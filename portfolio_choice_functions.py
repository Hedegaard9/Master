import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import importlib
from General_Functions import initial_weights_new, create_cov, pf_ts_fun, sigma_gam_adj, create_lambda
sqrtm_cpp = importlib.import_module("sqrtm_cpp")

# Funktion til beregning af M
def m_func(w, mu, rf, sigma_gam, gam, lambda_mat, iter):
    n = lambda_mat.shape[0]
    g_bar = np.ones(n)
    mu_bar_vec = np.ones(n) * (1 + rf + mu)

    # Beregning af sigma_gr
    sigma_gr = (1 + rf + mu) ** -2 * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)

    # Beregning af lambda^(-0.5), da lambda er diagonal
    lamb_neg05 = np.diag(np.diag(lambda_mat) ** -0.5)

    # Placeholder for x
    x = w ** -1 * lamb_neg05 @ sigma_gam @ lamb_neg05
    y = np.diag(1 + np.diag(sigma_gr))

    # Iteration på F
    sigma_hat = x + np.diag(1 + g_bar)

    # Brug C++-implementering af sqrtm, og tag realdelen
    sqrt_term = np.real(sqrtm_cpp.sqrtm_cpp(sigma_hat @ sigma_hat - 4 * np.eye(n)))

    m_tilde = 0.5 * (sigma_hat - sqrt_term)
    for _ in range(iter):
        m_tilde = np.linalg.inv(x + y - m_tilde * sigma_gr)  # Elementvis multiplikation

    # Output: Beregn elementvis kvadratrod af lambda
    return lamb_neg05 @ m_tilde @ np.diag(np.sqrt(np.diag(lambda_mat)))


# Statisk M
def m_static(sigma_gam, w, lambda_matrix, phi):
    return np.linalg.inv(sigma_gam + w / phi * lambda_matrix) @ (w / phi * lambda_matrix)

## Benchamark porteføljer
# Generisk vægtfunktion
def w_fun(data, dates, w_opt, wealth):
    # Opret et DataFrame med slutmåned (eom) og forrige slutmåned (eom_prev)
    dates_df = pd.DataFrame({'eom': dates})
    dates_df['eom_prev'] = dates_df['eom'].shift(1)

    # Udregn initiale vægte og fjern kolonnen 'w'
    w = initial_weights_new(data, w_type="vw").drop(columns=['w'])

    # Join: tilføj w_opt på (id, eom)
    w = pd.merge(w, w_opt, on=['id', 'eom'], how='left')

    # Join med dates_df for at få eom_prev
    w = pd.merge(w, dates_df, on='eom', how='left')

    # Tilføj 'tr_ld1' fra data til w_opt (join på id og eom)
    data_subset = data[['id', 'eom', 'tr_ld1']]
    w_opt = pd.merge(w_opt, data_subset, on=['id', 'eom'], how='left')

    # Tilføj 'mu_ld1' fra wealth (join på eom)
    wealth_subset = wealth[['eom', 'mu_ld1']]
    w_opt = pd.merge(w_opt, wealth_subset, on='eom', how='left')

    # Opdater 'w' ift. afkastjustering: w = w*(1+tr_ld1)/(1+mu_ld1)
    w_opt['w'] = w_opt['w'] * (1 + w_opt['tr_ld1']) / (1 + w_opt['mu_ld1'])

    # Omdøb kolonner: 'w' -> 'w_prev' og 'eom' -> 'eom_prev'
    w_opt = w_opt.rename(columns={'w': 'w_prev', 'eom': 'eom_prev'})

    # Join w med w_opt på (id, eom_prev)
    w = pd.merge(w, w_opt, left_on=['id', 'eom_prev'], right_on=['id', 'eom_prev'], how='left')

    # For alle rækker, hvor eom ikke er den mindste dato, sættes w_start = w_prev
    min_eom = w['eom'].min()
    w.loc[w['eom'] != min_eom, 'w_start'] = w['w_prev']

    # Udfyld eventuelle manglende værdier i w_start med 0
    w['w_start'] = w['w_start'].fillna(0)

    # Fjern de midlertidige kolonner
    w = w.drop(columns=['eom_prev', 'w_prev'])

    return w

# Implementering af tangensportefølje
def tpf_implement(data, cov_list, wealth, dates, gam):
    """
    Beregner tangency portfolio weights og porteføljestatistikker.

    Parametre:
      - data: DataFrame med kolonnerne 'id', 'eom', 'me', 'tr_ld1', 'pred_ld1' og 'valid'
      - cov_list: dictionary med nøgler som datoer (som strings, fx '2019-12-31') og værdier,
                  hvor hvert element er en dictionary med nøglerne 'fct_load', 'fct_cov' og 'ivol_vec'
      - wealth: DataFrame med kolonnerne 'eom', 'mu_ld1' og 'wealth'
      - dates: liste over datoer (som pd.Timestamp eller strings)
      - gam: risikoaversion parameter

    Returnerer:
      - en dictionary med nøglerne "w" (de faktiske vægte) og "pf" (porteføljestatistikker)

    #Eksempel på brug
    # tpf = tpf_implement(data=chars, cov_list=barra_cov, wealth=wealth, dates=dates_oos, gam=pf_set['gamma_rel'])
    """
    # Filtrer data for gyldige rækker med eom i dates
    data_rel = data[(data['valid'] == True) & (data['eom'].isin(dates))][['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']]
    data_rel = data_rel.sort_values(by=['id', 'eom'])

    # Opdel data_rel i grupper pr. eom
    data_split = {d: df for d, df in data_rel.groupby('eom')}

    tpf_opt_list = []

    for d in dates:
        # Konverter d til string, hvis d er en pd.Timestamp (cov_list-nøglerne forventes som strings)
        if isinstance(d, pd.Timestamp):
            d_key = d.strftime('%Y-%m-%d')
        else:
            d_key = d

        data_sub = data_split.get(d)
        if data_sub is None or data_sub.empty:
            continue

        ids = data_sub['id'].tolist()

        # Brug standalone funktionen create_cov her
        sigma = create_cov(cov_list[d_key], ids=ids)

        # Hent prediktioner for den aktuelle dato
        pred_vector = data_sub['pred_ld1'].values

        # Beregn de optimale vægte: w = (1/gam) * inv(sigma) * pred_vector
        weights = (1 / gam) * np.linalg.solve(sigma, pred_vector)  # Her beregnes de optimale vægte

        temp_df = data_sub[['id', 'eom']].copy()
        temp_df['w'] = weights
        tpf_opt_list.append(temp_df)

    if tpf_opt_list:
        tpf_opt = pd.concat(tpf_opt_list, ignore_index=True)
    else:
        tpf_opt = pd.DataFrame(columns=['id', 'eom', 'w'])

    # Beregn de faktiske vægte vha. w_fun (forventes defineret et andet sted)
    tpf_w = w_fun(data_rel, dates, tpf_opt, wealth)
    # Beregn porteføljestatistikker vha. pf_ts_fun (forventes defineret et andet sted)
    tpf_pf = pf_ts_fun(tpf_w, data, wealth, gam)
    tpf_pf['type'] = "Markowitz-ML"

    return {"w": tpf_w, "pf": tpf_pf}


# Counterfactual TPF
def tpf_cf_fun(data, cf_cluster, er_models, cluster_labels, wealth, gamma_rel, cov_list, dates, seed):
    np.random.seed(seed)
    if cf_cluster != "bm":
        cf = data.drop(columns=[col for col in data.columns if col.startswith("pred_ld")])
        cf["id_shuffle"] = cf.groupby("eom")["id"].transform(lambda x: np.random.permutation(x))
        chars_sub = cluster_labels.loc[cluster_labels["cluster"] == cf_cluster, "characteristic"]
        chars_data = cf[["id_shuffle", "eom"] + list(chars_sub)]
        cf.drop(columns=chars_sub, inplace=True)
        cf = pd.merge(cf, chars_data, on=["id_shuffle", "eom"])
        for model in er_models:
            sub_dates = model["pred"]["eom"].unique()
            cf_x = cf.loc[cf["eom"].isin(sub_dates), model["features"]].to_numpy()
            transformed = model["transform"](cf_x)
            cf.loc[cf["eom"].isin(sub_dates), "pred_ld1"] = model["fit"].predict(transformed)
    else:
        cf = data[data["valid"]]
    result = tpf_implement(cf, cov_list, wealth, dates, gamma_rel)
    result["pf"]["cluster"] = cf_cluster
    return result["pf"]

# Mean-Variance Efficient Portfolios of Risky Assets
def mv_risky_fun(data, cov_list, wealth, dates, gam, u_vec):
    # Filter relevant data
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me", "tr_ld1", "pred_ld1"]]
    data_rel = data_rel.sort_values(by=["id", "eom"])
    data_split = {key: group for key, group in data_rel.groupby("eom")}

    # Calculate optimal weights
    mv_opt_all = []
    for d in dates:
        data_sub = data_split[d]
        ids = data_sub["id"]
        sigma_inv = np.linalg.inv(create_cov(cov_list[d], ids))
        er = data_sub["pred_ld1"].values
        ones = np.ones_like(er)

        a = er.T @ sigma_inv @ er
        b = sigma_inv @ er
        c = sigma_inv @ ones
        d_aux = a * c.sum() - b.sum() ** 2

        for u in u_vec:
            weights = ((c.sum() * u - b.sum()) / d_aux) * (sigma_inv @ er) + \
                      ((a - b.sum() * u) / d_aux) * (sigma_inv @ ones)
            weights_df = pd.DataFrame({
                "id": data_sub["id"],
                "eom": d,
                "u": u * 12,
                "w": weights
            })
            mv_opt_all.append(weights_df)

    mv_opt_all = pd.concat(mv_opt_all)

    # Build portfolios
    results = []
    for u_sel in mv_opt_all["u"].unique():
        w = mv_opt_all[mv_opt_all["u"] == u_sel].copy()
        weights = w_fun(data_rel, dates, w, wealth)
        portfolio = pf_ts_fun(weights, data, wealth, gam)
        portfolio["u"] = u_sel
        results.append(portfolio)

    return pd.concat(results)

# High Minus Low (HML) Implementation
def factor_ml_implement(data, wealth, dates, n_pfs, gam):
    data = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me", "pred_ld1"]]
    data_split = {key: group for key, group in data_rel.groupby("eom")}

    hml_opt = []
    for d in dates:
        data_sub = data_split[d]
        er = data_sub["pred_ld1"]
        er_prank = er.rank(pct=True)

        data_sub["pos"] = 0
        data_sub.loc[er_prank >= 1 - 1 / n_pfs, "pos"] = 1
        data_sub.loc[er_prank <= 1 / n_pfs, "pos"] = -1

        data_sub["w"] = data_sub["me"] / data_sub.groupby("pos")["me"].transform("sum") * data_sub["pos"]
        hml_opt.append(data_sub[["id", "eom", "w"]])

    hml_opt = pd.concat(hml_opt)
    hml_w = w_fun(data, dates, hml_opt, wealth)
    hml_pf = pf_ts_fun(hml_w, data, wealth, gam)
    hml_pf["type"] = "Factor-ML"

    return {"w": hml_w, "pf": hml_pf}

# 1/N Portfolio Implementation
def ew_implement(data, wealth, dates, pf_set):
    # Filtrer data, så kun gyldige rækker med eom i dates bevares
    filtered = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()

    # Beregn antallet af rækker pr. eom
    counts = filtered.groupby('eom').size().reset_index(name='n')
    filtered = pd.merge(filtered, counts, on='eom', how='left')

    # Udregn optimale vægte: for hver (id, eom) sættes w = 1/n
    ew_opt = filtered[['id', 'eom']].copy()
    ew_opt['w'] = 1 / filtered['n']

    # Beregn vægte vha. w_fun
    weights_data = data[(data['eom'].isin(dates)) & (data['valid'] == True)].copy()
    ew_w = w_fun(weights_data, dates, ew_opt, wealth)

    # Beregn portefølje-statistikker vha. pf_ts_fun og tilføj porteføljetypen "1/N"
    ew_pf = pf_ts_fun(ew_w, data, wealth, gam=pf_set['gamma_rel'])
    ew_pf['type'] = "1/N"

    return {"w": ew_w, "pf": ew_pf}

# Market Portfolio Implementation
def mkt_implement(data, wealth, dates, pf_set):
    data = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates)]
    mkt_opt = data_rel.groupby("eom").apply(lambda x: pd.DataFrame({
        "id": x["id"],
        "eom": x["eom"],
        "w": x["me"] / x["me"].sum()
    })).reset_index(drop=True)

    mkt_w = w_fun(data, dates, mkt_opt, wealth)
    mkt_pf = pf_ts_fun(mkt_w, data, wealth, gam=pf_set['gamma_rel'])
    mkt_pf["type"] = "Market"

    return {"w": mkt_w, "pf": mkt_pf}

# Rank-Weighted Portfolio Implementation
def rw_implement(data, wealth, dates, pf_set):
    data = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me", "pred_ld1"]]
    data_split = {key: group for key, group in data_rel.groupby("eom")}

    rw_opt = []
    for d in dates:
        data_sub = data_split[d]
        er_rank = data_sub["pred_ld1"].rank()
        pos = (er_rank - er_rank.mean()) * 2 / er_rank.abs().sum()
        data_sub["w"] = pos
        rw_opt.append(data_sub[["id", "eom", "w"]])

    rw_opt = pd.concat(rw_opt)
    rw_w = w_fun(data, dates, rw_opt, wealth)
    rw_pf = pf_ts_fun(rw_w, data, wealth, gam=pf_set['gamma_rel'])  # Assuming `gam` is predefined or unnecessary
    rw_pf["type"] = "Rank-ML"

    return {"w": rw_w, "pf": rw_pf}

#Minimum Variance Portfolio
def mv_implement(data, cov_list, wealth, dates, pf_set):
    data = data[(data['valid'] == True) & (data['eom'].isin(dates))].copy()
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me"]]
    data_split = {key: group for key, group in data_rel.groupby("eom")}

    mv_opt = []
    for d in dates:
        data_sub = data_split[d]
        sig = create_cov(cov_list[d.strftime("%Y-%m-%d")], data_sub["id"])
        sig_inv = np.linalg.inv(sig)
        ones = np.ones((sig_inv.shape[0], 1))
        pos = (1 / (ones.T @ sig_inv @ ones)) * (sig_inv @ ones)
        data_sub["w"] = pos.flatten()
        mv_opt.append(data_sub[["id", "eom", "w"]])

    mv_opt = pd.concat(mv_opt)
    mv_w = w_fun(data, dates, mv_opt, wealth)
    mv_pf = pf_ts_fun(mv_w, data, wealth, gam=None)
    mv_pf["type"] = "Minimum Variance"

    return {"w": mv_w, "pf": mv_pf}

#Static Validation
def static_val_fun(data, dates, cov_list, lambda_list, wealth, cov_type, gamma_rel, k=None, g=None, u=None, hps=None):
    static_weights = initial_weights_new(data, w_type="vw")
    static_weights = pd.merge(static_weights, data[["id", "eom", "tr_ld1", "pred_ld1"]], on=["id", "eom"], how="left")
    static_weights = pd.merge(static_weights, wealth[["eom", "mu_ld1"]], on="eom", how="left")

    for d in dates:
        if hps is not None:
            # Filtrer alle hyperparametre med eom_ret < d
            hp_filtered = hps[hps["eom_ret"] < pd.Timestamp(d)]
            if not hp_filtered.empty:
                # Vælg den række med den højeste eom_ret (dvs. den seneste før d)
                hp = hp_filtered.loc[hp_filtered["eom_ret"].idxmax()]
                g = hp["g"]
                u = hp["u"]
                k = hp["k"]
            else:
                # Hvis der ikke findes hyperparametre før d, kan du f.eks. springe denne iteration over
                continue

        wealth_t = wealth.loc[wealth["eom"] == pd.Timestamp(d), "wealth"].values[0]
        ids = static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "id"]

        # Konverter d til en streng nøgle, fx "2019-12-31"
        key = pd.Timestamp(d).strftime('%Y-%m-%d')
        sigma_gam = create_cov(cov_list[key], ids) * gamma_rel
        sigma_gam = sigma_gam_adj(sigma_gam, g, cov_type)

        id_list = list(ids)
        lambda_vals = [lambda_list[key][i] for i in id_list]
        lambda_matrix = np.diag(lambda_vals) * k

        weights = np.linalg.solve(
            sigma_gam + wealth_t * lambda_matrix,
            (u * static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "pred_ld1"] +
             wealth_t * lambda_matrix @ static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "w_start"])
        )

        static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "w"] = weights

        next_index = dates.get_loc(d) + 1
        next_month = dates[next_index] if next_index < len(dates) else None
        if next_month:
            mask_current = static_weights["eom"] == pd.Timestamp(d)
            mask_next = static_weights["eom"] == pd.Timestamp(next_month)
            current_tr = static_weights.loc[mask_current, "tr_ld1"].values
            current_mu = static_weights.loc[mask_current, "mu_ld1"].values
            next_weights = weights * (1 + current_tr) / (1 + current_mu)
            static_weights.loc[mask_next, "w_start"] = next_weights
            static_weights.loc[:, "w_start"] = static_weights["w_start"].fillna(0)

    return static_weights


#Static Full Implementation
def static_implement(data_tc, cov_list, lambda_list, rf,
                     wealth, mu, gamma_rel,
                     dates_full, dates_oos, dates_hp, hp_years,
                     k_vec, u_vec, g_vec, cov_type,
                     validation=None, seed=None):
    data_tc = data_tc[(data_tc['valid'] == True) & (data_tc['eom'].isin(dates_oos))].copy()
    # Sæt seed, hvis nødvendigt
    if seed is not None:
        np.random.seed(seed)

    # HP grid: opret en DataFrame med alle kombinationer af k, u og g
    static_hps = pd.DataFrame(list(product(k_vec, u_vec, g_vec)), columns=['k', 'u', 'g'])

    # Udvælg relevant data: filter på 'valid' == True og eom i dates_hp og sorter efter id og eom
    data_rel = (data_tc.loc[(data_tc['valid']) & (data_tc['eom'].isin(dates_hp)),
    ['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']]
                .sort_values(by=['id', 'eom'])
                )

    # Validering: Hvis validation er None, beregn den
    if validation is None:
        validation_list = []
        for i in range(len(static_hps)):
            print(i)
            # Hent hyperparameterkombinationen
            hp = static_hps.iloc[i]
            # Kald funktionen static_val_fun på data_rel med de aktuelle hyperparametre
            static_w = static_val_fun(data_rel, dates=dates_hp, cov_list=cov_list,
                                      lambda_list=lambda_list, wealth=wealth, gamma_rel=gamma_rel,
                                      k=hp['k'], g=hp['g'], u=hp['u'], cov_type=cov_type)
            # Kald herefter pf_ts_fun på static_w og tilføj kolonner med hyperparametre og hp-nummer
            pf = pf_ts_fun(static_w, data=data_tc, wealth=wealth, gam=gamma_rel)
            pf = pf.copy()  # For at undgå advarsler om SettingWithCopy
            pf['hp_no'] = i
            pf['k'] = hp['k']
            pf['g'] = hp['g']
            pf['u'] = hp['u']
            validation_list.append(pf)

        # Kombinér resultaterne til én DataFrame (svarer til rbindlist)
        validation = pd.concat(validation_list, ignore_index=True)

    # Sorter validation efter hp_no og eom_ret
    validation.sort_values(by=['hp_no', 'eom_ret'], inplace=True)

    # Beregn kumulativ varians: cum_var = cummean(r^2) - (cummean(r))^2, beregnet pr. hp_no-gruppe
    validation['cum_var'] = (validation
                             .groupby('hp_no')['r']
                             .apply(lambda x: (x.pow(2).expanding().mean() - x.expanding().mean().pow(2)))
                             .reset_index(level=0, drop=True)
                             )

    # Beregn kumulativ objektiv funktion: cum_obj = cummean(r - tc - 0.5*cum_var*gamma_rel) pr. hp_no
    validation['cum_obj'] = (validation
                             .groupby('hp_no')
                             .apply(lambda group: (group['r'] - group['tc'] - 0.5 * group['cum_var'] * gamma_rel)
                                    .expanding().mean())
                             .reset_index(level=0, drop=True)
                             )

    # Beregn rang inden for hver eom_ret: højere cum_obj giver lavere rang (rank 1 er bedst)
    validation['rank'] = (validation
                          .groupby('eom_ret')['cum_obj']
                          .rank(ascending=False, method='min')
                          )

    # Plot: opret en 'name'-kolonne og filtrér for år >= 1981
    validation = validation.copy()
    validation['name'] = ("u:" + validation['u'].astype(str) +
                          " k:" + validation['k'].round(2).astype(str) +
                          " g:" + validation['g'].astype(str))

    plot_df = validation[validation['eom_ret'].dt.year >= 1981].copy()
    # For hver hp_no, beregn minimum rank og filtrér for dem med min_rank <= 1
    plot_df['min_rank'] = plot_df.groupby('hp_no')['rank'].transform('min')
    plot_df_filtered = plot_df[plot_df['min_rank'] <= 1]

    # Lav plot med seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df_filtered, x='eom_ret', y=plot_df_filtered['cum_obj'] * 12, hue='name')
    plt.xlabel('eom_ret')
    plt.ylabel('Kumulativ objektiv funktion * 12')
    plt.title('Static-implement: Kumulativ objektiv funktion pr. hyperparameter')
    plt.show()

    # Find optimale hyperparametre: vælg rækker med eom_ret i december og rank==1
    optimal_hps = (validation[(validation['eom_ret'].dt.month == 12) & (validation['rank'] == 1)]
                   .sort_values(by='eom_ret'))

    # Implementer det endelige portfolio:
    # Udvælg data_tc for datoer i dates_oos og hvor valid==True
    w_data = data_tc.loc[(data_tc['valid']) & (data_tc['eom'].isin(dates_oos)),
    ['id', 'eom', 'me', 'tr_ld1', 'pred_ld1']]
    w = static_val_fun(w_data, dates=dates_oos, cov_list=cov_list, lambda_list=lambda_list,
                       wealth=wealth, gamma_rel=gamma_rel, hps=optimal_hps, cov_type=cov_type)

    # Beregn portefølje-tidsserie og tilføj en 'type'-kolonne
    pf = pf_ts_fun(w, data=data_tc, wealth=wealth, gam=gamma_rel)
    pf = pf.copy()
    pf['type'] = "Static-ML*"

    # Returnér resultaterne som en ordbog
    return {"hps": validation, "best_hps": optimal_hps, "w": w, "pf": pf}

#Portfolio-ML Inputs
def pfml_input_fun(data_tc, cov_list, lambda_list, gamma_rel, wealth, mu, dates, lb, scale,
                   risk_free, features, rff_feat, seed, p_max, g, add_orig, iter, balanced):
    np.random.seed(seed)

    # Lookback dates
    dates_lb = pd.date_range(start=min(dates) - pd.DateOffset(months=lb + 1), end=max(dates), freq="M")

    # Generate random features if required
    if rff_feat:
        rff_x = generate_random_features(data_tc[features], p_max, g)
        rff_w = rff_x['weights']
        rff_features = rff_x['features']
        rff_features.columns = [f"rff{i + 1}" for i in range(rff_features.shape[1])]
        data = pd.concat([data_tc[["id", "eom", "valid", "ret_ld1", "tr_ld0", "mu_ld0"]], rff_features], axis=1)
        feat_new = rff_features.columns.tolist()
        if add_orig:
            data = pd.concat([data, data_tc[features]], axis=1)
            feat_new += features
    else:
        data = data_tc[["id", "eom", "valid", "ret_ld1", "tr_ld0", "mu_ld0"] + features]
        feat_new = features

    feat_cons = feat_new + ["constant"]

    # Scaling
    if scale:
        scales = []
        for d in dates_lb:
            sigma = create_cov(cov_list[d])
            diag_vol = np.sqrt(np.diag(sigma))
            scales.append(pd.DataFrame({"id": sigma.index, "eom": d, "vol_scale": diag_vol}))
        scales = pd.concat(scales)
        data = data.merge(scales, on=["id", "eom"], how="left")
        data["vol_scale"] = data.groupby("eom")["vol_scale"].transform(lambda x: x.fillna(x.median()))

    # Balanced data
    if balanced:
        data[feat_new] = data.groupby("eom")[feat_new].transform(lambda x: x - x.mean())
        data["constant"] = 1
        data[feat_cons] = data.groupby("eom")[feat_cons].transform(lambda x: x / np.sqrt((x ** 2).sum()))

    # Prepare signals and realizations
    inputs = {}
    for d in dates:
        if d.year % 10 == 0 and d.month == 1:
            print(f"--> PF-ML inputs: {d}")

        data_ret = data[(data["valid"]) & (data["eom"] == d)][["id", "ret_ld1"]]
        ids = data_ret["id"].values
        sigma = create_cov(cov_list[d], ids)
        lambda_matrix = create_lambda(lambda_list[d], ids)
        w = wealth.loc[wealth["eom"] == d, "wealth"].values[0]
        rf = risk_free.loc[risk_free["eom"] == d, "rf"].values[0]
        m = m_func(w=w, mu=mu, rf=rf, sigma_gam=sigma * gamma_rel, gam=gamma_rel, lambda_matrix=lambda_matrix,
                   iter=iter)

        # Prepare signals
        data_sub = data[(data["id"].isin(ids)) & (data["eom"] >= d - pd.DateOffset(months=lb))]
        if not balanced:
            data_sub[feat_new] = data_sub.groupby("eom")[feat_new].transform(lambda x: x - x.mean())
            data_sub["constant"] = 1
            data_sub[feat_cons] = data_sub.groupby("eom")[feat_cons].transform(lambda x: x / np.sqrt((x ** 2).sum()))

        signals = {}
        for d_new, group in data_sub.groupby("eom"):
            s = group[feat_cons].to_numpy()
            if scale:
                s = np.diag(1 / group["vol_scale"].values) @ s
            signals[d_new] = s

        # Weighted signals (omega)
        omega, const, omega_l1, const_l1 = 0, 0, 0, 0
        for i in range(lb + 1):
            d_new = d - pd.DateOffset(months=i)
            d_new_l1 = d_new - pd.DateOffset(months=1)
            s = signals.get(d_new, np.zeros((len(ids), len(feat_cons))))
            s_l1 = signals.get(d_new_l1, np.zeros((len(ids), len(feat_cons))))

            omega += m @ s
            const += m
            omega_l1 += m @ s_l1
            const_l1 += m

        omega = np.linalg.solve(const, omega)
        omega_l1 = np.linalg.solve(const_l1, omega_l1)
        omega_chg = omega - np.diag((1 + data_sub["tr_ld0"]) / (1 + data_sub["mu_ld0"])) @ omega_l1

        r_tilde = omega.T @ data_ret["ret_ld1"].values
        risk = gamma_rel * (omega.T @ sigma @ omega)
        tc = w * (omega_chg.T @ lambda_matrix @ omega_chg)
        denom = risk + tc

        inputs[d] = {
            "reals": {"r_tilde": r_tilde, "denom": denom, "risk": risk, "tc": tc},
            "signal_t": signals.get(d, np.zeros((len(ids), len(feat_cons))))
        }

    return inputs





