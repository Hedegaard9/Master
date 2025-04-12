import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import General_Functions
import importlib
from General_Functions import initial_weights_new, create_cov, pf_ts_fun, sigma_gam_adj, create_lambda
sqrtm_cpp = importlib.import_module("sqrtm_cpp")

# Funktion til beregning af M
def m_func(w, mu, rf, sigma_gam, gam, lambda_mat, iter): ## ændret denne i port_8 push, hvis problemer
    # Konverter inputs til NumPy-arrays, hvis de ikke allerede er det
    sigma_gam = np.array(sigma_gam)
    lambda_mat = np.array(lambda_mat)

    n = lambda_mat.shape[0]

    g_bar = np.ones(n)
    mu_bar_vec = np.ones(n) * (1 + rf + mu)

    # Beregn sigma_gr
    sigma_gr = (1 + rf + mu) ** -2 * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)
#    print("m_func: sigma_gr shape:", sigma_gr.shape)

    # Beregn lambda^(-0.5), da lambda er diagonal
    lamb_neg05 = np.diag(np.diag(lambda_mat) ** -0.5)
#    print("m_func: lamb_neg05 shape:", lamb_neg05.shape)

    # Beregn x
    x = (w ** -1) * (lamb_neg05 @ sigma_gam @ lamb_neg05)
#    print("m_func: x shape:", x.shape)

    # Beregn y
    y = np.diag(1 + np.diag(sigma_gr))
#    print("m_func: y shape:", y.shape)

    sigma_hat = x + np.diag(1 + g_bar)
#    print("m_func: sigma_hat shape:", sigma_hat.shape)

    # Brug C++-implementeringen af sqrtm, og tag realdelen
    sqrt_term = np.real(sqrtm_cpp.sqrtm_cpp(sigma_hat @ sigma_hat - 4 * np.eye(n)))
#    print("m_func: sqrt_term shape:", sqrt_term.shape)

    m_tilde = 0.5 * (sigma_hat - sqrt_term)
#    print("m_func: initial m_tilde shape:", m_tilde.shape)

    # Iteration på F med elementvis multiplikation (hvis det er tilsigtet)
    for i in range(iter):
        inv_arg = x + y - m_tilde * sigma_gr  # Elementvis multiplikation
        #print(f"m_func: Iteration {i}, inv_arg shape:", inv_arg.shape)
        m_tilde = np.linalg.inv(inv_arg)
        #print(f"m_func: Iteration {i}, m_tilde shape:", m_tilde.shape)

    # Beregn elementvis kvadratrod af lambda (dette giver en diagonal matrix)
    lambda_sqrt = np.diag(np.sqrt(np.diag(lambda_mat)))

    result = lamb_neg05 @ m_tilde @ lambda_sqrt
    #print("m_func: result shape:", result.shape)
    return result

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
        pos = (er_rank - er_rank.mean()) * 2 / (er_rank - er_rank.mean()).abs().sum() #So it sums to 2
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
    static_weights = General_Functions.initial_weights_new(data, w_type="vw")
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
                continue

        wealth_t = wealth.loc[wealth["eom"] == pd.Timestamp(d), "wealth"].values[0]
        ids = static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "id"]

        # Konverter d til en streng nøgle, fx "2019-12-31"
        key = pd.Timestamp(d).strftime('%Y-%m-%d')
        sigma_gam = General_Functions.create_cov(cov_list[key], ids) * gamma_rel
        sigma_gam = General_Functions.sigma_gam_adj(sigma_gam, g, cov_type)

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
    dates_hp = dates_hp[dates_hp >= pd.Timestamp(dates_oos[0].strftime('%Y-%m-%d'))]
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
            pf = General_Functions.pf_ts_fun(static_w, data=data_tc, wealth=wealth, gam=gamma_rel)
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
    pf = General_Functions.pf_ts_fun(w, data=data_tc, wealth=wealth, gam=gamma_rel)
    pf = pf.copy()
    pf['type'] = "Static-ML*"

    # Returnér resultaterne som en ordbog
    return {"hps": validation, "best_hps": optimal_hps, "w": w, "pf": pf}


def pfml_search_coef(pfml_input, p_vec, l_vec, hp_years, orig_feat):  # virker kun for et p so far.
    # Konverter nøglerne (datoer) til datetime-objekter
    reals = pfml_input["reals"]
    # Vi antager, at nøglerne kan konverteres med pd.to_datetime
    d_all = {pd.to_datetime(k): k for k in reals.keys()}

    # Bestem end_bef = min(hp_years) - 1 (fx hvis det mindste hp_year er 2016, så er end_bef 2015)
    end_bef = 2021  # - 1
    # Vælg træningsdata før: datoer < (min(hp_years)-2)-12-31
    cutoff_date = pd.to_datetime(f"{end_bef - 1}-12-31")
    train_bef = {k: reals[d_all[k]] for k in d_all if k < cutoff_date}

    # Beregn sum af r_tilde over træningssættet.
    # Her antages det, at hver x['r_tilde'] er en pandas Series med index svarende til feature-navne.
    r_tilde_list = [item['r_tilde'] for item in train_bef.values()]
    # Summering: her benyttes sum() til pandas Series (forudsætter, at listen ikke er tom)
    r_tilde_sum = sum(r_tilde_list) if r_tilde_list else pd.Series(0)

    # Tilsvarende: sum af 'denom'-matricer (antages at være pandas DataFrames med feature-navne som index/columns)
    denom_list = [item['denom'] for item in train_bef.values()]
    denom_raw_sum = sum(denom_list) if denom_list else pd.DataFrame(0)

    # n er antallet af observationer i train_bef
    n = len(train_bef)  # skal evt ændres

    # Sorter hp_years, og opret en ordbog til at holde koefficienterne
    hp_years = sorted(hp_years)
    coef_list = {}

    # For hvert hp_year
    for hp in hp_years:
        # Udvælg nye træningsobservationer: datoer i intervallet
        # [ (hp-2)-12-31, (hp-1)-11-30 ]
        lower_bound = pd.to_datetime(f"{hp - 2}-12-31")
        upper_bound = pd.to_datetime(f"{hp - 1}-11-30")

        train_new = {k: reals[d_all[k]] for k in d_all if lower_bound <= k <= upper_bound}

        # Opdater antallet af observationer
        n += len(train_new)
        # Opdater r_tilde_sum med de nye r_tilde-værdier
        r_tilde_new = sum([item['r_tilde'] for item in train_new.values()]) if len(train_new) > 0 else 0
        r_tilde_sum = r_tilde_sum + r_tilde_new

    # Opdater denom-summen
    denom_raw_new = sum([item['denom'] for item in train_new.values()]) if len(train_new) > 0 else 0
    denom_raw_sum = denom_raw_sum + denom_raw_new

    # For hvert p i p_vec, beregn de tilhørende koefficienter
    coef_by_hp = {}
    # Hent feature-navnene givet p og om de originale features skal medtages
    feat_p = pfml_feat_fun(p, orig_feat)
    # r_tilde_sub: udtræk de features fra r_tilde_sum og divider med n
    r_tilde_sum = pd.Series(r_tilde_sum, index=all_feat)

    r_tilde_sub = r_tilde_sum.loc[feat_p] / n
    # denom_sub: udtræk de relevante rækker og kolonner fra denom_raw_sum og divider med n
    denom_raw_sum.index = all_feat
    denom_raw_sum.columns = all_feat

    denom_sub = denom_raw_sum.loc[feat_p, feat_p] / n

    results = {}
    for l in l_vec:
        M = denom_sub + l * np.eye(len(feat_p))
        # Løs det lineære system: M * coef = r_tilde_sub
        coef = np.linalg.solve(M.values, r_tilde_sub.values)
        # Gem resultatet som en pandas Series med index = feat_p
        results[l] = pd.Series(coef, index=feat_p)
    # Gem resultaterne for den aktuelle p-værdi
    coef_by_hp[p] = results
    # Gem koefficienterne for det aktuelle hp_year, brug hp som streng
    coef_list[str(hp)] = coef_by_hp

    return coef_list


# Feature names
def pfml_feat_fun(p, orig_feat, features=None):
    """
    Returnerer en liste af feature-navne.

    Parametre:
      p         : Antal RFF-features (skal være deleligt med 2). Hvis p er 0, bruges ingen RFF-features.
      orig_feat : Bool – om de originale features skal inkluderes.
      features  : Liste af originale feature-navne. Skal gives, hvis orig_feat er True.

    Returnerer:
      En liste med feature-navne.
    """
    feat = ["constant"]
    if p != 0:
        half_p = p // 2
        feat.extend([f"rff{i}_cos" for i in range(1, half_p + 1)])
        feat.extend([f"rff{i}_sin" for i in range(1, half_p + 1)])
    if orig_feat and features is not None:
        feat.extend(features)
    return feat


#Portfolio-ML Inputs
def pfml_input_fun(data_tc, cov_list, lambda_list, gamma_rel, wealth, mu, dates, lb, scale,
                   risk_free, features, rff_feat, seed, p_max, g, add_orig, iter, balanced):
    # --- Lookback-datoer ---
    min_date = min(dates)
    max_date = max(dates)
    start_date = (min_date + pd.Timedelta(days=1)) - MonthEnd(lb + 1)
    dates_lb = pd.date_range(start=start_date, end=max_date, freq=MonthEnd())
    print(dates_lb)
    # --- Oprettelse af Random Fourier Features ---
    if rff_feat:
        np.random.seed(seed)
        X_features = data_tc[features].values
        rff_x = rff(X_features, p=p_max, g=g)
        rff_w = rff_x['W']
        X_cos = rff_x['X_cos']
        X_sin = rff_x['X_sin']
        rff_features = np.hstack([X_cos, X_sin])
        num = p_max // 2
        rff_colnames = [f"rff{i}_cos" for i in range(1, num + 1)] + [f"rff{i}_sin" for i in range(1, num + 1)]
        rff_df = pd.DataFrame(rff_features, columns=rff_colnames, index=data_tc.index)
        data = pd.concat([data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0']].reset_index(drop=True),
                          rff_df.reset_index(drop=True)], axis=1)
        feat_new = list(rff_df.columns)
        if add_orig:
            data = pd.concat([data, data_tc[features].reset_index(drop=True)], axis=1)
            feat_new = feat_new + features
    else:
        cols = ['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0'] + features
        data = data_tc[cols].copy()
        feat_new = features.copy()

    # Konverter 'eom' til string-format så vi får samme format i hele funktionen
    data['eom'] = pd.to_datetime(data['eom']).dt.strftime('%Y-%m-%d')

    feat_cons = feat_new + ['constant']

    # --- Tilføjelse af scales ---
    if scale:
        scales_list = []
        for d in dates_lb:
            d_str = d.strftime('%Y-%m-%d')
            sigma = General_Functions.create_cov(cov_list[d_str])
            if hasattr(sigma, 'values'):
                sigma_vals = sigma.values
                ids_sigma = sigma.index.astype(float)
            else:
                sigma_vals = sigma
                ids_sigma = np.arange(sigma.shape[0])
            diag_vol = np.sqrt(np.diag(sigma_vals))
            df_scales = pd.DataFrame({
                'id': ids_sigma,
                'eom': d_str,  # Brug d_str for konsistens
                'vol_scale': diag_vol
            })
            scales_list.append(df_scales)
        scales_df = pd.concat(scales_list, ignore_index=True)
        data = pd.merge(data, scales_df, on=['id', 'eom'], how='left')
        data['vol_scale'] = data.groupby('eom')['vol_scale'].transform(lambda x: x.fillna(x.median()))

    # --- Justering ved balanced panel ---
    if balanced:
        for col in feat_new:
            data[col] = data.groupby('eom')[col].transform(lambda x: x - x.mean())
        data['constant'] = 1
        for col in feat_cons:
            data[col] = data.groupby('eom')[col].transform(
                lambda x: x * np.sqrt(1 / np.sum(x ** 2)) if np.sum(x ** 2) != 0 else 0)

    # --- Beregning af signaler og realiseringer for hvert dato ---
    reals_dict = {}
    signal_t_dict = {}

    for d in dates:
        if d.year % 10 == 0 and d.month == 1:
            print(f"--> PF-ML inputs: {d}")

        d_str = d.strftime('%Y-%m-%d')
        print("d", d)
        print("d_str", d_str)
        data_ret = data[(data['valid'] == True) & (data['eom'] == d_str)][['id', 'ret_ld1']]
        ids = data_ret['id'].unique()  # ids som ints
        n = len(ids)
        r_vec = data_ret['ret_ld1'].values

        # Brug de korrekte key-typer til både sigma og lambda
        sigma = General_Functions.create_cov(cov_list[d_str], ids=ids)
        lambda_series = pd.Series(lambda_list[d_str])
        lambda_mat = General_Functions.create_lambda(lambda_series, ids=ids)

        w = wealth.loc[wealth['eom'] == d_str, 'wealth'].iloc[0]
        rf = risk_free.loc[risk_free['eom'] == d_str, 'rf'].iloc[0]
        m = m_func(w=w, mu=mu, rf=rf, sigma_gam=sigma * gamma_rel, gam=gamma_rel,
                   lambda_mat=lambda_mat, iter=iter)

        # Brug MonthEnd til at beregne lower_bound – konverter til string-format
        lower_bound = (d - MonthEnd(lb)).strftime('%Y-%m-%d')
        data_sub = data[(data['id'].isin(ids)) & (data['eom'] >= lower_bound) & (data['eom'] <= d_str)].copy()

        if not balanced:
            for col in feat_new:
                data_sub[col] = data_sub.groupby('eom')[col].transform(lambda x: x - x.mean())
            data_sub['constant'] = 1
            for col in feat_cons:
                data_sub[col] = data_sub.groupby('eom')[col].transform(
                    lambda x: x * np.sqrt(1 / np.sum(x ** 2)) if np.sum(x ** 2) != 0 else 0)

        data_sub = data_sub.sort_values(by=['eom', 'id'], ascending=[False, True])
        # Gruppér efter 'eom' – vi konverterer gruppenøglerne til string-format for konsistens
        groups = {k: group for k, group in data_sub.groupby('eom')}

        # Beregn signaler med diagnose: udskriv kun hvis NaN, Inf, tom DataFrame eller tomt index
        signals = {}
        for eom_val, group in groups.items():
            if group.empty or group.index.empty:
                print(f"Advarsel: Gruppe for {eom_val} er tom!")
            s = group[feat_cons].values
            if scale:
                s = np.diag(1 / group['vol_scale'].values) @ s
            # Tjek for NaN eller Inf
            if np.isnan(s).any():
                print(f"Advarsel: Signal for {eom_val} indeholder NaN. Form: {s.shape}")
            if np.isinf(s).any():
                print(f"Advarsel: Signal for {eom_val} indeholder Inf. Form: {s.shape}")
            signals[eom_val] = s
        print("signaler", signals)
        d_key = d_str  # d_str er allerede i '%Y-%m-%d'-format
        signal_current = signals.get(d_key, None)
        if signal_current is None:
            print(f"Advarsel: Ingen signal fundet for {d_key}.")
            continue
        elif np.isnan(signal_current).any() or np.isinf(signal_current).any():
            print(f"Advarsel: Signal for {d_key} indeholder NaN eller Inf, springer denne dato over.")
            continue

        # Beregn gtm for hver gruppe
        gtm = {}
        for eom_val, group in groups.items():
            gt = (1 + group['tr_ld0']) / (1 + group['mu_ld0'])
            gt = gt.fillna(1).values
            gtm[eom_val] = m @ np.diag(gt)

        # Aggregér gtm over lookback-perioden
        n_stocks = n
        gtm_agg = {}
        gtm_agg_l1 = {}
        gtm_agg[d_key] = np.eye(n_stocks)
        gtm_agg_l1[d_key] = np.eye(n_stocks)
        for i in range(1, lb + 1):
            d_i = (d - MonthEnd(i)).strftime('%Y-%m-%d')
            # DEBUG: Udskriv unikke 'eom'-datoer for denne lookback-dato
            lookback_date = pd.to_datetime(d_i)
            unique_dates = data[data['eom'] == lookback_date.strftime('%Y-%m-%d')]['eom'].unique()
            print(f"Lookup for lookback-dato: {d_i} - Unikke eom-datoer fundet: {unique_dates}")

            if d_i in gtm:
                gtm_agg[d_i] = gtm_agg[list(gtm_agg.keys())[-1]] @ gtm[d_i]
            else:
                gtm_agg[d_i] = gtm_agg[list(gtm_agg.keys())[-1]]
            d_i_l1 = (d - MonthEnd(i + 1)).strftime('%Y-%m-%d')
            if d_i_l1 in gtm:
                gtm_agg_l1[d_i] = gtm_agg_l1[list(gtm_agg_l1.keys())[-1]] @ gtm[d_i_l1]
            else:
                gtm_agg_l1[d_i] = gtm_agg_l1[list(gtm_agg_l1.keys())[-1]]

        # Summering over lookback: opbyg omega og konstanter
        omega_sum = None
        const_sum = None
        omega_l1_sum = None
        const_l1_sum = None
        for i in range(0, lb + 1):
            d_i = (d - MonthEnd(i)).strftime('%Y-%m-%d')
            s_i = signals.get(d_i, None)
            if s_i is None:
                print(f"Advarsel: Ingen signal fundet for lookback-dato d_i {d_i}")
                term = np.zeros((n_stocks, len(feat_cons)))
            else:
                term = gtm_agg.get(d_i, np.eye(n_stocks)) @ s_i
            if omega_sum is None:
                omega_sum = term
                const_sum = gtm_agg.get(d_i, np.eye(n_stocks))
            else:
                omega_sum = omega_sum + term
                const_sum = const_sum + gtm_agg.get(d_i, np.eye(n_stocks))

            d_i_l1 = (d - MonthEnd(i + 1)).strftime('%Y-%m-%d')
            s_i_l1 = signals.get(d_i_l1, None)
            if s_i_l1 is None:
                print(f"Advarsel: Ingen signal fundet for lookback-dato d_i_l1 {d_i_l1}")
                term_l1 = np.zeros((n_stocks, len(feat_cons)))
            else:
                term_l1 = gtm_agg_l1.get(d_i_l1, np.eye(n_stocks)) @ s_i_l1
            if omega_l1_sum is None:
                omega_l1_sum = term_l1
                const_l1_sum = gtm_agg_l1.get(d_i_l1, np.eye(n_stocks))
            else:
                omega_l1_sum = omega_l1_sum + term_l1
                const_l1_sum = const_l1_sum + gtm_agg_l1.get(d_i_l1, np.eye(n_stocks))

        # Løs de lineære systemer
        omega_final = np.linalg.solve(const_sum, omega_sum)
        omega_l1_final = np.linalg.solve(const_l1_sum, omega_l1_sum)

        # Beregn gt for den aktuelle gruppe (d)
        if d_key in signals:
            group_d = groups.get(d_key, None)
            if group_d is None:
                gt_vec = np.ones(n_stocks)
            else:
                gt_vec = ((1 + group_d['tr_ld0']) / (1 + group_d['mu_ld0'])).values
            gt_mat = np.diag(gt_vec)
        else:
            gt_mat = np.eye(n_stocks)

        omega_chg = omega_final - gt_mat @ omega_l1_final

        # --- Realiseringer ---
        r_tilde = omega_final.T @ r_vec
        risk_val = gamma_rel * (omega_final.T @ sigma @ omega_final)
        tc_val = w * (omega_chg.T @ lambda_mat @ omega_chg)
        denom = risk_val + tc_val

        reals = {
            "r_tilde": r_tilde.item() if np.isscalar(r_tilde) or r_tilde.size == 1 else r_tilde,
            "denom": denom.item() if np.isscalar(denom) or denom.size == 1 else denom,
            "risk": risk_val.item() if np.isscalar(risk_val) or risk_val.size == 1 else risk_val,
            "tc": tc_val.item() if np.isscalar(tc_val) or tc_val.size == 1 else tc_val
        }

        reals_dict[d_key] = reals
        signal_t_dict[d_key] = signals.get(d_key, None)

    return {"reals": reals_dict, "signal_t": signal_t_dict, "rff_w": rff_w if rff_feat else None}





