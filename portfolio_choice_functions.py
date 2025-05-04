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
#    print("m_func: n =", n)

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
    # Beregn porteføljestatistikker
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


def tpf_cf_fun(data, cf_cluster, er_models, cluster_labels, wealth, gamma_rel, cov_list, dates, seed):
    np.random.seed(seed)

    if cf_cluster != "bm":
        # Fjern kolonner, der starter med "pred_ld"
        cf = data.loc[:, ~data.columns.str.startswith("pred_ld")].copy()
        # Tilføj "id_shuffle" kolonne: for hver gruppe defineret af 'eom' laves en tilfældig permutation af 'id'
        cf['id_shuffle'] = cf.groupby('eom')['id'].transform(lambda x: np.random.permutation(x.values))

        # Hent de relevante karakteristika for den givne klynge fra cluster_labels
        chars_sub = cluster_labels.loc[
            (cluster_labels['cluster'] == cf_cluster) & (cluster_labels['characteristic'].isin(features)),
            'characteristic'
        ].tolist()

        # Filtrer for kun at bruge de kolonner, der findes i cf
        available_chars = [col for col in chars_sub if col in cf.columns]

        # Lav en DataFrame med de originale feature-værdier, men med 'id' omdøbt til 'id_shuffle'
        chars_data = cf[['id', 'eom'] + available_chars].copy().rename(columns={'id': 'id_shuffle'})
        # Fjern de oprindelige feature-kolonner fra cf
        cf = cf.drop(columns=available_chars)
        # Merge cf med chars_data på ['id_shuffle', 'eom']
        cf = pd.merge(cf, chars_data, on=['id_shuffle', 'eom'], how='left')

        # Forventede afkast: for hver model i er_models
        for m_sub in er_models.values():
            # Debug-print
            try:
                pred = m_sub['pred']
            except Exception as e:
                pred = None
            try:
                sub_dates = np.unique(pred['eom'])
            except Exception as e:
                sub_dates = np.array([pred])

            # Vælg rækker fra cf, hvor 'eom' er i sub_dates, og udtræk kolonnerne angivet i features
            cf_x = cf.loc[cf['eom'].isin(sub_dates), features].to_numpy()
            # Anvend rff-transformation med modelens W og passér p
            rff_x = rff(cf_x, p=m_sub['opt_hps']['p'], W=m_sub['W'])
            # Kombiner cosine- og sine-komponenter og gang med m_sub['opt_hps']['p']^(-0.5)
            cf_new_x = (m_sub['opt_hps']['p'] ** (-0.5)) * np.hstack([rff_x['X_cos'], rff_x['X_sin']])
            # Kalder predict() uden 'newx' og 's' keywords (sklearn Ridge forventer blot X)
            pred_values = m_sub['fit'].predict(cf_new_x)
            pred_values = np.ravel(pred_values)
            cf.loc[cf['eom'].isin(sub_dates), 'pred_ld1'] = pred_values
    else:
        cf = data[data['valid'] == True].copy()

    # Implementér TPF på de kontrafaktuelle data.
    op = tpf_implement(cf, cov_list=cov_list, wealth=wealth, dates=dates, gam=gamma_rel)
    op_pf = op['pf'].copy()
    op_pf['cluster'] = cf_cluster
    return op_pf


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


def pfml_search_coef(pfml_input, p_vec, l_vec, hp_years, orig_feat):
    """
    Beregner de bedste koefficienter for hver p og l for hvert hp-år.

    pfml_input: en dictionary med nøglen "reals". Forvent, at pfml_input["reals"] er en dictionary,
                hvor nøglerne er datoer (som pd.Timestamp eller kan konverteres til sådanne) og
                værdierne er dictionaries med mindst:
                  - "r_tilde": en numpy-array (vektor)
                  - "denom": en numpy-matrix
    p_vec: liste over p-værdier.
    l_vec: liste over l-værdier.
    hp_years: liste over hp-år (f.eks. årstal som integers).
    orig_feat: ekstra argument til pfml_feat_fun.

    Returnerer:
      coef_list: en dictionary, hvor nøglerne er hp-år (som str) og værdierne er dictionaries,
                 der for hver p (som str) indeholder en liste med koefficienter for hver l-værdi.
    """
    # Første del: Udregn gennemsnit før hp_years
    # Konverter nøglerne til pd.Timestamp (hvis de ikke allerede er)
    d_all = np.array([pd.to_datetime(k) for k in pfml_input["reals"].keys()])

    end_bef = min(hp_years) - 1
    # Filter: træningsdata før (end_bef-1)-12-31 (<, da returns er fra eom+months(1))
    cutoff_date = pd.to_datetime(f"{end_bef - 1}-12-31")
    train_bef = {d: pfml_input["reals"][d] for d in pfml_input["reals"]
                 if pd.to_datetime(d) < cutoff_date}

    # Sum af r_tilde-vektorer over træningsdataene
    # For hver træningsdato antages x["r_tilde"] at være en numpy-array
    r_tilde_sum = None
    for x in train_bef.values():
        if r_tilde_sum is None:
            r_tilde_sum = np.array(x["r_tilde"])
        else:
            r_tilde_sum += np.array(x["r_tilde"])

    # Beregn summen af 'denom'-matricerne over træningsdataene vha. denom_sum_fun
    denom_raw_sum = denom_sum_fun(train_bef)

    n = len(train_bef)

    # Sortér hp_years
    hp_years = sorted(hp_years)
    coef_list = {}  # Vil indeholde koefficienter for hvert hp-år (som str)

    for end in hp_years:
        # For dette hp-år
        # Træningsdata for d mellem (end-2)-12-31 og (end-1)-11-30
        lower_bound = pd.to_datetime(f"{end - 2}-12-31")
        upper_bound = pd.to_datetime(f"{end - 1}-11-30")
        train_new = {d: pfml_input["reals"][d] for d in pfml_input["reals"]
                     if lower_bound <= pd.to_datetime(d) <= upper_bound}

        n += len(train_new)
        # Opdater r_tilde_sum med nye data
        for x in train_new.values():
            r_tilde_sum += np.array(x["r_tilde"])
        # Opdater denom_raw_sum
        denom_raw_new = denom_sum_fun(train_new)
        denom_raw_sum += denom_raw_new

        # For hver p-værdi, beregn koefficienterne for hver l-værdi
        coef_by_hp = {}
        for p in p_vec:
            feat_p = pfml_feat_fun(p=p, orig_feat=orig_feat)  # skal returnere en liste/array af indekser
            feat_indices = [feat_p.index(f) for f in feat_p if f in feat_p]
            r_tilde_sub = r_tilde_sum[feat_indices] / n
            denom_sub = denom_raw_sum.to_numpy()[np.ix_(feat_indices, feat_indices)] / n
            # For hver l-værdi, beregn koefficienten som: solve(denom_sub + l*diag(p+1)) %*% r_tilde_sub
            # Bemærk: diag(p+1) antages at skabe en diagonal matrix med dimension len(feat_p)
            results = {}
            for l in l_vec:
                A = denom_sub + l * np.eye(len(feat_p))
                coef = np.linalg.solve(A, r_tilde_sub)
                results[str(l)] = coef
            coef_by_hp[str(p)] = results
        coef_list[str(end)] = coef_by_hp

    return coef_list


# Feature names
def pfml_feat_fun(p, orig_feat, features):
    feat = ["constant"]
    if p != 0:
        # Antag at p er et lige tal, så vi deler med 2 og laver features for cos og sin.
        feat += [f"rff{i}_cos" for i in range(1, int(p/2)+1)]
        feat += [f"rff{i}_sin" for i in range(1, int(p/2)+1)]
    if orig_feat:
        feat += features  # 'features' skal være defineret globalt (f.eks. en liste)
    return feat


def pfml_w(data, dates, cov_list, lambda_list, gamma_rel, iter, risk_free, wealth, mu, aims):
    """
    Beregner Portfolio-ML weights.

    Parametre:
      data      : DataFrame med f.eks. kolonnerne 'id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0'.
      dates     : Liste (eller array) af datoer (f.eks. som str i formatet 'YYYY-MM-DD').
      cov_list  : Dictionary med kovariansdata, nøgler svarende til datoer (som str eller pd.Timestamp).
      lambda_list: Dictionary med lambda-data, nøgler svarende til datoer.
      gamma_rel : Skalar.
      iter      : Antal iterationer til m_func.
      risk_free : DataFrame med kolonnerne 'eom' og 'rf'.
      wealth    : DataFrame med kolonnerne 'eom' og 'wealth'.
      mu        : Skalar (forventet afkast ud over rf).
      aims      : Aim-portefølje; hvis None, oprettes den ud fra signal_t og aim_coef (se nedenfor).

    Returnerer:
      fa_weights : DataFrame med de opdaterede vægte.
    """

    # --- Opret Aim Portfolio, hvis ikke angivet ---
    # Her antages det, at der findes en global dictionary 'signal_t' og en variabel 'aim_coef'
    # Som i R: hvis aims er None, så opret den ud fra hver dato i dates.
    if aims is None:
        aims_list = []
        for d in dates:
            # Filtrer data for denne dato
            data_d = data[data['eom'] == d][['id', 'eom']].copy()
            # Hent signal for d fra den globale signal_t
            s = signal_t.get(d, None)
            if s is None:
                print(f"Ingen signal fundet for dato {d}")
                continue
            # Hvis aim_coef er en liste (eller dict) med årstal, så brug koefficient for året for d
            # Vi antager, at d kan konverteres til et årstal.
            year_d = pd.to_datetime(d).year
            if isinstance(aim_coef, dict):
                coef = aim_coef.get(str(year_d), aim_coef)  # Hvis nøglen ikke findes, brug hele aim_coef
            else:
                coef = aim_coef
            # Beregn w_aim = s @ coef. For at fjerne overflødige dimensioner anvendes .squeeze()
            data_d['w_aim'] = np.squeeze(s @ coef)
            aims_list.append(data_d)
        if len(aims_list) > 0:
            aims = pd.concat(aims_list, ignore_index=True)
        else:
            aims = None

    # --- Start med initial weights: Brug VW (vægtet) hvis første observation, ellers brug eksisterende weights ---
    # Antag, at initial_weights_new er en hjælpefunktion, der returnerer en DataFrame med kolonnerne 'id', 'eom', 'w_start'
    fa_weights = General_Functions.initial_weights_new(data, w_type="vw")
    # Merge med de kolonner, der er nødvendige fra data (f.eks. 'tr_ld1')
    fa_weights = pd.merge(fa_weights, data[['id', 'eom', 'tr_ld1']], on=['id', 'eom'], how='left')
    # Merge med wealth for at få 'mu_ld1'
    fa_weights = pd.merge(fa_weights, wealth[['eom', 'mu_ld1']], on='eom', how='left')

    # --- For hver dato i dates: beregn m og opdater vægtene ---
    for d in dates:
        # d er en streng (f.eks. 'YYYY-MM-DD')
        # Opret datoen som pd.Timestamp for sammenligning
        d_ts = pd.to_datetime(d)
        # Hent ids for observationer i data for denne dato
        ids = data[data['eom'] == d]['id'].unique().astype(str).tolist()
        d_str = d.strftime('%Y-%m-%d')
        #     print("d_str")
        #     print(d_str)
        # Opret sigma og lambda for de givne ids. Her antages, at cov_list og lambda_list benyttes med nøglen d.
        # Vi konverterer nøglen til pd.Timestamp for at matche, hvis nødvendigt.
        # sigma = General_Functions.create_cov(cov_list[pd.to_datetime(d)], ids=ids)
        # sigma = General_Functions.create_cov(cov_list[d.strftime('%Y-%m-%d')], ids=ids)  # rettet her sidste # har virket
        sigma = General_Functions.create_cov(cov_list[d_str], ids=ids)
        lambda_series = pd.Series(lambda_list[d_str])
        lambda_series.index = lambda_series.index.astype(str)
        lam = create_lambda(lambda_series, ids=ids)
        w = wealth[wealth['eom'] == d]['wealth'].iloc[0]
        rf_val = risk_free[risk_free['eom'] == d]['rf'].iloc[0]
        m = m_func(w=w, mu=mu, rf=rf_val, sigma_gam=sigma * gamma_rel, gam=gamma_rel,
                                              lambda_mat=lam, iter=iter)
        # lam = create_lambda(lambda_list[pd.to_datetime(d)], ids=ids) # Gamle funktion
        # Hent wealth og rf for denne dato
        iden = np.eye(m.shape[0])

        # Hent de nuværende weights fra fa_weights for denne dato og merge med aims (baseret på id og eom)
        # Vi antager, at 'aims' har kolonnerne 'id', 'eom', 'w_aim'
        w_cur = pd.merge(fa_weights[fa_weights['eom'] == d], aims, on=['id', 'eom'], how='left')
        # Beregn w_opt: m @ w_start + (iden - m) @ w_aim
        # Her antages, at w_start og w_aim er kolonner i w_cur, og at de er arrays (eller skalarer)
        w_cur['w_opt'] = (m @ w_cur['w_start'].values) + ((iden - m) @ w_cur['w_aim'].values)
        print(w_cur['w_opt'])
        # Update weight matrix:
        # Find næste måned: antag, at dates er sorteret; brug listeindeks
        # Beregn næste måned direkte fra dates-listen
        # Bestem næste måneds dato (som streng)
        try:
            current_index = dates.get_loc(d)
            next_month = dates[current_index + 1]
            next_month_str = next_month.strftime('%Y-%m-%d')
            next_month = next_month_str
            print("next_month")
            print(next_month)
        except IndexError:
            next_month = None

        # Beregn w_opt_ld1 = w_opt * (1 + tr_ld1) / (1 + mu_ld1)
        w_cur['w_opt_ld1'] = w_cur['w_opt'] * (1 + w_cur['tr_ld1']) / (1 + w_cur['mu_ld1'])
        # print(w_cur.dtypes)

        fa_weights = pd.merge(w_cur[['id', 'w_opt', 'w_opt_ld1', 'eom']],
                              fa_weights, on=['id', 'eom'], how='right')

        mask_current = (fa_weights['eom'] == d) & (fa_weights['w_opt'].notna())
        fa_weights.loc[mask_current, 'w'] = fa_weights.loc[mask_current, 'w_opt']
        # print("fa_weights 2")
        # print(fa_weights)

        # For rækker med eom == next_month, sæt w_start til w_opt_ld1
        if next_month is not None:
            # Lav en kopi af w_cur og opdater 'eom' til next_month
            w_cur_next = w_cur.copy()
            # Bevar datatype som datetime
            w_cur_next['eom'] = pd.to_datetime(next_month)
            # print(w_cur_next.dtypes)

            # Merge w_cur_next med fa_weights for at få next_month-værdierne med w_opt_ld1
            fa_weights = pd.merge(w_cur_next[['id', 'w_opt_ld1', 'eom']],
                                  fa_weights, on=['id', 'eom'], how='right', suffixes=('_next', ''))

            mask_next = (fa_weights['eom'] == pd.to_datetime(next_month)) & (fa_weights['w_opt_ld1_next'].notna())
            fa_weights.loc[mask_next, 'w_start'] = fa_weights.loc[mask_next, 'w_opt_ld1_next']

            mask_next_na = (fa_weights['eom'] == pd.to_datetime(next_month)) & (fa_weights['w_start'].isna())
            fa_weights.loc[mask_next_na, 'w_start'] = 0
            fa_weights.drop(columns=['w_opt_ld1_next'], inplace=True)

        fa_weights.drop(columns=['w_opt', 'w_opt_ld1'], inplace=True)

    return fa_weights

#Portfolio-ML Inputs
def pfml_input_fun(data_tc, cov_list, lambda_list, gamma_rel, wealth, mu, dates, lb, scale,
                   risk_free, features, rff_feat, seed, p_max, g, add_orig, iter, balanced):
    # Konverter nøglerne i cov_list og lambda_list til pd.Timestamp
    cov_list = {pd.to_datetime(key): value for key, value in cov_list.items()}
    lambda_list = {pd.to_datetime(key): value for key, value in lambda_list.items()}

    # --- Lookback-datoer ---
    min_date = min(dates)
    max_date = max(dates)
    start_date = (min_date + pd.Timedelta(days=1)) - MonthEnd(lb + 1)
    dates_lb = pd.date_range(start=start_date, end=max_date, freq=MonthEnd())

    # --- Oprettelse af Random Fourier Features ---
    if rff_feat:
        np.random.seed(seed)
        X_features = data_tc[features].values
        rff_x = rff(X_features, p=p_max, g=g)  # Forudsætter, at funktionen rff er defineret
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

    # Konverter 'eom' til str for konsistens
    data['eom'] = pd.to_datetime(data['eom']).dt.strftime('%Y-%m-%d')

    feat_cons = feat_new + ['constant']

    # --- Tilføjelse af scales ---
    if scale:
        scales_list = []
        for d in dates_lb:
            d_str = d.strftime('%Y-%m-%d')
            # Bemærk: Her benyttes pd.to_datetime(d_str) for at sikre, at nøglen passer til cov_list
            sigma = General_Functions.create_cov(cov_list[pd.to_datetime(d_str)])
            if hasattr(sigma, 'values'):
                sigma_vals = sigma.values
                ids_sigma = sigma.index.astype(float)
            else:
                sigma_vals = sigma
                ids_sigma = np.arange(sigma.shape[0])
            diag_vol = np.sqrt(np.diag(sigma_vals))
            df_scales = pd.DataFrame({
                'id': ids_sigma,
                'eom': d_str,
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
        d_str = d.strftime('%Y-%m-%d')
        data_ret = data[(data['valid'] == True) & (data['eom'] == d_str)][['id', 'ret_ld1']]
        ids = data_ret['id'].unique()  # ids som ints
        n = len(ids)
        r_vec = data_ret['ret_ld1'].values

        # Her skal vi bruge cov_list med en nøgle, der er pd.Timestamp
        sigma = General_Functions.create_cov(cov_list[pd.to_datetime(d_str)], ids=ids)
        lambda_series = pd.Series(lambda_list[pd.to_datetime(d_str)])
        lambda_mat = create_lambda(lambda_series, ids=ids)

        w = wealth.loc[wealth['eom'] == pd.to_datetime(d_str), 'wealth'].iloc[0]
        rf_val = risk_free.loc[risk_free['eom'] == d_str, 'rf'].iloc[0]
        m = m_func(w=w, mu=mu, rf=rf_val, sigma_gam=sigma * gamma_rel, gam=gamma_rel,
                                              lambda_mat=lambda_mat, iter=iter)

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
        groups = {k: group for k, group in data_sub.groupby('eom')}

        signals = {}
        for eom_val, group in groups.items():
            s = group[feat_cons].values
            if scale:
                s = np.diag(1 / group['vol_scale'].values) @ s
            if np.isnan(s).any() or np.isinf(s).any():
                print(f"Advarsel: Signal for {eom_val} indeholder NaN eller Inf.")
            signals[eom_val] = s

        d_key = d_str
        signal_current = signals.get(d_key, None)
        if signal_current is None:
            print(f"Advarsel: Ingen signal fundet for {d_key}.")
            continue
        elif np.isnan(signal_current).any() or np.isinf(signal_current).any():
            print(f"Advarsel: Signal for {d_key} indeholder NaN eller Inf, springer denne dato over.")
            continue

        gtm = {}
        for eom_val, group in groups.items():
            gt = (1 + group['tr_ld0']) / (1 + group['mu_ld0'])
            gt = gt.fillna(1).values
            gtm[eom_val] = m @ np.diag(gt)

        n_stocks = n
        gtm_agg = {}
        gtm_agg_l1 = {}
        gtm_agg[d_key] = np.eye(n_stocks)
        gtm_agg_l1[d_key] = np.eye(n_stocks)
        for i in range(1, lb + 1):
            d_i = (d - MonthEnd(i)).strftime('%Y-%m-%d')
            if d_i in gtm:
                gtm_agg[d_i] = gtm_agg[list(gtm_agg.keys())[-1]] @ gtm[d_i]
            else:
                gtm_agg[d_i] = gtm_agg[list(gtm_agg.keys())[-1]]
            d_i_l1 = (d - MonthEnd(i + 1)).strftime('%Y-%m-%d')
            if d_i_l1 in gtm:
                gtm_agg_l1[d_i] = gtm_agg_l1[list(gtm_agg_l1.keys())[-1]] @ gtm[d_i_l1]
            else:
                gtm_agg_l1[d_i] = gtm_agg_l1[list(gtm_agg_l1.keys())[-1]]

        omega_sum = None
        const_sum = None
        omega_l1_sum = None
        const_l1_sum = None
        for i in range(0, lb + 1):
            d_i = (d - MonthEnd(i)).strftime('%Y-%m-%d')
            s_i = signals.get(d_i, None)
            if s_i is None:
                # print(f"Advarsel: Ingen signal fundet for lookback-dato {d_i}")
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
                # print(f"Advarsel: Ingen signal fundet for lookback-dato {d_i_l1}")
                term_l1 = np.zeros((n_stocks, len(feat_cons)))
            else:
                term_l1 = gtm_agg_l1.get(d_i_l1, np.eye(n_stocks)) @ s_i_l1
            if omega_l1_sum is None:
                omega_l1_sum = term_l1
                const_l1_sum = gtm_agg_l1.get(d_i_l1, np.eye(n_stocks))
            else:
                omega_l1_sum = omega_l1_sum + term_l1
                const_l1_sum = const_l1_sum + gtm_agg_l1.get(d_i_l1, np.eye(n_stocks))

        omega_final = np.linalg.solve(const_sum, omega_sum)
        omega_l1_final = np.linalg.solve(const_l1_sum, omega_l1_sum)

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

        reals_dict[d_str] = reals
        signal_t_dict[d_str] = signals.get(d_str, None)

    return {"reals": reals_dict, "signal_t": signal_t_dict, "rff_w": rff_w if rff_feat else None}


def pfml_cf_fun(data, cf_cluster, pfml_base, dates, cov_list, scale, orig_feat, gamma_rel,
                wealth, risk_free, mu, iter_, seed, features, lambda_list, dates_oos, cluster_labels):
    """
    Portefølje-ML counterfactual funktion.

    Parametre:
      cf_cluster : Navn på den cluster, der skal benyttes (fx "quality", "value", etc.).
      pfml_base  : Objekt (f.eks. et dictionary) med en Portfolio-ML base, der indeholder best_hps_list og hps.
      cov_list   : Dictionary med kovariansmatricer (nøgler som datoer).
      scale      : Bool, om skalering skal udføres.
      orig_feat  : Bool, om originale features skal medtages.
      gamma_rel  : Den aktuelle gamma-værdi.
      cluster_labels : pandas DataFrame med kolonnerne "cluster" og "characteristic".

    Returnerer:
      pf_cf : En pandas DataFrame med den counterfactual portefølje for cf_cluster.
    """
    np.random.seed(seed)

    # Hvis scale: byg en 'scales'-tabel for hver dato
    if scale:
        scales_list = []
        for d in dates:
            # Her antages cov_list er en dictionary med nøgler, der kan konverteres til str(d)
            sigma = create_cov(cov_list[str(d)])  # create_cov skal være defineret
            # Forvent, at sigma er en pandas DataFrame med index som 'id'
            diag_vol = np.sqrt(np.diag(sigma))
            df_scales = pd.DataFrame({
                'id': sigma.index.astype(float),
                'eom': pd.to_datetime(d),
                'vol_scale': diag_vol
            })
            scales_list.append(df_scales)
        scales_df = pd.concat(scales_list, ignore_index=True)
        # Merge med data (forudsætter at 'id' og 'eom' eksisterer)
        data = pd.merge(data, scales_df, on=['id', 'eom'], how='left')
        # Imputer manglende vol_scale med median pr. eom
        data['vol_scale'] = data.groupby('eom')['vol_scale'].transform(lambda x: x.fillna(x.median()))

    # Lav counterfactual data (cf) med udvalgte kolonner
    cf = data[['id', 'eom', 'vol_scale'] + features].copy()

    if cf_cluster != "bm":
        # Shuffle id'er inden for hver eom for at få reproducerbarhed
        cf = cf.groupby('eom', group_keys=False).apply(lambda df: df.assign(id_shuffle=np.random.permutation(df['id'])))
        # Udvælg karakteristika for cf_cluster fra cluster_labels
        chars_sub = cluster_labels.loc[cluster_labels['cluster'] == cf_cluster, 'characteristic'].tolist()
        # Vælg de relevante kolonner og omdøb 'id' til 'id_shuffle'
        chars_data = cf[['id', 'eom'] + chars_sub].rename(columns={'id': 'id_shuffle'})
        # Merge chars_data tilbage i cf (fjern kolonnerne med chars_sub i cf før merge)
        cf = pd.merge(cf.drop(columns=chars_sub), chars_data, on=['id_shuffle', 'eom'], how='left')

    # Beregn counterfactual aim portfolio for hver dato i 'dates'
    aim_cf_list = []
    for d in dates:
        # Filtrer cf for dato d
        cf_d = cf[pd.to_datetime(cf['eom']) == pd.to_datetime(d)]
        # Stocks er de aktuelle id'er
        stocks = cf_d['id'].tolist()
        # Hent base hyperparametre for dato d fra pfml_base
        best_hps = pfml_base['best_hps_list'][str(d)]
        best_g = best_hps['g']
        best_p = best_hps['p']
        aim_coef = best_hps['coef']  # Forventet som en numpy array
        # Hent rff-vægtene: antag pfml_base['hps'] er et dictionary med nøgler som str(best_g)
        W = pfml_base['hps'][str(best_g)]['rff_w'][:, :int(best_p / 2)]

        # Beregn rff for cf_d for de givne features med W (forvent at rff returnerer en dict med 'X_cos' og 'X_sin')
        rff_x = rff(cf_d[features].values, p=None, g=None, W=W)
        # Kombiner X_cos og X_sin til signalmatrix
        s = np.hstack([rff_x['X_cos'], rff_x['X_sin']])
        # Opret DataFrame med passende kolonnenavne
        col_names = [f"rff{i}_cos" for i in range(1, int(best_p / 2) + 1)] + [f"rff{i}_sin" for i in
                                                                              range(1, int(best_p / 2) + 1)]
        s_df = pd.DataFrame(s, columns=col_names, index=cf_d.index)
        # Demean: træk kolonnegennemsnit fra hver kolonne
        s_df = s_df - s_df.mean()
        # Scale: multiplicer hver kolonne med sqrt(1/sum(x^2))
        s_df = s_df.apply(lambda x: x * np.sqrt(1 / np.sum(x ** 2)) if np.sum(x ** 2) != 0 else x)
        # Reorder kolonnerne, fx ved at kalde pfml_feat_fun (forvent, at den returnerer en liste med ønskede feature-navne)
        feat = pfml_feat_fun(best_p, orig_feat)
        s_df = s_df[feat]
        if scale:
            # Anvend skalering: multiplicer med diagonal matrix 1/vol_scale for cf_d
            scales_arr = np.diag(1 / cf_d['vol_scale'].values)
            s_df = pd.DataFrame(scales_arr @ s_df.values, columns=s_df.columns, index=s_df.index)
        # Beregn aim-vægten: w_aim = s_df dot aim_coef
        w_aim = s_df.values @ aim_coef
        cf_d = cf_d[['id', 'eom']].copy()
        cf_d['w_aim'] = w_aim
        aim_cf_list.append(cf_d)
    aim_cf = pd.concat(aim_cf_list, ignore_index=True)

    # Beregn counterfactual portefølje ved at kalde pfml_w på out-of-sample data
    data_oos = data[(pd.to_datetime(data['eom']).isin(pd.to_datetime(dates_oos))) & (data['valid'] == True)]
    w_cf = pfml_w(data_oos, dates=dates, cov_list=cov_list, lambda_list=lambda_list, gamma_rel=gamma_rel,
                  iter_=iter_, risk_free=risk_free, wealth=wealth, mu=mu, aims=aim_cf)
    pf_cf = pf_ts_fun(w_cf, data=data, wealth=wealth, gam=gamma_rel)
    pf_cf['type'] = "Portfolio-ML"

    pf_cf['cluster'] = cf_cluster
    return pf_cf



