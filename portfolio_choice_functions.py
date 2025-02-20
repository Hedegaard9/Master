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
    # Konverter eom og eom_pred_last til datetime, hvis de ikke allerede er det
    data["eom"] = pd.to_datetime(data["eom"])
    data["eom_pred_last"] = pd.to_datetime(data["eom_pred_last"])


    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me", "tr_ld1", "pred_ld1"]].sort_values(
        by=["id", "eom"])
    data_split = {key: group for key, group in data_rel.groupby("eom")}
    tpf_opt = []
    for d in dates:
        data_sub = data_split[d]
        ids = data_sub["id"]
        sigma = create_cov(cov_list[d], ids)
        weights = np.linalg.solve(sigma, data_sub["pred_ld1"]) / gam
        tpf_opt.append(pd.DataFrame({"id": data_sub["id"], "eom": d, "w": weights}))
    tpf_opt = pd.concat(tpf_opt)
    tpf_w = w_fun(data_rel, dates, tpf_opt, wealth)
    tpf_pf = pf_ts_fun(tpf_w, data, wealth, gam)
    tpf_pf["type"] = "Markowitz-ML"
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
def mkt_implement(data, wealth, dates):
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates)]
    mkt_opt = data_rel.groupby("eom").apply(lambda x: pd.DataFrame({
        "id": x["id"],
        "eom": x["eom"],
        "w": x["me"] / x["me"].sum()
    })).reset_index(drop=True)

    mkt_w = w_fun(data, dates, mkt_opt, wealth)
    mkt_pf = pf_ts_fun(mkt_w, data, wealth, gam=None)  # Assuming `gam` is predefined or unnecessary
    mkt_pf["type"] = "Market"

    return {"w": mkt_w, "pf": mkt_pf}

# Rank-Weighted Portfolio Implementation
def rw_implement(data, wealth, dates):
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
    rw_pf = pf_ts_fun(rw_w, data, wealth, gam=None)  # Assuming `gam` is predefined or unnecessary
    rw_pf["type"] = "Rank-ML"

    return {"w": rw_w, "pf": rw_pf}

#Minimum Variance Portfolio
def mv_implement(data, cov_list, wealth, dates):
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me"]]
    data_split = {key: group for key, group in data_rel.groupby("eom")}

    mv_opt = []
    for d in dates:
        data_sub = data_split[d]
        sig = create_cov(cov_list[d], data_sub["id"])
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
            hp = hps[(hps["eom_ret"] < pd.Timestamp(d)) & (hps["eom_ret"] == hps["eom_ret"].max())]
            g = hp["g"].values[0]
            u = hp["u"].values[0]
            k = hp["k"].values[0]

        wealth_t = wealth.loc[wealth["eom"] == pd.Timestamp(d), "wealth"].values[0]
        ids = static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "id"]
        sigma_gam = create_cov(cov_list[d], ids) * gamma_rel
        sigma_gam = sigma_gam_adj(sigma_gam, g, cov_type)
        lambda_matrix = create_lambda(lambda_list[d], ids) * k

        weights = np.linalg.solve(sigma_gam + wealth_t * lambda_matrix,
                                  (u * static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "pred_ld1"] +
                                   wealth_t * lambda_matrix @ static_weights.loc[
                                       static_weights["eom"] == pd.Timestamp(d), "w_start"]))

        static_weights.loc[static_weights["eom"] == pd.Timestamp(d), "w"] = weights

        next_month = dates[dates.index(d) + 1] if dates.index(d) + 1 < len(dates) else None
        if next_month:
            next_weights = (weights * (1 + static_weights["tr_ld1"]) / (1 + static_weights["mu_ld1"]))
            static_weights.loc[static_weights["eom"] == pd.Timestamp(next_month), "w_start"] = next_weights
            static_weights["w_start"].fillna(0, inplace=True)

    return static_weights

#Static Full Implementation
def static_implement(data_tc, cov_list, lambda_list, rf, wealth, mu, gamma_rel, dates_full, dates_oos, dates_hp,
                     hp_years, k_vec, u_vec, g_vec, cov_type, validation=None, seed=None):
    static_hps = pd.DataFrame([(k, u, g) for k in k_vec for u in u_vec for g in g_vec], columns=["k", "u", "g"])
    data_rel = data_tc.loc[
        data_tc["valid"] & data_tc["eom"].isin(dates_hp), ["id", "eom", "me", "tr_ld1", "pred_ld1"]].sort_values(
        by=["id", "eom"])

    if validation is None:
        validation = []
        for i, hp in static_hps.iterrows():
            static_w = static_val_fun(data_rel, dates_hp, cov_list, lambda_list, wealth, cov_type, gamma_rel, hp["k"],
                                      hp["g"], hp["u"])
            portfolio = pf_ts_fun(static_w, data_tc, wealth, gamma_rel)
            portfolio["hp_no"] = i
            portfolio["k"] = hp["k"]
            portfolio["g"] = hp["g"]
            portfolio["u"] = hp["u"]
            validation.append(portfolio)
        validation = pd.concat(validation)

    validation["cum_var"] = validation.groupby("hp_no")["r"].transform(lambda x: np.cumsum(x ** 2) - np.cumsum(x) ** 2)
    validation["cum_obj"] = validation.groupby("hp_no").apply(
        lambda g: g["r"] - g["tc"] - 0.5 * g["cum_var"] * gamma_rel).reset_index(drop=True)
    validation["rank"] = validation.groupby("eom_ret")["cum_obj"].rank(ascending=False)

    optimal_hps = validation.loc[(validation["rank"] == 1) & (validation["eom_ret"].dt.month == 12)].sort_values(
        by="eom_ret")

    final_weights = static_val_fun(data_tc, dates_oos, cov_list, lambda_list, wealth, cov_type, gamma_rel,
                                   hps=optimal_hps)
    final_portfolio = pf_ts_fun(final_weights, data_tc, wealth, gamma_rel)
    final_portfolio["type"] = "Static-ML*"

    return {"hps": validation, "best_hps": optimal_hps, "w": final_weights, "pf": final_portfolio}


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





