import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

# Funktion til beregning af M
def m_func(w, mu, rf, sigma_gam, gam, lambda_matrix, iterations):
    n = lambda_matrix.shape[0]
    g_bar = np.ones(n)
    mu_bar_vec = np.ones(n) * (1 + rf + mu)
    sigma_gr = (1 + rf + mu)**-2 * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)
    lamb_neg05 = np.diag(np.diag(lambda_matrix)**-0.5)
    x = w**-1 * lamb_neg05 @ sigma_gam @ lamb_neg05
    y = np.diag(1 + np.diag(sigma_gr))
    sigma_hat = x + np.diag(1 + g_bar)
    m_tilde = 0.5 * (sigma_hat - sqrtm(sigma_hat @ sigma_hat - 4 * np.eye(n)))
    for _ in range(iterations):
        m_tilde = np.linalg.inv(x + y - m_tilde @ sigma_gr)
    return lamb_neg05 @ m_tilde @ np.sqrt(lambda_matrix)

# Statisk M
def m_static(sigma_gam, w, lambda_matrix, phi):
    return np.linalg.inv(sigma_gam + w / phi * lambda_matrix) @ (w / phi * lambda_matrix)

## Benchamark porteføljer
# Generisk vægtfunktion
def w_fun(data, dates, w_opt, wealth):
    data = data.copy()
    dates_prev = pd.DataFrame({'eom': dates, 'eom_prev': dates.shift(1)})
    data = initial_weights_new(data, w_type="vw").drop(columns=["w"])
    data = pd.merge(data, w_opt, on=["id", "eom"], how="left")
    data = pd.merge(data, dates_prev, on="eom", how="left")
    w_opt = pd.merge(w_opt, data[['id', 'eom', 'tr_ld1']], on=["id", "eom"], how="left")
    w_opt = pd.merge(w_opt, wealth[['eom', 'mu_ld1']], on="eom", how="left")
    w_opt["w"] = w_opt["w"] * (1 + w_opt["tr_ld1"]) / (1 + w_opt["mu_ld1"])
    w_opt.rename(columns={"w": "w_prev", "eom": "eom_prev"}, inplace=True)
    data = pd.merge(data, w_opt, on=["id", "eom_prev"], how="left")
    data.loc[data["eom"] != data["eom"].min(), "w_start"] = data["w_prev"]
    data["w_start"].fillna(0, inplace=True)
    data.drop(columns=["eom_prev", "w_prev"], inplace=True)
    return data

# Implementering af tangensportefølje
def tpf_implement(data, cov_list, wealth, dates, gam):
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates), ["id", "eom", "me", "tr_ld1", "pred_ld1"]].sort_values(by=["id", "eom"])
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
def ew_implement(data, wealth, dates):
    data_rel = data.loc[data["valid"] & data["eom"].isin(dates)]
    ew_opt = data_rel.groupby("eom").apply(lambda x: pd.DataFrame({
        "id": x["id"],
        "eom": x["eom"],
        "w": 1 / len(x)
    })).reset_index(drop=True)

    ew_w = w_fun(data, dates, ew_opt, wealth)
    ew_pf = pf_ts_fun(ew_w, data, wealth, gam=None)  # Assuming `gam` is predefined or unnecessary
    ew_pf["type"] = "1/N"

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
