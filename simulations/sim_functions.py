import numpy as np
import pandas as pd


def sim_prep_data(t, n, disp_x, tv_sds, tv_thetas, tv_rps, dolvol, dates_full, rf, seed, add_noise_feat=True,
                  feat_standardize=True):
    np.random.seed(seed)

    # Økonomiske parametre
    dt = 1
    mu_x = 1
    rpx = 0.05  # Årlig risikopræmie for X
    n_tv = len(tv_sds)
    features = ["x"] + [f"tv{k + 1}" for k in range(n_tv)]

    # Skaber en konstant faktor, der bestemmer kovarianser
    X = np.random.normal(mu_x, disp_x, size=(t, n))

    # Skaber flere tidsvarierende faktorer (tv_list)
    tv_list = []
    for k in range(n_tv):
        tv = np.zeros((t, n))
        tv[0, :] = np.random.normal(0, tv_sds[k], size=n)  # Initialværdi
        for j in range(1, t):
            tv[j, :] = tv[j - 1, :] + (1 - tv_thetas[k]) * (0 - tv[j - 1, :]) * dt + tv_sds[k] * np.random.normal(0,
                                                                                                                  np.sqrt(
                                                                                                                      dt),
                                                                                                                  size=n)
        tv_list.append(tv)

    # Simulerer afkast
    sig_m = 0.2
    sig_e = 0.4
    ivol = np.diag((sig_e ** 2) / 12 * np.ones(n))
    mkt_ret = np.random.normal(0, sig_m / np.sqrt(12), size=t)  # Markedsafkast
    e_ret = np.random.normal(0, sig_e / np.sqrt(12), size=(t, n))  # Individuel støj

    # Data gemmes i en liste af dataframes
    data_list = []
    for j in range(t):
        x = X[j, :]
        sigma = np.outer(x, x) * (sig_m ** 2 / 12) + ivol  # Kovariansmatrix
        mu = (rpx / 12) * x  # Forventet afkast

        for k in range(n_tv):
            mu += (tv_rps[k] / 12) * tv_list[k][j, :]

        # Realiserede afkast
        r = mu + x * mkt_ret[j] + e_ret[j, :]
        w_tpf = np.linalg.solve(sigma, mu)  # Markowitz-optimering

        # Dataframe for denne periode
        df = pd.DataFrame({
            "eom": dates_full[j],
            "id": np.arange(1, n + 1),
            "x": x,
            "ret_ld1": r,
            "er_true": mu,
            "sr": mu / np.sqrt(np.diag(sigma)),
            "w_tpf": w_tpf
        })

        # Tilføjer de tidsvarierende faktorer
        for k in range(n_tv):
            df[f"tv{k + 1}"] = tv_list[k][j, :]

        data_list.append(df)

    # Samler alle perioder i én dataframe
    data_tc = pd.concat(data_list, ignore_index=True)

    # Tilføjer støjfeatures, hvis nødvendigt
    if add_noise_feat:
        for i in range(1, 6):
            data_tc[f"feat{i}"] = np.random.normal(size=len(data_tc))
        features.extend([f"feat{i}" for i in range(1, 6)])

    data_tc["me"] = 1  # Markedsværdi

    # Beregner transaktionsomkostninger
    data_tc["lambda"] = 0.02 / dolvol
    data_tc["tr_ld1"] = data_tc["ret_ld1"] + rf
    data_tc["tr_ld0"] = data_tc.groupby("id")["tr_ld1"].shift(1).fillna(0)

    # Forudsigelser for fremtidige afkast
    for h in range(1, 13):
        data_tc[f"pred_ld{h}"] = (rpx / 12) * data_tc["x"]
        for k in range(n_tv):
            data_tc[f"pred_ld{h}"] += (tv_rps[k] / 12) * (
                    tv_thetas[k] ** (h - 1) * data_tc[f"tv{k + 1}"] + (1 - tv_rps[k] ** (h - 1)) * 0
            )

    # Beregner gennemsnitlige forudsigelser
    data_tc["pred_ld2_6"] = data_tc[[f"pred_ld{h}" for h in range(2, 7)]].mean(axis=1)
    data_tc["pred_ld7_12"] = data_tc[[f"pred_ld{h}" for h in range(7, 13)]].mean(axis=1)

    # Standardiserer features (hvis nødvendigt)
    if feat_standardize:
        for f in features:
            data_tc[f] = data_tc.groupby("eom")[f].transform(lambda x: (x.rank() - 1) / (len(x) - 1))

    # Markerer gyldige observationer
    data_tc["valid"] = True
    data_tc["mu_ld0"] = 0
    data_tc["eom_ret"] = pd.to_datetime(data_tc["eom"]) + pd.DateOffset(months=1)

    # Skaber en liste af kovariansmatricer for hver periode
    barra_cov = {
        dates_full[i]: {
            "fct_load": X[i, :].reshape(-1, 1),
            "fct_cov": np.array([[sig_m ** 2 / 12]]),
            "ivol_vec": np.diag(ivol)
        }
        for i in range(t)
    }

    return {
        "barra_cov": barra_cov,
        "data": data_tc,
        "features": features
    }
