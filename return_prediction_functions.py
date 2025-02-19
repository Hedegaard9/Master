import pandas as pd
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.utils import resample

def data_split(data, type, val_end, val_years, train_start, train_lookback, retrain_lookback, test_inc, test_end):
    """
    Splitter data i train, validation, test og fuld træning baseret på datoer.

    Args:
        data (pd.DataFrame): Dataset med kolonner 'eom' (slutning af måned) og 'eom_pred_last'.
        val_end (str): Slutdato for valideringsperiode (format: 'YYYY-MM-DD').
        val_years (int): Antal år for valideringsperiode.
        train_start (str): Startdato for træning (format: 'YYYY-MM-DD').
        train_lookback (int): Antal år til træningsdata.
        retrain_lookback (int): Antal år til fuld træning.
        test_inc (int): Antal år testdata inkluderer.
        test_end (str): Slutdato for testdata (format: 'YYYY-MM-DD').

    Returns:
        dict: Dictionary med train, validation, test og fuld træning.
    """

    # Konverter datoer til datetime-format
    val_end = pd.to_datetime(val_end)
    train_start = pd.to_datetime(train_start)
    test_end = pd.to_datetime(test_end)

    # Beregn grænser
    train_end = val_end - relativedelta(years=val_years)
    train_start = max(train_start, train_end - relativedelta(years=train_lookback))

    # Filtrer dataset
    op = {}
    op["val"] = data[(data["eom"] >= train_end) & (data["eom_pred_last"] <= val_end)]
    op["train"] = data[(data["eom"] >= train_start) & (data["eom_pred_last"] <= train_end)]
    op["train_full"] = data[(data["eom"] >= (val_end - relativedelta(years=retrain_lookback))) & (data["eom_pred_last"] <= val_end)]
    op["test"] = data[(data["eom"] >= val_end) & (data["eom"] < min(val_end + relativedelta(years=test_inc), test_end))]

    return op

def ols_fit(data, feat):
    """
    Tilpasser en OLS-regression på træningsdata og laver forudsigelser på testdata.

    Args:
        data (dict): Dictionary med trænings- og testdata (output fra `data_split`).
        feat (list): Liste af feature-kolonner, der skal bruges i modellen.

    Returns:
        dict: En dictionary med:
            - "fit": Den tilpassede OLS-model.
            - "pred": DataFrame med forudsigelser på testdata.

    Example:
        split_data = data_split(df, val_end, val_years, train_start, train_lookback, retrain_lookback, test_inc, test_end)
        features = ["feature1", "feature2", "feature3"]
        model_results = ols_fit(split_data, features)

        print(model_results["fit"].summary())  # Se modelresultater
        print(model_results["pred"].head())  # Se forudsigelser
    """

    # Definér trænings- og testdata
    train_data = data["train_full"]
    test_data = data["test"]

    # Tjek om features findes i dataset
    missing_features = [col for col in feat if col not in train_data.columns]
    if missing_features:
        raise ValueError(f"Manglende kolonner i træningsdata: {missing_features}")

    # Konstruér formel til OLS-regression (som i R's `lm` funktion)
    X_train = train_data[feat]
    X_train = sm.add_constant(X_train)  # Tilføj konstant (intercept)
    y_train = train_data["ret_pred"]

    # Fit OLS model
    model = sm.OLS(y_train, X_train).fit()

    # Lav forudsigelser på testdata
    X_test = test_data[feat]
    X_test = sm.add_constant(X_test)
    test_data["pred"] = model.predict(X_test)

    # Returnér model og forudsigelser
    return {
        "fit": model,
        "pred": test_data[["id", "eom", "eom_pred_last", "pred"]]
    }

def rff(X, p=None, g=None, W=None):
    """
    Genererer Random Fourier Features (RFF) fra input-data.

    Args:
        X (np.ndarray): Input-data som en matrix (n x d).
        p (int): Antal Fourier-funktioner (skal være delbart med 2).
        g (float): Varians/skala for vægtmatrixen W.
        W (np.ndarray, optional): Foruddefineret vægtmatrix (d x p/2). Hvis None, genereres tilfældige vægte.

    Returns:
        dict: En dictionary med:
            - "W": Vægtmatrixen (d x p/2).
            - "X_cos": Cosinus-transformeret data (n x p/2).
            - "X_sin": Sinus-transformeret data (n x p/2).

    Example:
        X = np.random.rand(100, 5)  # 100 eksempler, 5 features
        rff_result = rff(X, p=10, g=1.0)
        print("W:", rff_result["W"].shape)
        print("X_cos:", rff_result["X_cos"].shape)
        print("X_sin:", rff_result["X_sin"].shape)
    """

    # Tjek at p er delbart med 2
    if p % 2 != 0:
        raise ValueError("p skal være delbart med 2!")

    # Generér vægtmatrix W, hvis ikke angivet
    if W is None:
        k = X.shape[1]  # Antal features (dimensionen af X)
        W = np.random.multivariate_normal(
            mean=np.zeros(k),
            cov=g * np.eye(k),
            size=p // 2
        ).T  # Transponér så dimensionerne matcher (k x p/2)

    # Beregn X_new = X * W
    X_new = np.dot(X, W)  # Matrix-multiplikation

    # Transformer data til cosinus og sinus
    X_cos = np.cos(X_new)
    X_sin = np.sin(X_new)

    # Returnér vægte og transformeret data
    return {"W": W, "X_cos": X_cos, "X_sin": X_sin}


def rff_hp_search(data, feat, p_vec, g_vec, l_vec, seed):
    """
    Søger efter de optimale hyperparametre for Random Fourier Features og Ridge Regression.

    Args:
        data (dict): Dictionary med 'train', 'val', 'train_full' og 'test' DataFrames.
        feat (list): Liste over feature-kolonner.
        p_vec (list): Liste over værdier for antal Fourier-funktioner (skal være delbare med 2).
        g_vec (list): Liste over værdier for g (skala for vægte).
        l_vec (list): Liste over lambda (regulariseringsparametre) til Ridge Regression.
        seed (int): Seed for tilfældig vægtgeneration.

    Returns:
        dict: Dictionary med:
            - "fit": Den endelige Ridge Regression-model.
            - "pred": Forudsigelser for testdata som en DataFrame.
            - "hp_search": DataFrame med søgning over hyperparametre (g, p, lambda, mse).
            - "W": De optimale vægte (udtrukket fra rff_train for den optimale g).
            - "opt_hps": Den række med de optimale hyperparametre.
    """
    np.random.seed(seed)
    val_errors_list = []
    rff_info = {}  # Gemmer rff_train["W"] for hver g

    # Loop over g-værdier
    for g in g_vec:
        print(f"g: {g}")
        # Generer random Fourier features for træningsdata med p = max(p_vec)
        rff_train = rff(data["train"][feat].values, p=max(p_vec), g=g)
        # Brug de samme vægte til valideringsdata
        rff_val = rff(data["val"][feat].values, p=max(p_vec), W=rff_train["W"])

        # Loop over p-værdier
        for p in p_vec:
            print(f"  --> p: {p}")
            # Skalering: p^(-0.5) gange de første p/2 kolonner af cosine og sine
            half_p = int(p // 2)
            X_train = (p ** -0.5) * np.hstack((rff_train["X_cos"][:, :half_p],
                                               rff_train["X_sin"][:, :half_p]))
            X_val = (p ** -0.5) * np.hstack((rff_val["X_cos"][:, :half_p],
                                             rff_val["X_sin"][:, :half_p]))
            y_train = data["train"]["ret_pred"].values
            y_val = data["val"]["ret_pred"].values

            # Loop over lambda-værdier
            for lam in l_vec:
                model = Ridge(alpha=lam, fit_intercept=False)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                mse = np.mean((preds - y_val) ** 2)
                val_errors_list.append({"g": g, "p": p, "lambda": lam, "mse": mse})

        # Gem W for den aktuelle g (kun én gang pr. g)
        rff_info[str(g)] = rff_train["W"]

    # Saml resultaterne i en DataFrame
    val_errors_df = pd.DataFrame(val_errors_list)
    # Find de hyperparametre med lavest MSE
    opt_idx = val_errors_df["mse"].idxmin()
    opt_hps = val_errors_df.loc[opt_idx]
    print(f"Optimal g: {opt_hps['g']}, p: {opt_hps['p']}, lambda: {opt_hps['lambda']}")

    # Hent de optimale weights for den valgte g: tag de første p/2 kolonner
    opt_g = str(opt_hps["g"])
    opt_p = int(opt_hps["p"])
    half_opt_p = int(opt_p // 2)
    opt_W = rff_info[opt_g][:, :half_opt_p]

    # Re-fit på train_full data
    rff_train_full = rff(data["train_full"][feat].values, W=opt_W)
    X_train_full = (opt_p ** -0.5) * np.hstack((rff_train_full["X_cos"], rff_train_full["X_sin"]))
    y_train_full = data["train_full"]["ret_pred"].values
    final_model = Ridge(alpha=opt_hps["lambda"], fit_intercept=False)
    final_model.fit(X_train_full, y_train_full)

    # Forudsig på testdata
    rff_test = rff(data["test"][feat].values, W=opt_W)
    X_test = (opt_p ** -0.5) * np.hstack((rff_test["X_cos"], rff_test["X_sin"]))
    preds_test = final_model.predict(X_test)

    # Opret en DataFrame med forudsigelser
    pred_op = data["test"][["id", "eom", "eom_pred_last"]].copy()
    pred_op["pred"] = preds_test

    return {
        "fit": final_model,
        "pred": pred_op,
        "hp_search": val_errors_df,
        "W": opt_W,
        "opt_hps": opt_hps
    }

def ridge_hp_search(data, feat, vol_scale, lambdas):
    """
    Søger efter den optimale lambda for Ridge Regression.

    Args:
        data (dict): Dictionary med train, val, test og train_full data.
        feat (list): Liste over feature-kolonner.
        vol_scale (bool): Om forudsigelser skal skaleres med volatilitet (rvol_m).
        lambdas (list): Liste over lambda-værdier til hyperparametertuning.

    Returns:
        dict: Dictionary med:
            - "fit": Den endelige Ridge-model.
            - "hp_search": DataFrame med resultater fra hyperparametertuning.
            - "l_opt": Optimal lambda.
            - "pred": Forudsigelser på testdata.
            - "pred_val": Forudsigelser på valideringsdata.
            - "feat_imp": Feature-importance fra modellen.
    """

    # Træning og valideringsdata
    X_train = data["train"][feat].values
    y_train = data["train"]["ret_pred"].values
    X_val = data["val"][feat].values
    y_val = data["val"]["ret_pred"].values

    # Ridge regression hyperparametertuning
    val_errors = []
    for l in lambdas:
        ridge = Ridge(alpha=l, fit_intercept=True, normalize=True)  # Standardiserer features
        ridge.fit(X_train, y_train)
        y_pred_val = ridge.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        val_errors.append({"lambda": l, "mse": mse})

    # Konverter resultater til DataFrame
    lambda_search = pd.DataFrame(val_errors)

    # Plot fejl vs lambda
    plt.figure(figsize=(8, 6))
    plt.plot(np.log10(lambda_search["lambda"]), lambda_search["mse"], marker="o")
    plt.xlabel("log10(lambda)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Ridge Regression Hyperparameter Tuning")
    plt.grid(True)
    plt.show()

    # Find optimal lambda
    opt_lambda = lambda_search.loc[lambda_search["mse"].idxmin(), "lambda"]

    # Forudsigelser på valideringsdata
    ridge_opt = Ridge(alpha=opt_lambda, fit_intercept=True, normalize=True)
    ridge_opt.fit(X_train, y_train)
    pred_val_op = data["val"][["id", "eom", "eom_pred_last"]].copy()
    pred_val_op["pred"] = ridge_opt.predict(X_val)

    # Re-fit modellen på al træningsdata
    X_train_full = data["train_full"][feat].values
    y_train_full = data["train_full"]["ret_pred"].values
    ridge_opt.fit(X_train_full, y_train_full)

    # Forudsig på testdata
    X_test = data["test"][feat].values
    pred_op = data["test"][["id", "eom", "eom_pred_last"]].copy()
    pred_op["pred"] = ridge_opt.predict(X_test)

    # Feature-importance
    feat_imp = pd.DataFrame({
        "feature": feat,
        "importance": np.abs(ridge_opt.coef_)
    }).sort_values(by="importance", ascending=False)

    # Volatilitetsskala (valgfrit)
    if vol_scale:
        pred_val_op["pred_vol"] = pred_val_op["pred"]
        pred_op["pred_vol"] = pred_op["pred"]
        pred_val_op["pred"] *= data["val"]["rvol_m"].values
        pred_op["pred"] *= data["test"]["rvol_m"].values

    # Returnér resultater
    return {
        "fit": ridge_opt,
        "hp_search": lambda_search,
        "l_opt": opt_lambda,
        "pred": pred_op,
        "pred_val": pred_val_op,
        "feat_imp": feat_imp
    }


def fit_xgb(train, val, params, iter, es, cores, seed):
    """
    Træner en XGBoost-model med de angivne parametre.

    Args:
        train (xgb.DMatrix): Træningsdata som DMatrix.
        val (xgb.DMatrix): Valideringsdata som DMatrix.
        params (dict): Dictionary med XGBoost-hyperparametre.
        iter (int): Maksimalt antal træningsiterationer.
        es (int): Tidlig stoppetolerance (early stopping rounds).
        cores (int): Antal tråde til beregning.
        seed (int): Seed for reproducerbarhed.

    Returns:
        xgb.Booster: Den trænede XGBoost-model.
    """

    # Definer alle parametre
    params_all = {
        "objective": "reg:squarederror",
        "base_score": 0,
        "eval_metric": "rmse",
        "booster": "gbtree",
        "max_depth": params["tree_depth"],
        "eta": params["learn_rate"],
        "gamma": params["loss_reduction"],
        "subsample": params["sample_size"],
        "colsample_bytree": params["mtry"],
        "min_child_weight": params["min_n"],
        "lambda": params["penalty"],
        "seed": seed,
        "nthread": cores
    }

    # Træn modellen
    evals = [(train, "train"), (val, "val")]
    model = xgb.train(
        params=params_all,
        dtrain=train,
        num_boost_round=iter,
        evals=evals,
        early_stopping_rounds=es,
        verbose_eval=False
    )

    return model


def xgb_hp_search(data, feat, vol_scale, hp_grid, iter, es, cores, seed):
    """
    Udfører hyperparametertuning for XGBoost og finder de bedste parametre.

    Args:
        data (dict): Dictionary med train, val, test, og train_full data.
        feat (list): Liste over feature-kolonner.
        vol_scale (bool): Om forudsigelser skal skaleres med volatilitet (rvol_m).
        hp_grid (pd.DataFrame): Grid med hyperparametre.
        iter (int): Maksimalt antal træningsiterationer.
        es (int): Tidlig stoppetolerance (early stopping rounds).
        cores (int): Antal tråde til beregning.
        seed (int): Seed for reproducerbarhed.

    Returns:
        dict: Dictionary med:
            - "fit": Den endelige XGBoost-model.
            - "best_hp": De bedste hyperparametre.
            - "best_iter": Antal iterationer for den bedste model.
            - "hp_search": DataFrame med hyperparameter-søgningen.
            - "pred": Forudsigelser på testdata.
            - "feat_imp": Global feature-importance.
    """

    np.random.seed(seed)

    # Konverter trænings- og valideringsdata til DMatrix
    train = xgb.DMatrix(data=data["train"][feat].values, label=data["train"]["ret_pred"].values)
    val = xgb.DMatrix(data=data["val"][feat].values, label=data["val"]["ret_pred"].values)

    # Hyperparametertuning
    search_results = []
    for j in range(len(hp_grid)):
        print(f"HP: {j + 1}/{len(hp_grid)}")

        # Konverter række fra hp_grid til parametre
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "max_depth": int(hp_grid.iloc[j]["tree_depth"]),
            "eta": hp_grid.iloc[j]["learn_rate"],
            "gamma": hp_grid.iloc[j]["loss_reduction"],
            "subsample": hp_grid.iloc[j]["sample_size"],
            "colsample_bytree": hp_grid.iloc[j]["mtry"],
            "min_child_weight": hp_grid.iloc[j]["min_n"],
            "lambda": hp_grid.iloc[j]["penalty"],
            "seed": seed,
            "nthread": cores,
        }

        # Træn modellen
        model = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=iter,
            evals=[(train, "train"), (val, "val")],
            early_stopping_rounds=es,
            verbose_eval=False
        )

        # Gem resultater for denne model
        search_results.append({
            "hp_no": j + 1,
            "val_rmse": model.best_score,
            "best_iter": model.best_iteration
        })

    # Konverter resultater til DataFrame
    hp_search = pd.DataFrame(search_results)

    # Visualisering af RMSE over hyperparametre
    plt.figure(figsize=(8, 6))
    plt.plot(hp_search["hp_no"], hp_search["val_rmse"], marker="o")
    plt.xlabel("Hyperparameter-kombination")
    plt.ylabel("Validation RMSE")
    plt.title("Hyperparameter Tuning")
    plt.grid(True)
    plt.show()

    # Find bedste hyperparametre
    best_hp_no = hp_search.loc[hp_search["val_rmse"].idxmin(), "hp_no"]
    best_hp = hp_grid.iloc[best_hp_no - 1].to_dict()
    best_iter = int(hp_search.loc[hp_search["hp_no"] == best_hp_no, "best_iter"])

    print(f"Best HP: {best_hp}")
    print(f"With {best_iter} iterations")

    # Gen-træn på alle træningsdata
    train_full = xgb.DMatrix(data=data["train_full"][feat].values, label=data["train_full"]["ret_pred"].values)
    final_model = xgb.train(
        params=best_hp,
        dtrain=train_full,
        num_boost_round=best_iter,
        evals=[(train_full, "train")],
        verbose_eval=False
    )

    # Feature-importance (SHAP-bidrag)
    sample_data = resample(data["train_full"][feat], n_samples=min(10000, len(data["train_full"])), random_state=seed)
    shap_contrib = final_model.predict(xgb.DMatrix(sample_data), pred_contribs=True)
    global_imp = pd.DataFrame({
        "feature": feat,
        "importance": np.abs(shap_contrib).mean(axis=0)[:-1]  # Ekskluder BIAS
    }).sort_values(by="importance", ascending=False)

    # Visualisering af feature-importance
    top_features = global_imp.head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Global Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Features by Importance")
    plt.gca().invert_yaxis()
    plt.show()

    # Forudsigelser på testdata
    test = xgb.DMatrix(data=data["test"][feat].values)
    pred = final_model.predict(test)
    if vol_scale:
        pred_op = data["test"][["id", "eom"]].copy()
        pred_op["pred_vol"] = pred
        pred_op["pred"] = pred * data["test"]["rvol_m"].values
    else:
        pred_op = data["test"][["id", "eom"]].copy()
        pred_op["pred"] = pred

    # Returnér resultater
    return {
        "fit": final_model,
        "best_hp": best_hp,
        "best_iter": best_iter,
        "hp_search": hp_search,
        "pred": pred_op,
        "feat_imp": global_imp
    }
