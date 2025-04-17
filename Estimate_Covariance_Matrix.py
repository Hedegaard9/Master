import pandas as pd
import numpy as np
from Main import settings, features, pf_set
from pandas.tseries.offsets import MonthEnd
from Prepare_Data import process_risk_free_rate, process_return_data, wealth_func, load_and_filter_market_returns_test, process_all_data, process_cluster_labels
from General_Functions import size_screen_fun, addition_deletion_fun
import statsmodels.formula.api as smf
import sys
from math import sqrt
from tqdm import tqdm
import importlib
ewma = importlib.import_module("ewma")
#sqrtm_cpp = importlib.import_module("sqrtm_cpp")
import pandas as pd
import matplotlib.pyplot as plt
from General_Functions import long_horizon_ret
from Main import settings, features, pf_set
from Prepare_Data import process_risk_free_rate, process_return_data, wealth_func, load_and_filter_market_returns_test, process_all_data, process_cluster_labels
from General_Functions import size_screen_fun, addition_deletion_fun
from plotnine import ggplot, aes, geom_point, labs, theme, element_blank
import numpy as np
from scipy.stats import rankdata
import re
import gc
import ewma

import statsmodels.formula.api as smf
#New version:

def process_cluster_data(chars, daily, cluster_labels_path, factor_details_path):
    """
    Behandler cluster data ved at læse og forberede cluster labels, beregne cluster ranks,
    tilføje industry/market dummies, standardisere faktorer og flette daglige returdata.

    Parametre:
    - chars: DataFrame med karakteristika for aktier
    - daily: DataFrame med daglige afkastdata
    - cluster_labels_path: Sti til CSV-filen med cluster labels
    - factor_details_path: Sti til Excel-filen med factor details

    Returnerer:
    - cluster_data_d: Behandlet DataFrame med alle transformationer
    - ind_factors: Liste med industry/market faktorer
    - clusters: Liste over clusters
    - cluster_data_m: DataFrame med de mellemliggende data (før merging med daglige data)

    Eksempel på brug:
    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"
    cluster_data_d, ind_factors, clusters, cluster_data_m = process_cluster_data(chars, daily, file_path_cluster_labels, file_path_factor_details)
    """
    cluster_labels = process_cluster_labels(cluster_labels_path, factor_details_path)

    valid_chars = chars[chars["valid"] == True]
    cluster_data_m = valid_chars[["id", "eom", "size_grp", "ff12"] + features].copy()

    clusters = cluster_labels["cluster"].unique()
    cluster_ranks = {}

    for cl in clusters:
        chars_sub = cluster_labels[(cluster_labels["cluster"] == cl) & (cluster_labels["characteristic"].isin(features))]

        valid_features = [c for c in chars_sub["characteristic"].values if c in cluster_data_m.columns]
        data_sub = cluster_data_m[valid_features].copy()

        for c in valid_features:
            dir_value = chars_sub.loc[chars_sub["characteristic"] == c, "direction"].values[0]
            if dir_value == -1:
                data_sub[c] = 1 - data_sub[c]

        cluster_ranks[cl] = data_sub.mean(axis=1)

    cluster_ranks_df = pd.DataFrame(cluster_ranks)

    cluster_data_m = cluster_data_m[["id", "eom", "size_grp", "ff12"]].copy()
    cluster_data_m["eom_ret"] = cluster_data_m["eom"] + pd.DateOffset(months=1)
    cluster_data_m["eom_ret"] = cluster_data_m["eom_ret"] + MonthEnd(0)

    cluster_data_m = pd.concat([cluster_data_m, cluster_ranks_df], axis=1)

    if settings["cov_set"]["industries"]:
        industries = sorted(cluster_data_m["ff12"].unique())
        for ind in industries:
            cluster_data_m[str(ind)] = (cluster_data_m["ff12"] == ind).astype(int)
        ind_factors = industries
    else:
        cluster_data_m["mkt"] = 1
        ind_factors = ["mkt"]

    cluster_data_m[clusters] = cluster_data_m.groupby("eom")[clusters].transform(lambda x: (x - x.mean()) / x.std())

    min_eom = cluster_data_m["eom"].min()
    daily_filtered = daily[daily["date"] >= min_eom][["id", "date", "ret_exc", "eom"]].copy()
    daily_filtered.rename(columns={"eom": "eom_ret"}, inplace=True)

    cluster_data_d = cluster_data_m.merge(daily_filtered, on=["id", "eom_ret"], how="inner")

    cluster_data_d.dropna(inplace=True)

    return cluster_data_d, ind_factors, clusters, cluster_data_m


# Create factor returns --------------------

def run_regressions_by_date(cluster_data_d, ind_factors, clusters):
    """
    Kører en lineær regression pr. dato på 'cluster_data_d' og returnerer en DataFrame med resultaterne.

    Parametre:
      - cluster_data_d: DataFrame med mindst kolonnerne 'date' og 'ret_exc', samt de forklarende variable
      - ind_factors: Liste med navne på industry/market faktorer (f.eks. ['BusEq', 'Chems', ...])
      - clusters: Liste med navne på cluster faktorer (f.eks. ['low_leverage', 'investment', ...])

    Returnerer:
      En DataFrame med en række per dato, indeholdende:
        - date: Datoen for gruppen
        - data: Under-DataFrame med de data, der blev brugt for den dato
        - fit: Det tilpassede regressionsmodel-objekt (statsmodels)
        - res: Residualerne fra modellen
        - tidied: En "tidy" tabel med regressionsresultater (koefficienter, standardfejl, t-værdier, p-værdier mv.)
    Example:
    fct_ret_est = run_regressions_by_date(cluster_data_d, ind_factors, clusters)

    """
    all_factors = list(ind_factors) + list(clusters)
    formula = "ret_exc ~ -1 + " + " + ".join(all_factors)

    results = []
    for date, group in cluster_data_d.groupby('date'):
        model = smf.ols(formula=formula, data=group).fit()
        residuals = model.resid
        tidy_df = model.summary2().tables[1].reset_index().rename(columns={'index': 'term'})

        results.append({
            'date': date,
            'data': group,
            'fit': model,
            'res': residuals,
            'tidied': tidy_df
        })

    fct_ret_est = pd.DataFrame(results)
    return fct_ret_est


def create_factor_returns(fct_ret_est):
    """
    Omformer fct_ret_est til et datasæt med faktorafkast:

    For hver række i fct_ret_est:
      - "Unnestes" den nestede DataFrame i kolonnen 'tidied'
      - Kolonnen med koefficienter, som i vores tilfælde hedder "Coef.", omdøbes til "estimate"
      - 'date' tilføjes fra den overordnede række
      - Kun kolonnerne 'date', 'term' og 'estimate' bevares
    Herefter pivotteres dataene, så hver unik værdi i 'term' bliver en kolonne, og vi sorterer efter 'date'.

    Parametre:
      - fct_ret_est: DataFrame med mindst kolonnerne 'date' og 'tidied', hvor 'tidied' er en nestet DataFrame

    Returnerer:
      - fct_ret: Pivotteret DataFrame med én række pr. dato og en kolonne pr. term, indeholdende faktorafkast (estimate)

    Example:
    fct_ret = create_factor_returns(fct_ret_est)
    """
    tidied_list = []
    for _, row in fct_ret_est.iterrows():
        tidy_df = row['tidied'].copy()
        if 'Coef.' in tidy_df.columns:
            tidy_df = tidy_df.rename(columns={'Coef.': 'estimate'})
        tidy_df['date'] = row['date']
        tidied_list.append(tidy_df[['date', 'term', 'estimate']])

    combined = pd.concat(tidied_list, ignore_index=True)

    fct_ret = combined.pivot(index='date', columns='term', values='estimate').reset_index()

    fct_ret = fct_ret.sort_values('date').reset_index(drop=True)

    return fct_ret


# Factor Risk -----------------------------

def weighted_cov(
        df: pd.DataFrame,
        weights: np.ndarray,
        correlation: bool = False,
        center: bool = True,
        unbiased: bool = True
) -> np.ndarray:
    """
    Parametre:
    - df: DataFrame med observationsrækker og faktorkolonner (ingen 'date'-kolonne).
    - weights: Numpy-array med vægte (samme længde som df).
    - correlation: Hvis True, returneres en korrelationsmatrix. Ellers en kovariansmatrix.
    - center: Hvis True, fratrækkes det vægtede gennemsnit af hver kolonne.
    - unbiased: Hvis True, bruger vi (sum(weights) - 1) i nævneren, ellers sum(weights).

    Returnerer:
    - Et numpy 2D-array med kovarians- eller korrelationsmatricen.
    """
    if len(df) != len(weights):
        raise ValueError("Antal rækker i df skal matche længden af weights.")

    w_sum = weights.sum()

    if center:
        w_mean = np.average(df, axis=0, weights=weights)
    else:
        w_mean = np.zeros(df.shape[1])

    X_centered = df - w_mean

    expanded_weights = weights[:, np.newaxis]
    numerator = (X_centered.values * expanded_weights).T @ X_centered.values

    if unbiased:
        denominator = w_sum - 1
    else:
        denominator = w_sum

    cov_mat = numerator / denominator

    if correlation:
        std_diag = np.sqrt(np.diag(cov_mat))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_mat = cov_mat / np.outer(std_diag, std_diag)
            np.fill_diagonal(corr_mat, 1.0)
        return corr_mat
    else:
        return cov_mat


def factor_cov_estimate(calc_dates, fct_dates, fct_ret, w_cor, w_var, settings):
    """
      - For hver dato d i calc_dates:
          1. Find det tidligste tidspunkt (first_obs) i det rullende vindue
          2. Filtrer fct_ret til kun at indeholde data mellem first_obs og d
          3. Beregn vægtet korrelationsmatrix (cor_est) og vægtet variansmatrix (var_est)
          4. Kombiner dem til en endelig kovariansmatrix
          5. Returnér en dict eller liste af kovariansmatricer (én pr. dato)
    """
    factor_cov_est = {}

    for d in calc_dates:
        valid_fct_dates = [fd for fd in fct_dates if fd <= d]
        tail_fct_dates = valid_fct_dates[-settings["cov_set"]["obs"]:]

        if not tail_fct_dates:
            print(f"Advarsel: Ingen tilgængelige datoer for {d}")
            continue

        first_obs = min(tail_fct_dates)
        cov_data = fct_ret[(fct_ret["date"] >= first_obs) & (fct_ret["date"] <= d)]
        t = len(cov_data)

        if t < settings["cov_set"]["obs"] - 30:
            print("WARNING: INSUFFICIENT NUMBER OF OBSERVATIONS!!")

        w_cor_t = w_cor[-t:]
        w_var_t = w_var[-t:]

        factor_cols = [col for col in cov_data.columns if col != "date"]

        cor_est = weighted_cov(
            df=cov_data[factor_cols],
            weights=w_cor_t,
            correlation=True,
            center=True,
            unbiased=True
        )

        var_est = weighted_cov(
            df=cov_data[factor_cols],
            weights=w_var_t,
            correlation=False,
            center=True,
            unbiased=True
        )

        sd_diag = np.diag(np.sqrt(np.diag(var_est)))
        cov_est = sd_diag @ cor_est @ sd_diag

        cov_est_df = pd.DataFrame(cov_est, index=factor_cols, columns=factor_cols)

        factor_cov_est[d] = cov_est_df

    return factor_cov_est

# Specific Risk ---------------------------

def unnest_spec_risk(fct_ret_est):
    """
    Udpakker de nestede 'id' og 'res'-lister i fct_ret_est.

    Forudsætninger:
      - fct_ret_est er en Pandas DataFrame med kolonnerne:
          - 'date': Datoen for gruppen
          - 'data': En nestet DataFrame, som indeholder en kolonne 'id'
          - 'res': En liste eller kolonne med residualer, der svarer til rækkerne i den nestede 'data'

    Funktionen gør følgende:
      1. Arbejder på en kopi af fct_ret_est.
      2. Ekstraherer 'id'-kolonnen fra hver nestet DataFrame i 'data' og gemmer den som en liste i en ny kolonne 'id'.
      3. Beholder kun kolonnerne 'id', 'date' og 'res'.
      4. Kombinerer de to list-kolonner ('id' og 'res') pr. række til en liste af tuples, så parringen bevares.
      5. "Exploder" (unnester) den kombinerede kolonne, så hver tuple bliver til en enkelt række.
      6. Splitter tuple-kolonnen op i to separate kolonner: 'id' og 'res'.
      7. Sorterer resultaterne efter 'id' og 'date' og returnerer den fladede DataFrame.

    Returnerer:
      En Pandas DataFrame med én række pr. observation, der indeholder kolonnerne 'id', 'date' og 'res'.
    """
    df = fct_ret_est.copy()

    df['id'] = df['data'].apply(lambda nested_df: nested_df['id'].tolist())
    spec_risk = df[['id', 'date', 'res']].copy()
    spec_risk['id_res'] = spec_risk.apply(lambda row: list(zip(row['id'], row['res'])), axis=1)
    spec_risk = spec_risk.explode('id_res')
    spec_risk[['id', 'res']] = pd.DataFrame(spec_risk['id_res'].tolist(), index=spec_risk.index)

    spec_risk = spec_risk.drop(columns=['id_res'])
    spec_risk = spec_risk.sort_values(['id', 'date']).reset_index(drop=True)

    return spec_risk


def calculate_ewma(spec_risk, settings):
    """
    Beregner EWMA-volatilitet per gruppe i en DataFrame.

    Args:
        spec_risk (pd.DataFrame): DataFrame med kolonnerne 'id' og 'res'.
        settings (dict): Dictionary med 'cov_set' indeholdende 'hl_stock_var' og 'initial_var_obs'.

    Returns:
        pd.DataFrame: Opdateret DataFrame med en ny kolonne 'res_vol'.

    Example:
    spec_risk_res_vol = calculate_ewma(spec_risk,settings)
    spec_risk_res_vol
    """

    lambda_val = 0.5 ** (1.0 / settings['cov_set']['hl_stock_var'])
    start_val = settings['cov_set']['initial_var_obs']

    def apply_ewma(group):
        group = group.copy()  # Sikrer, at vi ikke ændrer originalen
        group['res_vol'] = ewma.ewma_c(group['res'].values, lambda_val, start_val)
        return group

    return spec_risk.groupby('id', group_keys=False, as_index=False).apply(apply_ewma)


def process_spec_risk_m(fct_dates, spec_risk):
    """
    Transformerer spec_risk-dataene og returnerer to DataFrames:

    1. spec_risk_m: Indeholder for hver id og eom (sidste dag i måneden) den seneste dato samt res_vol.
    2. spec_risk_res_vol: Den transformerede og filtrerede DataFrame.

    Parametre:
      fct_dates : array-lignende objekt med datoer.
      spec_risk : DataFrame med kolonnerne 'id', 'date' og 'res_vol'.

    Returnerer:
      spec_risk_m, spec_risk_res_vol : To DataFrames.

    Example:
    spec_risk_m, spec_risk_res_vol = process_spec_risk_m(fct_dates, spec_risk_res_vol)
    """
    td_range = pd.DataFrame({"date": fct_dates})
    td_range["td_252d"] = td_range["date"].shift(252)
    spec_risk_res_vol = spec_risk.merge(td_range, on="date", how="left")

    spec_risk_res_vol["date_200d"] = spec_risk_res_vol.groupby("id")["date"].shift(200)
    spec_risk_res_vol = spec_risk_res_vol[
        (spec_risk_res_vol["date_200d"] >= spec_risk_res_vol["td_252d"]) &
        (~spec_risk_res_vol["res_vol"].isna())
        ]
    spec_risk_res_vol = spec_risk_res_vol[["id", "date", "res_vol"]]

    spec_risk_res_vol["eom_ret"] = spec_risk_res_vol["date"] + MonthEnd(0)

    spec_risk_res_vol["max_date"] = spec_risk_res_vol.groupby(["id", "eom_ret"])["date"].transform("max")
    spec_risk_m = spec_risk_res_vol[spec_risk_res_vol["date"] == spec_risk_res_vol["max_date"]][
        ["id", "eom_ret", "res_vol"]]
    spec_risk_m = spec_risk_m.rename(columns={"eom_ret": "eom"})

    return spec_risk_m, spec_risk_res_vol


def calculate_barra_cov(calc_dates, cluster_data_m, spec_risk_m, factor_cov_est):
    """
    Beregner stock covariance matrix for hver dato i calc_dates.

    For hver dato d foretages følgende:
      1. Filtrering af cluster_data_m for rækker, hvor 'eom' matcher d.
      2. Merge med spec_risk_m på kolonnerne 'id' og 'eom' for at tilføje specific risk.
      3. Beregning af median specific risk (res_vol) for grupper defineret af 'size_grp' og 'eom'.
      4. Hvis der mangler median, beregnes median for grupper defineret af 'eom' og manglende værdier udfyldes.
      5. Manglende værdier i 'res_vol' erstattes med den beregnede median.
      6. Den faktor-kovariansmatrix, der hentes fra factor_cov_est for datoen, annualiseres ved at gange med 21.
      7. Data sorteres efter 'id', og de kolonner, som svarer til kolonnenavnene i fct_cov, udvælges til at danne faktorbelastningsmatricen X.
      8. Den individuelle varians (ivol_vec) beregnes som res_vol^2 * 21 (annualiseret).

    Returnerer en dictionary, hvor nøglerne er datoerne (som strings i formatet "YYYY-MM-DD"),
    og værdien for hver dato er en dictionary med:
      - "fct_load": Faktorbelastningsmatrix (som en DataFrame med 'id' som index)
      - "fct_cov": Faktor-kovariansmatrix (annualiseret)
      - "ivol_vec": En pandas Series med den annualiserede idiosynkratiske varians, indekseret med 'id'.

    Example:
    result = calculate_barra_cov(calc_dates, cluster_data_m, spec_risk_m, factor_cov_est)
    #For tjek af resultatet kan man med fordel skrive:
    print(barra_cov["2023-10-31"]["ivol_vec"])
    """
    cluster_data_m['eom'] = pd.to_datetime(cluster_data_m['eom'])
    spec_risk_m['eom'] = pd.to_datetime(spec_risk_m['eom'])

    factor_cov_est_str = {pd.to_datetime(k).strftime('%Y-%m-%d'): v for k, v in factor_cov_est.items()}

    barra_cov = {}

    for d in tqdm(calc_dates, desc="Processerer datoer"):
        d_date = pd.to_datetime(d)
        d_key = d_date.strftime('%Y-%m-%d')
        char_data = cluster_data_m[cluster_data_m['eom'] == d_date].copy()
        char_data = pd.merge(char_data, spec_risk_m, on=['id', 'eom'], how='left')

        char_data['med_res_vol'] = char_data.groupby(['size_grp', 'eom'])['res_vol'].transform('median')

        if char_data['med_res_vol'].isna().any():
            char_data['med_res_vol_all'] = char_data.groupby('eom')['res_vol'].transform('median')
            char_data['med_res_vol'] = char_data['med_res_vol'].fillna(char_data['med_res_vol_all'])

        char_data['res_vol'] = char_data['res_vol'].fillna(char_data['med_res_vol'])

        if d_key not in factor_cov_est_str:
            raise KeyError(f"Dato {d_key} ikke fundet i factor_cov_est.")
        fct_cov = factor_cov_est_str[d_key] * 21

        char_data.sort_values(by='id', inplace=True)

        if hasattr(fct_cov, 'columns'):
            factor_cols = list(fct_cov.columns)
        else:
            raise ValueError("fct_cov skal have attributten 'columns'.")

        X = char_data[factor_cols].to_numpy()
        fct_load = pd.DataFrame(X, index=char_data['id'].astype(str), columns=factor_cols)

        ivol_vec = (char_data['res_vol'] ** 2) * 21
        ivol_vec.index = char_data['id'].astype(str)

        barra_cov[d_key] = {"fct_load": fct_load, "fct_cov": fct_cov, "ivol_vec": ivol_vec}

    return barra_cov


def run_sanity_checks(run_checks, calc_dates, cluster_data_m, spec_risk_m, barra_cov, chars):
    if not run_checks:
        print("Sanity checks are disabled. Set run_checks=True to execute.")
        return None
    """
    # Eksempel på anvendelse:
    # results = run_sanity_checks(True, calc_dates, cluster_data_m, spec_risk_m, barra_cov, chars)
    # results["missing_values_plot"].show()
    # results["num_stocks_plot"].show()
    # results["volatility_plot"].show()
    # results["market_volatility_plot"].show()
    # print(results["valid_count"])
    """

    print("Running sanity checks...")
    rows = []
    for d in calc_dates:
        char_data = cluster_data_m[cluster_data_m["eom"] == d].copy()
        char_data = char_data.merge(spec_risk_m, on=["id", "eom"], how="left")
        n_miss = char_data["res_vol"].isna().sum()
        rows.append({"eom": d, "n_miss": n_miss})

    test = pd.DataFrame(rows)

    # Plot missing values over time
    p_missing = (ggplot(test, aes(x="eom", y="n_miss"))
                 + geom_point()
                 + labs(x="End of Month", y="Number of Missing res_vol"))

    # Sanity check: Beregn diagonal-elementer fra fct_cov
    rows = []
    for d, data in barra_cov.items():
        num_stocks = len(data["fct_load"])
        pred_var = np.diag(data["fct_cov"].values)
        sd_avg = np.sqrt(np.mean(pred_var) * 252)
        rows.append({"eom": pd.to_datetime(d), "n": num_stocks, "sd_avg": sd_avg})

    pred_sd_avg = pd.DataFrame(rows)

    # Plot antal stocks pr. måned
    p_n = (ggplot(pred_sd_avg, aes(x="eom", y="n"))
           + geom_point(size=1)
           + theme(axis_title_x=element_blank()))

    # Plot gennemsnitlig estimeret volatilitet
    p_sd_avg = (ggplot(pred_sd_avg, aes(x="eom", y="sd_avg"))
                + geom_point(size=1)
                + labs(y="Average Predicted Volatility (Annualized)", title="Covariance Sanity Check")
                + theme(axis_title_x=element_blank()))

    me = chars.dropna(subset=["me"])[["id", "eom", "me"]].copy()
    me = me.sort_values(by="eom")
    me_grouped = {date: group for date, group in me.groupby("eom")}

    rows = []
    for d, data in barra_cov.items():
        fct_load = data["fct_load"]
        fct_cov = data["fct_cov"]

        # Beregn aktie-kovariansmatrix
        stock_cov_matrix = fct_load @ fct_cov @ fct_load.T

        me_sub = me_grouped.get(pd.to_datetime(d), None)

        if me_sub is None or me_sub.empty:
            continue

        me_sub["id"] = me_sub["id"].astype(str)
        stock_cov_matrix.index = stock_cov_matrix.index.astype(str)
        stock_cov_matrix.columns = stock_cov_matrix.columns.astype(str)

        me_sub = me_sub[me_sub["id"].isin(stock_cov_matrix.columns)]

        if me_sub.empty or stock_cov_matrix.empty:
            continue  # Spring over hvis ingen matchende data

        me_sub["w"] = me_sub["me"] / me_sub["me"].sum()

        sigma_sub = stock_cov_matrix.loc[me_sub["id"], me_sub["id"]]

        if sigma_sub.empty:
            continue

        mkt_vol = np.sqrt(me_sub["w"].values @ sigma_sub.values @ me_sub["w"].values.T)
        rows.append({"eom": pd.to_datetime(d), "mkt_vol": mkt_vol})

    mkt_vol_df = pd.DataFrame(rows)

    p_mkt_vol = (ggplot(mkt_vol_df, aes(x="eom", y="mkt_vol * np.sqrt(252)"))
                 + geom_point()
                 + labs(y="Market Volatility (Annualized)")
                 + theme(axis_title_x=element_blank()))

    rows = []
    for d, data in barra_cov.items():
        ids = list(data["fct_load"].index.astype(str))  # Konverter til str

        for id in ids:
            rows.append({"eom": pd.to_datetime(d), "id": id, "valid_cov": True})

    valid_cov_est = pd.DataFrame(rows)

    chars["id"] = chars["id"].astype(str)

    test = valid_cov_est.merge(chars[["id", "eom", "valid"]], on=["id", "eom"], how="left")
    valid_count = test[(test["valid"] == True) & (test["eom"] >= valid_cov_est["eom"].min())].groupby(
        "valid_cov").size()

    return {
        "missing_values_plot": p_missing,
        "num_stocks_plot": p_n,
        "volatility_plot": p_sd_avg,
        "market_volatility_plot": p_mkt_vol,
        "valid_count": valid_count
    }




def main():
    # Indhenter chars og daily:
    file_path_usa_test = "./data_fifty/usa_test.parquet"
    daily_file_path = "./data_fifty/usa_dsf_test.parquet"
    file_path_world_ret = "./data_fifty/world_ret_test.csv"
    risk_free_path = "./data_fifty/risk_free_test.csv"
    market_path = "./data_fifty/market_returns_test.csv"

    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"

    # Kald funktionerne
    chars, daily = process_all_data(
        file_path_usa_test, daily_file_path, file_path_world_ret, risk_free_path, market_path
    )

    cluster_data_d, ind_factors, clusters, cluster_data_m  = process_cluster_data(
        chars, daily, file_path_cluster_labels, file_path_factor_details
    )

    fct_ret_est = run_regressions_by_date(cluster_data_d, ind_factors, clusters)
    fct_ret = create_factor_returns(fct_ret_est)

    w_cor = (0.5 ** (1 / settings['cov_set']['hl_cor'])) ** np.arange(settings['cov_set']['obs'], 0, -1)
    w_var = (0.5 ** (1 / settings['cov_set']['hl_var'])) ** np.arange(settings['cov_set']['obs'], 0, -1)

    fct_dates = np.sort(fct_ret["date"].unique())
    min_calc_date = fct_dates[settings["cov_set"]["obs"] - 1] - pd.DateOffset(months=1)
    calc_dates = np.sort(cluster_data_m.loc[cluster_data_m["eom"] >= min_calc_date, "eom"].unique())

    factor_cov_est = factor_cov_estimate(calc_dates, fct_dates, fct_ret, w_cor, w_var, settings)

    spec_risk = unnest_spec_risk(fct_ret_est)
    spec_risk_res_vol = calculate_ewma(spec_risk, settings)

    spec_risk_m, spec_risk_res_vol = process_spec_risk_m(fct_dates, spec_risk_res_vol)
    barra_cov = calculate_barra_cov(calc_dates, cluster_data_m, spec_risk_m, factor_cov_est)

    #results = run_sanity_checks(True, calc_dates, cluster_data_m, spec_risk_m, barra_cov, chars)
    return barra_cov


if __name__ == "__main__":
    main()
