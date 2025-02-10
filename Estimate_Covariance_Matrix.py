import pandas as pd
import numpy as np
from Main import settings, features, pf_set
from pandas.tseries.offsets import MonthEnd
from Prepare_Data import process_risk_free_rate, process_return_data, wealth_func, load_and_filter_market_returns_test, process_all_data, process_cluster_labels
from General_Functions import size_screen_fun, addition_deletion_fun



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

    Eksempel på brug:
    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"
    cluster_data_d = process_cluster_data(chars, daily, file_path_cluster_labels, file_path_factor_details)

    """

    # Læs cluster labels
    cluster_labels = pd.read_csv(cluster_labels_path)
    cluster_labels["cluster"] = cluster_labels["cluster"].str.lower().str.replace(r"\s|-", "_", regex=True)

    # Læs factor details
    factor_signs = pd.read_excel(factor_details_path, usecols=["abr_jkp", "direction"])
    factor_signs.rename(columns={"abr_jkp": "characteristic"}, inplace=True)
    factor_signs.dropna(subset=["characteristic"], inplace=True)
    factor_signs["direction"] = factor_signs["direction"].astype(float)

    # Merge factor signs med cluster labels
    cluster_labels = cluster_labels.merge(factor_signs, on="characteristic", how="left")

    # Tilføj specifikt cluster til rvol_252d
    new_row = pd.DataFrame({"characteristic": ["rvol_252d"], "cluster": ["low_risk"], "direction": [-1]})
    cluster_labels = pd.concat([cluster_labels, new_row], ignore_index=True)

    # Filtrer gyldige karakteristika
    valid_chars = chars[chars["valid"] == True]
    cluster_data_m = valid_chars[["id", "eom", "size_grp", "ff12"] + features].copy()

    # Beregn cluster ranks
    clusters = cluster_labels["cluster"].unique()
    cluster_ranks = {}

    for cl in clusters:
        chars_sub = cluster_labels[
            (cluster_labels["cluster"] == cl) & (cluster_labels["characteristic"].isin(features))]
        data_sub = cluster_data_m[chars_sub["characteristic"].values].copy()

        for c in chars_sub["characteristic"].values:
            dir_value = chars_sub.loc[chars_sub["characteristic"] == c, "direction"].values[0]
            if dir_value == -1:
                data_sub[c] = 1 - data_sub[c]

        cluster_ranks[cl] = data_sub.mean(axis=1)

    # Konverter cluster ranks til DataFrame
    cluster_ranks_df = pd.DataFrame(cluster_ranks)

    # Kombiner med oprindelige data
    cluster_data_m = cluster_data_m[["id", "eom", "size_grp", "ff12"]].copy()
    cluster_data_m["eom_ret"] = cluster_data_m["eom"] + pd.DateOffset(months=1)
    cluster_data_m["eom_ret"] = cluster_data_m["eom_ret"] + pd.offsets.MonthEnd(0)

    # Merge cluster rankings
    cluster_data_m = pd.concat([cluster_data_m, cluster_ranks_df], axis=1)

    # Tilføj industry/market dummies
    if settings["cov_set"]["industries"]:
        industries = sorted(cluster_data_m["ff12"].unique())
        for ind in industries:
            cluster_data_m[str(ind)] = (cluster_data_m["ff12"] == ind).astype(int)
        ind_factors = industries
    else:
        cluster_data_m["mkt"] = 1
        ind_factors = ["mkt"]

    # Standardiser faktorer per eom
    cluster_data_m[clusters] = cluster_data_m.groupby("eom")[clusters].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    cluster_data_m[clusters] = cluster_data_m[clusters].fillna(0)

    # Tilføj daglige returdata
    min_eom = cluster_data_m["eom"].min()
    daily_filtered = daily[daily["date"] >= min_eom][["id", "date", "ret_exc", "eom"]].copy()
    daily_filtered.rename(columns={"eom": "eom_ret"}, inplace=True)

    # Flet daglige data med cluster_data_m
    cluster_data_d = cluster_data_m.merge(daily_filtered, on=["id", "eom_ret"], how="inner")

    # Fjern rækker med manglende værdier
    cluster_data_d.dropna(inplace=True)

    return cluster_data_d



