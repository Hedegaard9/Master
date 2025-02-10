import pandas as pd
import numpy as np
from Main import settings, features, pf_set
from pandas.tseries.offsets import MonthEnd
from Prepare_Data import process_risk_free_rate, process_return_data, wealth_func, load_and_filter_market_returns_test, process_all_data, process_cluster_labels
from General_Functions import size_screen_fun, addition_deletion_fun



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

    Eksempel på brug:
    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"
    cluster_data_d = process_cluster_data(chars, daily, file_path_cluster_labels, file_path_factor_details)
    """

    # Indlæs cluster labels separat (bruger samme metode som i den anden version)
    cluster_labels = process_cluster_labels(cluster_labels_path, factor_details_path)

    # Filtrer gyldige karakteristika
    valid_chars = chars[chars["valid"] == True]
    cluster_data_m = valid_chars[["id", "eom", "size_grp", "ff12"] + features].copy()

    # Beregn cluster ranks
    clusters = cluster_labels["cluster"].unique()
    cluster_ranks = {}

    for cl in clusters:
        chars_sub = cluster_labels[(cluster_labels["cluster"] == cl) & (cluster_labels["characteristic"].isin(features))]

        # Undgå potentielle fejl ved at sikre, at kun eksisterende kolonner vælges
        valid_features = [c for c in chars_sub["characteristic"].values if c in cluster_data_m.columns]
        data_sub = cluster_data_m[valid_features].copy()

        for c in valid_features:
            dir_value = chars_sub.loc[chars_sub["characteristic"] == c, "direction"].values[0]
            if dir_value == -1:
                data_sub[c] = 1 - data_sub[c]

        cluster_ranks[cl] = data_sub.mean(axis=1)

    # Konverter cluster ranks til DataFrame
    cluster_ranks_df = pd.DataFrame(cluster_ranks)

    # Kombiner med de oprindelige data
    cluster_data_m = cluster_data_m[["id", "eom", "size_grp", "ff12"]].copy()
    cluster_data_m["eom_ret"] = cluster_data_m["eom"] + pd.DateOffset(months=1)
    cluster_data_m["eom_ret"] = cluster_data_m["eom_ret"] + pd.offsets.MonthEnd(0)  # Sikrer, at det er sidste dag i måneden

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

    # Standardiser faktorer per eom (Bruger samme metode som den anden version)
    cluster_data_m[clusters] = cluster_data_m.groupby("eom")[clusters].transform(lambda x: (x - x.mean()) / x.std())

    # Indlæs daglige returdata
    min_eom = cluster_data_m["eom"].min()
    daily_filtered = daily[daily["date"] >= min_eom][["id", "date", "ret_exc", "eom"]].copy()
    daily_filtered.rename(columns={"eom": "eom_ret"}, inplace=True)

    # Flet daglige data med cluster_data_m
    cluster_data_d = cluster_data_m.merge(daily_filtered, on=["id", "eom_ret"], how="inner")

    # Fjern manglende værdier
    cluster_data_d.dropna(inplace=True)

    return cluster_data_d



