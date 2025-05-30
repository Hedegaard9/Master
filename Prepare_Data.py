import pandas as pd
import numpy as np
import re
import gc
from pandas.tseries.offsets import MonthEnd
import General_Functions as GF
from pandas.tseries.offsets import MonthEnd
from Main import settings, features, pf_set
from General_Functions import size_screen_fun, addition_deletion_fun
from scipy.stats import rankdata



def merge_rvol_data(file_path_usa, file_path_rvol_252):
    """
    Indlæser USA data fra Parquet og rvol_252 fra CSV, konverterer relevante kolonner,
    merger dataene på 'id' og 'eom', udskriver antal ikke-NaN værdier før og efter merge,
    og gemmer den merged dataframe som 'usa_rvol.parquet' i ./Data/ mappen.

    Args:
    - file_path_usa (str): Sti til Parquet-filen.
    - file_path_rvol_252 (str): Sti til CSV-filen.
    Example:
    file_path_usa = "./Data/usa.parquet"
    file_path_rvol_252 = "./Data/rvol_252.csv"

    df_merged = merge_rvol_data(file_path_usa, file_path_rvol_252)
    Returns:
    - df (pd.DataFrame): Den merged dataframe.
    """

    df = pd.read_parquet(file_path_usa, engine='pyarrow')
    print("Fil indlæst med succes. Antal rækker før filtrering:", len(df))

    rvol_252 = pd.read_csv(file_path_rvol_252)

    df['id'] = df['id'].astype('int64')

    df['eom'] = pd.to_datetime(df['eom'])
    rvol_252['eom'] = pd.to_datetime(rvol_252['eom'])

    num_non_nan_rvol_252 = rvol_252['rvol_252d'].notna().sum()
    print(f"Antal ikke-NaN værdier i 'rvol_252d' i rvol_252 før merge: {num_non_nan_rvol_252}")

    df = df.merge(rvol_252[['id', 'eom', 'rvol_252d']], on=['id', 'eom'], how='left')

    num_non_nan_df = df['rvol_252d'].notna().sum()
    print(f"Antal ikke-NaN værdier i 'rvol_252d' i df efter merge: {num_non_nan_df}")

    print(df[['id', 'eom', 'rvol_252d']].head(10))

    save_path = "./Data/usa_rvol.parquet"
    df.to_parquet(save_path, engine="pyarrow", index=False)

    print("Filen 'usa_rvol.parquet' er gemt succesfuldt i ./Data mappen!")

    return df





def process_risk_free_rate(file_path):
    """
    Args:
        file_path (str): Stien til CSV-filen med risikofri rente-data.
    Example:
        #Duummy check
        rente_path = "Data/ff3_m.csv"
        risk_free_df = process_risk_free_rate(rente_path)
        print(risk_free_df)
    Returns:
        pd.DataFrame: En DataFrame med to kolonner: 'eom' (slutningen af måneden) og 'rf' (risikofri rente i procent).
    """
    risk_free = pd.read_csv(file_path, usecols=["yyyymm", "RF"])

    risk_free['rf'] = risk_free['RF'] / 100
    risk_free['eom'] = risk_free['yyyymm'].astype(str) + "01"
    risk_free['eom'] = pd.to_datetime(risk_free['eom'], format="%Y%m%d") + MonthEnd(0)

    return risk_free[['eom', 'rf']]




# Load market returns og filtrer på USA og mkt_vw_exc
def load_and_filter_market_returns(file_path):
    """
    Example:
    import Prepare_Data
    Prepare_Data.load_and_filter_market_returns(file_path_market_returns)
    :param file_path:
    :return: market_returns_df
    """
    try:
        market_returns_df = pd.read_csv(file_path)

        market_returns_df = market_returns_df[market_returns_df['excntry'] == 'USA'][['eom', 'mkt_vw_exc']].reset_index(
            drop=True)

        print("Filen er indlæst og filtreret succesfuldt.")
        return market_returns_df
    except Exception as e:
        print(f"Der opstod en fejl ved indlæsning eller filtrering af filen: {e}")
        return None

# Load market returns og filtrer på USA og mkt_vw_exc
def load_and_filter_market_returns_test(file_path):
    """
    Example:
    import Prepare_Data
    Prepare_Data.load_and_filter_market_returns(file_path_market_returns)
    :param file_path:
    :return: market_returns_df
    """
    try:
        market_returns_df = pd.read_csv(file_path)

        print("Filen er indlæst og filtreret succesfuldt.")
        return market_returns_df
    except Exception as e:
        print(f"Der opstod en fejl ved indlæsning eller filtrering af filen: {e}")
        return None



# Funktion til at indhente Factor Details og Cluster Labels
def process_cluster_labels(file_path_cluster_labels, file_path_factor_details):
    """
    Behandler cluster labels og faktor detaljer ved at kombinere og transformere data.

    Funktionens opgaver:
    1. Læs cluster labels fra en CSV-fil.
    2. Læs faktor detaljer fra en Excel-fil.
    3. Transformér cluster-navne til små bogstaver og erstat mellemrum og bindestreger med underscores.
    4. Slå cluster labels og faktor detaljer sammen på karakteristika.
    5. Tilføj en ny række for 'rvol_252d' med specifikke værdier.
    6. Returner den opdaterede DataFrame.

    Parametre:
    - file_path_cluster_labels (str): Sti til CSV-filen med cluster labels.
    - file_path_factor_details (str): Sti til Excel-filen med faktor detaljer.

    Krav:
    - openpyxl skal være installeret for at kunne læse Excel-filen.
      Installer med: `pip install openpyxl`


    #Eksempel på brug af funktionen:
    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"
    cluster_labels = process_cluster_labels(file_path_cluster_labels, file_path_factor_details)

    Returnerer:
    - pd.DataFrame: En opdateret DataFrame med de transformerede data.
    """
    cluster_labels = pd.read_csv(file_path_cluster_labels)
    factor_signs = pd.read_excel(file_path_factor_details, engine="openpyxl")

    cluster_labels['cluster'] = cluster_labels['cluster'].str.lower().str.replace(r"\s|-", "_", regex=True)

    factor_signs = (
        factor_signs[['abr_jkp', 'direction']]
        .rename(columns={'abr_jkp': 'characteristic'})
        .dropna(subset=['characteristic'])
    )
    factor_signs['direction'] = pd.to_numeric(factor_signs['direction'], errors='coerce')

    cluster_labels = pd.merge(cluster_labels, factor_signs, on='characteristic', how='left')

    new_row = {'characteristic': 'rvol_252d', 'cluster': 'low_risk', 'direction': -1}
    cluster_labels = pd.concat([cluster_labels, pd.DataFrame([new_row])], ignore_index=True)

    return cluster_labels




def process_return_data(file_path_world_ret, risk_free_path):
    """
    :param file_path_world_ret:
    :param risk_free_path:
    :return: en dataframe med afkast for næste periode (månedvis)

    Eksempel
    file_path_world_ret="./data_test/world_ret_test.csv"
    risk_free_path="./data_test/risk_free_test.csv"
    data_ret_ld1 = process_return_data(file_path_world_ret, risk_free_path)
    """
    monthly = pd.read_csv(file_path_world_ret, usecols=["excntry", "id", "eom", "ret_exc"], dtype={"eom": str})

    monthly = monthly[(monthly["excntry"] == "USA") & (monthly["id"] <= 99999)]

    monthly["eom"] = pd.to_datetime(monthly["eom"])

    horizon_K = settings["pf"]["hps"]["m1"]["K"]

    data_ret = GF.long_horizon_ret(monthly, h=horizon_K, impute="zero")
    data_ret = data_ret.drop(columns=["start", "end", "merge_date"])

    data_ret_ld1 = data_ret[["id", "eom", "ret_ld1"]].copy()


    data_ret_ld1["eom_ret"] = data_ret_ld1["eom"] + pd.DateOffset(months=1)
    data_ret_ld1["eom_ret"] = data_ret_ld1["eom_ret"] + MonthEnd(0)

    data_ret_ld1 = data_ret_ld1[["id", "eom", "eom_ret", "ret_ld1"]]

    risk_free = process_risk_free_rate(risk_free_path)
    data_ret_ld1 = data_ret_ld1.merge(risk_free, on="eom", how="left")

    data_ret_ld1["tr_ld1"] = data_ret_ld1["ret_ld1"] + data_ret_ld1["rf"]

    data_ret_ld1.drop(columns=["rf"], inplace=True)

    data_ret_ld1_shifted = data_ret_ld1[["id", "eom", "tr_ld1"]].copy()

    data_ret_ld1_shifted["eom"] = data_ret_ld1_shifted["eom"] + pd.DateOffset(months=1)
    data_ret_ld1_shifted["eom"] = data_ret_ld1_shifted["eom"] + MonthEnd(0)

    data_ret_ld1_shifted.rename(columns={"tr_ld1": "tr_ld0"}, inplace=True)

    data_ret_ld1 = data_ret_ld1.merge(data_ret_ld1_shifted, on=["id", "eom"], how="left")

    data_ret_ld1 = data_ret_ld1[["id", "eom", "tr_ld0", "eom_ret", "ret_ld1", "tr_ld1"]]

    return data_ret_ld1


# Wealth: Assumed portfolio growth
def wealth_func(wealth_end, end, market, risk_free):
    """
    Beregner porteføljevækst over tid med datoer justeret til den sidste dag i hver måned
    og forskudt en måned tilbage.

    Args:
        wealth_end (float): Slutværdien af porteføljen.
        end (str): Slutdato for perioden ("YYYY-MM-DD").
        market (pd.DataFrame): Markedsafkast (kolonner fx: 'eom' / 'eom_ret', 'mkt_vw_exc').
        risk_free (pd.DataFrame): Risikofrit afkast (kolonner fx: 'eom' / 'eom_ret', 'rf').

    Returns:
        pd.DataFrame: Med kolonner: 'eom', 'wealth', 'mu_ld1' (log-retur),
                      hvor 'eom' er den sidste dag i måneden og forskudt én måned tilbage.
    Eksempel på brug:
    risk_free_path="./data_test/risk_free_test.csv"
    market_path="./data_test/market_returns_test.csv"
    market = load_and_filter_market_returns_test(market_path)
    risk_free = process_risk_free_rate(risk_free_path)
    wealth_end = pf_set["wealth"]
    end = settings["split"]["test_end"]
    wealth = wealth_func(wealth_end, end, market, risk_free)
    """

    def find_date_col(df):
        if "eom" in df.columns:
            return "eom"
        elif "eom_ret" in df.columns:
            return "eom_ret"
        else:
            raise ValueError("Ingen 'eom' eller 'eom_ret' kolonne i DataFrame")

    risk_free_date = find_date_col(risk_free)
    market_date = find_date_col(market)

    risk_free = risk_free.rename(columns={risk_free_date: "date"})
    market = market.rename(columns={market_date: "date"})

    risk_free["date"] = pd.to_datetime(risk_free["date"])
    market["date"] = pd.to_datetime(market["date"])

    wealth = risk_free.merge(market, on="date", how="left")

    wealth["tret"] = wealth["mkt_vw_exc"] + wealth["rf"]

    wealth = wealth[wealth["date"] <= pd.to_datetime(end)]

    wealth = wealth.sort_values(by="date", ascending=False)

    wealth["wealth"] = (1 - wealth["tret"]).cumprod() * wealth_end

    wealth["eom"] = (wealth["date"] + pd.offsets.MonthEnd(0)) - pd.DateOffset(months=1)
    wealth["eom"] = wealth["eom"] + pd.offsets.MonthEnd(0)

    wealth = wealth.drop_duplicates(subset=["eom"], keep="last")

    final_row = pd.DataFrame({
        "eom": [pd.to_datetime(end) + pd.offsets.MonthEnd(0)],
        "wealth": [wealth_end],  # Wealth skal være wealth_end
        "mu_ld1": [np.nan]
    })
    wealth = pd.concat([wealth, final_row], ignore_index=True)

    wealth = wealth.sort_values(by="eom").reset_index(drop=True)

    return wealth[["eom", "wealth", "tret"]].rename(columns={"tret": "mu_ld1"})




# Prepare data -----------------------------------
def load_and_prepare_data(file_path, features):
    """
    Indlæser og forbereder rådata fra en Parquet-fil.

    Args:
        file_path (str): Stien til Parquet-filen (USA testfil).
        features (list): Liste over features, der skal indlæses.
    Example:
        file_path_usa_dsf_test = "./data_test/usa_dsf_test.parquet"
        data = load_and_prepare_data(file_path_usa_test, features)
    Returns:
        pd.DataFrame: DataFrame med forberedte data.
    """
    desired_cols = ["id", "eom", "sic", "ff49", "size_grp", "me", "crsp_exchcd", "rvol_252d", "dolvol_126d"] + features

    actual_cols = pd.read_parquet(file_path, engine="pyarrow").columns
    cols_to_load = [col for col in desired_cols if col in actual_cols]

    if not cols_to_load:
        raise ValueError("Ingen af de ønskede kolonner findes i Parquet-filen.")

    data = pd.read_parquet(file_path, engine="pyarrow", columns=cols_to_load)

    data = data[data["id"] <= 99999]

    data["eom"] = pd.to_datetime(data["eom"], errors="coerce")

    return data





def process_all_data(file_path_usa_test, daily_file_path, file_path_world_ret, risk_free_path, market_path):
    """
    Læs og processer 'chars' og 'daily' data og returner dem.

    Parametre:
      file_path_usa_test (str): Sti til Parquet-filen med USA-data (chars).
      daily_file_path (str): Sti til Parquet-filen med daily-data.
      file_path_world_ret (str): Sti til CSV-filen med world returns.
      risk_free_path (str): Sti til CSV-filen med risk free data.
      market_path (str): Sti til CSV-filen med market returns.

    Returnerer:
      tuple: (chars, daily)

     # Eksempel på parametre (disse skal defineres i din kode)
     file_path_usa_test = "./data_test/usa_test.parquet"
     daily_file_path = "./data_test/usa_dsf_test.parquet"
     file_path_world_ret = "./data_test/world_ret_test.csv"
     risk_free_path = "./data_test/risk_free_test.csv"
     market_path = "./data_test/market_returns_test.csv"


    # Kald funktionen
     chars, daily = process_all_data(file_path_usa_test, daily_file_path, file_path_world_ret, risk_free_path, market_path)
    """
    desired_cols = ["id", "eom", "sic", "size_grp", "me", "rvol_252d", "dolvol_126d"] + features
    actual_cols = pd.read_parquet(file_path_usa_test, engine="pyarrow").columns
    cols_to_load = [col for col in desired_cols if col in actual_cols]
    if not cols_to_load:
        raise ValueError("Ingen af de ønskede kolonner findes i Parquet-filen.")

    chars = pd.read_parquet(file_path_usa_test, engine="pyarrow", columns=cols_to_load)

    chars = chars[chars["id"] <= 99999]

    chars["eom"] = pd.to_datetime(chars["eom"], errors="coerce")

    chars = chars.loc[:, ~chars.columns.duplicated()]

    required_cols = ["dolvol_126d", "rvol_252d"]

    pi = settings["pi"]
    chars["dolvol"] = chars["dolvol_126d"]
    chars["lambda"] = 2 / chars["dolvol"] * pi
    chars["rvol_m"] = chars["rvol_252d"] * np.sqrt(21)

    data_ret_ld1 = process_return_data(file_path_world_ret, risk_free_path)
    chars = chars.merge(data_ret_ld1, on=["id", "eom"], how="left")

    market = load_and_filter_market_returns_test(market_path)
    risk_free = process_risk_free_rate(risk_free_path)
    wealth_end = pf_set["wealth"]
    end = settings["split"]["test_end"]
    wealth = wealth_func(wealth_end, end, market, risk_free)
    wealth["eom"] = wealth["eom"] + pd.offsets.MonthEnd(1)
    wealth_subset = wealth[["eom", "mu_ld1"]].rename(columns={"mu_ld1": "mu_ld0"})
    chars = chars.merge(wealth_subset, on="eom", how="left")

    chars = chars[(chars["eom"] >= settings["screens"]["start"]) &
                  (chars["eom"] <= settings["screens"]["end"])]

    n_start = len(chars)
    me_start = chars["me"].sum(skipna=True)
    chars = chars.dropna(subset=["me"])
    chars = chars.dropna(subset=["tr_ld0", "tr_ld1"])
    chars = chars.dropna(subset=["dolvol"])
    chars = chars[chars["dolvol"] > 0]
    chars = chars.dropna(subset=["sic"])
    chars = chars[chars["sic"] != ""]

    # Screening baseret på feature
    feat_available = chars[features].notna().sum(axis=1)
    min_feat = int(len(features) * settings["screens"]["feat_pct"])
    chars = chars[feat_available >= min_feat]

    run_sub = False
    if run_sub:
        np.random.seed(settings["seed"])
        sampled_ids = np.random.choice(chars["id"].unique(), 2500, replace=False)
        chars = chars[chars["id"].isin(sampled_ids)]

    if settings["feat_prank"]:
        chars[features] = chars[features].astype(float)
        for i, f in enumerate(features, 1):
            zero_mask = chars[f] == 0

            def ecdf_transform(x):
                non_na = x.dropna()
                if len(non_na) == 0:
                    return x
                ranks = rankdata(non_na, method="average") / len(non_na)
                rank_mapping = dict(zip(non_na, ranks))
                return x.map(lambda v: rank_mapping.get(v, np.nan))

            chars[f] = chars.groupby("eom")[f].transform(ecdf_transform)
            chars.loc[zero_mask, f] = 0
    if settings["feat_impute"]:
        if settings["feat_prank"]:
            chars[features] = chars[features].fillna(0.5)
        else:
            chars[features] = chars.groupby("eom")[features].transform(lambda x: x.fillna(x.median()))

    chars = chars.copy()
    chars["sic"] = pd.to_numeric(chars["sic"], errors="coerce")

    conditions = [
        chars["sic"].between(100, 999) | chars["sic"].between(2000, 2399) | chars["sic"].between(2700, 2749) |
        chars["sic"].between(2770, 2799) | chars["sic"].between(3100, 3199) | chars["sic"].between(3940, 3989),

        chars["sic"].between(2500, 2519) | chars["sic"].between(3630, 3659) | chars["sic"].between(3710, 3711) |
        chars["sic"].isin([3714, 3716]) | chars["sic"].between(3750, 3751) | chars["sic"].isin([3792]) |
        chars["sic"].between(3900, 3939) | chars["sic"].between(3990, 3999),

        chars["sic"].between(2520, 2589) | chars["sic"].between(2600, 2699) | chars["sic"].between(2750, 2769) |
        chars["sic"].between(3000, 3099) | chars["sic"].between(3200, 3569) | chars["sic"].between(3580, 3629) |
        chars["sic"].between(3700, 3709) | chars["sic"].between(3712, 3713) | chars["sic"].isin([3715]) |
        chars["sic"].between(3717, 3749) | chars["sic"].between(3752, 3791) | chars["sic"].between(3793, 3799) |
        chars["sic"].between(3830, 3839) | chars["sic"].between(3860, 3899),

        chars["sic"].between(1200, 1399) | chars["sic"].between(2900, 2999),

        chars["sic"].between(2800, 2829) | chars["sic"].between(2840, 2899),

        chars["sic"].between(3570, 3579) | chars["sic"].between(3660, 3692) | chars["sic"].between(3694, 3699) |
        chars["sic"].between(3810, 3829) | chars["sic"].between(7370, 7379),

        chars["sic"].between(4800, 4899),

        chars["sic"].between(4900, 4949),

        chars["sic"].between(5000, 5999) | chars["sic"].between(7200, 7299) | chars["sic"].between(7600, 7699),

        chars["sic"].between(2830, 2839) | chars["sic"].isin([3693]) | chars["sic"].between(3840, 3859) |
        chars["sic"].between(8000, 8099),

        chars["sic"].between(6000, 6999)
    ]
    choices = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems",
               "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money"]
    chars["ff12"] = np.select(conditions, choices, default="Other")

    chars = chars.copy()
    chars["valid_data"] = True
    chars = chars.sort_values(by=["id", "eom"]).copy()

    lb = pf_set["lb_hor"] + 1
    chars["eom_lag"] = chars.groupby("id")["eom"].shift(lb)
    chars["month_diff"] = ((chars["eom"] - chars["eom_lag"]).dt.days / 30.44).round().astype("Int64")

    chars["valid_data"] = chars["valid_data"] & (chars["month_diff"] == lb) & chars["month_diff"].notna()
    chars.drop(columns=["eom_lag", "month_diff"], inplace=True)
    del lb

    # Kør ekstern screening: Size og addition/deletion
    chars = size_screen_fun(chars, "all")
    chars = addition_deletion_fun(chars, addition_n=settings["addition_n"], deletion_n=settings["deletion_n"])


    # ----- Processering af daily-data -----
    daily = pd.read_parquet(daily_file_path)
    valid_ids_daily = chars.loc[chars['valid'] == True, 'id'].unique()
    daily = daily[
        daily['ret_exc'].notna() &
        (daily['id'] <= 99999) &
        (daily['id'].isin(valid_ids_daily))
        ]
    gc.collect()
    daily['date'] = pd.to_datetime(daily['date'], format='%Y-%m-%d', errors='coerce')
    gc.collect()
    daily['eom'] = daily['date'] + pd.offsets.MonthEnd(0)

    return chars, daily



def prepare_daily_returns(file_path, data):
    """
    Forbereder daglige afkastdata.

    Args:
        file_path (str): Stien til CSV-filen med daglige data. usa_dsf
        chars (pd.DataFrame): Filtreret DataFrame med gyldige observationer.

    Example:
        prepare_daily_returns("./data_test/usa_dsf_test.parquet", data)
        Her er data skabt fra tidligere ting i denne py fil.
        data = pd.read_parquet("./data_test/usa_test.parquet", engine="pyarrow")
        print(data.head())


            df = prepare_daily_returns("./data_test/usa_dsf_test.parquet", data)
            print(df.head())
        Dog giver dette bedst mening at gøre med data efterbehandling fra usa filen.

    Returns:
        pd.DataFrame: DataFrame med daglige afkast.
    """
    usecols = ["id", "date", "ret_exc"]
    daily = pd.read_parquet(file_path, engine="pyarrow", columns=usecols)
    daily = daily[daily["ret_exc"].notna() & daily["id"].isin(data["id"].unique())]
    daily["date"] = pd.to_datetime(daily["date"], format="%Y%m%d")
    daily["eom"] = daily["date"] + pd.offsets.MonthEnd(0)

    return daily



def main():
    # Definer filstier (tilpas disse til dine data)
    file_path_usa = "./Data/usa.parquet"
    file_path_rvol_252 = "./Data/rvol_252.csv"
    file_path_usa_test = "./data_test/usa_test.parquet"
    daily_file_path = "./data_test/usa_dsf_test.parquet"
    file_path_world_ret = "./data_test/world_ret_test.csv"
    risk_free_path = "./data_test/risk_free_test.csv"
    market_path = "./data_test/market_returns_test.csv"
    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"
    rente_path = "Data/ff3_m.csv"

    # Hent ekstra indstillinger
    wealth_end = pf_set["wealth"]
    end = settings["split"]["test_end"]

    df_merged = merge_rvol_data(file_path_usa, file_path_rvol_252)
    risk_free = process_risk_free_rate(rente_path)
    market_test = load_and_filter_market_returns_test(market_path)
    data_ret_ld1 = process_return_data(file_path_world_ret, risk_free_path)
    wealth = wealth_func(wealth_end, end, market_test, risk_free)
    data = load_and_prepare_data(file_path_usa_test, features)
    chars, daily = process_all_data(file_path_usa_test, daily_file_path, file_path_world_ret, risk_free_path, market_path)
    df_daily_returns = prepare_daily_returns(daily_file_path, data)


    return wealth

if __name__ == "__main__":
    main()
