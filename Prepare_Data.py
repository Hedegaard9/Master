import pandas as pd
from pandas.tseries.offsets import MonthEnd
import General_Functions as GF
from pandas.tseries.offsets import MonthEnd
from Main import settings
import numpy as np

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

    # Indlæs data
    df = pd.read_parquet(file_path_usa, engine='pyarrow')
    print("Fil indlæst med succes. Antal rækker før filtrering:", len(df))

    rvol_252 = pd.read_csv(file_path_rvol_252)

    # Konverter 'id' til samme datatype (int64)
    df['id'] = df['id'].astype('int64')

    # Konverter 'eom' til datetime
    df['eom'] = pd.to_datetime(df['eom'])
    rvol_252['eom'] = pd.to_datetime(rvol_252['eom'])

    # Udskriv antal ikke-NaN værdier i rvol_252d i rvol_252 før merge
    num_non_nan_rvol_252 = rvol_252['rvol_252d'].notna().sum()
    print(f"Antal ikke-NaN værdier i 'rvol_252d' i rvol_252 før merge: {num_non_nan_rvol_252}")

    # Merge dataene baseret på 'id' og 'eom'
    df = df.merge(rvol_252[['id', 'eom', 'rvol_252d']], on=['id', 'eom'], how='left')

    # Udskriv antal ikke-NaN værdier i 'rvol_252d' efter merge
    num_non_nan_df = df['rvol_252d'].notna().sum()
    print(f"Antal ikke-NaN værdier i 'rvol_252d' i df efter merge: {num_non_nan_df}")

    # Tjek om vi nu får nogle ikke-NaN værdier
    print(df[['id', 'eom', 'rvol_252d']].head(10))

    # Gem den merged dataframe som Parquet i ./Data/ mappen
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
    # Læs data fra filen
    risk_free = pd.read_csv(file_path, usecols=["yyyymm", "RF"])

    # Opret nye kolonner for risikofri rente og slutningen af måneden
    risk_free['rf'] = risk_free['RF'] / 100  # Konverter risikofri rente til procent
    risk_free['eom'] = risk_free['yyyymm'].astype(str) + "01"
    risk_free['eom'] = pd.to_datetime(risk_free['eom'], format="%Y%m%d") + MonthEnd(0)

    # Returner kun de nødvendige kolonner
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
        # Læs filen
        market_returns_df = pd.read_csv(file_path)

        # Filtrér og vælg kun de ønskede kolonner, derefter nulstil indekset
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
        # Læs filen
        market_returns_df = pd.read_csv(file_path)

        print("Filen er indlæst og filtreret succesfuldt.")
        return market_returns_df
    except Exception as e:
        print(f"Der opstod en fejl ved indlæsning eller filtrering af filen: {e}")
        return None



# Funktion til at indhente Factor Details og Cluster Labels og datahåndtering
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

    Eksempel:
    # file_path_cluster_labels = "./Data/Cluster Labels.csv"
    # file_path_factor_details = "./Data/Factor Details.xlsx"

    # Brug funktionen som følger:
    #rente_path = "Data/ff3_m.csv"
    #risk_free = process_risk_free_rate(rente_path)
    #h_list = [1,2]
    #file_path_usa = "Data/usa.parquet"
    #result = monthly_returns(risk_free=risk_free, h_list=h_list, file_path=file_path_usa)

    Returnerer:
    - pd.DataFrame: En opdateret DataFrame med de transformerede data.
    """
    # Læs data fra CSV og Excel
    cluster_labels = pd.read_csv(file_path_cluster_labels)
    factor_signs = pd.read_excel(file_path_factor_details, engine="openpyxl")

    # Transformer 'cluster' kolonnen
    cluster_labels['cluster'] = cluster_labels['cluster'].str.lower().str.replace(r"\s|-", "_", regex=True)

    # Vælg og transformer relevante kolonner fra Excel-data
    factor_signs = (
        factor_signs[['abr_jkp', 'direction']]
        .rename(columns={'abr_jkp': 'characteristic'})
        .dropna(subset=['characteristic'])
    )
    factor_signs['direction'] = pd.to_numeric(factor_signs['direction'], errors='coerce')

    # Slå de to datasæt sammen
    cluster_labels = pd.merge(cluster_labels, factor_signs, on='characteristic', how='left')

    # Tilføj en ny række for 'rvol_252d'
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
    # Indlæs data
    monthly = pd.read_csv(file_path_world_ret, usecols=["excntry", "id", "eom", "ret_exc"], dtype={"eom": str})

    # Filtrér kun USA og CRSP-observationer
    monthly = monthly[(monthly["excntry"] == "USA") & (monthly["id"] <= 99999)]

    # Konverter 'eom' fra string til datetime
    monthly["eom"] = pd.to_datetime(monthly["eom"])

    # Hent tidshorisont
    horizon_K = settings["pf"]["hps"]["m1"]["K"]

    # Anvend long_horizon_ret
    data_ret = GF.long_horizon_ret(monthly, h=horizon_K, impute="zero")
    data_ret = data_ret.drop(columns=["start", "end", "merge_date"])

    # Behold nødvendige kolonner
    data_ret_ld1 = data_ret[["id", "eom", "ret_ld1"]].copy()

    # Beregn eom_ret: sidste dag i næste måned
    data_ret_ld1["eom_ret"] = data_ret_ld1["eom"] + pd.DateOffset(months=1)
    data_ret_ld1["eom_ret"] = data_ret_ld1["eom_ret"] + MonthEnd(0)

    # Omarranger kolonnerne
    data_ret_ld1 = data_ret_ld1[["id", "eom", "eom_ret", "ret_ld1"]]

    # Merge med risk_free data
    risk_free = process_risk_free_rate(risk_free_path)
    data_ret_ld1 = data_ret_ld1.merge(risk_free, on="eom", how="left")

    # Beregn total afkast: excess return + rf
    data_ret_ld1["tr_ld1"] = data_ret_ld1["ret_ld1"] + data_ret_ld1["rf"]

    # Fjern 'rf' kolonnen efter brug
    data_ret_ld1.drop(columns=["rf"], inplace=True)

    # Opret en kopi med justeret eom
    data_ret_ld1_shifted = data_ret_ld1[["id", "eom", "tr_ld1"]].copy()

    # Flyt eom til næste måneds sidste dag
    data_ret_ld1_shifted["eom"] = data_ret_ld1_shifted["eom"] + pd.DateOffset(months=1)
    data_ret_ld1_shifted["eom"] = data_ret_ld1_shifted["eom"] + MonthEnd(0)

    # Omdøb tr_ld1 til tr_ld0
    data_ret_ld1_shifted.rename(columns={"tr_ld1": "tr_ld0"}, inplace=True)

    # Merge for at inkludere tr_ld0
    data_ret_ld1 = data_ret_ld1.merge(data_ret_ld1_shifted, on=["id", "eom"], how="left")

    # Omarranger kolonner
    data_ret_ld1 = data_ret_ld1[["id", "eom", "tr_ld0", "eom_ret", "ret_ld1", "tr_ld1"]]

    return data_ret_ld1


# Wealth: Assumed portfolio growth
def wealth_func(wealth_end, end, market, risk_free):
    """
    Beregner porteføljevækst over tid.

    Args:
        wealth_end (float): Slutværdien af porteføljen.
        end (str): Slutdato for perioden ("YYYY-MM-DD").
        market (pd.DataFrame): Markedsafkast (kolonner fx: 'eom' / 'eom_ret', 'mkt_vw_exc').
        risk_free (pd.DataFrame): Risikofrit afkast (kolonner fx: 'eom' / 'eom_ret', 'rf').

    Example:
        market = load_and_filter_market_returns_test("data_test/market_returns_test.csv")
        print(market.head())
        risk_free = process_risk_free_rate("data_test/risk_free_test.csv")
        print(risk_free.head())
        wealth_end = pf_set["wealth"]
        end = "2023-11-30"

        wealth = wealth_func(wealth_end, end, market, risk_free)
        print(wealth.head())
        print(wealth.tail())

    Returns:
        pd.DataFrame: Med kolonner: 'eom', 'wealth', 'mu_ld1' (log-retur).
    """

    # --- 1) Find dato-kolonnen i hver DataFrame ---
    def find_date_col(df):
        if "eom" in df.columns:
            return "eom"
        elif "eom_ret" in df.columns:
            return "eom_ret"
        else:
            raise ValueError("Ingen 'eom' eller 'eom_ret' kolonne i DataFrame")

    risk_free_date = find_date_col(risk_free)
    market_date    = find_date_col(market)

    # --- 2) Omdøb til et fælles navn, fx "date" ---
    risk_free = risk_free.rename(columns={risk_free_date: "date"})
    market    = market.rename(columns={market_date: "date"})

    # --- 3) Konverter 'date' til datetime i begge DataFrames ---
    risk_free["date"] = pd.to_datetime(risk_free["date"])
    market["date"]    = pd.to_datetime(market["date"])

    # --- 4) Merge på "date" ---
    wealth = risk_free.merge(market, on="date", how="left")

    # --- 5) Beregninger ---
    # a) total return (tret = mkt_vw_exc + rf)
    wealth["tret"] = wealth["mkt_vw_exc"] + wealth["rf"]

    # b) Filtrér på slutdato
    wealth = wealth[wealth["date"] <= pd.to_datetime(end)]

    # c) Sortér i faldende rækkefølge (seneste dato øverst)
    wealth = wealth.sort_values(by="date", ascending=False)

    # d) Kumulativ formue: (1 - afkast) da mkt_vw_exc er eks. risikofri rente.
    #    Ret evt. efter definition af "tret" i dit datasæt
    wealth["wealth"] = (1 - wealth["tret"]).cumprod() * wealth_end

    # e) Lav 'eom' som sidste dag i måneden
    wealth["eom"] = wealth["date"].dt.to_period("M").dt.to_timestamp("M")

    # f) Tilføj en slut-række med wealth_end (så dataset slutter på 'end' i tabellen)
    final_row = pd.DataFrame({
        "eom": [pd.to_datetime(end)],
        "wealth": [wealth_end],
        "mu_ld1": [np.nan]
    })
    wealth = pd.concat([wealth, final_row], ignore_index=True)

    # g) Sortér i stigende rækkefølge (ældste dato først)
    wealth = wealth.sort_values(by="eom").reset_index(drop=True)

    # --- 6) Returnér de vigtige kolonner ---
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
    # Liste over kolonner, vi gerne vil have
    desired_cols = ["id", "eom", "sic", "ff49", "size_grp", "me", "crsp_exchcd", "rvol_252d", "dolvol_126d"] + features

    # Tjek hvilke kolonner der faktisk findes i Parquet-filen
    actual_cols = pd.read_parquet(file_path, engine="pyarrow").columns
    cols_to_load = [col for col in desired_cols if col in actual_cols]

    if not cols_to_load:
        raise ValueError("Ingen af de ønskede kolonner findes i Parquet-filen.")

    # Indlæs kun de kolonner, der faktisk findes
    data = pd.read_parquet(file_path, engine="pyarrow", columns=cols_to_load)

    # Filtrér observationer
    data = data[data["id"] <= 99999]

    # Konverter datoformat for 'eom'
    data["eom"] = pd.to_datetime(data["eom"], errors="coerce")

    return data









# last one - load dauly returns from usa_dsf_test
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
