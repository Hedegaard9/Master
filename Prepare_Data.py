import pandas as pd
from pandas.tseries.offsets import MonthEnd

def process_risk_free_rate(file_path):
    """
    Args:
        file_path (str): Stien til CSV-filen med risikofri rente-data.
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
#Duummy check
rente_path = "Data/ff3_m.csv"
risk_free_df = process_risk_free_rate(rente_path)
print(risk_free_df)



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
    # result = process_cluster_labels(file_path_cluster_labels, file_path_factor_details)

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


def monthly_returns(risk_free, h_list, file_path):
    """
    Behandler data ved at udvælge kolonner, filtrere USA-data og beregne langsigtede afkast.

    Parameters:
        risk_free (DataFrame): DataFrame med risikofri afkast, skal indeholde 'eom' og 'rf'.
        h_list (list): Liste af horisonter til beregning af langsigtede afkast.
        file_path (str): Sti til datafilen (CSV), der indeholder 'excntry', 'id', 'eom', 'ret_exc_lead1m'.

    Example:
    h_list = [1,2]  # Liste af horisonter

    # Kald funktionen:
    result = process_data(risk_free=risk_free, h_list=h_list, file_path=file_path_usa)

    Returns:
        DataFrame: Den samlede DataFrame med resultater for alle horisonter.
    """
    # Læs data fra filen
    df_usa = pd.read_parquet(file_path, engine='pyarrow')
    kolonner = ["excntry", "id", "eom", "ret_exc_lead1m"]
    monthly = df_usa[kolonner]

    # Omdøb kolonnen 'ret_exc_lead1m' til 'ret_exc'
    monthly = monthly.rename(columns={"ret_exc_lead1m": "ret_exc"})

    # Filtrer kun USA og id <= 99999
    monthly = monthly[(monthly["excntry"] == "USA") & (monthly["id"] <= 99999)]

    # Konverter 'eom' til datoformat
    monthly["eom"] = pd.to_datetime(monthly["eom"], format="%Y%m%d")

    # Initialiser en tom liste til resultater
    results = []

    # Loop over horisonter i h_list
    for h in h_list:
        # Beregn langsigtede afkast
        data_ret = General_Functions.long_horizon_ret(data=monthly, h=h, impute="zero")

        # Tilføj 'eom_ret'
        data_ret["eom_ret"] = data_ret["eom"] + MonthEnd(1)

        # Filtrer til relevante kolonner
        data_ret_ld1 = data_ret[["id", "eom", "eom_ret", f"ret_ld{h}"]]

        # Merge med risikofri data
        data_ret_ld1 = data_ret_ld1.merge(risk_free, on="eom", how="left")

        # Beregn total return
        data_ret_ld1["tr_ld1"] = data_ret_ld1[f"ret_ld{h}"] + data_ret_ld1["rf"]
        data_ret_ld1.drop(columns=["rf"], inplace=True)

        # Tilføj horisont som kolonne
        data_ret_ld1["horizon"] = h

        # Gem resultatet for denne horisont
        results.append(data_ret_ld1)

    # Kombiner alle resultater til én samlet DataFrame
    final_result = pd.concat(results, ignore_index=True)
    final_result.to_csv("Data/monthly_preprocessed.csv", index=False)
    # Returner den kombinerede DataFrame
    return final_result