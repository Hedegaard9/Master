import pandas as pd

def get_latest_ticker_data(input_file, output_file):
    """
    Indlæser en CSV-fil, filtrerer relevante kolonner, og returnerer den seneste dato pr. PERMNO.

    Parametre:
    - input_file (str): Stien til input CSV-fil (f.eks. 'data/crsp_a_stock_codes.csv').
    - output_file (str): Stien til output CSV-fil.

    Returnerer:
    - pd.DataFrame: Filtreret DataFrame med den seneste dato per PERMNO.
    # Eksempel på brug:
    #latest_ticker_data = get_latest_ticker_data('data/crsp_a_stock_codes.csv', 'Data/ticker_and_id.csv')

    """
    # Indlæs data
    ticker_data = pd.read_csv(input_file)

    # Konverter 'date' til datetime
    ticker_data["date"] = pd.to_datetime(ticker_data["date"])

    # Filtrer relevante kolonner
    df_filtered = ticker_data[["PERMNO", "TICKER", "COMNAM", "date"]]

    # Behold kun den seneste dato per unikt PERMNO
    df_latest = df_filtered.sort_values("date").drop_duplicates(subset=["PERMNO"], keep="last")

    # Gem som CSV (skal angives af brugeren)
    df_latest.to_csv(output_file, index=False)

    # Returner den filtrerede DataFrame
    return df_latest


#Gammel funktion for Prepare matrix of future monthly returns ---------------
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
        data_ret = GF.long_horizon_ret(data=monthly, h=h, impute="zero")

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