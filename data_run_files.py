import pandas as pd
from pandas.tseries.offsets import MonthEnd
import General_Functions as GF
from Prepare_Data import load_and_filter_market_returns

def filter_ids_from_dataset(file_path_input, file_path_id_test, output_path, start_date):
    """
    Indlæser dataset fra en Parquet-fil, beholder kun ID'er fra en liste, filtrerer på startdato og gemmer resultatet.

    Args:
        file_path_input (str): Sti til den originale Parquet-fil (kan være usa_dsf eller usa).
        file_path_id_test (str): Sti til CSV-filen med ID'er, der skal beholdes.
        output_path (str): Sti til at gemme det filtrerede dataset.
        start_date (str): Startdato i formatet 'YYYY-MM-DD' for at filtrere data.

    Example:
        # Kør på usa_dsf data
        file_path_usa_dsf = "./Data/usa_dsf.parquet"
        file_path_id_test = "./data_test/top_5_percent_ids.csv"
        output_path = "./data_test/usa_dsf_test.parquet"
        start_date = "2010-01-31"
        filter_ids_from_dataset(file_path_usa_dsf, file_path_id_test, output_path, start_date)

        # Kør på usa data
        file_path_usa = "./Data/usa.parquet"
        file_path_id_test = "./data_test/top_5_percent_ids.csv"
        output_path = "./data_test/usa_test.parquet"
        start_date = "2010-01-31"
        filter_ids_from_dataset(file_path_usa, file_path_id_test, output_path, start_date)
    """

    # Indlæs det oprindelige dataset
    df = pd.read_parquet(file_path_input, engine='pyarrow')
    print(f"Fil {file_path_input} indlæst med succes. Antal rækker før filtrering:", len(df))

    # Indlæs ID'erne, der skal beholdes
    df_ids = pd.read_csv(file_path_id_test)
    ids_to_keep = set(df_ids['id'])  # Konverter til et set for hurtigere opslag

    # Filtrer dataset ved kun at beholde de ID'er fra CSV-filen
    df_filtered = df[df['id'].isin(ids_to_keep)].copy()  # Tilføj .copy() for at undgå advarsler
    print("Filtrering af ID'er udført. Antal rækker efter filtrering:", len(df_filtered))

    # Tjek om 'eom' findes, og find alternativ hvis ikke
    if 'eom' not in df_filtered.columns:
        alternative_columns = ['date', 'timestamp', 'end_of_month', 'period']
        found_col = None
        for col in alternative_columns:
            if col in df_filtered.columns:
                found_col = col
                df_filtered.loc[:, 'eom'] = df_filtered[col]  # Brug .loc for at undgå warning
                print(f"Bruger '{col}' i stedet for 'eom'.")
                break
        if not found_col:
            raise KeyError("Kolonnen 'eom' findes ikke i dataset. Tjek dine data.")

    # Konverter 'eom' til datetime for filtrering
    df_filtered.loc[:, 'eom'] = pd.to_datetime(df_filtered['eom'], errors='coerce')

    # Filtrer på startdato
    df_filtered = df_filtered[df_filtered['eom'] >= pd.to_datetime(start_date)]
    print(f"Filtrering på startdato {start_date} udført. Antal rækker efter filtrering:", len(df_filtered))

    # Udskriv antal unikke virksomheder efter filtrering
    unique_companies = df_filtered['id'].nunique()
    print(f"Antal unikke virksomheder efter filtrering: {unique_companies}")

    # Gem det filtrerede dataset som en Parquet-fil
    df_filtered.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Fil gemt som {output_path}")


def process_risk_free_rate(file_path, start_date, end_date=None, output_path="./data_test/risk_free_test.csv"):
    """
    Indlæser en CSV-fil med risikofri rente-data, konverterer datoer og filtrerer på startdato (og evt. slutdato),
    og gemmer resultatet som en CSV-fil i 'data_test'-mappen.

    Args:
        file_path (str): Stien til CSV-filen med risikofri rente-data.
        start_date (str): Startdato i formatet 'YYYY-MM-DD'.
        end_date (str, optional): Slutdato i formatet 'YYYY-MM-DD'. Hvis None, filtreres kun på startdato.
        output_path (str, optional): Sti til den filtrerede CSV-fil. Standard er "./data_test/risk_free_test.csv".

    Example:
        rente_path = "Data/ff3_m.csv"
        start_date = "2010-01-31"

        # Med både start- og slutdato
        end_date = "2023-11-30"
        process_risk_free_rate(rente_path, start_date, end_date)

        # Kun med startdato (ingen slutdato)
        process_risk_free_rate(rente_path, start_date)

    Returns:
        pd.DataFrame: En DataFrame med to kolonner: 'eom' (slutningen af måneden) og 'rf' (risikofri rente i procent).
    """
    # Læs data fra filen
    risk_free = pd.read_csv(file_path, usecols=["yyyymm", "RF"])

    # Opret nye kolonner for risikofri rente og slutningen af måneden
    risk_free['rf'] = risk_free['RF'] / 100  # Konverter risikofri rente til procent
    risk_free['eom'] = risk_free['yyyymm'].astype(str) + "01"
    risk_free['eom'] = pd.to_datetime(risk_free['eom'], format="%Y%m%d") + MonthEnd(0)

    # Filtrer på startdato
    risk_free = risk_free[risk_free['eom'] >= pd.to_datetime(start_date)]

    # Hvis slutdato er angivet, filtrer også på den
    if end_date is not None:
        risk_free = risk_free[risk_free['eom'] <= pd.to_datetime(end_date)]

    print(f"Filtrering udført. Antal rækker efter filtrering: {len(risk_free)}")

    # Gem det filtrerede dataset som CSV i `data_test`
    risk_free.to_csv(output_path, index=False)
    print(f"Fil gemt som {output_path}")

    # Returner det filtrerede DataFrame
    return risk_free[['eom', 'rf']]



def world_ret_monthly_test_filter(file_path_id_test, file_path_world_ret, start_date, end_date, output_file="./data_test/world_ret_test.csv"):
    """
    Filtrerer world_ret-data for kun at inkludere ID'er fra id_test samt rækker inden for en bestemt dato-interval.

    Parameters:
    file_path_id_test (str): Sti til CSV-fil med ID'er.
    file_path_world_ret (str): Sti til CSV-fil med world_ret-data.
    start_date (str): Startdato i formatet "YYYY-MM-DD".
    end_date (str): Slutdato i formatet "YYYY-MM-DD".
    output_file (str): Sti til output CSV-fil (default: "filtered_world_ret_filtered_by_date.csv").

    Returns:
    None: Funktionen gemmer den filtrerede DataFrame som en CSV-fil.

    Example:
    file_path_id_test = "./data_test/top_5_percent_ids.csv"
    file_path_world_ret = "./Data/world_ret_monthly.csv"
    start_date = "2010-01-31"
    end_date = "2023-11-30"

    world_ret_monthly_test_filter(file_path_id_test, file_path_world_ret, start_date, end_date)
    """
    # Læs CSV-filer
    world_ret = pd.read_csv(file_path_world_ret)
    id_test = pd.read_csv(file_path_id_test)

    # Filtrér world_ret for kun at inkludere id'er fra id_test
    filtered_world_ret = world_ret[world_ret['id'].isin(id_test['id'])]

    # Konverter 'eom' til datetime-format
    filtered_world_ret['eom'] = pd.to_datetime(filtered_world_ret['eom'], format='%Y%m%d')

    # Filtrér baseret på dato-intervallet
    filtered_world_ret = filtered_world_ret[
        (filtered_world_ret['eom'] >= start_date) & (filtered_world_ret['eom'] <= end_date)
        ]

    # Gem den filtrerede DataFrame til en ny CSV-fil
    filtered_world_ret.to_csv(output_file, index=False)

    # Bekræftelse
    print(f"Data gemt i: {output_file}")
    print(filtered_world_ret.head())
    print(filtered_world_ret.tail())
    print(f"Shape efter dato-filter: {filtered_world_ret.shape}")

    # Tæl unikke ID'er efter filtrering
    unique_ids_count = filtered_world_ret['id'].nunique()
    print(f"Antal unikke ID'er efter dato-filter: {unique_ids_count}")



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
    final_result.to_csv("data_test/monthly_preprocessed_test.csv", index=False)
    # Returner den kombinerede DataFrame
    return final_result

def filter_and_save_data(file_path, start_date, end_date, output_path):
    """
    Indlæser vores market returns, filtrerer data efter start/slutdato, og gemmer den filtrerede version.

    Parametre:
    file_path (str): Sti til input CSV-fil.
    start_date (str): Startdato i format 'YYYY-MM-DD'.
    end_date (str): Slutdato i format 'YYYY-MM-DD'.
    output_path (str): Sti til output CSV-fil.

    Example:
    file_path_market_returns = "./Data/market_returns.csv"
    output_path = "data_test/market_returns_test.csv"
    start_date = "2010-01-31"
    end_date = "2023-11-30"
    filter_and_save_data(file_path_market_returns, start_date, end_date, output_path)
    """

    # Læs CSV-fil
    market_returns = load_and_filter_market_returns(file_path)

    # Konverter eom til datetime-format
    market_returns['eom'] = pd.to_datetime(market_returns['eom'])

    # Filtrer data baseret på dato
    filtered_df = market_returns[(market_returns['eom'] >= start_date) & (market_returns['eom'] <= end_date)]

    # Gem den filtrerede data til en ny CSV-fil
    filtered_df.to_csv(output_path, index=False)

    print(f"Fil gemt som {output_path} med {len(filtered_df)} rækker.")


# ===============================
# MAIN FUNKTION
# ===============================
def main():
    # Filstier og parametre
    file_path_usa_dsf = "./Data/usa_dsf.parquet"
    file_path_usa = "./Data/usa_rvol.parquet"
    file_path_id_test = "./data_test/top_5_percent_ids.csv"
    file_path_market_returns = "./Data/market_returns.csv"
    output_path_usa_dsf = "./data_test/usa_dsf_test.parquet"
    output_path_usa = "./data_test/usa_test.parquet"
    output_path_market_returns = "data_test/market_returns_test.csv"
    file_path_world_ret = "./Data/world_ret_monthly.csv"
    start_date = "2010-01-31"
    end_date = "2023-11-30"

    # Filtrér ID'er for usa_dsf og usa
    filter_ids_from_dataset(file_path_usa_dsf, file_path_id_test, output_path_usa_dsf, start_date)
    filter_ids_from_dataset(file_path_usa, file_path_id_test, output_path_usa, start_date)

    # Processér risikofri rente
    rente_path = "Data/ff3_m.csv"
    risk_free = process_risk_free_rate(rente_path, start_date)

    #Dan ret_monthly_test fil
    world_ret_monthly_test_filter(file_path_id_test, file_path_world_ret, start_date, end_date)

    # lav testsæt for market returns
    filter_and_save_data(file_path_market_returns, start_date, end_date, output_path_market_returns)
    # Beregn månedlige afkast
    h_list = [1]  # Horisonter
    dataret = monthly_returns(risk_free, h_list, output_path_usa)
    print(dataret.head())
    print("Alle processer er gennemført!")


# ===============================
# KØR SCRIPTET
# ===============================
if __name__ == "__main__":
    main()