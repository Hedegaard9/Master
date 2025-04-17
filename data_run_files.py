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

    df = pd.read_parquet(file_path_input, engine='pyarrow')
    print(f"Fil {file_path_input} indlæst med succes. Antal rækker før filtrering:", len(df))

    df_ids = pd.read_csv(file_path_id_test)
    ids_to_keep = set(df_ids['id'])

    df_filtered = df[df['id'].isin(ids_to_keep)].copy()
    print("Filtrering af ID'er udført. Antal rækker efter filtrering:", len(df_filtered))

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

    df_filtered.loc[:, 'eom'] = pd.to_datetime(df_filtered['eom'], errors='coerce')

    df_filtered = df_filtered[df_filtered['eom'] >= pd.to_datetime(start_date)]
    print(f"Filtrering på startdato {start_date} udført. Antal rækker efter filtrering:", len(df_filtered))

    unique_companies = df_filtered['id'].nunique()
    print(f"Antal unikke virksomheder efter filtrering: {unique_companies}")

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
    risk_free = pd.read_csv(file_path, usecols=["yyyymm", "RF"])

    risk_free['rf'] = risk_free['RF'] / 100
    risk_free['eom'] = risk_free['yyyymm'].astype(str) + "01"
    risk_free['eom'] = pd.to_datetime(risk_free['eom'], format="%Y%m%d") + MonthEnd(0)

    risk_free = risk_free[risk_free['eom'] >= pd.to_datetime(start_date)]

    if end_date is not None:
        risk_free = risk_free[risk_free['eom'] <= pd.to_datetime(end_date)]

    print(f"Filtrering udført. Antal rækker efter filtrering: {len(risk_free)}")

    risk_free.to_csv(output_path, index=False)
    print(f"Fil gemt som {output_path}")


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
    world_ret = pd.read_csv(file_path_world_ret)
    id_test = pd.read_csv(file_path_id_test)

    filtered_world_ret = world_ret[world_ret['id'].isin(id_test['id'])]

    filtered_world_ret['eom'] = pd.to_datetime(filtered_world_ret['eom'], format='%Y%m%d')

    filtered_world_ret = filtered_world_ret[
        (filtered_world_ret['eom'] >= start_date) & (filtered_world_ret['eom'] <= end_date)
        ]

    filtered_world_ret.to_csv(output_file, index=False)

    print(f"Data gemt i: {output_file}")
    print(filtered_world_ret.head())
    print(filtered_world_ret.tail())
    print(f"Shape efter dato-filter: {filtered_world_ret.shape}")

    unique_ids_count = filtered_world_ret['id'].nunique()
    print(f"Antal unikke ID'er efter dato-filter: {unique_ids_count}")



def monthly_returns(risk_free, h_list, file_path):
    """
    Behandler data ved at udvælge kolonner, filtrere USA-data og beregne langsigtede afkast.

    Parameters:
        risk_free (DataFrame): DataFrame med risikofri afkast, skal indeholde 'eom' og 'rf'.
        h_list (list): Liste af horisonter til beregning af langsigtede afkast.
        file_path (str): Sti til datafilen (CSV), der indeholder 'excntry', 'id', 'eom', 'ret_exc_lead1m'.

    # Kald funktionen:
    result = process_data(risk_free=risk_free, h_list=h_list, file_path=file_path_usa)

    Returns:
        DataFrame: Den samlede DataFrame med resultater for alle horisonter.
    """
    df_usa = pd.read_parquet(file_path, engine='pyarrow')
    kolonner = ["excntry", "id", "eom", "ret_exc_lead1m"]
    monthly = df_usa[kolonner]

    monthly = monthly.rename(columns={"ret_exc_lead1m": "ret_exc"})

    # Filtrer kun USA og id <= 99999 for at undgå index osv.
    monthly = monthly[(monthly["excntry"] == "USA") & (monthly["id"] <= 99999)]

    monthly["eom"] = pd.to_datetime(monthly["eom"], format="%Y%m%d")

    results = []

    for h in h_list:
        data_ret = GF.long_horizon_ret(data=monthly, h=h, impute="zero")

        data_ret["eom_ret"] = data_ret["eom"] + MonthEnd(1)

        data_ret_ld1 = data_ret[["id", "eom", "eom_ret", f"ret_ld{h}"]]

        data_ret_ld1 = data_ret_ld1.merge(risk_free, on="eom", how="left")

        data_ret_ld1["tr_ld1"] = data_ret_ld1[f"ret_ld{h}"] + data_ret_ld1["rf"]
        data_ret_ld1.drop(columns=["rf"], inplace=True)

        data_ret_ld1["horizon"] = h

        results.append(data_ret_ld1)

    final_result = pd.concat(results, ignore_index=True)
    final_result.to_csv(file_path, index=False)
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

    market_returns = load_and_filter_market_returns(file_path)
    market_returns['eom'] = pd.to_datetime(market_returns['eom'])

    filtered_df = market_returns[(market_returns['eom'] >= start_date) & (market_returns['eom'] <= end_date)]

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
    start_date = "2009-12-31"
    end_date = "2023-11-30"

    # Filtrér ID'er for usa_dsf og usa
    filter_ids_from_dataset(file_path_usa_dsf, file_path_id_test, output_path_usa_dsf, start_date)
    filter_ids_from_dataset(file_path_usa, file_path_id_test, output_path_usa, start_date)

    # Processér risikofri rente
    rente_path = "Data/ff3_m.csv"
    risk_free = process_risk_free_rate(rente_path, start_date)

    #Dan ret_monthly_test fil
    world_ret_monthly_test_filter(file_path_id_test, file_path_world_ret, start_date, end_date)

    filter_and_save_data(file_path_market_returns, start_date, end_date, output_path_market_returns)
    h_list = [1]  # Horisont
    dataret = monthly_returns(risk_free, h_list, output_path_usa)
    print(dataret.head())
    print("Alle processer er gennemført!")


# ===============================
# KØR SCRIPTET
# ===============================
if __name__ == "__main__":
    main()