import numpy as np
import pandas as pd

def filter_ids_from_dataset(file_path_usa_dsf, file_path_id_test, output_path, start_date):
    """
    Indlæser dataset fra en Parquet-fil, beholder kun ID'er fra en liste, filtrerer på startdato og gemmer resultatet.

    Args:
        file_path_usa_dsf (str): Sti til den originale Parquet-fil.
        file_path_id_test (str): Sti til CSV-filen med ID'er, der skal beholdes.
        output_path (str): Sti til at gemme det filtrerede dataset.
        start_date (str): Startdato i formatet 'YYYY-MM-DD' for at filtrere data.

        Example:
        file_path_usa_dsf = "./Data/usa_dsf.parquet"  # Din input Parquet-fil
        file_path_id_test = "./data_test/top_5_percent_ids.csv"  # CSV-fil med ID'er
        output_path = "./data_test/filtered_data.parquet"  # Output-fil
        start_date = "2010-01-31"  # Vælg din startdato

        filter_ids_from_dataset(file_path_usa_dsf, file_path_id_test, output_path, start_date)
    """
    # Indlæs det oprindelige dataset
    df = pd.read_parquet(file_path_usa_dsf, engine='pyarrow')
    print("Fil indlæst med succes. Antal rækker før filtrering:", len(df))

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
