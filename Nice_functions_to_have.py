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

