import pandas as pd


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