import pandas as pd
import numpy as np
import os
import pickle
from pandas.tseries.offsets import MonthEnd
from Main import settings, features, pf_set
from datetime import datetime
import data_run_files
import Prepare_Data
import Estimate_Covariance_Matrix as ECM

def add_return_predictions(chars, settings, get_from_path_model):
    """
    Tilføjer return predictions til `chars` baseret på data fra pickle-filer.

    Args:
        chars (pd.DataFrame): Dataframe med aktie-karakteristika.
        settings (dict): Indstillinger, herunder antal prediction-filer.
        get_from_path_model (str): Sti til mappen med model-filer.

    Returns:
        pd.DataFrame: Opdateret `chars` med return predictions.

    Example:
        output_path_usa = "./data_test/usa_test.parquet"
        start_date = "2010-01-31"
        rente_path = "Data/ff3_m.csv"
        daily_file_path = "./data_test/usa_dsf_test.parquet"
        file_path_world_ret = "./data_test/world_ret_test.csv"
        file_path_usa_test = "./data_test/usa_test.parquet"
        risk_free_path = "./data_test/risk_free_test.csv"
        market_path = "./data_test/market_returns_test.csv"
        daily_file_path = "./data_test/usa_dsf_test.parquet"
        risk_free = data_run_files.process_risk_free_rate(rente_path, start_date)
        h_list = [1]  # Horisonter
        data_ret = data_run_files.monthly_returns(risk_free, h_list, output_path_usa)
        chars, daily = Prepare_Data.process_all_data(file_path_usa_test, daily_file_path, file_path_world_ret, risk_free_path, market_path)
    """
    for h in range(1, settings['pf']['hps']['m1']['K'] + 1):
        file_path = os.path.join(get_from_path_model, f"model_{h}.pkl")
        with open(file_path, 'rb') as f:
            model_dict = pickle.load(f)  # Indlæs pickle-fil

        # Iterer over dato-nøgler og udtræk forudsigelser
        all_preds = []
        for date_key, subdict in model_dict.items():
            # Antag at hver subdictionary indeholder en key "pred"
            pred_df = pd.DataFrame(subdict['pred'])
            all_preds.append(pred_df)

        # Saml alle forudsigelsesdata
        pred_df_all = pd.concat(all_preds, ignore_index=True)
        pred_df_all = pred_df_all[['id', 'eom', 'pred']].rename(columns={'pred': f'pred_ld{h}'})

        # Merge forudsigelserne ind i chars
        chars = chars.merge(pred_df_all, on=['id', 'eom'], how='left')

    return chars


# Create lambda list ------------------------------------- virker
def create_lambda_list(chars):
    """
    Opretter en dictionary (`lambda_list`) af aktiers lambda-værdier pr. dato.

    Args:
        chars (pd.DataFrame): Dataframe med aktie-karakteristika inkl. `eom` og `lambda`.

    Returns:
        dict: `lambda_list` hvor nøgler er datoer i formatet 'YYYY-MM-DD' og værdier er dictionaries af {id: lambda}.

    Example: lambda_list = create_lambda_list(chars)
    """
    lambda_dates = chars['eom'].unique()

    lambda_list = {
        pd.to_datetime(d).strftime('%Y-%m-%d'): chars[chars['eom'] == d][['id', 'lambda']].set_index('id')['lambda'].to_dict()
        for d in lambda_dates
    }

    return lambda_list



# Important dates ---------------------- virker

def define_important_dates(barra_cov, settings):
    """
    Definerer vigtige datoer for analyse baseret på `barra_cov` og `settings`.

    Args:
        barra_cov (dict): Dictionary med kovariansdata (faktor-kovariansmatricer pr. dato).
        settings (dict): Indstillinger, inkl. start- og slutår for analyse.

    Returns:
        tuple: first_cov_date, hp_years, start_oos.

    Example:
        first_cov_date, hp_years, start_oos = define_important_dates(barra_cov, settings)
    """
    first_cov_date = min(pd.to_datetime(list(barra_cov.keys())))  # Find den tidligste dato i barra_cov
    hp_years = list(range(settings['pf']['dates']['start_year'], settings['pf']['dates']['end_yr'] + 1))
    start_oos = settings['pf']['dates']['start_year'] + settings['pf']['dates']['split_years']

    return first_cov_date, hp_years, start_oos

def create_date_ranges(settings, first_cov_date, start_oos, hp_years):
    """
    Opretter datointervaller til forskellige analyseperioder.

    Args:
        settings (dict): Indstillinger, inkl. trænings- og testperioder.
        first_cov_date (datetime): Tidligste dato i `barra_cov`.
        start_oos (int): Startår for out-of-sample periode.
        hp_years (list): Liste over holdperioder.

    Returns:
        dict: Datoer for `m1`, `m2`, `oos` og `hp` perioder.

    Example:
        date_ranges = create_date_ranges(settings, first_cov_date, start_oos, hp_years)
        date_ranges["dates_m1"]
        print(date_ranges)
    """
    end_date = settings['split']['test_end'] - MonthEnd(1)
    start_date_oos = datetime(start_oos - 1, 12, 31)
    start_date_hp = datetime(min(hp_years), 1, 1) - MonthEnd(1)

    dates_m1 = pd.date_range(start=settings['split']['train_end'], end=end_date, freq='ME')

    dates_m2 = pd.date_range(start=first_cov_date + pd.DateOffset(months=pf_set['lb_hor'] + 1), end=end_date, freq='ME')

    dates_oos = pd.date_range(start=start_date_oos, end=end_date, freq='ME')

    dates_hp = pd.date_range(start=start_date_hp,end=end_date, freq='ME')

    return {
        "dates_m1": dates_m1,
        "dates_m2": dates_m2,
        "dates_oos": dates_oos,
        "dates_hp": dates_hp
    }

def main(barra_cov):
    get_from_path_model = "./data_test/"
    output_path_usa = "./data_test/usa_test.parquet"
    start_date = "2010-01-31"
    rente_path = "Data/ff3_m.csv"
    daily_file_path = "./data_test/usa_dsf_test.parquet"
    file_path_world_ret = "./data_test/world_ret_test.csv"
    file_path_usa_test = "./data_test/usa_test.parquet"
    risk_free_path = "./data_test/risk_free_test.csv"
    market_path = "./data_test/market_returns_test.csv"
    daily_file_path = "./data_test/usa_dsf_test.parquet"
    risk_free = data_run_files.process_risk_free_rate(rente_path, start_date)
    h_list = [1]  # Horisonter
    data_ret = data_run_files.monthly_returns(risk_free, h_list, output_path_usa)
    chars, daily = Prepare_Data.process_all_data(file_path_usa_test, daily_file_path, file_path_world_ret,risk_free_path, market_path)
    chars = add_return_predictions(chars, settings, get_from_path_model) # mangler pickle funktion
    lambda_list = create_lambda_list(chars)
    first_cov_date, hp_years, start_oos = define_important_dates(barra_cov, settings)
    date_ranges = create_date_ranges(settings, first_cov_date, start_oos, hp_years)

    return chars, lambda_list, first_cov_date, hp_years, start_oos, date_ranges


