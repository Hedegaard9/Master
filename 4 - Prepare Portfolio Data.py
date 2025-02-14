import pandas as pd
import numpy as np
import os
import pickle
from pandas.tseries.offsets import MonthEnd


def add_return_predictions(chars, settings, get_from_path_model): # ikke testet endnu
    """
    Tilføjer return predictions til `chars` baseret på data fra pickle-filer.

    Args:
        chars (pd.DataFrame): Dataframe med aktie-karakteristika.
        settings (dict): Indstillinger, herunder antal prediction-filer.
        get_from_path_model (str): Sti til mappen med model-filer.

    Returns:
        pd.DataFrame: Opdateret `chars` med return predictions.

    Example:
        chars = add_return_predictions(chars, settings, get_from_path_model)
    """
    for h in range(1, settings['pf']['hps']['m1']['K'] + 1):
        file_path = os.path.join(get_from_path_model, f"model_{h}.pkl")  # Brug pickle i stedet for RDS

        with open(file_path, 'rb') as f:
            pred_data = pickle.load(f)  # Indlæs pickle-fil

        pred_df = pd.DataFrame(pred_data['pred'])  # Konverter til DataFrame
        pred_df = pred_df[['id', 'eom', 'pred']].rename(columns={'pred': f'pred_ld{h}'})  # Omdøb kolonne

        chars = chars.merge(pred_df, on=['id', 'eom'], how='left')  # Merge forudsigelser ind i chars

    return chars


# Create lambda list ------------------------------------- virker
def create_lambda_list(chars):
    """
    Opretter en dictionary (`lambda_list`) af aktiers lambda-værdier pr. dato.

    Args:
        chars (pd.DataFrame): Dataframe med aktie-karakteristika inkl. `eom` og `lambda`.

    Returns:
        dict: `lambda_list` hvor nøgler er datoer og værdier er dictionaries af {id: lambda}.

    Example: lambda_list = create_lambda_list(chars)
    """
    lambda_dates = chars['eom'].unique()

    lambda_list = {
        str(d): chars[chars['eom'] == d][['id', 'lambda']].set_index('id')['lambda'].to_dict()
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
    dates_m1 = pd.date_range(start=settings['split']['train_end'] + pd.DateOffset(days=1),
                             end=settings['split']['test_end'] + pd.DateOffset(days=1) - MonthEnd(1), freq='ME')

    dates_m2 = pd.date_range(start=first_cov_date + pd.DateOffset(months=pf_set['lb_hor'] + 1),
                             end=settings['split']['test_end'] + pd.DateOffset(days=1) - MonthEnd(1), freq='ME')

    dates_oos = pd.date_range(start=datetime(start_oos, 1, 1),
                              end=settings['split']['test_end'] + pd.DateOffset(days=1) - MonthEnd(1), freq='ME')

    dates_hp = pd.date_range(start=datetime(min(hp_years), 1, 1),
                             end=settings['split']['test_end'] + pd.DateOffset(days=1) - MonthEnd(1), freq='ME')

    return {
        "dates_m1": dates_m1,
        "dates_m2": dates_m2,
        "dates_oos": dates_oos,
        "dates_hp": dates_hp
    }