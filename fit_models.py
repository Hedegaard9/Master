
import pandas as pd
from Main import settings, features, pf_set
import time
import pickle
import pandas as pd
from pandas.tseries.offsets import DateOffset
import Prepare_Data
import data_run_files
import return_prediction_functions
# exec(open("Prepare_Data.py").read()) # udkommenteret, da jeg ikke lige ved, om den bliver brugt.
# exec(open("data_run_files.py").read()) # mulighed for denne løsning
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

output_path = "./data_test/"

file_path_usa_test = "./data_test/usa_test.parquet"
daily_file_path = "./data_test/usa_dsf_test.parquet"
file_path_world_ret = "./data_test/world_ret_test.csv"
risk_free_path = "./data_test/risk_free_test.csv"
market_path = "./data_test/market_returns_test.csv"

rente_path = "Data/ff3_m.csv"


# Kører funktioner fra data_run_files
risk_free = data_run_files.process_risk_free_rate(rente_path, start_date)
h_list = [1]  # Horisonter
data_ret = data_run_files.monthly_returns(risk_free, h_list, output_path_usa)
chars, daily = Prepare_Data.process_all_data(file_path_usa_test, daily_file_path, file_path_world_ret, risk_free_path, market_path)

print("risk_free.head")
print(risk_free.head())
print(h_list)
print("data_ret.head")
print(data_ret.head())
print("chars.head")
print(chars.head())
print("daily.head")
print(daily.head())
# Version med 12 horisonter
#search_grid = pd.DataFrame({
#    'name': [f'm{i}' for i in range(1, 13)],
#    'horizon': list(range(1, 13))
#})


# Version med kun én horizon (kun h=1)
search_grid_single = pd.DataFrame({
    'name': ['m1'],
    'horizon': [1]
})



search_grid = search_grid_single

start_time = time.time()
models = []  # Liste til at gemme output for hver horizon

# Iterer over rækkerne i search_grid
for i in range(len(search_grid)):
    # Forbered y-variablen:
    h = search_grid.iloc[i]["horizon"]  # fx 1
    # Konstruer kolonnenavnet, fx "ret_ld1"
    col_name = "ret_ld" + str(h)
    # Beregn gennemsnittet pr. række – hvis der kun er én kolonne, svarer det til kolonneværdierne
    pred_y_values = data_ret[col_name]  # antages at være en Pandas Series
    # Opret en DataFrame med de nødvendige kolonner: id, eom
    pred_y_df = data_ret[['id', 'eom']].copy()
    # Beregn eom_pred_last: Vi antager, at vi ønsker at beregne slutdatoen for forudsigelsen som:
    # eom + (h+1) måneder - 1 dag (svarer nogenlunde til R's eom+1+months(max(h))-1)
    pred_y_df['eom_pred_last'] = pred_y_df['eom'] + DateOffset(months=int(h) + 1) - DateOffset(days=1)
    pred_y_df['ret_pred'] = pred_y_values

    # Join med de rækker i 'chars', hvor valid er True, på kolonnerne id og eom
    valid_chars = chars[chars['valid'] == True]
    data_pred = pd.merge(pred_y_df, valid_chars, on=['id', 'eom'], how='inner')
    print("data_pred.head")
    print(data_pred.head())
    print("data_pred.tail")
    print(data_pred.tail())
    print("horizons:", [h])

    # Bestem validerings-slutdatoer og test-inc baseret på settings
    update_freq = settings['split']['model_update_freq']
    if update_freq == "once":
        val_ends = [settings['split']['train_end']]
        test_inc = 1000
    elif update_freq == "yearly":
        # Opret en liste af datoer med årligt interval
        val_ends = pd.date_range(start=settings['split']['train_end'],
                                 end=settings['split']['test_end'],
                                 freq='YE').to_pydatetime().tolist()
        test_inc = 1
    elif update_freq == "decade":
        start_date = pd.to_datetime(settings['split']['train_end'])
        end_date = pd.to_datetime(settings['split']['test_end'])
        val_ends = []
        current = start_date
        while current <= end_date:
            val_ends.append(current)
            current += pd.DateOffset(years=10)
        test_inc = 10
    else:
        raise ValueError("Ugyldig model_update_freq i settings.")

    op = {}  # Dictionary til at gemme modeloutput for hver val_end
    inner_start = time.time()
    # Iterer over hver validerings-slutdato
    for val_end in val_ends:
        print(val_end)
        train_test_val = return_prediction_functions.data_split(
            data_pred,
            type=update_freq,
            val_end=val_end,
            val_years=settings['split']['val_years'],
            train_start=settings['screens']['start'],
            train_lookback=settings['split']['train_lookback'],
            retrain_lookback=settings['split']['retrain_lookback'],
            test_inc=test_inc,
            test_end=settings['split']['test_end']
        )
        print("Train data:")
        print(train_test_val["train"].head())
        print("\nValidation data:")
        print(train_test_val["val"].head())
        print("\nTrain full data:")
        print(train_test_val["train_full"].head())
        print("\nTest data:")
        print(train_test_val["test"].head())
        model_start = time.time()
        model_op = return_prediction_functions.rff_hp_search(
            train_test_val,
            feat=features,
            p_vec=settings['rff']['p_vec'],
            g_vec=settings['rff']['g_vec'],
            l_vec=settings['rff']['l_vec'],
            seed=settings['seed_no']
        )
        model_time = time.time() - model_start
        print("Model training time:", model_time, "seconds")
        op[val_end] = model_op
    inner_time = time.time() - inner_start
    print("Total time for current horizon:", inner_time, "seconds")

    # Gem model-output for den aktuelle horizon til en pickle-fil (svarer til R's saveRDS)
    model_filename = f"{output_path}/model_{h}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(op, f)

    models.append(op)

total_time = time.time() - start_time
print("Total run time:", total_time, "seconds")