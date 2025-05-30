
import pandas as pd
from Main import settings, features, pf_set
import time
import pickle
import pandas as pd
from pandas.tseries.offsets import DateOffset
import Prepare_Data
import data_run_files
import return_prediction_functions
from pandas.tseries.offsets import MonthEnd
# exec(open("Prepare_Data.py").read()) # udkommenteret, da jeg ikke lige ved, om den bliver brugt.
# exec(open("data_run_files.py").read()) # mulighed for denne løsning
file_path_usa_dsf = "./data_fifty/usa_dsf.parquet"
file_path_usa = "./data_fifty/usa_rvol.parquet"
file_path_id_test = "./data_fifty/top_5_percent_ids.csv"
file_path_market_returns = "./data_fifty/market_returns_test.csv"
output_path_usa_dsf = "./data_fifty/usa_dsf_test.parquet"
output_path_usa = "./data_fifty/usa_test.parquet"
output_path_market_returns = "data_fifty/market_returns_test.csv"
file_path_world_ret = "./data_fifty/world_ret_test.csv"
start_date = pd.to_datetime('1952-12-31')
end_date = pd.to_datetime('2020-12-31')

output_path = "./data_fifty/"

file_path_usa_test = "./data_fifty/usa_test.parquet"
daily_file_path = "./data_fifty/usa_dsf_test.parquet"
file_path_world_ret = "./data_fifty/world_ret_test.csv"
risk_free_path = "./data_fifty/risk_free_test.csv"
market_path = "./data_fifty/market_returns_test.csv"

rente_path = "data_fifty/ff3_m.csv"

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
models = []

# Iterer over rækkerne i search_grid
for i in range(len(search_grid)):
    h = search_grid.iloc[i]["horizon"]  # fx 1
    col_name = "ret_ld" + str(h)
    pred_y_values = data_ret[col_name]
    pred_y_df = data_ret[['id', 'eom']].copy()
    pred_y_df['eom_pred_last'] = pred_y_df['eom'] + MonthEnd(1)
    pred_y_df['ret_pred'] = pred_y_values
    valid_chars = chars[chars['valid'] == True]
    data_pred = pd.merge(pred_y_df, valid_chars, on=['id', 'eom'], how='inner')
    print("data_pred.head")
    print(data_pred.head())
    print("data_pred.tail")
    print(data_pred.tail())
    print("horizons:", [h])

    update_freq = settings['split']['model_update_freq']
    if update_freq == "once":
        val_ends = [settings['split']['train_end']]
        test_inc = 1000
    elif update_freq == "yearly":
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

    op = {}
    inner_start = time.time()
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

    model_filename = f"{output_path}/model_{h}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(op, f)

    models.append(op)

total_time = time.time() - start_time
print("Total run time:", total_time, "seconds")