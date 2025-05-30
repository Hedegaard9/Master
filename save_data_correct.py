
import pandas as pd
from Main import settings, features, pf_set
import time
import pickle
from pandas.tseries.offsets import DateOffset
import Prepare_Data
import data_run_files
import return_prediction_functions
from return_prediction_functions import rff
import numpy as np
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import Ridge
import re
import matplotlib.pyplot as plt
import seaborn as sns
import General_Functions
import Estimate_Covariance_Matrix
import datetime


file_path_usa_dsf = "./data_fifty/usa_dsf.parquet"
file_path_usa = "./data_fifty/usa_rvol.parquet"
file_path_id_test = "./data_fifty/top_5_percent_ids.csv"
file_path_market_returns = "./data_fifty/market_returns_test.csv"
output_path_usa_dsf = "./data_fifty/usa_dsf_test.parquet"
output_path_usa = "./data_fifty/usa_test.parquet"
output_path_market_returns = "data_fifty/market_returns_test.csv"
file_path_world_ret = "./data_fifty/world_ret_test.csv"
start_date = pd.to_datetime('1952-12-31')
end_date = pd.to_datetime('2022-12-31')

output_path = "./data_fifty/"

file_path_usa_test = "./data_fifty/usa_test.parquet"
daily_file_path = "./data_fifty/usa_dsf_test.parquet"
file_path_world_ret = "./data_fifty/world_ret_test.csv"
risk_free_path = "./data_fifty/risk_free_test.csv"
market_path = "./data_fifty/market_returns_test.csv"

#rente_path = "data_fifty/ff3_m.csv"
wealth_end = pf_set["wealth"]
end = settings["split"]["test_end"]
market_test = Prepare_Data.load_and_filter_market_returns_test(market_path)

risk_free = data_run_files.process_risk_free_rate(risk_free_path, start_date)
h_list = [1]
wealth = Prepare_Data.wealth_func(wealth_end, end, market_test, risk_free)


def create_data_ret_and_data_ret_ld1(file_path_world_ret, risk_free, output_data_ret_csv, output_data_ret_ld1_csv):
    df_test = pd.read_csv(file_path_world_ret)
    kolonner = ["excntry", "id", "eom", "ret_exc"]
    monthly = df_test[kolonner]
    monthly = monthly[(monthly["excntry"] == "USA") & (monthly["id"] <= 99999)]
    monthly["eom"] = pd.to_datetime(monthly["eom"])

    data_ret = General_Functions.long_horizon_ret(data=monthly,
                                                  h=settings['pf']['hps']['m1']['K'],
                                                  impute="zero")

    data_ret_ld1 = data_ret[['id', 'eom', 'ret_ld1']].copy()
    data_ret_ld1['eom_ret'] = data_ret_ld1['eom'] + MonthEnd(1)

    data_ret_ld1 = data_ret_ld1.merge(risk_free, on='eom', how='left')
    data_ret_ld1['tr_ld1'] = data_ret_ld1['ret_ld1'] + data_ret_ld1['rf']


    data_ret_ld1 = data_ret_ld1.drop(columns=['rf'])

    shifted = data_ret_ld1[['id', 'eom', 'tr_ld1']].copy()
    shifted['eom'] = shifted['eom'] + MonthEnd(1)
    shifted = shifted.rename(columns={'tr_ld1': 'tr_ld0'})

    data_ret_ld1 = data_ret_ld1.merge(shifted[['id', 'eom', 'tr_ld0']], on=['id', 'eom'], how='left')


    del monthly
    data_ret.to_csv(output_data_ret_csv, index=False)
    data_ret_ld1.to_csv(output_data_ret_ld1_csv, index=False)

    return data_ret, data_ret_ld1


output_data_ret_csv = "./data_fifty/data_ret.csv"
output_data_ret_ld1_csv = "./data_fifty/data_ret_ld1.csv"
data_ret, data_ret_ld1 = create_data_ret_and_data_ret_ld1(file_path_world_ret, risk_free, output_data_ret_csv, output_data_ret_ld1_csv)


data_ret = pd.read_csv(output_data_ret_csv)
data_ret_ld1 = pd.read_csv(output_data_ret_ld1_csv)
data_ret['eom'] = pd.to_datetime(data_ret['eom'])
data_ret_ld1['eom'] = pd.to_datetime(data_ret_ld1['eom'])
data_ret_ld1['eom_ret'] = pd.to_datetime(data_ret_ld1['eom_ret'])
## Skab data_ret_ld1




chars = pd.read_parquet(output_path_usa)



selected_cols = list(set(["id", "eom", "sic", "size_grp", "me", "rvol_252d", "dolvol_126d"] + features))
chars = pd.read_parquet(output_path_usa, columns=selected_cols)

chars = chars[chars['id'] <= 99999]
chars['eom'] = pd.to_datetime(chars['eom'])


chars['dolvol'] = chars['dolvol_126d']
chars['lambda'] = 2 / chars['dolvol'] * settings['pi']
chars['rvol_m'] = chars['rvol_252d'] * np.sqrt(21)

chars = pd.merge(chars, data_ret_ld1, on=['id', 'eom'], how='left')


temp = wealth.copy()
temp['eom'] = temp['eom'] + pd.offsets.MonthEnd(1)
temp = temp.rename(columns={'mu_ld1': 'mu_ld0'})
chars = pd.merge(chars, temp[['eom', 'mu_ld0']], on='eom', how='left')

date_excludes = ((chars['eom'] < settings['screens']['start']) | (chars['eom'] > settings['screens']['end'])).mean() * 100
print(f"   Date screen excludes {date_excludes:.2f}% of the observations")

chars = chars[(chars['eom'] >= settings['screens']['start']) & (chars['eom'] <= settings['screens']['end'])]

n_start = chars.shape[0]
me_start = chars['me'].sum(skipna=True)

me_missing_pct = chars['me'].isna().mean() * 100
print(f"   Non-missing me excludes {me_missing_pct:.2f}% of the observations")
chars = chars[~chars['me'].isna()]

valid_return_excludes = ((chars['tr_ld1'].isna()) | (chars['tr_ld0'].isna())).mean() * 100
print(f"   Valid return req excludes {valid_return_excludes:.2f}% of the observations")
chars = chars[chars['tr_ld0'].notna() & chars['tr_ld1'].notna()]

dolvol_excludes = ((chars['dolvol'].isna()) | (chars['dolvol'] == 0)).mean() * 100
print(f"   Non-missing/non-zero dolvol excludes {dolvol_excludes:.2f}% of the observations")

chars = chars[(~chars['dolvol'].isna()) & (chars['dolvol'] > 0)]

# --- Valid SIC code ---
sic_excludes = (chars['sic'] == "").mean() * 100
print(f"   Valid SIC code excludes {sic_excludes:.2f}% of the observations")

chars = chars[~chars['sic'].isna()]

# --- Feature screens ---
feat_available = chars[features].notna().sum(axis=1)

min_feat = int(np.floor(len(features) * settings['screens']['feat_pct']))

feat_excludes = (feat_available < min_feat).mean() * 100
print(f"   At least {settings['screens']['feat_pct']*100:.0f}% of feature excludes {feat_excludes:.2f}% of the observations")

chars = chars[feat_available >= min_feat]

final_obs_pct = (len(chars) / n_start) * 100
final_me_pct = (chars['me'].sum() / me_start) * 100
print(
    f"   In total, the final dataset has {final_obs_pct:.2f}% of the observations and {final_me_pct:.2f}% of the market cap in the post {settings['screens']['start']} data")
run_sub = False
if run_sub:
    np.random.seed(settings['seed'])
    unique_ids = chars['id'].unique()
    sample_ids = np.random.choice(unique_ids, size=2500, replace=False)
    chars = chars[chars['id'].isin(sample_ids)]
if settings['feat_prank']:
    chars[features] = chars[features].astype(float)


    def ecdf_transform(s):
        return s.rank(method='average', pct=True)


    for i, f in enumerate(features):
        if (i + 1) % 10 == 0:
            print(f"Feature {i + 1} out of {len(features)}")

        zero = (chars[f] == 0)

        chars[f] = chars.groupby('eom')[f].transform(lambda s: ecdf_transform(s))

        chars.loc[zero, f] = 0

if settings['feat_impute']:
    if settings['feat_prank']:
        for f in features:
            chars[f] = chars[f].fillna(0.5)
    else:
        for f in features:
            chars[f] = chars.groupby('eom')[f].transform(lambda s: s.fillna(s.median()))


chars['sic'] = pd.to_numeric(chars['sic'], errors='coerce')


cond_no_dur = (
    chars['sic'].between(100, 999) |
    chars['sic'].between(2000, 2399) |
    chars['sic'].between(2700, 2749) |
    chars['sic'].between(2770, 2799) |
    chars['sic'].between(3100, 3199) |
    chars['sic'].between(3940, 3989)
)

cond_durbl = (
    chars['sic'].between(2500, 2519) |
    chars['sic'].between(3630, 3659) |
    chars['sic'].isin([3710, 3711, 3714, 3716]) |
    chars['sic'].between(3750, 3751) |
    (chars['sic'] == 3792) |
    chars['sic'].between(3900, 3939) |
    chars['sic'].between(3990, 3999)
)

cond_manuf = (
    chars['sic'].between(2520, 2589) |
    chars['sic'].between(2600, 2699) |
    chars['sic'].between(2750, 2769) |
    chars['sic'].between(3000, 3099) |
    chars['sic'].between(3200, 3569) |
    chars['sic'].between(3580, 3629) |
    chars['sic'].between(3700, 3709) |
    chars['sic'].between(3712, 3713) |
    (chars['sic'] == 3715) |
    chars['sic'].between(3717, 3749) |
    chars['sic'].between(3752, 3791) |
    chars['sic'].between(3793, 3799) |
    chars['sic'].between(3830, 3839) |
    chars['sic'].between(3860, 3899)
)

cond_enrgy = (
    chars['sic'].between(1200, 1399) |
    chars['sic'].between(2900, 2999)
)

cond_chems = (
    chars['sic'].between(2800, 2829) |
    chars['sic'].between(2840, 2899)
)

cond_buseq = (
    chars['sic'].between(3570, 3579) |
    chars['sic'].between(3660, 3692) |
    chars['sic'].between(3694, 3699) |
    chars['sic'].between(3810, 3829) |
    chars['sic'].between(7370, 7379)
)

cond_telcm = chars['sic'].between(4800, 4899)
cond_utils = chars['sic'].between(4900, 4949)

cond_shops = (
    chars['sic'].between(5000, 5999) |
    chars['sic'].between(7200, 7299) |
    chars['sic'].between(7600, 7699)
)

cond_hlth = (
    chars['sic'].between(2830, 2839) |
    (chars['sic'] == 3693) |
    chars['sic'].between(3840, 3859) |
    chars['sic'].between(8000, 8099)
)

cond_money = chars['sic'].between(6000, 6999)

conditions = [
    cond_no_dur,
    cond_durbl,
    cond_manuf,
    cond_enrgy,
    cond_chems,
    cond_buseq,
    cond_telcm,
    cond_utils,
    cond_shops,
    cond_hlth,
    cond_money
]

choices = [
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "Chems",
    "BusEq",
    "Telcm",
    "Utils",
    "Shops",
    "Hlth",
    "Money"
]

chars['ff12'] = np.select(conditions, choices, default="Other")

chars.sort_values(['id', 'eom'], inplace=True)

lb = pf_set['lb_hor'] + 1

eom_lag_series = chars.groupby('id')['eom'].shift(lb)

def calc_month_diff(row):
    if pd.isna(row['eom_lag']):
        return np.nan
    return (row['eom'].year - row['eom_lag'].year) * 12 + (row['eom'].month - row['eom_lag'].month)

temp_df = pd.concat([chars['eom'], eom_lag_series.rename('eom_lag')], axis=1)
temp_df['month_diff'] = temp_df.apply(lambda row: calc_month_diff(row), axis=1)


exclusion_rate = (((temp_df['month_diff'] != lb) | (temp_df['month_diff'].isna())).mean()) * 100
print(f"   Valid lookback observation screen excludes {exclusion_rate:.2f}% of the observations")

valid_data_series = (temp_df['month_diff'] == lb) & (~temp_df['month_diff'].isna())

new_cols = pd.DataFrame({
    'valid_data': valid_data_series,
    'eom_lag': temp_df['eom_lag'],
    'month_diff': temp_df['month_diff']
}, index=chars.index)

chars = pd.concat([chars, new_cols], axis=1)

chars['valid_data'] = chars['valid_data']

chars.drop(columns=['eom_lag', 'month_diff'], inplace=True)


def investment_universe(add, delete):
    """
    Beregner en logisk vektor, der angiver om et aktiv (i en tidsserie)
    skal inkluderes i investeringsuniverset.

    Parametre:
      add (list eller 1D-array af bool): Tilføjningsflag pr. periode.
      delete (list eller 1D-array af bool): Sletningsflag pr. periode.

    Returnerer:
      included (list af bool): Resultat for hver periode.
    """
    n = len(add)
    included = [False] * n
    state = False
    for i in range(1, n):

        if (not state) and add[i] and (not add[i - 1]):
            state = True

        if state and delete[i]:
            state = False
        included[i] = state
    return included



def size_screen_fun(chars, type_screen):
    """
    Size-based screen function. Modifies DataFrame 'chars' in place.

    Parametre:
      chars (pd.DataFrame): DataFrame med mindst kolonnerne 'valid_data', 'me',
                            og evt. 'size_grp'.
      type_screen (str): Angiver, hvilken type screen der skal anvendes. Skal være
                         præcis én af:
                           - "all"
                           - "topN" (f.eks. "top1000")
                           - "bottomN" (f.eks. "bottom100")
                           - "size_grp_{gruppe}" (f.eks. "size_grp_small")
                           - En procent-baseret string indeholdende "perc", "low", "high" og "min"
                             f.eks. "perc_low20high80min50"

    Funktionen tjekker, at præcis én screen-type er anvendt. Hvis ikke, kastes en fejl.
    """
    count = 0
    # --- Screen: All ---
    if type_screen == "all":
        print("No size screen")
        chars.loc[chars['valid_data'] == True, 'valid_size'] = True
        count += 1

    # --- Screen: Top N ---
    if "top" in type_screen:
        top_n = int(re.sub(r"\D", "", type_screen))
        chars.loc[chars['valid_data'] == True, 'me_rank'] = chars.groupby('eom')['me'].rank(ascending=False,
                                                                                            method='first')
        chars['valid_size'] = (chars['me_rank'] <= top_n) & (~chars['me_rank'].isna())
        chars.drop(columns='me_rank', inplace=True)
        count += 1

    # --- Screen: Bottom N ---
    if "bottom" in type_screen:
        bot_n = int(re.sub(r"\D", "", type_screen))
        chars.loc[chars['valid_data'] == True, 'me_rank'] = chars.groupby('eom')['me'].rank(ascending=True,
                                                                                            method='first')
        chars['valid_size'] = (chars['me_rank'] <= bot_n) & (~chars['me_rank'].isna())
        chars.drop(columns='me_rank', inplace=True)
        count += 1

    # --- Screen: Size group ---
    if "size_grp_" in type_screen:
        size_grp_screen = type_screen.replace("size_grp_", "")
        chars['valid_size'] = (chars['size_grp'] == size_grp_screen) & (chars['valid_data'] == True)
        count += 1

    # --- Screen: Percentile-based ---
    if "perc" in type_screen:
        low_match = re.search(r"(?<=low)\d+", type_screen)
        high_match = re.search(r"(?<=high)\d+", type_screen)
        min_match = re.search(r"(?<=min)\d+", type_screen)
        if low_match and high_match and min_match:
            low_p = int(low_match.group())
            high_p = int(high_match.group())
            min_n = int(min_match.group())
        else:
            raise ValueError("Percentile screen format invalid.")
        print(f"Percentile-based screening: Range {low_p}% - {high_p}%, min_n: {min_n} stocks")
        chars.loc[chars['valid_data'] == True, 'me_perc'] = chars.groupby('eom')['me'].transform(
            lambda s: s.rank(method='average', pct=True))
        chars['valid_size'] = (chars['me_perc'] > (low_p / 100)) & (chars['me_perc'] <= (high_p / 100)) & (
            ~chars['me_perc'].isna())
        group_stats = chars.groupby('eom').apply(lambda g: pd.Series({
            'n_tot': g['valid_data'].sum(),
            'n_size': g['valid_size'].sum(),
            'n_less': ((g['valid_data'] == True) & (g['me_perc'] <= (low_p / 100))).sum(),
            'n_more': ((g['valid_data'] == True) & (g['me_perc'] > (high_p / 100))).sum()
        }))
        chars = chars.merge(group_stats, left_on='eom', right_index=True, how='left')
        chars['n_miss'] = np.maximum(min_n - chars['n_size'], 0)
        chars['n_below'] = np.ceil(np.minimum(chars['n_miss'] / 2, chars['n_less']))
        chars['n_above'] = np.ceil(np.minimum(chars['n_miss'] / 2, chars['n_more']))
        cond = (chars['n_below'] + chars['n_above'] < chars['n_miss']) & (chars['n_above'] > chars['n_below'])
        chars.loc[cond, 'n_above'] = chars.loc[cond, 'n_above'] + (
                    chars.loc[cond, 'n_miss'] - chars.loc[cond, 'n_above'] - chars.loc[cond, 'n_below'])
        cond = (chars['n_below'] + chars['n_above'] < chars['n_miss']) & (chars['n_above'] < chars['n_below'])
        chars.loc[cond, 'n_below'] = chars.loc[cond, 'n_below'] + (
                    chars.loc[cond, 'n_miss'] - chars.loc[cond, 'n_above'] - chars.loc[cond, 'n_below'])

        chars['valid_size'] = (chars['me_perc'] > (low_p / 100 - chars['n_below'] / chars['n_tot'])) & \
                              (chars['me_perc'] <= (high_p / 100 + chars['n_above'] / chars['n_tot'])) & \
                              (~chars['me_perc'].isna())
        for col in ['me_perc', 'n_tot', 'n_size', 'n_less', 'n_more', 'n_miss', 'n_below', 'n_above']:
            if col in chars.columns:
                chars.drop(columns=col, inplace=True)
        count += 1

    if count != 1:
        raise ValueError("Invalid size screen applied!!!!")

    return chars


def addition_deletion_fun(chars, addition_n, deletion_n, pf_set):
    """
    Anvender addition/deletion-reglen på DataFrame 'chars'.

    Parametre:
      chars (pd.DataFrame): Datasættet med aktieobservationer.
      addition_n (int): Vindueslængde for addition (rullende sum).
      deletion_n (int): Vindueslængde for deletion (rullende sum).
      pf_set (dict): Indstillinger; herunder forventes pf_set['lb_hor'] at være defineret.

    Funktionen:
      - Opretter en midlertidig kolonne 'valid_temp' (valid_data AND valid_size)
      - Sorterer data efter id og eom.
      - Beregner rullende summer for valid_temp over vinduerne addition_n og deletion_n.
      - Definerer flagget 'add' (når addition_count == addition_n) og 'delete' (når deletion_count == 0).
      - For hver aktie (id) med mere end én observation, anvendes investment_universe til at udlede en kolonne 'valid'.
      - For aktier med kun én observation sættes valid til False.
      - Hvis valid_data er False, sættes valid til False.
      - Beregner turnover (ændring) både for den rå validitet (valid_temp) og den justerede validitet (valid).
      - Udregner og printer gennemsnitlig turnover pr. eom.
      - Fjerner de midlertidige kolonner.
    """
    chars['valid_temp'] = (chars['valid_data'] == True) & (chars['valid_size'] == True)

    chars.sort_values(['id', 'eom'], inplace=True)

    chars['valid_int'] = chars['valid_temp'].astype(int)
    chars['addition_count'] = chars.groupby('id')['valid_int'] \
        .transform(lambda s: s.rolling(window=addition_n, min_periods=addition_n).sum())
    chars['deletion_count'] = chars.groupby('id')['valid_int'] \
        .transform(lambda s: s.rolling(window=deletion_n, min_periods=deletion_n).sum())

    chars['add'] = (chars['addition_count'] == addition_n)
    chars['add'] = chars['add'].fillna(False)
    chars['delete'] = (chars['deletion_count'] == 0)
    chars['delete'] = chars['delete'].fillna(False)

    chars['n'] = chars.groupby('id')['id'].transform('count')

    def apply_investment_universe(df):
        add_list = df['add'].tolist()
        delete_list = df['delete'].tolist()
        valid_list = investment_universe(add_list, delete_list)
        return pd.Series(valid_list, index=df.index)

    chars.loc[chars['n'] > 1, 'valid'] = chars.groupby('id').apply(
        lambda g: apply_investment_universe(g) if len(g) > 1 else False
    ).reset_index(level=0, drop=True)

    chars.loc[chars['n'] == 1, 'valid'] = False
    chars.loc[chars['valid_data'] == False, 'valid'] = False

    chars['chg_raw'] = chars.groupby('id')['valid_temp'].transform(lambda s: s != s.shift(1))
    chars['chg_adj'] = chars.groupby('id')['valid'].transform(lambda s: s != s.shift(1))

    chars['valid_temp_int'] = chars['valid_temp'].astype(int)
    chars['valid_int_new'] = chars['valid'].astype(int)
    chars['chg_raw_int'] = chars['chg_raw'].astype(int)
    chars['chg_adj_int'] = chars['chg_adj'].astype(int)

    turnover = chars.groupby('eom').agg(
        raw_n=pd.NamedAgg(column='valid_temp_int', aggfunc='sum'),
        adj_n=pd.NamedAgg(column='valid_int_new', aggfunc='sum'),
        raw=pd.NamedAgg(column='chg_raw_int', aggfunc='sum'),
        total_valid_temp=pd.NamedAgg(column='valid_temp_int', aggfunc='sum'),
        total_valid=pd.NamedAgg(column='valid_int_new', aggfunc='sum'),
        adj_sum=pd.NamedAgg(column='chg_adj_int', aggfunc='sum')
    ).reset_index()

    turnover['raw'] = turnover.apply(
        lambda row: row['raw'] / row['total_valid_temp'] if row['total_valid_temp'] > 0 else np.nan, axis=1)
    turnover['adj'] = turnover.apply(
        lambda row: row['adj_sum'] / row['total_valid'] if row['total_valid'] > 0 else np.nan, axis=1)

    valid_turnover = turnover[(~turnover['raw'].isna()) & (~turnover['adj'].isna()) & (turnover['adj'] != 0)]

    mean_raw = valid_turnover['raw'].mean()
    mean_adj = valid_turnover['adj'].mean()

    print(f"Turnover wo addition/deletion rule: {mean_raw * 100:.2f}%")
    print(f"Turnover w  addition/deletion rule: {mean_adj * 100:.2f}%")

    cols_to_drop = ['n', 'addition_count', 'deletion_count', 'add', 'delete',
                    'valid_temp', 'valid_data', 'valid_size',
                    'chg_raw', 'chg_adj', 'valid_temp_int', 'valid_int_new',
                    'chg_raw_int', 'chg_adj_int']
    chars.drop(columns=cols_to_drop, inplace=True)

    return chars


type_screen = "all"
chars = size_screen_fun(chars, type_screen)
chars = addition_deletion_fun(chars, addition_n=settings['addition_n'], deletion_n=settings['deletion_n'], pf_set=settings)


# Plot investable universe - udkommenteret
valid_counts = chars.loc[chars['valid'] == True].groupby('eom').size().reset_index(name='N')
#plt.figure(figsize=(10,6))
#sns.scatterplot(data=valid_counts, x='eom', y='N')
#plt.ylabel("Valid stocks")
#plt.axhline(y=0, color='gray', linestyle='--')
#plt.title("Investable Universe")
#plt.show()

valid_pct = (chars['valid'].mean() * 100)
market_cap_valid = (chars.loc[chars['valid'] == True, 'me'].sum() / chars['me'].sum()) * 100
print(f"   The valid_data subset has {valid_pct:.2f}% of the observations and {market_cap_valid:.2f}% of the market cap")
chars.drop(columns="valid_int", inplace=True)


chars.to_parquet("./data_fifty/chars_behandlet.parquet", index=False)


#-------------------------------------------------- daily
daily_file_path = "./data_fifty/usa_dsf_test.parquet"
daily = pd.read_parquet(daily_file_path, engine='pyarrow')
daily = daily[(daily['ret_exc'].notna()) &
              (daily['id'] <= 99999) &
              (daily['id'].isin(chars.loc[chars['valid'] == True, 'id'].unique()))]

daily['eom'] = daily['date'] + MonthEnd(0)
daily.to_csv("./data_fifty/daily.csv", index=False)


daily_path = "./data_fifty/daily.csv"
chars_path = "./data_fifty/chars_behandlet.parquet"
daily = pd.read_csv(daily_path, parse_dates=["date", "eom"])
chars = pd.read_parquet(chars_path, engine='pyarrow')

risk_free = data_run_files.process_risk_free_rate(risk_free_path, start_date)
market_path = "./data_fifty/market_returns_test.csv"
wealth_end = pf_set["wealth"]
end = settings["split"]["test_end"]
market_test = Prepare_Data.load_and_filter_market_returns_test(market_path)
wealth = Prepare_Data.wealth_func(wealth_end, end, market_test, risk_free)
wealth.to_csv("./data_fifty/wealth.csv", index=False)
wealth['eom'] = pd.to_datetime(data_ret_ld1['eom'])



search_grid_single = pd.DataFrame({
    'name': ['m1'],
    'horizon': [1]
})
search_grid = search_grid_single

start_time = time.time()
models = []

for i in range(len(search_grid)):
    # Forbered y-variablen:
    h = search_grid.iloc[i]["horizon"]  # fx 1

    col_name = "ret_ld" + str(h)
    pred_y_values = data_ret[col_name]

    pred_y_df = data_ret[['id', 'eom']].copy()
    pred_y_df['eom_pred_last'] = pred_y_df['eom'] + MonthEnd(1)
    pred_y_df['ret_pred'] = pred_y_values

    valid_chars = chars[chars['valid'] == True]
    data_pred = pd.merge(pred_y_df, valid_chars, on=['id', 'eom'], how='inner')

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
        if train_test_val["test"].empty:
            break
        model_start = time.time()
        model_op = return_prediction_functions.rff_hp_search(
            train_test_val,
            feat=features,
            p_vec=settings['rff']['p_vec'],
            g_vec=settings['rff']['g_vec'],
            l_vec=settings['rff']['l_vec'],
            seed= 1
        )
        model_time = time.time() - model_start
        print("Model training time:", model_time, "seconds")
        op[val_end] = model_op
    inner_time = time.time() - inner_start
    print("Total time for current horizon:", inner_time, "seconds")

    monthly_means_df = (data_pred[['eom', 'mean_ret']]
                         .drop_duplicates()
                         .sort_values('eom')
                         .reset_index(drop=True))
    for key in list(op.keys()):
        if isinstance(key, datetime.datetime):
            model_dict = op[key]
            if 'pred' in model_dict:
                pred_df = model_dict['pred']
                # Merge på 'eom_pred_last' i pred_df og 'eom' i monthly_means_df
                pred_df = pred_df.merge(monthly_means_df, left_on='eom_pred_last', right_on='eom', how='left')
                pred_df = pred_df.drop(columns=['eom_x']).rename(columns={'eom_y': 'eom'}) # ny sørg for kun en eom
                print("pred_df før tilføjelse af pred_df", pred_df)
                # Re-add de-meanede afkast ved at lægge mean_ret til forudsigelsen
                pred_df['pred'] = pred_df['pred'] + pred_df['mean_ret']
                print("pred_df efter tilføjelse af pred_df", pred_df)
                model_dict['pred'] = pred_df
                op[key] = model_dict

    # Gem model-output for den aktuelle horizon til en pickle-fil
    model_filename = f"{output_path}/demeaned_model_{h}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(op, f)

    models.append(op)

total_time = time.time() - start_time
print("Total run time:", total_time, "seconds")




file_path_cluster_labels = "Data/Cluster Labels.csv"
file_path_factor_details = "Data/Factor Details.xlsx"


cluster_data_d, ind_factors, clusters, cluster_data_m  = Estimate_Covariance_Matrix.process_cluster_data(chars, daily, file_path_cluster_labels, file_path_factor_details)

fct_ret_est = Estimate_Covariance_Matrix.run_regressions_by_date(cluster_data_d, ind_factors, clusters)
fct_ret = Estimate_Covariance_Matrix.create_factor_returns(fct_ret_est)

w_cor = (0.5 * (1 / settings['cov_set']['hl_cor'])) * np.arange(settings['cov_set']['obs'], 0, -1)
w_var = (0.5 * (1 / settings['cov_set']['hl_var'])) * np.arange(settings['cov_set']['obs'], 0, -1)

fct_dates = np.sort(fct_ret["date"].unique())
min_calc_date = fct_dates[settings["cov_set"]["obs"] - 1] - pd.DateOffset(months=1)
calc_dates = np.sort(cluster_data_m.loc[cluster_data_m["eom"] >= min_calc_date, "eom"].unique())

factor_cov_est = Estimate_Covariance_Matrix.factor_cov_estimate(calc_dates, fct_dates, fct_ret, w_cor, w_var, settings)

spec_risk = Estimate_Covariance_Matrix.unnest_spec_risk(fct_ret_est)
spec_risk_res_vol = Estimate_Covariance_Matrix.calculate_ewma(spec_risk, settings)

spec_risk_m, spec_risk_res_vol = Estimate_Covariance_Matrix.process_spec_risk_m(fct_dates, spec_risk_res_vol)
barra_cov = Estimate_Covariance_Matrix.calculate_barra_cov(calc_dates, cluster_data_m, spec_risk_m, factor_cov_est)

with open('./data_fifty/barra_cov.pkl', 'wb') as f:
    pickle.dump(barra_cov, f)

#Done print
print("færdig med data_ret, data_ret_ld1")
print("chars er gemt")
print("daily er gemt")
print("wealth er gemt")
print("Predictions er gemt i sin pickle fil")
print("barra_cov er gemt")
print("Færdig")
# Eksempel:
#data_ret = pd.read_csv(output_data_ret_csv)
#data_ret_ld1 = pd.read_csv(output_data_ret_ld1_csv)
#data_ret['eom'] = pd.to_datetime(data_ret['eom'])
#data_ret_ld1['eom'] = pd.to_datetime(data_ret_ld1['eom'])
#data_ret_ld1['eom_ret'] = pd.to_datetime(data_ret_ld1['eom_ret'])