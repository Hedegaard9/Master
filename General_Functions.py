import numpy as np
import pandas as pd
import re
import gc
import xgboost
from pandas.tseries.offsets import MonthEnd

# Read configuration file
def read_config(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    config = {}
    for line in lines:
        key, value = map(str.strip, line.split('=', 1))
        config[key] = eval(value)
    return config

# Create covariance matrix
def create_cov(x, ids=None):
    if ids is None:
        load = x['fct_load']
        ivol = x['ivol_vec']
    else:
        # Konverter ids til strings
        ids = pd.Index(ids).astype(str)
        load = x['fct_load'].loc[ids]
        ivol = x['ivol_vec'].loc[ids]
    return load @ x['fct_cov'] @ load.T + np.diag(ivol)





# Create lambda
def create_lambda(x, ids):
    return np.diag(x[ids])

# Compute expected risk
def expected_risk_fun(ws, dates, cov_list):
    ws = ws.sort_values(['type', 'eom', 'id'])
    types = ws['type'].unique()
    w_list = {key: group for key, group in ws.groupby('eom')}
    result = []
    for d in dates:
        w_sub = w_list.get(d)
        ids = w_sub['id'].unique()
        sigma = create_cov(cov_list[str(d)], ids=ids)
        for t in types:
            w = w_sub[w_sub['type'] == t]['w'].values
            pf_var = w.T @ sigma @ w
            result.append({'type': t, 'pf_var': pf_var, 'eom': d})
    return pd.DataFrame(result)

# Long horizon returns
def long_horizon_ret(data, h, impute):
    # Udtræk unikke datoer og opret 'merge_date' (samme som 'eom')
    dates = data.loc[data['ret_exc'].notna(), ['eom']].drop_duplicates().assign(merge_date=lambda x: x['eom'])
    # Fjern den oprindelige 'eom'-kolonne – behold kun 'merge_date'
    dates['key'] = 1
    dates = dates[['merge_date', 'key']]

    # Beregn for hver id den tidligste (start) og seneste (end) dato med ikke-NA ret_exc
    ids = data.loc[data['ret_exc'].notna()].groupby('id').agg(start=('eom', 'min'),
                                                              end=('eom', 'max')).reset_index()
    ids['key'] = 1

    # Alternativ krydsjoin via nøglekolonne
    full_ret = ids.merge(dates, on='key').drop('key', axis=1)

    # Filtrer, så kun rækker hvor merge_date ligger mellem start og end medtages
    full_ret = full_ret[(full_ret['merge_date'] >= full_ret['start']) &
                        (full_ret['merge_date'] <= full_ret['end'])]

    # Omdøb merge_date til eom, så vi matcher den oprindelige datanøgle
    full_ret = full_ret.rename(columns={'merge_date': 'eom'})

    # Merge med original data for at få ret_exc for hver id og eom
    full_ret = full_ret.merge(data[['id', 'eom', 'ret_exc']], on=['id', 'eom'], how='left')

    # Sorter data
    full_ret = full_ret.sort_values(['id', 'eom'])

    # Opret lead-kolonner: ret_ld1, ret_ld2, ... ret_ldh
    for l in range(1, h + 1):
        full_ret[f'ret_ld{l}'] = full_ret.groupby('id')['ret_exc'].shift(-l)

    # Fjern den originale ret_exc kolonne
    full_ret.drop(columns=['ret_exc'], inplace=True)

    # Fjern rækker, hvor alle ret_ld-kolonner er NaN
    lead_cols = [f'ret_ld{l}' for l in range(1, h + 1)]
    all_missing = full_ret[lead_cols].isna().all(axis=1)
    print(f"All missing excludes {all_missing.mean() * 100:.2f}% of the observations")
    full_ret = full_ret[~all_missing].reset_index(drop=True)

    # Imputér manglende værdier kun i lead-kolonnerne
    if impute == "zero":
        full_ret[lead_cols] = full_ret[lead_cols].fillna(0)
    elif impute == "mean":
        full_ret[lead_cols] = full_ret.groupby('eom')[lead_cols].transform(lambda x: x.fillna(x.mean()))
    elif impute == "median":
        full_ret[lead_cols] = full_ret.groupby('eom')[lead_cols].transform(lambda x: x.fillna(x.median()))

    # Fjern hjælpekolonnerne kolonnerne "start" og "end"
    full_ret = full_ret.drop(columns=['start', 'end'])

    return full_ret

# Portfolio Function
def sigma_gam_adj(sigma_gam, g, cov_type):
    if cov_type == "cov_mult":
        return sigma_gam * g
    elif cov_type == "cov_add":
        return sigma_gam + np.diag(np.diag(sigma_gam) * g)
    elif cov_type == "cor_shrink":
        if abs(g) > 1:
            raise ValueError("g must be between -1 and 1")
        sd_vec = np.sqrt(np.diag(sigma_gam))
        sd_vec_inv = np.linalg.inv(np.diag(sd_vec))
        cor_mat = sd_vec_inv @ sigma_gam @ sd_vec_inv
        cor_mat_adj = cor_mat * (1 - g) + np.eye(len(cor_mat)) * g
        return np.diag(sd_vec) @ cor_mat_adj @ np.diag(sd_vec)


# Initial weights
def initial_weights_new(data, w_type, udf_weights=None):
    """
    Beregner initiale vægte baseret på w_type.

    Parametre:
      - data: Pandas DataFrame med mindst kolonnerne 'id', 'eom' og (afhængigt af w_type) 'me'
      - w_type: str, en af "vw", "ew", "rand_pos" eller "udf"
      - udf_weights: DataFrame med brugerdefinerede vægte (bruges, hvis w_type == "udf")

    Returnerer:
      - pf_w: DataFrame med kolonnerne 'id', 'eom', 'w_start' og 'w'
    """
    # Lav en kopi af data for at undgå sideeffekter
    df = data.copy()

    if w_type == "vw":
        # For hver eom beregnes w_start som me divideret med summen af me i gruppen
        df['w_start'] = df.groupby('eom')['me'].transform(lambda x: x / x.sum())
        pf_w = df[['id', 'eom', 'w_start']].copy()

    elif w_type == "ew":
        # For hver eom sættes w_start = 1 / antal elementer i gruppen
        df['w_start'] = df.groupby('eom')['id'].transform(lambda x: 1 / len(x))
        pf_w = df[['id', 'eom', 'w_start']].copy()

    elif w_type == "rand_pos":
        # Vælg kun de relevante kolonner
        pf_w = df[['id', 'eom']].copy()
        # Generér tilfældige tal for hver række
        pf_w['w_start'] = np.random.uniform(0, 1, size=len(pf_w))
        # Normalisér inden for hver eom, så summen af w_start = 1
        pf_w['w_start'] = pf_w.groupby('eom')['w_start'].transform(lambda x: x / x.sum())

    elif w_type == "udf":
        if udf_weights is None:
            raise ValueError("udf_weights skal angives, når w_type er 'udf'")
        # Merge udf_weights med data på id og eom
        # Her antages det, at udf_weights indeholder kolonnen 'w_start'
        pf_w = pd.merge(data[['id', 'eom']], udf_weights, on=['id', 'eom'], how='left')

    else:
        raise ValueError("Ugyldigt w_type angivet.")

    # For alle rækker, hvor eom ikke er den mindste (tidligste) dato, sættes w_start til NaN
    min_eom = pf_w['eom'].min()
    pf_w.loc[pf_w['eom'] != min_eom, 'w_start'] = np.nan

    # Tilføj kolonnen 'w' og sæt den til NaN for alle rækker
    pf_w['w'] = np.nan

    return pf_w


def pf_ts_fun(weights, data, wealth, gam):
    # Join weights med relevante data-kolonner (id, eom, ret_ld1, pred_ld1, lambda)
    data_subset = data[['id', 'eom', 'ret_ld1', 'pred_ld1', 'lambda']]
    comb = pd.merge(weights, data_subset, on=['id', 'eom'], how='left')

    # Join med wealth for at tilføje 'wealth'-kolonnen (join på eom)
    wealth_subset = wealth[['eom', 'wealth']]
    comb = pd.merge(comb, wealth_subset, on='eom', how='left')

    # Aggregér porteføljestatistikker pr. eom
    def agg_func(g):
        inv = np.sum(np.abs(g['w']))
        shorting = np.sum(np.abs(g.loc[g['w'] < 0, 'w']))
        turnover = np.sum(np.abs(g['w'] - g['w_start']))
        r = np.sum(g['w'] * g['ret_ld1'])
        # For tc antages det, at der er et unikt wealth pr. eom
        unique_wealth = g['wealth'].unique()[0] if len(g['wealth'].unique()) > 0 else np.nan
        tc = (unique_wealth / 2) * np.sum(g['lambda'] * ((g['w'] - g['w_start']) ** 2))
        return pd.Series({
            'inv': inv,
            'shorting': shorting,
            'turnover': turnover,
            'r': r,
            'tc': tc
        })

    pf = comb.groupby('eom').apply(agg_func).reset_index()

    # Udregn eom_ret som eom plus 1 måned (forudsætter at eom er en datetime)
    pf['eom'] = pd.to_datetime(pf['eom'])
    #pf['eom_ret'] = pf['eom'] + pd.DateOffset(months=1)
    pf['eom_ret'] = pf['eom'] + MonthEnd(1)

    # Fjern den oprindelige eom-kolonne, hvis ønsket
    pf = pf.drop(columns=['eom'])

    return pf







def size_screen_fun(chars, type):
    """
    Funktion til at filtrere aktier baseret på størrelse.
    Opdaterer kolonnen 'valid_size' i DataFrame'en chars.
    """

    count = 0  # Sikrer, at kun én screening anvendes

    # All - Ingen filtrering
    if type == "all":
        print("No size screen")
        chars.loc[chars['valid_data'] == True, 'valid_size'] = True
        count += 1

    # Top N - Behold de største N aktier
    if re.match(r"top\d+", type):
        top_n = int(re.sub(r"top", "", type))  # Ekstraher tallet fra strengen
        chars.loc[chars['valid_data'] == True, 'me_rank'] = chars.loc[chars['valid_data'] == True].groupby('eom')['me'].rank(ascending=False, method='dense')
        chars['valid_size'] = (chars['me_rank'] <= top_n) & chars['me_rank'].notna()
        chars.drop(columns=['me_rank'], inplace=True)
        count += 1

    # Bottom N - Behold de mindste N aktier
    if re.match(r"bottom\d+", type):
        bot_n = int(re.sub(r"bottom", "", type))
        chars.loc[chars['valid_data'] == True, 'me_rank'] = chars.loc[chars['valid_data'] == True].groupby('eom')['me'].rank(ascending=True, method='dense')
        chars['valid_size'] = (chars['me_rank'] <= bot_n) & chars['me_rank'].notna()
        chars.drop(columns=['me_rank'], inplace=True)
        count += 1

    # Size group - Filtrér efter bestemt gruppe
    if re.match(r"size_grp_", type):
        size_grp_screen = re.sub(r"size_grp_", "", type)
        chars['valid_size'] = (chars['size_grp'] == size_grp_screen) & (chars['valid_data'] == True)
        count += 1

    # Percentil range - minimum N aktier
    if re.match(r"perc", type):
        low_p = int(re.search(r"low(\d+)", type).group(1))
        high_p = int(re.search(r"high(\d+)", type).group(1))
        min_n = int(re.search(r"min(\d+)", type).group(1))

        print(f"Percentile-based screening: Range {low_p}% - {high_p}%, min_n: {min_n} stocks")

        # Beregn percentiler baseret på 'me' værdien
        chars['me_perc'] = chars.groupby('eom')['me'].transform(lambda x: x.rank(pct=True))

        # Primær filtrering baseret på percentilgrænser
        chars['valid_size'] = (chars['me_perc'] > low_p / 100) & (chars['me_perc'] <= high_p / 100) & chars['me_perc'].notna()

        # Beregn antal aktier der opfylder kravene
        chars['n_tot'] = chars.groupby('eom')['valid_data'].transform(lambda x: x.sum())
        chars['n_size'] = chars.groupby('eom')['valid_size'].transform(lambda x: x.sum())
        chars['n_less'] = chars.groupby('eom').apply(lambda x: (x['valid_data'] & (x['me_perc'] <= low_p / 100)).sum()).reset_index(level=0, drop=True)
        chars['n_more'] = chars.groupby('eom').apply(lambda x: (x['valid_data'] & (x['me_perc'] > high_p / 100)).sum()).reset_index(level=0, drop=True)

        # Beregn hvor mange aktier, der mangler for at opfylde min_n
        chars['n_miss'] = np.maximum(min_n - chars['n_size'], 0)
        chars['n_below'] = np.ceil(np.minimum(chars['n_miss'] / 2, chars['n_less']))
        chars['n_above'] = np.ceil(np.minimum(chars['n_miss'] / 2, chars['n_more']))

        # Juster hvis der stadig mangler aktier
        chars.loc[(chars['n_below'] + chars['n_above'] < chars['n_miss']) & (chars['n_above'] > chars['n_below']), 'n_above'] += (chars['n_miss'] - chars['n_above'] - chars['n_below'])
        chars.loc[(chars['n_below'] + chars['n_above'] < chars['n_miss']) & (chars['n_above'] < chars['n_below']), 'n_below'] += (chars['n_miss'] - chars['n_above'] - chars['n_below'])

        # Opdater valid_size baseret på de nye justeringer
        chars['valid_size'] = (chars['me_perc'] > (low_p / 100 - chars['n_below'] / chars['n_tot'])) & \
                              (chars['me_perc'] <= (high_p / 100 + chars['n_above'] / chars['n_tot'])) & \
                              chars['me_perc'].notna()

        # Fjern midlertidige kolonner
        chars.drop(columns=["me_perc", "n_tot", "n_size", "n_less", "n_more", "n_miss", "n_below", "n_above"], inplace=True)
        count += 1

    # Sikrer, at kun én screening-metode er valgt
    if count != 1:
        raise ValueError("Invalid size screen applied!!!!")

    return chars


def investment_universe(add, delete):
    """
    Hjælpefunktion svarende til R-versionen.
    """
    n = len(add)
    included = [False] * n
    state = False
    # Start fra index 1 (svarende til R's 2:n)
    for i in range(1, n):
        # Inkluder hvis aktien har været valid i 12 måneder
        if not state and add[i] and not add[i - 1]:
            state = True
        # Ekskluder hvis aktien ikke har været valid i de foregående 12 måneder
        if state and delete[i]:
            state = False
        included[i] = state
    return included



def addition_deletion_fun(chars, addition_n, deletion_n):
    """
    Python-version af funktionen addition_deletion_fun med rettelser til groupby.apply.
    """
    # Opret 'valid_temp'
    chars['valid_temp'] = chars['valid_data'] & chars['valid_size']

    # Sorter efter 'id' og 'eom'
    chars.sort_values(by=['id', 'eom'], inplace=True)

    # Beregn rullende summer for additions- og deletionsreglen
    chars['addition_count'] = chars.groupby('id')['valid_temp'] \
        .transform(lambda x: x.rolling(window=addition_n, min_periods=addition_n).sum())
    chars['deletion_count'] = chars.groupby('id')['valid_temp'] \
        .transform(lambda x: x.rolling(window=deletion_n, min_periods=deletion_n).sum())

    # Bestem 'add' og 'delete'
    chars['add'] = (chars['addition_count'] == addition_n)
    chars['add'] = chars['add'].fillna(False)
    chars['delete'] = (chars['deletion_count'] == 0)
    chars['delete'] = chars['delete'].fillna(False)

    # Tæl antal rækker pr. 'id'
    chars['n'] = chars.groupby('id')['id'].transform('count')

    # Beregn 'valid' for hver gruppe
    def compute_valid(group):
        if len(group) > 1:
            return pd.Series(
                investment_universe(group['add'].tolist(), group['delete'].tolist()),
                index=group.index
            )
        else:
            return pd.Series([False] * len(group), index=group.index)

    # Her får vi et MultiIndex (id, original_index)
    valid_series = chars.groupby('id').apply(compute_valid)
    # Fjern det ekstra gruppe-niveau, så indekset stemmer overens med chars.index
    valid_series.index = valid_series.index.droplevel(0)
    chars['valid'] = valid_series

    # Sikr, at hvis valid_data er False, så sættes valid til False
    chars.loc[~chars['valid_data'], 'valid'] = False

    # Beregn ændringer for valid_temp og valid.
    chg_raw = chars.groupby('id')['valid_temp'].apply(lambda x: x.ne(x.shift()))
    chg_raw.index = chg_raw.index.droplevel(0)
    chars['chg_raw'] = chg_raw.fillna(False)

    chg_adj = chars.groupby('id')['valid'].apply(lambda x: x.ne(x.shift()))
    chg_adj.index = chg_adj.index.droplevel(0)
    chars['chg_adj'] = chg_adj.fillna(False)

    # Aggregér turnover-statistikker pr. 'eom'
    agg = chars.groupby('eom').agg(
        raw_n=('valid_temp', 'sum'),
        adj_n=('valid', 'sum'),
        sum_chg_raw=('chg_raw', 'sum'),
        sum_chg_adj=('chg_adj', 'sum')
    ).reset_index()

    agg['raw'] = agg.apply(lambda row: row['sum_chg_raw'] / row['raw_n'] if row['raw_n'] != 0 else np.nan, axis=1)
    agg['adj'] = agg.apply(lambda row: row['sum_chg_adj'] / row['adj_n'] if row['adj_n'] != 0 else np.nan, axis=1)

    # Fjern rækker med NaN eller hvor adj. turnover er 0
    agg = agg[(agg['raw'].notna()) & (agg['adj'].notna()) & (agg['adj'] != 0)]

    n_months = len(agg)
    n_raw = agg['raw_n'].mean() if n_months > 0 else np.nan
    n_adj = agg['adj_n'].mean() if n_months > 0 else np.nan
    turnover_raw = agg['raw'].mean() if n_months > 0 else np.nan
    turnover_adjusted = agg['adj'].mean() if n_months > 0 else np.nan

    print(f"Turnover wo addition/deletion rule: {round(turnover_raw * 100, 2)}%")
    print(f"Turnover w  addition/deletion rule: {round(turnover_adjusted * 100, 2)}%")

    # Fjern de mellemliggende kolonner
    cols_to_drop = ["n", "addition_count", "deletion_count", "add", "delete",
                    "valid_temp", "valid_data", "valid_size", "chg_raw", "chg_adj"]
    chars.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return chars

