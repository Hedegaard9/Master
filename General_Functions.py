import numpy as np
import pandas as pd

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
    dates = data.loc[data['ret_exc'].notna(), ['eom']].drop_duplicates().assign(merge_date=lambda x: x['eom'])
    ids = data.loc[data['ret_exc'].notna()].groupby('id').agg(start=('eom', 'min'), end=('eom', 'max')).reset_index()
    full_ret = ids.merge(dates, how='cross')
    full_ret = full_ret.merge(data[['id', 'eom', 'ret_exc']], on=['id', 'eom'], how='left')
    full_ret = full_ret.sort_values(['id', 'eom'])
    for l in range(1, h + 1):
        full_ret[f'ret_ld{l}'] = full_ret.groupby('id')['ret_exc'].shift(-l)
    full_ret.drop(columns=['ret_exc'], inplace=True)

    # Fjern rækker, hvor alle ret_ld værdier er NaN
    all_missing = full_ret.iloc[:, -h:].isna().all(axis=1)
    print(f"All missing excludes {all_missing.mean() * 100:.2f}% of the observations")
    full_ret = full_ret[~all_missing].reset_index(drop=True)

    if impute == "zero":
        full_ret.update(full_ret.iloc[:, -h:].fillna(0))
    elif impute == "mean":
        full_ret.update(full_ret.groupby('eom').transform(lambda x: x.fillna(x.mean())))
    elif impute == "median":
        full_ret.update(full_ret.groupby('eom').transform(lambda x: x.fillna(x.median())))
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
    if w_type == "vw":
        data['w_start'] = data.groupby('eom')['me'].apply(lambda x: x / x.sum())
    elif w_type == "ew":
        data['w_start'] = data.groupby('eom').transform(lambda x: 1 / len(x))
    elif w_type == "rand_pos":
        data['w_start'] = data.groupby('eom').transform(lambda x: np.random.rand(len(x)) / np.random.rand(len(x)).sum())
    elif w_type == "udf":
        data = data.merge(udf_weights, on=['id', 'eom'], how='left')
    data.loc[data['eom'] != data['eom'].min(), 'w_start'] = np.nan
    data['w'] = np.nan
    return data
