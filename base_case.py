import pandas as pd
import numpy as np
import os
import pickle
from pandas.tseries.offsets import MonthEnd
from Main import settings, features, pf_set
from datetime import datetime
import portfolio_choice_functions

import prepare_portfolio_data
import data_run_files
import Estimate_Covariance_Matrix
import Prepare_Data
import Estimate_Covariance_Matrix as ECM
import Prepare_Data

# indhent barra_cov
barra_cov = ECM.main()
# indhent wealth
wealth = Prepare_Data.main()
wealth = wealth.dropna(subset=['wealth'])

chars, lambda_list, first_cov_date, hp_years, start_oos, date_ranges = prepare_portfolio_data.main(barra_cov)
dates_oos = date_ranges["dates_oos"]
# Benchmark portfolios ------------------------------------------
# Markowitz-ML
tpf = portfolio_choice_functions.tpf_implement(data=chars, cov_list=barra_cov, wealth=wealth, dates=dates_oos, gam=pf_set["gamma_rel"])

# Factor-ML
factor_ml = portfolio_choice_functions.factor_ml_implement(data = chars, wealth = wealth, dates= dates_oos, n_pfs=settings["factor_ml"]["n_pfs"]
 , gam=pf_set['gamma_rel'])

# Market
mkt = portfolio_choice_functions.mkt_implement(data=chars, wealth=wealth, dates=dates_oos, pf_set= pf_set)

# 1/n
ew = portfolio_choice_functions.ew_implement(data=chars, wealth=wealth, dates=dates_oos, pf_set=pf_set)

# Fama-MacBeth / Rank weighted portfolios
rw = portfolio_choice_functions.rw_implement(data=chars, wealth=wealth, dates=dates_oos, pf_set= pf_set)

# Minimum variance
mv = portfolio_choice_functions.mv_implement(data=chars, cov_list=barra_cov ,wealth=wealth, dates=dates_oos, pf_set= pf_set)


# Saml porteføljerne i én DataFrame
#bm_pfs = pd.concat([tpf["pf"], factor_ml["pf"], ew["pf"], mkt["pf"], rw["pf"], mv["pf"]], ignore_index=True)

# Gem resultatet som en CSV-fil
#bm_pfs.to_csv(f"{output_path}/bms.csv", index=False)
