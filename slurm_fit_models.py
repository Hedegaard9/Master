import sys
import os
import shutil
import pickle
import time
from datetime import datetime
import pandas as pd
from Main import settings, features, pf_set

# --- Kør "Main.py" (svarer til source("Main.R") i R) ---


# --- Opdater settings ---
settings['screens']['size_screen'] = "all"
settings['screens']['nyse_stocks'] = False  # Inkludér også ikke-NYSE aktier
run_sub = False  # True ved test, False ellers

# --- Range of Prediction Horizons ---
# Opret en DataFrame svarende til tibble i R
search_grid = pd.DataFrame({
    'name': [f"m{i}" for i in range(1, 13)],
    'horizon': list(range(1, 13))
})

# --- Læs konfiguration ---
# Forbered til at modtage konfigurationsfilen via kommandolinjen (f.eks. fra slurm)
if True:
    args = sys.argv[1:]
    config_file = args[0] if len(args) > 0 else None
    if config_file is not None:
        # Forudsætter, at read_config er defineret et sted og returnerer en dictionary
        config_params = {'horizon': [1]} # read_config(config_file)
    else:
        config_params = {'horizon': [1]}
else:
    config_params = {'horizon': [1]}

# Sørg for, at config_params['horizon'] er en liste
if not isinstance(config_params['horizon'], list):
    config_params['horizon'] = [config_params['horizon']]

# Filtrer search_grid, så kun de relevante horisonter beholdes
names_to_include = [f"m{x}" for x in config_params['horizon']]
search_grid = search_grid[search_grid['name'].isin(names_to_include)]

# --- Opret output-mappe, hvis den ikke findes ---
output_path = os.path.join("Data", "Generated", "Models",
                           f"{datetime.now().strftime('%Y%m%d')}-{settings['screens']['size_screen']}")
if not os.path.exists(output_path):
    os.makedirs(output_path)
    # Kopiér eventuelt relevante konfigurationsfiler til output mappen
    if os.path.exists("Main.py"):
        shutil.copy("Main.py", os.path.join(output_path, "Main.py"))
    if os.path.exists("slurm_fit_models.py"):
        shutil.copy("slurm_fit_models.py", os.path.join(output_path, "slurm_fit_models.py"))
    # Gem settings som pickle (i stedet for RDS)
    with open(os.path.join(output_path, "settings.pkl"), "wb") as f:
        pickle.dump(settings, f)

# --- Start total tidstagning ---
tic = time.time()

# --- Kør "1 - Prepare Data.py" ---
# Her antages det, at du har en Python-version af scriptet.
with open("Prepare_Data.py", encoding="utf-8") as f:
    exec(f.read())

# --- Kør "2 - Fit Models.py" ---
with open("fit_models.py", encoding="utf-8") as f:
    exec(f.read())

print(f"Total run time: {time.time() - tic:.2f} sekunder")