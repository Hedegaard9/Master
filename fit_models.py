from Main import settings, features, pf_set

import Prepare_Data


####
import pandas as pd

# Version med 12 horisonter
search_grid = pd.DataFrame({
    'name': [f'm{i}' for i in range(1, 13)],
    'horizon': list(range(1, 13))
})

print("Search Grid med 12 horisonter:")
print(search_grid)

# Version med kun én horizon (kun h=1)
search_grid_single = pd.DataFrame({
    'name': ['m1'],
    'horizon': [1]
})

print("\nSearch Grid med kun én horizon:")
print(search_grid_single)

exec(open("Prepare_Data.py").read())
