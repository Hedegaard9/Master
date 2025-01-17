import pandas as pd

# Sti til usa_dsf.parquet
file_path = "./Data/usa_dsf.parquet"

# Læs datafilen
try:
    df = pd.read_parquet(file_path, engine='pyarrow')
    print("Fil indlæst med succes.")

    # Få en oversigt over kolonner
    print("\nKolonner i datafilen:")
    print(df.columns.tolist())

    # Få info om dataens størrelse
    print("\nDataens størrelse:")
    print(df.shape)

    # Få en kort oversigt over dataen
    print("\nOversigt over data:")
    print(df.head())

except Exception as e:
    print(f"Kunne ikke læse filen. Fejl: {e}")


print("\nDatatyper for hver kolonne:")
print(df.dtypes)


print("\nAntal manglende værdier pr. kolonne:")
print(df.isnull().sum())


print("\nBasisstatistik for numeriske kolonner:")
print(df.describe())

# Udskift 'kolonnenavn' med den kolonne, du vil undersøge
column_name = 'kolonnenavn'

if column_name in df.columns:
    print(f"\nUnikke værdier i '{column_name}':")
    print(df[column_name].unique())

    print(f"\nAntal værdier i '{column_name}':")
    print(df[column_name].value_counts())
else:
    print(f"Kolonnen '{column_name}' findes ikke i datafilen.")


df['date'] = pd.to_datetime(df['date'])
print(df.dtypes)  # Bekræft ændringen


import matplotlib.pyplot as plt

df['ret_exc'].hist(bins=100)
plt.title("Fordeling af ret_exc")
plt.xlabel("ret_exc")
plt.ylabel("Antal observationer")
plt.show()
