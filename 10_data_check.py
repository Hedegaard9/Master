import pandas as pd
import os
import pyarrow
import pyarrow.parquet
# Sti til data-mappen
data_folder = "./Data"

# Find alle Parquet-filer i mappen
parquet_files = [f for f in os.listdir(data_folder) if f.endswith('.parquet')]

# Variabel til at holde styr på det samlede antal kolonner
total_columns = 0

# Liste til at gemme oplysninger om fil, kolonnenavne og første række
data_summary = []

# Iterér over filerne og hent kolonnenavne og første række
for file in parquet_files:
    file_path = os.path.join(data_folder, file)
    try:
        # Læs Parquet-filen med pyarrow
        df = pd.read_parquet(file_path, engine='pyarrow')
        column_count = len(df.columns)  # Antal kolonner i denne fil
        total_columns += column_count  # Opdater totalen

        print(f"Kolonner i filen '{file}':")
        print(df.columns.tolist())  # Udskriver kolonnenavne som en liste
        print(f"Antal kolonner i denne fil: {column_count}")
        print("-" * 50)

        # Tilføj kolonner og første række til data_summary
        for col in df.columns:
            first_value = df[col].iloc[0] if not df.empty else None  # Hent første række, hvis ikke tom
            data_summary.append({"File": file, "Column": col, "First Row Value": first_value})
    except Exception as e:
        print(f"Kunne ikke læse filen '{file}'. Fejl: {e}")

# Konverter data_summary til en DataFrame
summary_df = pd.DataFrame(data_summary)

# Udskriv totalen af kolonner på tværs af alle filer
print(f"Det samlede antal kolonner på tværs af alle filer: {total_columns}")

# Udskriv det samlede DataFrame
print(summary_df)

# Gem DataFrame som CSV, hvis nødvendigt
summary_df.to_csv("column_summary.csv", index=False)
