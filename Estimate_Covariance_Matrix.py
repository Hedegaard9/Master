import pandas as pd
import numpy as np
from Main import settings, features, pf_set
from pandas.tseries.offsets import MonthEnd
from Prepare_Data import process_risk_free_rate, process_return_data, wealth_func, load_and_filter_market_returns_test, process_all_data, process_cluster_labels
from General_Functions import size_screen_fun, addition_deletion_fun



#New version:

def process_cluster_data(chars, daily, cluster_labels_path, factor_details_path):
    """
    Behandler cluster data ved at læse og forberede cluster labels, beregne cluster ranks,
    tilføje industry/market dummies, standardisere faktorer og flette daglige returdata.

    Parametre:
    - chars: DataFrame med karakteristika for aktier
    - daily: DataFrame med daglige afkastdata
    - cluster_labels_path: Sti til CSV-filen med cluster labels
    - factor_details_path: Sti til Excel-filen med factor details

    Returnerer:
    - cluster_data_d: Behandlet DataFrame med alle transformationer

    Eksempel på brug:
    file_path_cluster_labels = "Data/Cluster Labels.csv"
    file_path_factor_details = "Data/Factor Details.xlsx"
    cluster_data_d, ind_factors, clusters = process_cluster_data(chars, daily, file_path_cluster_labels, file_path_factor_details)
    """

    # Indlæs cluster labels separat (bruger samme metode som i den anden version)
    cluster_labels = process_cluster_labels(cluster_labels_path, factor_details_path)

    # Filtrer gyldige karakteristika
    valid_chars = chars[chars["valid"] == True]
    cluster_data_m = valid_chars[["id", "eom", "size_grp", "ff12"] + features].copy()

    # Beregn cluster ranks
    clusters = cluster_labels["cluster"].unique()
    cluster_ranks = {}

    for cl in clusters:
        chars_sub = cluster_labels[(cluster_labels["cluster"] == cl) & (cluster_labels["characteristic"].isin(features))]

        # Undgå potentielle fejl ved at sikre, at kun eksisterende kolonner vælges
        valid_features = [c for c in chars_sub["characteristic"].values if c in cluster_data_m.columns]
        data_sub = cluster_data_m[valid_features].copy()

        for c in valid_features:
            dir_value = chars_sub.loc[chars_sub["characteristic"] == c, "direction"].values[0]
            if dir_value == -1:
                data_sub[c] = 1 - data_sub[c]

        cluster_ranks[cl] = data_sub.mean(axis=1)

    # Konverter cluster ranks til DataFrame
    cluster_ranks_df = pd.DataFrame(cluster_ranks)

    # Kombiner med de oprindelige data
    cluster_data_m = cluster_data_m[["id", "eom", "size_grp", "ff12"]].copy()
    cluster_data_m["eom_ret"] = cluster_data_m["eom"] + pd.DateOffset(months=1)
    cluster_data_m["eom_ret"] = cluster_data_m["eom_ret"] + pd.offsets.MonthEnd(0)  # Sikrer, at det er sidste dag i måneden

    # Merge cluster rankings
    cluster_data_m = pd.concat([cluster_data_m, cluster_ranks_df], axis=1)

    # Tilføj industry/market dummies
    if settings["cov_set"]["industries"]:
        industries = sorted(cluster_data_m["ff12"].unique())
        for ind in industries:
            cluster_data_m[str(ind)] = (cluster_data_m["ff12"] == ind).astype(int)
        ind_factors = industries # vi bruger ik den her?
    else:
        cluster_data_m["mkt"] = 1
        ind_factors = ["mkt"] # vi bruger ik den her?

    # Standardiser faktorer per eom (Bruger samme metode som den anden version)
    cluster_data_m[clusters] = cluster_data_m.groupby("eom")[clusters].transform(lambda x: (x - x.mean()) / x.std())

    # Indlæs daglige returdata
    min_eom = cluster_data_m["eom"].min()
    daily_filtered = daily[daily["date"] >= min_eom][["id", "date", "ret_exc", "eom"]].copy()
    daily_filtered.rename(columns={"eom": "eom_ret"}, inplace=True)

    # Flet daglige data med cluster_data_m
    cluster_data_d = cluster_data_m.merge(daily_filtered, on=["id", "eom_ret"], how="inner")

    # Fjern manglende værdier
    cluster_data_d.dropna(inplace=True)

    return cluster_data_d, ind_factors, clusters

# Create factor returns --------------------

def run_regressions_by_date(cluster_data_d, ind_factors, clusters):
    """
    Kører en lineær regression pr. dato på 'cluster_data_d' og returnerer en DataFrame med resultaterne.

    Parametre:
      - cluster_data_d: DataFrame med mindst kolonnerne 'date' og 'ret_exc', samt de forklarende variable
      - ind_factors: Liste med navne på industry/market faktorer (f.eks. ['BusEq', 'Chems', ...])
      - clusters: Liste med navne på cluster faktorer (f.eks. ['low_leverage', 'investment', ...])

    Returnerer:
      En DataFrame med en række per dato, indeholdende:
        - date: Datoen for gruppen
        - data: Under-DataFrame med de data, der blev brugt for den dato
        - fit: Det tilpassede regressionsmodel-objekt (statsmodels)
        - res: Residualerne fra modellen
        - tidied: En "tidy" tabel med regressionsresultater (koefficienter, standardfejl, t-værdier, p-værdier mv.)
    Example:
    fct_ret_est = run_regressions_by_date(cluster_data_d, ind_factors, clusters)

    """
    # Kombiner de forklarende variable
    all_factors = list(ind_factors) + list(clusters)
    formula = "ret_exc ~ -1 + " + " + ".join(all_factors)

    results = []
    # Gruppér data efter 'date'
    for date, group in cluster_data_d.groupby('date'):
        # Kør regressionen uden intercept (-1)
        model = smf.ols(formula=formula, data=group).fit()
        # Udtræk residualer
        residuals = model.resid
        # Opret en "tidy" oversigt over regressionsresultaterne vha. summary2
        tidy_df = model.summary2().tables[1].reset_index().rename(columns={'index': 'term'})

        results.append({
            'date': date,
            'data': group,
            'fit': model,
            'res': residuals,
            'tidied': tidy_df
        })

    fct_ret_est = pd.DataFrame(results)
    return fct_ret_est


def create_factor_returns(fct_ret_est):
    """
    Omformer fct_ret_est til et datasæt med faktorafkast på samme måde som R-koden:
      fct_ret <- fct_ret_est %>%
                   unnest(tidied) %>%
                   select(date, term, estimate) %>%
                   pivot_wider(names_from = term, values_from = estimate) %>%
                   arrange(date) %>%
                   setDT()

    For hver række i fct_ret_est:
      - "Unnestes" den nestede DataFrame i kolonnen 'tidied'
      - Kolonnen med koefficienter, som i vores tilfælde hedder "Coef.", omdøbes til "estimate"
      - 'date' tilføjes fra den overordnede række
      - Kun kolonnerne 'date', 'term' og 'estimate' bevares
    Herefter pivotteres dataene, så hver unik værdi i 'term' bliver en kolonne, og vi sorterer efter 'date'.

    Parametre:
      - fct_ret_est: DataFrame med mindst kolonnerne 'date' og 'tidied', hvor 'tidied' er en nestet DataFrame

    Returnerer:
      - fct_ret: Pivotteret DataFrame med én række pr. dato og en kolonne pr. term, indeholdende faktorafkast (estimate)

    Example:
    fct_ret = create_factor_returns(fct_ret_est)
    """
    tidied_list = []
    # Gå igennem hver række i fct_ret_est
    for _, row in fct_ret_est.iterrows():
        # Kopiér den nestede DataFrame
        tidy_df = row['tidied'].copy()
        # Hvis kolonnen hedder "Coef.", omdøb den til "estimate"
        if 'Coef.' in tidy_df.columns:
            tidy_df = tidy_df.rename(columns={'Coef.': 'estimate'})
        # Tilføj datoen fra den overordnede række
        tidy_df['date'] = row['date']
        # Vælg kun de ønskede kolonner
        tidied_list.append(tidy_df[['date', 'term', 'estimate']])

    # Sammensæt alle de unnestede DataFrames
    combined = pd.concat(tidied_list, ignore_index=True)

    # Pivotér dataene: 'date' bliver indeks, 'term' bliver kolonnenavne, og cellerne udfyldes med 'estimate'
    fct_ret = combined.pivot(index='date', columns='term', values='estimate').reset_index()

    # Sortér efter 'date'
    fct_ret = fct_ret.sort_values('date').reset_index(drop=True)

    return fct_ret


# Factor Risk -----------------------------

def unnest_spec_risk(fct_ret_est):
    """
    Udpakker de nestede 'id' og 'res'-lister i fct_ret_est.

    Forudsætninger:
      - fct_ret_est er en Pandas DataFrame med kolonnerne:
          - 'date': Datoen for gruppen
          - 'data': En nestet DataFrame, som indeholder en kolonne 'id'
          - 'res': En liste eller kolonne med residualer, der svarer til rækkerne i den nestede 'data'

    Funktionen gør følgende:
      1. Arbejder på en kopi af fct_ret_est.
      2. Ekstraherer 'id'-kolonnen fra hver nestet DataFrame i 'data' og gemmer den som en liste i en ny kolonne 'id'.
      3. Beholder kun kolonnerne 'id', 'date' og 'res'.
      4. Kombinerer de to list-kolonner ('id' og 'res') pr. række til en liste af tuples, så parringen bevares.
      5. "Exploder" (unnester) den kombinerede kolonne, så hver tuple bliver til en enkelt række.
      6. Splitter tuple-kolonnen op i to separate kolonner: 'id' og 'res'.
      7. Sorterer resultaterne efter 'id' og 'date' og returnerer den fladede DataFrame.

    Returnerer:
      En Pandas DataFrame med én række pr. observation, der indeholder kolonnerne 'id', 'date' og 'res'.
    """
    # Arbejd på en kopi af fct_ret_est
    df = fct_ret_est.copy()

    # 1. Udtræk 'id'-kolonnen fra den nestede 'data'-kolonne
    df['id'] = df['data'].apply(lambda nested_df: nested_df['id'].tolist())

    # 2. Vælg kun de relevante kolonner
    spec_risk = df[['id', 'date', 'res']].copy()

    # 3. Kombiner de to list-kolonner (id og res) pr. række til en liste af tuples
    spec_risk['id_res'] = spec_risk.apply(lambda row: list(zip(row['id'], row['res'])), axis=1)

    # 4. Eksploder (unnest) den nye kolonne 'id_res'
    spec_risk = spec_risk.explode('id_res')

    # 5. Split tuple-kolonnen 'id_res' op i to separate kolonner: 'id' og 'res'
    spec_risk[['id', 'res']] = pd.DataFrame(spec_risk['id_res'].tolist(), index=spec_risk.index)

    # 6. Fjern hjælpekolonnen 'id_res'
    spec_risk = spec_risk.drop(columns=['id_res'])

    # 7. Sorter resultaterne efter 'id' og 'date'
    spec_risk = spec_risk.sort_values(['id', 'date']).reset_index(drop=True)

    return spec_risk


def ewma_variance(x, lambda_, start):
    """
    Beregner EWMA-variance (og volatilitet) for en 1-dimensionel array eller pd.Series 'x'.
    Forudsætter, at middelværdien er 0.

    Parametre:
      - x: En 1-dimensionel array eller pd.Series med residualer.
      - lambda_: Vægtningsparameteren (0 < lambda_ < 1). Typisk beregnet som 0.5**(1/hl_stock_var).
      - start: Antallet af observationer, der skal bruges til initial beregning af varians.

    Returnerer:
      - En numpy-array med volatilitet (dvs. kvadratroden af variansen) for hver observation.
    """
    x = np.asarray(x)  # sikre at vi arbejder med en numpy-array
    n = len(x)
    var_vec = np.full(n, np.nan)  # initialiserer en array med NaN

    # Hvis der ikke er nok observationer, returneres en array med NaN
    if n < start:
        return var_vec

    # Beregn initial varians baseret på de første 'start' observationer (med antaget middelværdi 0)
    na_count = 0
    initial_sum = 0.0
    for i in range(start):
        if np.isnan(x[i]):
            na_count += 1
        else:
            initial_sum += x[i] ** 2
    denominator = start - 1 - na_count
    if denominator <= 0:
        initial_value = np.nan
    else:
        initial_value = initial_sum / denominator

    # Sæt den første beregnede varians på position 'start - 1'
    var_vec[start - 1] = initial_value

    # Iterativ opdatering af EWMA-variansen for de efterfølgende observationer
    for j in range(start, n):
        if np.isnan(x[j - 1]):
            var_vec[j] = var_vec[j - 1]
        else:
            var_vec[j] = lambda_ * var_vec[j - 1] + (1 - lambda_) * (x[j - 1] ** 2)

    # Returner volatiliteten (kvadratroden af variansen)
    return np.sqrt(var_vec)