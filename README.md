# Masterprojekt: Implementable Efficient Frontier med Machine Learning

Dette repository indeholder koden og analyserne til mit masterprojekt, som bygger videre på Jensen, Kelly, Malamud og Pedersen (2024) og deres paper:  
[Machine Learning and the Implementable Efficient Frontier](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4187217)

Projektet undersøger og anvender forskellige porteføljevalgmetoder – både klassiske og machine learning-baserede – med fokus på return prediction, transaktionsomkostninger og konstruktionen af den implementerbare effektive rand (IEF).

## 📚 Citation

Hvis du bruger kode eller inspiration fra dette projekt, henvis da også til originalartiklen:

```bibtex
@article{JensenKellyMalamudPedersen2024,
  author = {Jensen, Theis Ingerslev og Kelly, Bryan og Malamud, Semyon og Pedersen, Lasse Heje},
  title = {Machine Learning and the Implementable Efficient Frontier},
  year = {2024}
}
```

## 📁 Projektstruktur - funktioner og data

- 📂 **`much_more_data/`** – Det dataset, der bliver benyttet i vores implementering
- 📂 **`data_fifty/`** – For at bruge mappen skal id filen bruges på de resterende datasæt 
- 📂 **`data_test/`** – For at bruge mappen skal id filen bruges på de resterende datasæt
- 📄 **`Main.py`** – Hovedscript der definerer de valgte settings 
- 📄 **`Prepare_Data.py`** – Funktioner til at klargøre og transformerer rådata
- 📄 **`fit_models.py`** – Træner ML-modeller til return prediction  
- 📄 **`Estimate_Covariance_Matrix.py`** – Estimerer kovariansmatricer (fx Barra)  
- 📄 **`portfolio_choice_functions.py`** – Funktioner til porteføljevalg og optimering  
- 📄 **`requirements.txt`** – Python-pakker som kræves for at køre projektet  
- 📄 **`README.md`** – Denne beskrivelse  
- 📓 **`*.ipynb`** – Jupyter-notebooks til visualisering, test og analyse. Bliver også benyttet til at køre funktioner.

## 📓 Jupyter Notebooks

- 📓 **`Implementable efficient frontier.ipynb`** – Genererer og visualiserer den implementerbare effektive rand (IEF) 
- 📓 **`Base analysis_plots.ipynb`** – Visualiseringer af base case-resultater og performance  
- 📓 **`IEF_1.ipynb`** – Generer IEF for Portfolio-ML, Static-ML m.m, Performance og diverse analyser  
- 📓 **`TC_optimering_Markowitz_ML.ipynb`** – Max-norm løsningen og L2-norm løsningen med Markowitz-ML som modelportefølje (TC-optimering)  
- 📓 **`Cost_opt_markowitz.ipynb`** – Markowitz-optimering med fokus på handelsomkostninger
- 📓 **`Prediction_new.ipynb`** – Alternativ version af forudsigelser med justerede parametre  
- 📓 **`Fit_models_Jonas-Copy1 (4) (1).ipynb`** – Testversion af modeltræning  
- 📓 **`Fit_models_Jonas-Copy1.ipynb`** – Eksperimentel notebook med ML-modeltilpasning  
- 📓 **`Collect_data.ipynb`** – Indsamler og strukturerer finansielle datasæt  
- 📓 **`Create testsets.ipynb`** – Genererer og gemmer testdatasæt til kørsel
- 📓 **`Barra_cov_check.ipynb`** – Skaber en barracov over et valgt dataset
- 📓 **`data_analysis.ipynb`** – Eksplorativ analyse og datatjek i tidligt stadie  
