# Masterprojekt: Implementable Efficient Frontier med Machine Learning

Dette repository indeholder koden og analyserne til vores masterprojekt, som bygger videre på Jensen, Kelly, Malamud og Pedersen (2024) og deres paper:  
[Machine Learning and the Implementable Efficient Frontier](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4187217)

Projektet undersøger og anvender forskellige porteføljevalgmetoder både klassiske og ML-baserede med fokus på return prediction, transaktionsomkostninger og konstruktionen af den implementerbare efficiente Frontier (IEF).

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
- 📂 **`Data/`** – De dataset med rå data før behandling og alle ids.
- 📂 **`much_more_data/`** – De dataset, der bliver benyttet i vores implementering
- 📂 **`data_fifty/`** – For at bruge mappen skal id filen bruges på de resterende rå datasæt 
- 📂 **`data_test/`** – For at bruge mappen skal id filen bruges på de resterende rå datasæt
- 📄 **`Main.py`** – Hovedscript der definerer de valgte settings 
- 📄 **`Prepare_Data.py`** – Funktioner til at klargøre og transformerer rådata
- 📄 **`fit_models.py`** – Træner ML-modeller til return prediction  
- 📄 **`Estimate_Covariance_Matrix.py`** – Estimerer kovariansmatricer (fx Barra) 
- 📄 **`return_prediction_functions.py`** – Funktioner til prediktion (fx Barra) 
- 📄 **`portfolio_choice_functions.py`** – Funktioner til porteføljevalg og optimering  
- 📄 **`requirements.txt`** – Python-pakker som kræves for at køre projektet  
- 📄 **`README.md`** – Denne beskrivelse  
- 📓 **`*.ipynb`** – Jupyter-notebooks til visualisering, test og analyse. Bliver også benyttet til at køre funktioner.

## 📓 Jupyter Notebooks

- 📓 **`Implementable efficient frontier.ipynb`** – Genererer og visualiserer den implementerbare effektive rand (IEF) 
- 📓 **`Bindende_bibetingelse_illustration.ipynb`** – Visualiseringer af bindende bibetingelse
- 📓 **`Base analysis_plots.ipynb`** – Visualiseringer af faktorer i hver klynge 
- 📓 **`IEF_1.ipynb`** – – Genererer IEF for Portfolio-ML, Static-ML m.m, Performance og diverse analyser 
- 📓 **`TC_optimering_Markowitz_ML.ipynb`** – Max-norm løsningen og L2-norm løsningen med Markowitz-ML som modelportefølje (TC-optimering)
- 📓 **`Prediction_new.ipynb`** – Alternativ version af prædiktioner med forskellige slags kørsler  
- 📓 **`new_demean_ucloud.ipynb`** – Mest rene prædiktionsfil brugt i uCloud.  
- 📓 **`Portfolios_factors_overview.ipynb`** – Kørsel af modeller og faktor eskponering. Mange plots af resultater 
- 📓 **`Create testsets.ipynb`** – Genererer og gemmer testdatasæt til kørsel ud fra valgte ids.
- 📓 **`Barra_cov_check.ipynb`** – Skaber en barracov over et valgt dataset
- 📓 **`data_analysis.ipynb`** – Eksplorativ analyse og datatjek i tidligt stadie  
