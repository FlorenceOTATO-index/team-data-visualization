# U.S. Traffic Accident Data Visualization

SJSU DATA 230 — Data Visualization Group Project

## Team

- **Saida Mahmood** — data cleaning & processing

- **Tzu-Yang Huang** - preliminary ML

- **Nino Pelko** - data exploratory analysis

- **Jue Wang** - data introduction and Tableau dashboard

  
This project explores patterns in the U.S. Accidents dataset through exploratory data analysis (EDA), interactive visualizations, and a preliminary
machine-learning plan. The project focuses on identifying how accident severity varies across time, location, weather, road, and environmental conditions.

## Project question

**What patterns in time, geography, weather, and road conditions are associated with differences in U.S. traffic-accident severity?**

The analysis is intended to help viewers understand where and when accidents occur, which conditions appear alongside more severe accidents, and which variables
may be useful for future severity prediction.

## Dataset

- **Source:** [US Accidents — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

- **Dataset type:** Nationwide traffic-accident records collected from multiple public and real-time data sources

- **Approximate size:** 7.7 million accident records and 46 features in the commonly distributed release

- **Geographic coverage:** United States, across multiple states and cities

- **Target variable:** `Severity`, an ordinal accident-severity score from 1 to 4

The dataset includes accident identifiers, timestamps, geographic coordinates, address information, weather conditions, visibility, precipitation, temperature,
wind, road features, and other contextual variables. The exact row count and available columns may depend on the downloaded version, so the project notebook
reports the values used in this analysis.

## Why this dataset?

Traffic accidents are an important public-safety problem with strong spatial and temporal structure. This dataset was selected because it is large, rich in
features, and suitable for both visual analytics and predictive modeling. It also presents realistic data-science challenges, including high dimensionality,
missing weather observations, categorical variables, skewed numeric variables, class imbalance, and millions of records that require efficient processing.

## Project workflow

1. Inspect the dataset structure, data types, missingness, duplicates, and class distribution.

2. Clean and prepare variables for visualization.

3. Perform univariate, bivariate, and multivariate EDA.

4. Build at least six meaningful visualizations connected to accident severity.

5. Interpret the visual evidence and document analysis decisions.

6. Present a preliminary machine-learning direction for future work.

7. Share a dashboard that communicates the most important findings.

---

## Setup (do this once)

You need: Git, Python 3.10+.

```
# 1. Get the code
git clone https://github.com/FlorenceOTATO-index/team-data-visualization.git
cd team-data-visualization

# 2. Make a Python environment and install packages
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn plotly pyarrow psutil jupyter
```

`pyarrow` is needed for the compressed Parquet output of the cleaning pipeline.

```
# 3. Get the data (3.06 GB — not in git)
python scripts/download_data.py    # → data/raw/US_Accidents_March23.csv
```

The download needs a Kaggle account. Alternatively download
`US_Accidents_March23.csv` by hand from the Kaggle page above and place it
in `data/raw/`.

```
# 4. Run the cleaning pipeline (about 7 minutes, ~3.5 GB RAM)
# Open notebooks/00_sm_cleaning_local.ipynb, set CSV_PATH in the first
# code cell to your CSV, and choose Restart & Run All.
# Wait for "All checks passed."
```

## Data rules (please read)

- **`data/raw/` is never committed.** The Kaggle CSV is 3 GB. Everyone
  downloads it with the script above.
- **`data/processed/` is committed**, so we all share the same small
  derived artifacts (`data_dictionary.csv`, `cleaning_log.json`,
  `imputation_stats*.csv`, …). Never edit these by hand — the notebook
  rebuilds them.
- **Notebooks are one person each** (`NN_initials_topic.ipynb`). They merge
  badly — never two people editing the same one.
- Every notebook starts from the cleaned data, never the raw CSV.
- Put shared code in `src/` instead of copy-pasting cells.

## How we work with Git

**Short version: never push directly to `main`.** Make a branch, push it,
and open a Pull Request (PR) so a teammate can look at it before it's merged.

```
git checkout main
git pull                                   # get everyone's latest work
git checkout -b yourname/short-description # e.g. jue/eda-severity-time

# ... work ...
git add notebooks/02_jw_severity_by_hour.ipynb
git commit -m "Add severity-by-hour EDA"

git push -u origin yourname/short-description
```

Then open GitHub, click **Compare & pull request**, write a sentence or two,
and request a teammate as reviewer. After approval: **Squash and merge**,
delete the branch, everyone else runs `git checkout main && git pull`.

---

## Data cleaning and processing

The analysis addresses the following issues:

- **Missing values:** Missingness is measured by column and handled according to each variable’s role. Variables with substantial missingness are either
excluded from a specific visualization or clearly labeled as unavailable.

- **Duplicate records:** Exact duplicates are checked before analysis to reduce the risk of counting the same accident more than once.

- **Data types:** Date/time fields are converted to datetime values. Numeric fields are coerced to numeric types when necessary, and categorical fields are
standardized for grouping.

- **Derived features:** Time-based features such as year, month, day of week, and hour are created from the start timestamp. Accident duration may be derived
from start and end times when both values are valid.

- **Skewed variables:** Highly skewed variables, such as distance, are examined with appropriate transformations or visual scales when useful.

- **Outliers:** Extreme values are investigated using summary statistics and plots. Values are not removed automatically unless they are invalid, outside the
documented meaning of the field, or clearly caused by a data-entry issue.

- **Severity imbalance:** Severity is treated as an ordinal target. Because the classes are unevenly distributed, counts and percentages are shown together and
future modeling should use stratification and imbalance-aware evaluation metrics.

### Challenges encountered

- The dataset is large, so loading and plotting all records can be computationally expensive.

- Many weather and environmental fields contain missing values.

- Severity classes are imbalanced, making raw counts easy to misinterpret.

- Several categorical variables have many distinct levels and require grouping or filtering.

- Accident duration and other derived variables depend on the quality and consistency of timestamp fields.

- Geographic and weather patterns may be difficult to compare fairly when exposure or record coverage differs across locations.

Full details: [`docs/cleaning_log.md`](docs/cleaning_log.md) and the complete
[`docs/cleaning_report.md`](docs/cleaning_report.md).

## Exploratory data analysis

The project includes at least six visualizations selected to answer the project question. Each visualization is interpreted in terms of what it shows, what
decision it supports, and what limitation should be considered.

Current visualization topics include:

1. **Severity distribution:** Shows the number and percentage of records in each severity class and documents class imbalance.

2. **Accidents over time:** Examines changes in accident frequency across years, months, days of the week, or hours.

3. **Geographic distribution:** Compares accident patterns across states, cities, or mapped locations.

4. **Weather conditions and severity:** Compares severity across weather categories and identifies categories requiring careful grouping.

5. **Visibility, precipitation, or temperature and severity:** Examines relationships between environmental measurements and severity.

6. **Duration or distance and severity:** Uses derived or transformed numeric variables to compare distributions across severity levels.

7. **Multivariate view:** Combines two or more environmental or temporal variables with severity to reveal interactions that are not visible in a
single-variable plot.

> Replace the topics above with links to the final notebook, exported figures, or dashboard tabs once the project artifacts are finalized.

### Main EDA decisions and preliminary insights

- Severity should be analyzed as an ordinal outcome rather than treated as an ordinary continuous measurement.

- Percentages and severity-stratified comparisons are more informative than accident counts alone because the target classes are imbalanced.

- Time features should be retained because accident records show temporal structure that can support both visualization and modeling.

- Weather, visibility, precipitation, and road-context variables should be compared jointly where possible because individual relationships may be confounded by
other conditions.

- Geographic comparisons should report the number of records and use appropriate normalization when making rate-like comparisons.

- Highly skewed numeric variables should be visualized with transformations or robust summaries so that the central pattern is not hidden by extreme values.

## Dashboard

The dashboard is designed to communicate the project’s most important findings without requiring the viewer to inspect code. It should allow users to explore
accident severity by relevant dimensions such as time, location, weather, and road conditions.

- **Dashboard link:** `[TBD]`

- **Dashboard file:** `[TBD]` → place in `dashboards/`

- **Dashboard tool:** `[Tableau]`

Recommended dashboard elements include:

- A severity overview showing counts and percentages

- A time trend with filters for severity and location

- A geographic view or state-level comparison

- A weather or environmental comparison

- A clear explanation of the main findings and limitations

## Preliminary machine-learning direction

The proposed future task is **multiclass ordinal classification**, using the accident `Severity` score as the target.

Candidate predictors include timestamp-derived features, geographic variables, weather conditions, visibility, precipitation, temperature, wind, road features,
and accident duration or distance where valid. The EDA supports this direction because severity is categorical and imbalanced, while the dataset contains
multiple contextual variables that may help distinguish severity levels.

Future modeling should consider:

- Stratified train/validation/test splits

- Class weighting or resampling methods

- Baseline models such as logistic regression, decision trees, and random forests

- Macro-F1, balanced accuracy, confusion matrices, and per-class recall

- Leakage checks for identifiers, post-accident fields, and variables unavailable at prediction time

- Ordinal-aware evaluation because confusing neighboring severity levels may be less serious than confusing the lowest and highest levels

No machine-learning results are required for the mid-presentation; this section documents the proposed direction and its motivation from EDA.

---

## Repository structure

```
team-data-visualization/
├── data/
│   ├── raw/              ← Kaggle CSV (NOT in git; see Setup)
│   └── processed/        ← committed cleaning artifacts (data_dictionary.csv,
│                           cleaning_log.json, imputation_stats*.csv, …)
├── notebooks/            ← 00_sm_cleaning_local.ipynb, 01_sm_cleaning_colab.ipynb,
│                           EDA notebooks (NN_initials_topic.ipynb)
├── src/
│   ├── cleaning/         ← us_accidents_cleaning.py (cleaning/figure/loading fns)
│   ├── eda/              ← shared plotting helpers
│   └── dashboard/        ← dashboard app code
├── dashboards/           ← exported dashboard file or shareable link doc
├── docs/
│   ├── cleaning_log.md   ← cleaning results + run guide
│   ├── cleaning_report.md← full cleaning report
│   ├── dataset.md        ← dataset intro [TODO]
│   ├── eda.md            ← EDA decisions & conclusions [TODO]
│   ├── ml_direction.md   ← preliminary ML direction [TODO]
│   └── contributions.md  ← contribution evidence [TODO]
├── reports/
│   ├── figures/          ← committed figures + figure_captions.md
│   └── mid-presentation/ ← slides [TODO]
└── scripts/              ← download_data.py
```

| Path | Description |
|---|---|
| `README.md` | Project overview, methodology, findings, and contribution record |
| `notebooks/` | Cleaning and EDA notebooks |
| `src/` | Reusable cleaning, plotting, and dashboard code |
| `dashboards/` | Exported dashboard file or shareable link |
| `docs/` | Dataset, cleaning log/report, EDA decisions, ML direction, contributions |
| `reports/figures/` | Exported visualizations used in the presentation and dashboard |
| `reports/mid-presentation/` | Slides |
| `data/` | Local data instructions; small derived artifacts in `data/processed/` (never commit the raw Kaggle CSV) |
| `scripts/` | One-off utilities (dataset download) |

## Mid-presentation checklist (due Wed Oct 7, 11:59pm — 10 min + 4 min Q&A)

| Requirement | Status | Lives in |
|---|---|---|
| Dataset intro & motivation | Drafted (above) → extract to `docs/dataset.md` | `docs/dataset.md` |
| Data cleaning (missing values, outliers, transforms, challenges) | Done | `docs/cleaning_log.md` |
| EDA: ≥ 6 meaningful visualizations tied to the project question, with decisions/conclusions | In progress | `notebooks/` → `docs/eda.md` |
| Preliminary ML direction (task + justification, no results) | Drafted (above) → extract to `docs/ml_direction.md` | `docs/ml_direction.md` |
| One dashboard, built by one member | TBD (Tableau) | `dashboards/` |
| Evidence of each member's contribution (required last slide) | TBD | `docs/contributions.md` |

Submit the repo link plus the dashboard link/file.

---

## Reproducibility

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
   (`python scripts/download_data.py`, or by hand into `data/raw/`).

2. Install the dependencies (see Setup).

3. Run the cleaning notebook (`notebooks/00_sm_cleaning_local.ipynb`) from
   start to finish — wait for "All checks passed."

4. Run the EDA notebooks in numeric order.

5. Open the dashboard using the published link or the file in `dashboards/`.

Because the full dataset is large and subject to Kaggle access requirements, it is not stored in this repository.

## Conventions

| Thing | Convention | Example |
|---|---|---|
| Branches | `name/what-youre-doing` | `jue/eda-severity-time` |
| Notebooks | number, initials, topic | `notebooks/02_jw_severity_by_hour.ipynb` |
| Figures | descriptive, committed | `reports/figures/severity_by_hour.png` |
| Code style | `snake_case` names | `severity_by_state` |

### Contribution evidence

The final presentation must include evidence of each member’s contribution. Add specific responsibilities and links to commits, notebooks, dashboard work, or
presentation sections below.

| Member | Contribution |
|---|---|
| Saida Mahmood | Data cleaning pipeline: `src/cleaning/us_accidents_cleaning.py`, `notebooks/00_sm_cleaning_local.ipynb`, `docs/cleaning_report.md` |
| Tzu-Yang Huang | `[Add contribution]` |
| Nino Pelko | `[Add contribution]` |
| Jue Wang | `[Add contribution]` |

## Course

**SJSU DATA 230 — Data Visualization, Fall 2026**

## License and data attribution

This repository contains student analysis code, visualizations, and documentation. The U.S. Accidents dataset remains subject to the terms and attribution
requirements of its [Kaggle source](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) (CC BY-NC-SA 4.0). If you use this dataset, please cite:
Moosavi et al., "A Countrywide Traffic Accident Dataset", 2019; and Moosavi et al., "Accident Risk Prediction based on Heterogeneous Sparse Data", ACM SIGSPATIAL 2019.
