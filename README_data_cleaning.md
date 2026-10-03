# Data cleaning and processing

SJSU DATA 230 group project: U.S. Traffic Accident Data Visualization
**Owner:** Saida Mahmood

This part of the project turns the raw Kaggle file (`US_Accidents_March23.csv`, 7,728,394 rows × 46 columns, 3.06 GB) into one cleaned table that the EDA, the dashboard and the model all read from. The full file is processed in 500,000-row blocks; no sampling. Every step saves a checkpoint so a crashed or disconnected session resumes where it stopped.

For the project overview, see the main [README.md](README.md). For the reasoning behind every decision below, see the full [Data_Cleaning_Report.md](Data_Cleaning_Report.md).

## Files

| Path | Use |
|---|---|
| [notebooks/DATA230_Cleaning_Colab.ipynb](notebooks/DATA230_Cleaning_Colab.ipynb) | The full cleaning run with all outputs (Google Colab) |
| [src/us_accidents_cleaning.py](src/us_accidents_cleaning.py) | All cleaning functions and the loading helpers |
| [Data_Cleaning_Report.md](Data_Cleaning_Report.md) / [.docx](Data_Cleaning_Report.docx) | Full written report |
| [figures/](figures/) | Cleaning figures C1–C7 (500 dpi) |
| [data/clean/data_dictionary.csv](data/clean/data_dictionary.csv) | Type, role, missing % and distinct values of every column |
| [data/clean/imputation_stats.csv](data/clean/imputation_stats.csv), [imputation_stats_random.csv](data/clean/imputation_stats_random.csv) | Training-only medians for each split |
| [data/clean/invalid_values_log.csv](data/clean/invalid_values_log.csv) | Every masked value with its original reading and tier |
| [data/clean/cleaning_log.json](data/clean/cleaning_log.json) | Counts recorded at every cleaning step |
| [data/clean/report_numbers.csv](data/clean/report_numbers.csv) | Every number quoted in the report |
| `data/clean/us_accidents_clean/` | Cleaned table (16 Parquet files, 0.81 GB). **Not committed**; rebuilt by the notebook |

The raw CSV, the Parquet files and the checkpoints are excluded by `.gitignore`.

## What was done

| Issue | Treatment |
|---|---|
| **Duplicate records** | The unique `ID` hides duplicates, so rows are compared without it and after normalising timestamps, numbers and whitespace. **117,695** copies removed (102,338 visible as raw text + 15,357 stored with a second timestamp format). Rows with the same start/end time and location, and reports within ~110 m and 10 minutes, are **flagged, not deleted**, and kept on one side of every train/test split. |
| **Data types** | All columns read as text, then converted explicitly: timestamps parsed with a fixed format (743,166 values carry fractional seconds), booleans to 0/1, numbers to float32, text to categories. |
| **Invalid values** | Two tiers, both set to missing with a `*_was_invalid` flag; rows are kept and every original value is logged: **definition-invalid** (impossible: 3 zero-pressure readings) and **implausible** (beyond documented U.S. records, e.g. wind > 150 mph: 279 readings). |
| **Outliers** | IQR outliers are reported, not removed: e.g. low pressure comes from high-elevation states (Wyoming, Colorado). |
| **Missing values** | Kept visible in the shared table. Weather gaps follow the collection period (precipitation missing 87.1% before April 2019, 3.7% after), so missing-value flags are not used as model inputs by default. Imputation (State × Month medians) is fitted on training rows only. Day/night gaps are filled from the sun's position (22,724 rows; 99.77% agreement with the provider's label); time-zone gaps from the county's time zone (7,703 rows; 3 left missing). |
| **Inconsistent labels** | Placeholder text ("Unknown") → missing; whitespace and letter case standardised; wind directions 24 → 18 labels; weather vocabulary change of April 2019 harmonised (1,193,320 rows); weather grouped into families. |
| **Derived features** | Time parts, holiday, rush hour, road type (from street name), Census region, weather-observation lag, duplicate-group and split columns. |
| **Leakage** | Every column has a role. `End_Time`, `End_Lat/Lng`, `Distance(mi)`, duration and `Description` describe the accident after it happened and are **never model inputs**; `Source` and `Start_Year` describe how Severity was labelled and need a team decision. |

## Result

7,610,699 rows × 79 columns (44 cleaned original columns + 35 derived and audit columns), stored as 16 Parquet files of 0.81 GB in total. Only the 33 columns returned by `feature_columns()` enter the baseline model.

**Splits:** `Split_Time` (chronological hold-out from May 2022, primary) and `Split` (random, grouped by event). Severity shifts strongly over time (Severity 3 is 20.1% of training rows but 2.8% of the hold-out), so stratified random splits alone would overstate model performance.

## Figures

| | |
|---|---|
| ![C1 Missing values before and after](figures/cleaning_C1_missing_before_after.png) | ![C2 Invalid values](figures/cleaning_C2_invalid_values.png) |
| ![C3 Pressure by state](figures/cleaning_C3_pressure_by_state.png) | ![C4 Weather observation lag](figures/cleaning_C4_weather_lag.png) |
| ![C5 Missing values by month](figures/cleaning_C5_missing_by_month.png) | ![C6 Near duplicates](figures/cleaning_C6_near_duplicates.png) |
| ![C7 Duplicate tiers](figures/cleaning_C7_duplicate_tiers.png) | |

## Using the cleaned data

Run from the repository root, after the notebook has built `data/clean/us_accidents_clean/`:

```python
import sys; sys.path.append("src")
import us_accidents_cleaning as uc

df = uc.load_clean("data/clean/us_accidents_clean")              # EDA and dashboard
stats = uc.load_imputation_stats("data/clean/imputation_stats.csv")
X_train, y_train = uc.build_model_frame(df[df.Split_Time == "train"], stats)   # 33 model inputs
X_test,  y_test  = uc.build_model_frame(df[df.Split_Time == "test"],  stats)
```

For accident **counts** on the dashboard, exclude exact repeats (`Same_Event_Repeat`) rather than whole event groups; for rates and models, use all rows.

## Rebuilding the cleaned table (Google Colab)

1. In Google Drive, create the folder `MyDrive/D230` and upload `US_Accidents_March23.csv` (or the Kaggle `archive.zip`). If neither is there, the notebook downloads the dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
2. Open `notebooks/DATA230_Cleaning_Colab.ipynb` in Colab (**File → Upload notebook**).
3. Choose **Runtime → Run all** and allow access to Google Drive.
4. Wait for **"All checks passed."** (about 23 minutes). Progress lines such as `cleaning: block 9 saved` show it is working.
5. If the session disconnects, choose **Runtime → Run all** again; finished steps are skipped.
6. Results are saved in `MyDrive/D230/`: `figures/`, `data/clean/` and `src/`.

## Decisions for the team

1. **KJRB precipitation values (9.93–10.00 in/h, about 370 readings):** almost certainly an instrument error. Mask them, or keep them as they are?
2. **Label-process columns** (`Source`, `Start_Year`, `Start_Month`, `Weather_Period`): leave them out of the baseline model (default) or include them?
3. **Evaluation:** use the chronological split as the headline result, with the random split as a secondary check.
