# Data Cleaning and Processing — US Accidents (2016–2023)

DATA 230 Group Project · Section owner: **Saida Mahmood**
Full report: [`docs/Data_Cleaning_Report.md`](docs/Data_Cleaning_Report.md) ([Word version](docs/Data_Cleaning_Report.docx))

This part of the repository turns the raw Kaggle file into **one shared, documented, leakage-safe dataset** for the team's EDA, dashboard and machine-learning direction. The full file is processed; nothing is sampled.

## Results at a glance

| | Raw file | Cleaned table |
|---|---|---|
| Rows | 7,728,394 | **7,610,699** |
| Columns | 46 | **79** (44 cleaned + 35 created; 33 used by the model) |
| Size on disk | 3.06 GB (CSV) | **0.83 GB** (31 Parquet files) |
| Duplicate rows | 117,695 (102,338 visible as text + 15,357 hidden by a second timestamp format) | 0 |
| Invalid / implausible readings | 282 (3 impossible + 279 beyond U.S. records) | 0 — set to missing, flagged, logged |
| Missing day/night labels / time zones | 23,246 / 7,808 | 0 / 3 (filled from solar position / county time zone) |
| Train / test (chronological, from May 2022) | — | 6,248,916 / 1,361,783 |

## What the cleaning does

| Issue | Treatment |
|---|---|
| **Duplicates** | Rows compared without the unique `ID`, after normalising timestamps, numbers and whitespace; 117,695 proven copies removed. Probable repeats (181,573 rows with identical start/end time and location; reports within ~110 m and 10 minutes) are **flagged, not deleted**, and kept on one side of every train/test split. |
| **Data types** | Every column read as text, then converted explicitly: timestamps with a fixed format (0 parse failures), True/False → 0/1, numbers → float32, repeated text → categories. |
| **Invalid values** | Two tiers, both set to missing with a `*_was_invalid` flag (rows kept, originals in `invalid_values_log.csv`): **impossible** (e.g. pressure = 0) and **implausible** (beyond documented U.S. records, e.g. wind > 150 mph). |
| **Outliers** | IQR outliers are reported, not removed: low pressure comes from high-elevation states, and 19.6% of visibility values differ from the dominant 10 miles. |
| **Missing values** | Kept visible. Weather gaps follow the April 2019 collection change (precipitation missing 87.1% → 3.7%), so missing-value flags are not model inputs. Imputation (State × Month medians) is fitted on training rows only. Day/night is filled from the sun's position (99.77% agreement with the provider's label); time zones from the county. |
| **Labels** | Placeholders ("Unknown") → missing; whitespace and case standardised; wind directions 24 → 18; April 2019 weather vocabulary harmonised (1,193,320 rows); weather grouped into families. |
| **Features** | Time parts, holidays, rush hour, road type (from street name), Census region, weather-observation lag, collection period, duplicate and split columns. |
| **Leakage** | Every column has a role. `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)`, `Duration_min` and `Description` are known only after the accident and are **never model inputs**. Verified: 0 post-event columns and 0 missing values in the 33-column model input. |
| **Splits** | `Split_Time` (chronological hold-out from 1 May 2022, primary) and `Split` (random 80/20, grouped by event). Severity 3 falls from 20.1% to 2.8% over time, so stratified random splits alone would overstate performance. |

## Figures

| Figure | File | Shows |
|---|---|---|
| 1 | `figures/cleaning_fig1_missing_values.png` | Missing values by column (raw data) |
| 2 | `figures/cleaning_fig2_repeated_records.png` | Duplicates removed vs probable repeats flagged |
| 3 | `figures/cleaning_fig3_missing_by_month.png` | Missingness follows the collection period |
| 4 | `figures/cleaning_fig4_pressure_by_state.png` | Outliers are not errors (high elevation) |
| 5 | `figures/cleaning_fig5_severity_by_split.png` | Severity labels drift over time |

All figures follow the course visualization standards (SKILL.md, Tier 1 + Tier 2): units on every axis, legends outside the data, colour-blind palette with hatching, light dashed gridlines, 500 dpi. Their captions are in `figures/figure_captions.md`.

## Files

```
notebooks/DATA230_Cleaning_Local.ipynb    cleaning run on a laptop (Jupyter)
notebooks/DATA230_Cleaning_Colab.ipynb    same code for Google Colab (Google Drive)
src/us_accidents_cleaning.py              cleaning, figure and loading functions
figures/cleaning_fig1..fig5.png           figures + figure_captions.md
data/clean/report_numbers.csv             every number in the report
data/clean/data_dictionary.csv            type, role, missing %, distinct values per column
data/clean/imputation_stats.csv           training-only medians (chronological split)
data/clean/imputation_stats_random.csv    training-only medians (random split)
data/clean/invalid_values_log.csv         every masked value with its original reading
data/clean/cleaning_log.json              counts from every cleaning step
docs/Data_Cleaning_Report.md / .docx      full report
```

The raw CSV, the cleaned Parquet folder and the checkpoints are **not committed** (see `.gitignore`); the notebook rebuilds them.

## How to run

**Requirements:** Python 3.10+, about 4 GB of free memory and 5 GB of disk space.

### On your computer (fastest: about 7 minutes)
1. Download `US_Accidents_March23.csv` from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).
2. Install the packages once:
   ```bash
   pip install pandas numpy matplotlib pyarrow psutil jupyter
   ```
   `pyarrow` is needed for compressed Parquet output (0.83 GB instead of about 2.8 GB).
3. Open `notebooks/DATA230_Cleaning_Local.ipynb`, set `CSV_PATH` in the first code cell to your CSV, and choose **Restart & Run All**.
4. Wait for **"All checks passed."** Results are saved next to the CSV: `data/clean/`, `figures/`, `src/`.

### On Google Colab (about 21 minutes)
1. In Google Drive, create `MyDrive/D230` and upload the CSV or the Kaggle `archive.zip`. If neither is there, the notebook downloads the dataset from Kaggle.
2. Open `notebooks/DATA230_Cleaning_Colab.ipynb` in Colab and choose **Runtime → Run all**; allow Google Drive access.
3. Results are saved to `MyDrive/D230/`.

### Good to know
- **Resume:** each pass saves a checkpoint. If a run stops, **Run all** continues from the last finished pass. Set `RESUME = False` to rebuild everything from the raw CSV.
- **Progress:** every cell prints its run time and memory use; each pass prints a line per 250,000-row block.
- **Memory:** peak use was 3.1 GB (Colab) and 3.5 GB (laptop). With 16 GB of RAM or more, `READ_WHOLE_FILE = True` reads the CSV once; results are identical.

## Using the cleaned data

```python
import us_accidents_cleaning as uc
df = uc.load_clean("data/clean/us_accidents_clean")              # EDA / dashboard (all rows)
counts = uc.events_only(df)                                       # one row per accident, for counts
stats = uc.load_imputation_stats("data/clean/imputation_stats.csv")
X_train, y_train = uc.build_model_frame(df[df.Split_Time == "train"], stats)   # 33 model inputs
X_test,  y_test  = uc.build_model_frame(df[df.Split_Time == "test"],  stats)
```

Rules for everyone: do not refit imputation on all rows; never use `leakage_post_event` columns as features; include `label_process` columns (`Source`, `Start_Year`, `Start_Month`, `Weather_Period`) only if the team decides to (`include_label_process=True`).

## Open decisions for the team

1. Station KJRB reports precipitation of exactly 9.93–10.00 in/h about 370 times (likely an instrument error): mask or keep?
2. Include label-process columns in the model, or leave them out (default)?
3. Use the chronological split as the headline evaluation, with the random split as a secondary check.
