# US Accidents (2016–2023): Data Cleaning and Processing

**DATA 230 Group Project — Data Cleaning and Processing section**
Saida Mahmood · San José State University · Fall 2026
Project repository: <https://github.com/FlorenceOTATO-index/team-data-visualization>

---

## 1. Purpose and scope

This section prepares the U.S. Accidents dataset for the project's exploratory analysis, dashboard and preliminary machine-learning direction. The project asks which patterns in time, geography, weather and road conditions are associated with differences in accident severity. The cleaning therefore aims at one shared, documented dataset in which:

- every accident is counted once;
- every value is either valid or missing with a recorded reason;
- nothing known only *after* an accident can enter a severity model;
- every step can be reproduced from code.

All work uses the **full file of 7,728,394 rows**; no sampling was done at any stage.

## 2. The dataset

| Item | Value |
|---|---|
| Source | US Accidents (2016–2023), Kaggle, March 2023 release (`US_Accidents_March23.csv`, 3.06 GB) |
| Size | 7,728,394 rows × 46 columns |
| Target | `Severity`, 1–4 (impact on traffic) |
| Coverage | Contiguous United States (48 states and DC), January 2016 – March 2023 |
| Providers | Three data sources (`Source1`–`Source3`) |

The columns cover an identifier, the target, the provider, start and end times, start and end coordinates, distance, a free-text description, address fields, a weather-station code and observation time, eight weather measurements, a weather description, thirteen True/False road features, and four day/night indicators.

Severity is strongly imbalanced: Severity 1 = 67,366 (0.87%), Severity 2 = 6,156,981 (79.67%), Severity 3 = 1,299,337 (16.81%), Severity 4 = 204,710 (2.65%). The largest class is 91 times the smallest.

## 3. Data-quality problems found

A full-file audit read every column as text, in blocks of 250,000 rows, and counted each problem before any change was made. Most problems raise no error in pandas and would distort the analysis silently.

| Problem | Evidence (raw file) |
|---|---|
| Missing values | 22 of 46 columns; largest: `End_Lat`/`End_Lng` 44.03%, `Precipitation(in)` 28.51%, `Wind_Chill(F)` 25.87%, `Wind_Speed(mph)` 7.39% (Figure 1) |
| Hidden duplicates | `ID` is unique, so a normal check finds none. Without `ID`, 102,338 rows are copies, plus 15,357 that differ only in timestamp format: 117,695 in total (Figure 2) |
| Two timestamp formats | 743,166 start times end in `.000000000`; date inference would turn them into missing values |
| Non-random missingness | Precipitation is missing for 87.1% of accidents in March 2019 and 3.7% in April 2019, for every provider (Figure 3) |
| Invalid readings that look like numbers | e.g. temperature 207 °F, wind speed 1,087 mph, pressure 0 inHg |
| Implausible end times | 4,229 accidents "last" more than one year; 93,903 weather readings were taken more than one hour from the accident |
| Inconsistent text | 1,696,520 `Street` values with leading or trailing spaces; "Unknown" used as a description 34 times; city and county names written in several letter cases |
| Inconsistent categories | `Wind_Direction` uses 24 labels for 18 directions; in April 2019, three weather labels were replaced by synonyms (Clear → Fair, Overcast → Cloudy, Thunderstorm → T-Storm; 1,193,320 rows); "N/A Precipitation" used 3,252 times |
| Mixed code formats | `Zipcode` mixes 5-digit, ZIP+4 and other formats |
| Constant columns | `Country` (always US), `Turning_Loop` (always False) |
| Leakage risk | `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)` and `Description` describe the accident after it happened |
| Labels changing over time | Severity 3 is 20.1% of accidents before May 2022 but 2.8% afterwards (Figure 5) |

![Figure 1](../figures/cleaning_fig1_missing_values.png)

**Figure 1. Missing values by column in the raw data.** Share of the 7,728,394 raw rows with no value in each column; columns without gaps are not shown. 22 of 46 columns have gaps. The largest are the end coordinates (44.0%, recorded after the accident and never used as model inputs), precipitation (28.5%) and wind chill (25.9%); all other weather fields are below 7%. Missing values were kept visible in the cleaned table and imputed only at modelling time, from training rows, because their pattern follows the data-collection period (Figure 3). *Data: US Accidents (Kaggle, March 2023 release), full file.*

## 4. Cleaning methodology

The pipeline (`src/us_accidents_cleaning.py`, run from `notebooks/DATA230_Cleaning_Local.ipynb` or `DATA230_Cleaning_Colab.ipynb`) works in three passes:

1. **Audit (pass 1).** Count every problem above and compute a fingerprint of each row: all columns except `ID`, after normalising timestamps, numbers and whitespace.
2. **Row-level cleaning (pass 2).** Remove proven duplicates, validate the target, parse dates, set invalid readings to missing, convert types, standardise labels and create features. Write the result as compressed Parquet blocks.
3. **Global steps (pass 3).** Group repeated reports of the same accident, assign two train/test splits, and fit imputation values on training rows only.

Steps that need no information from other rows run block by block, so they cannot leak information between rows. Steps that need the whole dataset use only a few key columns. Each pass saves a checkpoint, so an interrupted run resumes where it stopped.

Design principles:

- **Mask values, never delete accidents:** an invalid reading becomes missing and gets a flag; the accident row stays.
- **Keep missingness visible:** the shared table is not imputed.
- **Remove only what is proven:** exact copies are removed; probable repeats are flagged.
- **Give every column a role** (`feature`, `label_process`, `leakage_post_event`, `eda_only`, `quality_flag`, `split`, `target`, `identifier`), so the model can only use information available when an accident is first reported.
- **Name every threshold** as a constant in the code, and log every masked value with its original reading.

## 5. Cleaning steps and rationale

### 5.1 Duplicate records

Rows were compared without `ID`. A raw-text comparison finds 102,338 copies. Comparing what each row *means* — timestamps cut to one format, numbers parsed, whitespace trimmed — finds 15,357 more: the same record stored once as `05:46:00` and once as `05:46:00.000000000`. **117,695 rows (1.52%) were removed**, keeping the first copy.

Two looser kinds of repetition were **flagged, not removed**, because they are not proven duplicates:

- **181,573 rows** share the exact start time, end time and location with another row but differ in other fields (`Same_Event_Repeat`), for example an updated report of the same incident.
- **1,045,620 rows** fall into 435,150 groups of reports within about 110 m and 10 minutes of each other (`Event_Key`); 8,566 of them involve two different providers.

Every report of a group is placed on the same side of each train/test split, so a model is never tested on a report of an accident it was trained on. Groups whose reports carry different Severity values (147,120 rows) are potential conflicts requiring investigation: provider disagreement, separate nearby accidents grouped by the heuristic, repeated reports, or inconsistent labels.

![Figure 2](../figures/cleaning_fig2_repeated_records.png)

**Figure 2. Four kinds of repeated records, from strictest to loosest definition.** Proven copies (top two bars, 117,695 rows) were removed, keeping the first copy; the second bar only appears when the same instant written as '05:46:00' and '05:46:00.000000000' is treated as equal. The lower bars count every row of a group, including its first report; these probable repeats were kept, flagged, and assigned to one side of every train/test split so that no accident is in both training and test data. *Data: US Accidents (Kaggle, March 2023 release), full file.*

### 5.2 Target and timestamps

`Severity` is 1–4 in every row (0 invalid values) and is never imputed or changed. Timestamps are parsed from their first 19 characters with an explicit format, with a strict ISO-8601 retry for anything else; **0 start times failed to parse**. Durations that are negative or longer than one year are set to missing and flagged (3,636 rows after duplicate removal). Seven accidents start before the documented February 2016 coverage; they are kept and flagged.

### 5.3 Invalid and implausible readings

Readings are checked in two tiers. Both set the value to missing, add a `*_was_invalid` flag, keep the accident row, and write the original value to `invalid_values_log.csv`:

- **Definition-invalid (impossible)** — violates the definition of the quantity: pressure ≤ 0 inHg, humidity outside 0–100%, negative amounts, temperatures below absolute zero, coordinates outside the contiguous United States. **3 readings** (pressure = 0).
- **Implausible** — beyond documented U.S. records: temperature outside −70 to 134 °F, wind above 150 mph, precipitation above 12 in/h, pressure outside 15–35 inHg, visibility above 100 mi, wind chill above the air temperature. **279 readings.**

In total **282 values** were masked (pressure 116, temperature 63, wind speed 51, wind chill 31, visibility 14, precipitation 7), out of about 53 million weather values.

### 5.4 Outliers

Values that are valid but far from the bulk were **reported, not removed**. A standard Tukey rule (1.5 × IQR) would flag 19.6% of visibility readings (every value other than 10 miles), 12.5% of distances, 9.7% of precipitation readings (every non-zero amount) and 5.8% of pressure readings, which are mostly genuine values. Figure 4 shows why for pressure: the low readings come from high-elevation states (median 23.4 inHg in Wyoming, 24.5 in Colorado, 24.9 in New Mexico).

![Figure 4](../figures/cleaning_fig4_pressure_by_state.png)

**Figure 4. Low air-pressure readings come from high-elevation states, not from errors.** Station air pressure in the 5 states with the lowest and the 5 with the highest median (boxes: interquartile range; whiskers: 1.5 × IQR; line: median; outliers not drawn; n under each state). A dataset-wide IQR rule would flag 5.8% of all readings below the dashed fence, yet the low-median states (WY 23.4, CO 24.5, NM 24.9 inHg) are high-elevation states where low station pressure is physically normal. Statistical outliers were therefore reported, not removed. *Data: US Accidents (Kaggle, March 2023 release), full file.*

### 5.5 Missing values

Missing values stay visible in the shared table. How they are handled depends on the column's role:

| Column(s) | Missing (raw) | Treatment |
|---|---|---|
| `End_Lat`, `End_Lng` | 44.03% | Kept for EDA; never a model input (post-event) |
| `Precipitation`, `Wind_Speed`, `Visibility`, `Humidity`, `Temperature`, `Pressure` | 1.8–28.5% | Imputed at modelling time with State × Month medians (then State, then overall), fitted on training rows only |
| `Wind_Chill(F)` | 25.87% | Kept for EDA; not a model input (correlation 0.994 with temperature) |
| `Wind_Direction`, `Weather_Condition` | 2.2–2.3% | Explicit `Unknown` category at modelling time |
| `Sunrise_Sunset` | 0.30% | Filled from the sun's position at the accident's time and place (22,724 rows); the calculation agrees with the provider's own label for 99.77% of accidents |
| `Timezone` | 0.10% | Filled from the county's time zone, or from the state when the whole state has one zone (7,703 rows, flagged); 3 rows stay missing |

The weather gaps are not random (Figure 3). Precipitation is missing for 86–89% of accidents before April 2019 and 4–7% afterwards, at a similar level for all three providers within each period. A "was missing" indicator would therefore tell a model *when* an accident was recorded, not anything about the road, so missing-value flags are not used as model inputs by default.

![Figure 3](../figures/cleaning_fig3_missing_by_month.png)

**Figure 3. Weather values are missing by collection period, not at random.** Monthly share of accidents with no precipitation, wind-speed or temperature value (months with at least 1,000 accidents); the shaded area is the period before the April 2019 change in weather data. Precipitation is missing for 87.1% of accidents in the month before the change and 3.7% in the month after, for every provider. A 'value was missing' indicator would therefore encode when an accident was recorded, so missing-value flags are not used as model inputs by default. *Data: US Accidents (Kaggle, March 2023 release), full file.*

### 5.6 Data types and inconsistent labels

- True/False text → 0/1; numbers → 32-bit floats; repeated text → categories.
- Whitespace and letter case standardised; placeholder text such as "Unknown" → missing.
- `Wind_Direction`: 24 labels → 18 (calm, variable and 16 compass points).
- `Weather_Condition`: the April 2019 vocabulary change was harmonised (1,193,320 rows), and the labels were grouped into weather families (`Weather_Group`).
- `Zipcode` is kept as text; a 5-digit `Zip5` was added.

### 5.7 Feature creation and lookups

| New column(s) | Purpose |
|---|---|
| `Start_Year`, `Start_Month`, `Start_DayOfWeek`, `Start_Hour`, `Is_Weekend`, `Is_Rush_Hour`, `Is_Holiday` | Time patterns (local clock time; U.S. federal holidays) |
| `Road_Type` | Interstate, freeway/expressway, U.S./state highway or local road, from the street name |
| `Region` | U.S. Census region (lookup on `State`; 0 unmatched) |
| `Weather_Group`, `Weather_Windy` | Weather families and a windy indicator |
| `Weather_Lag_min`, `Weather_Stale` | Gap between weather reading and accident; flagged above 60 minutes |
| `Weather_Period` | Before / from April 2019 |
| `Duration_min` | End minus start time, for EDA and the dashboard only |
| Duplicate, quality and split columns | Described in sections 5.1–5.5 and 6 |

No second dataset was merged. The two lookups (Census region, federal holidays) add context without changing the row count.

## 6. Data-leakage controls

| Risk | Control | Verified result |
|---|---|---|
| Post-event fields used as predictors | Role `leakage_post_event`; the model-input builder refuses them | 0 post-event columns among the 33 model inputs |
| Same accident in training and test data | Splits assigned per event group | 0 groups in both halves, for both splits |
| Test data influencing imputation | Medians fitted on the training rows of each split separately | Overall median temperature 63 °F (chronological training rows) vs 64 °F (random training rows) |
| Collection period hidden in missingness | Missing-value flags and `Weather_All_Missing` are not model inputs | — |
| Provider and period shaping the label | `Source`, `Start_Year`, `Start_Month`, `Weather_Period` excluded unless the team opts in | — |
| Incomplete model input | Built block by block for all 6,248,916 training rows | 0 missing values left |

**Two splits are provided.** `Split_Time` holds out the most recent 18% of accidents (from 1 May 2022: 1,361,783 rows; training 6,248,916) and is the primary test, because it measures performance on future accidents. `Split` is a random 80/20 split grouped by event (6,089,098 / 1,521,601).

![Figure 5](../figures/cleaning_fig5_severity_by_split.png)

**Figure 5. Severity labels change over time.** Share of accidents at each severity level in the chronological training period and the hold-out period (the most recent 18% of accidents). Severity 3 falls from 20.1% to 2.8% and Severity 1 rises from 0.5% to 2.6%. A random split would hide this drift, so the chronological split is the primary test of future performance. *Data: US Accidents (Kaggle, March 2023 release), full file.*

## 7. Results: before and after

| | Before (raw file) | After (shared cleaned table) |
|---|---|---|
| Rows | 7,728,394 | **7,610,699** |
| Columns | 46 | **79** (44 cleaned + 35 created; 2 constants removed) |
| Size on disk | 3.06 GB (CSV) | **0.83 GB** (31 compressed Parquet files) |
| Duplicate rows | 117,695 | 0 |
| Unparsed timestamps | 743,166 at risk | 0 |
| Invalid or implausible readings | present | 0 (282 masked after duplicate removal, all logged) |
| Invalid Severity | 0 | 0 |
| `Wind_Direction` labels | 24 | 18 |
| Missing `Sunrise_Sunset` / `Timezone` | 23,246 / 7,808 | 0 / 3 |
| Severity 1 / 2 / 3 / 4 | 67,366 / 6,156,981 / 1,299,337 / 204,710 | 65,570 / 6,048,391 / 1,295,235 / 201,503 |

Removing duplicates barely changes the class balance (Severity 2: 79.67% → 79.47%), so the cleaning did not distort the target.

**Column roles in the cleaned table:** 33 model features, 19 quality flags, 13 EDA-only columns, 6 post-event (leakage) columns, 4 label-process columns, 2 split columns, the target and the identifier. Only the 33 features enter the baseline model.

**Reproducibility and performance.** The same pipeline was run on a laptop and on Google Colab and produced identical results. Run time and peak memory were measured per notebook cell:

| Environment | Audit | Cleaning | Final step and reports | Total | Peak memory |
|---|---|---|---|---|---|
| Laptop (Jupyter) | 1 min 54 s | 1 min 27 s | 3 min 1 s | 6.6 min | 3.5 GB |
| Google Colab (free) | 7 min 39 s | 6 min 54 s | 4 min 48 s | 20.6 min | 3.1 GB |

## 8. Challenges and solutions

### 8.1 Data challenges

1. **Duplicates a normal check cannot see.** The unique `ID` hid 102,338 copies, and 15,357 more differed only in timestamp format. *Solution:* compare a fingerprint of each row's meaning, without `ID`.
2. **Deciding what counts as "the same accident".** Deleting near-identical reports would remove real data; ignoring them would let one accident appear in both training and test data. *Solution:* flag them, keep them, and split by event group.
3. **Timestamps in two formats.** Format inference silently turns 743,166 start times into missing values. *Solution:* explicit format plus a strict retry; 0 failures.
4. **Invalid values that look normal.** A wind speed of 1,087 mph is not missing, so `isna()` cannot find it. *Solution:* two-tier rules, with every masked value logged so the decision can be reversed.
5. **"Outlier" is not "error".** A generic IQR rule flags 19.6% of visibility readings and the normal pressure of mountain states. *Solution:* physical limits for errors; IQR for reporting only (Figure 4).
6. **Missing values that are not random.** Weather gaps follow the April 2019 collection change (Figure 3). *Solution:* keep missingness visible, exclude missing-value flags from the default model, impute only from training rows.
7. **Categories written several ways.** 24 wind labels for 18 directions; a weather vocabulary change. *Solution:* canonical mappings.
8. **Filling time zones correctly.** 24 states span several time zones (El Paso, Texas uses Mountain time). *Solution:* county-level time zones; 3 rows left missing rather than guessed.
9. **Severity labels changing over time.** *Solution:* a chronological hold-out as the primary test (Figure 5).
10. **A faulty weather station.** Station KJRB reports precipitation of exactly 9.93–10.00 in/h about 370 times, within the 12-inch limit. *Solution:* listed in the audit for a team decision.

### 8.2 Processing challenges

11. **A 3 GB file with every value read as text.** Large blocks pushed memory to 6.7 GB during the audit on a free 12 GB Colab machine. *Solution:* 250,000-row blocks, memory released after each block, and reports that load only the 25 columns they need and summarise the full table one block at a time. Peak memory fell to 3.1–3.5 GB.
12. **Crashed or disconnected sessions.** A restart erases all variables. *Solution:* a checkpoint after each pass, so **Run all** resumes where it stopped; a version number prevents results from old rules being reused.
13. **Long runs without visible progress.** *Solution:* progress messages for every block, a run-time and memory line after every cell, and faster text cleaning (each distinct value cleaned once).
14. **Different environments.** Colab cannot read files on a laptop. *Solution:* two notebook versions with identical code — one for Colab with Google Drive, one for a local Jupyter installation — each writing its own cleaning module.
15. **Computing day/night without a time-zone database.** *Solution:* U.S. time-zone offsets and daylight-saving rules are built into the code; the result agrees with the provider's own label for 99.77% of accidents.

## 9. Hand-off to the team

| File (repository path) | Use |
|---|---|
| `notebooks/DATA230_Cleaning_Local.ipynb`, `notebooks/DATA230_Cleaning_Colab.ipynb` | Full cleaning run with all outputs |
| `src/us_accidents_cleaning.py` | Cleaning functions, figure functions and loading helpers |
| `data/clean/us_accidents_clean/` | Cleaned table (Parquet; not committed — rebuilt by the notebook) |
| `data/clean/data_dictionary.csv` | Type, role, missing % and distinct values of every column |
| `data/clean/imputation_stats.csv`, `imputation_stats_random.csv` | Training-only medians for each split |
| `data/clean/invalid_values_log.csv` | Every masked value with its original reading and rule |
| `data/clean/report_numbers.csv` | Every number in this report |
| `figures/cleaning_fig1` … `fig5`, `figure_captions.md` | The five figures (500 dpi) and their captions |

```python
import us_accidents_cleaning as uc
df = uc.load_clean("data/clean/us_accidents_clean")              # EDA and dashboard
counts = uc.events_only(df)                                       # one row per accident, for counts
stats = uc.load_imputation_stats("data/clean/imputation_stats.csv")
X_train, y_train = uc.build_model_frame(df[df.Split_Time == "train"], stats)   # 33 model inputs
X_test,  y_test  = uc.build_model_frame(df[df.Split_Time == "test"],  stats)
```

## 10. Decisions for the team

1. **KJRB precipitation values:** mask them as an instrument error, or keep them?
2. **Label-process columns** (`Source`, `Start_Year`, `Start_Month`, `Weather_Period`): exclude from the baseline model (default) or include?
3. **Evaluation:** report the chronological split as the headline result, with the random split as a secondary check.

---

## Appendix A. Presentation material (about 2.5 minutes)

**Slide 1 — "7.7 million records, problems that raise no error"** (Figure 2)
- 117,695 duplicate rows hidden by a unique ID; 15,357 of them only visible when timestamps are compared as instants
- 743,166 timestamps in a second format; 282 invalid weather readings such as 1,087 mph wind

**Slide 2 — "Missing values are not random"** (Figure 1 + Figure 3)
- 22 of 46 columns have gaps; precipitation is missing for 87% of accidents before April 2019 and 4% after
- Missingness kept visible; imputed only from training rows, by state and month

**Slide 3 — "Outliers are not errors; one leakage-safe dataset"** (Figure 4 + Figure 5)
- Low pressure comes from mountain states: reported, not removed
- Severity 3 falls from 20.1% to 2.8% over time, so the chronological split is the primary test
- 7,610,699 rows × 79 columns, 33 model inputs; runs in 6.6 minutes on a laptop

**Speaking script.** "My part was turning the raw file into one dataset the team can trust. The file has 7.7 million accidents and 46 columns, and most of its problems raise no error. Every row has a unique ID, so a normal duplicate check finds nothing; without the ID, 102,338 rows are copies, and 15,357 more appear once we notice the same time written in two formats — 117,695 duplicates removed. We also found readings such as a wind speed of 1,087 miles per hour; we set those values to missing and flagged them, instead of deleting the accident. Missing weather turned out not to be random: precipitation is missing for 87% of accidents before April 2019 and under 4% after, so we keep missing values visible and fill them only for modelling, from training data. We kept genuine extremes — the low pressure readings come from mountain states — and tagged every column so that fields known only after an accident, like its end time or distance, can never enter a model. Finally, Severity 3 drops from 20% of accidents to under 3% in the most recent period, so we test on future accidents. The result is 7.6 million rows and 79 columns, reproducible in under seven minutes on a laptop."
