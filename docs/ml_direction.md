# Preliminary ML Direction

*Proposal for the mid-presentation. No model has been trained yet.*

## 1. Problem formulation

The target is accident `Severity` (1–4), so the primary task is **supervised multiclass classification** of all four levels, using only information available when an accident is first reported. Because the classes are ordered, evaluation will also be **ordinal-aware**, distinguishing small errors (4 → 3) from large ones (4 → 1). The binary Severity 3–4 contrast used in parts of the EDA is descriptive, not the modeling target.

## 2. Modeling dataset and leakage control

Modeling uses the shared cleaned table `us_accidents_clean` (7,610,699 rows × 79 columns: the full file minus 117,695 duplicates, with no sampling). `build_model_frame()` in `src/cleaning/us_accidents_cleaning.py` builds the model inputs. Columns follow the cleaning pipeline's roles:

- **Features (33):** time of day and week, holiday, location (State, Region, Timezone, coordinates), road type and road-feature flags, weather measurements, `Weather_Group`, day/night.
- **Post-event, never inputs:** `End_Time`, `Duration_min`, `End_Lat`, `End_Lng`, `Distance(mi)`, `Description`, all known only after the accident.
- **Label-process, not predictors:** `Source`, `Start_Year`, `Start_Month`, `Weather_Period`. They describe how and when a record was produced, so they are kept for diagnostics and stratified evaluation; as predictors the model would learn collection conventions.
- **EDA-only:** high-cardinality fields (City, County, Zipcode, Street, Airport_Code) and raw or redundant fields such as raw `Weather_Condition`; excluded from model inputs.

Missing weather values are imputed by the pipeline using State × Month medians fitted on training rows only. Missingness indicators are currently off because they risk encoding the collection period: precipitation is missing for 87.1% of accidents in March 2019 and 3.7% in April 2019.

## 3. Current evidence

**Established by committed cleaning outputs**
- **Class imbalance:** Severity 1–4 = 0.86% / 79.47% / 17.02% / 2.65%. Always predicting Severity 2 gives 79.5% accuracy overall and 91.8% on the chronological holdout.
- **Temporal label shift:** from the chronological training period to the holdout, Severity 3 falls from 20.1% to 2.8% and Severity 1 rises from 0.5% to 2.6%.
- **Collection-period break:** weather missingness and weather vocabulary change abruptly in April 2019.
- **Repeated reports:** 1,045,620 rows fall in near-duplicate event groups, so splits must keep each event on one side.

**Preliminary, not yet established**
- Preliminary inspection suggests that provider (`Source`) composition may contribute to the temporal label shift. Section 9.1 tests this explicitly, and detailed findings will be added after that analysis.
- The existing EDA uses the raw dataset and redefines weather and time features independently; for example, post-2019 "Fair" is grouped with "Other" rather than with "Clear". Relationships used for ML decisions will therefore be re-estimated with the cleaned feature definitions.

## 4. Baseline model: Multinomial Logistic Regression

Multinomial Logistic Regression is the baseline because it is interpretable, computationally efficient at this scale, and gives a clear reference for the value of added complexity.
* If more flexible models substantially outperform it, nonlinear effects or interactions carry important predictive information.
* If the gap is small, the simpler model is preferable.

Regularization will be used to reduce coefficient instability. Redundant representations of the same information, such as State, Region and Timezone, can be reviewed separately.

**Class imbalance:** class weighting will be the first imbalance treatment evaluated before resampling. Its effect is judged by Macro F1 and per-class recall, because accuracy is dominated by Severity 2.

## 5. Nonlinear model candidates

Random Forest and Gradient Boosting will be compared with the baseline if Section 9.2 shows **nonlinear relationships or thresholds** between the features and Severity. They capture such patterns without manually specifying transformations or cut points. The existing EDA does not yet show how Severity changes across the range of the continuous features, so this question remains open.

## 6. Validation strategy

The cleaned table already defines two splits:
* **Chronological holdout (`Split_Time`, primary):** accidents from 1 May 2022 onward (1,361,783 rows) against earlier accidents (6,248,916 rows). It tests generalization to future accidents under the observed label shift.
* **Random event-grouped split (`Split`, secondary):** an 80/20 split by event group, not stratified. All reports of one accident stay on the same side, which prevents near-duplicate leakage. The gap between the two splits will indicate the cost of temporal shift.

Model selection and tuning will be performed entirely within the training partition, so the chronological holdout remains untouched until final evaluation. If Section 9.1 confirms provider-specific label differences, holdout results will also be reported per `Source`.

**Open question:** geographic holdout or region-level evaluation, considered only if Section 9.3 shows regional differences that add information beyond the temporal shift.

## 7. Evaluation metrics

**Macro F1**, **per-class precision and recall** and the **confusion matrix** evaluate performance across all four Severity levels. **Accuracy** is reported as a secondary metric. An ordinal error metric such as **Mean Absolute Error** between the predicted and true levels measures how far wrong predictions are.

## 8. Model interpretation

**Permutation importance** (all models) and **SHAP** (tree models) will identify which temporal, weather, road and geographic features drive predictions. Errors will also be compared across subgroups such as year, region and `Source`. Importance reflects association with the recorded labels, not causal effect.

## 9. Remaining analysis before modeling

### 9.1 Severity mix by Source over time (required)
Compute the monthly share of each Severity level within each `Source` on the cleaned data, with the April 2019 break and the May 2022 split boundary marked. Purpose: determine whether the temporal label shift is partly provider-specific, and therefore what the chronological holdout measures.

### 9.2 Nonlinear feature–Severity relationships (required)
On the chronological training partition:
- **Continuous weather variables** (`Temperature(F)`, `Visibility(mi)`, `Wind_Speed(mph)`, optionally `Humidity(%)`): plot the empirical log-odds of each Severity level against Severity 2 across binned values, to check whether a linear-logit specification is reasonable.
- **`Start_Hour`:** treat it as a cyclic time feature and examine whether categorical or cyclical encoding is appropriate.

Purpose: test whether these relationships are nonlinear enough to justify transformations or tree-based models beyond the baseline.

### 9.3 Geographic shift (optional, if time permits after 9.1 and 9.2)
Compare each Census region's Severity distribution with the rest of the data, alongside the train/holdout difference. Purpose: decide whether region-level evaluation adds information beyond the established temporal shift.
