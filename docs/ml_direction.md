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

**Established by committed outputs** (cleaning pipeline and `notebooks/03_preliminary_ml.ipynb`)
- **Class imbalance:** Severity 1–4 = 0.86% / 79.47% / 17.02% / 2.65%. Always predicting Severity 2 gives 79.5% accuracy overall and 91.8% on the chronological holdout.
- **Temporal label shift:** from the chronological training period to the holdout, Severity 3 falls from 20.1% to 2.8% and Severity 1 rises from 0.5% to 2.6%.
- **Source-associated label shift:** the temporal label shift is strongly associated with `Source`, consistent with provider-specific label regimes. Source1 supplies 91.85% of holdout rows and has no Severity 1 or Severity 3 records there (Section 9.1).
- **Collection-period break:** weather missingness and weather vocabulary change abruptly in April 2019.
- **Repeated reports:** 1,045,620 rows fall in near-duplicate event groups, so splits must keep each event on one side.
- **Functional form:** the nonlinear feature–Severity diagnostic is completed; temperature is curved, visibility is threshold-like and humidity is closer to linear (Section 9.2).

**Note on the existing EDA**
- The existing EDA uses the raw dataset and redefines weather and time features independently; for example, post-2019 "Fair" is grouped with "Other" rather than with "Clear". Relationships used for ML decisions will therefore be re-estimated with the cleaned feature definitions.

## 4. Baseline model: Multinomial Logistic Regression

Multinomial Logistic Regression is the baseline because it is interpretable, computationally efficient at this scale, and gives a clear reference for the value of added complexity.
* If more flexible models substantially outperform it, nonlinear effects or interactions carry important predictive information.
* If the gap is small, the simpler model is preferable.

Regularization will be used to reduce coefficient instability. Redundant representations of the same information, such as State, Region and Timezone, can be reviewed separately.

**Class imbalance:** class weighting will be the first imbalance treatment evaluated before resampling. Its effect is judged by Macro F1 and per-class recall, because accuracy is dominated by Severity 2.

## 5. Nonlinear model candidates

The completed diagnostics (Section 9.2) show that some continuous predictors have **curvature or threshold behavior** that a single linear term cannot represent. This justifies comparing nonlinear alternatives with the Logistic Regression baseline: flexible specifications (bins, splines) and tree-based models such as Random Forest and Gradient Boosting, which capture such patterns without manually specified cut points. Whether any of them performs better will only be known after training.

## 6. Validation strategy

The cleaned table already defines two splits:
* **Chronological holdout (`Split_Time`, primary):** accidents from 1 May 2022 onward (1,361,783 rows) against earlier accidents (6,248,916 rows). It tests generalization to future accidents under the observed label shift.
* **Random event-grouped split (`Split`, secondary):** an 80/20 split by event group, not stratified. All reports of one accident stay on the same side, which prevents near-duplicate leakage. The gap between the two splits will indicate the cost of temporal shift.

Model selection and tuning will be performed entirely within the training partition, so the chronological holdout remains untouched until final evaluation. Because the completed Source × time analysis (Section 9.1) shows Source-associated label differences, chronological holdout results will also be reported per `Source`, alongside the per-class metrics.

**Open question:** geographic holdout or region-level evaluation, considered only if Section 9.3 shows regional differences that add information beyond the temporal shift.

## 7. Evaluation metrics

**Macro F1**, **per-class precision and recall** and the **confusion matrix** evaluate performance across all four Severity levels. **Accuracy** is reported as a secondary metric. An ordinal error metric such as **Mean Absolute Error** between the predicted and true levels measures how far wrong predictions are.

## 8. Model interpretation

**Permutation importance** (all models) and **SHAP** (tree models) will identify which temporal, weather, road and geographic features drive predictions. Errors will also be compared across subgroups such as year, region and `Source`. Importance reflects association with the recorded labels, not causal effect.

## 9. Preliminary ML diagnostics

Completed in `notebooks/03_preliminary_ml.ipynb`; no model has been trained.

### 9.1 Severity mix by Source over time (completed)
Figure: `reports/figures/ml_severity_source_over_time.png`. Monthly within-Source Severity shares change differently across Sources over time, consistent with provider-specific label regimes; the temporal label shift is strongly associated with `Source`. Source1 dominates the chronological holdout (91.85% of rows) and contains no Severity 1 or Severity 3 records there, so all holdout Severity 1 (35,898) and Severity 3 (38,371) cases come from Source2 and Source3.

*Implication:* this supports the chronological holdout as the primary evaluation, because a random split would hide the shift. Overall metrics alone can hide subgroup behavior, so results are reported per class and per `Source`. `Source` stays a diagnostic and evaluation variable rather than automatically becoming a predictor.

### 9.2 Nonlinear feature–Severity relationships (completed)
Figure: `reports/figures/ml_nonlinear_feature_severity.png` shows Severity 3 vs Severity 2 log-odds by feature bin in Source2's chronological training rows, against a weighted straight-line reference. The notebook also covers the other contrasts and Source1.
- `Temperature(F)`: clearly curved (straight-line R² 0.03); a single linear term is too restrictive.
- `Visibility(mi)`: threshold-like, especially at very low visibility (R² 0.18).
- `Humidity(%)`: closer to linear (weighted RMSE 0.04 from the straight line); a linear term is a reasonable approximation.
- `Start_Hour`: cyclic, with sharp hourly steps; categorical (24-level) encoding is preferred over a simple linear numeric term, with sine/cosine terms as a compact fallback.
- `Wind_Speed(mph)`: not used as evidence of nonlinearity. Recording precision changes across collection periods (decimals before April 2019, whole numbers after), which can create an artifact, so wind needs a common precision before modeling and must not act as a proxy for collection period.

*Implication:* Multinomial Logistic Regression remains the baseline. The results justify comparing feature transformations and flexible specifications, and nonlinear models such as Random Forest or Gradient Boosting; they do not show that tree models will outperform Logistic Regression.

### 9.3 Geographic shift (optional, not yet done)
Compare each Census region's Severity distribution with the rest of the data, alongside the train/holdout difference. Purpose: decide whether region-level evaluation adds information beyond the established temporal shift.
