# Analytical decision log

1. Retain the current authoritative full dataset (3516 rows, 32 variables) without current-population duplicate exclusions. Historical research reproduces its existing explicit protocol as a separate experiment.
2. Use survey CGPA categories in their documented order; do not infer continuous scores. High performance is the upper two bands, operationally >=3.50.
3. Classify academic movement from the signed ordinal band difference. Fit improvement and decline separately against all other records.
4. Always include Year-1 starting band in adjusted mobility; include it in the main adjusted Year-2 model. Include current Year-2 standing in research association models, while keeping it out of historical-horizon research prediction.
5. Use seed 1 for splits, forests and bootstrap draws. Keep model configurations and preprocessing in metric JSON files. New categorical imputation learns only from training data.
6. Preserve Study A's frozen configuration and existing splits. Select one common Study C configuration from mean C1/C2 validation F1 to isolate the feature comparison. Never select with Test results.
7. Encode research as any Yes versus No; retain missing outcomes as missing. Historical nominal-attribute WEKA is independently refitted and compared with saved-model replay. Its header is used only for exact symbolic encoding, never for target-driven tuning.
8. Specify all background covariates rather than choose predictors from p-values. Add contemporary contexts to association models for mobility/research only. Missing-covariate exclusions are model-specific complete cases, not population exclusions.
9. Center graduation year at 2020. Treat selected bands as ordered trends. Reference categories are recorded per model; no nominal integer labels imply an order.
10. Pool Faculty ECE into Others only in fitted association models to avoid sparse-category instability; retain original descriptive/subgroup reporting.
11. Apply BH FDR to the predeclared exploratory families in [methodology](METHODOLOGY.md). Use nested LR interaction tests and joint factor Wald tests, with raw p and q preserved.
12. Standardize marginal predictions over each model's complete-case covariate population. Use Wilson proportion CIs, seeded percentile bootstrap metric/correlation CIs, and parameter-covariance delta CIs for standardized probabilities.
13. Compare checkpoints after computation; preserve discrepancies and record dataset, engine, feature set, denominator and configuration rather than optimize toward historical values.
14. Generate every result table, figure, conclusion status and numerical report statement from executable outputs. Preserve prior artifacts and mark their metric prose as historical snapshots.
