# Final Model Card

## Intended task

Predict the five-class `Year1CGPA` category from 23 survey/context predictors.

## Model

scikit-learn `RandomForestClassifier` inside an end-to-end preprocessing Pipeline.

Locked configuration selected on Validation:

- `n_estimators = 100`
- `max_features = "sqrt"`
- `min_samples_leaf = 1`
- `random_state = 1`

## Training data for final fit

Train + Validation = 2,988 rows.

## Final evaluation

Held-out Test = 528 rows.

- Accuracy: 0.5985
- Weighted Precision: 0.6202
- Weighted Recall: 0.5985
- Weighted F1: 0.5984
- Macro F1: 0.6033
- Cohen's Kappa: 0.4811
- Weighted ROC-AUC: ~0.862
- Weighted PRC-AUC: ~0.715

## Main error pattern

The confusion matrix shows many errors occur between neighboring CGPA bands, especially around 3.00–3.75. This is plausible because the target is an ordered set of ranges but is modeled as nominal multiclass classification.

## Feature-selection finding

CFS + BestFirst selected a compact four-feature core (CurrentInstitution, HSCMath, HSCPhysics, HSCChemistry), but using this subset inside leakage-controlled cross-validation reduced accuracy to 36.6894%. The final model therefore keeps the broader predictor set.

## Limitations

- Test was not prospectively reserved before the earlier WEKA exploratory work.
- RandomForest impurity importance is predictive, not causal.
- High-cardinality categorical predictors can receive inflated impurity importance.
- WEKA and scikit-learn forests are not identical implementations.
- The final target bands are ordinal in meaning but treated as nominal classes.
- Data provenance includes synthetic-response artifacts; disclosure is required if synthetic records are part of the final modeling table.
