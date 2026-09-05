# End-to-End Architecture

## Data layer

`data/raw/academic_performance_raw.csv` is the reproducible source used by `main.py`.

## Cleaning layer

The code selects the 24 fields used for the Year-1 prediction problem and standardizes their names. The historical cleaning files are preserved separately to show the evolution of the dataset.

## Split layer

`train_test_split(..., stratify=Year1CGPA, random_state=1)` creates a 70/15/15 split. The second split divides the 30% temporary pool equally into Validation and Test.

## Preprocessing layer

- `HSCGraduationYear`: numeric passthrough.
- All other predictors: categorical one-hot encoding with `handle_unknown="ignore"`.
- Preprocessing is inside the scikit-learn Pipeline so transformations are learned from training data rather than fitted globally.

## Model-selection layer

RandomForest candidate configurations are fitted on Train and evaluated on Validation. Weighted F1 is the primary selection metric because the five target classes are not exactly equal in frequency. Accuracy and Cohen's Kappa are tie-breakers.

## Model-locking layer

After validation, the winning parameter set is frozen. No parameter is changed in response to Test performance.

## Final training layer

Train and Validation are combined. The preprocessing + RandomForest pipeline is refitted on those 2,988 records.

## Test layer

The locked pipeline predicts `test.csv` once and reports:

- Accuracy
- Weighted and macro Precision/Recall/F1
- Cohen's Kappa
- Multiclass one-vs-rest ROC-AUC
- PRC-AUC / average precision
- Per-class report
- Confusion matrix

## Serialization layer

The fitted end-to-end pipeline is stored as `models/final/final_random_forest_pipeline.joblib`, ensuring later predictions use the same encoder and classifier.
