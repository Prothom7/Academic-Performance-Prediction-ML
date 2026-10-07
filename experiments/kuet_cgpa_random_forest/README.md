# KUET-only CGPA Random Forest experiment

This isolated experiment asks how well **22 background predictors** classify Year-1 and Year-2 CGPA bands among KUET respondents. The two targets have independent stratified splits, validation searches, locked models, and held-out evaluations. Results describe prediction **within KUET**; they do not establish causal relationships or generalize by themselves to other institutions.

## Source and subset

The sole input is the preserved [authoritative v3 dataset](../../data/processed/intermediate/academic_performance_clean_v3.csv), expected to have **3,516 rows and 32 columns**, SHA256 `7cb8520f8d2b97a2f4baf13b93de8cc93ed7c7bcc26a99d4153eb027ea8bbc80`. The runner verifies this contract and filters `CurrentInstitution` by an exact match to `kuet` after Unicode NFKC normalization, outer-whitespace trimming, and case folding. The verified KUET population is **1,377**. `data/kuet_full.csv` copies all source fields for these respondents and adds `source_row_id`, the zero-based row position in authoritative v3. No student record is manually changed.

Both targets use the canonical [`cgpa_band()` mapping](../../src/research/common.py): `0 = <3.00`, `1 = 3.00–3.25`, `2 = 3.25–3.50`, `3 = 3.50–3.75`, `4 = >3.75`. Raw bilingual target values remain in exported data; numeric codes are used only as model labels and in summaries. Missing targets or absent classes cause a hard failure.

## Predictors and model

The model imports the canonical `BASE_FEATURES` definition and removes `CurrentInstitution` because it is constant after KUET filtering. The same **22 predictors** are used for both targets: `Faculty`, `HSCGraduationYear`, `Gender`, `CollegeCategory`, `CollegeLocation`, `HSCLearningSource`, `CollegeAttendance`, `HSCMath`, `HSCPhysics`, `HSCChemistry`, `FamilyMembers`, `BirthOrder`, `HouseholdIncomeYearly`, `HSCMonthlyExpenditure`, `InternetAvailability`, `ComputerAvailability`, `UniversityPreferenceOrder`, `DepartmentPreferenceOrder`, `FatherEducation`, `MotherEducation`, `FatherEmploymentSector`, and `MotherEmploymentSector`.

Year-1 predicts `Year1CGPA`; Year-2 predicts `Year2CGPA` from those same background variables. The primary Year-2 model excludes `Year1CGPA` so the feature timing and scope are comparable across the two experiments. Neither target, `source_row_id`, research participation, later university outcomes, nor derived trajectory variables enter the predictor list.

`HSCGraduationYear` passes through as numeric. Other predictors receive most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore")`. These steps live inside an sklearn `Pipeline` and are fitted only on train for candidate evaluation, then on train plus validation for the final model. Raw category strings are retained. `RandomForestClassifier` uses seed **1**; tree fitting may use up to four workers, while prediction uses one worker for deterministic probability aggregation.

## Split, selection, and final test

For **each target separately**, stratified 70/15/15 splitting with seed **1** gives 963 train, 207 validation, and 207 test respondents. There is no overlap within a target; every KUET `source_row_id` occurs once. Every split contains all five classes, with proportions within two percentage points of the full KUET distribution. Year-1 and Year-2 splits need not contain the same respondents in corresponding partitions.

The train-only grid has 27 configurations: `n_estimators = {100, 250, 500}`, `max_features = {"sqrt", 2, 4}`, and `min_samples_leaf = {1, 2, 3}`. Validation selection ranks **weighted F1**, then **accuracy**, then **Cohen's kappa**, with stable grid order for an exact tie. After locking the parameters separately for each target, the runner refits on train plus validation and evaluates that fitted model once on its held-out test set. The test set has no role in selection.

## Reproduce

From the repository root, with Python 3.13 and packages from the root `requirements-lock.txt` installed:

```bash
python experiments/kuet_cgpa_random_forest/src/run_experiment.py
python -m pytest -q experiments/kuet_cgpa_random_forest/tests
```

The runner writes only under this experiment folder. `data/` holds the full KUET subset, independently stratified splits, and class summaries. `results/summary.csv` and `.json` compare the final experiments. Each `results/year*/` folder holds all 27 validation rows, selected configuration, held-out metrics, classification report, confusion matrix CSV/PNG, and source-row-linked test predictions with class probabilities. `models/` holds final train-plus-validation pipelines. Accuracy has a 95% Wilson interval; weighted and macro F1 have 1,500-draw row-bootstrap intervals with seed 1. The baseline predicts the training/development majority class on the same held-out test rows.

The completed run selected `n_estimators=250`, `max_features="sqrt"`, and `min_samples_leaf=1` independently for both targets. Year-1 held-out accuracy is **0.5556** (weighted F1 **0.5456**, macro F1 **0.5326**) versus majority accuracy **0.3285**. Year-2 held-out accuracy is **0.5749** (weighted F1 **0.5560**, macro F1 **0.5359**) versus majority accuracy **0.3092**. These are target-specific results from different test rows; the machine-readable metrics include all other requested measures and intervals.

## Limits

These estimates come from one KUET subset and one split per target. Bootstrap intervals condition on the final fitted model and test sample; they do not capture model-selection or between-cohort uncertainty. The Year-1 and Year-2 test sets are independently sampled, so their metrics are a descriptive comparison, not a paired significance test. CGPA is observed only as survey bands. The study does not identify causes of academic performance.
