# Academic Performance Prediction — Complete Project Snapshot

**Snapshot date:** 2026-09-05  
**Primary task:** Multiclass prediction of first-year university CGPA category (`Year1CGPA`)  
**Final modeling dataset:** 3,516 instances, 23 predictors, 5 target classes

This archive is the complete organized snapshot of the work completed so far: raw/source data, synthetic-data generation/QC artifacts, all cleaning stages, WEKA datasets/screenshots/logs, the broad model sweep, RandomForest tuning, feature analysis, train/validation/test files, Python code, serialized models, validation results, final held-out test results, the reference paper, and previous intermediate packages.

## Start here

1. Read `docs/PROJECT_STATUS.md` for what is complete and what remains.
2. Read `docs/PROJECT_HISTORY.md` for the full experiment story.
3. Read `docs/ARCHITECTURE.md` for the end-to-end ML pipeline.
4. Read `experiments/weka/WEKA_MODEL_RESULTS.md` for the model sweep.
5. Read `docs/MODEL_CARD.md` for the final RandomForest and limitations.
6. Run the reproducible pipeline with `python main.py`.

## Final coded pipeline

```text
Raw survey data
    ↓
Cleaning / field selection
    ↓
Final modeling table: 3,516 × 24
    ↓
Stratified 70 / 15 / 15 split
    ↓
Train (2,461) ──→ RandomForest candidates ──→ Validation (527)
                                                ↓
                                        lock best parameters
                                                ↓
                                  Train + Validation (2,988)
                                                ↓
                                          final fit
                                                ↓
                                         Test (528)
                                                ↓
        Accuracy / Precision / Recall / F1 / Kappa / ROC-AUC / PRC-AUC / CM
```

## Final held-out test result

| Metric | Value |
|---|---:|
| Accuracy | **59.85%** |
| Weighted Precision | **62.02%** |
| Weighted Recall | **59.85%** |
| Weighted F1 | **59.84%** |
| Macro F1 | **60.33%** |
| Cohen's Kappa | **0.4811** |
| Weighted ROC-AUC | **~0.862** |
| Weighted PRC-AUC | **~0.715** |

The validation-selected scikit-learn RandomForest configuration was `n_estimators=100`, `max_features="sqrt"`, `min_samples_leaf=1`, `random_state=1`.

## Strongest WEKA development result

The best development-stage WEKA RandomForest result was **60.9499% accuracy**, weighted F1 **0.609**, Kappa **0.4928**, with 250 trees, automatic K, and M tuned to 3.

## Important methodology notes

- The final coded pipeline keeps `test.csv` out of model selection and tuning.
- However, the full 3,516-record dataset had already been explored in WEKA before the fixed 70/15/15 split was created. Therefore the test set is held out from the **coded pipeline**, but it was not prospectively reserved before every exploratory analysis.
- Synthetic-data generation and quality-control artifacts are included under `data_generation/`. If synthetic responses contributed to the modeling dataset, they should be disclosed as synthetic/augmented data rather than represented as independently collected human responses.
- WEKA and scikit-learn RandomForest implementations and categorical-data handling differ, so their parameter values and probability metrics should not be treated as numerically identical configurations.

## Run the code

### Windows

Double-click `RUN_WINDOWS.bat`, or:

```bash
pip install -r requirements.txt
python main.py
```

### Linux/macOS

```bash
./RUN_LINUX_MAC.sh
```

## Folder guide

- `data/` — raw, intermediate, final and split datasets
- `data_generation/` — synthetic generation, form-related legacy tool, QC reports
- `src/` — reusable Python pipeline modules
- `scripts/standalone/` — individual scripts created during development
- `experiments/weka/` — WEKA screenshots, raw logs and consolidated model results
- `models/` — validation-stage and final serialized pipelines
- `results/` — validation search and final test outputs
- `references/` — base/reference paper
- `docs/` — architecture, history, data dictionary, model card, project status, presentation notes
- `archive/` — duplicate/intermediate packages retained only for provenance
