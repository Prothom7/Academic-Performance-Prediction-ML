# Academic Performance Machine Learning Project

**Project snapshot:** 2026-09-05  
**Current status:** Phase 1 and Phase 2 modelling complete; Phase 3 robustness/reproducibility next  
**Completed prediction tasks:**

1. **Year-1 CGPA prediction** — 5-class classification
2. **Research Participation prediction** — binary classification

This repository contains the organized research record for the Academic Performance ML project: source and processed data, fixed splits, WEKA experiments, Python code, serialized models, model-selection records, final held-out evaluations, feature analysis, documentation, and research artifacts.

---

## Current headline results

| Task | Final model | Test set | Accuracy | Main F1 | ROC-AUC | PRC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Year-1 CGPA | scikit-learn RandomForest | 528 | **59.85%** | **0.598 weighted F1** | **~0.862 weighted** | **~0.715 weighted** |
| Research Participation | WEKA RandomForest, 100 trees, K=12 | 522 | **81.03%** | **0.699 Participated F1** | **0.879** | **0.827** |

The two F1 values are not directly interchangeable: Year-1 CGPA is a **five-class problem** reported with weighted F1, while Research Participation is a **binary problem** whose frozen primary metric is F1 for the minority/positive `Participated` class.

---

## Project structure at a glance

### Phase 1 — Year-1 CGPA prediction — COMPLETE

**Task:** predict the student's first-year university CGPA category from information available before or at university entry.

- Raw observations: **3,516**
- Predictors: **23**
- Target: `Year1CGPA`
- Target classes: **5**
- Fixed split:
  - Train: **2,461**
  - Validation: **527**
  - Test: **528**
- Final scikit-learn RandomForest:
  - `n_estimators=100`
  - `max_features="sqrt"`
  - `min_samples_leaf=1`
  - `random_state=1`

### Final Year-1 CGPA held-out result

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

The strongest WEKA development-stage RandomForest result was **60.9499% accuracy**, weighted F1 **0.609**, and Kappa **0.4928** using 250 trees, automatic K, and M=3. That result belongs to development; the held-out Python result above is the final coded result.

---

## Phase 2 — Research Participation prediction — COMPLETE IN WEKA

**Task:** predict whether a student belongs to `Participated` or `NotParticipated` in research activity.

### Data audit

- Raw rows: **3,516**
- Missing target rows removed: **37**
- Exact duplicate rows removed: **4**
- Final binary dataset: **3,475 rows**
- Predictors: **24**
- Target distribution:
  - `NotParticipated`: **2,239 (64.43%)**
  - `Participated`: **1,236 (35.57%)**

### Fixed split

| Split | Instances | NotParticipated | Participated |
|---|---:|---:|---:|
| Train | **2,432** | 1,567 | 865 |
| Validation | **521** | 336 | 185 |
| Test — locked during development | **522** | 336 | 186 |

Broad model benchmarking used **10-fold stratified cross-validation on Train only**. The frozen primary model-selection metric was **F1 for `Participated`**, not overall accuracy.

### Broad WEKA benchmark

- Benchmark slots tracked: **33**
- Substantive completed model results: **28**
- Unavailable/skipped algorithms: **4**
- Non-informative duplicate Vote configuration: **1**
- Broad-benchmark leader: **RandomForest default**
  - Train-CV Accuracy: **79.11%**
  - Participated F1: **0.650**
  - ROC-AUC: **0.849**
  - PRC-AUC: **0.803**

The permanent reconstructed benchmark archive is kept under:

```text
experiments/weka/research_participation/benchmark/
```

The reconstructed spreadsheet preserves the experiment ledger and is **not** a substitute for raw per-model WEKA result buffers that were not individually archived.

### Validation and tuning

Default RandomForest validation:

- Accuracy: **81.19%**
- Participated F1: **0.682**

The final configuration was selected after limited RandomForest tuning:

```text
weka.classifiers.trees.RandomForest
-P 100 -I 100 -num-slots 1 -K 12 -M 1.0 -V 0.001 -S 1
```

For K=12:

- Train 10-fold CV F1: **0.671**
- Validation Accuracy: **81.96%**
- Validation Precision: **0.818**
- Validation Recall: **0.632**
- Validation F1: **0.713**
- Validation ROC-AUC: **0.870**
- Validation PRC-AUC: **0.823**
- Validation MCC: **0.595**

After this point, the model and hyperparameters were frozen.

### Final locked Test evaluation

Train and Validation were combined (**2,953 instances**) and the frozen model was fitted before the locked Test was evaluated.

| Metric | Final Test |
|---|---:|
| Accuracy | **81.0345%** |
| Participated Precision | **0.804** |
| Participated Recall | **0.618** |
| Participated F1 | **0.699** |
| ROC-AUC | **0.879** |
| PRC-AUC | **0.827** |
| Cohen's Kappa | **0.5641** |
| MCC | **0.574** |
| Weighted F1 | **0.804** |
| Balanced Accuracy | **0.7675** |

Confusion matrix:

```text
                       Predicted
                    No        Yes
Actual No           308        28
Actual Yes           71       115
```

The Validation-to-Test F1 change was only **-0.014** (0.713 → 0.699), while ROC-AUC increased from 0.870 to 0.879 and PRC-AUC from 0.823 to 0.827.

No model selection or hyperparameter tuning should be performed using this Test result.

---

## Research Participation feature importance

WEKA RandomForest impurity-based importance ranked the following features highest:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | HSCGraduationYear | **0.81** |
| 2 | Gender | **0.80** |
| 3 | Faculty | **0.76** |
| 4 | CollegeLocation | **0.70** |
| 5 | CollegeCategory | **0.69** |
| 6 | HSCMath | **0.69** |
| 7 | HSCLearningSource | **0.67** |
| 8 | CollegeAttendance | **0.65** |
| 9 | HSCPhysics | **0.64** |
| 10 | ComputerAvailability | **0.64** |

These values represent **model-specific predictive importance**, not causal effects. Impurity-based feature importance may also favor variables with more available split points or categories.

---

## Running `main.py`

`main.py` is now a project launcher as well as the Phase-1 reproducibility entry point.

### Show current project status — default and recommended

```bash
python main.py
```

or:

```bash
python main.py status
```

The default command **does not run any experiment or evaluate any Test set**.

### Reproduce Year-1 cleaning, split, and Train → Validation selection

```bash
python main.py run-year1
```

This intentionally stops before the final held-out Test evaluation.

### Reproduce the already-frozen Year-1 final Test evaluation

Only do this when intentionally reproducing the final recorded pipeline:

```bash
python main.py run-year1 --evaluate-test
```

### Research Participation Python status

The Research Participation study is complete in WEKA, including the final locked Test evaluation and saved model. A faithful Python reproduction is a **Phase 3 task** because WEKA and scikit-learn RandomForest handling of categorical predictors and `K`/`max_features` are not numerically identical.

---

## Methodology and test-governance notes

### Year-1 CGPA

- The coded Python pipeline keeps its fixed Test set out of model selection and tuning.
- The full 3,516-row dataset had been explored in WEKA before the fixed 70/15/15 split was created. Therefore the Test set is held out from the coded pipeline, but it was not prospectively reserved before every exploratory analysis in the entire historical project.

### Research Participation

- A fixed Train/Validation/Test split was created before the Research Participation model sweep.
- The 522-row Test split was not evaluated during algorithm selection or hyperparameter tuning.
- The final configuration was frozen before the official Test evaluation.
- During later feature-importance extraction, WEKA remained configured with the supplied Test set and repeated the identical final evaluation. No parameters or model-selection decisions were changed after seeing the Test result; the first locked-test run remains the official final estimate.

### Synthetic-data provenance

Synthetic-data generation and quality-control artifacts are stored under `data_generation/`. If synthetic responses contributed to any modelling dataset, they must be disclosed as **synthetic/simulated/augmented records** rather than represented as independently collected human respondents.

---

## Repository guide

```text
Academic_Performance_ML_Complete/
├── main.py
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── research_participation/
│   └── splits/
│       └── research_participation/
├── data_generation/
├── src/
├── scripts/
│   └── standalone/
├── experiments/
│   └── weka/
│       └── research_participation/
│           ├── benchmark/
│           ├── validation/
│           ├── tuning/
│           └── final/
├── models/
│   ├── reproduction/
│   └── research_participation/
│       └── final/
├── results/
│   ├── reproduction/
│   └── research_participation/
│       └── final/
├── docs/
│   └── research_participation/
├── references/
├── verification/
└── archive/
```

### Important Research Participation artifacts

Expected project locations include:

```text
experiments/weka/research_participation/benchmark/
    ResearchParticipation_Benchmark_28_Reconstructed.xlsx
    ResearchParticipation_Benchmark_28_Reconstructed.csv

experiments/weka/research_participation/final/
    ResearchParticipation_Final_RandomForest_Test_Result.txt
    ResearchParticipation_Final_RandomForest_FeatureImportance.txt

models/research_participation/final/
    ResearchParticipation_Final_RandomForest_K12.model

results/research_participation/final/
    ResearchParticipation_Final_Classification_Report.xlsx
    ResearchParticipation_Final_Confusion_Matrix.xlsx
    ResearchParticipation_Final_Feature_Importance.xlsx
    ResearchParticipation_Complete_Experiment_Summary.xlsx

docs/research_participation/
    ResearchParticipation_Phase2_Methodology_and_Results.docx
```

---

## Reference paper positioning

The included reference article, *Dataset of academic performance evolution for engineering students* (Delahoz-Dominguez, Zuluaga, & Fontalvo-Herrera, 2020), is primarily a **dataset/data-description paper** based on 12,411 engineering students and 44 variables. It motivates educational-data-mining analysis but does not provide a directly comparable classifier benchmark. The present project therefore should **not** claim to outperform that paper on accuracy or F1; the contribution here is a deeper predictive modelling and evaluation workflow on a different dataset.

---

## Next phase

### Phase 3 — Research Participation Robustness, Feature Analysis, and Reproducibility

Planned work:

1. Information Gain feature ranking
2. CFS + BestFirst feature selection
3. Leakage-controlled feature-selection experiments
4. RP-A vs RP-B predictor timing comparison
5. Python reproduction of Research Participation modelling
6. Descriptive-statistics / participant-characteristics tables
7. Confidence intervals or repeated-CV uncertainty analysis
8. Final paper integration and reproducibility audit

No additional model tuning should be performed against the locked Research Participation Test set.
