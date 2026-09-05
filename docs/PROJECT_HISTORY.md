# Project History / Experiment Story

## 1. Data generation and quality control

The project began with survey-response data and synthetic response generation/QC artifacts. The original and updated synthetic datasets, quality-control reports, and the legacy form-submission script are preserved under `data_generation/` for provenance.

## 2. Cleaning iterations

Several cleaning versions were produced (`academic_performance_clean.csv`, `v2`, `v3`) along with WEKA ARFF working files. These are archived as intermediate evidence rather than used as the canonical final input.

The final Year-1 CGPA modeling table contains 24 attributes: 23 predictors plus the five-class `Year1CGPA` target.

## 3. Broad WEKA model sweep

A wide set of classification families was tested with 10-fold cross-validation. RandomForest emerged as the strongest baseline at 60.3242% accuracy. MLP, LMT, RandomCommittee, KStar, IBk and RandomTree were also among the stronger models.

## 4. RandomForest tuning

- RF-T01: tuning tree count selected 250 trees; accuracy 60.7509%.
- RF-T02: tuning K selected K=2; accuracy 60.6086%.
- RF-T03: tuning minimum leaf weight selected M=3; accuracy **60.9499%**.

The gains were modest but consistent enough to motivate RandomForest as the development champion.

## 5. Feature analysis

RandomForest impurity importance emphasized institutional, parental, attendance, preference and contextual features. Information Gain ranked HSC Math, Current Institution, HSC Physics and HSC Chemistry as the strongest individual predictors. CFS + BestFirst selected exactly those four features.

This produced two complementary interpretations: a small core of directly predictive academic/institutional variables, and a broader set of contextual variables that may contribute through nonlinear interactions.

## 6. Leakage-controlled feature-selection test

FS-E01 used `AttributeSelectedClassifier` so CFS selection occurred inside cross-validation. It also nested RandomForest M tuning inside the training folds.

Result: accuracy fell to **36.6894%** and weighted F1 to **0.351**, versus 60.9499% / 0.609 for the full tuned forest. Therefore aggressive reduction to the four-feature core removed substantial complementary signal.

## 7. Fixed Train/Validation/Test pipeline

A reproducible stratified split was created:

- Train: 2,461 (69.99%)
- Validation: 527 (14.99%)
- Test: 528 (15.02%)

Test was excluded from the coded tuning stage.

## 8. Python validation tuning

The Python pipeline used one-hot encoding for categorical predictors and passed `HSCGraduationYear` as numeric. A compact RandomForest grid was evaluated on Validation using weighted F1 as the primary selection criterion, with accuracy and Kappa as tie-breakers.

Selected configuration: 100 trees, `max_features=sqrt`, `min_samples_leaf=1`.

Validation: 55.60% accuracy, weighted F1 55.33%, Kappa 0.4255.

## 9. Final held-out test

The selected model was refit on Train+Validation (2,988 rows) and evaluated once on Test (528 rows).

Final: 59.85% accuracy, 59.84% weighted F1, 0.4811 Kappa, ~0.862 weighted ROC-AUC and ~0.715 weighted PRC-AUC.

## 10. Reproducible packaging

The workflow was consolidated into `main.py` + reusable modules under `src/`, with datasets, results, models, experiment evidence and reference material in a professional project structure.
