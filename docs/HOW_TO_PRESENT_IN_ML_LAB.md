# How to Present This Project in the ML Lab

## 30-second architecture explanation

“We predict a student's first-year CGPA category from pre-university academic, institutional, family, socioeconomic and preference variables. I first cleaned the survey data and removed fields that were irrelevant to the Year-1 prediction setup, then created a stratified 70/15/15 Train/Validation/Test split. Categorical inputs are one-hot encoded inside the pipeline, RandomForest is tuned using only Train and Validation, the configuration is locked, and the final model is evaluated once on the held-out Test set.”

## Why RandomForest?

The broad WEKA benchmark showed RandomForest was the strongest model family. The baseline forest achieved 60.3242% cross-validation accuracy, and tuning reached 60.9499%, outperforming the other recorded classifiers.

## What increased accuracy?

- Moving from baseline models to nonlinear ensemble methods produced the largest improvement.
- Increasing RandomForest trees from 100 to a selected 250 produced a modest development gain.
- Mild leaf regularization (M=3 in WEKA) produced the best WEKA development accuracy.

## What decreased accuracy?

Aggressive CFS feature selection reduced the feature set to only CurrentInstitution + the three HSC science/math score variables. In leakage-controlled cross-validation this fell to 36.6894% accuracy, showing that weaker contextual predictors still contain complementary information useful to the forest.

## Why these metrics?

- **Accuracy:** easy overall correctness measure.
- **Precision/Recall/F1:** show class-level behavior beyond raw accuracy.
- **Weighted F1:** primary validation metric because class sizes are unequal.
- **Macro F1:** gives each class equal weight.
- **Cohen's Kappa:** measures agreement beyond chance.
- **ROC-AUC:** measures multiclass ranking/discrimination quality.
- **PRC-AUC:** useful when class frequencies are unequal and focuses on precision-recall behavior.
- **Confusion matrix:** shows which CGPA ranges are confused with each other.

## Important WEKA MAE/RMSE clarification

For this nominal-class problem, WEKA's MAE/RMSE describe errors in the predicted class-probability distributions. They are **not** numeric CGPA regression errors.

## Contribution wording

A safe contribution statement is:

“This project builds and evaluates a reproducible academic-performance classification pipeline on the collected/assembled survey dataset. The work contributes a broad multi-family benchmark, systematic RandomForest tuning, multiple feature-analysis methods, an explicit leakage-controlled feature-selection test, and a final Train/Validation/Test implementation with reproducible code and several evaluation metrics.”

Do not describe predictive feature importance as causal influence.
