# Project Status

## Completed

- Source/raw survey data captured.
- Synthetic survey generation artifacts and quality-control reports preserved.
- Multiple cleaning iterations completed and archived.
- Final Year-1 CGPA modeling dataset prepared for WEKA/Python.
- Broad WEKA classifier sweep completed across baselines, Bayesian models, linear/SVM, neural networks, trees, ensembles, lazy learners, rule learners and meta-classifiers.
- RandomForest identified as the strongest model family.
- RandomForest tuning experiments completed for number of trees, K and minimum leaf weight M.
- RandomForest impurity importance generated.
- Information Gain ranking completed.
- CFS + BestFirst feature subset search completed.
- Leakage-controlled `AttributeSelectedClassifier` feature-selection experiment completed.
- Fixed stratified 70/15/15 Train/Validation/Test split created.
- Python RandomForest validation tuning completed without using Test.
- Final configuration locked and refit on Train+Validation.
- Final held-out Test evaluation completed.
- Final metrics, classification report, confusion matrix and serialized model saved.
- End-to-end reproducible Python pipeline created and verified.
- Complete project archive organized.

## Remaining for the academic submission / presentation

- Formal base/reference-paper comparison and contribution wording.
- Final written report/documentation in the required course format.
- Presentation slides, if required.
- Optional statistical uncertainty/repeated split analysis if the course expects inferential comparison.

## Current final result

Held-out test accuracy: **59.85%**  
Weighted F1: **59.84%**  
Kappa: **0.4811**  
Weighted ROC-AUC: **~0.862**  
Weighted PRC-AUC: **~0.715**
