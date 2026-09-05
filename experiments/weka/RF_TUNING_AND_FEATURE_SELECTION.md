# RandomForest Tuning and Feature-Selection Experiments

## RF baseline

100 trees, K=0, M=1, seed 1: 60.3242% accuracy, weighted F1 0.604, Kappa 0.4855.

## RF-T01 — tree count

CVParameterSelection over 50–500 trees selected 250. Outer 10-fold CV: 60.7509% accuracy, weighted F1 0.608, Kappa 0.4901, ROC 0.867, PRC 0.729.

## RF-T02 — K

With 250 trees, tuning K selected K=2. Accuracy 60.6086%, weighted F1 0.605, Kappa 0.4865, ROC 0.870, PRC 0.731. Accuracy was slightly below RF-T01.

## RF-T03 — minimum leaf weight M

With 250 trees and K=0, tuning M=1..10 selected M=3. Accuracy **60.9499%**, weighted F1 **0.609**, Kappa **0.4928**, ROC 0.860, PRC 0.723. This became the WEKA development champion by accuracy.

## Feature analysis

### Information Gain top four

1. HSCMath
2. CurrentInstitution
3. HSCPhysics
4. HSCChemistry

### CFS + BestFirst

Selected exactly:

- CurrentInstitution
- HSCMath
- HSCPhysics
- HSCChemistry

### RandomForest impurity importance

The highest values included CurrentInstitution, MotherEducation, FatherEducation, CollegeAttendance, UniversityPreferenceOrder, DepartmentPreferenceOrder, FatherEmploymentSector and HSCLearningSource. These are predictive importance values, not causal effects.

## FS-E01 — leakage-controlled subset evaluation

Exact WEKA scheme:

```text
weka.classifiers.meta.AttributeSelectedClassifier
-E "weka.attributeSelection.CfsSubsetEval -P 1 -E 1"
-S "weka.attributeSelection.BestFirst -D 1 -N 5"
-W weka.classifiers.meta.CVParameterSelection --
-P "M 1.0 10.0 10.0" -X 10 -S 1
-W weka.classifiers.trees.RandomForest --
-P 100 -I 250 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1
```

The full-data CFS display selected the same four features and the full-data internal tuning display selected M=2. Under outer 10-fold CV:

- Accuracy: **36.6894%**
- Weighted F1: **0.351**
- Kappa: **0.1739**
- Weighted ROC-AUC: **0.660**
- Weighted PRC-AUC: **0.337**

Compared with the full tuned RF, accuracy dropped about 24.26 percentage points. The compact core contains strong direct signal, but the full nonlinear ensemble benefits from broader contextual information and interactions.
