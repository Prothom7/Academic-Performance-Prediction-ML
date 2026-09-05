"""
FINAL-EVAL-01: Locked RandomForest evaluation on the held-out test set.

Workflow:
1. Hyperparameters were selected earlier using train.csv + validation.csv only.
2. The locked hyperparameters are:
       n_estimators = 100
       max_features = "sqrt"
       min_samples_leaf = 1
       random_state = 1
3. Train and validation sets are combined.
4. Preprocessing is refit on Train+Validation only.
5. The locked model is fitted once.
6. test.csv is read only for the final evaluation.

Important project limitation:
The original 3516-row dataset had already been explored in WEKA before this
70/15/15 split was created, so this test set is a held-out test for the coded
pipeline, but it was not prospectively reserved before all exploratory work.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

TRAIN = Path("train.csv")
VALIDATION = Path("validation.csv")
TEST = Path("test.csv")
TARGET = "Year1CGPA"
RANDOM_STATE = 1

LOCKED_PARAMS = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
}

train_df = pd.read_csv(TRAIN, encoding="utf-8-sig")
val_df = pd.read_csv(VALIDATION, encoding="utf-8-sig")
test_df = pd.read_csv(TEST, encoding="utf-8-sig")

dev_df = pd.concat([train_df, val_df], ignore_index=True)

X_dev = dev_df.drop(columns=[TARGET])
y_dev = dev_df[TARGET].astype(str)
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(str)

numeric_features = ["HSCGraduationYear"]
categorical_features = [c for c in X_dev.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=1,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_dev, y_dev)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)
classes = list(model.classes_)
y_test_bin = label_binarize(y_test, classes=classes)

precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    y_test, pred, average="weighted", zero_division=0
)
precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
    y_test, pred, average="macro", zero_division=0
)

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "weighted_precision": precision_w,
    "weighted_recall": recall_w,
    "weighted_f1": f1_w,
    "macro_precision": precision_m,
    "macro_recall": recall_m,
    "macro_f1": f1_m,
    "cohen_kappa": cohen_kappa_score(y_test, pred),
    "weighted_roc_auc_ovr": roc_auc_score(
        y_test, proba, labels=classes, multi_class="ovr", average="weighted"
    ),
    "weighted_prc_auc": average_precision_score(
        y_test_bin, proba, average="weighted"
    ),
}

print("FINAL HELD-OUT TEST METRICS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, pred, labels=classes, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, pred, labels=classes))

joblib.dump(model, "final_random_forest_pipeline.joblib")
