"""
Train/validation stage for the Year1CGPA machine-learning project.

Important evaluation rule:
- train.csv is used to fit candidate models.
- validation.csv is used to select/tune the RandomForest.
- test.csv is NOT read here and remains locked for final evaluation.

Primary model-selection metric:
    Weighted F1
Tie-breakers:
    Accuracy, then Cohen's kappa

Why weighted F1?
The target has five classes with unequal frequencies, so weighted F1 balances
precision/recall while respecting class prevalence better than accuracy alone.
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
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)

TRAIN = Path("train.csv")
VALIDATION = Path("validation.csv")
TARGET = "Year1CGPA"
RANDOM_STATE = 1

train_df = pd.read_csv(TRAIN, encoding="utf-8-sig")
val_df = pd.read_csv(VALIDATION, encoding="utf-8-sig")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(str)
X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET].astype(str)

numeric_features = ["HSCGraduationYear"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

grid = [
    {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
    }
    for n_estimators in [100, 250, 500]
    for max_features in ["sqrt", 2, 4]
    for min_samples_leaf in [1, 2, 3]
]

classes = sorted(y_train.unique())
y_val_bin = label_binarize(y_val, classes=classes)

results = []
best = None

for params in grid:
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_features=params["max_features"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_val, pred, average="weighted", zero_division=0
    )

    class_order = list(model.classes_)
    aligned_proba = np.column_stack(
        [proba[:, class_order.index(c)] for c in classes]
    )

    row = {
        **params,
        "accuracy": accuracy_score(y_val, pred),
        "weighted_precision": precision_w,
        "weighted_recall": recall_w,
        "weighted_f1": f1_w,
        "macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_val, pred),
        "weighted_roc_auc_ovr": roc_auc_score(
            y_val, proba, labels=model.classes_, multi_class="ovr", average="weighted"
        ),
        "weighted_prc_auc": average_precision_score(
            y_val_bin, aligned_proba, average="weighted"
        ),
    }
    results.append(row)

    score = (row["weighted_f1"], row["accuracy"], row["kappa"])
    if best is None or score > best["score"]:
        best = {"score": score, "params": params, "model": model, "metrics": row}

results_df = pd.DataFrame(results).sort_values(
    ["weighted_f1", "accuracy", "kappa"], ascending=False
)
results_df.to_csv("validation_random_forest_results.csv", index=False, encoding="utf-8-sig")

joblib.dump(best["model"], "best_validation_pipeline.joblib")

with open("best_validation_config.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "selection_rule": "maximize weighted_f1; tie-break by accuracy then kappa",
            "best_parameters": best["params"],
            "best_validation_metrics": {
                k: float(v)
                for k, v in best["metrics"].items()
                if k not in {"n_estimators", "max_features", "min_samples_leaf"}
            },
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Best parameters:", best["params"])
print("Best validation metrics:")
for k, v in best["metrics"].items():
    if k not in {"n_estimators", "max_features", "min_samples_leaf"}:
        print(f"  {k}: {v:.4f}")
