import numpy as np
import pandas as pd

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

from .config import TARGET, RANDOM_STATE

NUMERIC_FEATURES = ["HSCGraduationYear"]

def build_pipeline(X: pd.DataFrame, rf_params: dict) -> Pipeline:
    categorical_features = [c for c in X.columns if c not in NUMERIC_FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_features=rf_params["max_features"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", rf),
    ])

def calculate_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y, pred, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )

    y_bin = label_binarize(y, classes=classes)

    return {
        "accuracy": accuracy_score(y, pred),
        "weighted_precision": p_w,
        "weighted_recall": r_w,
        "weighted_f1": f1_w,
        "macro_precision": p_m,
        "macro_recall": r_m,
        "macro_f1": f1_m,
        "cohen_kappa": cohen_kappa_score(y, pred),
        "weighted_roc_auc_ovr": roc_auc_score(
            y, proba, labels=classes, multi_class="ovr", average="weighted"
        ),
        "macro_roc_auc_ovr": roc_auc_score(
            y, proba, labels=classes, multi_class="ovr", average="macro"
        ),
        "weighted_prc_auc": average_precision_score(
            y_bin, proba, average="weighted"
        ),
        "macro_prc_auc": average_precision_score(
            y_bin, proba, average="macro"
        ),
    }

def tune_random_forest(train_df: pd.DataFrame, validation_df: pd.DataFrame):
    """
    Tune only on train -> validation.

    Primary selection metric: weighted F1.
    Tie-breakers: accuracy, then Cohen's kappa.
    """
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(str)
    X_val = validation_df.drop(columns=[TARGET])
    y_val = validation_df[TARGET].astype(str)

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

    results = []
    best = None

    for params in grid:
        model = build_pipeline(X_train, params)
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_val, y_val)
        row = {**params, **metrics}
        results.append(row)

        score = (
            metrics["weighted_f1"],
            metrics["accuracy"],
            metrics["cohen_kappa"],
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "params": params,
                "metrics": metrics,
            }

    results_df = pd.DataFrame(results).sort_values(
        ["weighted_f1", "accuracy", "cohen_kappa"],
        ascending=False,
    ).reset_index(drop=True)

    return best, results_df
