"""Train-only grid search and development-set refit for each target."""
import itertools
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import FEATURES, SEED

GRID = tuple(dict(n_estimators=n, max_features=k, min_samples_leaf=m)
             for n, k, m in itertools.product((100, 250, 500), ("sqrt", 2, 4), (1, 2, 3)))


def make_pipeline(parameters):
    numeric = ["HSCGraduationYear"]
    categorical = [name for name in FEATURES if name not in numeric]
    transform = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("numeric", "passthrough", numeric),
    ])
    forest = RandomForestClassifier(**parameters, random_state=SEED,
                                    n_jobs=min(4, os.cpu_count() or 1))
    return Pipeline([("preprocess", transform), ("model", forest)])


def fit_model(parameters, frame, labels):
    model = make_pipeline(parameters)
    model.fit(frame.loc[:, FEATURES], labels)
    # Deterministic tree-order summation for prediction probabilities.
    model.named_steps["model"].set_params(n_jobs=1)
    return model


def validation_search(train, validation, train_labels, validation_labels):
    rows = []
    for params in GRID:
        model = fit_model(params, train, train_labels)
        predictions = model.predict(validation.loc[:, FEATURES])
        rows.append({**params,
                     "weighted_f1": float(f1_score(validation_labels, predictions, average="weighted", zero_division=0)),
                     "accuracy": float(accuracy_score(validation_labels, predictions)),
                     "cohen_kappa": float(cohen_kappa_score(validation_labels, predictions))})
    table = pd.DataFrame(rows).sort_values(
        ["weighted_f1", "accuracy", "cohen_kappa"],
        ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    best = table.iloc[0]
    parameters = {"n_estimators": int(best.n_estimators),
                  "max_features": best.max_features,
                  "min_samples_leaf": int(best.min_samples_leaf)}
    return table, parameters
