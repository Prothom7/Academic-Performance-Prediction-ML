import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from .config import TARGET
from .modeling import build_pipeline, calculate_metrics

def final_test_evaluation(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    locked_params: dict,
    results_dir: Path,
    models_dir: Path,
):
    """
    Refit the locked model on Train+Validation and evaluate once on Test.

    No parameter is changed after observing the test results.
    """
    dev_df = pd.concat([train_df, validation_df], ignore_index=True)

    X_dev = dev_df.drop(columns=[TARGET])
    y_dev = dev_df[TARGET].astype(str)
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(str)

    model = build_pipeline(X_dev, locked_params)
    model.fit(X_dev, y_dev)

    pred = model.predict(X_test)
    metrics = calculate_metrics(model, X_test, y_test)

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "final_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "locked_parameters": locked_params,
                "train_plus_validation_rows": len(dev_df),
                "test_rows": len(test_df),
                **{k: float(v) for k, v in metrics.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    classes = list(model.classes_)
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report_df.to_csv(
        results_dir / "final_classification_report.csv",
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(y_test, pred, labels=classes)
    short_labels = [
        "3.00-3.25",
        "3.25-3.50",
        "3.50-3.75",
        "Above 3.75",
        "Below 3.00",
    ]

    pd.DataFrame(
        cm,
        index=[f"Actual: {x}" for x in short_labels],
        columns=[f"Predicted: {x}" for x in short_labels],
    ).to_csv(
        results_dir / "final_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm)
    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Final Held-Out Test Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(
        results_dir / "final_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    joblib.dump(model, models_dir / "final_random_forest_pipeline.joblib")
    return metrics
