"""Single held-out evaluation and local experiment artifacts."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from statsmodels.stats.proportion import proportion_confint

from config import BANDS, CLASSES, FEATURES, OUTPUT_MODELS, OUTPUT_RESULTS, SEED


def save_json(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def bootstrap_f1(y, prediction, draws=1500):
    rng = np.random.default_rng(SEED)
    values = np.empty((draws, 2))
    for i in range(draws):
        sample = rng.integers(len(y), size=len(y))
        for j, average in enumerate(("weighted", "macro")):
            values[i, j] = f1_score(y[sample], prediction[sample], average=average,
                                    labels=CLASSES, zero_division=0)
    intervals = np.quantile(values, (0.025, 0.975), axis=0)
    return {f"{average}_f1_CI_{bound}": float(intervals[i, j])
            for j, average in enumerate(("weighted", "macro"))
            for i, bound in enumerate(("low", "high"))}


def metrics(y, predicted, probabilities):
    output = {"test_N": int(len(y)),
              "accuracy": float(accuracy_score(y, predicted)),
              "cohen_kappa": float(cohen_kappa_score(y, predicted)),
              "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
              "matthews_correlation_coefficient": float(matthews_corrcoef(y, predicted))}
    for average in ("weighted", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predicted, labels=CLASSES, average=average, zero_division=0)
        output.update({f"{average}_precision": float(precision),
                       f"{average}_recall": float(recall),
                       f"{average}_f1": float(f1)})
        try:
            output[f"{average}_roc_auc_ovr"] = float(roc_auc_score(
                y, probabilities, labels=CLASSES, multi_class="ovr", average=average))
        except ValueError:
            output[f"{average}_roc_auc_ovr"] = None
    low, high = proportion_confint(int(np.sum(y == predicted)), len(y), method="wilson")
    output.update(accuracy_CI_low=float(low), accuracy_CI_high=float(high))
    output.update(bootstrap_f1(y, predicted))
    output["test_class_distribution"] = {str(c): int(np.sum(y == c)) for c in CLASSES}
    return output


def save_confusion_figure(matrix, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=BANDS, yticklabels=BANDS,
           xlabel="Predicted CGPA band", ylabel="Actual CGPA band", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2
    for row in range(5):
        for column in range(5):
            ax.text(column, row, int(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_and_save(study, target, frames, labels, model, parameters, validation_grid):
    folder = OUTPUT_RESULTS / study
    folder.mkdir(parents=True, exist_ok=True)
    development = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
    development_labels = np.concatenate([labels["train"], labels["validation"]])
    test = frames["test"]
    y = labels["test"]
    predicted = model.predict(test.loc[:, FEATURES])
    probabilities = model.predict_proba(test.loc[:, FEATURES])
    if tuple(model.classes_) != CLASSES:
        raise ValueError(f"Fitted classes {model.classes_} differ from canonical bands")

    output = metrics(y, predicted, probabilities)
    baseline = DummyClassifier(strategy="most_frequent", random_state=SEED)
    baseline.fit(development.loc[:, FEATURES], development_labels)
    baseline_prediction = baseline.predict(test.loc[:, FEATURES])
    output["majority_class"] = int(baseline.classes_[np.argmax(baseline.class_prior_)])
    output["majority_baseline_accuracy"] = float(accuracy_score(y, baseline_prediction))
    output["accuracy_above_majority_baseline"] = output["accuracy"] - output["majority_baseline_accuracy"]
    output["target"] = target
    output["class_order"] = list(CLASSES)
    output["class_labels"] = list(BANDS)
    output["selected_parameters"] = parameters
    output["seed"] = SEED

    validation_grid.to_csv(folder / "validation_grid.csv", index=False, lineterminator="\n")
    save_json(folder / "best_config.json", {
        "target": target, "parameters": parameters, "validation_weighted_f1": float(validation_grid.iloc[0].weighted_f1),
        "validation_accuracy": float(validation_grid.iloc[0].accuracy),
        "validation_cohen_kappa": float(validation_grid.iloc[0].cohen_kappa),
        "selection_order": ["weighted_f1", "accuracy", "cohen_kappa"],
        "candidate_count": int(len(validation_grid)), "seed": SEED,
        "features": list(FEATURES)})
    save_json(folder / "test_metrics.json", output)

    report = pd.DataFrame(classification_report(
        y, predicted, labels=CLASSES, output_dict=True, zero_division=0)).T
    report.index.name = "class"
    report.reset_index().to_csv(folder / "classification_report.csv", index=False, lineterminator="\n")
    matrix = confusion_matrix(y, predicted, labels=CLASSES)
    pd.DataFrame(matrix, index=BANDS, columns=BANDS).rename_axis("actual_band").to_csv(
        folder / "confusion_matrix.csv", lineterminator="\n")
    save_confusion_figure(matrix, folder / "confusion_matrix.png", f"KUET {study} held-out test")
    predictions = pd.DataFrame({"source_row_id": test.source_row_id.to_numpy(),
                                "actual_band": y, "predicted_band": predicted})
    for i, band in enumerate(CLASSES):
        predictions[f"probability_{band}"] = probabilities[:, i]
    predictions.to_csv(folder / "test_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_MODELS / f"{study}_random_forest.joblib", compress=3)
    return output
