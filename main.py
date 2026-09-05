"""Academic Performance ML Project entry point.

This file is intentionally conservative about held-out test data.

Current project status
----------------------
1. Year-1 CGPA prediction
   - Multiclass task (5 classes)
   - Python + WEKA workflow completed
   - Final held-out test accuracy: 59.85%
   - Final weighted F1: 0.5984

2. Research Participation prediction
   - Binary task: NotParticipated vs Participated
   - WEKA benchmarking, validation, tuning, locked-test evaluation,
     model saving, and feature-importance analysis completed
   - Final held-out test accuracy: 81.03%
   - Participated F1: 0.699
   - Python reproduction is intentionally left for the next phase

Running ``python main.py`` prints project status only. This avoids accidental
re-execution of a held-out test. To reproduce the coded Year-1 CGPA workflow,
run ``python main.py run-year1``. Add ``--evaluate-test`` only when you
intentionally want to reproduce the already-frozen final held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


YEAR1_FINAL = {
    "task": "Year-1 CGPA (5-class classification)",
    "instances": 3516,
    "predictors": 23,
    "train": 2461,
    "validation": 527,
    "test": 528,
    "model": "scikit-learn RandomForest",
    "configuration": {
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "random_state": 1,
    },
    "metrics": {
        "accuracy": 0.5985,
        "weighted_precision": 0.6202,
        "weighted_recall": 0.5985,
        "weighted_f1": 0.5984,
        "macro_f1": 0.6033,
        "kappa": 0.4811,
        "weighted_roc_auc": 0.862,
        "weighted_prc_auc": 0.715,
    },
}


RESEARCH_PARTICIPATION_FINAL = {
    "task": "Research Participation (binary classification)",
    "instances": 3475,
    "predictors": 24,
    "train": 2432,
    "validation": 521,
    "test": 522,
    "development_fit": 2953,
    "positive_class": "Participated",
    "primary_metric": "Participated F1",
    "model": "WEKA RandomForest",
    "configuration": "-P 100 -I 100 -num-slots 1 -K 12 -M 1.0 -V 0.001 -S 1",
    "metrics": {
        "accuracy": 0.810345,
        "precision_participated": 0.804,
        "recall_participated": 0.618,
        "f1_participated": 0.699,
        "roc_auc": 0.879,
        "prc_auc": 0.827,
        "kappa": 0.5641,
        "mcc": 0.574,
        "weighted_f1": 0.804,
        "balanced_accuracy": 0.7675,
    },
    "confusion_matrix": {
        "tn": 308,
        "fp": 28,
        "fn": 71,
        "tp": 115,
    },
    "benchmark": {
        "tracked_slots": 33,
        "substantive_completed_models": 28,
        "skipped_unavailable": 4,
        "non_informative_duplicate_configurations": 1,
    },
}


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def print_status() -> None:
    """Print the current project snapshot without running any experiment."""
    print("=" * 78)
    print("ACADEMIC PERFORMANCE ML PROJECT — CURRENT STATUS")
    print("=" * 78)

    print("\n[PHASE 1] YEAR-1 CGPA PREDICTION — COMPLETE")
    print(f"Dataset: {YEAR1_FINAL['instances']} instances, {YEAR1_FINAL['predictors']} predictors")
    print(
        "Split: "
        f"Train={YEAR1_FINAL['train']}, "
        f"Validation={YEAR1_FINAL['validation']}, "
        f"Test={YEAR1_FINAL['test']}"
    )
    print(f"Final model: {YEAR1_FINAL['model']}")
    print(f"Test accuracy: {_pct(YEAR1_FINAL['metrics']['accuracy'])}")
    print(f"Weighted F1:  {YEAR1_FINAL['metrics']['weighted_f1']:.4f}")
    print(f"Kappa:        {YEAR1_FINAL['metrics']['kappa']:.4f}")
    print(f"ROC-AUC:      ~{YEAR1_FINAL['metrics']['weighted_roc_auc']:.3f}")
    print(f"PRC-AUC:      ~{YEAR1_FINAL['metrics']['weighted_prc_auc']:.3f}")

    rp = RESEARCH_PARTICIPATION_FINAL
    print("\n[PHASE 2] RESEARCH PARTICIPATION PREDICTION — COMPLETE (WEKA)")
    print(f"Dataset: {rp['instances']} instances, {rp['predictors']} predictors")
    print(
        "Split: "
        f"Train={rp['train']}, Validation={rp['validation']}, Test={rp['test']}"
    )
    print(f"Final fit: Train+Validation={rp['development_fit']}")
    print(f"Final model: {rp['model']}")
    print(f"WEKA options: {rp['configuration']}")
    print(f"Test accuracy:       {_pct(rp['metrics']['accuracy'])}")
    print(f"Participated F1:     {rp['metrics']['f1_participated']:.3f}")
    print(f"Participated recall: {rp['metrics']['recall_participated']:.3f}")
    print(f"ROC-AUC:             {rp['metrics']['roc_auc']:.3f}")
    print(f"PRC-AUC:             {rp['metrics']['prc_auc']:.3f}")
    print(f"MCC:                 {rp['metrics']['mcc']:.3f}")
    print(
        "Confusion matrix (TN, FP, FN, TP): "
        f"({rp['confusion_matrix']['tn']}, {rp['confusion_matrix']['fp']}, "
        f"{rp['confusion_matrix']['fn']}, {rp['confusion_matrix']['tp']})"
    )
    print(
        "Broad benchmark: "
        f"{rp['benchmark']['substantive_completed_models']} substantive completed models"
    )

    print("\n[NEXT] PHASE 3 — ROBUSTNESS, FEATURE ANALYSIS, AND REPRODUCIBILITY")
    print("- Information Gain and CFS/BestFirst feature-selection analysis")
    print("- Leakage-controlled feature-selection experiment")
    print("- RP-A vs RP-B timing / predictor-set comparison")
    print("- Python reproduction of the Research Participation model")
    print("- Descriptive-statistics and uncertainty analysis")

    print("\nSafety note: no experiment is run by the default command.")
    print("Use `python main.py run-year1` only when you intentionally want to reproduce Phase 1.")


def run_year1_pipeline(evaluate_test: bool = False) -> None:
    """Reproduce the existing coded Year-1 CGPA pipeline.

    The held-out Test evaluation is opt-in because the final model has already
    been frozen and evaluated. Running without ``--evaluate-test`` performs
    cleaning, splitting, and Train->Validation model selection only.
    """
    from src.config import (
        RAW_DATA,
        CLEAN_DATA,
        TRAIN_DATA,
        VALIDATION_DATA,
        TEST_DATA,
        RESULTS_DIR,
        MODELS_DIR,
    )
    from src.data import clean_raw_dataset, create_stratified_split
    from src.modeling import tune_random_forest

    print("=" * 78)
    print("YEAR-1 CGPA REPRODUCIBLE PIPELINE")
    print("=" * 78)

    print("\n[1/4] Cleaning raw dataset...")
    clean_df = clean_raw_dataset(RAW_DATA, CLEAN_DATA)
    print(f"Clean data: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")

    print("\n[2/4] Creating reproducible stratified 70/15/15 split...")
    train_df, val_df, test_df = create_stratified_split(
        CLEAN_DATA,
        TRAIN_DATA,
        VALIDATION_DATA,
        TEST_DATA,
    )
    print(f"Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    print("\n[3/4] Tuning RandomForest on Train -> Validation...")
    best, validation_results = tune_random_forest(train_df, val_df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(
        RESULTS_DIR / "validation_random_forest_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (RESULTS_DIR / "best_validation_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_rule": "maximize weighted_f1; tie-break by accuracy then kappa",
                "best_parameters": best["params"],
                "best_validation_metrics": {
                    k: float(v) for k, v in best["metrics"].items()
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Best validation parameters:", best["params"])
    print(f"Validation weighted F1: {best['metrics']['weighted_f1']:.4f}")
    print(f"Validation accuracy:    {best['metrics']['accuracy']:.4f}")

    if not evaluate_test:
        print("\n[4/4] Held-out Test evaluation NOT RUN.")
        print("The final Test result is already frozen in the project record.")
        print("To reproduce it intentionally, run:")
        print("  python main.py run-year1 --evaluate-test")
        return

    from src.evaluation import final_test_evaluation

    print("\n[4/4] Reproducing the frozen held-out Test evaluation...")
    final_metrics = final_test_evaluation(
        train_df,
        val_df,
        test_df,
        best["params"],
        RESULTS_DIR,
        MODELS_DIR,
    )

    print("\nFINAL HELD-OUT TEST RESULTS")
    for name, value in final_metrics.items():
        print(f"{name:26s}: {value:.4f}")

    print("\nPipeline complete.")
    print("Results:", RESULTS_DIR)
    print("Model:  ", MODELS_DIR / "final_random_forest_pipeline.joblib")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Academic Performance ML project launcher and reproducibility entry point."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show the current two-phase project status.")

    year1 = subparsers.add_parser(
        "run-year1",
        help="Reproduce the coded Year-1 CGPA cleaning/split/validation pipeline.",
    )
    year1.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Also reproduce the already-frozen final Test evaluation. "
            "Omit this flag during normal development."
        ),
    )

    subparsers.add_parser(
        "research-status",
        help="Show the completed Research Participation WEKA result without rerunning Test.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "status", "research-status"):
        print_status()
        return

    if args.command == "run-year1":
        run_year1_pipeline(evaluate_test=args.evaluate_test)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
