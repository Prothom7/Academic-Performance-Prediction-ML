"""
End-to-end academic performance classification pipeline.

Architecture:
Raw Survey
  -> Cleaning / leakage-aware feature removal
  -> Stratified 70/15/15 split
  -> One-hot encoding of categorical predictors
  -> RandomForest validation tuning
  -> Lock configuration
  -> Refit on Train+Validation
  -> One-time held-out Test evaluation
  -> Metrics + confusion matrix + saved model
"""

import json
import pandas as pd

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
from src.evaluation import final_test_evaluation

def main():
    print("=" * 72)
    print("ACADEMIC PERFORMANCE ML PIPELINE")
    print("=" * 72)

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
    print(
        f"Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}"
    )

    print("\n[3/4] Tuning RandomForest on Train -> Validation...")
    best, validation_results = tune_random_forest(train_df, val_df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(
        RESULTS_DIR / "validation_random_forest_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (RESULTS_DIR / "best_validation_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "selection_rule": (
                    "maximize weighted_f1; tie-break by accuracy then kappa"
                ),
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

    print("\n[4/4] Locking model and evaluating once on held-out Test...")
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

if __name__ == "__main__":
    main()
