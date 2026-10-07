"""Run both independent KUET Random Forest studies from the repository root."""
import json

import numpy as np
import pandas as pd

from config import FEATURES, OUTPUT_RESULTS, SEED, SOURCE_SHA256, TARGETS, cgpa_band
from evaluation import evaluate_and_save
from modeling import fit_model, validation_search
from prepare_data import load_kuet, make_splits, write_data


def main():
    kuet = load_kuet()
    splits = {study: make_splits(kuet, target) for study, target in TARGETS.items()}
    write_data(kuet, splits)
    results = {}
    for study, target in TARGETS.items():
        print(f"{study}: tuning 27 configurations on train/validation", flush=True)
        frames = splits[study]
        labels = {name: frame[target].map(cgpa_band).to_numpy(dtype=int)
                  for name, frame in frames.items()}
        grid, parameters = validation_search(
            frames["train"], frames["validation"], labels["train"], labels["validation"])
        development = pd.concat([frames["train"], frames["validation"]], ignore_index=True)
        development_labels = np.concatenate([labels["train"], labels["validation"]])
        final_model = fit_model(parameters, development, development_labels)
        results[study] = evaluate_and_save(
            study, target, frames, labels, final_model, parameters, grid)
        print(f"{study}: locked {parameters}; test accuracy {results[study]['accuracy']:.4f}", flush=True)

    OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame([{
        "study": study, "target": TARGETS[study], "KUET_N": len(kuet),
        "train_N": len(splits[study]["train"]),
        "validation_N": len(splits[study]["validation"]),
        **{key: value for key, value in results[study].items()
           if isinstance(value, (int, float)) and not isinstance(value, bool)}
    } for study in TARGETS])
    comparison.to_csv(OUTPUT_RESULTS / "summary.csv", index=False, lineterminator="\n")
    summary_json = json.dumps({
        "source_sha256": SOURCE_SHA256, "KUET_N": len(kuet), "seed": SEED,
        "features": list(FEATURES), "experiments": results,
        "interpretation": "Within-KUET held-out predictive comparison; no causal claim."
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (OUTPUT_RESULTS / "summary.json").write_bytes(summary_json.encode("utf-8"))
    print("\nFinal held-out comparison (independent test splits):")
    print(comparison[["study", "test_N", "accuracy", "weighted_f1", "macro_f1",
                      "majority_baseline_accuracy"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
