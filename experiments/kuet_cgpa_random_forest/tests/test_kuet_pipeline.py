"""Source, split, leakage and local fitting invariants for the KUET study."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
import joblib

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from config import (BANDS, CLASSES, EXPECTED_KUET_N, EXPECTED_SOURCE_SHAPE,
                    FEATURES, OUTPUT_DATA, OUTPUT_MODELS, OUTPUT_RESULTS, SOURCE,
                    SOURCE_SHA256, SPLIT_SIZES, TARGETS, cgpa_band)
from modeling import GRID, fit_model, make_pipeline
from prepare_data import load_kuet, make_splits, normalized_institution, verify_splits


@pytest.fixture(scope="module")
def kuet():
    return load_kuet()


@pytest.fixture(scope="module", params=list(TARGETS))
def study(request, kuet):
    name = request.param
    target = TARGETS[name]
    return name, target, make_splits(kuet, target)


def test_source_contract_and_identity():
    import hashlib
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    assert source.shape == EXPECTED_SOURCE_SHAPE
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    assert set(FEATURES).issubset(source.columns)


def test_exact_kuet_filter_and_row_traceability(kuet):
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    expected = source.index[source.CurrentInstitution.map(normalized_institution).eq("kuet")]
    assert len(kuet) == EXPECTED_KUET_N == len(expected)
    assert kuet.source_row_id.is_unique
    assert kuet.source_row_id.tolist() == expected.tolist()
    assert (kuet.CurrentInstitution == "KUET").all()
    assert normalized_institution(" kuet ") == "kuet"
    assert normalized_institution("KUET-other") != "kuet"
    assert kuet.drop(columns="source_row_id").reset_index(drop=True).equals(
        source.loc[expected].reset_index(drop=True))


@pytest.mark.parametrize("target", list(TARGETS.values()))
def test_targets_complete_and_five_classes(kuet, target):
    assert not kuet[target].isna().any()
    assert set(kuet[target].map(cgpa_band)) == set(CLASSES)


def test_canonical_cgpa_mapping():
    raw_labels = ("Less than 3.00", "3.00 - 3.25", "3.25 - 3.50",
                  "3.50 - 3.75", "Above 3.75")
    assert tuple(cgpa_band(label) for label in raw_labels) == CLASSES
    assert len(BANDS) == 5
    with pytest.raises(ValueError):
        cgpa_band("not a CGPA category")


def test_predictor_contract_and_no_leakage():
    assert len(FEATURES) == 22
    assert "CurrentInstitution" not in FEATURES
    assert "source_row_id" not in FEATURES
    assert "Year1CGPA" not in FEATURES and "Year2CGPA" not in FEATURES
    assert not set(FEATURES) & {"ResearchParticipation", "LivingArrangement",
                                "PartTimeEmployment", "UniversityMonthlyExpenditure",
                                "StudentClubMember", "Timestamp"}


def test_grid_is_27_train_only_candidates():
    assert len(GRID) == 27
    assert {p["n_estimators"] for p in GRID} == {100, 250, 500}
    assert {p["max_features"] for p in GRID} == {"sqrt", 2, 4}
    assert {p["min_samples_leaf"] for p in GRID} == {1, 2, 3}


def test_split_sizes_overlap_coverage_and_classes(study, kuet):
    _, target, frames = study
    verify_splits(kuet, frames, target)
    assert {name: len(frame) for name, frame in frames.items()} == SPLIT_SIZES
    ids = [set(frame.source_row_id) for frame in frames.values()]
    assert len(set.union(*ids)) == EXPECTED_KUET_N
    assert not any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))
    for frame in frames.values():
        assert set(frame[target].map(cgpa_band)) == set(CLASSES)
        assert frame.source_row_id.is_unique


def test_split_reproducibility_seed_one(study, kuet):
    _, target, first = study
    second = make_splits(kuet, target)
    for name in SPLIT_SIZES:
        np.testing.assert_array_equal(first[name].source_row_id, second[name].source_row_id)


def test_pipeline_trains_for_both_targets(study):
    _, target, frames = study
    sample = frames["train"]
    model = fit_model({"n_estimators": 10, "max_features": "sqrt",
                       "min_samples_leaf": 1}, sample, sample[target].map(cgpa_band))
    assert tuple(model.classes_) == CLASSES
    predictions = model.predict(frames["validation"].loc[:, FEATURES])
    assert len(predictions) == SPLIT_SIZES["validation"]
    assert set(predictions).issubset(CLASSES)
    assert model.named_steps["preprocess"].transformers_[1][2] == ["HSCGraduationYear"]


def test_saved_experiment_consistency(study):
    name, target, frames = study
    folder = OUTPUT_DATA / name
    for split in SPLIT_SIZES:
        saved = pd.read_csv(folder / f"{split}.csv", encoding="utf-8-sig")
        np.testing.assert_array_equal(saved.source_row_id, frames[split].source_row_id)
    summary = pd.read_csv(folder / "split_summary.csv")
    assert summary.groupby("split")["count"].sum().to_dict() == SPLIT_SIZES
    assert summary.groupby("split")["proportion"].sum().round(10).eq(1).all()
    output = json.loads((OUTPUT_RESULTS / name / "test_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(OUTPUT_RESULTS / name / "test_predictions.csv")
    assert output["target"] == target and output["test_N"] == SPLIT_SIZES["test"]
    assert predictions.source_row_id.tolist() == frames["test"].source_row_id.tolist()
    assert np.isclose((predictions.actual_band == predictions.predicted_band).mean(), output["accuracy"])
    assert np.allclose(predictions[[f"probability_{i}" for i in CLASSES]].sum(axis=1), 1)
    saved_model = joblib.load(OUTPUT_MODELS / f"{name}_random_forest.joblib")
    np.testing.assert_array_equal(saved_model.predict(frames["test"].loc[:, FEATURES]), predictions.predicted_band)
    np.testing.assert_allclose(saved_model.predict_proba(frames["test"].loc[:, FEATURES]),
                               predictions[[f"probability_{i}" for i in CLASSES]], atol=1e-11)
    grid = pd.read_csv(OUTPUT_RESULTS / name / "validation_grid.csv")
    assert len(grid) == 27 and grid.iloc[0]["rank"] == 1
    assert output["selected_parameters"] == json.loads(
        (OUTPUT_RESULTS / name / "best_config.json").read_text(encoding="utf-8"))["parameters"]
