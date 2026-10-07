"""Verify the source, retain source row IDs, and make independent splits."""
import hashlib
import json
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (BANDS, CLASSES, EXPECTED_KUET_N, EXPECTED_SOURCE_SHAPE,
                    FEATURES, OUTPUT_DATA, SEED, SOURCE, SOURCE_SHA256,
                    SPLIT_SIZES, TARGETS, cgpa_band)


def normalized_institution(value):
    """Unicode-normalized, whitespace-trimmed, case-folded exact identifier."""
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def load_kuet():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Authoritative source missing: {SOURCE}")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Source SHA256 changed: {digest}; expected {SOURCE_SHA256}")
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    if source.shape != EXPECTED_SOURCE_SHAPE:
        raise ValueError(f"Source shape {source.shape}; expected {EXPECTED_SOURCE_SHAPE}")
    required = set(FEATURES) | set(TARGETS.values()) | {"CurrentInstitution"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source columns missing: {sorted(missing)}")
    mask = source.CurrentInstitution.map(normalized_institution).eq("kuet")
    kuet = source.loc[mask].copy()
    kuet.insert(0, "source_row_id", kuet.index.astype(int))
    kuet.reset_index(drop=True, inplace=True)
    if len(kuet) != EXPECTED_KUET_N or not kuet.source_row_id.is_unique:
        raise ValueError(f"KUET contract failed: N={len(kuet)}, unique IDs={kuet.source_row_id.is_unique}")
    if not kuet.CurrentInstitution.map(normalized_institution).eq("kuet").all():
        raise ValueError("KUET subset contains a different institution")
    for target in TARGETS.values():
        if kuet[target].isna().any():
            raise ValueError(f"Missing {target} in KUET subset")
        bands = kuet[target].map(cgpa_band)
        if set(bands) != set(CLASSES):
            raise ValueError(f"{target} does not contain exactly five canonical CGPA bands")
    return kuet


def band_distribution(frame, target, split):
    codes = frame[target].map(cgpa_band)
    counts = codes.value_counts().reindex(CLASSES, fill_value=0)
    return [{"split": split, "target": target, "band": int(band),
             "band_label": BANDS[band], "count": int(count),
             "proportion": float(count / len(frame)), "split_N": len(frame)}
            for band, count in counts.items()]


def make_splits(kuet, target):
    """Two stratified draws yield 70/15/15 with a sealed final test set."""
    ids = kuet.source_row_id.to_numpy()
    labels = kuet[target].map(cgpa_band).to_numpy()
    train_ids, temporary_ids = train_test_split(
        ids, test_size=0.30, random_state=SEED, stratify=labels)
    labels_by_id = pd.Series(labels, index=ids)
    validation_ids, test_ids = train_test_split(
        temporary_ids, test_size=0.50, random_state=SEED,
        stratify=labels_by_id.loc[temporary_ids].to_numpy())
    by_id = kuet.set_index("source_row_id", drop=False)
    frames = {name: by_id.loc[split_ids].reset_index(drop=True)
              for name, split_ids in (("train", train_ids),
                                      ("validation", validation_ids),
                                      ("test", test_ids))}
    verify_splits(kuet, frames, target)
    return frames


def verify_splits(kuet, frames, target):
    if set(frames) != set(SPLIT_SIZES):
        raise ValueError("Split names do not match train/validation/test")
    all_ids = []
    population = kuet[target].map(cgpa_band).value_counts(normalize=True).reindex(CLASSES)
    for name, frame in frames.items():
        if len(frame) != SPLIT_SIZES[name]:
            raise ValueError(f"{name} size {len(frame)}; expected {SPLIT_SIZES[name]}")
        if frame.source_row_id.duplicated().any() or frame[target].isna().any():
            raise ValueError(f"Duplicate row ID or missing target in {name}")
        distribution = frame[target].map(cgpa_band).value_counts(normalize=True).reindex(CLASSES, fill_value=0)
        if (distribution == 0).any() or (distribution - population).abs().max() > 0.02:
            raise ValueError(f"Class coverage/proportions failed in {name}")
        all_ids.extend(frame.source_row_id.tolist())
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(kuet.source_row_id):
        raise ValueError("Split overlap or incomplete KUET population coverage")


def write_data(kuet, splits):
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    kuet.to_csv(OUTPUT_DATA / "kuet_full.csv", index=False, lineterminator="\n")
    rows = []
    for study, target in TARGETS.items():
        rows.extend(band_distribution(kuet, target, "full"))
        folder = OUTPUT_DATA / study
        folder.mkdir(parents=True, exist_ok=True)
        split_rows = []
        for name, frame in splits[study].items():
            frame.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
            split_rows.extend(band_distribution(frame, target, name))
        pd.DataFrame(split_rows).to_csv(folder / "split_summary.csv", index=False, lineterminator="\n")
        summary_json = json.dumps({"target": target, "seed": SEED, "split_sizes": SPLIT_SIZES,
                                   "class_counts": split_rows}, ensure_ascii=False, indent=2) + "\n"
        (folder / "split_summary.json").write_bytes(summary_json.encode("utf-8"))
    pd.DataFrame(rows).to_csv(OUTPUT_DATA / "dataset_summary.csv", index=False, lineterminator="\n")
