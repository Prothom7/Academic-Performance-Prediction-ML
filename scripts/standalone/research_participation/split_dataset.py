from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# RP-04
# CREATE AND LOCK RESEARCH PARTICIPATION SPLITS
# ============================================================


# ============================================================
# 1. LOCATE PROJECT ROOT
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

# Expected script location:
#
# Academic_Performance_ML_Complete/
#   scripts/
#     standalone/
#       research_participation/
#         split_dataset.py
#
# parents[3] = Academic_Performance_ML_Complete

ROOT = SCRIPT_PATH.parents[3]

if ROOT.name != "Academic_Performance_ML_Complete":
    raise RuntimeError(
        "\nProject root detection failed.\n"
        f"Detected root: {ROOT}\n"
        "Expected: Academic_Performance_ML_Complete\n"
    )


# ============================================================
# 2. INPUT / OUTPUT PATHS
# ============================================================

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "research_participation"
    / "ResearchParticipation_Binary_FinalClean.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "splits"
    / "research_participation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

AUDIT_PATH = OUTPUT_DIR / "RP04_SPLIT_AUDIT.txt"
MANIFEST_PATH = OUTPUT_DIR / "SPLIT_MANIFEST.json"
CHECKSUM_PATH = OUTPUT_DIR / "SHA256SUMS.txt"
TEST_LOCK_PATH = OUTPUT_DIR / "TEST_SET_LOCKED.txt"


# ============================================================
# 3. FILE HASH HELPER
# ============================================================

def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# 4. LOAD RP-03 CLEAN DATASET
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "\nRP-03 cleaned dataset was not found.\n"
        f"Expected:\n{INPUT_PATH}\n"
    )

df = pd.read_csv(INPUT_PATH)

print("=" * 78)
print("RP-04: CREATE FIXED TRAIN / VALIDATION / TEST SPLIT")
print("=" * 78)

print("\nProject root:")
print(ROOT)

print("\nInput dataset:")
print(INPUT_PATH)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 5. VALIDATE INPUT
# ============================================================

EXPECTED_SHAPE = (3475, 25)

if df.shape != EXPECTED_SHAPE:
    raise ValueError(
        "\nUnexpected RP-03 dataset shape.\n"
        f"Expected: {EXPECTED_SHAPE}\n"
        f"Found:    {df.shape}\n"
    )

TARGET = "ResearchParticipationBinary"

if TARGET not in df.columns:
    raise ValueError(
        f"\nTarget column not found: {TARGET}\n"
    )

if df[TARGET].isna().any():
    raise ValueError(
        "\nTarget contains missing values.\n"
    )

expected_classes = {
    "NotParticipated",
    "Participated",
}

actual_classes = set(
    df[TARGET].unique()
)

if actual_classes != expected_classes:
    raise ValueError(
        "\nUnexpected target classes.\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {actual_classes}\n"
    )


# ============================================================
# 6. DISPLAY ORIGINAL CLASS BALANCE
# ============================================================

print("\nFull dataset class distribution:")
print(
    df[TARGET].value_counts()
)

print("\nFull dataset class percentages:")
print(
    (
        df[TARGET]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# 7. FIRST STRATIFIED SPLIT
#
# 70% TRAIN
# 30% TEMPORARY
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=1,
    stratify=df[TARGET],
    shuffle=True,
)


# ============================================================
# 8. SECOND STRATIFIED SPLIT
#
# TEMPORARY 30%:
#   15% VALIDATION
#   15% TEST
# ============================================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=1,
    stratify=temp_df[TARGET],
    shuffle=True,
)


# ============================================================
# 9. RESET SPLIT INDICES
# ============================================================

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 10. VALIDATE SPLIT SIZES
# ============================================================

expected_sizes = {
    "Train": 2432,
    "Validation": 521,
    "Test": 522,
}

actual_sizes = {
    "Train": len(train_df),
    "Validation": len(validation_df),
    "Test": len(test_df),
}

if actual_sizes != expected_sizes:
    raise ValueError(
        "\nUnexpected split sizes.\n"
        f"Expected: {expected_sizes}\n"
        f"Found:    {actual_sizes}\n"
    )


# ============================================================
# 11. VALIDATE EXPECTED CLASS COUNTS
# ============================================================

expected_class_counts = {
    "Train": {
        "NotParticipated": 1567,
        "Participated": 865,
    },
    "Validation": {
        "NotParticipated": 336,
        "Participated": 185,
    },
    "Test": {
        "NotParticipated": 336,
        "Participated": 186,
    },
}


def class_counts(dataframe):
    counts = dataframe[TARGET].value_counts()

    return {
        "NotParticipated": int(
            counts.get("NotParticipated", 0)
        ),
        "Participated": int(
            counts.get("Participated", 0)
        ),
    }


actual_class_counts = {
    "Train": class_counts(train_df),
    "Validation": class_counts(validation_df),
    "Test": class_counts(test_df),
}

if actual_class_counts != expected_class_counts:
    raise ValueError(
        "\nUnexpected stratified class counts.\n"
        f"Expected:\n{expected_class_counts}\n\n"
        f"Found:\n{actual_class_counts}\n"
    )


# ============================================================
# 12. ENSURE TOTAL ROW COUNT IS PRESERVED
# ============================================================

total_split_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

if total_split_rows != len(df):
    raise ValueError(
        "\nSplit rows do not sum to full dataset.\n"
        f"Full dataset: {len(df)}\n"
        f"Split total:  {total_split_rows}\n"
    )


# ============================================================
# 13. SAVE SPLITS
# ============================================================

train_df.to_csv(
    TRAIN_PATH,
    index=False,
    encoding="utf-8-sig",
)

validation_df.to_csv(
    VALIDATION_PATH,
    index=False,
    encoding="utf-8-sig",
)

test_df.to_csv(
    TEST_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. COMPUTE CHECKSUMS
# ============================================================

source_hash = sha256_file(INPUT_PATH)

train_hash = sha256_file(TRAIN_PATH)
validation_hash = sha256_file(VALIDATION_PATH)
test_hash = sha256_file(TEST_PATH)


# ============================================================
# 15. CREATE MACHINE-READABLE MANIFEST
# ============================================================

manifest = {
    "phase": "RP-04",
    "target": TARGET,
    "prediction_horizon": (
        "End of Year 1, predicting later "
        "research participation"
    ),
    "source_dataset": str(INPUT_PATH),
    "source_sha256": source_hash,
    "random_state": 1,
    "stratified": True,
    "split_protocol": {
        "stage_1": (
            "train_test_split(test_size=0.30, "
            "random_state=1, stratify=target)"
        ),
        "stage_2": (
            "split temporary set 50/50 using "
            "random_state=1 and stratification"
        ),
    },
    "rows": {
        "total": int(len(df)),
        "train": int(len(train_df)),
        "validation": int(len(validation_df)),
        "test": int(len(test_df)),
    },
    "class_counts": actual_class_counts,
    "files": {
        "train": {
            "path": str(TRAIN_PATH),
            "sha256": train_hash,
        },
        "validation": {
            "path": str(VALIDATION_PATH),
            "sha256": validation_hash,
        },
        "test": {
            "path": str(TEST_PATH),
            "sha256": test_hash,
        },
    },
    "test_policy": (
        "TEST IS LOCKED. Do not use for model selection, "
        "feature selection, hyperparameter tuning, "
        "threshold tuning, or exploratory evaluation. "
        "Evaluate once after the model is fully locked."
    ),
}

with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 16. SAVE CHECKSUM FILE
# ============================================================

checksum_text = (
    f"{source_hash}  "
    "ResearchParticipation_Binary_FinalClean.csv\n"

    f"{train_hash}  train.csv\n"

    f"{validation_hash}  validation.csv\n"

    f"{test_hash}  test.csv\n"
)

CHECKSUM_PATH.write_text(
    checksum_text,
    encoding="utf-8",
)


# ============================================================
# 17. CREATE EXPLICIT TEST-LOCK NOTICE
# ============================================================

test_lock_text = """
RESEARCH PARTICIPATION FINAL TEST SET
=====================================

STATUS: LOCKED

File:
test.csv

Rows:
522

Purpose:
One-time final evaluation only.

DO NOT USE THIS FILE FOR:
- WEKA broad model comparison
- feature selection
- hyperparameter tuning
- model-family selection
- preprocessing decisions based on performance
- threshold tuning
- exploratory performance inspection

Allowed use:
Only after the complete model configuration has been
selected and locked using Train and Validation data.

Random state:
1

Prediction horizon:
End of Year 1, predicting later research participation.

IMPORTANT:
Do not open Test results simply to see how well a
development model performs. Doing so would weaken the
independence of the final evaluation.
""".strip()

TEST_LOCK_PATH.write_text(
    test_lock_text,
    encoding="utf-8",
)


# ============================================================
# 18. CREATE HUMAN-READABLE AUDIT
# ============================================================

def percentage(part, total):
    return round(
        (part / total) * 100,
        2
    )


audit_text = f"""
RP-04 FIXED STRATIFIED DATA SPLIT
=================================

SOURCE
------
{INPUT_PATH}

Source SHA256:
{source_hash}


DATASET
-------
Rows:
{len(df)}

Columns:
{df.shape[1]}

Target:
{TARGET}


SPLIT DESIGN
------------
Random state:
1

Stratification:
Yes

Train:
{len(train_df)} rows
{percentage(len(train_df), len(df))}%

Validation:
{len(validation_df)} rows
{percentage(len(validation_df), len(df))}%

Test:
{len(test_df)} rows
{percentage(len(test_df), len(df))}%


CLASS COUNTS
------------

Train
-----
NotParticipated:
{actual_class_counts["Train"]["NotParticipated"]}

Participated:
{actual_class_counts["Train"]["Participated"]}


Validation
----------
NotParticipated:
{actual_class_counts["Validation"]["NotParticipated"]}

Participated:
{actual_class_counts["Validation"]["Participated"]}


Test
----
NotParticipated:
{actual_class_counts["Test"]["NotParticipated"]}

Participated:
{actual_class_counts["Test"]["Participated"]}


OUTPUT FILES
------------
Train:
{TRAIN_PATH}

Validation:
{VALIDATION_PATH}

Test:
{TEST_PATH}

Manifest:
{MANIFEST_PATH}

Checksums:
{CHECKSUM_PATH}

Test lock notice:
{TEST_LOCK_PATH}


SHA256
------
Train:
{train_hash}

Validation:
{validation_hash}

Test:
{test_hash}


TEST POLICY
-----------
The 522-row Test set is now LOCKED.

It must not be used for:

- broad model benchmarking
- model-family selection
- feature selection
- hyperparameter tuning
- preprocessing decisions based on performance
- threshold tuning
- exploratory test evaluation

Model development must use Train and Validation only.

The Test set will be evaluated once after the final
model configuration has been locked.


STATUS
------
RP-04 completed.

No model was trained.
No Test performance was inspected.
""".strip()

AUDIT_PATH.write_text(
    audit_text,
    encoding="utf-8",
)


# ============================================================
# 19. PRINT SPLIT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("SPLIT RESULTS")
print("=" * 78)

for name, split_df in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    print(f"\n{name}")
    print("-" * 30)

    print(f"Rows: {len(split_df)}")

    print(
        split_df[TARGET]
        .value_counts()
    )

    print("\nPercentages:")

    print(
        (
            split_df[TARGET]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )


# ============================================================
# 20. SUCCESS SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("RP-04 COMPLETED SUCCESSFULLY")
print("=" * 78)

print("\nTrain:")
print(TRAIN_PATH)

print("\nValidation:")
print(VALIDATION_PATH)

print("\nTest:")
print(TEST_PATH)

print("\nManifest:")
print(MANIFEST_PATH)

print("\nTest lock notice:")
print(TEST_LOCK_PATH)

print("\nIMPORTANT:")
print(
    "The 522-row Test split is now LOCKED."
)

print(
    "Do not use Test during WEKA experiments, "
    "feature selection, or tuning."
)

print(
    "Development must use Train and Validation only."
)

print("\nNo machine-learning model was trained.")