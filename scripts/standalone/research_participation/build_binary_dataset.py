from pathlib import Path
import pandas as pd


# ============================================================
# RP-03
# BUILD RESEARCH PARTICIPATION BINARY DATASET
# ============================================================


# ============================================================
# 1. LOCATE PROJECT ROOT
# ============================================================

# This script is expected at:
#
# Academic_Performance_ML_Complete/
#   scripts/
#     standalone/
#       research_participation/
#         build_binary_dataset.py
#
# Therefore:
# parents[0] = research_participation
# parents[1] = standalone
# parents[2] = scripts
# parents[3] = Academic_Performance_ML_Complete

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]

if ROOT.name != "Academic_Performance_ML_Complete":
    raise RuntimeError(
        "\nProject root detection failed.\n"
        f"Detected root: {ROOT}\n"
        "Expected folder name: Academic_Performance_ML_Complete\n"
    )

RAW_PATH = (
    ROOT
    / "data"
    / "raw"
    / "academic_performance_raw.csv"
)

if not RAW_PATH.exists():
    raise FileNotFoundError(
        "\nRaw dataset was not found.\n"
        f"Expected path:\n{RAW_PATH}\n"
    )


# ============================================================
# 2. DEFINE OUTPUT LOCATIONS
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "research_participation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = (
    OUTPUT_DIR
    / "ResearchParticipation_Binary_FinalClean.csv"
)

ARFF_PATH = (
    OUTPUT_DIR
    / "ResearchParticipation_Binary_FinalClean.arff"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "RP03_DATASET_AUDIT.txt"
)

DUPLICATE_AUDIT_PATH = (
    OUTPUT_DIR
    / "RP03_EXACT_DUPLICATE_GROUPS.csv"
)


# ============================================================
# 3. LOAD RAW DATASET
# ============================================================

df = pd.read_csv(RAW_PATH)

print("=" * 78)
print("RP-03: RESEARCH PARTICIPATION BINARY DATASET")
print("=" * 78)

print("\nProject root:")
print(ROOT)

print("\nRaw dataset:")
print(RAW_PATH)

print("\nRaw shape:")
print(df.shape)


# The raw project dataset was audited as 3516 x 32.
if df.shape != (3516, 32):
    raise ValueError(
        "\nUnexpected raw dataset dimensions.\n"
        f"Expected: (3516, 32)\n"
        f"Found:    {df.shape}\n"
    )


# ============================================================
# 4. DETECT RESEARCH PARTICIPATION TARGET
# ============================================================

# The real CSV header contains spaces/newlines and Bangla text,
# so we identify it using the stable English portion instead
# of manually reproducing the complete header.

target_candidates = [
    col
    for col in df.columns
    if "Participation in Research Projects" in str(col)
]

if len(target_candidates) != 1:
    raise ValueError(
        "\nCould not uniquely identify the Research Participation "
        "target column.\n"
        f"Candidates found: {target_candidates}\n"
    )

target_col = target_candidates[0]

print("\nDetected ResearchParticipation column:")
print(repr(target_col))


# ============================================================
# 5. RAW TARGET AUDIT
# ============================================================

raw_rows = len(df)

missing_target_count = int(
    df[target_col].isna().sum()
)

raw_duplicate_copies = int(
    df.duplicated(keep="first").sum()
)

print("\nRaw ResearchParticipation distribution:")
print(
    df[target_col]
    .value_counts(dropna=False)
)

print("\nMissing ResearchParticipation rows:")
print(missing_target_count)

print("\nExact duplicate copies in raw dataset:")
print(raw_duplicate_copies)


# ============================================================
# 6. REMOVE ROWS WITHOUT A TARGET
# ============================================================

labeled = (
    df.loc[df[target_col].notna()]
    .copy()
)

rows_after_missing_removal = len(labeled)

print("\nRows after removing missing target:")
print(rows_after_missing_removal)


if rows_after_missing_removal != 3479:
    raise ValueError(
        "\nUnexpected labeled-row count.\n"
        f"Expected: 3479\n"
        f"Found:    {rows_after_missing_removal}\n"
    )


# ============================================================
# 7. AUDIT EXACT DUPLICATES
# ============================================================

# Find every member of every exact duplicate group.
# This file is saved only as evidence.
#
# It contains BOTH the copy we keep and the duplicate copy
# that will be removed.

duplicate_group_mask = labeled.duplicated(
    keep=False
)

duplicate_groups = (
    labeled.loc[duplicate_group_mask]
    .copy()
)

if len(duplicate_groups) > 0:
    duplicate_groups.insert(
        0,
        "OriginalPandasIndex",
        duplicate_groups.index
    )

    duplicate_groups.to_csv(
        DUPLICATE_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig"
    )


# Now identify only copies that should actually be removed.

duplicate_copy_mask = labeled.duplicated(
    keep="first"
)

removed_duplicate_indices = (
    labeled.index[duplicate_copy_mask]
    .tolist()
)

duplicate_copies_removed = len(
    removed_duplicate_indices
)

print("\nExact duplicate copies to remove:")
print(duplicate_copies_removed)

print("\nDuplicate pandas indices removed:")
print(removed_duplicate_indices)


# ============================================================
# 8. REMOVE DUPLICATE COPIES
# ============================================================

working = (
    labeled.loc[~duplicate_copy_mask]
    .copy()
)

rows_after_duplicate_removal = len(working)

print("\nRows after duplicate removal:")
print(rows_after_duplicate_removal)


# Our RP-01 audit found four exact duplicate copies.
if duplicate_copies_removed != 4:
    raise ValueError(
        "\nDuplicate count differs from RP-01 audit.\n"
        f"Expected duplicate copies removed: 4\n"
        f"Found: {duplicate_copies_removed}\n"
    )

if rows_after_duplicate_removal != 3475:
    raise ValueError(
        "\nUnexpected final row count after duplicate removal.\n"
        f"Expected: 3475\n"
        f"Found:    {rows_after_duplicate_removal}\n"
    )


# ============================================================
# 9. SELECT THE 24 LEAKAGE-SAFE PREDICTORS
# ============================================================

# Prediction horizon:
#
# End of Year 1, predicting later research participation.
#
# Included:
#   Raw columns 2-19
#   Raw columns 21-22
#   Raw columns 29-32
#
# Excluded:
#   1  Timestamp
#   20 Year2CGPA
#   23 Living arrangement
#   24 Part-time employment
#   25 Current university expenditure
#   26 Current financial support
#   27 Student club membership
#   28 ResearchParticipation target
#
# Python indexes are zero-based.

safe_positions = (
    list(range(1, 19))
    + [20, 21]
    + [28, 29, 30, 31]
)

safe_columns = [
    df.columns[index]
    for index in safe_positions
]


clean_column_names = [
    "Faculty",
    "CurrentInstitution",
    "HSCGraduationYear",
    "Gender",
    "CollegeCategory",
    "CollegeLocation",
    "HSCLearningSource",
    "CollegeAttendance",
    "HSCMath",
    "HSCPhysics",
    "HSCChemistry",
    "FamilyMembers",
    "BirthOrder",
    "HouseholdIncomeYearly",
    "HSCMonthlyExpenditure",
    "InternetAvailability",
    "ComputerAvailability",
    "Year1CGPA",
    "UniversityPreferenceOrder",
    "DepartmentPreferenceOrder",
    "FatherEducation",
    "MotherEducation",
    "FatherEmploymentSector",
    "MotherEmploymentSector",
]


if len(safe_columns) != 24:
    raise ValueError(
        f"Expected 24 safe predictor columns, "
        f"found {len(safe_columns)}"
    )

if len(clean_column_names) != 24:
    raise ValueError(
        f"Expected 24 cleaned column names, "
        f"found {len(clean_column_names)}"
    )


# ============================================================
# 10. BUILD CLEAN PREDICTOR DATAFRAME
# ============================================================

clean = (
    working.loc[:, safe_columns]
    .copy()
)

clean.columns = clean_column_names


# ============================================================
# 11. CREATE BINARY TARGET
# ============================================================

def map_research_participation(value):
    """
    Map every original Yes category to Participated
    and No to NotParticipated.
    """

    normalized = (
        str(value)
        .strip()
        .lower()
    )

    if normalized.startswith("yes"):
        return "Participated"

    if normalized.startswith("no"):
        return "NotParticipated"

    raise ValueError(
        f"Unexpected ResearchParticipation value: {value!r}"
    )


clean["ResearchParticipationBinary"] = (
    working[target_col]
    .apply(map_research_participation)
    .to_numpy()
)


# ============================================================
# 12. FINAL DATA VALIDATION
# ============================================================

target_counts = (
    clean["ResearchParticipationBinary"]
    .value_counts()
)

target_percentages = (
    clean["ResearchParticipationBinary"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "=" * 78)
print("FINAL RP-03 DATASET CHECK")
print("=" * 78)

print("\nFinal shape:")
print(clean.shape)

print("\nFinal binary target counts:")
print(target_counts)

print("\nFinal binary target percentages:")
print(target_percentages)

print("\nMissing values in final dataset:")

final_missing = (
    clean.isna()
    .sum()
)

missing_only = (
    final_missing[
        final_missing > 0
    ]
)

if len(missing_only) == 0:
    print("None")
else:
    print(missing_only)


# ============================================================
# 13. HARD VALIDATION CHECKS
# ============================================================

if clean.shape != (3475, 25):
    raise ValueError(
        "\nUnexpected final dataset shape.\n"
        f"Expected: (3475, 25)\n"
        f"Found:    {clean.shape}\n"
    )

if (
    target_counts.get(
        "Participated",
        0
    )
    != 1236
):
    raise ValueError(
        "\nUnexpected Participated count.\n"
        f"Expected: 1236\n"
        f"Found: "
        f"{target_counts.get('Participated', 0)}\n"
    )

if (
    target_counts.get(
        "NotParticipated",
        0
    )
    != 2239
):
    raise ValueError(
        "\nUnexpected NotParticipated count.\n"
        f"Expected: 2239\n"
        f"Found: "
        f"{target_counts.get('NotParticipated', 0)}\n"
    )


# ============================================================
# 14. SAVE FINAL CSV
# ============================================================

clean.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved CSV:")
print(CSV_PATH)


# ============================================================
# 15. WEKA-COMPATIBLE ARFF HELPERS
# ============================================================

def quote_arff_value(value):
    """
    Quote categorical values so commas, spaces,
    Bangla text, etc. are safe inside ARFF files.
    """

    value = str(value)

    value = value.replace(
        "\\",
        "\\\\"
    )

    value = value.replace(
        '"',
        '\\"'
    )

    value = value.replace(
        "\r",
        " "
    )

    value = value.replace(
        "\n",
        " "
    )

    return f'"{value}"'


# ============================================================
# 16. SAVE WEKA ARFF
# ============================================================

with open(
    ARFF_PATH,
    "w",
    encoding="utf-8"
) as arff:

    arff.write(
        "@RELATION "
        "ResearchParticipation_Binary_FinalClean"
        "\n\n"
    )

    for column in clean.columns:

        # Keep the same main preprocessing philosophy
        # used in the completed Year-1 pipeline:
        # HSCGraduationYear is numeric and other fields
        # remain categorical/nominal.

        if column == "HSCGraduationYear":

            arff.write(
                f"@ATTRIBUTE {column} NUMERIC\n"
            )

        else:

            unique_values = (
                clean[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            nominal_definition = ",".join(
                quote_arff_value(value)
                for value in unique_values
            )

            arff.write(
                f"@ATTRIBUTE {column} "
                f"{{{nominal_definition}}}\n"
            )

    arff.write("\n@DATA\n")

    for _, row in clean.iterrows():

        output_values = []

        for column in clean.columns:

            value = row[column]

            if pd.isna(value):

                output_values.append("?")

            elif column == "HSCGraduationYear":

                output_values.append(
                    str(value)
                )

            else:

                output_values.append(
                    quote_arff_value(value)
                )

        arff.write(
            ",".join(output_values)
            + "\n"
        )


print("\nSaved ARFF:")
print(ARFF_PATH)


# ============================================================
# 17. SAVE RP-03 AUDIT RECORD
# ============================================================

audit_text = f"""
RP-03 RESEARCH PARTICIPATION BINARY DATASET
============================================

PROJECT ROOT
------------
{ROOT}

RAW DATASET
-----------
{RAW_PATH}

Raw shape:
{df.shape}

Raw rows:
{raw_rows}

Raw columns:
{df.shape[1]}


TARGET
------
Original target column:
{repr(target_col)}

Missing target rows removed:
{missing_target_count}

Rows after missing-target removal:
{rows_after_missing_removal}


DUPLICATE AUDIT
---------------
Exact duplicate copies in raw dataset:
{raw_duplicate_copies}

Exact labeled duplicate copies removed:
{duplicate_copies_removed}

Removed pandas indices:
{removed_duplicate_indices}

Rows after duplicate removal:
{rows_after_duplicate_removal}

Duplicate evidence file:
{DUPLICATE_AUDIT_PATH}


PREDICTION HORIZON
------------------
Prediction is made at the end of Year 1,
using information reasonably available by that point,
to predict later ResearchParticipation.


PRIMARY FEATURE SET
-------------------
Safe predictors:
24

Included predictors:
{chr(10).join("- " + col for col in clean_column_names)}


EXCLUDED FROM PRIMARY MODEL
---------------------------
- Timestamp
- Year2CGPA
- CurrentLivingArrangement
- PartTimeEmployment
- UniversityMonthlyExpenditure
- FinancialSupportSource
- StudentClubMember
- Original ResearchParticipation field


BINARY TARGET
-------------
Target name:
ResearchParticipationBinary

Participated:
{target_counts.get("Participated", 0)}

NotParticipated:
{target_counts.get("NotParticipated", 0)}


FINAL DATASET
-------------
Rows:
{clean.shape[0]}

Predictors:
24

Target:
1

Total columns:
{clean.shape[1]}

Final shape:
{clean.shape}


MISSING PREDICTOR VALUES
------------------------
{missing_only.to_string() if len(missing_only) > 0 else "None"}


OUTPUT FILES
------------
CSV:
{CSV_PATH}

ARFF:
{ARFF_PATH}

Audit:
{AUDIT_PATH}

Duplicate evidence:
{DUPLICATE_AUDIT_PATH}


IMPORTANT
---------
The completed Year-1 CGPA datasets, splits,
models, experiments, and results were not modified.

The ResearchParticipation Test split has NOT
been created yet.

No machine-learning model has been trained
during RP-03.
""".strip()


AUDIT_PATH.write_text(
    audit_text,
    encoding="utf-8"
)

print("\nSaved audit:")
print(AUDIT_PATH)


# ============================================================
# 18. SUCCESS SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("RP-03 COMPLETED SUCCESSFULLY")
print("=" * 78)

print("\nFinal dataset:")
print(f"Rows       : {clean.shape[0]}")
print("Predictors : 24")
print("Target     : 1")
print(f"Columns    : {clean.shape[1]}")

print("\nClass distribution:")
print(target_counts)

print("\nNo model was trained.")
print("The final Test split has not been created yet.")