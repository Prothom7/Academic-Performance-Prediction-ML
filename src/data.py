import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TARGET, RANDOM_STATE

STANDARD_COLUMNS = [
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

KEEP_COLUMN_INDEXES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 21, 28, 29, 30, 31,
]


def clean_raw_dataset(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Recreate the final 24-column Year1CGPA modeling table."""
    with raw_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        if len(raw_header) < 32:
            raise ValueError(f"Expected at least 32 raw columns, found {len(raw_header)}.")
        rows = [[row[i] for i in KEEP_COLUMN_INDEXES] for row in reader]

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with clean_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(STANDARD_COLUMNS)
        writer.writerows(rows)

    clean_df = pd.read_csv(clean_path, encoding="utf-8-sig")
    clean_df["HSCGraduationYear"] = pd.to_numeric(
        clean_df["HSCGraduationYear"], errors="raise"
    )
    return clean_df


def create_stratified_split(
    clean_path: Path,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the exact reproducible stratified 70/15/15 split used in the project."""
    with clean_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    labels = [row[TARGET] for row in rows]

    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    temp_labels = [row[TARGET] for row in temp_rows]
    validation_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    def save_csv(path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    save_csv(train_path, train_rows)
    save_csv(validation_path, validation_rows)
    save_csv(test_path, test_rows)

    return (
        pd.read_csv(train_path, encoding="utf-8-sig"),
        pd.read_csv(validation_path, encoding="utf-8-sig"),
        pd.read_csv(test_path, encoding="utf-8-sig"),
    )
