from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "academic_performance_raw.csv"
CLEAN_DATA = PROJECT_ROOT / "data" / "processed" / "final" / "E1_Year1CGPA_FinalClean.csv"

TRAIN_DATA = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALIDATION_DATA = PROJECT_ROOT / "data" / "splits" / "validation.csv"
TEST_DATA = PROJECT_ROOT / "data" / "splits" / "test.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "reproduction"
MODELS_DIR = PROJECT_ROOT / "models" / "reproduction"

TARGET = "Year1CGPA"
RANDOM_STATE = 1

# Locked only AFTER validation.
FINAL_RF_PARAMS = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
}
