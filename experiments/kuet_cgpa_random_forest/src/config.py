"""Read-only links to canonical definitions and local experiment settings."""
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE = REPOSITORY_ROOT / "data/processed/intermediate/academic_performance_clean_v3.csv"
SOURCE_SHA256 = "7cb8520f8d2b97a2f4baf13b93de8cc93ed7c7bcc26a99d4153eb027ea8bbc80"
OUTPUT_DATA = EXPERIMENT_ROOT / "data"
OUTPUT_RESULTS = EXPERIMENT_ROOT / "results"
OUTPUT_MODELS = EXPERIMENT_ROOT / "models"

# Import only the canonical, read-only feature and band definitions. The
# experiment never calls the canonical pipeline or writes to its directories.
sys.path.insert(0, str(REPOSITORY_ROOT))
from src.research.common import BASE_FEATURES, BANDS, SEED, cgpa_band  # noqa: E402

FEATURES = tuple(feature for feature in BASE_FEATURES if feature != "CurrentInstitution")
TARGETS = {"year1": "Year1CGPA", "year2": "Year2CGPA"}
CLASSES = tuple(range(5))
EXPECTED_SOURCE_SHAPE = (3516, 32)
EXPECTED_KUET_N = 1377
SPLIT_SIZES = {"train": 963, "validation": 207, "test": 207}

assert SEED == 1
assert len(BASE_FEATURES) == 23 and len(FEATURES) == 22
assert "CurrentInstitution" not in FEATURES
assert not set(FEATURES) & {"source_row_id", "Year1CGPA", "Year2CGPA", "ResearchParticipation"}
