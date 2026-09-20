from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.common import load_population

@pytest.fixture(scope='session')
def population():
    return load_population()
