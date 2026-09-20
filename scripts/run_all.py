"""Single portable entry point: recompute, test and audit the entire project."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Keep numerical linear algebra deterministic and avoid nested thread pools.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['SM_FORMULA_ENGINE']='patsy'

def main():
    from src.research.pipeline import run_all
    run_all()

if __name__=='__main__':
    main()
