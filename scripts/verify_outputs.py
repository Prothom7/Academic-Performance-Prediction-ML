"""Verify existing outputs and input hashes without refitting any model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.audit import verify

if __name__=='__main__':
    result=verify(check_result_hashes=True)
    print(f'PASS: {result["checks"]} audit checks; {result["tables"]} CSV tables; {result["figures"]} PNG figures.')
