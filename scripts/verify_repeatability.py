"""Compare regenerated numerical bytes with an independent full-run baseline."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.common import sha256,save_json

def main():
    baseline=json.loads((ROOT/'config/numerical_repeatability_baseline.json').read_text(encoding='utf-8'))
    differences=[name for name,value in baseline.items() if sha256(ROOT/name)!=value]
    result=dict(compared_numeric_artifacts=len(baseline),byte_identical=len(baseline)-len(differences),
        differences=differences,excluded='Historical-checkpoint comparison extended to cover additional supplied raw rates and intervals between runs; engine timings/manifests/docs excluded.',
        status='passed' if not differences else 'failed')
    save_json(ROOT/'results/logs/numerical_repeatability.json',result)
    print(json.dumps(result,indent=2))
    if differences:
        raise SystemExit(1)

if __name__=='__main__':
    main()
