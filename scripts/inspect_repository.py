"""Read-only repository inspection; explicit one-time baseline creation option.

Baseline creation is not called by run_all: changed inputs must never be silently
accepted. The baseline names the original commit and protected working artifacts.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research.common import sha256,relative,save_json,git

def inspect(create_baseline=False):
    original_commit='ba2b8ee'  # Inspected starting commit, not a result checkpoint.
    files=(git('ls-tree','-r','--name-only',original_commit) or '').splitlines()
    protected=[]
    original_inventory=[]
    exceptions={'.gitignore','requirements.txt','README.md','docs/DATA_DICTIONARY.md'}
    for name in files:
        path=ROOT/name
        if not path.exists():
            raise FileNotFoundError('Original artifact missing: '+name)
        blob=subprocess.run(['git','show',original_commit+':'+name],cwd=ROOT,capture_output=True,check=True).stdout
        original_inventory.append(dict(path=name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        if name not in exceptions:
            protected.append(dict(path=name,sha256=sha256(path),bytes=path.stat().st_size,
                preexisting_user_modified=name.startswith('presentation/') and sha256(path)!=hashlib.sha256(blob).hexdigest()))
    print(f'Original tracked inventory: {len(files)} files; protected artifacts: {len(protected)}')
    print('Current branch:',git('branch','--show-current'))
    print('Remotes:',git('remote','-v'))
    print('Recent history:',git('log','-5','--oneline'))
    print('Largest original artifact:',max(original_inventory,key=lambda r:r['bytes'])['bytes'],'bytes')
    if create_baseline:
        target=ROOT/'config/repository_baseline.json'
        if target.exists():
            raise RuntimeError('Baseline already exists; refusing to overwrite historical protection.')
        save_json(target,dict(starting_commit=original_commit,starting_branch='main',
            remote='https://github.com/Prothom7/Academic-Performance-Prediction-ML.git',
            original_tracked_inventory=original_inventory,protected_artifacts=protected,
            original_documentation_backups={'README.md':'archive/prior_documentation/README.md','docs/DATA_DICTIONARY.md':'archive/prior_documentation/DATA_DICTIONARY.md'},
            inspection_note='Starting tree, branch, status, logs, remotes, code, datasets, outputs, docs and environment inspected before analysis edits. Original object inventory reconstructed from inspected starting commit; pre-existing presentation change separately protected.'))
        records=[]
        for path in sorted((ROOT/'data').rglob('*')):
            if not path.is_file() or path.suffix not in ['.csv','.arff'] or path.name=='DATA_MANIFEST.csv':
                continue
            n,p=None,None
            if path.suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as stream:
                    reader=csv.reader(stream); header=next(reader); n=sum(1 for row in reader);p=len(header)
            records.append(dict(path=relative(path),sha256=sha256(path),bytes=path.stat().st_size,
                row_count=n,column_count=p,role='authoritative analytical population' if path.name=='academic_performance_clean_v3.csv' else 'preserved raw/intermediate/historical modeling artifact'))
        import pandas as pd
        pd.DataFrame(records).to_csv(ROOT/'data/DATA_MANIFEST.csv',index=False,lineterminator='\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--create-baseline',action='store_true')
    inspect(parser.parse_args().create_baseline)
