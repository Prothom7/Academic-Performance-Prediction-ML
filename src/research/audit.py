"""Input/output hash, link, preservation, schema and publication-text checks."""
import json
import re
from pathlib import Path
import pandas as pd
from .common import *

def verify_inputs():
    manifest=ROOT/'data/DATA_MANIFEST.csv'
    if not manifest.exists():
        raise FileNotFoundError('Required pre-inspected data manifest missing; do not silently rebaseline changed data.')
    inputs=pd.read_csv(manifest)
    for row in inputs.itertuples():
        path=ROOT/row.path
        if not path.is_file() or sha256(path)!=row.sha256 or path.stat().st_size!=row.bytes:
            raise AssertionError('Input integrity failure: '+row.path)
    baseline=json.loads((ROOT/'config/repository_baseline.json').read_text(encoding='utf-8'))
    originals={r['path']:r['sha256'] for r in baseline['original_tracked_inventory']}
    for row in baseline['protected_artifacts']:
        accepted={row['sha256']}
        # The pre-existing presentation change is explicitly outside the commit.
        # A fresh clone therefore legitimately has its original committed bytes.
        if row.get('preexisting_user_modified'):
            accepted.add(originals[row['path']])
        if sha256(ROOT/row['path']) not in accepted:
            raise AssertionError('Preserved historical artifact modified: '+row['path'])
    return len(inputs)+len(baseline['protected_artifacts'])

def generated_docs():
    return [ROOT/'README.md']+[DOCS/name for name in [
        'DATA_DICTIONARY.md','RESEARCH_OVERVIEW.md','METHODOLOGY.md','MASTER_EVIDENCE_TABLE.md',
        'FINDINGS.md','REFERENCE_PAPER_COMPARISON.md','LIMITATIONS.md','REPRODUCIBILITY.md',
        'DECISION_LOG.md','CHANGELOG_RESEARCH.md','FULL_RESEARCH_REPORT.md','FIGURE_INDEX.md',
        'HISTORICAL_DISCREPANCIES.md','REPRODUCIBILITY_AUDIT.md']]

def verify(check_result_hashes=False,allow_pending_audit=False):
    count=verify_inputs()
    required=['dataset_summary','categorical_distributions','numeric_distributions',
        'transition_counts','transition_row_percent','trajectory_distribution','persistence_statistics',
        'master_unadjusted_associations','spearman_associations','adjusted_year1','adjusted_year2',
        'adjusted_improvement','adjusted_decline','adjusted_research','adjusted_model_diagnostics',
        'adjusted_factor_tests','model_specifications','adjusted_marginal_probabilities',
        'interaction_tests','interaction_coefficients','interaction_marginal_probabilities',
        'descriptive_contrasts','subgroup_persistence','master_evidence_table','verified_conclusions',
        'historical_checkpoint_comparison','figure_catalog','ml_test_metrics','year2_added_information']
    for name in required:
        for ext in ['csv','md']:
            path=TABLES/f'{name}.{ext}'
            assert path.is_file() and path.stat().st_size>0,str(path)
            count+=1
    catalog=pd.read_csv(TABLES/'figure_catalog.csv')
    assert len(catalog)==21,'Expected 21 publication figure concepts'
    for row in catalog.itertuples():
        for field in ['png','pdf','svg']:
            assert (ROOT/getattr(row,field)).is_file(),getattr(row,field)
            count+=1
        for path in row.source_tables.split('; '):
            assert (ROOT/path).is_file(),path
            count+=1
    for path in generated_docs():
        if allow_pending_audit and path.name=='REPRODUCIBILITY_AUDIT.md':
            continue
        assert path.is_file(),str(path)
        text=path.read_text(encoding='utf-8')
        # Ignore code blocks when checking Markdown links. Network links are
        # source citations, not local artifact references.
        without_code=re.sub(r'```.*?```','',text,flags=re.S)
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',without_code):
            if link.startswith(('http:','https:','mailto:','#')):
                continue
            target=link.split('#')[0].strip('<>')
            if allow_pending_audit and Path(target).name=='REPRODUCIBILITY_AUDIT.md':
                continue
            assert (path.parent/target).resolve().exists(),f'Broken link in {relative(path)}: {link}'
            count+=1
        assert not re.search(r'\b(caused|led to|improved because of|best university|worst university|lower-quality university)\b',without_code,re.I),str(path)
        count+=1
    evidence=pd.read_csv(TABLES/'master_evidence_table.csv')
    for field in ['supporting_script','supporting_table','unadjusted_table','supporting_figure']:
        for link in evidence[field].unique():
            assert (ROOT/link).is_file(),link
            count+=1
    for path in [*TABLES.glob('adjusted_*.csv')]:
        frame=pd.read_csv(path)
        if {'odds_ratio','CI_low','CI_high'}.issubset(frame):
            assert (frame.CI_low<=frame.odds_ratio).all() and (frame.odds_ratio<=frame.CI_high).all(),str(path)
            count+=1
    # Scan all versioned text plus intended new source/docs/config outputs. Never
    # print matching secret values; report only the offending filename.
    tracked=(git('ls-files') or '').splitlines()
    added=[relative(p) for base in [ROOT/'src/research',ROOT/'scripts',ROOT/'tests',ROOT/'config',DOCS,TABLES,METRICS,LOGS] for p in base.rglob('*') if p.is_file()]
    secret=re.compile(r'(?:sk-[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for name in sorted(set(tracked+added)):
        path=ROOT/name
        if path.suffix.lower() not in ['.py','.md','.txt','.json','.csv','.yml','.yaml','.cff','.sh','.bat']:
            continue
        text=path.read_text(encoding='utf-8',errors='replace')
        assert not secret.search(text),'Possible secret in '+name
        count+=1
    if check_result_hashes:
        manifest=pd.read_csv(ROOT/'results/result_manifest.csv')
        for row in manifest.itertuples():
            assert (ROOT/row.artifact).is_file(),row.artifact
            assert sha256(ROOT/row.artifact)==row.hash,'Result hash mismatch: '+row.artifact
            count+=1
        run=json.loads((ROOT/'results/run_manifest.json').read_text(encoding='utf-8'))
        assert run['status']=='completed; tests and audit passed','Run is incomplete or failed'
        for path in run['generated_artifact_list']:
            assert (ROOT/path).is_file(),path
            count+=1
    tables=len(list(TABLES.glob('*.csv')))
    figures=len(list(FIGURES.glob('*.png')))
    return dict(checks=count,tables=tables,figures=figures,status='passed')
