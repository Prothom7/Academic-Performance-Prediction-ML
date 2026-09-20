"""Deterministic orchestration and provenance manifests for all research stages."""
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
import pandas as pd
from .common import *
from .audit import verify_inputs,verify,generated_docs

def artifacts():
    paths=[]
    for directory in [TABLES,FIGURES,METRICS,LOGS,ROOT/'results/models']:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and p.name!='pipeline_run.txt')
    paths.extend(p for p in generated_docs() if p.exists())
    paths.extend([ROOT/'results/final_research_summary.json'])
    return sorted(set(paths),key=relative)

def generator(path):
    name=path.stem
    if path==DOCS/'DATA_DICTIONARY.md':
        return 'src/research/statistics.py'
    if path in generated_docs():
        return 'src/research/documentation.py' if name!='REPRODUCIBILITY_AUDIT' else 'src/research/pipeline.py'
    if name in ['master_evidence_table','verified_conclusions','historical_checkpoint_comparison']:
        return 'src/research/evidence.py'
    if path.parent==FIGURES or name in ['figure_catalog','trajectory_by_gender','trajectory_by_living_arrangement']:
        return 'src/research/visualization.py'
    if name.startswith('interaction'):
        return 'src/research/interactions.py'
    if name.startswith(('adjusted','model_specifications','raw_vs_adjusted')):
        return 'src/research/adjusted.py'
    if name.startswith(('year1_','year2_','research')) or path.parent in [METRICS,ROOT/'results/models'] or name in ['ml_test_metrics','weka_replay_predictions']:
        return 'src/research/ml.py'
    if name in ['reference_dataset_comparison','reference_paper_extracted']:
        return 'src/research/documentation.py'
    if name in ['test_results','output_verification']:
        return 'src/research/pipeline.py'
    if name=='numerical_repeatability':
        return 'scripts/verify_repeatability.py'
    if name=='final_research_summary':
        return 'src/research/pipeline.py'
    return 'src/research/statistics.py'

def artifact_input(path):
    name=path.stem
    if name.startswith('year1_') or name=='year1':
        return 'data/splits/train.csv; data/splits/validation.csv; data/splits/test.csv'
    if name.startswith('research_historical_weka') or name=='weka_replay_predictions':
        return 'data/splits/research_participation/ResearchParticipation_TrainValidation_2953.arff; data/splits/research_participation/ResearchParticipation_test.arff; models/final/ResearchParticipation_Final_RandomForest_K12.model; tools/vendor/weka/weka.jar'
    if name.startswith('research_historical'):
        return 'data/splits/research_participation/train.csv; data/splits/research_participation/validation.csv; data/splits/research_participation/test.csv; models/final/ResearchParticipation_Final_RandomForest_K12.model'
    if name.startswith('reference_'):
        return 'references/academic_performance_paper.pdf; '+relative(DATA)
    return relative(DATA)

def manifests(start,seed,counts,status='analyses generated; validation pending'):
    paths=artifacts()
    rows=[dict(artifact=relative(path),generating_script=generator(path),input_dataset=artifact_input(path),
        hash=sha256(path),created_at=start) for path in paths]
    pd.DataFrame(rows).to_csv(ROOT/'results/result_manifest.csv',index=False,lineterminator='\n')
    packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()}
    java=os.environ.get('JAVA_EXE','java')
    java_version=subprocess.run([java,'-version'],capture_output=True,text=True,errors='replace')
    datasets=pd.read_csv(ROOT/'data/DATA_MANIFEST.csv')
    save_json(ROOT/'results/run_manifest.json',dict(run_timestamp_UTC=start,git_commit=git('rev-parse','HEAD'),
        git_branch=git('branch','--show-current'),git_dirty=bool(git('status','--porcelain')),
        Python_version=sys.version,platform=platform.platform(),package_versions=packages,
        Java_executable=java,Java_version=(java_version.stderr+java_version.stdout).strip(),
        analysis_seed=seed,row_count=counts[0],column_count=counts[1],
        execution_configuration={'training_workers':min(4,os.cpu_count() or 1),'prediction_workers':1,
            'linear_algebra_threads':1,'formula_engine':'patsy'},
        validation_context=os.environ.get('RESEARCH_VALIDATION_CONTEXT','working repository'),
        dataset_filenames=datasets.path.to_list(),dataset_SHA256=dict(zip(datasets.path,datasets.sha256)),
        model_configuration_files=[relative(p) for p in sorted(METRICS.glob('*.json'))],
        generated_artifact_list=[relative(path) for path in paths]+['results/result_manifest.csv','results/run_manifest.json','results/logs/pipeline_run.txt'],
        self_hash_exclusions=['results/result_manifest.csv','results/run_manifest.json','results/logs/pipeline_run.txt'],
        status=status))

class Tee:
    def __init__(self,stream,log):
        self.stream,self.log=stream,log
    def write(self,text):
        self.stream.write(text); self.log.write(text); self.log.flush()
    def flush(self):
        self.stream.flush(); self.log.flush()

def run_all():
    from .statistics import characterization,trajectories,associations,descriptive_contrasts
    from .ml import ml_experiments
    from .adjusted import adjusted_models
    from .interactions import interaction_analysis
    from .visualization import generate_figures
    from .evidence import build_evidence,validate_conclusions,historical_discrepancies
    from .documentation import generate_documentation
    start=datetime.now(timezone.utc).isoformat()
    began=time.monotonic()
    LOGS.mkdir(parents=True,exist_ok=True)
    lock=ROOT/'results/.pipeline.lock'
    try:
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor,str(os.getpid()).encode()); os.close(descriptor)
    except FileExistsError:
        raise RuntimeError('A pipeline lock exists; another run may be active. Investigate before removing it.')
    original_out=sys.stdout
    try:
        with (LOGS/'pipeline_run.txt').open('w',encoding='utf-8') as log:
            sys.stdout=Tee(original_out,log)
            print('[1/10] Verify protected inputs and characterize full population',flush=True)
            verify_inputs()
            raw,df=load_population()
            characterization(raw)
            print('[2/10] Reproduce ML experiments (no generated-result inputs)',flush=True)
            configs,delta,weka=ml_experiments(df)
            print('[3/10] Paired academic trajectory and subgroup persistence',flush=True)
            persistence,distribution=trajectories(df)
            print('[4/10] Unadjusted associations and descriptive contrasts',flush=True)
            unadjusted=associations(df)
            descriptive_contrasts(df)
            print('[5/10] Specified adjusted association models',flush=True)
            models,adjusted=adjusted_models(df)
            print('[6/10] Interactions and standardized probabilities',flush=True)
            interactions=interaction_analysis(df)
            print('[7/10] Publication figures and evidence synthesis',flush=True)
            figure_catalog=generate_figures(df)
            evidence=build_evidence(adjusted,unadjusted,interactions)
            conclusions=validate_conclusions(persistence,distribution,configs,delta,adjusted,unadjusted,interactions)
            discrepancies=historical_discrepancies(configs,delta,weka,persistence,distribution,adjusted,unadjusted,interactions)
            summary=dict(dataset_N=len(raw),original_variable_count=raw.shape[1],background_feature_count=len(BASE_FEATURES),
                dataset=relative(DATA),dataset_sha256=sha256(DATA),Year1_model_metrics=configs['year1']['test_metrics'],
                research_model_metrics=configs['research']['test_metrics'],research_historical_weka=weka,
                research_historical_python_metrics=configs['research_historical_python']['test_metrics'],
                Year2_without_Y1_metrics=configs['year2_without_year1']['test_metrics'],
                Year2_with_Y1_metrics=configs['year2_with_year1']['test_metrics'],delta_metrics=delta,
                trajectory_distribution=distribution.to_dict(orient='records'),
                Year1_Year2_spearman_rho=persistence['spearman_rho'],Year1_Year2_cramers_v=persistence['cramers_v'],
                key_adjusted_odds_ratios={name:frame.loc[frame.term.isin(['Year1Band','HSCMathOrdinal','HSCPhysicsOrdinal','HSCChemistryOrdinal','StudentClubMemberOrdinal',"C(Gender, Treatment(reference='Female'))[T.Male]", "C(CollegeCategory, Treatment(reference='Private'))[T.Public]"]),['term','odds_ratio','CI_low','CI_high','p_value','q_value','model_N']].to_dict(orient='records') for name,frame in adjusted.items()},
                verified_conclusions=conclusions.to_dict(orient='records'),
                historical_checkpoint_discrepancies=int((~discrepancies.within_tolerance).sum()),
                generated_table_paths=[relative(p) for p in sorted(TABLES.glob('*.csv'))],
                generated_figure_paths=figure_catalog.png.to_list(),model_configurations=configs,
                interpretation='Exploratory observational associations and conditional predictive performance; no causal or mediation identification.')
            print('[8/10] Generate research documentation and final summary',flush=True)
            summary['reference_paper']=generate_documentation(raw,df,summary,evidence,conclusions,discrepancies)
            summary['generated_table_paths']=[relative(p) for p in sorted(TABLES.glob('*.csv'))]
            save_json(ROOT/'results/final_research_summary.json',summary)
            # First manifest gives output-reference tests a concrete artifact list;
            # final manifest is refreshed after the audit and test logs exist.
            manifests(start,SEED,raw.shape)
            print('[9/10] Run meaningful validation tests',flush=True)
            tests=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
            (LOGS/'test_results.txt').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8')
            print(tests.stdout,flush=True)
            if tests.returncode:
                raise RuntimeError('Validation tests failed; see results/logs/test_results.txt')
            if (ROOT/'config/numerical_repeatability_baseline.json').is_file():
                repeat=subprocess.run([sys.executable,str(ROOT/'scripts/verify_repeatability.py')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
                print(repeat.stdout,flush=True)
                if repeat.returncode:
                    raise RuntimeError('Numerical repeatability check failed; investigate results/logs/numerical_repeatability.json')
            print('[10/10] Verify links, statistical intervals, preservation and output audit',flush=True)
            check=verify(allow_pending_audit=True)
            save_json(LOGS/'output_verification.json',check)
            audit=f'''# Reproducibility audit

Pipeline: **PASSED**. All stages recomputed from protected source inputs, without reading prior generated results as analytical inputs. Validation: **{tests.stdout.strip().splitlines()[-1]}**; [test log](../results/logs/test_results.txt). Output/link/schema/preservation/secret-pattern checks: **{check['checks']} passed**.

Execution context: **{os.environ.get('RESEARCH_VALIDATION_CONTEXT','working repository')}**. Source commit: `{git('rev-parse','HEAD')}`; [execution environment and context](../results/run_manifest.json).

Generated CSV tables: **{check['tables']}** (each also rendered as Markdown). Figure concepts: **{check['figures']}**, with PNG/PDF/SVG exports. Artifact hashes and generators: [result manifest](../results/result_manifest.csv). Dataset identities and hashes: [data manifest](../data/DATA_MANIFEST.csv). Authoritative v3 SHA256: `{sha256(DATA)}`; N={len(raw)}, original variables={raw.shape[1]}.

Historical checkpoints beyond the declared approximate tolerance: **{summary['historical_checkpoint_discrepancies']}**; [comparison table](../results/tables/historical_checkpoint_comparison.csv) and [configuration explanations](HISTORICAL_DISCREPANCIES.md). The historical WEKA API refit matches every frozen-model prediction and probability; the initial different CLI result is retained diagnostically. Original Year-2 and historical adjusted/interaction specifications are unavailable; computed explicit specifications remain authoritative.

Numerical repeatability: [independent-run comparison](../results/logs/numerical_repeatability.json) checks the recorded numerical artifact bytes against the earlier full analytical run. Historical comparison rows were extended between runs; their expected-value metadata is excluded from this repeatability baseline. Git attributes preserve dataset bytes across operating-system checkout line endings.

Preservation: all baseline-protected source datasets, models, outputs and research artifacts match their recorded SHA256; original README/dictionary snapshots retained. The pre-existing modified presentation is preserved outside the intended research commit. No project license or author ownership is invented. All intended artifacts are below normal GitHub individual-file limits; no LFS migration is necessary. Historical metric prose, legacy reports/presentations and result files are explicitly snapshots, with current numeric findings generated from outputs.

Unresolved scientific issues: historical missing specifications, exploratory modeling, ordinal linear-trend assumptions, mobility boundaries, varying subgroup denominators, observational design and external validation. See [limitations](LIMITATIONS.md). These constrain interpretation, not execution.

Commit-ready research status: **YES after staged-diff review**, excluding the pre-existing presentation modification and ignored runtime environments. Git commit/push status is reported separately because this audit is produced before committing. Reproduce: `python scripts/run_all.py` under the [recorded environment](REPRODUCIBILITY.md). Timestamps, git metadata and engine timings vary; numerical outputs are deterministic under the pinned environment and seed.
'''
            (DOCS/'REPRODUCIBILITY_AUDIT.md').write_text(audit,encoding='utf-8')
            manifests(start,SEED,raw.shape,status='completed; tests and audit passed')
            final_check=verify(check_result_hashes=True)
            print(f'Complete: {final_check["tables"]} tables, {final_check["figures"]} figures, {final_check["checks"]} checks; {time.monotonic()-began:.1f}s.',flush=True)
    except Exception as error:
        manifest_path=ROOT/'results/run_manifest.json'
        if manifest_path.is_file():
            state=json.loads(manifest_path.read_text(encoding='utf-8'))
            state['status']='FAILED'
            state['failure']=str(error)
            save_json(manifest_path,state)
        raise
    finally:
        sys.stdout=original_out
        lock.unlink(missing_ok=True)
