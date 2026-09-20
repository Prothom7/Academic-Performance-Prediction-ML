"""Generate reports from authoritative outputs, never from checkpoint prose."""
import json
import re
import pandas as pd
from pypdf import PdfReader
from .common import *

QUESTIONS = [
    'How do academic-performance distributions change between Year 1 and Year 2?',
    'How strongly does first-year academic standing persist into the second year?',
    'Which pre-university, socioeconomic, educational and institutional characteristics are associated with academic standing?',
    'Which characteristics distinguish improved, stable and declined academic standing?',
    'How do associations vary across gender, faculty, institution, college background and living arrangement?',
    'Which academic, institutional, socioeconomic and extracurricular characteristics are associated with research participation?',
    'How much additional predictive information does Year-1 standing provide for Year-2 standing beyond background variables?',
    'How similar are the association profiles of attainment, mobility and research engagement?',
]

def write(name,text):
    (DOCS/name).write_text(text.strip()+'\n',encoding='utf-8')

def table_link(name,label=None):
    return f'[{label or name.replace("_"," ")}](../results/tables/{name}.csv)'

def reference_comparison(raw):
    paper=ROOT/'references/academic_performance_paper.pdf'
    text='\n'.join(page.extract_text() for page in PdfReader(paper).pages)
    (LOGS/'reference_paper_extracted.txt').write_text(text,encoding='utf-8')
    n_match=re.search(r'contains\s+([\d,]+)\s+observations',text,re.I)
    p_match=re.search(r'has\s+(\d+)\s+vari',text,re.I)
    if not n_match or not p_match:
        raise RuntimeError('Reference-paper dimensional extraction failed.')
    reference_n=int(n_match.group(1).replace(',','')); reference_p=int(p_match.group(1))
    comparison=pd.DataFrame([
        {'dimension':'Population size','reference_dataset':str(reference_n),'current_project':str(len(raw))},
        {'dimension':'Variables','reference_dataset':str(reference_p),'current_project':str(raw.shape[1])},
        {'dimension':'Design','reference_dataset':'Linked Colombian ICFES assessment databases','current_project':'Current assembled academic-performance survey population'},
        {'dimension':'Educational stages','reference_dataset':'Secondary-school Saber 11 and end-of-training Saber Pro','current_project':'HSC preparation plus Year-1 and Year-2 CGPA bands'},
        {'dimension':'Academic evolution','reference_dataset':'Competencies assessed across educational stages','current_project':'Ordinal persistence, transition matrix and mobility categories'},
        {'dimension':'Context','reference_dataset':'Academic, social and economic variables','current_project':'Academic, socioeconomic, institutional, preferences, living and financial variables'},
        {'dimension':'Research engagement','reference_dataset':'Not an outcome described in the article','current_project':'Binary research participation; extracurricular and contextual associations'},
        {'dimension':'Comparison scope','reference_dataset':'Data-description article','current_project':'Predictive and explanatory association analyses; no accuracy-superiority claim'},
    ])
    save_table('reference_dataset_comparison',comparison)
    write('REFERENCE_PAPER_COMPARISON.md',f'''# Reference-paper comparison

The local [reference PDF](../references/academic_performance_paper.pdf) is Delahoz-Dominguez, Zuluaga and Fontalvo-Herrera (2020), *Dataset of academic performance evolution for engineering students*, Data in Brief 30, 105537, [DOI](https://doi.org/10.1016/j.dib.2020.105537). Its population size and variable count below are extracted and checked by `src/research/documentation.py`, with the extracted text retained in [the analysis log](../results/logs/reference_paper_extracted.txt).

{comparison.to_markdown(index=False)}

These datasets concern different populations, educational systems, stages and measurements. Neither accuracy nor distributions constitute a direct population comparison. The current project adds explicit university-band mobility and research-engagement analyses; it does not establish superiority over the reference article. See {table_link('reference_dataset_comparison')} for a machine-readable comparison.
''')
    return {'reference_N':reference_n,'reference_variables':reference_p,'paper_sha256':sha256(paper)}

def generate_documentation(raw,df,summary,evidence,conclusions,discrepancies):
    metrics=pd.read_csv(TABLES/'ml_test_metrics.csv')
    interaction=pd.read_csv(TABLES/'interaction_tests.csv')
    diagnostics=pd.read_csv(TABLES/'adjusted_model_diagnostics.csv')
    figures=pd.read_csv(TABLES/'figure_catalog.csv')
    reference=reference_comparison(raw)
    rq='\n'.join(f'{i}. {q}' for i,q in enumerate(QUESTIONS,1))
    report_metrics=metrics[['study','train_N','validation_N','test_N','accuracy','weighted_f1','macro_f1','accuracy_CI_low','accuracy_CI_high']].copy()
    metric_text=report_metrics.to_markdown(index=False,floatfmt='.6f')
    finding_text=[]
    for _,row in conclusions.iterrows():
        finding_text.append(f'''## {row.conclusion_id}. {row.candidate_conclusion} — {row.status}

{row.statistics}

Evidence: [{Path(row.supporting_table).stem}](../{row.supporting_table}), [figure](../{row.supporting_figure}), [generating script](../{row.supporting_script}).

Interpretation and caveat: {row.caution}
''')
    findings='\n'.join(finding_text)
    write('FINDINGS.md','# Regenerated findings\n\nGenerated by `src/research/documentation.py` from computed tables. Status labels evaluate the candidate claims against the explicit model specifications.\n\n'+findings)
    selected=evidence[['factor','outcome','coefficient_contrast','unadjusted_effect','FDR_q','adjusted_effect_OR_or_beta','adjusted_CI_low','adjusted_CI_high','adjusted_q','sample_size','evidence_category']]
    write('MASTER_EVIDENCE_TABLE.md',f'''# Master evidence table

Generated from {table_link('master_evidence_table')}. Every row represents an explicit coefficient contrast, grouped by factor and outcome; categorical factors may therefore have several rows. Factor-level survival uses joint-Wald FDR, and coefficient-level survival uses separate within-model coefficient FDR. Raw outcomes are multiclass bands/trajectory, whereas adjusted attainment/mobility outcomes are binary: these are related but distinct questions.

Evidence categories use transparent rules: Central denotes the retained Year-1 term in the Year-2 model; Context-dependent denotes a retained corresponding interaction; Strong secondary denotes retained factor-level adjusted FDR; Weakens after adjustment denotes a raw retained association without retained factor-level adjusted FDR; all remaining rows are Negative/non-conclusive. No evidence score is constructed.

{selected.to_markdown(index=False,floatfmt='.5g')}

The CSV includes descriptive contrasts, both test families, interpretations, caveats and real script/table/figure paths.
''')
    write('RESEARCH_OVERVIEW.md',f'''# Research overview

This project evaluates attainment, academic mobility and research engagement in the current full analytical population, retaining all {len(raw)} records and {raw.shape[1]} original variables. The conceptual proposition is assessed through the {table_link('master_evidence_table')} and {table_link('verified_conclusions')}, rather than assumed.

## Research questions

{rq}

Study A reproduces frozen five-class Year-1 prediction. Study B reproduces the original WEKA research experiment, its Python approximation, and a current-population Python experiment. Study C compares Year-2 prediction without and with Year-1 standing on identical rows and a common configuration. Descriptive and adjusted association analyses address the scientific questions beyond predictive metrics.
''')
    write('METHODOLOGY.md',f'''# Methodology

## Population and measurements

The authoritative input is [academic_performance_clean_v3.csv](../{relative(DATA)}). All {len(raw)} records remain in the analytical population. Column names, levels, missingness and distributions are regenerated in the [dictionary](DATA_DICTIONARY.md). Labels are reduced to their English display portion for analyses; original bilingual values remain preserved in the dataset and dictionary exports. Timestamp is administrative and excluded from predictors.

CGPA bands are ordered as `{BANDS}` with scores 0 through 4. Survey band boundaries can overlap in their printed wording; exact within-band CGPA is unavailable. The high-performance definition selects the upper two bands (operationally >=3.50); low Year-2 selects the lowest band. A band change equals Year2Band minus Year1Band: positive Improved, zero Stable, negative Declined. Band movement is not a continuous CGPA change.

Research participation maps each original Yes category to 1 and No to 0. Missing outcomes stay missing and are never coded No. The original outcome-category distributions remain available. No current-population duplicate exclusions are applied. Historical research splits are replayed under their existing explicit protocol and reported separately.

## Predictor timing and ML

The 23 background predictors are listed in each [metric/configuration JSON](../results/metrics/year1.json). Study A uses the existing splits and frozen forest (100 trees, sqrt features, leaf size 1, seed {SEED}), reconstructing its 27-candidate validation sweep without changing the recorded final model. Historical labels and preprocessing are retained. Both Year-2 versions use the same 70/15/15 stratified split (Year-2 target), seed {SEED}, original 23 background predictors, candidate grid and selected forest; only C2 adds Year1CGPA. The common forest maximizes the mean of C1 and C2 validation weighted F1, breaking ties by mean accuracy and kappa. Test does not participate in selection.

Current-population research prediction uses the historical 24 end-of-Year-1 feature set, excluding clubs, contemporary living/financial context and Year-2 standing. It uses a predeclared 100-tree, max_features=12, leaf-size=1 Python forest. This is an approximation because encoded columns differ from original WEKA attributes. Exact historical WEKA uses 100 trees, K=12, M=1, seed {SEED}, original training order and nominal header. The bundled API helper explicitly reads UTF-8 and checks unique English label prefixes against the preserved model header, independently refits the forest and replays the saved model. Both predictions and probabilities are checked. A CLI converter path is also retained as a diagnostic; its differing output is not selected as the historical experiment.

All preprocessing is fitted inside the training pipeline. New Python analyses use training-fitted most-frequent categorical imputation and one-hot encoding with unseen levels ignored; graduation year is numeric. Historical Python reproductions keep the existing NaN encoding. Final models refit on Train+Validation before Test evaluation. Majority-class baselines, per-class and macro/weighted metrics, kappa, balanced accuracy, MCC, ROC AUC, average precision, confusion matrices and individual test predictions are exported. Accuracy CIs use Wilson; F1 CIs use 1,500 seeded row-bootstrap draws. C1/C2 difference intervals use the same sampled Test indices; exact McNemar assesses paired correctness. CIs are conditional on the fitted model and split, not uncertainty over model development.

Forest fitting uses up to four workers with independently assigned fixed tree seeds. Prediction uses one worker to keep probability summation order deterministic; numerical linear algebra uses one thread. The actual worker settings and Patsy formula engine are recorded in the run manifest.

## Descriptive and unadjusted analyses

Continuous descriptive statistics are used only for actual numeric measurements (graduation year). Banded variables are summarized by level and counts. Category percentages include explicit population and observed-value denominators. Proportions use Wilson CIs. Year comparisons are paired: transition counts, row percentages, Spearman correlation, bootstrap correlation CIs (1,000 full-population draws; 500 subgroup draws), signed-rank and exact direction sign tests are exported.

Categorical tests use Pearson chi-square without Yates correction; conventional uncorrected Cramers V is sqrt(chi-square/[N*(min(rows,columns)-1)]). Each result records chi-square, degrees of freedom, N, p, BH q, minimum expected count and fraction below five. Sparse cells make asymptotic tests approximate and limit interpretations. Unadjusted ordered relationships additionally use Spearman rho and 300-draw percentile bootstrap intervals. Trajectory Spearman uses signed band change. Relationships between starting bands and mobility are mathematically constrained, not independent mechanisms.

## Adjusted association models

Five binomial-logit GLMs model high Year-1, high Year-2, improvement versus all others, decline versus all others, and binary research participation. All measured background variables are included. Year-2 adds Year-1 standing; both mobility models always include starting Year-1 standing and contemporary living, work, expenditure, financial-support and club variables. The research association model adds these contemporary variables and current Year-2 standing. This model differs in purpose and feature timing from research prediction.

Ordered HSC subjects, attendance, family size, income, expenditure, preferences, parental education, employment hours and clubs are scored by ordered levels; they assume a linear trend on log odds, not equal distances on the original measurement scale. Graduation year is centered at 2020 for stable intercept estimation. Other predictors are nominal treatment contrasts. Reference levels and complete formulas are exported in {table_link('model_specifications')}. Female, Private college, unavailable computer, CSE faculty, KUET institution, first-child, parents/family living and parents/guardians funding are explicit references; remaining nominal references follow sorted levels. Faculty ECE is pooled into Others **only within fitted association models** because its N is very small; original records and descriptive categories are retained.

Each model uses complete cases for its own variables; per-model omitted counts, events and N are reported in {table_link('adjusted_model_diagnostics')}. Maximum-likelihood coefficient CIs are Wald intervals exponentiated to ORs. Reported diagnostics include convergence, matrix rank, condition number, events per parameter, likelihood, AIC, deviance, McFadden pseudo-R2, model LR, and explicitly in-sample AUC/Brier. These are not validation scores. Sparse/unstable models fail the pipeline rather than silently receiving invalid penalized-model p-values.

## Multiplicity, interactions and subgroups

Benjamini–Hochberg FDR families are defined separately for all unadjusted categorical tests per outcome, ordered tests per outcome, coefficients per adjusted model (excluding intercept), joint factor Wald tests per adjusted model, and the nine interaction/overall-term tests jointly. Subgroup correlation tests are corrected separately within grouping factor. The selected raw/adjusted OR comparisons have a per-model raw-test family. Paired global trajectory tests and the single paired ML comparison are targeted tests. The raw p-values and corrected q-values remain available; FDR q<0.05 denotes exploratory support.

Interactions use nested likelihood-ratio tests on identical complete-case rows. Gender by each HSC subject is tested for Year-1, gender by Math for Year-2 before and after Year-1 adjustment, college/faculty by clubs for research, and gender by living arrangement for decline. The adjusted overall trajectory term in research is tested separately from product interactions but shares the nine-test exploratory family. Standardized probabilities average model predictions over the complete-case empirical covariate distribution; CIs use the delta method on the logit of the average with fitted parameter covariance, holding covariate distribution fixed. Unsupported interactions remain non-conclusive, regardless of plotted curves. Reported subgroup N is the observed combination count; correlations below N=20 are suppressed. Institutions are contextual groups, not rankings.

## Interpretation

All findings are observational associations within this assembled dataset. Conditioning on Year-1 can capture predictive information but does not identify mediation. OR attenuation can involve confounding, conditioning and noncollapsibility. Differences in significance across outcomes are not formal tests of coefficient differences. Attainment, mobility and engagement are operationally distinct outcomes; the analysis does not establish a latent-dimension or causal model. Historical checkpoints are comparison metadata, never fitting targets.

Implementation: [statistics](../src/research/statistics.py), [ML](../src/research/ml.py), [adjusted models](../src/research/adjusted.py), [interactions](../src/research/interactions.py), [evidence synthesis](../src/research/evidence.py).

Implementation references: [SciPy contingency association](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html), [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), and [statsmodels GLMs](https://www.statsmodels.org/stable/glm.html).
''')
    write('LIMITATIONS.md','''# Limitations

The design is observational. The current full assembled dataset is the authorized analytical population; its records are not asserted to be an independently sampled, representative cohort. Survey wording indicates reported measures. Existing provenance disclosures are retained in the historical documentation without changing the current-population inclusion decision. No analysis here identifies a causal effect or mediation pathway.

CGPA, HSC marks and most contextual measures are bands. Printed boundaries can overlap and exact scores are unavailable; ordinal trends assume linear log odds per category. Improvement and decline have structural floor/ceiling constraints, while logistic trend adjustment approximates starting position. These models describe observed category movement rather than precise academic growth. Some mobility models have modest events per parameter; their finite estimates and nominal CIs require independent confirmation.

Complete-case adjusted models have distinct denominators and assume that omitted records do not invalidate the conditional associations. Small category counts can make asymptotic chi-square and Wald tests approximate. Faculty ECE is retained descriptively but pooled with Others only in association models. No institution-specific result is a quality ranking. Standardized probabilities extrapolate combinations and hold the empirical covariate distribution fixed for interval estimation.

The historical Year-1 Test was held out from the coded validation sweep but the full dataset had already been explored in WEKA. Historical research uses its existing separate protocol, whereas current-population research uses available outcomes without duplicate exclusions. Implementation and label-processing differences distinguish approximate Python and WEKA models. Predictor timing is defined conceptually but not independently verified from prospective collection times. Research association covariates such as clubs may be concurrent with the outcome and should not be treated as available prospective predictors.

A single deterministic split provides conditional test estimates; bootstrap CIs do not capture training-selection variability or transportability to a new cohort. FDR controls specified exploratory families under its assumptions; it does not turn exploratory results into confirmed findings. Cross-outcome significance differences are not formal coefficient-difference tests, and association-profile comparisons do not establish psychometric dimensional separation. Confirmatory preregistration, external cohorts and sensitivity to ordinal/nonlinear specifications are appropriate next steps.

No project-wide LICENSE or authorship terms are invented. Dataset redistribution and reuse remain subject to the repository owner's applicable permissions; bundled WEKA retains its own GPL terms.
''')
    write('DECISION_LOG.md',f'''# Analytical decision log

1. Retain the current authoritative full dataset ({len(raw)} rows, {raw.shape[1]} variables) without current-population duplicate exclusions. Historical research reproduces its existing explicit protocol as a separate experiment.
2. Use survey CGPA categories in their documented order; do not infer continuous scores. High performance is the upper two bands, operationally >=3.50.
3. Classify academic movement from the signed ordinal band difference. Fit improvement and decline separately against all other records.
4. Always include Year-1 starting band in adjusted mobility; include it in the main adjusted Year-2 model. Include current Year-2 standing in research association models, while keeping it out of historical-horizon research prediction.
5. Use seed {SEED} for splits, forests and bootstrap draws. Keep model configurations and preprocessing in metric JSON files. New categorical imputation learns only from training data.
6. Preserve Study A's frozen configuration and existing splits. Select one common Study C configuration from mean C1/C2 validation F1 to isolate the feature comparison. Never select with Test results.
7. Encode research as any Yes versus No; retain missing outcomes as missing. Historical nominal-attribute WEKA is independently refitted and compared with saved-model replay. Its header is used only for exact symbolic encoding, never for target-driven tuning.
8. Specify all background covariates rather than choose predictors from p-values. Add contemporary contexts to association models for mobility/research only. Missing-covariate exclusions are model-specific complete cases, not population exclusions.
9. Center graduation year at 2020. Treat selected bands as ordered trends. Reference categories are recorded per model; no nominal integer labels imply an order.
10. Pool Faculty ECE into Others only in fitted association models to avoid sparse-category instability; retain original descriptive/subgroup reporting.
11. Apply BH FDR to the predeclared exploratory families in [methodology](METHODOLOGY.md). Use nested LR interaction tests and joint factor Wald tests, with raw p and q preserved.
12. Standardize marginal predictions over each model's complete-case covariate population. Use Wilson proportion CIs, seeded percentile bootstrap metric/correlation CIs, and parameter-covariance delta CIs for standardized probabilities.
13. Compare checkpoints after computation; preserve discrepancies and record dataset, engine, feature set, denominator and configuration rather than optimize toward historical values.
14. Generate every result table, figure, conclusion status and numerical report statement from executable outputs. Preserve prior artifacts and mark their metric prose as historical snapshots.
''')
    write('REPRODUCIBILITY.md','''# Reproducing the project

From the repository root, create a virtual environment with Python 3.13 and install the fully pinned environment:

```bash
python -m venv .venv
python -m pip install -r requirements-lock.txt
python scripts/run_all.py
```

Activate the virtual environment before the last two commands if your shell does not automatically use it. On Windows use `.venv\\Scripts\\activate`; on Linux/macOS use `source .venv/bin/activate`. Install a working Java runtime, ensure `java -version` succeeds, or set `JAVA_EXE` to the executable. The bundled WEKA JAR and precompiled Java-8-compatible replay helper need no network download or Java compiler during ordinary reproduction. Python package versions and the actual Java version are recorded in the run manifest. `requirements.txt` lists the top-level packages; the lock file pins all tested transitive dependencies.

`scripts/run_all.py` recomputes all outputs without reading previous generated results as analytical inputs. It validates input checksums against `data/DATA_MANIFEST.csv`, fails on missing required inputs, reconstructs experiments, generates tables and figures, evaluates candidate conclusions, writes reports/manifests, runs tests and performs the final audit. Historical source/result artifacts are never overwritten by this entry point. It can be invoked from any working directory. Full-data statistical analyses and ML validation sweeps are deliberately distinct stages.

```bash
python scripts/verify_outputs.py
python -m pytest -q
python -m src.research.interactions
```

The last command reruns only interaction analyses and changes their outputs; run the full entry point afterwards to synchronize reports and manifests. Numerical tables are deterministic under the recorded environment and seed; timestamps, git metadata and engine timings differ between executions. Result manifest hashing excludes itself and the run manifest to avoid recursive self-hashes. Nonanalytic runtime/cache folders are ignored. Model files use compressed joblib serialization; load them only from trusted sources.

Git attributes preserve exact generated bytes and the inspected historical checkout line endings across platforms. Historical mixed endings are retained byte-for-byte in the research branch; prior normalized Git representations remain in existing history. The baseline presentation check accepts either the recorded local user modification or the original committed version, because that local modification is excluded from research commits and does not enter any analysis.

See [the audit](REPRODUCIBILITY_AUDIT.md), [run manifest](../results/run_manifest.json), [artifact manifest](../results/result_manifest.csv), [data manifest](../data/DATA_MANIFEST.csv), [summary JSON](../results/final_research_summary.json), and [historical checkpoint comparison](../results/tables/historical_checkpoint_comparison.csv).
''')
    write('CHANGELOG_RESEARCH.md','''# Research implementation changelog

Added a current-population reproducible analysis layer without deleting historical research datasets, experiments, result files or models. Reconstructed frozen Year-1 validation/Test and original research WEKA/API training, added current-population and historical Python research experiments, and paired Year-2 feature comparisons. Added dataset characterization, ordinal transitions, unadjusted associations, five adjusted models, dedicated interactions, standardized predictions, subgroup persistence, publication figures, evidence tables and generated reports. Added a pinned environment, data/artifact hashes, meaningful validation tests and an executable reproducibility audit. The existing uncommitted presentation is outside the research change set.
''')
    write('FIGURE_INDEX.md','# Generated figures\n\nEach plot is generated from computed tables by `src/research/visualization.py`. PNG, PDF and SVG exports are listed in '+table_link('figure_catalog')+'.\n\n'+
        '\n\n'.join(f"- [{r.figure}](../{r.png}): {r.title}. Inputs: {r.source_tables}. {r.note if pd.notna(r.note) else ''}" for _,r in figures.iterrows()))
    discrepancy_count=int((~discrepancies.within_tolerance).sum())
    write('HISTORICAL_DISCREPANCIES.md',f'''# Historical checkpoint comparison

Generated comparison: {table_link('historical_checkpoint_comparison')}. {discrepancy_count} checkpoints differ beyond the declared approximate tolerance. Each row retains the expected checkpoint, computed result, difference, denominator/configuration context and authoritative output path. These are research changes or unresolved historical-specification differences, not targets for numerical tuning.

The exact historical WEKA API refit matches saved-model predictions and probabilities. A separate CLI route initially produced different output. The preserved model contains historically corrupted bilingual symbolic labels. WEKA's [data-dependent RNG implementation](https://github.com/Waikato/weka-3.8/blob/master/weka/src/main/java/weka/core/Instances.java) includes a row-string hash, so symbolic text can change randomized training even when numeric nominal indices and the explicit seed match. The replay helper verifies zero input-cell differences and records different derived RNG values for the original versus UTF-8 source headers. It aligns the current source to the unique English prefixes of the preserved nominal header, restores the original symbolic encoding, and independently refits; no predictors, outcomes, rows or configuration are tuned to checkpoints. The CLI diagnostic remains in [its JSON](../results/metrics/research_historical_weka_cli_diagnostic.json) and [executed log](../results/logs/research_historical_weka.txt).

Year-2 uses the newly documented directly comparable feature-set protocol because no original Year-2 executable/configuration is present. Adjusted and interaction specifications include all measured background variables, ordinal trends and documented contextual controls; historical specifications and test types are unavailable. Those configuration differences remain unresolved rather than being retrospectively reconstructed to target p-values. Unadjusted research tests use the binary outcome; any historical multi-category target may therefore differ. Missingness denominators and Faculty ECE pooling are explicit in the model diagnostics. Coefficient CIs and interaction evidence are taken from the computed tables.

{discrepancies[['checkpoint','historical_expected','computed','difference','within_tolerance']].to_markdown(index=False,floatfmt='.6g')}
''')
    report=f'''# Academic attainment, mobility and research engagement: a reproducible survey analysis

## Abstract-style summary

The current full dataset contains {len(raw)} records and {raw.shape[1]} original variables. Year-1/Year-2 standing is evaluated as ordered survey bands. The regenerated analyses assess persistence, background and contextual associations, mobility, research engagement and paired predictive information. Exact numerical findings and candidate-status decisions below come from {table_link('verified_conclusions')}; observational associations do not establish mechanisms or causality.

## 1. Introduction

Academic attainment, movement between standing bands and research engagement are operationally different outcomes. This project tests whether their measured association profiles support a differentiated conceptual account, using predictive models alongside interpretable statistics.

## 2. Dataset

The analytical population is [the full v3 dataset](../{relative(DATA)}). {table_link('dataset_summary')} and the [dictionary](DATA_DICTIONARY.md) characterize all variables. Original/raw/intermediate/model-specific inputs remain preserved. Missing research outcomes remain missing and adjusted complete-case denominators are explicit.

## 3. Research questions

{rq}

## 4. Methodology

See [complete methodology](METHODOLOGY.md), [decision log](DECISION_LOG.md), and {table_link('model_specifications')}. Band construction, split governance, feature timing, multiplicity, references and interval methods are explicit.

## 5. Year-1 academic performance

Frozen historical five-class prediction is regenerated with original splits. The metrics below come from {table_link('ml_test_metrics')}. Class-level error structure appears in {table_link('year1_classification_report')} and {table_link('year1_confusion_matrix')}. Inferential results remain separate in {table_link('adjusted_year1')}.

{metric_text}

## 6. Research participation

The exact original WEKA refit gives accuracy {summary['research_historical_weka']['accuracy']:.6f} and Participated F1 {summary['research_historical_weka']['participated_f1']:.6f}, Test N={summary['research_historical_weka']['test_N']}; see [executed prediction-derived metrics](../results/metrics/research_historical_weka.json) and [API predictions](../results/logs/weka_replay_predictions.csv). Python historical/current-population studies are reported separately above. Scientific associations use {table_link('adjusted_research')} with contemporary contexts and current standing, distinct from the prediction horizon.

## 7. Year-2 academic performance

The without/with-Year-1 versions share rows, preprocessing and a common forest. {table_link('year2_added_information')} contains paired metric changes, bootstrap CIs and exact McNemar p, with configurations in [without-Year-1](../results/metrics/year2_without_year1.json) and [with-Year-1](../results/metrics/year2_with_year1.json).

## 8. Academic progression and persistence

{table_link('trajectory_distribution')}, {table_link('transition_counts')}, {table_link('transition_row_percent')} and {table_link('persistence_statistics')} provide exact movement, paired change tests and association estimates; [transition figure](../results/figures/04_year1_vs_year2_transition_heatmap.png). Correlations use ordinal standing rather than continuous CGPA.

## 9. Unadjusted association analysis

{table_link('master_unadjusted_associations')} retains chi-square, degrees of freedom, raw p, outcome-family BH q, Cramers V, N and sparse-cell diagnostics. {table_link('spearman_associations')} adds ordered trends and bootstrap intervals. These findings are not adjusted independent relationships.

## 10. Multivariable analysis

Five adjusted binomial-logit models export ORs, Wald CIs and coefficient q-values; {table_link('adjusted_factor_tests')} additionally supplies omnibus factor inference. Starting Year-1 standing is controlled in both mobility models and the main Year-2 model. Model diagnostics:

{diagnostics[['model','N','omitted_due_to_missing','events','parameters','events_per_parameter','converged','rank','condition_number','McFadden_R2']].to_markdown(index=False,floatfmt='.6g')}

## 11. Interaction and subgroup analysis

Nested LR tests on identical complete cases are corrected jointly. Unsupported curves remain non-conclusive. {table_link('interaction_marginal_probabilities')} provides standardized predictions, parameter-covariance CIs and observed subgroup sizes. {table_link('subgroup_persistence')} describes contextual persistence and suppresses tiny-group correlations.

{interaction[['analysis','LR_statistic','df','p_value','q_value','N','supported']].to_markdown(index=False,floatfmt='.6g')}

## 12. Cross-outcome synthesis

The evidence supports a descriptive distinction among attainment, movement and engagement when retained factors and effect profiles differ. The [master evidence table](MASTER_EVIDENCE_TABLE.md) gives each coefficient contrast and supports. Significance differences are not coefficient-difference tests, and no causal arrows, formal mediation or latent-dimension model are established.

## 13. Comparison with reference dataset

The local article's extracted dimensions are {reference['reference_N']} observations and {reference['reference_variables']} variables; see {table_link('reference_dataset_comparison')} and the [source-grounded comparison](REFERENCE_PAPER_COMPARISON.md). Educational stages and populations differ, preventing direct accuracy or population comparisons.

## 14. Principal findings

{findings}

## 15. Limitations

See [limitations](LIMITATIONS.md): observational measures, band discretization, starting-position boundaries, missingness, subgroup variation, conditional test uncertainty and exploratory multiplicity. Historical test-governance disclosures remain preserved. Model associations do not identify effects of institutions, clubs or gender.

## 16. Contribution

The contribution is an executable evidence chain spanning dataset characterization, historical experiment reproduction, paired Year-2 information, ordinal mobility, multivariable association, contextual interactions and engagement, with machine-readable provenance for each reported numerical result.

## 17. Reproducibility

Run `python scripts/run_all.py`; see [reproduction instructions](REPRODUCIBILITY.md), [audit](REPRODUCIBILITY_AUDIT.md) and [final JSON](../results/final_research_summary.json). Data and artifact manifests map reported results to executable generators. Historical discrepancies are retained rather than fitted away.

## 18. Next research steps

Evaluate ordinal/nonlinear specification sensitivity, repeated-split development uncertainty and external validation. Preregister confirmatory interaction hypotheses and assess predictor timing prospectively. Broader mechanism or dimension claims need designs and measurements capable of testing them.
'''
    write('FULL_RESEARCH_REPORT.md',report)
    stable=summary['trajectory_distribution']
    README=f'''# Academic Performance Machine Learning Research

An auditable analysis of academic attainment, Year-1 to Year-2 mobility, and research engagement in the current full {len(raw)}-record, {raw.shape[1]}-variable dataset. Numerical claims are generated from executable results; all findings are observational associations within this dataset.

## Setup and reproduction

Use Python 3.13, create and activate a virtual environment, and install a working Java runtime (or set `JAVA_EXE`). The exact historical WEKA engine and GPL license are bundled.

```bash
python -m pip install -r requirements-lock.txt
python scripts/run_all.py
```

The entry point computes all analyses, figures, reports and manifests, runs tests and audits outputs. [Full instructions](docs/REPRODUCIBILITY.md).

## Dataset and studies

Authoritative population: [academic_performance_clean_v3.csv]({relative(DATA)}). All records remain included in characterization and trajectory analysis; unavailable outcomes and covariates have explicit analysis-specific denominators. [Dictionary](docs/DATA_DICTIONARY.md), [data guide](data/README.md), [SHA256 manifest](data/DATA_MANIFEST.csv).

- Study A: exact frozen Year-1 five-class Python prediction with existing Train/Validation/Test.
- Study B: independent historical WEKA refit and saved-model replay; historical Python approximation; current-population prediction and adjusted research association analysis.
- Study C: directly comparable Year-2 prediction without and with Year-1, identical split rows and common forest configuration.

## Research questions

{rq}

## Regenerated predictive results

{metric_text}

All proportions above are fractions. [Metric table](results/tables/ml_test_metrics.csv), [per-study configurations](results/metrics/year1.json), [paired added-information analysis](results/tables/year2_added_information.csv). Exact historical WEKA metrics are [reported separately](results/metrics/research_historical_weka.json).

## Findings and evidence

Year-1/Year-2 Spearman rho is {summary['Year1_Year2_spearman_rho']:.6f}; [persistence table](results/tables/persistence_statistics.csv). Trajectory proportions and confidence intervals are in [the distribution table](results/tables/trajectory_distribution.csv). The conceptual proposition that attainment, mobility and engagement have differing association profiles is evaluated with explicit caveats in [the twelve candidate-conclusion decisions](results/tables/verified_conclusions.csv).

[Findings](docs/FINDINGS.md), [full report](docs/FULL_RESEARCH_REPORT.md), [master evidence](docs/MASTER_EVIDENCE_TABLE.md), [methodology](docs/METHODOLOGY.md), [limitations](docs/LIMITATIONS.md), [checkpoint discrepancies](docs/HISTORICAL_DISCREPANCIES.md). These generated documents distinguish descriptive, unadjusted, adjusted, interaction and predictive results. They do not imply causation, mediation or institutional rankings. [Legacy documentation index](config/legacy_documentation_index.json) explicitly identifies preserved historical prose, including stale paths and metrics, and points to its current replacement.

## Repository guide

- `data/raw`, `data/processed/intermediate`, `data/processed/final`, `data/splits`: preserved source versions and historical modeling inputs.
- `src/research`: data contracts, statistics, ML, adjusted models, interactions, figures, evidence and document generators.
- `scripts/run_all.py`, `scripts/verify_outputs.py`, `tests`: orchestration, verification and scientific invariants.
- `results/tables`, `results/metrics`, `results/figures`, `results/models`, `results/logs`: current authoritative exports.
- `results/final`, `results/validation`, `results/reproduction`, `experiments/weka`, `models`, `archive`, `presentation`: preserved historical research artifacts. Historical metric prose and presentations are snapshots; consult regenerated findings for current inference.
- `docs`: research report, assumptions, data dictionary, decisions, evidence and audit.
- `tools/vendor/weka`: versioned historical engine, replay source/bytecode and GPL license.

Each major table has CSV and generated Markdown exports. The [figure index](docs/FIGURE_INDEX.md) lists all {len(figures)} figures, each exported as PNG/PDF/SVG with computed source-table references. [Final summary JSON](results/final_research_summary.json), [run manifest](results/run_manifest.json), [artifact manifest](results/result_manifest.csv).

## Validation

```bash
python scripts/verify_outputs.py
python -m pytest -q
```

[Reproducibility audit](docs/REPRODUCIBILITY_AUDIT.md). Prior datasets, result files and models are protected by a baseline inventory; the entry point verifies their hashes. Historical documentation snapshots are retained under [archive/prior_documentation](archive/prior_documentation/README.md).

## Reference and citation

The included [Data in Brief article](references/academic_performance_paper.pdf), Delahoz-Dominguez, Zuluaga and Fontalvo-Herrera (2020), *Dataset of academic performance evolution for engineering students*, DOI [10.1016/j.dib.2020.105537](https://doi.org/10.1016/j.dib.2020.105537), concerns a different Colombian assessment population. [Comparison](docs/REFERENCE_PAPER_COMPARISON.md). No directly comparable benchmark or accuracy-superiority claim is made.

For project citation, identify the repository title, GitHub URL, exact research-branch commit and access date; see [citation guidance](CITATION.md). Original authorship and ownership terms are not inferred, and no project-wide license is invented. Bundled third-party WEKA keeps its [existing license](tools/vendor/weka/COPYING).
'''
    (ROOT/'README.md').write_text(README,encoding='utf-8')
    return reference
