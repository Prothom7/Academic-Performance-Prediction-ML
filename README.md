# Academic Performance Machine Learning Research

An auditable analysis of academic attainment, Year-1 to Year-2 mobility, and research engagement in the current full 3516-record, 32-variable dataset. Numerical claims are generated from executable results; all findings are observational associations within this dataset.

## Setup and reproduction

Use Python 3.13, create and activate a virtual environment, and install a working Java runtime (or set `JAVA_EXE`). The exact historical WEKA engine and GPL license are bundled.

```bash
python -m pip install -r requirements-lock.txt
python scripts/run_all.py
```

The entry point computes all analyses, figures, reports and manifests, runs tests and audits outputs. [Full instructions](docs/REPRODUCIBILITY.md).

## Dataset and studies

Authoritative population: [academic_performance_clean_v3.csv](data/processed/intermediate/academic_performance_clean_v3.csv). All records remain included in characterization and trajectory analysis; unavailable outcomes and covariates have explicit analysis-specific denominators. [Dictionary](docs/DATA_DICTIONARY.md), [data guide](data/README.md), [SHA256 manifest](data/DATA_MANIFEST.csv).

- Study A: exact frozen Year-1 five-class Python prediction with existing Train/Validation/Test.
- Study B: independent historical WEKA refit and saved-model replay; historical Python approximation; current-population prediction and adjusted research association analysis.
- Study C: directly comparable Year-2 prediction without and with Year-1, identical split rows and common forest configuration.

## Research questions

1. How do academic-performance distributions change between Year 1 and Year 2?
2. How strongly does first-year academic standing persist into the second year?
3. Which pre-university, socioeconomic, educational and institutional characteristics are associated with academic standing?
4. Which characteristics distinguish improved, stable and declined academic standing?
5. How do associations vary across gender, faculty, institution, college background and living arrangement?
6. Which academic, institutional, socioeconomic and extracurricular characteristics are associated with research participation?
7. How much additional predictive information does Year-1 standing provide for Year-2 standing beyond background variables?
8. How similar are the association profiles of attainment, mobility and research engagement?

## Regenerated predictive results

| study                      |   train_N |   validation_N |   test_N |   accuracy |   weighted_f1 |   macro_f1 |   accuracy_CI_low |   accuracy_CI_high |
|:---------------------------|----------:|---------------:|---------:|-----------:|--------------:|-----------:|------------------:|-------------------:|
| year1                      |      2461 |            527 |      528 |   0.598485 |      0.598370 |   0.603309 |          0.556106 |           0.639441 |
| research                   |      2435 |            522 |      522 |   0.827586 |      0.820562 |   0.798611 |          0.792819 |           0.857568 |
| research_historical_python |      2432 |            521 |      522 |   0.827586 |      0.820945 |   0.799289 |          0.792819 |           0.857568 |
| year2_without_year1        |      2461 |            527 |      528 |   0.649621 |      0.646766 |   0.657557 |          0.607979 |           0.689102 |
| year2_with_year1           |      2461 |            527 |      528 |   0.912879 |      0.913085 |   0.907935 |          0.885744 |           0.934049 |

All proportions above are fractions. [Metric table](results/tables/ml_test_metrics.csv), [per-study configurations](results/metrics/year1.json), [paired added-information analysis](results/tables/year2_added_information.csv). Exact historical WEKA metrics are [reported separately](results/metrics/research_historical_weka.json).

## Findings and evidence

Year-1/Year-2 Spearman rho is 0.846810; [persistence table](results/tables/persistence_statistics.csv). Trajectory proportions and confidence intervals are in [the distribution table](results/tables/trajectory_distribution.csv). The conceptual proposition that attainment, mobility and engagement have differing association profiles is evaluated with explicit caveats in [the twelve candidate-conclusion decisions](results/tables/verified_conclusions.csv).

[Findings](docs/FINDINGS.md), [full report](docs/FULL_RESEARCH_REPORT.md), [master evidence](docs/MASTER_EVIDENCE_TABLE.md), [methodology](docs/METHODOLOGY.md), [limitations](docs/LIMITATIONS.md), [checkpoint discrepancies](docs/HISTORICAL_DISCREPANCIES.md). These generated documents distinguish descriptive, unadjusted, adjusted, interaction and predictive results. They do not imply causation, mediation or institutional rankings. [Legacy documentation index](config/legacy_documentation_index.json) explicitly identifies preserved historical prose, including stale paths and metrics, and points to its current replacement.

## Repository guide

- `data/raw`, `data/processed/intermediate`, `data/processed/final`, `data/splits`: preserved source versions and historical modeling inputs.
- `src/research`: data contracts, statistics, ML, adjusted models, interactions, figures, evidence and document generators.
- `scripts/run_all.py`, `scripts/verify_outputs.py`, `tests`: orchestration, verification and scientific invariants.
- `results/tables`, `results/metrics`, `results/figures`, `results/models`, `results/logs`: current authoritative exports.
- `results/final`, `results/validation`, `results/reproduction`, `experiments/weka`, `models`, `archive`, `presentation`: preserved historical research artifacts. Historical metric prose and presentations are snapshots; consult regenerated findings for current inference.
- `docs`: research report, assumptions, data dictionary, decisions, evidence and audit.
- `tools/vendor/weka`: versioned historical engine, replay source/bytecode and GPL license.

Each major table has CSV and generated Markdown exports. The [figure index](docs/FIGURE_INDEX.md) lists all 21 figures, each exported as PNG/PDF/SVG with computed source-table references. [Final summary JSON](results/final_research_summary.json), [run manifest](results/run_manifest.json), [artifact manifest](results/result_manifest.csv).

## Validation

```bash
python scripts/verify_outputs.py
python -m pytest -q
```

[Reproducibility audit](docs/REPRODUCIBILITY_AUDIT.md). Prior datasets, result files and models are protected by a baseline inventory; the entry point verifies their hashes. Historical documentation snapshots are retained under [archive/prior_documentation](archive/prior_documentation/README.md).

## Reference and citation

The included [Data in Brief article](references/academic_performance_paper.pdf), Delahoz-Dominguez, Zuluaga and Fontalvo-Herrera (2020), *Dataset of academic performance evolution for engineering students*, DOI [10.1016/j.dib.2020.105537](https://doi.org/10.1016/j.dib.2020.105537), concerns a different Colombian assessment population. [Comparison](docs/REFERENCE_PAPER_COMPARISON.md). No directly comparable benchmark or accuracy-superiority claim is made.

For project citation, identify the repository title, GitHub URL, exact research-branch commit and access date; see [citation guidance](CITATION.md). Original authorship and ownership terms are not inferred, and no project-wide license is invented. Bundled third-party WEKA keeps its [existing license](tools/vendor/weka/COPYING).
