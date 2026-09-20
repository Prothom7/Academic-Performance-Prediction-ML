# Academic attainment, mobility and research engagement: a reproducible survey analysis

## Abstract-style summary

The current full dataset contains 3516 records and 32 original variables. Year-1/Year-2 standing is evaluated as ordered survey bands. The regenerated analyses assess persistence, background and contextual associations, mobility, research engagement and paired predictive information. Exact numerical findings and candidate-status decisions below come from [verified conclusions](../results/tables/verified_conclusions.csv); observational associations do not establish mechanisms or causality.

## 1. Introduction

Academic attainment, movement between standing bands and research engagement are operationally different outcomes. This project tests whether their measured association profiles support a differentiated conceptual account, using predictive models alongside interpretable statistics.

## 2. Dataset

The analytical population is [the full v3 dataset](../data/processed/intermediate/academic_performance_clean_v3.csv). [dataset summary](../results/tables/dataset_summary.csv) and the [dictionary](DATA_DICTIONARY.md) characterize all variables. Original/raw/intermediate/model-specific inputs remain preserved. Missing research outcomes remain missing and adjusted complete-case denominators are explicit.

## 3. Research questions

1. How do academic-performance distributions change between Year 1 and Year 2?
2. How strongly does first-year academic standing persist into the second year?
3. Which pre-university, socioeconomic, educational and institutional characteristics are associated with academic standing?
4. Which characteristics distinguish improved, stable and declined academic standing?
5. How do associations vary across gender, faculty, institution, college background and living arrangement?
6. Which academic, institutional, socioeconomic and extracurricular characteristics are associated with research participation?
7. How much additional predictive information does Year-1 standing provide for Year-2 standing beyond background variables?
8. How similar are the association profiles of attainment, mobility and research engagement?

## 4. Methodology

See [complete methodology](METHODOLOGY.md), [decision log](DECISION_LOG.md), and [model specifications](../results/tables/model_specifications.csv). Band construction, split governance, feature timing, multiplicity, references and interval methods are explicit.

## 5. Year-1 academic performance

Frozen historical five-class prediction is regenerated with original splits. The metrics below come from [ml test metrics](../results/tables/ml_test_metrics.csv). Class-level error structure appears in [year1 classification report](../results/tables/year1_classification_report.csv) and [year1 confusion matrix](../results/tables/year1_confusion_matrix.csv). Inferential results remain separate in [adjusted year1](../results/tables/adjusted_year1.csv).

| study                      |   train_N |   validation_N |   test_N |   accuracy |   weighted_f1 |   macro_f1 |   accuracy_CI_low |   accuracy_CI_high |
|:---------------------------|----------:|---------------:|---------:|-----------:|--------------:|-----------:|------------------:|-------------------:|
| year1                      |      2461 |            527 |      528 |   0.598485 |      0.598370 |   0.603309 |          0.556106 |           0.639441 |
| research                   |      2435 |            522 |      522 |   0.827586 |      0.820562 |   0.798611 |          0.792819 |           0.857568 |
| research_historical_python |      2432 |            521 |      522 |   0.827586 |      0.820945 |   0.799289 |          0.792819 |           0.857568 |
| year2_without_year1        |      2461 |            527 |      528 |   0.649621 |      0.646766 |   0.657557 |          0.607979 |           0.689102 |
| year2_with_year1           |      2461 |            527 |      528 |   0.912879 |      0.913085 |   0.907935 |          0.885744 |           0.934049 |

## 6. Research participation

The exact original WEKA refit gives accuracy 0.810345 and Participated F1 0.699088, Test N=522; see [executed prediction-derived metrics](../results/metrics/research_historical_weka.json) and [API predictions](../results/logs/weka_replay_predictions.csv). Python historical/current-population studies are reported separately above. Scientific associations use [adjusted research](../results/tables/adjusted_research.csv) with contemporary contexts and current standing, distinct from the prediction horizon.

## 7. Year-2 academic performance

The without/with-Year-1 versions share rows, preprocessing and a common forest. [year2 added information](../results/tables/year2_added_information.csv) contains paired metric changes, bootstrap CIs and exact McNemar p, with configurations in [without-Year-1](../results/metrics/year2_without_year1.json) and [with-Year-1](../results/metrics/year2_with_year1.json).

## 8. Academic progression and persistence

[trajectory distribution](../results/tables/trajectory_distribution.csv), [transition counts](../results/tables/transition_counts.csv), [transition row percent](../results/tables/transition_row_percent.csv) and [persistence statistics](../results/tables/persistence_statistics.csv) provide exact movement, paired change tests and association estimates; [transition figure](../results/figures/04_year1_vs_year2_transition_heatmap.png). Correlations use ordinal standing rather than continuous CGPA.

## 9. Unadjusted association analysis

[master unadjusted associations](../results/tables/master_unadjusted_associations.csv) retains chi-square, degrees of freedom, raw p, outcome-family BH q, Cramers V, N and sparse-cell diagnostics. [spearman associations](../results/tables/spearman_associations.csv) adds ordered trends and bootstrap intervals. These findings are not adjusted independent relationships.

## 10. Multivariable analysis

Five adjusted binomial-logit models export ORs, Wald CIs and coefficient q-values; [adjusted factor tests](../results/tables/adjusted_factor_tests.csv) additionally supplies omnibus factor inference. Starting Year-1 standing is controlled in both mobility models and the main Year-2 model. Model diagnostics:

| model       |    N |   omitted_due_to_missing |   events |   parameters |   events_per_parameter | converged   |   rank |   condition_number |   McFadden_R2 |
|:------------|-----:|-------------------------:|---------:|-------------:|-----------------------:|:------------|-------:|-------------------:|--------------:|
| year1       | 3514 |                        2 |     1412 |           56 |               25.2143  | True        |     56 |            128.154 |     0.0614442 |
| year2       | 3514 |                        2 |     1365 |           57 |               23.9474  | True        |     57 |            132.163 |     0.530911  |
| improvement | 3482 |                       34 |      381 |           68 |                5.60294 | True        |     68 |            138.652 |     0.148636  |
| decline     | 3482 |                       34 |      491 |           68 |                7.22059 | True        |     68 |            138.652 |     0.162892  |
| research    | 3445 |                       71 |     1225 |           68 |               18.0147  | True        |     68 |            138.314 |     0.121937  |

## 11. Interaction and subgroup analysis

Nested LR tests on identical complete cases are corrected jointly. Unsupported curves remain non-conclusive. [interaction marginal probabilities](../results/tables/interaction_marginal_probabilities.csv) provides standardized predictions, parameter-covariance CIs and observed subgroup sizes. [subgroup persistence](../results/tables/subgroup_persistence.csv) describes contextual persistence and suppresses tiny-group correlations.

| analysis                           |   LR_statistic |   df |     p_value |     q_value |    N | supported   |
|:-----------------------------------|---------------:|-----:|------------:|------------:|-----:|:------------|
| gender_HSCMath_year1               |       48.6224  |    1 | 3.10301e-12 | 2.79271e-11 | 3514 | True        |
| gender_HSCPhysics_year1            |       40.6684  |    1 | 1.80381e-10 | 8.11714e-10 | 3514 | True        |
| gender_HSCChemistry_year1          |       12.3731  |    1 | 0.000435558 | 0.000980005 | 3514 | True        |
| gender_HSCMath_year2_without_year1 |       39.3693  |    1 | 3.50774e-10 | 1.05232e-09 | 3514 | True        |
| gender_HSCMath_year2_with_year1    |        1.51699 |    1 | 0.218076    | 0.218076    | 3514 | False       |
| clubs_college_research             |        5.0774  |    1 | 0.0242397   | 0.0363596   | 3445 | True        |
| clubs_faculty_research             |        6.04029 |    4 | 0.196159    | 0.218076    | 3445 | False       |
| gender_living_decline              |        8.6262  |    5 | 0.124935    | 0.160631    | 3482 | False       |
| trajectory_research_adjusted       |        8.77876 |    2 | 0.0124084   | 0.0223351   | 3479 | True        |

## 12. Cross-outcome synthesis

The evidence supports a descriptive distinction among attainment, movement and engagement when retained factors and effect profiles differ. The [master evidence table](MASTER_EVIDENCE_TABLE.md) gives each coefficient contrast and supports. Significance differences are not coefficient-difference tests, and no causal arrows, formal mediation or latent-dimension model are established.

## 13. Comparison with reference dataset

The local article's extracted dimensions are 12411 observations and 44 variables; see [reference dataset comparison](../results/tables/reference_dataset_comparison.csv) and the [source-grounded comparison](REFERENCE_PAPER_COMPARISON.md). Educational stages and populations differ, preventing direct accuracy or population comparisons.

## 14. Principal findings

## 1. Year-1 standing strongly persists into Year 2. — VERIFIED

Spearman rho=0.846810, 95% bootstrap CI [0.830888, 0.861351], Cramers V=0.695379.

Evidence: [persistence_statistics](../results/tables/persistence_statistics.csv), [figure](../results/figures/04_year1_vs_year2_transition_heatmap.png), [generating script](../src/research/statistics.py).

Interpretation and caveat: Academic persistence within this dataset; ordinal categories do not measure exact score changes.

## 2. Year 1 is a major academic checkpoint. — MODIFIED

Added test accuracy=26.325758 percentage points; gender–Mathematics Year-2 interaction q before=1.05232e-09, after Year-1 adjustment=0.218076.

Evidence: [year2_added_information](../results/tables/year2_added_information.csv), [figure](../results/figures/09_hsc_math_gender_interaction.png), [generating script](../src/research/ml.py).

Interpretation and caveat: MODIFIED: Year-1 standing captures substantial additional information; temporal mediation and causal pathways are not identified.

## 3. HSC preparation relates more strongly to academic level than mobility. — VERIFIED

HSCMath: V(Y1)=0.169702, V(Y2)=0.156397, V(trajectory)=0.033097; HSCPhysics: V(Y1)=0.154832, V(Y2)=0.133840, V(trajectory)=0.059546; HSCChemistry: V(Y1)=0.149006, V(Y2)=0.134890, V(trajectory)=0.080665

Evidence: [master_unadjusted_associations](../results/tables/master_unadjusted_associations.csv), [figure](../results/figures/07_hsc_subject_vs_year1.png), [generating script](../src/research/statistics.py).

Interpretation and caveat: Descriptive comparison of observed association strengths, with different outcome cardinalities.

## 4. Mathematics has the clearest independent Year-1 association, followed by Physics. — VERIFIED

Mathematics: OR=1.783942, CI [1.551015, 2.051848], q=2.82375e-14; Physics: OR=1.260907, CI [1.097413, 1.448758], q=0.0146913; Chemistry: OR=0.949705, CI [0.816817, 1.104214], q=0.690576

Evidence: [adjusted_year1](../results/tables/adjusted_year1.csv), [figure](../results/figures/17_adjusted_year1_forest_plot.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: Ordered per-band estimates depend on linear log-odds specification; ranking point estimates is not a formal difference test.

## 5. Academic attainment and mobility have different associated factors. — MODIFIED

HSCMath: adjusted factor q(Y1)=1.18084e-14, q(decline)=0.896195; HSCChemistry: adjusted factor q(Y1)=0.550069, q(decline)=0.0576339; Gender: adjusted factor q(Y1)=0.0126247, q(decline)=0.0038737

Evidence: [adjusted_factor_tests](../results/tables/adjusted_factor_tests.csv), [figure](../results/figures/19_adjusted_decline_forest_plot.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: MODIFIED: association profiles differ descriptively; differences in significance are not tests of coefficient differences.

## 6. Gender trajectory contrasts involve stability and movement in both directions. — VERIFIED

Female: stable=86.038961%, improved=5.519481%, declined=8.441558%, N=1232; Male: stable=69.045534%, improved=13.704028%, declined=17.250438%, N=2284

Evidence: [trajectory_by_gender](../results/tables/trajectory_by_gender.csv), [figure](../results/figures/10_trajectory_by_gender.png), [generating script](../src/research/visualization.py).

Interpretation and caveat: Within-dataset descriptive distribution; avoid attributing differences to gender itself.

## 7. Institutional context is associated with mobility and research participation. — VERIFIED

Institution V(trajectory)=0.125114, V(research)=0.327030; model omnibus q-values in adjusted factor tests.

Evidence: [adjusted_factor_tests](../results/tables/adjusted_factor_tests.csv), [figure](../results/figures/06_year1_year2_persistence.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: Institutional heterogeneity does not justify university rankings; subgroup sizes vary.

## 8. Clubs are associated more strongly with research than CGPA. — VERIFIED

Club OR(research)=1.361290, CI [1.257958, 1.473110], q=1.26856e-12; V(research)=0.252200, V(Y1)=0.064105, V(Y2)=0.053015.

Evidence: [adjusted_research](../results/tables/adjusted_research.csv), [figure](../results/figures/12_club_vs_research.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: Concurrent measurement; selection into clubs and research is possible.

## 9. College background independently relates to research more clearly than academic standing. — VERIFIED

year1: college-category omnibus q=0.400676; year2: college-category omnibus q=0.709394; research: college-category omnibus q=0.00100564

Evidence: [adjusted_factor_tests](../results/tables/adjusted_factor_tests.csv), [figure](../results/figures/21_adjusted_research_forest_plot.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: Conditional odds ratios use Private as reference; this is not a causal or population-comparability claim.

## 10. Research participation is not represented by CGPA alone. — VERIFIED

After Year-2-band adjustment: club OR=1.361290, q=1.26856e-12; institutional omnibus q=2.1122e-12.

Evidence: [adjusted_research](../results/tables/adjusted_research.csv), [figure](../results/figures/21_adjusted_research_forest_plot.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: Independent associations in this specification; no latent-dimension measurement model was fitted.

## 11. Some unadjusted associations weaken after adjustment. — VERIFIED

year2/HSCPhysics: unadjusted OR=1.382758, adjusted OR=1.108932, adjusted p=0.30366; year2/HSCChemistry: unadjusted OR=1.261263, adjusted OR=1.227826, adjusted p=0.0721396; year2/Gender: unadjusted OR=1.474731, adjusted OR=1.149358, adjusted p=0.267971; decline/HSCPhysics: unadjusted OR=0.848801, adjusted OR=0.878121, adjusted p=0.195739; decline/StudentClubMember: unadjusted OR=1.100181, adjusted OR=1.051677, adjusted p=0.373315; research/HSCMath: unadjusted OR=0.778252, adjusted OR=1.068652, adjusted p=0.352585; research/HSCPhysics: unadjusted OR=0.756305, adjusted OR=1.042332, adjusted p=0.578185; research/HSCChemistry: unadjusted OR=0.709993, adjusted OR=0.997170, adjusted p=0.972885

Evidence: [raw_vs_adjusted_effects](../results/tables/raw_vs_adjusted_effects.csv), [figure](../results/figures/16_raw_vs_adjusted_effects.png), [generating script](../src/research/adjusted.py).

Interpretation and caveat: Adjusted conditional ORs can differ through confounding, conditioning and noncollapsibility; attenuation alone does not establish a mechanism.

## 12. Year-1 CGPA adds information for Year-2 prediction. — VERIFIED

Delta accuracy=26.325758 percentage points, paired 95% bootstrap CI [22.159091, 30.402462], exact McNemar p=9.0023e-32.

Evidence: [year2_added_information](../results/tables/year2_added_information.csv), [figure](../results/figures/04_year1_vs_year2_transition_heatmap.png), [generating script](../src/research/ml.py).

Interpretation and caveat: Model-specific added predictive information; test observations are paired and are not an independent cohort.


## 15. Limitations

See [limitations](LIMITATIONS.md): observational measures, band discretization, starting-position boundaries, missingness, subgroup variation, conditional test uncertainty and exploratory multiplicity. Historical test-governance disclosures remain preserved. Model associations do not identify effects of institutions, clubs or gender.

## 16. Contribution

The contribution is an executable evidence chain spanning dataset characterization, historical experiment reproduction, paired Year-2 information, ordinal mobility, multivariable association, contextual interactions and engagement, with machine-readable provenance for each reported numerical result.

## 17. Reproducibility

Run `python scripts/run_all.py`; see [reproduction instructions](REPRODUCIBILITY.md), [audit](REPRODUCIBILITY_AUDIT.md) and [final JSON](../results/final_research_summary.json). Data and artifact manifests map reported results to executable generators. Historical discrepancies are retained rather than fitted away.

## 18. Next research steps

Evaluate ordinal/nonlinear specification sensitivity, repeated-split development uncertainty and external validation. Preregister confirmatory interaction hypotheses and assess predictor timing prospectively. Broader mechanism or dimension claims need designs and measurements capable of testing them.
