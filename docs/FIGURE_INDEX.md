# Generated figures

Each plot is generated from computed tables by `src/research/visualization.py`. PNG, PDF and SVG exports are listed in [figure catalog](../results/tables/figure_catalog.csv).

- [01_dataset_composition](../results/figures/01_dataset_composition.png): Dataset composition. Inputs: results/tables/categorical_distributions.csv. 

- [02_year1_cgpa_distribution](../results/figures/02_year1_cgpa_distribution.png): Year-1 CGPA bands. Inputs: results/tables/yearly_cgpa_distribution.csv. 

- [03_year2_cgpa_distribution](../results/figures/03_year2_cgpa_distribution.png): Year-2 CGPA bands. Inputs: results/tables/yearly_cgpa_distribution.csv. 

- [04_year1_vs_year2_transition_heatmap](../results/figures/04_year1_vs_year2_transition_heatmap.png): Year-1 to Year-2 academic transitions. Inputs: results/tables/transition_counts.csv; results/tables/transition_row_percent.csv. 

- [05_academic_trajectory_distribution](../results/figures/05_academic_trajectory_distribution.png): Academic mobility. Inputs: results/tables/trajectory_distribution.csv. 

- [06_year1_year2_persistence](../results/figures/06_year1_year2_persistence.png): Academic persistence across faculty contexts. Inputs: results/tables/subgroup_persistence.csv. 

- [07_hsc_subject_vs_year1](../results/figures/07_hsc_subject_vs_year1.png): HSC achievement and high Year-1 CGPA. Inputs: results/tables/descriptive_contrasts.csv. High performance uses the upper two CGPA bands. Wilson confidence intervals; unadjusted comparisons.

- [08_hsc_subject_vs_year2](../results/figures/08_hsc_subject_vs_year2.png): HSC achievement and high Year-2 CGPA. Inputs: results/tables/descriptive_contrasts.csv. High performance uses the upper two CGPA bands. Wilson confidence intervals; unadjusted comparisons.

- [09_hsc_math_gender_interaction](../results/figures/09_hsc_math_gender_interaction.png): HSC Mathematics by gender: Year-1 high CGPA. Inputs: results/tables/interaction_marginal_probabilities.csv; results/tables/interaction_tests.csv. Adjusted standardized probabilities with 95% delta-method CI. Interaction FDR q=2.79e-11; exploratory association.

- [10_trajectory_by_gender](../results/figures/10_trajectory_by_gender.png): Academic mobility by gender. Inputs: results/tables/trajectory_by_gender.csv. 

- [11_trajectory_by_living_arrangement](../results/figures/11_trajectory_by_living_arrangement.png): Academic mobility by living arrangement. Inputs: results/tables/trajectory_by_living_arrangement.csv. 

- [12_club_vs_research](../results/figures/12_club_vs_research.png): Club involvement and research participation. Inputs: results/tables/descriptive_contrasts.csv; results/tables/adjusted_marginal_probabilities.csv. 

- [13_club_research_by_college_background](../results/figures/13_club_research_by_college_background.png): Clubs and research participation by college background. Inputs: results/tables/interaction_marginal_probabilities.csv; results/tables/interaction_tests.csv. Exploratory nested interaction: p=0.0242; FDR q=0.0364. Curves alone do not establish interaction.

- [14_club_research_by_faculty](../results/figures/14_club_research_by_faculty.png): Clubs and research participation by faculty. Inputs: results/tables/interaction_marginal_probabilities.csv; results/tables/interaction_tests.csv. Exploratory nested interaction: p=0.196; FDR q=0.218. Curves alone do not establish interaction.

- [15_research_by_trajectory](../results/figures/15_research_by_trajectory.png): Research participation across academic trajectories. Inputs: results/tables/descriptive_contrasts.csv; results/tables/interaction_marginal_probabilities.csv. 

- [16_raw_vs_adjusted_effects](../results/figures/16_raw_vs_adjusted_effects.png): Research associations before and after adjustment. Inputs: results/tables/raw_vs_adjusted_effects.csv. Same outcome, coefficient contrast and complete-case rows; odds ratios remain observational associations.

- [17_adjusted_year1_forest_plot](../results/figures/17_adjusted_year1_forest_plot.png): Adjusted associations: year1. Inputs: results/tables/adjusted_year1.csv. Complete-case logistic association model, N=3514. Reference categories and q-values in source table.

- [18_adjusted_year2_forest_plot](../results/figures/18_adjusted_year2_forest_plot.png): Adjusted associations: year2. Inputs: results/tables/adjusted_year2.csv. Complete-case logistic association model, N=3514. Reference categories and q-values in source table.

- [19_adjusted_decline_forest_plot](../results/figures/19_adjusted_decline_forest_plot.png): Adjusted associations: decline. Inputs: results/tables/adjusted_decline.csv. Complete-case logistic association model, N=3482. Reference categories and q-values in source table.

- [20_adjusted_improvement_forest_plot](../results/figures/20_adjusted_improvement_forest_plot.png): Adjusted associations: improvement. Inputs: results/tables/adjusted_improvement.csv. Complete-case logistic association model, N=3482. Reference categories and q-values in source table.

- [21_adjusted_research_forest_plot](../results/figures/21_adjusted_research_forest_plot.png): Adjusted associations: research. Inputs: results/tables/adjusted_research.csv. Complete-case logistic association model, N=3445. Reference categories and q-values in source table.
