"""Auditable evidence synthesis and comparison to nonauthoritative checkpoints."""
import json
import numpy as np
import pandas as pd
from .common import *

def read(name):
    return pd.read_csv(TABLES/(name+'.csv'))

def factor_for_term(term):
    if term.startswith('C('):
        return term[2:].split(',')[0].split(')')[0]
    return term.removesuffix('Ordinal')

def build_evidence(adjusted,associations,interactions):
    factor_tests = read('adjusted_factor_tests')
    contrasts = read('descriptive_contrasts')
    outcomes = {'year1':'Year1Band','year2':'Year2Band','improvement':'Trajectory','decline':'Trajectory','research':'research'}
    rows = []
    for model,coefficients in adjusted.items():
        for _,coef in coefficients[coefficients.term!='Intercept'].iterrows():
            factor = factor_for_term(coef.term)
            outcome = outcomes[model]
            unadjusted = associations[(associations.factor==factor)&(associations.outcome==outcome)]
            unadjusted = unadjusted.iloc[0] if len(unadjusted) else None
            overall = factor_tests[(factor_tests.model==model)&(factor_tests.factor==factor)].iloc[0]
            corresponding = {'year1':'high_year1','year2':'high_year2','improvement':'improved','decline':'declined','research':'research'}[model]
            descriptive = contrasts[(contrasts.factor==factor)&(contrasts.outcome==corresponding)]
            text = '; '.join(f'{r.group}: {r.percent:.3f}% (N={r.N})' for _,r in descriptive.iterrows())
            if not text:
                text = 'See categorical distributions and model coefficient contrast.'
            related = interactions[interactions.interaction.str.contains(factor,regex=False)&(interactions.outcome==corresponding)]
            survives = bool(overall.q_value<.05)
            context = bool(len(related) and related.supported.any())
            raw_significant = bool(unadjusted is not None and unadjusted.q_value<.05)
            category = ('Central' if factor=='Year1Band' and model=='year2' and survives else
                'Context-dependent' if context else 'Strong secondary' if survives else
                'Weakens after adjustment' if raw_significant else 'Negative/non-conclusive')
            contrast_survives = coef.q_value<.05
            interpretation = (f'{coef.term}: adjusted odds ratio {coef.odds_ratio:.4g} for {corresponding}; '+
                ('coefficient retained after within-model FDR.' if contrast_survives else 'coefficient not retained after within-model FDR.'))
            rows.append(dict(factor=factor,outcome=corresponding,model=model,coefficient_contrast=coef.term,
                descriptive_result=text,unadjusted_effect=unadjusted.cramers_v if unadjusted is not None else np.nan,
                unadjusted_effect_type='Cramers V: full band or trajectory-category outcome',
                unadjusted_p=unadjusted.p_value if unadjusted is not None else np.nan,
                FDR_q=unadjusted.q_value if unadjusted is not None else np.nan,
                adjusted_effect_OR_or_beta=coef.odds_ratio,adjusted_CI_low=coef.CI_low,adjusted_CI_high=coef.CI_high,
                adjusted_p=coef.p_value,adjusted_q=coef.q_value,adjusted_factor_p=overall.p_value,
                adjusted_factor_q=overall.q_value,interaction_test_if_any='; '.join(f'{r.analysis}: p={r.p_value:.4g}, q={r.q_value:.4g}' for _,r in related.iterrows()),
                sample_size=coef.model_N,survives_adjustment=survives,contrast_survives_adjustment=contrast_survives,
                evidence_category=category,interpretation=interpretation,
                caution='Exploratory observational association; raw multiclass association and adjusted binary coefficient answer different questions.',
                supporting_script='src/research/adjusted.py',
                supporting_table=f'results/tables/adjusted_{model}.csv',
                unadjusted_table='results/tables/master_unadjusted_associations.csv',
                supporting_figure=f'results/figures/{dict(year1="17",year2="18",decline="19",improvement="20",research="21")[model]}_adjusted_{model}_forest_plot.png'))
    return save_table('master_evidence_table',pd.DataFrame(rows))

def validate_conclusions(persistence,distribution,configs,delta,adjusted,associations,interactions):
    tests=read('adjusted_factor_tests')
    def coefficient(model,factor):
        table=adjusted[model]
        return table.loc[table.term==factor+'Ordinal'].iloc[0]
    def adjusted_sig(model,factor):
        return tests.loc[(tests.model==model)&(tests.factor==factor),'q_value'].iloc[0]<.05
    def v(factor,outcome):
        return float(associations.loc[(associations.factor==factor)&(associations.outcome==outcome),'cramers_v'].iloc[0])
    def interaction(name):
        return interactions.loc[interactions.analysis==name].iloc[0]
    math,physics,chem = [coefficient('year1',x) for x in ['HSCMath','HSCPhysics','HSCChemistry']]
    gender_distribution=read('trajectory_by_gender')
    movement_gap=gender_distribution.set_index('group').loc['Female','Stable_percent']-gender_distribution.set_index('group').loc['Male','Stable_percent']
    raw_adjusted=read('raw_vs_adjusted_effects')
    weakens=(raw_adjusted.raw_q<.05)&(raw_adjusted.adjusted_p>=.05)
    club=coefficient('research','StudentClubMember')
    added=delta['delta_accuracy']
    # A checkpoint/mechanistic interpretation is broader than the observational
    # design. It remains MODIFIED even when the quantitative pattern is clear.
    entries = [
        ('Year-1 standing strongly persists into Year 2.',persistence['rho_CI_low']>.7,
            f"Spearman rho={persistence['spearman_rho']:.6f}, 95% bootstrap CI [{persistence['rho_CI_low']:.6f}, {persistence['rho_CI_high']:.6f}], Cramers V={persistence['cramers_v']:.6f}.",
            'Academic persistence within this dataset; ordinal categories do not measure exact score changes.',
            'persistence_statistics','04_year1_vs_year2_transition_heatmap','src/research/statistics.py'),
        ('Year 1 is a major academic checkpoint.',True,
            f"Added test accuracy={added*100:.6f} percentage points; gender–Mathematics Year-2 interaction q before={interaction('gender_HSCMath_year2_without_year1').q_value:.6g}, after Year-1 adjustment={interaction('gender_HSCMath_year2_with_year1').q_value:.6g}.",
            'MODIFIED: Year-1 standing captures substantial additional information; temporal mediation and causal pathways are not identified.',
            'year2_added_information','09_hsc_math_gender_interaction','src/research/ml.py'),
        ('HSC preparation relates more strongly to academic level than mobility.',
            all(v(s,'Year1Band')>v(s,'Trajectory') and v(s,'Year2Band')>v(s,'Trajectory') for s in ['HSCMath','HSCPhysics','HSCChemistry']),
            '; '.join(f"{s}: V(Y1)={v(s,'Year1Band'):.6f}, V(Y2)={v(s,'Year2Band'):.6f}, V(trajectory)={v(s,'Trajectory'):.6f}" for s in ['HSCMath','HSCPhysics','HSCChemistry']),
            'Descriptive comparison of observed association strengths, with different outcome cardinalities.',
            'master_unadjusted_associations','07_hsc_subject_vs_year1','src/research/statistics.py'),
        ('Mathematics has the clearest independent Year-1 association, followed by Physics.',
            math.q_value<.05 and physics.q_value<.05 and math.odds_ratio>physics.odds_ratio and chem.q_value>=.05,
            '; '.join(f'{s}: OR={r.odds_ratio:.6f}, CI [{r.CI_low:.6f}, {r.CI_high:.6f}], q={r.q_value:.6g}' for s,r in [('Mathematics',math),('Physics',physics),('Chemistry',chem)]),
            'Ordered per-band estimates depend on linear log-odds specification; ranking point estimates is not a formal difference test.',
            'adjusted_year1','17_adjusted_year1_forest_plot','src/research/adjusted.py'),
        ('Academic attainment and mobility have different associated factors.',
            any(adjusted_sig('year1',f)!=adjusted_sig('decline',f) for f in ['HSCMath','HSCChemistry','Gender']),
            '; '.join(f"{f}: adjusted factor q(Y1)={tests[(tests.model=='year1')&(tests.factor==f)].q_value.iloc[0]:.6g}, q(decline)={tests[(tests.model=='decline')&(tests.factor==f)].q_value.iloc[0]:.6g}" for f in ['HSCMath','HSCChemistry','Gender']),
            'MODIFIED: association profiles differ descriptively; differences in significance are not tests of coefficient differences.',
            'adjusted_factor_tests','19_adjusted_decline_forest_plot','src/research/adjusted.py'),
        ('Gender trajectory contrasts involve stability and movement in both directions.',movement_gap>0,
            '; '.join(f'{r.group}: stable={r.Stable_percent:.6f}%, improved={r.Improved_percent:.6f}%, declined={r.Declined_percent:.6f}%, N={r.N}' for _,r in gender_distribution.iterrows()),
            'Within-dataset descriptive distribution; avoid attributing differences to gender itself.',
            'trajectory_by_gender','10_trajectory_by_gender','src/research/visualization.py'),
        ('Institutional context is associated with mobility and research participation.',
            adjusted_sig('research','CurrentInstitution') and (adjusted_sig('decline','CurrentInstitution') or adjusted_sig('improvement','CurrentInstitution')),
            f"Institution V(trajectory)={v('CurrentInstitution','Trajectory'):.6f}, V(research)={v('CurrentInstitution','research'):.6f}; model omnibus q-values in adjusted factor tests.",
            'Institutional heterogeneity does not justify university rankings; subgroup sizes vary.',
            'adjusted_factor_tests','06_year1_year2_persistence','src/research/adjusted.py'),
        ('Clubs are associated more strongly with research than CGPA.',club.q_value<.05 and v('StudentClubMember','research')>max(v('StudentClubMember','Year1Band'),v('StudentClubMember','Year2Band')),
            f"Club OR(research)={club.odds_ratio:.6f}, CI [{club.CI_low:.6f}, {club.CI_high:.6f}], q={club.q_value:.6g}; V(research)={v('StudentClubMember','research'):.6f}, V(Y1)={v('StudentClubMember','Year1Band'):.6f}, V(Y2)={v('StudentClubMember','Year2Band'):.6f}.",
            'Concurrent measurement; selection into clubs and research is possible.',
            'adjusted_research','12_club_vs_research','src/research/adjusted.py'),
        ('College background independently relates to research more clearly than academic standing.',
            adjusted_sig('research','CollegeCategory') and not adjusted_sig('year1','CollegeCategory') and not adjusted_sig('year2','CollegeCategory'),
            '; '.join(f"{m}: college-category omnibus q={tests[(tests.model==m)&(tests.factor=='CollegeCategory')].q_value.iloc[0]:.6g}" for m in ['year1','year2','research']),
            'Conditional odds ratios use Private as reference; this is not a causal or population-comparability claim.',
            'adjusted_factor_tests','21_adjusted_research_forest_plot','src/research/adjusted.py'),
        ('Research participation is not represented by CGPA alone.',adjusted_sig('research','CurrentInstitution') and adjusted_sig('research','StudentClubMember'),
            f"After Year-2-band adjustment: club OR={club.odds_ratio:.6f}, q={club.q_value:.6g}; institutional omnibus q={tests[(tests.model=='research')&(tests.factor=='CurrentInstitution')].q_value.iloc[0]:.6g}.",
            'Independent associations in this specification; no latent-dimension measurement model was fitted.',
            'adjusted_research','21_adjusted_research_forest_plot','src/research/adjusted.py'),
        ('Some unadjusted associations weaken after adjustment.',bool(weakens.any()),
            '; '.join(f'{r.model}/{r.factor}: unadjusted OR={r.raw_OR:.6f}, adjusted OR={r.adjusted_OR:.6f}, adjusted p={r.adjusted_p:.6g}' for _,r in raw_adjusted[weakens].iterrows()),
            'Adjusted conditional ORs can differ through confounding, conditioning and noncollapsibility; attenuation alone does not establish a mechanism.',
            'raw_vs_adjusted_effects','16_raw_vs_adjusted_effects','src/research/adjusted.py'),
        ('Year-1 CGPA adds information for Year-2 prediction.',delta['delta_accuracy_CI_low']>0,
            f"Delta accuracy={added*100:.6f} percentage points, paired 95% bootstrap CI [{delta['delta_accuracy_CI_low']*100:.6f}, {delta['delta_accuracy_CI_high']*100:.6f}], exact McNemar p={delta['exact_McNemar_p']:.6g}.",
            'Model-specific added predictive information; test observations are paired and are not an independent cohort.',
            'year2_added_information','04_year1_vs_year2_transition_heatmap','src/research/ml.py'),
    ]
    rows=[]
    for i,(claim,supported,statistic,caution,table,figure,script) in enumerate(entries,1):
        status='MODIFIED' if supported and (i in [2,5] or caution.startswith('MODIFIED')) else 'VERIFIED' if supported else 'NOT SUPPORTED'
        rows.append(dict(conclusion_id=i,candidate_conclusion=claim,status=status,statistics=statistic,caution=caution,
            supporting_script=script,supporting_table=f'results/tables/{table}.csv',supporting_figure=f'results/figures/{figure}.png'))
    return save_table('verified_conclusions',pd.DataFrame(rows))

def historical_discrepancies(configs,delta,weka,persistence,distribution,adjusted,associations,interactions):
    # Historical expected values are checkpoint metadata ONLY. Results and
    # documents are generated from the computed values passed into this function.
    rows=[]
    def add(name,expected,computed,tolerance,source,reason):
        rows.append(dict(checkpoint=name,historical_expected=expected,computed=computed,
            difference=computed-expected,tolerance=tolerance,within_tolerance=abs(computed-expected)<=tolerance,
            source_table=source,input_dataset=relative(DATA),configuration=reason,
            disposition='Authoritative computed result retained; checkpoint never used for fitting/selection.'))
    expected_ml={'year1':(.5985,.5984,.6033),'year2_without_year1':(.5587,.5569,.5670),'year2_with_year1':(.8712,.8713,.8736)}
    for study,expected in expected_ml.items():
        for metric,e in zip(['accuracy','weighted_f1','macro_f1'],expected):
            add(study+'/'+metric,e,configs[study]['test_metrics'][metric],.0001,'results/metrics/'+study+'.json',
                'Historical frozen Year-1 configuration and preserved splits.' if study=='year1' else 'Shared Year-2 split; common forest selected by mean C1/C2 validation F1; original Year-2 configuration unavailable.')
    add('research_historical_weka/accuracy',.8103,weka['accuracy'],.0001,'results/metrics/research_historical_weka.json','Existing 3475-row protocol; API refit with original header and frozen parameters.')
    add('research_historical_weka/participated_f1',.699,weka['participated_f1'],.001,'results/metrics/research_historical_weka.json','Existing original nominal-attribute WEKA configuration.')
    for level,expected in [('Stable',75.),('Improved',10.84),('Declined',14.16)]:
        add('trajectory/'+level,expected,float(distribution.loc[distribution.trajectory==level,'percent'].iloc[0]),.01,'results/tables/trajectory_distribution.csv','Full population; exact ordered-band differences.')
    add('persistence/spearman',.847,persistence['spearman_rho'],.001,'results/tables/persistence_statistics.csv','Full-population ordinal ranks.')
    add('persistence/cramers_v',.695,persistence['cramers_v'],.001,'results/tables/persistence_statistics.csv','Uncorrected Pearson chi-square; conventional uncorrected Cramers V.')
    expected_v={'HSCMath':(.170,.156,.033,.109),'HSCPhysics':(.155,.134,.060,.104),'HSCChemistry':(.149,.135,.081,.107),
        'CurrentInstitution':(.169,.133,.125,.327),'CollegeCategory':(.176,.128,.080,.254),'Gender':(.118,.122,.188,.024),
        'LivingArrangement':(.104,.079,.135,.195),'StudentClubMember':(.064,.053,.087,.252),'PartTimeEmployment':(.069,.049,.069,.153),
        'ComputerAvailability':(.072,.058,.099,.025),'MotherEducation':(.101,.090,.084,.099),'HouseholdIncomeYearly':(.078,.052,.031,.091),
        'DepartmentPreferenceOrder':(.065,.064,.089,.058),'UniversityPreferenceOrder':(.058,.046,.076,.089),
        'FinancialSupportSource':(.106,.085,.109,.052),'CollegeLocation':(.088,.070,.017,.148),'FatherEducation':(.086,.063,.088,.103)}
    for factor,expectations in expected_v.items():
        for outcome,expected in zip(['Year1Band','Year2Band','Trajectory','research'],expectations):
            value=associations.loc[(associations.factor==factor)&(associations.outcome==outcome),'cramers_v'].iloc[0]
            add(f'CramersV/{factor}/{outcome}',expected,value,.001,'results/tables/master_unadjusted_associations.csv',
                'Full population; pairwise available cases; binary research (Yes versus No); bilingual labels normalized only.')
    expected_or=[('year2','Year1Band',11.10),('year1','HSCMathOrdinal',1.70),('year1','HSCPhysicsOrdinal',1.24),
        ('decline','HSCChemistryOrdinal',.75),('decline',"C(Gender, Treatment(reference='Female'))[T.Male]",1.82),
        ('improvement',"C(Gender, Treatment(reference='Female'))[T.Male]",2.24),('decline','Year1Band',1.65),('improvement','Year1Band',.60),
        ('research','StudentClubMemberOrdinal',1.354),('research',"C(CollegeCategory, Treatment(reference='Private'))[T.Public]",.669)]
    for model,term,expected in expected_or:
        value=adjusted[model].loc[adjusted[model].term==term,'odds_ratio'].iloc[0]
        add(f'adjusted/{model}/{term}',expected,value,.01,'results/tables/adjusted_'+model+'.csv',
            'Explicit all-background trend-score specification; context covariates and Year-1/Year-2 controls as documented; historical adjusted specification unavailable.')
    expected_p={'gender_HSCMath_year1':1e-12,'gender_HSCPhysics_year1':1.4e-10,'gender_HSCChemistry_year1':.0003,
        'gender_HSCMath_year2_without_year1':2.4e-12,'gender_HSCMath_year2_with_year1':.194,
        'clubs_college_research':.034,'clubs_faculty_research':.0017,'gender_living_decline':.078,'trajectory_research_adjusted':.0155}
    for name,expected in expected_p.items():
        value=interactions.loc[interactions.analysis==name,'p_value'].iloc[0]
        add('interaction/'+name,expected,value,max(expected*.2,1e-13),'results/tables/interaction_tests.csv',
            'Nested LR tests on identical complete cases, explicit adjusted model; historical test type/covariates unavailable. Math Y1 historical entry is an upper-bound checkpoint.')
    contrasts=read('descriptive_contrasts')
    for factor,expected_rates in {'HSCMath':(58.1,34.3,57.6,32.6),'HSCPhysics':(59.8,36.0,58.0,34.7),'HSCChemistry':(55.2,37.8,56.5,36.0)}.items():
        for (outcome,group),expected in zip([('high_year1','190–200'),('high_year1','All lower categories'),
            ('high_year2','190–200'),('high_year2','All lower categories')],expected_rates):
            computed=contrasts.loc[(contrasts.factor==factor)&(contrasts.outcome==outcome)&(contrasts.group==group),'percent'].iloc[0]
            add(f'raw_rate/{factor}/{outcome}/{group}',expected,computed,.1,'results/tables/descriptive_contrasts.csv','Full-population upper-two-band proportions; observed group denominators.')
    gender=read('trajectory_by_gender').set_index('group')
    for group,expectations in {'Female':(5.5,86.0,8.4),'Male':(13.7,69.1,17.3)}.items():
        for movement,expected in zip(['Improved','Stable','Declined'],expectations):
            add(f'raw_rate/gender/{group}/{movement}',expected,gender.loc[group,movement+'_percent'],.1,
                'results/tables/trajectory_by_gender.csv','Full population; mutually exclusive movement classes.')
    clubs={'No':24.4,'Yes, one club/organization':45.9,'Yes, two clubs/organizations':46.9,'Yes, three or more clubs/organizations':54.8}
    for group,expected in clubs.items():
        computed=contrasts.loc[(contrasts.factor=='StudentClubMember')&(contrasts.outcome=='research')&(contrasts.group==group),'percent'].iloc[0]
        add(f'raw_rate/clubs/{group}',expected,computed,.1,'results/tables/descriptive_contrasts.csv','Available research outcomes; clubs measured at survey.')
    margins=read('adjusted_marginal_probabilities')
    for level,expected in enumerate([29.9,36.0,42.5,49.2]):
        computed=float(margins.loc[(margins.model=='research')&(margins.factor=='StudentClubMemberOrdinal')&(margins.level.astype(str)==str(level)),'probability'].iloc[0])*100
        add(f'adjusted_probability/clubs/{level}',expected,computed,.1,'results/tables/adjusted_marginal_probabilities.csv',
            'Standardized full adjusted model; empirical complete-case covariate population; historical specification unavailable.')
    for outcome,expected_public,expected_private in [('high_year2',38.9,38.8),('low_year2',11.5,19.1),('research',25.8,50.6)]:
        for group,expected in [('Public',expected_public),('Private',expected_private)]:
            computed=contrasts.loc[(contrasts.factor=='CollegeCategory')&(contrasts.outcome==outcome)&(contrasts.group==group),'percent'].iloc[0]
            add(f'raw_rate/college/{group}/{outcome}',expected,computed,.1,'results/tables/descriptive_contrasts.csv','Current-population available outcome; no duplicate exclusions.')
    for group,expected in [('1st choice',44.1),('5th choice or lower',33.1)]:
        computed=contrasts.loc[(contrasts.factor=='DepartmentPreferenceOrder')&(contrasts.outcome=='high_year2')&(contrasts.group==group),'percent'].iloc[0]
        add(f'raw_rate/department_preference/{group}',expected,computed,.1,'results/tables/descriptive_contrasts.csv','Operational high CGPA uses upper two bands.')
    for group,expected in [('Declined',31.),('Stable',36.),('Improved',40.)]:
        computed=contrasts.loc[(contrasts.factor=='Trajectory')&(contrasts.outcome=='research')&(contrasts.group==group),'percent'].iloc[0]
        add(f'raw_rate/trajectory_research/{group}',expected,computed,1.,'results/tables/descriptive_contrasts.csv','Available research outcomes; raw rate checkpoint.')
    subgroup=read('subgroup_persistence')
    for faculty,expected in [('CSE',.843),('EEE',.858),('ME',.821),('CE',.824),('Others',.884)]:
        computed=subgroup.loc[(subgroup.factor=='Faculty')&(subgroup.subgroup==faculty),'spearman_rho'].iloc[0]
        add('faculty_rho/'+faculty,expected,computed,.001,'results/tables/subgroup_persistence.csv','Original faculty categories; no descriptive pooling.')
    for model,term,expected_low,expected_high in [('year2','Year1Band',9.44,13.04),('year1','HSCMathOrdinal',1.49,1.95),
        ('year1','HSCPhysicsOrdinal',1.08,1.42),('research','StudentClubMemberOrdinal',1.25,1.46)]:
        coefficient=adjusted[model].loc[adjusted[model].term==term].iloc[0]
        for bound,expected in [('low',expected_low),('high',expected_high)]:
            add(f'adjusted_CI/{model}/{term}/{bound}',expected,coefficient['CI_'+bound],.01,'results/tables/adjusted_'+model+'.csv',
                'Wald CI under documented full adjusted specification; historical covariate/test specification unavailable.')
    return save_table('historical_checkpoint_comparison',pd.DataFrame(rows))
