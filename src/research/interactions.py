"""Dedicated nested-model interaction tests and standardized probabilities."""
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from .common import *
from .adjusted import model_data, specifications, background_terms, nominal, fit, coefficients, marginal

def interaction_analysis(df):
    data = model_data(df)
    specs = specifications()
    analyses = []
    for subject in ['HSCMath','HSCPhysics','HSCChemistry']:
        analyses.append((f'gender_{subject}_year1','high_year1',specs['year1'][1],nominal('Gender','Female'),subject+'Ordinal','Gender',subject+'Ordinal'))
    for control in [False,True]:
        analyses.append((f'gender_HSCMath_year2_{"with" if control else "without"}_year1','high_year2',
            background_terms()+(['Year1Band'] if control else []),nominal('Gender','Female'),'HSCMathOrdinal','Gender','HSCMathOrdinal'))
    analyses.extend([
        ('clubs_college_research','research',specs['research'][1],nominal('CollegeCategory','Private'),'StudentClubMemberOrdinal','CollegeCategory','StudentClubMemberOrdinal'),
        ('clubs_faculty_research','research',specs['research'][1],nominal('Faculty','CSE'),'StudentClubMemberOrdinal','Faculty','StudentClubMemberOrdinal'),
        ('gender_living_decline','declined',specs['decline'][1],nominal('Gender','Female'),nominal('LivingArrangement','With parents/family'),'Gender','LivingArrangement'),
    ])
    tests,coefs,predictions = [],[],[]
    for name,outcome,terms,left,right,factor,axis in analyses:
        baseline = outcome+' ~ '+' + '.join(terms)
        full = baseline+f' + {left}:{right}'
        print(f'  Testing {name}',flush=True)
        extended = fit(full,data)
        same_rows = data.loc[extended.model.data.row_labels]
        reduced = fit(baseline,same_rows)
        statistic = max(0,2*(extended.llf-reduced.llf))
        dof = int(extended.df_model-reduced.df_model)
        tests.append(dict(analysis=name,outcome=outcome,interaction=f'{left}:{right}',
            LR_statistic=statistic,df=dof,p_value=chi2.sf(statistic,dof),N=int(extended.nobs),
            unstable_model=extended.research_unstable,specification=full,
            note='Nested likelihood-ratio test with identical complete-case rows.'))
        coefs.append(coefficients(extended,name,full))
        for level in sorted(same_rows[factor].dropna().unique()):
            table = marginal(extended,data,axis,sorted(same_rows[axis].dropna().unique()),{factor:level})
            table['analysis'] = name
            table['subgroup'] = level
            table['unstable_model'] = extended.research_unstable
            predictions.append(table)
    # I: trajectory is an adjusted term rather than a product interaction.
    terms = ['Year2Band',nominal('CurrentInstitution','KUET'),nominal('Faculty','CSE'),
        'StudentClubMemberOrdinal',nominal('CollegeCategory','Private'),nominal('Gender','Female'),
        'HSCMathOrdinal','HSCPhysicsOrdinal','HSCChemistryOrdinal']
    baseline = 'research ~ '+' + '.join(terms)
    full = baseline+" + C(Trajectory, Treatment(reference='Stable'))"
    extended = fit(full,data)
    reduced = fit(baseline,data.loc[extended.model.data.row_labels])
    statistic = max(0,2*(extended.llf-reduced.llf))
    dof = int(extended.df_model-reduced.df_model)
    tests.append(dict(analysis='trajectory_research_adjusted',outcome='research',interaction='Overall trajectory term',
        LR_statistic=statistic,df=dof,p_value=chi2.sf(statistic,dof),N=int(extended.nobs),
        unstable_model=extended.research_unstable,specification=full,
        note='Adjusted overall categorical trajectory term, not a product interaction.'))
    coefs.append(coefficients(extended,'trajectory_research_adjusted',full))
    table = marginal(extended,data,'Trajectory',['Declined','Stable','Improved'])
    table['analysis'] = 'trajectory_research_adjusted'
    table['subgroup'] = 'All complete cases'
    table['unstable_model'] = extended.research_unstable
    predictions.append(table)
    tests = pd.DataFrame(tests)
    tests['q_value'] = multipletests(tests.p_value,method='fdr_bh')[1]
    tests['supported'] = (tests.q_value<.05)&(~tests.unstable_model)
    save_table('interaction_tests',tests)
    save_table('interaction_coefficients',pd.concat(coefs,ignore_index=True))
    save_table('interaction_marginal_probabilities',pd.concat(predictions,ignore_index=True))
    return tests

if __name__ == '__main__':
    interaction_analysis(load_population()[1])
