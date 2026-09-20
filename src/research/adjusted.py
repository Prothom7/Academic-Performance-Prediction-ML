"""Specified logistic association models with coefficient and fit diagnostics."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from scipy.special import expit, logit
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, brier_score_loss
from .common import *

def design_info(result):
    # statsmodels 0.15 renamed the stored Patsy specification.
    return getattr(result.model.data,'model_spec',None) or result.model.data.design_info

def references(result):
    import re
    values = []
    for factor,info in design_info(result).factor_infos.items():
        if info.type!='categorical':
            continue
        expression = factor.name()
        match = re.search(r"reference=['\"]([^'\"]+)['\"]",expression)
        reference = match.group(1) if match else info.categories[0]
        values.append(f'{expression}: {reference}')
    return '; '.join(values)

def nominal(column,reference=None):
    if reference is None:
        return f'C({column})'
    return f'C({column}, Treatment(reference={reference!r}))'

def model_data(df):
    df = df.copy()
    # Centering leaves all slope estimates unchanged and avoids a huge
    # extrapolated intercept at calendar year zero.
    df['HSCGraduationYear'] = df.HSCGraduationYear - 2020
    # ECE (N=4) remains intact in descriptive/subgroup tables. Pool solely in
    # fitted association models to avoid unestimable sparse-cell contrasts.
    df['Faculty'] = df.Faculty.replace({'ECE':'Others'})
    return df

def background_terms():
    references = {'Gender':'Female','CollegeCategory':'Private','ComputerAvailability':'Unavailable',
        'Faculty':'CSE','CurrentInstitution':'KUET','BirthOrder':'First-child'}
    return [f'{c}Ordinal' if c in ORDER_PREFIXES else c if c=='HSCGraduationYear' else nominal(c,references.get(c))
        for c in BASE_FEATURES]

def context_terms():
    return [nominal('LivingArrangement','With parents/family'), 'PartTimeEmploymentOrdinal',
        'UniversityMonthlyExpenditureOrdinal', nominal('FinancialSupportSource','Parents/Guardians'),
        'StudentClubMemberOrdinal']

def specifications():
    background = background_terms()
    return {
        'year1':('high_year1',background),
        'year2':('high_year2',background+['Year1Band']),
        'improvement':('improved',background+context_terms()+['Year1Band']),
        'decline':('declined',background+context_terms()+['Year1Band']),
        'research':('research',background+context_terms()+['Year2Band']),
    }

def fit(formula,data):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result = smf.glm(formula,data=data,family=sm.families.Binomial(),missing='drop').fit(maxiter=200,tol=1e-10)
    if not result.converged or not np.all(np.isfinite(result.params)):
        raise RuntimeError(f'Nonconverged/nonfinite logistic model: {formula}')
    result.research_warnings = sorted(set(str(w.message) for w in captured))
    # A large coefficient or standard error flags instability; do not silently
    # substitute penalized estimates with invalid Wald inferential claims.
    result.research_unstable = bool((result.bse > 10).any() or (result.params.abs() > 25).any())
    return result

def coefficients(result,name,formula):
    ci = result.conf_int()
    frame = pd.DataFrame({'term':result.params.index,'coefficient':result.params.to_numpy(),
        'standard_error':result.bse.to_numpy(),'odds_ratio':np.exp(result.params.to_numpy()),
        'CI_low':np.exp(ci[0].to_numpy()),'CI_high':np.exp(ci[1].to_numpy()),'p_value':result.pvalues.to_numpy()})
    frame['q_value'] = np.nan
    selected = frame.term != 'Intercept'
    frame.loc[selected,'q_value'] = multipletests(frame.loc[selected,'p_value'],method='fdr_bh')[1]
    frame['model'] = name
    frame['model_N'] = int(result.nobs)
    frame['specification'] = formula
    frame['reference_categories'] = references(result)
    frame['unstable_model'] = result.research_unstable
    return frame

def diagnostics(result,name,formula,population_n):
    x = result.model.exog
    y = result.model.endog
    predictions = np.asarray(result.fittedvalues)
    null = sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial()).fit()
    stat = 2*(result.llf-null.llf)
    return dict(model=name,N=int(result.nobs),population_N=population_n,
        omitted_due_to_missing=population_n-int(result.nobs),events=int(y.sum()),parameters=x.shape[1],
        events_per_parameter=float(y.sum()/x.shape[1]),converged=result.converged,
        rank=int(np.linalg.matrix_rank(x)),condition_number=float(np.linalg.cond(x)),
        log_likelihood=result.llf,AIC=result.aic,deviance=result.deviance,
        McFadden_R2=1-result.llf/null.llf,likelihood_ratio_statistic=stat,
        likelihood_ratio_p=chi2.sf(stat,result.df_model),in_sample_AUC=roc_auc_score(y,predictions),
        in_sample_Brier=brier_score_loss(y,predictions),minimum_fitted_probability=predictions.min(),
        maximum_fitted_probability=predictions.max(),unstable_model=result.research_unstable,
        warnings='; '.join(result.research_warnings),specification=formula,
        note='In-sample diagnostics describe association-model fit, not held-out predictive performance.')

def marginal(result,df,factor,levels,extra=None):
    # Standardize over the model's complete-case empirical covariate population.
    base = df.loc[result.model.data.row_labels].copy()
    rows = []
    covariance = np.asarray(result.cov_params())
    for level in levels:
        data = base.copy()
        data[factor] = level
        if extra:
            for col,value in extra.items():
                data[col] = value
        x = np.asarray(build_design_matrices([design_info(result)],data)[0])
        p = expit(x @ np.asarray(result.params))
        mean = p.mean()
        gradient = np.mean(x*(p*(1-p))[:,None],axis=0)
        se = np.sqrt(max(0,gradient@covariance@gradient))
        # Delta method on the logit of the averaged probability.
        logit_se = se/(mean*(1-mean))
        ci = expit(logit(mean)+np.array([-1,1])*1.959963984540054*logit_se)
        mask = base[factor]==level
        if extra:
            for col,value in extra.items():
                mask &= base[col]==value
        rows.append(dict(factor=factor,level=level,probability=mean,CI_low=ci[0],CI_high=ci[1],
            standard_error=se,model_N=len(base),subgroup_N=int(mask.sum()),
            conditioning='; '.join(f'{k}={v}' for k,v in (extra or {}).items()),
            CI_method='delta method, logit of standardized mean; covariate distribution fixed'))
    return pd.DataFrame(rows)

def adjusted_models(df):
    data = model_data(df)
    models,tables,diags,margins = {},{},[],[]
    for name,(outcome,terms) in specifications().items():
        formula = outcome+' ~ '+' + '.join(terms)
        print(f'  Fitting adjusted {name}',flush=True)
        result = fit(formula,data)
        if result.research_unstable:
            raise RuntimeError(f'Unstable primary adjusted model: {name}; inspect sparse cells before reporting.')
        models[name] = result
        tables[name] = save_table('adjusted_'+name,coefficients(result,name,formula))
        diags.append(diagnostics(result,name,formula,len(df)))
        factors = ['StudentClubMemberOrdinal','CollegeCategory'] if name=='research' else ['LivingArrangement'] if name in ['improvement','decline'] else []
        for factor in factors:
            table = marginal(result,data,factor,sorted(data[factor].dropna().unique()))
            table['model'] = name
            margins.append(table)
    save_table('adjusted_model_diagnostics',pd.DataFrame(diags))
    save_table('adjusted_marginal_probabilities',pd.concat(margins,ignore_index=True))
    rows = []
    for name,result in models.items():
        rows.append(dict(model=name,specification=result.model.formula,N=int(result.nobs),
            reference_categories=tables[name].reference_categories.iloc[0],
            preprocessing='Complete cases for each specification; ordered trend scores; ECE pooled with Others only in fitted models.'))
    save_table('model_specifications',pd.DataFrame(rows))
    comparisons,omnibus = [],[]
    for name,result in models.items():
        data_same = data.loc[result.model.data.row_labels]
        outcome,terms = specifications()[name]
        for factor in BASE_FEATURES+(['Year1Band'] if name in ['year2','improvement','decline'] else [])+(['LivingArrangement','PartTimeEmployment','UniversityMonthlyExpenditure','FinancialSupportSource','StudentClubMember'] if name in ['research','decline','improvement'] else [])+(['Year2Band'] if name=='research' else []):
            matching = [i for i,t in enumerate(result.params.index) if t==factor or t==factor+'Ordinal' or t.startswith('C('+factor+',') or t.startswith('C('+factor+')')]
            if not matching:
                continue
            contrast = np.eye(len(result.params))[matching]
            wald = result.wald_test(contrast,scalar=True)
            omnibus.append(dict(model=name,factor=factor,chi_square=float(wald.statistic),df=len(matching),
                p_value=float(wald.pvalue),N=int(result.nobs),test='Joint Wald test of the factor coefficients'))
        for factor in ['HSCMath','HSCPhysics','HSCChemistry','Gender','CollegeCategory','StudentClubMember']:
            term = next((t for t in terms if t==factor+'Ordinal' or t.startswith('C('+factor+',')),None)
            if term is None:
                continue
            raw = fit(outcome+' ~ '+term,data_same)
            raw_coefficients = coefficients(raw,name,outcome+' ~ '+term)
            adjusted_coefficients = tables[name]
            for _,r in raw_coefficients[raw_coefficients.term!='Intercept'].iterrows():
                adjusted_row = adjusted_coefficients.loc[adjusted_coefficients.term==r.term].iloc[0]
                comparisons.append(dict(model=name,factor=factor,term=r.term,N=int(result.nobs),
                    raw_OR=r.odds_ratio,raw_CI_low=r.CI_low,raw_CI_high=r.CI_high,raw_p=r.p_value,
                    adjusted_OR=adjusted_row.odds_ratio,adjusted_CI_low=adjusted_row.CI_low,
                    adjusted_CI_high=adjusted_row.CI_high,adjusted_p=adjusted_row.p_value,
                    note='Same outcome, contrast and complete-case rows; ORs on the same scale.'))
    omnibus = pd.DataFrame(omnibus)
    omnibus['q_value'] = omnibus.groupby('model').p_value.transform(lambda p:multipletests(p,method='fdr_bh')[1])
    save_table('adjusted_factor_tests',omnibus)
    comparisons = pd.DataFrame(comparisons)
    comparisons['raw_q'] = comparisons.groupby('model').raw_p.transform(lambda p:multipletests(p,method='fdr_bh')[1])
    save_table('raw_vs_adjusted_effects',comparisons)
    return models,tables
