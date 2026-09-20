"""Scientific/data-governance assertions, not fitted-metric checkpoint tests."""
import json
import numpy as np
import pandas as pd
import pytest
from src.research.common import *
from src.research.adjusted import specifications

def read(name):
    return pd.read_csv(TABLES/(name+'.csv'))

def config(study):
    return json.loads((METRICS/(study+'.json')).read_text(encoding='utf-8'))

def test_population_contract(population):
    raw,df=population
    assert raw.shape==(3516,32)
    assert set(REQUIRED_COLUMNS)==set(raw.columns)
    assert len(df)==len(raw)
    assert df.index.equals(raw.index)

def test_band_order_and_unknown_values():
    inputs=['Less than 3.00','3.00 - 3.25','3.25 - 3.50','3.50 - 3.75','Above 3.75']
    assert [cgpa_band(v) for v in inputs]==list(range(5))
    with pytest.raises(ValueError):
        cgpa_band('unknown')

def test_trajectory_sign_and_all_boundaries():
    old,new=np.meshgrid(range(5),range(5))
    delta=new-old
    classes=trajectory(delta)
    assert np.all(classes[delta>0]=='Improved')
    assert np.all(classes[delta==0]=='Stable')
    assert np.all(classes[delta<0]=='Declined')
    assert np.all(classes[old==4]!='Improved')
    assert np.all(classes[old==0]!='Declined')

def test_transition_totals_and_marginals(population):
    raw,df=population
    matrix=read('transition_counts').drop(columns='Year1Band').to_numpy()
    assert matrix.sum()==len(raw)
    assert np.array_equal(matrix.sum(axis=1),df.Year1Band.value_counts().sort_index())
    assert np.array_equal(matrix.sum(axis=0),df.Year2Band.value_counts().sort_index())
    normalized=read('transition_row_percent').drop(columns='Year1Band').to_numpy()
    assert np.allclose(normalized.sum(axis=1),100)
    assert np.allclose(normalized,matrix/matrix.sum(axis=1,keepdims=True)*100)

def test_percentages_and_high_outcomes(population):
    raw,df=population
    distribution=read('trajectory_distribution')
    assert distribution['count'].sum()==len(raw)
    assert np.isclose(distribution.percent.sum(),100)
    assert np.allclose(distribution.percent,100*distribution['count']/len(raw))
    assert np.array_equal(df.high_year1,df.Year1Band>=3)
    assert np.array_equal(df.high_year2,df.Year2Band>=3)
    assert np.array_equal(df.trajectory_delta,df.Year2Band-df.Year1Band)
    categories=read('categorical_distributions')
    assert np.allclose(categories.groupby('variable').percent_population.sum(),100)

def test_missing_research_not_encoded_as_negative(population):
    raw,df=population
    assert df.research.isna().sum()==raw.ResearchParticipation.isna().sum()
    assert np.isnan(research_binary(np.nan))
    assert research_binary('No')==0
    for label in ['Yes, as a research assistant','Yes, as part of a course','Participated']:
        assert research_binary(label)==1

def test_feature_timing_and_target_leakage():
    without=config('year2_without_year1');with_year1=config('year2_with_year1')
    assert 'Year1CGPA' not in without['features']
    assert 'Year1CGPA' in with_year1['features']
    assert set(with_year1['features'])-set(without['features'])=={'Year1CGPA'}
    for study in ['year1','year2_without_year1','year2_with_year1','research','research_historical_python']:
        c=config(study)
        assert c['target'] not in c['features']
        assert not set(c['features']) & {'Year2CGPA','Year2Band','research','ResearchParticipation','ResearchParticipationBinary','Trajectory','trajectory_delta','Timestamp'}
    assert 'Year1CGPA' not in config('year1')['features']
    assert not set(config('research')['features']) & {'StudentClubMember','LivingArrangement','PartTimeEmployment','FinancialSupportSource'}
    assert without['parameters']==with_year1['parameters']
    assert without['seed']==with_year1['seed']==SEED

@pytest.mark.parametrize('study',['year2','research'])
def test_split_disjointness_population_coverage(study,population):
    _,df=population
    ids=read(study+'_split_assignments')
    assert ids.row_id.is_unique
    expected=set(df.index) if study=='year2' else set(df.index[df.research.notna()])
    assert set(ids.row_id)==expected
    assert set(ids['split'])=={'train','validation','test'}
    classes=df.Year2Band if study=='year2' else df.research
    for name,group in ids.groupby('split'):
        assert set(classes.loc[group.row_id])==set(classes.dropna())

def test_paired_year2_predictions_and_deltas():
    old=read('year2_without_year1_test_predictions');new=read('year2_with_year1_test_predictions')
    assert np.array_equal(old.row_id,new.row_id)
    assert np.array_equal(old.actual,new.actual)
    added=json.loads((METRICS/'year2_added_information.json').read_text())
    for key in ['accuracy','weighted_f1','macro_f1']:
        expected=config('year2_with_year1')['test_metrics'][key]-config('year2_without_year1')['test_metrics'][key]
        assert np.isclose(expected,added['delta_'+key])
        assert added['delta_'+key+'_CI_low']<=expected<=added['delta_'+key+'_CI_high']

@pytest.mark.parametrize('name',['year1','year2','improvement','decline','research'])
def test_adjusted_schema_intervals_and_diagnostics(name):
    frame=read('adjusted_'+name)
    assert {'coefficient','odds_ratio','CI_low','CI_high','p_value','q_value','model_N','reference_categories','specification'}.issubset(frame.columns)
    assert np.isfinite(frame[['coefficient','odds_ratio','CI_low','CI_high','p_value']].to_numpy()).all()
    assert (frame.CI_low<=frame.odds_ratio).all() and (frame.odds_ratio<=frame.CI_high).all()
    assert np.allclose(np.exp(frame.coefficient),frame.odds_ratio)
    assert frame.model_N.nunique()==1
    assert frame.loc[frame.term!='Intercept','q_value'].between(0,1).all()
    diag=read('adjusted_model_diagnostics').set_index('model').loc[name]
    assert diag.converged and diag['rank']==diag.parameters
    assert not diag.unstable_model
    assert diag.N==frame.model_N.iloc[0]
    if name in ['year2','improvement','decline']:
        assert 'Year1Band' in set(frame.term)

def test_unadjusted_schema_and_fdr():
    frame=read('master_unadjusted_associations')
    assert {'factor','outcome','chi_square','df','p_value','q_value','cramers_v','N','fraction_expected_below_5'}.issubset(frame.columns)
    assert frame.q_value.between(0,1).all()
    assert frame.cramers_v.between(0,1).all()
    assert (frame.q_value+1e-12>=frame.p_value).all()

def test_marginal_probability_intervals():
    for name in ['adjusted_marginal_probabilities','interaction_marginal_probabilities']:
        frame=read(name)
        assert frame.probability.between(0,1).all()
        assert (frame.CI_low<=frame.probability).all() and (frame.probability<=frame.CI_high).all()
        assert (frame.subgroup_N<=frame.model_N).all()
        assert frame.CI_low.between(0,1).all() and frame.CI_high.between(0,1).all()

def test_interaction_support_matches_fdr():
    frame=read('interaction_tests')
    assert len(frame)==9
    assert np.array_equal(frame.supported,(frame.q_value<.05)&(~frame.unstable_model))
    assert (frame['df']>=1).all()

def test_evidence_and_conclusion_references():
    evidence=read('master_evidence_table')
    assert {'descriptive_result','unadjusted_effect','FDR_q','adjusted_effect_OR_or_beta',
        'adjusted_CI_low','adjusted_CI_high','sample_size','survives_adjustment','evidence_category','interpretation','caution'}.issubset(evidence.columns)
    for table in [evidence,read('verified_conclusions')]:
        for field in ['supporting_script','supporting_table','supporting_figure']:
            assert all((ROOT/path).is_file() for path in table[field].unique())
    conclusions=read('verified_conclusions')
    assert len(conclusions)==12
    assert set(conclusions.status)<={'VERIFIED','MODIFIED','NOT SUPPORTED'}

def test_manifests_resolve_to_existing_files():
    manifest=pd.read_csv(ROOT/'results/result_manifest.csv')
    assert {'artifact','generating_script','input_dataset','hash','created_at'}.issubset(manifest.columns)
    assert manifest.artifact.is_unique
    for row in manifest.itertuples():
        assert (ROOT/row.artifact).is_file()
        assert (ROOT/row.generating_script).is_file()
        for path in row.input_dataset.split('; '):
            assert (ROOT/path).is_file()
    run=json.loads((ROOT/'results/run_manifest.json').read_text(encoding='utf-8'))
    assert all((ROOT/path).is_file() for path in run['generated_artifact_list'])
    assert run['analysis_seed']==SEED

def test_preserved_input_hashes():
    from src.research.audit import verify_inputs
    assert verify_inputs()>0

def test_weka_independent_refit_matches_saved_model():
    weka=config('research_historical_weka')
    assert weka['refit_matches_frozen_predictions']
    assert weka['maximum_probability_difference_from_frozen']<1e-12
    predictions=pd.read_csv(LOGS/'weka_replay_predictions.csv')
    assert np.array_equal(predictions.frozen_predicted,predictions.refit_predicted)

def test_original_test_metrics_are_computable_from_predictions():
    from sklearn.metrics import accuracy_score,f1_score
    for study in ['year1','research','year2_without_year1','year2_with_year1']:
        predictions=read(study+'_test_predictions');saved=config(study)['test_metrics']
        assert np.isclose(accuracy_score(predictions.actual,predictions.predicted),saved['accuracy'])
        assert np.isclose(f1_score(predictions.actual,predictions.predicted,average='weighted'),saved['weighted_f1'])
