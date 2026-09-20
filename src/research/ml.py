"""Frozen historical reproductions and directly comparable Year-2 experiments."""
import itertools
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
    cohen_kappa_score, balanced_accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_fscore_support, matthews_corrcoef)
from statsmodels.stats.proportion import proportion_confint
from .common import *

GRID = [dict(n_estimators=n,max_features=k,min_samples_leaf=m)
    for n,k,m in itertools.product([100,250,500],['sqrt',2,4],[1,2,3])]

def pipeline(x,parameters,impute=True):
    numeric = ['HSCGraduationYear']
    categorical = [c for c in x if c not in numeric]
    categorical_transform = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))]) if impute else OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer([('categorical',categorical_transform,categorical),('numeric','passthrough',numeric)])
    return Pipeline([('preprocess',preprocessor),('model',RandomForestClassifier(**parameters,random_state=SEED,n_jobs=1))])

def fit_pipeline(model,x,y):
    # Trees have independent fixed seeds and preserve their assigned order.
    # Parallel fitting is deterministic; prediction returns to one worker so
    # floating-point probability summation also follows a fixed order.
    model.named_steps['model'].set_params(n_jobs=min(4,os.cpu_count() or 1))
    model.fit(x,y)
    model.named_steps['model'].set_params(n_jobs=1)
    return model

def metrics(y,pred,probability=None,classes=None):
    result = dict(accuracy=accuracy_score(y,pred),weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),
        macro_f1=f1_score(y,pred,average='macro',zero_division=0),cohen_kappa=cohen_kappa_score(y,pred),
        balanced_accuracy=balanced_accuracy_score(y,pred),MCC=matthews_corrcoef(y,pred))
    for average in ['weighted','macro']:
        precision,recall,_,_ = precision_recall_fscore_support(y,pred,average=average,zero_division=0)
        result[average+'_precision'] = precision
        result[average+'_recall'] = recall
    low,high = proportion_confint(int(np.sum(np.asarray(y)==np.asarray(pred))),len(y),method='wilson')
    result.update(accuracy_CI_low=low,accuracy_CI_high=high)
    if probability is not None:
        if len(classes)==2:
            positive_index = list(classes).index('Participated') if 'Participated' in classes else list(classes).index(1)
            binary_y = (np.asarray(y)==classes[positive_index]).astype(int)
            result['ROC_AUC'] = roc_auc_score(binary_y,probability[:,positive_index])
            result['average_precision'] = average_precision_score(binary_y,probability[:,positive_index])
            result['participated_f1'] = f1_score(binary_y,np.asarray(pred)==classes[positive_index])
        else:
            binary = np.asarray(y)[:,None]==np.asarray(classes)[None,:]
            for average in ['weighted','macro']:
                result[average+'_ROC_AUC'] = roc_auc_score(y,probability,labels=classes,multi_class='ovr',average=average)
                result[average+'_average_precision'] = average_precision_score(binary,probability,average=average)
    return result

def bootstrap_metrics(y,pred,replicates=1500):
    y,pred = np.asarray(y),np.asarray(pred)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(replicates):
        indices = rng.integers(len(y),size=len(y))
        draws.append([f1_score(y[indices],pred[indices],average=a,zero_division=0) for a in ['weighted','macro']])
    cis = np.quantile(draws,[.025,.975],axis=0)
    return {f'{average}_f1_CI_{bound}':float(cis[j,i]) for i,average in enumerate(['weighted','macro']) for j,bound in enumerate(['low','high'])}

def evaluate(study,frames,target,features,parameters,impute=True):
    train,val,test = frames
    development = pd.concat([train,val],ignore_index=True)
    model = pipeline(development[features],parameters,impute=impute)
    fit_pipeline(model,development[features],development[target])
    y = test[target].to_numpy()
    pred = model.predict(test[features])
    probability = model.predict_proba(test[features])
    result = {**metrics(y,pred,probability,model.classes_),**bootstrap_metrics(y,pred)}
    classes = list(model.classes_)
    report = pd.DataFrame(classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)).T.rename_axis('class').reset_index()
    save_table(study+'_classification_report',report)
    cm = confusion_matrix(y,pred,labels=classes)
    save_table(study+'_confusion_matrix',pd.DataFrame(cm,columns=[str(c) for c in classes]).assign(actual_class=[str(c) for c in classes]))
    predictions = pd.DataFrame({'row_id':test.index,'actual':y,'predicted':pred})
    for i,cls in enumerate(classes):
        predictions['probability_'+str(cls)] = probability[:,i]
    save_table(study+'_test_predictions',predictions)
    baseline = DummyClassifier(strategy='most_frequent').fit(development[features],development[target])
    baseline_metrics = metrics(y,baseline.predict(test[features]))
    configuration = dict(study=study,target=target,features=features,parameters=parameters,seed=SEED,
        train_N=len(train),validation_N=len(val),test_N=len(test),class_order=classes,
        preprocessing='HSCGraduationYear numeric passthrough; categorical one-hot within pipeline; '+
            ('training-fitted most-frequent imputation.' if impute else 'historical NaN category retained.'),
        test_metrics=result,majority_baseline=baseline_metrics)
    save_json(METRICS/(study+'.json'),configuration)
    model_dir = ROOT/'results/models'
    model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,model_dir/(study+'.joblib'),compress=3)
    return configuration,pred

def tune(train,val,target,features,study,impute=True):
    rows = []
    for parameters in GRID:
        model = pipeline(train[features],parameters,impute=impute)
        fit_pipeline(model,train[features],train[target])
        pred = model.predict(val[features])
        rows.append({**parameters,**metrics(val[target],pred)})
    table = pd.DataFrame(rows).sort_values(['weighted_f1','accuracy','cohen_kappa'],ascending=False,kind='stable')
    save_table(study+'_validation_grid',table)
    best = table.iloc[0]
    parameters = dict(n_estimators=int(best.n_estimators),max_features=best.max_features,min_samples_leaf=int(best.min_samples_leaf))
    return parameters

def split_ids(df,target,study):
    train,temp = train_test_split(df.index.to_numpy(),test_size=.30,random_state=SEED,stratify=df[target])
    val,test = train_test_split(temp,test_size=.50,random_state=SEED,stratify=df.loc[temp,target])
    ids = pd.concat([pd.DataFrame({'row_id':indices,'split':name}) for name,indices in [('train',train),('validation',val),('test',test)]],ignore_index=True)
    save_table(study+'_split_assignments',ids)
    return [df.loc[i].copy() for i in [train,val,test]]

def historical_year1():
    frames = [pd.read_csv(ROOT/f'data/splits/{name}.csv',encoding='utf-8-sig') for name in ['train','validation','test']]
    features = list(frames[0].drop(columns='Year1CGPA').columns)
    # Historical preprocessing deliberately retains its missing-value category.
    selected = tune(frames[0],frames[1],'Year1CGPA',features,'year1',impute=False)
    frozen = dict(n_estimators=100,max_features='sqrt',min_samples_leaf=1)
    result,pred = evaluate('year1',frames,'Year1CGPA',features,frozen,impute=False)
    result['regenerated_validation_selected_parameters'] = selected
    result['test_policy'] = 'Historical frozen configuration; validation sweep is reconstructed without changing the recorded final experiment.'
    save_json(METRICS/'year1.json',result)
    return result

def research_experiments(df):
    # Primary current-population prediction: outcomes missing remain missing;
    # no complete-population duplicate exclusions are applied.
    available = df.loc[df.research.notna()].copy()
    available['research'] = available.research.astype(int)
    frames = split_ids(available,'research','research')
    features = feature_list('research')
    # Frozen upfront Python approximation to WEKA K=12; no test-driven tuning.
    parameters = dict(n_estimators=100,max_features=12,min_samples_leaf=1)
    model = pipeline(frames[0][features],parameters)
    fit_pipeline(model,frames[0][features],frames[0].research)
    save_table('research_validation',pd.DataFrame([{**parameters,**metrics(frames[1].research,model.predict(frames[1][features]))}]))
    result,_ = evaluate('research',frames,'research',features,parameters)
    result['population_N'] = len(df)
    result['labeled_N'] = len(available)
    result['missing_outcome_N'] = int(df.research.isna().sum())
    result['feature_timing'] = 'Historical 24 end-of-Year-1 predictors; survey timing is not independently validated.'
    save_json(METRICS/'research.json',result)
    historical = [pd.read_csv(ROOT/f'data/splits/research_participation/{name}.csv',encoding='utf-8-sig') for name in ['train','validation','test']]
    historic_features = list(historical[0].drop(columns='ResearchParticipationBinary').columns)
    historical_result,_ = evaluate('research_historical_python',historical,'ResearchParticipationBinary',historic_features,parameters,impute=False)
    historical_result['implementation_note'] = 'Python approximation: K=12 selects one-hot encoded columns, whereas WEKA selects original attributes; not an identical forest.'
    save_json(METRICS/'research_historical_python.json',historical_result)
    return result,historical_result

def year2_experiments(df):
    frames = split_ids(df,'Year2Band','year2')
    # Select one shared configuration by average validation weighted F1. Both
    # feature sets share exact rows, candidate grid, seed and selected forest.
    validation = []
    for study in ['year2_without_year1','year2_with_year1']:
        features = feature_list(study)
        for candidate,parameters in enumerate(GRID,1):
            model = pipeline(frames[0][features],parameters)
            fit_pipeline(model,frames[0][features],frames[0].Year2Band)
            validation.append(dict(study=study,**parameters,**metrics(frames[1].Year2Band,model.predict(frames[1][features]))))
            if candidate%9==0:
                print(f'    {study}: {candidate}/{len(GRID)} validation configurations',flush=True)
    validation = pd.DataFrame(validation)
    save_table('year2_validation_grid',validation)
    shared = validation.groupby(['n_estimators','max_features','min_samples_leaf'],sort=False).agg(
        weighted_f1=('weighted_f1','mean'),accuracy=('accuracy','mean'),cohen_kappa=('cohen_kappa','mean')).reset_index()
    best = shared.sort_values(['weighted_f1','accuracy','cohen_kappa'],ascending=False,kind='stable').iloc[0]
    parameters = dict(n_estimators=int(best.n_estimators),max_features=best.max_features,min_samples_leaf=int(best.min_samples_leaf))
    results,predictions = {},{}
    for study in ['year2_without_year1','year2_with_year1']:
        results[study],predictions[study] = evaluate(study,frames,'Year2Band',feature_list(study),parameters)
        results[study]['selection_rule'] = 'One common configuration maximizing mean C1/C2 validation weighted F1; accuracy and kappa tie-breakers.'
        results[study]['class_labels'] = BANDS
        save_json(METRICS/(study+'.json'),results[study])
    y = frames[2].Year2Band.to_numpy()
    before,after = [predictions[s] for s in ['year2_without_year1','year2_with_year1']]
    keys = ['accuracy','weighted_f1','macro_f1']
    delta = {f'delta_{k}':results['year2_with_year1']['test_metrics'][k]-results['year2_without_year1']['test_metrics'][k] for k in keys}
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(1500):
        ids = rng.integers(len(y),size=len(y))
        old,new = metrics(y[ids],before[ids]),metrics(y[ids],after[ids])
        draws.append([new[k]-old[k] for k in keys])
    ci = np.quantile(draws,[.025,.975],axis=0)
    for i,key in enumerate(keys):
        delta[f'delta_{key}_CI_low'] = ci[0,i]
        delta[f'delta_{key}_CI_high'] = ci[1,i]
    delta.update(test_N=len(y),CI_method='paired percentile bootstrap; 1500 draws; seed=1')
    from scipy.stats import binomtest
    old_correct,new_correct = before==y,after==y
    gains = int((~old_correct&new_correct).sum())
    losses = int((old_correct&~new_correct).sum())
    delta.update(discordant_gains=gains,discordant_losses=losses,
        exact_McNemar_p=binomtest(gains,gains+losses,.5).pvalue if gains+losses else 1.)
    save_json(METRICS/'year2_added_information.json',delta)
    save_table('year2_added_information',pd.DataFrame([delta]))
    return results,delta

def reproduce_weka():
    jar = ROOT/'tools/vendor/weka/weka.jar'
    java = os.environ.get('JAVA_EXE') or shutil.which('java')
    if not java:
        raise RuntimeError('Java runtime required to reproduce historical WEKA. Install Java 8+ or set JAVA_EXE.')
    output = ROOT/'results/logs/research_historical_weka.txt'
    output.parent.mkdir(parents=True,exist_ok=True)
    environment = os.environ.copy()
    environment['WEKA_HOME'] = str(ROOT/'.weka-runtime')
    command = [java,'-Djava.awt.headless=true','-Dfile.encoding=UTF-8','-cp',str(jar),
        'weka.classifiers.trees.RandomForest','-t',str(ROOT/'data/splits/research_participation/ResearchParticipation_TrainValidation_2953.arff'),
        '-T',str(ROOT/'data/splits/research_participation/ResearchParticipation_test.arff'),'-v',
        '-P','100','-I','100','-num-slots','1','-K','12','-M','1.0','-V','0.001','-S','1']
    run = subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,encoding='utf-8',errors='replace',timeout=300)
    output.write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    if run.returncode or 'Weka exception:' in run.stderr:
        raise RuntimeError(f'Historical WEKA reproduction failed; see {output}: {run.stderr[-1500:]}')
    # Parse the newly executed confusion matrix, not checkpoint text.
    import re
    matches = re.findall(r'^\s*(\d+)\s+(\d+)\s*\|\s*\w\s*=\s*(NotParticipated|Participated)',run.stdout,re.MULTILINE)
    matches = matches[-2:]
    if len(matches)!=2:
        raise RuntimeError('Could not parse regenerated WEKA confusion matrix.')
    cm = np.array([[int(a),int(b)] for a,b,_ in matches])
    labels = [c for _,_,c in matches]
    y,pred = [],[]
    for i in range(2):
        for j in range(2):
            y.extend([labels[i]]*cm[i,j]); pred.extend([labels[j]]*cm[i,j])
    cli_result = metrics(y,pred)
    save_json(METRICS/'research_historical_weka_cli_diagnostic.json',cli_result)
    # Reproduce the GUI/API training path with explicit UTF-8 reading and the
    # preserved header. The helper also independently replays the saved model.
    prediction_file = LOGS/'weka_replay_predictions.csv'
    replay_command = [java,'-Djava.awt.headless=true','-Dfile.encoding=UTF-8','-cp',
        str(jar)+os.pathsep+str(jar.parent),'ReplayResearch',
        str(ROOT/'models/final/ResearchParticipation_Final_RandomForest_K12.model'),
        str(ROOT/'data/splits/research_participation/ResearchParticipation_TrainValidation_2953.arff'),
        str(ROOT/'data/splits/research_participation/ResearchParticipation_test.arff'),str(prediction_file)]
    replay = subprocess.run(replay_command,cwd=ROOT,env=environment,capture_output=True,encoding='utf-8',errors='replace',timeout=300)
    (LOGS/'research_historical_weka_api.txt').write_text(replay.stdout+'\n'+replay.stderr,encoding='utf-8')
    if replay.returncode:
        raise RuntimeError('WEKA API replay failed: '+replay.stderr[-1500:])
    predictions = pd.read_csv(prediction_file)
    probability = np.column_stack([1-predictions.refit_probability,predictions.refit_probability])
    result = {**metrics(predictions.actual,predictions.refit_predicted,probability,[0,1]),
        **bootstrap_metrics(predictions.actual,predictions.refit_predicted)}
    result['frozen_model_replay'] = metrics(predictions.actual,predictions.frozen_predicted,
        np.column_stack([1-predictions.frozen_probability,predictions.frozen_probability]),[0,1])
    result['refit_matches_frozen_predictions'] = bool((predictions.refit_predicted==predictions.frozen_predicted).all())
    result['maximum_probability_difference_from_frozen'] = float(np.max(np.abs(predictions.refit_probability-predictions.frozen_probability)))
    y = predictions.actual.map({0:'NotParticipated',1:'Participated'}).to_numpy()
    pred = predictions.refit_predicted.map({0:'NotParticipated',1:'Participated'}).to_numpy()
    labels = ['NotParticipated','Participated']
    cm = confusion_matrix(y,pred,labels=labels)
    result['participated_f1'] = f1_score(np.asarray(y)=='Participated',np.asarray(pred)=='Participated')
    result.update(test_N=len(y),jar_sha256=sha256(jar),implementation='WEKA 3.8.7; original nominal attributes; frozen K=12',
        parameters={'trees':100,'K':12,'M':1.,'seed':1},generating_script='src/research/ml.py',
        note='Independent API refit with explicit UTF-8 and preserved nominal header; compared against saved-model replay. CLI path retained as a diagnostic with different results.')
    save_table('research_historical_weka_confusion_matrix',pd.DataFrame(cm,columns=labels).assign(actual_class=labels))
    report = pd.DataFrame(classification_report(y,pred,output_dict=True,zero_division=0)).T.rename_axis('class').reset_index()
    save_table('research_historical_weka_classification_report',report)
    save_json(METRICS/'research_historical_weka.json',result)
    return result

def ml_experiments(df):
    print('  Reproducing frozen Year-1 experiment and validation sweep',flush=True)
    year1 = historical_year1()
    print('  Reproducing research participation',flush=True)
    research,historical = research_experiments(df)
    weka = reproduce_weka()
    print('  Running paired Year-2 feature comparison',flush=True)
    year2,delta = year2_experiments(df)
    configs = dict(year1=year1,research=research,research_historical_python=historical,**year2)
    save_table('ml_test_metrics',pd.DataFrame([dict(study=name,train_N=config['train_N'],validation_N=config['validation_N'],test_N=config['test_N'],**config['test_metrics']) for name,config in configs.items()]))
    return configs,delta,weka
