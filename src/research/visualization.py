"""Publication figures derived exclusively from executable output tables."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from .common import *

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':120,'savefig.dpi':300,
    'svg.fonttype':'none','svg.hashsalt':'academic-performance-seed-1','pdf.fonttype':42})
COLORS = ['#c66a46','#426c90','#438b72','#9a78a4','#d2a349']

def read(name):
    return pd.read_csv(TABLES/(name+'.csv'))

def short(value):
    substitutions = {'Rented apartment/house (shared with others)':'Shared rental',
        'Rented apartment/house (alone)':'Rented alone','University residence hall/dormitory':'University hall',
        'With parents/family':'Parents/family','With relatives':'Relatives',
        'Yes, three or more clubs/organizations':'3+ clubs','Yes, two clubs/organizations':'2 clubs',
        'Yes, one club/organization':'1 club','No':'None',
        'StudentClubMemberOrdinal':'Club involvement (per band)',
        'HSCMathOrdinal':'HSC Mathematics (per band)','HSCPhysicsOrdinal':'HSC Physics (per band)',
        'HSCChemistryOrdinal':'HSC Chemistry (per band)','Year1Band':'Year-1 standing (per band)'}
    value = substitutions.get(str(value),str(value))
    if '[T.' in value:
        factor=value[2:].split(',')[0].split(')')[0] if value.startswith('C(') else value.split('[')[0]
        contrast=value[value.index('[T.')+3:].rstrip(']')
        value = human_name(factor)+': '+substitutions.get(contrast,contrast)
    return '\n'.join(textwrap.wrap(value,32))

def save(fig,name,title,source,catalog,note=''):
    FIGURES.mkdir(parents=True,exist_ok=True)
    fig.suptitle(title,fontsize=13,fontweight='bold')
    fig.text(.01,.005,note or 'Source: '+', '.join(source),fontsize=7,color='#444444',wrap=True)
    fig.tight_layout(rect=[0,.035,1,.93])
    for extension in ['png','pdf','svg']:
        metadata = {'Date':None} if extension=='svg' else {'CreationDate':None,'ModDate':None} if extension=='pdf' else None
        fig.savefig(FIGURES/(name+'.'+extension),bbox_inches='tight',metadata=metadata)
    plt.close(fig)
    catalog.append(dict(figure=name,title=title,png='results/figures/'+name+'.png',
        pdf='results/figures/'+name+'.pdf',svg='results/figures/'+name+'.svg',
        source_tables='; '.join('results/tables/'+s+'.csv' for s in source),
        generating_script='src/research/visualization.py',note=note))

def rate_plot(ax,frame,label='group',percentage=True):
    value = 'percent' if percentage else 'probability'
    multiplier = 1 if percentage else 100
    x = np.arange(len(frame))
    y = frame[value].to_numpy()*multiplier
    ax.bar(x,y,color=COLORS[1],width=.65)
    ax.errorbar(x,y,yerr=np.vstack([y-frame.CI_low.to_numpy()*multiplier,
        frame.CI_high.to_numpy()*multiplier-y]),fmt='none',color='#333333',capsize=3,linewidth=1)
    labels=[short(v) for v in frame[label]]
    if not percentage and frame.factor.eq('StudentClubMemberOrdinal').all():
        labels=[['None','1 club','2 clubs','3+ clubs'][int(v)] for v in frame[label]]
    ax.set_xticks(x,labels,rotation=30,ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel('Proportion (%) with 95% CI')
    ax.set_ylim(0,min(100,max(frame.CI_high*multiplier)*1.12))
    for i,(_,row) in enumerate(frame.iterrows()):
        ax.text(i,row.CI_high*multiplier+1.5,f'n={int(row.get("N",row.get("subgroup_N",0)))}',ha='center',fontsize=8)

def interaction_plot(ax,frame):
    for i,(subgroup,group) in enumerate(frame.groupby('subgroup',sort=True)):
        levels = group.level.to_list()
        x = np.arange(len(levels))
        y = group.probability.to_numpy()*100
        ax.plot(x,y,'o-',label=short(subgroup),color=COLORS[i%len(COLORS)])
        ax.fill_between(x,group.CI_low*100,group.CI_high*100,color=COLORS[i%len(COLORS)],alpha=.12)
        labels=[short(v) for v in levels]
        if group.factor.eq('StudentClubMemberOrdinal').all():
            labels=[['None','1 club','2 clubs','3+ clubs'][int(v)] for v in levels]
        elif group.factor.str.startswith('HSC').all():
            labels=[['<120','120–140','140–160','160–190','190–200'][int(v)] for v in levels]
        ax.set_xticks(x,labels)
    ax.set_ylabel('Standardized probability (%)')
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(fontsize=8,frameon=False)

def generate_figures(df):
    catalog = []
    composition = read('categorical_distributions')
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for ax,factor in zip(axes,['Gender','Faculty','CollegeCategory']):
        data = composition[composition.variable==factor].copy()
        ax.bar([short(english(x)) for x in data.category],data.percent_population,color=COLORS[:len(data)] if len(data)<=5 else COLORS[1])
        ax.set_title(human_name(factor))
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        for i,n in enumerate(data['count']):
            ax.text(i,data.percent_population.iloc[i]+1,f'n={n}',ha='center',fontsize=8)
    save(fig,'01_dataset_composition','Dataset composition',['categorical_distributions'],catalog)
    years = read('yearly_cgpa_distribution')
    for year in [1,2]:
        fig,ax = plt.subplots(figsize=(8,4))
        rate_plot(ax,years[years.year==year],label='label')
        save(fig,f'{year+1:02d}_year{year}_cgpa_distribution',f'Year-{year} CGPA bands',['yearly_cgpa_distribution'],catalog)
    transition = read('transition_counts').drop(columns='Year1Band').to_numpy()
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    normalized = transition/transition.sum(axis=1,keepdims=True)*100
    for ax,values,title,percentage in zip(axes,[transition,normalized],['Counts','Row percentages'],[False,True]):
        im = ax.imshow(values,cmap='Blues')
        ax.set_xticks(range(5),BANDS,rotation=30,ha='right'); ax.set_yticks(range(5),BANDS)
        ax.set_xlabel('Year-2 band'); ax.set_ylabel('Year-1 band'); ax.set_title(title)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{values[i,j]:.1f}%' if percentage else str(int(values[i,j])),ha='center',va='center',
                    color='white' if values[i,j]>values.max()*.55 else '#111111',fontsize=8)
        fig.colorbar(im,ax=ax,shrink=.65)
    save(fig,'04_year1_vs_year2_transition_heatmap','Year-1 to Year-2 academic transitions',['transition_counts','transition_row_percent'],catalog)
    fig,ax = plt.subplots(figsize=(7,4)); rate_plot(ax,read('trajectory_distribution'),label='trajectory')
    save(fig,'05_academic_trajectory_distribution','Academic mobility',['trajectory_distribution'],catalog)
    sub = read('subgroup_persistence'); sub = sub[(sub.factor=='Faculty')&sub.spearman_rho.notna()]
    fig,ax = plt.subplots(figsize=(7,4)); x=np.arange(len(sub))
    ax.errorbar(x,sub.spearman_rho,yerr=np.vstack([sub.spearman_rho-sub.rho_CI_low,sub.rho_CI_high-sub.spearman_rho]),fmt='o',capsize=3,color=COLORS[1])
    ax.set_xticks(x,[f'{s}\nN={n}' for s,n in zip(sub.subgroup,sub.N)]); ax.set_ylim(0,1); ax.set_ylabel('Spearman rho (95% bootstrap CI)')
    save(fig,'06_year1_year2_persistence','Academic persistence across faculty contexts',['subgroup_persistence'],catalog)
    contrasts = read('descriptive_contrasts')
    for year in [1,2]:
        fig,axes = plt.subplots(1,3,figsize=(13,4))
        for ax,factor in zip(axes,['HSCMath','HSCPhysics','HSCChemistry']):
            data=contrasts[(contrasts.factor==factor)&(contrasts.outcome==f'high_year{year}')]
            rate_plot(ax,data); ax.set_title(human_name(factor))
        save(fig,f'{6+year:02d}_hsc_subject_vs_year{year}',f'HSC achievement and high Year-{year} CGPA',['descriptive_contrasts'],catalog,
            'High performance uses the upper two CGPA bands. Wilson confidence intervals; unadjusted comparisons.')
    interaction = read('interaction_marginal_probabilities'); tests = read('interaction_tests')
    fig,ax=plt.subplots(figsize=(7,4)); interaction_plot(ax,interaction[interaction.analysis=='gender_HSCMath_year1'])
    q=tests.loc[tests.analysis=='gender_HSCMath_year1','q_value'].iloc[0]
    save(fig,'09_hsc_math_gender_interaction','HSC Mathematics by gender: Year-1 high CGPA',['interaction_marginal_probabilities','interaction_tests'],catalog,
        f'Adjusted standardized probabilities with 95% delta-method CI. Interaction FDR q={q:.3g}; exploratory association.')
    for number,factor,suffix in [(10,'Gender','gender'),(11,'LivingArrangement','living_arrangement')]:
        table=pd.crosstab(df[factor],df.Trajectory).reindex(columns=['Declined','Stable','Improved'],fill_value=0)
        table.columns.name=None; table.index.name='group'
        data=table.reset_index(); data['N']=table.sum(axis=1).to_numpy()
        for col in ['Declined','Stable','Improved']:
            data[col+'_percent']=data[col]/data.N*100
        save_table('trajectory_by_'+suffix,data)
        fig,ax=plt.subplots(figsize=(9,4))
        bottom=np.zeros(len(data))
        for i,col in enumerate(['Declined','Stable','Improved']):
            ax.bar(np.arange(len(data)),data[col+'_percent'],bottom=bottom,label=col,color=COLORS[i]); bottom+=data[col+'_percent']
        ax.set_xticks(range(len(data)),[f'{short(s)}\nn={n}' for s,n in zip(data.group,data.N)],rotation=25,ha='right')
        ax.set_ylabel('Within-group proportion (%)'); ax.yaxis.set_major_formatter(PercentFormatter(100)); ax.legend(frameon=False,ncol=3)
        save(fig,f'{number:02d}_trajectory_by_{suffix}',f'Academic mobility by {human_name(factor).lower()}',['trajectory_by_'+suffix],catalog)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    raw_clubs=contrasts[(contrasts.factor=='StudentClubMember')&(contrasts.outcome=='research')].set_index('group').loc[
        ['No','Yes, one club/organization','Yes, two clubs/organizations','Yes, three or more clubs/organizations']].reset_index()
    rate_plot(axes[0],raw_clubs); axes[0].set_title('Unadjusted rates')
    margins=read('adjusted_marginal_probabilities')
    rate_plot(axes[1],margins[(margins.model=='research')&(margins.factor=='StudentClubMemberOrdinal')],label='level',percentage=False); axes[1].set_title('Adjusted standardized probabilities')
    save(fig,'12_club_vs_research','Club involvement and research participation',['descriptive_contrasts','adjusted_marginal_probabilities'],catalog)
    for number,name,title in [(13,'clubs_college_research','college background'),(14,'clubs_faculty_research','faculty')]:
        fig,ax=plt.subplots(figsize=(8,5)); interaction_plot(ax,interaction[interaction.analysis==name])
        test=tests[tests.analysis==name].iloc[0]
        save(fig,f'{number:02d}_club_research_by_'+('college_background' if number==13 else 'faculty'),
            'Clubs and research participation by '+title,['interaction_marginal_probabilities','interaction_tests'],catalog,
            f'Exploratory nested interaction: p={test.p_value:.3g}; FDR q={test.q_value:.3g}. Curves alone do not establish interaction.')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rate_plot(axes[0],contrasts[(contrasts.factor=='Trajectory')&(contrasts.outcome=='research')]); axes[0].set_title('Unadjusted rates')
    rate_plot(axes[1],interaction[interaction.analysis=='trajectory_research_adjusted'],label='level',percentage=False); axes[1].set_title('Adjusted standardized probabilities')
    save(fig,'15_research_by_trajectory','Research participation across academic trajectories',['descriptive_contrasts','interaction_marginal_probabilities'],catalog)
    effects=read('raw_vs_adjusted_effects'); effects=effects[(effects.model=='research')&effects.factor.isin(['StudentClubMember','CollegeCategory','Gender'])]
    fig,ax=plt.subplots(figsize=(8,4)); y=np.arange(len(effects))
    for shift,prefix,label,color in [(-.08,'raw','Unadjusted',COLORS[0]),(.08,'adjusted','Adjusted',COLORS[1])]:
        estimates=effects[prefix+'_OR']; low=effects[prefix+'_CI_low']; high=effects[prefix+'_CI_high']
        ax.errorbar(estimates,y+shift,xerr=np.vstack([estimates-low,high-estimates]),fmt='o',capsize=3,label=label,color=color)
    ax.set_yticks(y,[short(t) for t in effects.term]); ax.axvline(1,color='#888888',linestyle='--'); ax.set_xscale('log'); ax.set_xlabel('Odds ratio with 95% CI'); ax.legend(frameon=False)
    save(fig,'16_raw_vs_adjusted_effects','Research associations before and after adjustment',['raw_vs_adjusted_effects'],catalog,
        'Same outcome, coefficient contrast and complete-case rows; odds ratios remain observational associations.')
    for number,model in [(17,'year1'),(18,'year2'),(19,'decline'),(20,'improvement'),(21,'research')]:
        table=read('adjusted_'+model); table=table[table.term!='Intercept'].iloc[::-1]
        fig,ax=plt.subplots(figsize=(10,max(6,len(table)*.24)))
        y=np.arange(len(table)); estimate=table.odds_ratio
        ax.errorbar(estimate,y,xerr=np.vstack([estimate-table.CI_low,table.CI_high-estimate]),fmt='o',capsize=2,color=COLORS[1],markersize=3)
        ax.set_yticks(y,[short(t).replace('\n',' ') for t in table.term],fontsize=8)
        ax.set_xscale('log'); ax.axvline(1,linestyle='--',color='#888888'); ax.set_xlabel('Adjusted odds ratio with 95% CI')
        save(fig,f'{number:02d}_adjusted_{model}_forest_plot',f'Adjusted associations: {model}',['adjusted_'+model],catalog,
            f'Complete-case logistic association model, N={int(table.model_N.iloc[0])}. Reference categories and q-values in source table.')
    return save_table('figure_catalog',pd.DataFrame(catalog))
