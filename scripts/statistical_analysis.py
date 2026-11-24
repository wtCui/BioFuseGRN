import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

p=argparse.ArgumentParser()
p.add_argument('--results', required=True, help='CSV with dataset,network,gene_count,seed,method,auroc,auprc; matched splits across methods')
p.add_argument('--output', default='statistics.csv')
a=p.parse_args()
df=pd.read_csv(a.results)
rows=[]
for key,g in df.groupby(['dataset','network','gene_count']):
    for metric in ['auroc','auprc']:
        x=g.pivot(index='seed',columns='method',values=metric)
        if 'BioFuseGRN' not in x or x.shape[1]<2 or x.isna().any().any():
            raise ValueError(f'Missing paired results: {key}')
        baseline=x.drop(columns='BioFuseGRN').mean().idxmax()
        delta=x['BioFuseGRN']-x[baseline]
        pv=1.0 if np.all(delta==0) else wilcoxon(delta,alternative='two-sided').pvalue
        rows.append(dict(zip(['dataset','network','gene_count'],key),metric=metric,baseline=baseline,n=len(delta),mean_difference=delta.mean(),p=float(pv)))
r=pd.DataFrame(rows)
order=np.argsort(r.p.to_numpy()); adj=np.empty(len(r)); adj[order]=np.minimum(1,np.maximum.accumulate(r.p.to_numpy()[order]*(len(r)-np.arange(len(r)))))
r['p_holm']=adj;r['significant']=adj<0.05;r.to_csv(a.output,index=False)
