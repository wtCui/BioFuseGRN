import argparse
from pathlib import Path
import pandas as pd
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--validation',required=True,help='CSV with tf,target columns from independent evidence');p.add_argument('--tfs',nargs='+',default=['SOX2','NANOG']);p.add_argument('--top-k',type=int,default=20);p.add_argument('--output',default='case_study.csv');a=p.parse_args()
r=Path(a.run);genes=pd.read_csv(r/'genes.csv').gene.astype(str).tolist();tfset=set(pd.read_csv(r/'tfs.csv').tf.astype(str));z=np.load(r/'embeddings.npy');w=np.load(r/'link_weight.npy');known=pd.read_csv(r/'train_pairs.csv');known=set(map(tuple,known.loc[known.label==1,['tf','target']].values));v=pd.read_csv(a.validation);evidence=set(map(tuple,v[['tf','target']].values));rows=[]
for tf in a.tfs:
    if tf not in tfset: raise ValueError(f'Unknown TF: {tf}')
    scores=1/(1+np.exp(-np.clip(z[genes.index(tf)]@w@z.T,-80,80)))
    ids=[i for i in np.argsort(-scores) if genes[i]!=tf and (tf,genes[i]) not in known][:a.top_k]
    rows.extend({'tf':tf,'target':genes[i],'rank':rank,'score':scores[i],'supported':(tf,genes[i]) in evidence} for rank,i in enumerate(ids,1))
pd.DataFrame(rows).to_csv(a.output,index=False)
