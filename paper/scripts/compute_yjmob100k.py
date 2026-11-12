"""YJMob100K utility, risk and fairness measures, and matched-utility comparison. Run from the repository root."""
import pandas as pd,numpy as np,json,glob
B='reports/figures'
S=f'{B}/yjmob100k_sample'
p=pd.read_parquet(f'{S}/unified_sample.parquet',columns=['user_id','cell_id'])
uids=sorted(p.user_id.unique()); n=len(uids); assert n==2000
cells={c:i for i,c in enumerate(p.cell_id.unique())}
P=np.zeros((n,len(cells)),np.float32)
cnt=p.groupby(['user_id','cell_id']).size().reset_index()
P[cnt.user_id.values-1,cnt.cell_id.map(cells).values]=cnt[0].values
P/=P.sum(1,keepdims=True)
LOC=np.zeros((n,n))
for i in range(n):
    s=np.nonzero(P[i])[0]; a=P[i,s][None,:]; b=P[:,s]; m=(a+b)/2
    with np.errstate(divide='ignore',invalid='ignore'):
        t=a*np.log(a/m)+np.where(b>0,b*np.log(b/m),0)
    shared=0.5*t.sum(1); rest=0.5*np.log(2)*(1-b.sum(1))
    LOC[i]=1-np.sqrt(np.clip((shared+rest)/np.log(2),0,1))
LOC=(LOC+LOC.T)/2
z=np.load(f'{S}/similarities.npz'); remap=json.load(open(f'{S}/user_id_remap.json')); orig=json.load(open(f'{S}/user_ids.json'))
order=np.array([remap[str(u)]-1 for u in orig]); inv=np.argsort(order)
sims={'loc':LOC,'traj':z['trajectory_shape'][np.ix_(inv,inv)],'rog':z['radius_of_gyration'][np.ix_(inv,inv)]}
iu=np.triu_indices(n,1)
rows=[]
for path in sorted(glob.glob(f'{B}/retraining_to_convergence/yjmob100k_*')):
    nm=path.split('yjmob100k_')[1]
    if 'auditseed' in nm or nm=='plain_lp0.8': continue
    df=pd.read_csv(path+'/per_user_04d.csv').sort_values('user_id'); assert (df.user_id.values==np.arange(1,n+1)).all() and (df.n_windows>=10).all()
    s=json.load(open(path+'/summary.json')); r=df.post_top1.values; d=np.abs(r[iu[0]]-r[iu[1]]); h=df.s_unc.values
    row={'run':nm,'util':s['utility_mean'],'mean':r.mean(),'dec0':r[df.entropy_decile.values==0].mean(),'dec9':r[df.entropy_decile.values==9].mean()}
    allS={'ent':-np.abs(h[iu[0]]-h[iu[1]]),**{k:v[iu] for k,v in sims.items()}}
    for k,v in allS.items():
        top=v>=np.quantile(v,0.9); row[k+'_abs']=d[top].mean(); row[k+'_rel']=d[top].mean()/r.mean()
    rows.append(row)
D=pd.DataFrame(rows); D.to_csv('paper/data/results_yjmob.csv',index=False); pd.set_option('display.width',250); print(D.round(4).to_string(index=False))
# matched
c=D[D.run.str.startswith('plain')].copy(); c['lp']=c.run.str.extract(r'lp([\d.]+)')[0].astype(float); c=c.sort_values('lp')
u=c.util.values
for _,q in D[D.run.str.startswith('ed_soft')].iterrows():
    seg=[i for i in range(len(u)-1) if min(u[i],u[i+1])<=q.util<=max(u[i],u[i+1])]
    if not seg: print(q.run,'outside'); continue
    i=seg[-1]; a,b=c.iloc[i],c.iloc[i+1]; t=(q.util-a.util)/(b.util-a.util)
    out=f"{q.run} util {q.util:.4f} seg lp{a.lp}-lp{b.lp}: "
    for k in ['mean','dec0','dec9']: out+=f"{k} {a[k]+t*(b[k]-a[k]):.3f}->{q[k]:.3f}  "
    for k in ['ent','loc','traj','rog']:
        for m in ['abs','rel']: mo=a[f'{k}_{m}']+t*(b[f'{k}_{m}']-a[f'{k}_{m}']); out+=f"{k}_{m} {100*(q[f'{k}_{m}']/mo-1):+.0f}% "
    print(out)
