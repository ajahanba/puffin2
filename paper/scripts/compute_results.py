"""Run from the repository root."""
import json, numpy as np, pandas as pd, sys
import os, argparse
ap=argparse.ArgumentParser(description='Per-run utility, risk and individual-fairness measures (D_abs, D_rel) for GeoLife, Foursquare, Gowalla.')
ap.add_argument('--top',type=float,default=0.10,help='fraction of most similar pairs (default 0.10)')
ap.add_argument('--out',default='paper/data/results_main.csv')
args=ap.parse_args()
base='reports/figures'
DEF=[('entropy','ent'),('location_distribution','loc'),('trajectory_shape','traj'),('radius_of_gyration','rog')]
def sims_for(ds):
    if ds=='gowalla_sf_bay':
        out={}
        for k,_ in DEF:
            z=np.load(f'{base}/individual_fairness/gowalla_sf_bay_similarities_{k}.npz'); out[k]=z[z.files[0]].astype(np.float64)
        return out
    z=np.load(f'{base}/individual_fairness/{ds}_similarities.npz'); return {k:z[k] for k,_ in DEF}
def run_rows(ds, runs):
    sims=sims_for(ds); uids=json.load(open(f'{base}/individual_fairness/{ds}_user_ids.json')); rows=[]
    for name,path in runs:
        df=pd.read_csv((path if path.startswith('/') else f'{base}/{path}')+'/per_user_04d.csv'); s=json.load(open((path if path.startswith('/') else f'{base}/{path}')+'/summary.json'))
        df=df[df.n_windows>=10]; risk=dict(zip(df.user_id,df.post_top1)); dec=dict(zip(df.user_id,df.entropy_decile))
        idx=[i for i,u in enumerate(uids) if u in risk]; r=np.array([risk[uids[i]] for i in idx]); dd=np.array([dec[uids[i]] for i in idx])
        iu=np.triu_indices(len(idx),1); d=np.abs(r[iu[0]]-r[iu[1]])
        row={'ds':ds,'run':name,'util':s.get('utility_mean'),'util_sd':s.get('utility_std'),'n':len(idx),'mean':r.mean(),'dec0':r[dd==0].mean(),'dec9':r[dd==9].mean()}
        for k,lab in DEF:
            S=sims[k][np.ix_(idx,idx)][iu]; top=S>=np.quantile(S,1-args.top)
            row[lab+'_abs']=d[top].mean(); row[lab+'_rel']=d[top].mean()/r.mean(); row[lab+'_ratio']=d[top].mean()/d.mean()
        rows.append(row)
    return rows

def pool5(ds,b,x):
    """Pool the 3-seed and the 2-seed (seeds 3-4) Lipschitz runs into one 5-seed run directory."""
    T='task2_formulations'; a=f'{base}/{T}/{ds}_lipschitz_{b}_lambda{x}_3seed'; c=a.replace('_3seed','_seeds34')
    if not os.path.exists(c+'/summary.json'): return None
    A=pd.read_csv(a+'/per_user_04d.csv'); C=pd.read_csv(c+'/per_user_04d.csv')
    ca=[k for k in A if k.startswith('post_top1_encseed')]; cc=[k for k in C if k.startswith('post_top1_encseed')]
    M=A.merge(C[['user_id']+cc],on='user_id'); M['post_top1']=M[ca+cc].mean(axis=1)
    u=json.load(open(a+'/summary.json'))['utility_per_encoder_seed']+json.load(open(c+'/summary.json'))['utility_per_encoder_seed']
    out=f'paper/data/pooled/{ds}_lipschitz_{b}_lambda{x}_5seed'; os.makedirs(out,exist_ok=True)
    M.to_csv(out+'/per_user_04d.csv',index=False); json.dump({'utility_mean':float(np.mean(u))},open(out+'/summary.json','w'))
    return '/'+os.path.abspath(out).lstrip('/')

def pool_trajgan(ds):
    """Average the external attacker's per-user risk over TrajGAN's three generator seeds."""
    ps=[f'{base}/paper/{ds}_trajgan_genseed{g}' for g in range(3)]
    if not all(os.path.exists(p+'/summary.json') for p in ps): return None
    dfs=[pd.read_csv(p+'/per_user_04d.csv') for p in ps]
    M=dfs[0].drop(columns=[c for c in dfs[0] if c.startswith('post_top1')]).copy()
    M['post_top1']=pd.concat([x.set_index('user_id').post_top1 for x in dfs],axis=1).mean(axis=1).reindex(M.user_id).values
    out=f'paper/data/pooled/{ds}_trajgan_3seed'; os.makedirs(out,exist_ok=True); M.to_csv(out+'/per_user_04d.csv',index=False)
    json.dump({'utility_mean':float(np.mean([json.load(open(p+'/summary.json'))['utility_mean'] for p in ps]))},open(out+'/summary.json','w'))
    return '/'+os.path.abspath(out).lstrip('/')

R='retraining_to_convergence'
cfg={
 'geolife':[('no-privacy',f'{R}/geolife_no_privacy')]+[(f'Mo-PAE lp{x}',f'{R}/geolife_plain_lp{x}') for x in ['0.02','0.1','0.2','0.3','0.5','0.8']]+[('Mo-PAE lp1.2',f'{R}/geolife_plain_lp1.2_10seed'),('Mo-PAE lp2',f'{R}/geolife_plain_lp2_10seed_corrected'),('Mo-PAE lp3',f'{R}/geolife_plain_lp3_10seed_corrected')]+[(f'PUFFIN {x}',f'{R}/geolife_ed_soft_lambda{x}_10seed') for x in ['0.3','1.0','3.0']]+[(f'PUFFIN-loc {x}',f'task2_formulations/geolife_ed_soft_location_distribution_lambda{x}_10seed') for x in ['0.3','1.0']],
 'foursquare':[('no-privacy',f'{R}/foursquare_no_privacy')]+[(f'Mo-PAE lp{x}',f'{R}/foursquare_plain_lp{x}') for x in ['0.02','0.1','0.2','0.3','0.5','0.8']]+[(f'PUFFIN {x}',f'{R}/foursquare_ed_soft_lambda{x}') for x in ['0.3','1.0','3.0']],
 'gowalla_sf_bay':[('no-privacy',f'{R}/gowalla_sf_bay_no_privacy')]+[(f'Mo-PAE lp{x}',f'{R}/gowalla_sf_bay_plain_lp{x}') for x in ['0.02','0.1','0.2','0.3','0.5','0.8']]+[(f'PUFFIN {x}',f'{R}/gowalla_sf_bay_ed_soft_lambda{x}') for x in ['0.3','1.0','3.0']],
}
T='task2_formulations'
for ds,ns in [('geolife','10seed'),('foursquare','3seed'),('gowalla_sf_bay','3seed')]:
    for b,lab in [('entropy','Lip-ent'),('location_distribution','Lip-loc')]:
        cfg[ds]+=[(f'{lab} {x}',(pool5(ds,b,x) if ds=='gowalla_sf_bay' else None) or f'{T}/{ds}_lipschitz_{b}_lambda{x}_{ns}') for x in ['0.3','1.0','3.0']]
    cfg[ds]+=[(f'GI-DP {e}',f'gi_dp/{ds}_gi_dp_eps{e}') for e in ['0.0005','0.001','0.002','0.005','0.01']]
cfg['geolife']+=[(f'LocGM {x}',f'paper/geolife_ed_soft_location_gradmatched_lambda{x}_'+('3seed' if x=='3.0' else '10seed')) for x in ['0.3','1.0','3.0']]
for ds in ['geolife','foursquare','gowalla_sf_bay']:
    p=pool_trajgan(ds)
    if p: cfg[ds].append(('TrajGAN',p))
cfg['geolife']+=[(f'GroupDRO {x}',f'{T}/geolife_group_dro_lambda{x}_3seed') for x in ['0.3','1.0']]
allrows=[]
for ds,runs in cfg.items(): allrows+=run_rows(ds,runs)
df=pd.DataFrame(allrows); df.to_csv(args.out,index=False)
