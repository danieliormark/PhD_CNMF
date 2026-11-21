# Own spread of W-tied shared child hyperedges vs the mixes of their three contexts (articles, parents, W neighbours);
# study A pass 3, approach 1, K 13, group 1, seeds 1–10, r 0 / 0.3 / 1 (RL-326). Run from fullscale_pipeline/m1.
import glob, os, json, numpy as np, scipy.sparse as sp, sys
sys.path.insert(0, ".")
from fit import data as D, measures as M
from fit.aggregate import load_fit, load_UZ
b=D.Build(D.BUILD); W=sp.load_npz(b.path+"W_child_child.npz").tocsr(); d=np.asarray(W.sum(1)).ravel(); tied=d>0
Wn=sp.diags(np.where(d>0,1/np.maximum(d,1e-300),0))@W
rds=D.fit_data(b,1,group=0); ca=next(r for r in rds if r.name=="child_article"); sem=[r for r in rds if r.name in("child_article","parent_child")]
df=np.bincount(ca.tie_col,minlength=ca.shape[1]); m=tied&(df>=2)
out={}
for sp_ in glob.glob("/mnt/hum01-rds/Basov/p91688di/m1_fits/stage0_pass3/*/spec.json"):
    j=json.load(open(sp_))["job"]
    if j["approach"]!=1 or j["group"]!=0 or j["K"]!=13: continue
    f=load_fit(os.path.dirname(sp_)); U,Z=load_UZ(f["dir"]); wm=M.weighted_memberships(U,Z,{k:tuple(v) for k,v in f["result"]["relations"].items()})
    P=wm["child"]["P"]; own=M.entropy_bits(P)
    exc=M.excess_spread(sem,wm,"child"); ctx=own-exc                      # most mixed article/parent context spread
    wmix=M.entropy_bits(np.asarray(Wn@P))
    # articles' mix alone
    T=sp.csr_matrix((ca.tie_val,(ca.tie_col,ca.tie_row)),shape=(ca.shape[1],ca.shape[0])); s=np.asarray(T.sum(1)).ravel()
    amix=M.entropy_bits(np.asarray(T@wm["article"]["P"])/np.maximum(s,1e-300)[:,None])
    pc=next(x for x in rds if x.name=="parent_child")
    Tp=sp.csr_matrix((pc.tie_val,(pc.tie_col,pc.tie_row)),shape=(pc.shape[1],pc.shape[0])); sp_=np.asarray(Tp.sum(1)).ravel()
    pmix=M.entropy_bits(np.asarray(Tp@wm["parent"]["P"])/np.maximum(sp_,1e-300)[:,None])
    mean3=(amix+pmix+wmix)/3; med3=np.median(np.vstack([amix,pmix,wmix]),0)
    out.setdefault(("mean",j["arm"]["r"]),[]).append([np.median(pmix[m]),np.median((own-mean3)[m]),np.mean((own-mean3)[m]>0),np.median((own-med3)[m]),np.mean((own-med3)[m]>0)])
    r=j["arm"]["r"]; out.setdefault(r,[]).append([np.median(own[m]),np.median(amix[m]),np.median(ctx[m]),np.median(wmix[m]),
        np.median((own-np.maximum(ctx,wmix))[m]), np.mean((own-np.maximum(ctx,wmix))[m]>0), np.median(M.entropy_bits(wm["article"]["P"]))])
for k,v in sorted([kv for kv in out.items() if isinstance(kv[0],tuple)], key=lambda kv: kv[0][1]):
    x=np.median(np.array(v),0); print(f"r {k[1]}: parents' mix {x[0]:.2f} | own − mean of 3 contexts {x[1]:+.2f} (share>0 {x[2]:.2f}) | own − median of 3 {x[3]:+.2f} (share>0 {x[4]:.2f})")
for r,v in sorted([kv for kv in out.items() if not isinstance(kv[0],tuple)]):
    x=np.median(np.array(v),0); print(f"r {r}: own {x[0]:.2f} | articles' mix {x[1]:.2f} | most mixed of articles/parents {x[2]:.2f} | W-neighbours' mix {x[3]:.2f} | excess incl. W {x[4]:+.2f} (share>0 {x[5]:.2f}) | single article spread {x[6]:.2f}")
