# Per-facet stability with each facet matched on its own (numbering-free) vs the decided joint matching, at K*.
import json, os, glob, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1")
os.environ.pop("M1_ALIGNED", None)
from fit import measures as M
from fit.aggregate import load_fit, load_UZ, latest_fits
from fit.align import apply
ROOT="/mnt/hum01-rds/Basov/p91688di/m1_fits/stage0_pass1"; KSTAR={2:20,1:13}; L={2:"squared_error",1:"KL"}
fs=[load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT,"*","measures.json"))]
cells=defaultdict(list)
for f in latest_fits(fs):
    j=f["job"]
    if j["K"]!=KSTAR[j["beta"]] or (j.get("terms") or {}) or (j.get("arm") or {}) or not f["result"]["converged"]: continue
    cells[(j["approach"],j["beta"],j["group"])].append(f)
rng=np.random.default_rng(0)
def trackB(wm1,wm2,perm,f):
    Y1,Y2=wm1[f]["Y"],wm2[f]["Y"][:,perm]
    cos=(Y1*Y2).sum(1)/np.maximum(np.linalg.norm(Y1,axis=1)*np.linalg.norm(Y2,axis=1),1e-300); w=wm1[f]["mass"]
    return float((w*cos).sum()/w.sum())
out=defaultdict(lambda: defaultdict(list))
for (a,b,g),F in sorted(cells.items()):
    wf,wa=[],[]
    for f in F:
        U,Z=load_UZ(f["dir"]); rel={r:tuple(v) for r,v in f["result"]["relations"].items()}
        wf.append(M.weighted_memberships(U,Z,rel))
        perms=json.load(open(os.path.join(f["dir"],"aligned.json")))["perms"]
        Ua,Za=apply(U,Z,rel,perms); wa.append(M.weighted_memberships(Ua,Za,rel))
    facets=sorted(wf[0])
    for i in range(len(F)):
        for j in range(i+1,len(F)):
            for lab,W in (("joint_fitted",wf),("joint_aligned",wa)):
                p=M.match(W[i],W[j],facets); sh=M._shuffled(W[j],rng); pc=M.match(W[i],sh,facets)
                for f in facets:
                    r,c=trackB(W[i],W[j],p,f),trackB(W[i],sh,pc,f); out[(a,b)][(lab,f)].append((r-c)/(1-c))
            sh=M._shuffled(wf[j],rng)
            for f in facets:
                p=M.match(wf[i],wf[j],[f]); pc=M.match(wf[i],sh,[f])
                r,c=trackB(wf[i],wf[j],p,f),trackB(wf[i],sh,pc,f); out[(a,b)][("per_facet",f)].append((r-c)/(1-c))
    print("done",a,b,g,len(F),flush=True)
rep={}
for (a,b),d in out.items():
    name=f"approach_{a}_{L[b]}"; rep[name]={}
    for (lab,f),v in d.items(): rep[name].setdefault(f,{})[lab]=float(np.mean(v))
    print(name); [print("  ",f,{k:round(x,3) for k,x in v.items()}) for f,v in sorted(rep[name].items())]
json.dump(rep,open("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/aligned/perfacet_stability_kstar.json","w"),indent=1)
