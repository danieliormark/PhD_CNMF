# W3 report (2026-11-23, RL-352): the combined held-out score of a fit weights relations by that fit's own relation
# weights (run_fit → measures.combined_two_part), so mass-weight fits were scored with more weight on the affiliation
# relation. Rescore mass-weight fits and their pass-3 twins per relation and combined with the SAME (current) weights;
# semantic-equal scores as decided (2026-11-02). Paired by twin, mean per hold-out group, mean ± SE over 5 groups.
# Saved measures only. Usage: PYTHONPATH=. python -m fit.weight_check_rescore > fit/results/stage0/weight_check_rescore.log
import os; os.environ["M1_ALIGNED"]="0"
import glob, json, math
from collections import defaultdict
import numpy as np
from fit.aggregate import load_fit, latest_fits
from fit.weight_check_analysis import key
ROOT="/mnt/hum01-rds/Basov/p91688di/m1_fits"
chk=latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(f"{ROOT}/weight_check/*/measures.json")])
p3=latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(f"{ROOT}/stage0_pass3/*/measures.json")])
twin={key(f["job"]):f for f in p3}
cur_w={"parent_child":0.25,"child_article":0.25}
rows=defaultdict(list)
for f in chk:
    if f["job"]["arm"].get("swap_p"): continue
    k=key(f["job"]); t=twin.get(k)
    if t is None or not(f["result"]["converged"] and t["result"]["converged"]): continue
    rows[(k[0],k[1],k[2])].append((k[3],f,t))
def ex(f):
    tp=f["measures"]["heldout_two_part"]; out={}
    for r,v in tp.items():
        if v is None: continue
        out[f"AP_{r}"]=v.get("ap_within_class"); 
        if v.get("value_partial_spearman") is not None: out[f"val_{r}"]=v["value_partial_spearman"]
    return out
res={}
for cell,rs in sorted(rows.items()):
    # current weights from a twin's spec (relation_weights)
    W=json.load(open(os.path.join(rs[0][2]["dir"],"spec.json")))["relation_weights"]
    by=defaultdict(lambda: defaultdict(list))
    for g,f,t in rs:
        a,b=ex(f),ex(t)
        for kk in a:
            if a[kk] is None or b.get(kk) is None: continue
            by[kk][g].append((a[kk],b[kk]))
        for lab,src in (("m",a),("c",b)):
            pass
        # combined with current weights
        for nm,keys in (("existence_current_weights",[k for k in a if k.startswith("AP_")]),):
            def comb(s):
                rr=[k[3:] for k in keys if s.get(k) is not None]; tot=sum(W[r] for r in rr)
                return sum(W[r]*s["AP_"+r] for r in rr)/tot
            by[nm][g].append((comb(a),comb(b)))
        for nm,rels in (("existence_semantic_equal",("child_article","parent_child")),("value_semantic_equal",("child_article","parent_child"))):
            pre="AP_" if nm.startswith("existence") else "val_"
            by[nm][g].append((np.mean([a[pre+r] for r in rels]),np.mean([b[pre+r] for r in rels])))
    out={}
    for kk,gg in by.items():
        m=np.array([np.mean([x[0] for x in v]) for v in gg.values()]); c=np.array([np.mean([x[1] for x in v]) for v in gg.values()]); d=m-c
        out[kk]=(m.mean(),c.mean(),d.mean(),d.std(ddof=1)/math.sqrt(len(d)),d.mean()/c.mean())
    res[cell]=out
    print(f"\napproach {cell[0]} K {cell[1]} r {cell[2]:g}  pairs {len(rs)}   (mass, current, diff ± SE over 5 groups, relative)")
    for kk in sorted(out):
        m,c,d,se,rel=out[kk]; print(f"   {kk:34s} {m:.4f} {c:.4f} {d:+.4f} ± {se:.4f}  ({rel:+.0%})")
