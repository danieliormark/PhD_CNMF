"""Aggregate t3_<scenario>_<regime>.json: per scenario, regime, K and weight, the mean over
replicates of recovery accuracy (all / persisting / new / the scenario's special group),
the change against weight 0 within the same replicate (paired), accepted pairs, false
matches, emerged community matched, spurious new-only communities, convergence."""
import glob, json, collections
import numpy as np

SPECIAL = {"growth": "new", "migration": "migrant", "split": "split", "merge": "merged",
           "dissolve_emerge": "emerged"}
rows = []
for path in sorted(glob.glob("t3_*_*.json")):
    sc, rg = path[3:-5].rsplit("_", 1)
    for r in json.load(open(path)):
        r.update(scenario=sc, regime=rg)
        rows.append(r)

by = collections.defaultdict(list)
for r in rows:
    by[(r["scenario"], r["regime"], r["K"], r["lambda"])].append(r)
base = {(r["scenario"], r["regime"], r["K"], r["rep"]): r for r in rows if r["lambda"] == 0.0}

summary = []
print(f"{'scenario':16s} {'reg':5s} K  lambda | acc_all pers  new   special (d vs 0) | accepted false emerged_m new_only conv")
for key in sorted(by):
    sc, rg, K, lam = key
    rs = by[key]
    sp = SPECIAL[sc]
    acc = lambda g: [r["accuracy"].get(g) for r in rs if r["accuracy"].get(g) is not None]
    d_sp = [r["accuracy"].get(sp, np.nan) - base[(sc, rg, K, r["rep"])]["accuracy"].get(sp, np.nan) for r in rs]
    d_pers = [r["accuracy"].get("persisting", np.nan) - base[(sc, rg, K, r["rep"])]["accuracy"].get("persisting", np.nan) for r in rs]
    rec = dict(scenario=sc, regime=rg, K=K, weight=lam, n=len(rs),
               acc_all=float(np.mean(acc("all"))), acc_persisting=float(np.mean(acc("persisting"))),
               acc_new=float(np.mean(acc("new"))) if acc("new") else None,
               acc_special=float(np.mean(acc(sp))) if acc(sp) else None,
               d_special=float(np.nanmean(d_sp)), d_persisting=float(np.nanmean(d_pers)),
               accepted=float(np.mean([r["accepted"] for r in rs])),
               false_rate=float(sum(r["false_matches"] for r in rs) / max(1, sum(r["accepted"] for r in rs))),
               emerged_matched=sum(bool(r["emerged_matched"]) for r in rs),
               new_only=float(np.mean([len(r["new_only"]) for r in rs])),
               converged=sum(bool(r["converged"]) for r in rs))
    summary.append(rec)
    print(f"{sc:16s} {rg:5s} {K}  {lam:<5} | {rec['acc_all']:.3f} {rec['acc_persisting']:.3f} "
          f"{(rec['acc_new'] or 0):.3f} {sp}={(rec['acc_special'] or 0):.3f} ({rec['d_special']:+.3f}) "
          f"pers d {rec['d_persisting']:+.3f} | {rec['accepted']:.1f} {rec['false_rate']:.2f} "
          f"{rec['emerged_matched']} {rec['new_only']:.1f} {rec['converged']}/{rec['n']}")
json.dump(summary, open("t3_summary.json", "w"), indent=1)
