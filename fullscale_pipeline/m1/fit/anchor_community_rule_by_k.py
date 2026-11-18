# Owner's anchor requirement (2026-11-18, DECIDED): in child×article, every community's own-numbered cell of the parts
# table must be more than half of its child×article tie weight (row and column side). Read on all stage-0 pass-1 fits,
# every K, aligned numbering. No fits. Per setting and K: fits in which every community passes; median number of
# failing communities; median share of child×article tie weight (half credit) in failing communities.
# Also on stage-0 pass-2 fits (λ_child on), grouped by K and r (--stage stage0_pass2).
# Usage: python -m fit.anchor_community_rule_by_k [--stage stage0_pass2] --out results/stage0/aligned/anchor_community_rule_by_k[_pass2].json
import argparse, json, os
from collections import defaultdict
import numpy as np
os.environ["M1_ALIGNED"] = "1"
from . import data as D, measures as M
from .aggregate import load_UZ
from .facet_detachment_pass1 import all_fits, LOSS
from .aggregate import load_fit, latest_fits
import glob

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def run(stage="stage0_pass1"):
    fits = all_fits() if stage == "stage0_pass1" else latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, stage, "*", "measures.json"))])
    build = D.Build(D.BUILD); cache = {}; rows = defaultdict(list)
    for f in sorted(fits, key=lambda f: (f["job"]["approach"], f["job"]["group"])):
        j = f["job"]
        if not f["result"]["converged"]:
            continue
        key = (j["approach"], j["group"])
        if key not in cache:
            cache.clear(); cache[key] = {r.name: r for r in D.fit_data(build, j["approach"], group=j["group"])}
        U, Z = load_UZ(f["dir"]); P, T = M.parts(U, Z, cache[key]["child_article"])
        d = np.diag(P); rw, cw = P.sum(1), P.sum(0); h = 0.5 * (rw + cw) / T
        ok = (d / np.maximum(rw, 1e-300) > 0.5) & (d / np.maximum(cw, 1e-300) > 0.5)
        rows[(f"approach_{j['approach']}_{LOSS[j['beta']]}", j["K"] if stage == "stage0_pass1" else f"K{j['K']}_r{(j.get('arm') or {}).get('r')}")].append((bool(ok.all()), int((~ok).sum()), float(h[~ok].sum())))
    rep = defaultdict(dict)
    for (cell, K), L in sorted(rows.items()):
        a = np.array(L, dtype=float)
        rep[cell][K] = {"fits": len(L), "fits_all_pass": int(a[:, 0].sum()), "failing_median": float(np.median(a[:, 1])),
                        "failing_weight_median": float(np.median(a[:, 2]))}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--stage", default="stage0_pass1")
    a = ap.parse_args(); rep = run(a.stage); json.dump(rep, open(a.out, "w"), indent=1)
    for c, v in rep.items():
        print(c); [print(f"  {K}: all communities pass in {s['fits_all_pass']}/{s['fits']} fits; failing median {s['failing_median']:.0f}, weight {s['failing_weight_median']:.2f}") for K, s in v.items()]


if __name__ == "__main__":
    main()
