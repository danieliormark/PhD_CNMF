# Reference values for the owner's relaxed rule (2026-11-18): in every relation other than child×article, a community's
# own-numbered cell of the parts table should exceed every other cell of its row (and/or column). Read on stage-0 pass-1
# fits at K* under the aligned numbering (M1_ALIGNED=1). No fits. Per fit and relation:
#   row share  = share of reconstructed tie weight in row communities whose part(k,k) > part(k,l) for every l ≠ k;
#   col share  = the same for column communities (part(k,k) > part(l,k) for every l ≠ k);
#   both share = share of tie weight (half credit) in communities satisfying both;
#   n_row / n_col / n_both = number of communities (of K) satisfying it, unweighted.
# Usage: M1_ALIGNED=1 python -m fit.relaxed_rule_reference --out results/stage0/aligned/relaxed_rule_reference.json
import argparse, json, os
from collections import defaultdict
import numpy as np
os.environ["M1_ALIGNED"] = "1"
from . import data as D, measures as M
from .aggregate import load_UZ
from .facet_detachment_pass1 import all_fits, KSTAR, LOSS


def rule(P):
    K = P.shape[0]; d = np.diag(P); T = P.sum()
    off = P - np.diag(np.full(K, np.inf))
    row_ok = d > off.max(1); col_ok = d > off.max(0)
    rw, cw = P.sum(1) / T, P.sum(0) / T; h = 0.5 * (rw + cw)
    both = row_ok & col_ok
    return {"row_share": float(rw[row_ok].sum()), "col_share": float(cw[col_ok].sum()), "both_share": float(h[both].sum()),
            "n_row": int(row_ok.sum()), "n_col": int(col_ok.sum()), "n_both": int(both.sum()), "K": K}


def run():
    fits = [f for f in all_fits() if f["job"]["K"] == KSTAR[f["job"]["beta"]] and f["job"]["seed"] in range(1, 11)]
    build = D.Build(D.BUILD); cache = {}; rows = defaultdict(list)
    for f in sorted(fits, key=lambda f: (f["job"]["approach"], f["job"]["group"])):
        j = f["job"]; key = (j["approach"], j["group"])
        if key not in cache:
            cache.clear(); cache[key] = D.fit_data(build, j["approach"], group=j["group"])
        U, Z = load_UZ(f["dir"])
        for rd in cache[key]:
            P, _ = M.parts(U, Z, rd)
            rows[(f"approach_{j['approach']}_{LOSS[j['beta']]}", rd.name)].append(rule(P) | {"group": j["group"]})
    rep = {}
    for (cell, name), L in rows.items():
        s = {}
        for sub, LL in (("all_groups", L), ("group_0", [x for x in L if x["group"] == 0])):
            s[sub] = {k: float(np.median([x[k] for x in LL])) for k in ("row_share", "col_share", "both_share", "n_row", "n_col", "n_both")}
            s[sub]["fits"] = len(LL)
        s["K"] = L[0]["K"]
        rep.setdefault(cell, {})[name] = s
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    rep = run(); json.dump(rep, open(ap.parse_args().out, "w"), indent=1)
    for cell, v in sorted(rep.items()):
        print(cell)
        for r, s in sorted(v.items()):
            a = s["all_groups"]
            print(f"  {r:22s} K={s['K']}  row {a['row_share']:.2f} ({a['n_row']:.0f})  col {a['col_share']:.2f} ({a['n_col']:.0f})  both {a['both_share']:.2f} ({a['n_both']:.0f})")


if __name__ == "__main__":
    main()
