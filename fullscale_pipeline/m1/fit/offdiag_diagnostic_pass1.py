# Off-diagonal diagnostic of stage-0 pass 1 across K (ticket 107; decided 2026-11-06 as a stage-0 per-matrix diagnostic
# feeding O1/O3, reported mainly in tie-weight units (2026-11-07) with the squared core-cell form beside it; run across
# all K on the owner's go-ahead 2026-11-17). Per relation, loss, approach and K, median over converged λ = 0 fits
# (5 hold-out groups × 10 seeds):
#   off-diagonal share, tie-weight form  1 − Σ_k part(k,k) ÷ Σ part   (measures.presence "off_diagonal_share")
#   off-diagonal share, core-cell form   Σ_{k≠l} z² ÷ Σ z² on the scaled core (measures.presence "off_diagonal_core_cells")
#   within share after one-to-one renumbering of the column facet (Hungarian, maximal diagonal tie weight) — removes
#     the arbitrary numbering of single-relation facets (organisations, journals, parents; RL-255)
#   effective number of partner communities (label-free; RL-255) and main-partner share
# Usage: python -m fit.offdiag_diagnostic_pass1 --out results/stage0/offdiag_diagnostic_pass1.json
import argparse, json
from collections import defaultdict
import numpy as np
from . import data as D
from .aggregate import load_UZ
from .facet_detachment_pass1 import all_fits, LOSS
from .start_test_analysis import link


def run():
    fits = sorted(all_fits(), key=lambda f: (f["job"]["approach"], f["job"]["group"]))
    build = D.Build(D.BUILD)
    cache, acc = {}, defaultdict(list)
    for n, f in enumerate(fits):
        j = f["job"]; key = (j["approach"], j["group"])
        if key not in cache:
            cache.clear(); cache[key] = {rd.name: rd for rd in D.fit_data(build, j["approach"], group=j["group"])}
        U, Z = load_UZ(f["dir"])
        U = {k: v.astype(np.float64) for k, v in U.items()}; Z = {k: v.astype(np.float64) for k, v in Z.items()}
        pres = f["measures"]["presence"]
        for name, rd in cache[key].items():
            s, _, _ = link(U, Z, rd)
            acc[(j["approach"], j["beta"], name, j["K"])].append(
                {"off_tie_weight": pres[name]["off_diagonal_share"], "off_core_cells": pres[name]["off_diagonal_core_cells"],
                 "off_renumbered": 1 - s["within_renumbered"], "effective_partners": s["effective_partners"],
                 "main_partner": s["main_partner"]})
        if n % 100 == 0:
            print(n, "of", len(fits), flush=True)
    rep = {}
    for (a, b, name, K), L in sorted(acc.items()):
        cell = rep.setdefault(f"approach_{a}_{LOSS[b]}", {}).setdefault(name, {})
        cell[str(K)] = {"fits": len(L), **{k: float(np.median([x[k] for x in L])) for k in L[0]},
                        "effective_partners_q10_q90": [float(np.quantile([x["effective_partners"] for x in L], q)) for q in (0.1, 0.9)]}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
