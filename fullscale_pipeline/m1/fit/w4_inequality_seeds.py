# W4 follow-up, owner's point 3 (2026-11-27; RL-423): does λ_child's content-independent addition co-vary, across seeds,
# with how much λ_child raises the inequality of mean article memberships (Σ_k m_k², m_k = Σ_i p_ik ÷ n)? Natural
# variation only, not an intervention. Per cell and seed, after the swap (p 1): Δ inequality = Σ m_k² (λ_child on) −
# (off); addition = excess (on) − (off) in the unweighted-pair and article-level units (RL-420). Spearman across the 10
# seeds per cell, and pooled after standardising within cell. Saved fits only.
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.w4_inequality_seeds fit/results/stage0/w4_inequality_seeds.json
import json, os, sys
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from fit import w4_swap_analysis as W, w4_h1h2_check as H, w4_units as Un
from fit.plan import W4_CELLS


def one(task):
    cell, seed, fo, fn, T0, E0 = task
    perm = np.load(os.path.join(fo["dir"], "swap_perm.npy"))
    T, E = Un.permuted(T0, E0, perm); U = sp.triu(T, 1).tocoo(); iu = (U.row, U.col)
    Xo, Xn = H.load(fo)[2]["article"], H.load(fn)[2]["article"]
    ro, rn = Un.readings(Xo, T, E, iu, U.data), Un.readings(Xn, T, E, iu, U.data)
    return cell, seed, {"d_ineq": float((Xn.mean(0) ** 2).sum() - (Xo.mean(0) ** 2).sum()),
                        "d_entropy_bits": float((H.entropy(Xn).mean() - H.entropy(Xo).mean()) / np.log(2)),
                        **{f"add_{u}": rn[u] - ro[u] for u in ("pairs", "article", "tie_weighted")}}


def main():
    fits = W.collect(W.ROOT); T0, E0 = Un.base_structures(fits); tasks = []
    for ap_, K, r in W4_CELLS:
        on, off = (ap_, K, r, 1.0), (ap_, K, 0.0, 1.0)
        for s in sorted(set(fits[on]) & set(fits[off])):
            tasks.append(((ap_, K, r), s, fits[off][s], fits[on][s], T0, E0))
    with Pool(12) as P:
        res = P.map(one, tasks)
    rows = {}
    for cell, s, x in res:
        rows.setdefault(f"A{cell[0]} K{cell[1]} r{cell[2]:g}", {})[s] = x
    out = {"per_seed": rows, "cells": {}, "pooled": {}}
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    pool = {u: ([], []) for u in ("pairs", "article", "tie_weighted")}
    for cell, per in rows.items():
        d = np.array([per[s]["d_ineq"] for s in per])
        c = {"d_ineq": H.ms(d), "d_ineq_cv": float(d.std(ddof=1) / abs(d.mean()))}
        for u in pool:
            a = np.array([per[s][f"add_{u}"] for s in per])
            rho, pv = spearmanr(d, a); c[u] = {"addition": H.ms(a), "spearman": float(rho), "p": float(pv)}
            pool[u][0].extend(z(d)); pool[u][1].extend(z(a))
        out["cells"][cell] = c
    for u, (x, y) in pool.items():
        rho, pv = spearmanr(x, y); out["pooled"][u] = {"spearman": float(rho), "p": float(pv), "n": len(x)}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for cell, c in out["cells"].items():
        print(cell, f"Δineq {c['d_ineq'][0]:+.4f}±{c['d_ineq'][1]:.4f} (CV {c['d_ineq_cv']:.2f})",
              " | ".join(f"{u}: add {c[u]['addition'][0]:+.4f} ρ {c[u]['spearman']:+.2f} (p {c[u]['p']:.2f})" for u in pool))
    print("pooled", {u: {k: round(v, 3) for k, v in d.items()} for u, d in out["pooled"].items()})


if __name__ == "__main__":
    main()
