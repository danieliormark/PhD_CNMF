# W4 follow-up (owner 2026-11-27: "present the three-unit results, then run the check"; RL-421). Saved fits only.
# Question: does the model's co-membership of article pairs that share a person rise with the pair's content overlap,
# and does the word-similarity regulariser (λ_child, child×child W) change that? Works where the swap has no power.
# Per fit (W4 cells, λ_child off / on, swap p 0 / 1, 10 seeds), with C_ij = Σ_k p_ik p_jk (weighted memberships):
#   overlaps of a pair: shared child hyperedges (count), Jaccard of child-hyperedge sets, W-aware cosine (each article's
#   child-hyperedge vector x extended to x + x·W_rownorm, as swap_link_measure) — visible ties of the data the fit saw;
#   CHANGED after the one-seed test (2026-11-27): co-membership falls with overlap even for random pairs (Spearman
#   ≈ −0.4 by count), because articles with many child hyperedges have more concentrated memberships and C = ‖p_i‖‖p_j‖
#   × cosine mixes concentration with similarity. Main similarity is therefore the cosine of the two membership vectors
#   (co-membership reported beside), and every ρ partials out the pair's article sizes (log size_i + log size_j).
#   ρ_same   = Spearman(similarity, overlap) over the 1,107 pairs with a co-author tie (= sharing a person; after the swap, the
#              pairs in the swapped data);
#   ρ_partial = the same, with the rank of the co-author tie value partialled out (tie strength may drive both);
#   ρ_random = Spearman(C, overlap) over 50,000 random article pairs without a co-author tie (fixed sample) — the
#              model's generic content dependence, the yardstick;
#   gap      = ρ_same − ρ_random (negative: a person's articles are grouped less by content than articles in general).
# Summary: mean ± SE over seeds; λ_child on − off paired by seed.
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.w4_content_dependence fit/results/stage0/w4_content_dependence.json
import json, os, sys
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
from scipy.stats import rankdata
sys.path.insert(0, ".")
from fit import w4_swap_analysis as W, w4_h1h2_check as H, w4_units as Un, swap_link_measure as S
from fit.align import fit_rds
from fit.plan import W4_CELLS

_cache = {}
OVERLAPS = ("count", "jaccard", "w_cosine")


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, *zs):
    """Spearman of a and b after regressing both ranks on the ranks of the covariates zs (with intercept)."""
    ra, rb = rankdata(a), rankdata(b)
    Z = np.column_stack([np.ones(len(a))] + [rankdata(z) for z in zs])
    res = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def content(f):
    ca = {x.name: x for x in fit_rds(f["job"], _cache)}["child_article"]
    X = sp.csr_matrix((np.ones(len(ca.tie_row)), (ca.tie_row, ca.tie_col)), shape=ca.shape)
    Xw = (X + X @ S._WN).tocsr(); nw = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel())
    return X, np.asarray(X.sum(1)).ravel(), Xw, nw


def overlaps(Xc, r, c):
    X, size, Xw, nw = Xc
    inter = np.asarray(X[r].multiply(X[c]).sum(1)).ravel()
    union = size[r] + size[c] - inter
    wc = np.asarray(Xw[r].multiply(Xw[c]).sum(1)).ravel() / np.maximum(nw[r] * nw[c], 1e-300)
    return {"count": inter, "jaccard": np.where(union > 0, inter / np.maximum(union, 1), 0.0), "w_cosine": wc}


def one(task):
    (ap_, K, r), seed, fs, T0, E0, rand = task
    res = {}
    for p in (0.0, 1.0):
        fo = fs[("off", p)]
        perm = np.load(os.path.join(fo["dir"], "swap_perm.npy")) if p > 0 else np.arange(T0.shape[0])
        T, _ = Un.permuted(T0, E0, perm)
        U = sp.triu(T, 1).tocoo(); rs, cs, w = U.row, U.col, U.data
        Xc = content(fo)                                   # semantic ties are not swapped; same for on and off
        ov_s, ov_r = overlaps(Xc, rs, cs), overlaps(Xc, *rand)
        size = Xc[1]; ls = np.log(np.maximum(size, 1))
        zs_s, zs_r = ls[rs] + ls[cs], ls[rand[0]] + ls[rand[1]]
        for t in ("off", "on"):
            P = H.load(fs[(t, p)])[2]["article"]; nP = np.sqrt((P * P).sum(1))
            d = {}
            for sim in ("cosine", "comembership"):
                f = (lambda i, j: (P[i] * P[j]).sum(1) / np.maximum(nP[i] * nP[j], 1e-300)) if sim == "cosine" else (lambda i, j: (P[i] * P[j]).sum(1))
                Ss, Sr = f(rs, cs), f(*rand)
                d[sim] = {o: {"same": partial_spearman(Ss, ov_s[o], zs_s), "partial": partial_spearman(Ss, ov_s[o], zs_s, w),
                              "random": partial_spearman(Sr, ov_r[o], zs_r), "same_raw": spearman(Ss, ov_s[o]),
                              "random_raw": spearman(Sr, ov_r[o])} for o in OVERLAPS}
            res[f"{t}_p{int(p)}"] = d
    return (ap_, K, r), seed, res


def main():
    fits = W.collect(W.ROOT); T0, E0 = Un.base_structures(fits); n = T0.shape[0]
    rng = np.random.default_rng(107)
    a = rng.integers(0, n, 70000); b = rng.integers(0, n, 70000); keep = a != b
    a, b = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
    keep = np.asarray(T0[a, b]).ravel() == 0                 # random pairs without a co-author tie (unswapped data)
    rand = (a[keep][:50000], b[keep][:50000])
    tasks = []
    for ap_, K, r in W4_CELLS:
        keys = {(t, p): (ap_, K, r if t == "on" else 0.0, p) for t in ("on", "off") for p in (0.0, 1.0)}
        seeds = sorted(set.intersection(*[set(fits[k]) for k in keys.values()]))
        tasks += [((ap_, K, r), s, {tp: fits[k][s] for tp, k in keys.items()}, T0, E0, rand) for s in seeds]
    with Pool(12) as P:
        res = P.map(one, tasks)
    rows = {}
    for cell, s, x in res:
        rows.setdefault(cell, {})[s] = x
    out = {}
    for cell, per in rows.items():
        seeds = sorted(per); c = {"seeds": len(seeds)}
        for sim, o in [(a, b) for a in ("cosine", "comembership") for b in OVERLAPS]:
            d = {}
            for q in ("p0", "p1"):
                g = lambda t, k: np.array([per[s][f"{t}_{q}"][sim][o][k] for s in seeds])
                for k in ("same", "partial", "random", "same_raw", "random_raw"):
                    d[f"{k}_off_{q}"] = H.ms(g("off", k)); d[f"{k}_on_minus_off_{q}"] = H.ms(g("on", k) - g("off", k))
                d[f"gap_off_{q}"] = H.ms(g("off", "same") - g("off", "random"))
                d[f"gap_on_minus_off_{q}"] = H.ms((g("on", "same") - g("on", "random")) - (g("off", "same") - g("off", "random")))
            c[f"{sim}|{o}"] = d
        out[f"A{cell[0]} K{cell[1]} r{cell[2]:g}"] = c
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    f = lambda v: f"{v[0]:+.3f}±{v[1]:.3f}"
    for cell, c in out.items():
        print(f"== {cell} seeds {c['seeds']}")
        for o in [k for k in c if "|" in k]:
            for q in ("p0", "p1"):
                d = c[o]
                print(f"  {o:22s} {q}: same off {f(d[f'same_off_{q}'])} on−off {f(d[f'same_on_minus_off_{q}'])} | partial off {f(d[f'partial_off_{q}'])}"
                      f" on−off {f(d[f'partial_on_minus_off_{q}'])} | random off {f(d[f'random_off_{q}'])} on−off {f(d[f'random_on_minus_off_{q}'])}"
                      f" | gap off {f(d[f'gap_off_{q}'])} on−off {f(d[f'gap_on_minus_off_{q}'])}")


if __name__ == "__main__":
    main()
