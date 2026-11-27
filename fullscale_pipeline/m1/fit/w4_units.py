# W4 follow-up (owner 2026-11-27: "For tie-weighting for articles check, I'd actually consider making analysis with all 3
# units"; RL-419). Saved fits only (the W4 cells, λ_child off / on × swap p 0 / 1 × 10 seeds), no new fits.
# λ_child = weight of the word-similarity regulariser (child×child W); off = r 0, on = the cell's r.
# C_ij = Σ_k p_ik p_jk (weighted memberships, ticket 100); m = mean C over all article pairs. Weights w_ij = co-author tie
# value (P5: Σ_shared persons 1/(n_p − 1) ÷ |authors(A) ∪ authors(B)|, then ÷ (s_A s_B)^¼), the SAME in both approaches;
# persons from the author×article relation (P1). After the swap both move with the article's social bundle (the fit's
# own saved permutation), so every unit is read in the data that fit saw.
# Units (excess over m in each):
#   pairs (unweighted)  mean C over article pairs with a co-author tie (= pairs sharing a person) — RL-415's reading
#   (a) tie-weighted    Σ w_ij C_ij ÷ Σ w_ij over those pairs
#   (b) article level   c_i = Σ_j w_ij C_ij ÷ Σ_j w_ij over the articles j tied to i; mean over articles with a tie
#   (c) person level    mean C over pairs of a person's articles; mean over persons with ≥ 2 articles (unweighted)
# Per unit: the four levels, response to the swap R = excess(p 1) − excess(p 0) with λ_child off / on, decided W4
# flatness f = −sign(R̄_off)(R_on − R_off) (flag f > 2 SE; testable |R̄_off| > 2 SE), λ_child's addition at p 0 / p 1,
# and the RL-417 counterfactual shares at p 1 (per-article concentration, mean article membership, both).
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.w4_units fit/results/stage0/w4_units.json
import json, math, os, sys
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
sys.path.insert(0, ".")
from fit import w4_swap_analysis as W
from fit import w4_h1h2_check as H
from fit.align import fit_rds
from fit.swap_link_measure import incidence
from fit.plan import W4_CELLS

_cache = {}
UNITS = ("pairs", "tie_weighted", "article", "person")


def base_structures(fits):
    """Unswapped co-author tie matrix (from an approach-2 p 0 fit) and article×person incidence (approach-1 p 0 fit)."""
    f2 = next(iter(fits[next(k for k in fits if k[0] == 2 and k[3] == 0.0)].values()))
    f1 = next(iter(fits[next(k for k in fits if k[0] == 1 and k[3] == 0.0)].values()))
    rd = {r.name: r for r in fit_rds(f2["job"], _cache)}["article_article"]
    n = rd.shape[0]
    T = sp.csr_matrix((rd.tie_val, (rd.tie_row, rd.tie_col)), shape=(n, n)); T = T.maximum(T.T).tocsr()
    E = incidence({r.name: r for r in fit_rds(f1["job"], _cache)}["author_article"], n).tocsr()
    return T, E


def permuted(T, E, perm):
    """Swapped index of original article a = perm[a] (worlds.graded_swap)."""
    n = len(perm); Pm = sp.csr_matrix((np.ones(n), (perm, np.arange(n))), shape=(n, n))
    return (Pm @ T @ Pm.T).tocsr(), (Pm @ E).tocsr()


def readings(X, T, E, iu, w):
    m = H.all_mean(X); r, c = iu
    C = (X[r] * X[c]).sum(1)
    out = {"pairs": float(C.mean()) - m, "tie_weighted": float((w * C).sum() / w.sum()) - m}
    n = len(X); num = np.zeros(n); den = np.zeros(n)
    np.add.at(num, r, w * C); np.add.at(num, c, w * C); np.add.at(den, r, w); np.add.at(den, c, w)
    has = den > 0
    out["article"] = float((num[has] / den[has]).mean()) - m
    Y = E.T @ X; sq = E.T @ (X * X).sum(1); npers = np.asarray(E.sum(0)).ravel(); ok = npers >= 2
    out["person"] = float((((Y[ok] ** 2).sum(1) - sq[ok]) / (npers[ok] * (npers[ok] - 1))).mean()) - m
    return out


def one(task):
    (ap_, K, r), seed, fs, T0, E0 = task
    res = {}
    for p in (0.0, 1.0):
        fo, fn = fs[("off", p)], fs[("on", p)]
        n = T0.shape[0]
        if p > 0:
            perm = np.load(os.path.join(fo["dir"], "swap_perm.npy")); perm_on = np.load(os.path.join(fn["dir"], "swap_perm.npy"))
            assert (perm == perm_on).all(), "on and off fits must share the swap"
        else:
            perm = np.arange(n)
        T, E = permuted(T0, E0, perm)
        # sanity: the rebuilt structures equal the fit's own social relation
        rds = {x.name: x for x in fit_rds(fo["job"], _cache)}
        if ap_ == 1:
            assert (abs(incidence(rds["author_article"], n) - E) > 0).nnz == 0
        else:
            own = sp.csr_matrix((rds["article_article"].tie_val, (rds["article_article"].tie_row, rds["article_article"].tie_col)), shape=(n, n))
            assert abs(own.maximum(own.T) - T).max() < 1e-12
        U = sp.triu(T, 1).tocoo(); iu = (U.row, U.col); w = U.data
        Xoff = H.load(fo)[2]["article"]; Xon = H.load(fn)[2]["article"]
        V = {"off": Xoff, "on": Xon,
             "conc1": H.match_entropy(Xoff, H.entropy(Xon))[0], "conc2": H.match_entropy(Xon, H.entropy(Xoff))[0],
             "size1": H.match_sizes(Xoff, Xon.mean(0)), "size2": H.match_sizes(Xon, Xoff.mean(0)),
             "both1": H.match_both(Xoff, Xon.mean(0), H.entropy(Xon)), "both2": H.match_both(Xon, Xoff.mean(0), H.entropy(Xoff))}
        res[f"p{int(p)}"] = {k: readings(x, T, E, iu, w) for k, x in V.items()}
        res[f"p{int(p)}"]["_n"] = {"pairs": len(w), "articles": int(np.unique(np.r_[iu[0], iu[1]]).size),
                                   "persons_2plus": int((np.asarray(E.sum(0)).ravel() >= 2).sum())}
    return (ap_, K, r), seed, res


def summarise(rows):
    out = {}
    for cell, per in rows.items():
        seeds = sorted(per); c = {"seeds": len(seeds), "counts": per[seeds[0]]["p0"]["_n"]}
        for u in UNITS:
            g = lambda pk, k: np.array([per[s][pk][k][u] for s in seeds])
            d = {f"{t}_p{q}": H.ms(g(f"p{q}", t)) for t in ("off", "on") for q in (0, 1)}
            Roff, Ron = g("p1", "off") - g("p0", "off"), g("p1", "on") - g("p0", "on")
            sgn = np.sign(Roff.mean()); f = -sgn * (Ron - Roff)
            d.update({"R_off": H.ms(Roff), "R_on": H.ms(Ron), "flatness": H.ms(f)})
            d["testable"] = bool(abs(d["R_off"][0]) > 2 * d["R_off"][1]); d["flag"] = bool(d["testable"] and d["flatness"][0] > 2 * d["flatness"][1])
            for q in (0, 1):
                tot = g(f"p{q}", "on") - g(f"p{q}", "off"); d[f"addition_p{q}"] = H.ms(tot)
            tot = g("p1", "on") - g("p1", "off")
            for v in ("conc", "size", "both"):
                o1, o2 = g("p1", v + "1") - g("p1", "off"), g("p1", "on") - g("p1", v + "2")
                d[f"{v}_share_p1"] = H.share((o1 + o2) / 2, tot)
            c[u] = d
        out[f"A{cell[0]} K{cell[1]} r{cell[2]:g}"] = c
    return out


def main():
    fits = W.collect(W.ROOT); T0, E0 = base_structures(fits); tasks = []
    for ap_, K, r in W4_CELLS:
        keys = {(t, p): (ap_, K, r if t == "on" else 0.0, p) for t in ("on", "off") for p in (0.0, 1.0)}
        seeds = sorted(set.intersection(*[set(fits[k]) for k in keys.values()]))
        tasks += [((ap_, K, r), s, {tp: fits[k][s] for tp, k in keys.items()}, T0, E0) for s in seeds]
    with Pool(12) as P:
        res = P.map(one, tasks)
    rows = {}
    for cell, s, x in res:
        rows.setdefault(cell, {})[s] = x
    out = summarise(rows)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    f = lambda v: f"{v[0]:+.4f}±{v[1]:.4f}"; sh = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for cell, c in out.items():
        print(f"== {cell} seeds {c['seeds']} counts {c['counts']}")
        for u in UNITS:
            d = c[u]
            print(f"  {u:12s} off {d['off_p0'][0]:.4f}→{d['off_p1'][0]:.4f} on {d['on_p0'][0]:.4f}→{d['on_p1'][0]:.4f} | R_off {f(d['R_off'])} R_on {f(d['R_on'])}"
                  f" f {f(d['flatness'])} testable {d['testable']} flag {d['flag']} | add p0 {f(d['addition_p0'])} p1 {f(d['addition_p1'])}"
                  f" | conc {sh(d['conc_share_p1'])} size {sh(d['size_share_p1'])} both {sh(d['both_share_p1'])}")


if __name__ == "__main__":
    main()
