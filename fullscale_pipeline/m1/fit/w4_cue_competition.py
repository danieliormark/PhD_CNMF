# W4 follow-up (owner 2026-11-27: "you can run c"; RL-422). Saved fits only. Question: when an article's social cues
# and its content cues disagree, which does the fitted model group by more? After the swap (p 1) an article's social
# ties (authors, organisations, journal, co-author ties, moved together) are independent of its content, so the cues
# compete without being confounded; p 0 (cues aligned in the data) is reported beside.
# Per fit (W4 cells, λ_child = word-similarity regulariser weight off / on, p 0 / 1, 10 seeds):
#   similarity of two articles = cosine of their weighted-membership vectors (main; ignores how concentrated each
#   membership is, RL-421) and co-membership Σ_k p_ik p_jk (beside); excess = group mean − mean over all article pairs.
#   Social cue groups, exclusive, in the data the fit saw: person = pairs sharing a person (co-author tie);
#   organisation = sharing an organisation but no person (approach 1: through their authors; approach 2: article×
#   organisation); journal = sharing a journal but no person and no organisation.
#   Content cue groups, rank-matched: for each social group of N pairs, the N pairs with the highest W-aware content
#   overlap (cosine of x + x·W_rownorm, as swap_link_measure) among pairs sharing no social tie of any kind. Each pair
#   of groups is "the N pairs most strongly linked by that cue".
#   Also: each group's mean W-aware overlap and the all-pairs mean (the swap should leave social groups at about the
#   all-pairs mean content overlap at p 1).
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.w4_cue_competition fit/results/stage0/w4_cue_competition.json
import json, os, sys
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
sys.path.insert(0, ".")
from fit import w4_swap_analysis as W, w4_h1h2_check as H, w4_units as Un, swap_link_measure as S
from fit.align import fit_rds
from fit.plan import W4_CELLS

_cache = {}
G = {}            # per approach: base social incidences, content Gram, candidate content pairs (set in main, inherited by fork)
CUES = ("person", "organisation", "journal")
NCAND = 400000


def base_social(ap_, f):
    rds = {x.name: x for x in fit_rds(f["job"], _cache)}
    n = rds["child_article"].shape[0]
    Ej = S.incidence(rds["article_journal"], n).tocsr()
    if ap_ == 1:
        Ea = S.incidence(rds["author_article"], n); A2 = rds["author_affiliation"]
        Eo = sp.csr_matrix((Ea @ sp.csr_matrix((np.ones(len(A2.tie_row)), (A2.tie_row, A2.tie_col)), shape=A2.shape)) > 0, dtype=float)
    else:
        Eo = S.incidence(rds["article_affiliation"], n).tocsr()
    ca = rds["child_article"]
    X = sp.csr_matrix((np.ones(len(ca.tie_row)), (ca.tie_row, ca.tie_col)), shape=ca.shape)
    Xw = (X + X @ S._WN).tocsr(); nw = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel())
    Gm = (Xw @ Xw.T).toarray() / np.maximum(np.outer(nw, nw), 1e-300)
    iu = np.triu_indices(n, 1); vals = Gm[iu]
    top = np.argpartition(-vals, NCAND)[:NCAND]; top = top[np.argsort(-vals[top])]
    return {"Eo": Eo.tocsr(), "Ej": Ej, "gram": Gm, "cand": (iu[0][top], iu[1][top]), "all_overlap": float(vals.mean())}


def perm_rows(E, perm):
    n = len(perm); Pm = sp.csr_matrix((np.ones(n), (perm, np.arange(n))), shape=(n, n))
    return (Pm @ E).tocsr()


def groups(ap_, perm, T0):
    g = G[ap_]
    T, _ = Un.permuted(T0, sp.csr_matrix((T0.shape[0], 1)), perm)
    Eo, Ej = perm_rows(g["Eo"], perm), perm_rows(g["Ej"], perm)
    So = (Eo @ Eo.T).tocsr(); Sj = (Ej @ Ej.T).tocsr()
    up = lambda M: (sp.triu(M, 1) > 0).astype(float).tocsr()
    Pp, Po, Pj = up(T), up(So), up(Sj)
    excl = {"person": Pp, "organisation": Po - Po.multiply(Pp), "journal": Pj - Pj.multiply(((Pp + Po) > 0).astype(float))}
    pairs = {}
    for k, M in excl.items():
        M = sp.csr_matrix(M); M.eliminate_zeros(); M = M.tocoo(); keep = M.data > 0
        pairs[k] = (M.row[keep], M.col[keep])
    tied = ((Pp + Po + Pj) > 0).tocsr()
    r, c = g["cand"]; free = np.asarray(tied[r, c]).ravel() == 0
    rf, cf = r[free], c[free]
    cont = {k: (rf[:len(pairs[k][0])], cf[:len(pairs[k][0])]) for k in CUES}
    assert all(len(cont[k][0]) == len(pairs[k][0]) for k in CUES), "not enough untied candidate pairs"
    return pairs, cont


def sims(P):
    q = P / np.maximum(np.sqrt((P * P).sum(1, keepdims=True)), 1e-300)
    n = len(P); sq = q.sum(0)
    allc = float((sq @ sq - (q * q).sum()) / (n * (n - 1)))
    allm = H.all_mean(P)
    return q, allc, allm


def one(task):
    (ap_, K, r), seed, fs, T0 = task
    res = {}
    for p in (0.0, 1.0):
        perm = np.load(os.path.join(fs[("off", p)]["dir"], "swap_perm.npy")) if p > 0 else np.arange(T0.shape[0])
        pairs, cont = groups(ap_, perm, T0)
        gm = G[ap_]["gram"]
        d = {"_overlap": {**{f"social_{k}": float(gm[pairs[k]].mean()) for k in CUES},
                          **{f"content_{k}": float(gm[cont[k]].mean()) for k in CUES}, "all": G[ap_]["all_overlap"]},
             "_n": {k: int(len(pairs[k][0])) for k in CUES}}
        for t in ("off", "on"):
            P = H.load(fs[(t, p)])[2]["article"]; q, allc, allm = sims(P)
            cos = lambda ij: float((q[ij[0]] * q[ij[1]]).sum(1).mean()) - allc
            com = lambda ij: float((P[ij[0]] * P[ij[1]]).sum(1).mean()) - allm
            d[t] = {k: {"social_cos": cos(pairs[k]), "content_cos": cos(cont[k]), "social_com": com(pairs[k]), "content_com": com(cont[k])}
                    for k in CUES}
            d[t]["_all"] = {"cos": allc, "com": allm}
        res[f"p{int(p)}"] = d
    return (ap_, K, r), seed, res


def main():
    fits = W.collect(W.ROOT); T0, _ = Un.base_structures(fits)
    for ap_ in (1, 2):
        f = next(iter(fits[next(k for k in fits if k[0] == ap_ and k[3] == 0.0)].values()))
        G[ap_] = base_social(ap_, f)
    tasks = []
    for ap_, K, r in W4_CELLS:
        keys = {(t, p): (ap_, K, r if t == "on" else 0.0, p) for t in ("on", "off") for p in (0.0, 1.0)}
        seeds = sorted(set.intersection(*[set(fits[k]) for k in keys.values()]))
        tasks += [((ap_, K, r), s, {tp: fits[k][s] for tp, k in keys.items()}, T0) for s in seeds]
    with Pool(12) as Pl:
        res = Pl.map(one, tasks)
    rows = {}
    for cell, s, x in res:
        rows.setdefault(cell, {})[s] = x
    out = {}
    for cell, per in rows.items():
        seeds = sorted(per); c = {"seeds": len(seeds)}
        for q in ("p0", "p1"):
            first = per[seeds[0]][q]
            c[q] = {"n_pairs": first["_n"], "overlap": {k: H.ms([per[s][q]["_overlap"][k] for s in seeds]) for k in first["_overlap"]}}
            for k in CUES:
                g = lambda t, m: np.array([per[s][q][t][k][m] for s in seeds])
                e = {}
                for t in ("off", "on"):
                    for sim in ("cos", "com"):
                        so, co = g(t, f"social_{sim}"), g(t, f"content_{sim}")
                        e[f"{t}_{sim}"] = {"social": H.ms(so), "content": H.ms(co), "social_minus_content": H.ms(so - co),
                                           "social_over_content": float(so.mean() / co.mean())}
                for sim in ("cos", "com"):
                    so = g("on", f"social_{sim}") - g("off", f"social_{sim}"); co = g("on", f"content_{sim}") - g("off", f"content_{sim}")
                    e[f"on_minus_off_{sim}"] = {"social": H.ms(so), "content": H.ms(co)}
                c[q][k] = e
        out[f"A{cell[0]} K{cell[1]} r{cell[2]:g}"] = c
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    f = lambda v: f"{v[0]:+.4f}±{v[1]:.4f}"
    for cell, c in out.items():
        for q in ("p1", "p0"):
            print(f"== {cell} {q} pairs {c[q]['n_pairs']} overlap " + " ".join(f"{k} {v[0]:.4f}" for k, v in c[q]["overlap"].items()))
            for k in CUES:
                e = c[q][k]
                for t in ("off", "on"):
                    x = e[f"{t}_cos"]; y = e[f"{t}_com"]
                    print(f"  {k:12s} {t:3s} cos social {f(x['social'])} content {f(x['content'])} s−c {f(x['social_minus_content'])} s/c {x['social_over_content']:.2f}"
                          f" | com social {f(y['social'])} content {f(y['content'])} s/c {y['social_over_content']:.2f}")
                print(f"  {k:12s} on−off cos social {f(e['on_minus_off_cos']['social'])} content {f(e['on_minus_off_cos']['content'])}")


if __name__ == "__main__":
    main()
