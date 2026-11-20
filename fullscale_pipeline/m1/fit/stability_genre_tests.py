# Lower seed stability in study B (no genre hyperedges; RL-303/305): tests 1–3 (owner 2026-11-20: "Run tests 1–3").
# Saved fits only: stage-0 pass 1, KL, λ = 0, K 5 and 13, both approaches, hold-out groups 1–5 × seeds 1–10, study A
# (stage0_pass1) and study B (B_stage0_pass1). Seed pairs within a group, communities matched by measures.match (the
# Track A/B matching); Track A / B as measures.track_pair (A: 1 − JS distance over placed entities; B: mass-weighted
# cosine), chance = fit 2's rows shuffled per facet, adjusted (real − chance) / (1 − chance), mean over pairs.
#   Test 1 (measurement, H4): study A's stability with the genre child hyperedges left out of the child facet (Track A:
#     placed ∩ substantive; Track B: substantive rows only), beside the usual reading; per facet raw values too.
#   Test 2 (where the instability sits, H1 anchor vs H2 less data): per article, agreement across seeds = 1 − JS
#     distance between its matched memberships, mean over the group's seed pairs, then over groups. Study A's curve
#     of agreement against log(1 + the article's child×article ties) (lowess-free: means in tie-count bins) predicts
#     study B's agreement from the article's remaining ties; the residual (B − predicted) is related to the share of the
#     article's ties that were genre (genre loss): H2 → residual ≈ 0 and flat in genre loss; H1 → negative and falling
#     with genre loss. Also within bins of remaining ties: B − A agreement for articles above vs below the bin's median
#     genre loss.
#   Test 3 (solution modes, H3): per group, the 10 × 10 matrix of seed-pair agreement (model-level Track A, raw); two
#     clusters by average linkage; sizes, mean agreement within clusters and between them; final data loss per seed and
#     the loss difference between the clusters (in units of the group's loss SD).
# Usage: python -m fit.stability_genre_tests --out fit/results/stage0/stability_genre_tests.json [--workers 8]
import argparse, glob, json, os
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits, load_entity

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
STAGES = {"A": "stage0_pass1", "B": "B_stage0_pass1"}
KS = (5, 13)
_G = {}


def setup():
    from .generic_hyperedges import labels
    b = D.Build(D.BUILD); _G["genre"] = labels(b) == "G"


def tracks(wm1, wm2, perm, placed, masks):
    """Per facet Track A and B with optional row masks {facet: bool}."""
    A, B = {}, {}
    for f in wm1:
        m = placed[f] & masks.get(f, True)
        P1, P2 = wm1[f]["P"], wm2[f]["P"][:, perm]
        A[f] = float(1 - M._js_distance_e(P1[m], P2[m]).mean()) if m.any() else np.nan
        mm = np.ones(len(P1), bool) & masks.get(f, True)
        Y1, Y2 = wm1[f]["Y"][mm], wm2[f]["Y"][:, perm][mm]
        cos = (Y1 * Y2).sum(1) / np.maximum(np.linalg.norm(Y1, axis=1) * np.linalg.norm(Y2, axis=1), 1e-300)
        w = wm1[f]["mass"][mm]
        B[f] = float((w * cos).sum() / w.sum()) if w.sum() > 0 else np.nan
    return A, B


def task(args):
    study, a, K, g, dirs = args
    fits = [load_fit(d) for d in dirs]
    fits = sorted([f for f in fits if f["result"]["converged"]], key=lambda f: f["job"]["seed"])
    ents = [load_entity(f["dir"]) for f in fits]
    placed = {fa: M.placed_entities([e[fa]["reproduced_share"] for e in ents], ents[0][fa]["reproduced_share_degree_only"])
              for fa in ents[0]}
    rel = {r: tuple(v) for r, v in fits[0]["result"]["relations"].items()}
    wms = [M.weighted_memberships(*load_UZ(f["dir"]), rel) for f in fits]
    facets = sorted(wms[0]); n = len(wms)
    masks_sub = {"child": ~_G["genre"]} if study == "A" else {}
    rng = np.random.default_rng(0)
    pairs, art = [], np.zeros(len(wms[0]["article"]["P"])); agree = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        perm = M.match(wms[i], wms[j], facets)
        sh = M._shuffled(wms[j], rng)
        out = {}
        for lab, masks in (("all", {}), ("substantive_child", masks_sub)):
            if lab == "substantive_child" and not masks:
                continue
            rA, rB = tracks(wms[i], wms[j], perm, placed, masks)
            cA, cB = tracks(wms[i], sh, M.match(wms[i], sh, facets), placed, masks)
            mA, mB = np.nanmean(list(rA.values())), np.nanmean(list(rB.values()))
            cAm, cBm = np.nanmean(list(cA.values())), np.nanmean(list(cB.values()))
            out[lab] = {"A_facet": rA, "B_facet": rB, "A_model": mA, "B_model": mB,
                        "A_adj": (mA - cAm) / (1 - cAm), "B_adj": (mB - cBm) / (1 - cBm)}
        pairs.append(out); agree[i, j] = agree[j, i] = out["all"]["A_model"]
        P1, P2 = wms[i]["article"]["P"], wms[j]["article"]["P"][:, perm]
        art += 1 - M._js_distance_e(P1, P2)
    art /= len(pairs)
    loss = np.array([float(f["result"]["data_loss_history"][-1]) for f in fits])
    # test 3: two clusters by average linkage on 1 − agreement
    dist = 1 - agree; np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), 2, "maxclust")
    within = [agree[i, j] for i, j in combinations(range(n), 2) if lab[i] == lab[j]]
    between = [agree[i, j] for i, j in combinations(range(n), 2) if lab[i] != lab[j]]
    c1, c2 = loss[lab == 1], loss[lab == 2]
    modes = {"sizes": sorted([int((lab == 1).sum()), int((lab == 2).sum())]), "within": float(np.mean(within)) if within else None,
             "between": float(np.mean(between)) if between else None,
             "pair_agreement_min": float(np.nanmin(agree)), "pair_agreement_median": float(np.nanmedian(agree)),
             "loss_sd": float(loss.std(ddof=1)), "loss_mean": float(loss.mean()),
             "loss_gap_between_clusters_in_sd": float(abs(c1.mean() - c2.mean()) / max(loss.std(ddof=1), 1e-300)) if len(c1) and len(c2) else None,
             "lower_loss_cluster_size": int(min((c1, c2), key=lambda c: c.mean()).size) if len(c1) and len(c2) else None,
             "seeds": [f["job"]["seed"] for f in fits], "cluster": lab.tolist(), "loss": loss.tolist(),
             "agreement_matrix": np.round(agree, 4).tolist()}
    summ = {}
    for labk in pairs[0]:
        summ[labk] = {"A_adj": float(np.mean([p[labk]["A_adj"] for p in pairs])), "B_adj": float(np.mean([p[labk]["B_adj"] for p in pairs])),
                      "A_facet": {f: float(np.mean([p[labk]["A_facet"][f] for p in pairs])) for f in facets},
                      "B_facet": {f: float(np.mean([p[labk]["B_facet"][f] for p in pairs])) for f in facets}}
    return {"study": study, "approach": a, "K": K, "group": g, "n": n, "tracks": summ, "modes": modes, "article_agreement": art.tolist()}


def run(workers):
    tasks = []
    for study, st in STAGES.items():
        fs = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, st, "*", "measures.json"))])
        by = defaultdict(list)
        for f in fs:
            j = f["job"]
            if j["beta"] == 1 and j["K"] in KS and not (j.get("terms") or {}) and not j.get("arm"):
                by[(j["approach"], j["K"], j["group"])].append(f["dir"])
        tasks += [(study, a, K, g, ds) for (a, K, g), ds in sorted(by.items())]
    with Pool(workers, initializer=setup) as pool:
        res = pool.map(task, tasks)
    setup()
    return res


def analyse(res):
    b = D.Build(D.BUILD); TF = b.load("raw_child_article_tf").tocsr(); g = _G["genre"]
    ties = np.diff(TF.indptr); gt = np.asarray((TF > 0).astype(float) @ g.astype(float)).ravel()
    rem = ties - gt; loss_share = gt / np.maximum(ties, 1)
    rep = {"test1": {}, "test2": {}, "test3": {}}
    by = defaultdict(list)
    for r in res:
        by[(r["study"], r["approach"], r["K"])].append(r)
    for a in (1, 2):
        for K in KS:
            A, B = by[("A", a, K)], by[("B", a, K)]
            t1 = {}
            for nm, rs, lab in (("A_all", A, "all"), ("A_substantive_child", A, "substantive_child"), ("B", B, "all")):
                t1[nm] = {"A_adj": float(np.mean([r["tracks"][lab]["A_adj"] for r in rs])),
                          "B_adj": float(np.mean([r["tracks"][lab]["B_adj"] for r in rs])),
                          "child_A_raw": float(np.mean([r["tracks"][lab]["A_facet"]["child"] for r in rs])),
                          "child_B_raw": float(np.mean([r["tracks"][lab]["B_facet"]["child"] for r in rs])),
                          "article_A_raw": float(np.mean([r["tracks"][lab]["A_facet"]["article"] for r in rs]))}
            rep["test1"][f"approach_{a}_K{K}"] = t1
            # test 2
            agA = np.mean([r["article_agreement"] for r in A], 0); agB = np.mean([r["article_agreement"] for r in B], 0)
            lt_all, lt_rem = np.log1p(ties), np.log1p(rem)
            edges = np.quantile(lt_all, np.linspace(0, 1, 11)); edges[-1] += 1e-9
            binA = np.clip(np.searchsorted(edges, lt_all, "right") - 1, 0, 9)
            curve = np.array([agA[binA == k].mean() for k in range(10)]); mid = np.array([lt_all[binA == k].mean() for k in range(10)])
            pred = np.interp(lt_rem, mid, curve)
            resid = agB - pred
            X = np.column_stack([np.ones_like(loss_share), loss_share, lt_rem])
            coef, res_, *_ = np.linalg.lstsq(X, agB - agA, rcond=None)
            sig2 = ((agB - agA - X @ coef) ** 2).sum() / (len(X) - 3); se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
            q = np.quantile(lt_rem, np.linspace(0, 1, 6)); q[-1] += 1e-9; bins = np.clip(np.searchsorted(q, lt_rem, "right") - 1, 0, 4)
            within = []
            for k in range(5):
                m = bins == k; med = np.median(loss_share[m]); hi, lo = m & (loss_share > med), m & (loss_share <= med)
                within.append({"remaining_ties_median": float(np.median(rem[m])), "genre_loss_median": float(med),
                               "drop_high_genre_loss": float((agA - agB)[hi].mean()) if hi.any() else None,
                               "drop_low_genre_loss": float((agA - agB)[lo].mean()) if lo.any() else None,
                               "n": int(m.sum())})
            rep["test2"][f"approach_{a}_K{K}"] = {
                "agreement_A_mean": float(agA.mean()), "agreement_B_mean": float(agB.mean()),
                "residual_B_vs_A_curve_mean": float(resid.mean()),
                "residual_slope_on_genre_loss": float(np.polyfit(loss_share, resid, 1)[0]),
                "drop_regression": {"intercept": float(coef[0]), "genre_loss": float(coef[1]), "genre_loss_se": float(se[1]),
                                    "log_remaining_ties": float(coef[2]), "log_remaining_ties_se": float(se[2])},
                "within_bins_of_remaining_ties": within,
                "genre_loss_quantiles": np.quantile(loss_share, [0.1, 0.5, 0.9]).tolist()}
            # test 3
            rep["test3"][f"approach_{a}_K{K}"] = {s: [{k: r["modes"][k] for k in ("sizes", "within", "between", "pair_agreement_min",
                                                                              "pair_agreement_median", "loss_gap_between_clusters_in_sd",
                                                                              "lower_loss_cluster_size", "loss_sd")} | {"group": r["group"],
                                                   "A_adj": r["tracks"]["all"]["A_adj"]}
                                                  for r in sorted(by[(s, a, K)], key=lambda r: r["group"])] for s in ("A", "B")}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    res = run(a.workers); rep = analyse(res)
    rep["_raw"] = [{k: v for k, v in r.items() if k != "article_agreement"} for r in res]
    json.dump(rep, open(a.out, "w"), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (int(x) if isinstance(x, np.integer) else float(x)))
    for t in ("test1", "test2"):
        for k, v in rep[t].items():
            print(t, k, json.dumps(v, default=float)[:1500])
    for k, v in rep["test3"].items():
        for s in ("A", "B"):
            print("test3", k, s, [(x["group"], round(x["A_adj"], 3), x["sizes"], round(x["within"] or 0, 3), round(x["between"] or 0, 3),
                                   round(x["loss_gap_between_clusters_in_sd"] or 0, 2)) for x in v[s]])


if __name__ == "__main__":
    main()
