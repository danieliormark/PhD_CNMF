# Community-quality readings across the weighting points (owner 2026-11-24: "Yes, please prepare the job"; RL-362).
# Question (owner): does the loss of child×article prediction at semantic pull 0.65 (approach 2) make the communities
# more biased? Saved fits only, study A, K 5, r 0.3, both approaches, hold-out groups 1–5 × seeds 1–10, at W1 (pass 3),
# 0.75, 0.65 (weight_curve) and W2 0.50 (weight_check). Per fit (relation data with the fit's own weights):
#   5–8  pass3_readings.one_fit — within-community share of child×article tie weight (anchor_within_share; continuous T1),
#        crossing ties per relation (share of tie weight between different main communities; co-membership on crossing /
#        within ties and its chance level), article coherence "mosaic vs blend" (floor_diagnostics), closure ratio per
#        relation, flattening of W-tied child hyperedges, off-diagonal shares, domain presence;
#   9    idea coherence of article groupings: over all article pairs, co-membership Σ_k p_k q_k (weighted memberships)
#        against shared ideas — Pearson correlation with the W-aware cosine of their child-hyperedge sets, and the mean
#        co-membership of pairs sharing ≥ 1 child hyperedge minus that of all pairs ("excess");
#   10   people coherence: the same excess for pairs sharing a social entity (swap_link_measure.one at p = 0: approach 1
#        author, organisation via authors, journal; approach 2 co-author tie, organisation, journal), and the Pearson
#        correlation of co-membership with the number of shared authors / co-author tie.
# Summary per point and approach: every numeric reading as point − W1, paired by (group, seed), mean per group, mean ±
# SE over the 5 groups (hold-out-draw noise; a yardstick, not a p-value), relative to W1, and in W1 seed SDs.
# Usage: python -m fit.weight_curve_community --out fit/results/stage0/weight_curve_community.json --workers 8
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits
from .align import fit_rds
from . import pass3_readings as P3
from .swap_link_measure import one as swap_one, incidence, _WN
from .weight_curve_analysis import point_of

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
K, R = 5, 0.3
_C = {}


def idea_people(f, rds):
    ca = rds["child_article"]; n = ca.shape[0]
    X = sp.csr_matrix((np.ones(len(ca.tie_row)), (ca.tie_row, ca.tie_col)), shape=ca.shape)
    Xw = X + X @ _WN; nw = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel())
    Jw = (Xw @ Xw.T).toarray() / np.maximum(np.outer(nw, nw), 1e-300)
    shared = (X @ X.T).toarray()
    U, Z = load_UZ(f["dir"]); P = M.weighted_memberships(U, Z, {k: tuple(v) for k, v in f["result"]["relations"].items()})["article"]["P"]
    C = P @ P.T
    iu = np.triu_indices(n, 1); c, jw, sh = C[iu], Jw[iu], shared[iu]
    out = {"idea_corr_w": float(np.corrcoef(c, jw)[0, 1]), "idea_excess": float(c[sh > 0].mean() - c.mean()),
           "idea_pairs_share": float((sh > 0).mean()), "co_membership_mean": float(c.mean())}
    soc = rds["author_article"] if f["job"]["approach"] == 1 else rds["article_article"]
    E = incidence(soc, n)
    S = (E @ E.T).toarray() if f["job"]["approach"] == 1 else E.toarray() + E.toarray().T
    out["people_corr"] = float(np.corrcoef(c, S[iu])[0, 1])
    return out


def task(d):
    f = load_fit(d); cache = _C.setdefault("rds", {})
    rds = {rd.name: rd for rd in fit_rds(f["job"], cache)}
    j = f["job"]
    row = {"approach": j["approach"], "group": j["group"], "seed": j["seed"], "point": point_of(j)}
    try:
        row["p3"] = P3.one_fit(f, rds)
        row["idea_people"] = idea_people(f, rds)
        row["swap_p0"] = {e: v["model_excess"] for e, v in swap_one(f, cache).items()}
    except Exception as e:                     # one failing fit must not lose the others; reported
        row["error"] = repr(e)
    return row


def init():
    P3.setup(D.BUILD, "A")


def flat(x, pre=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{pre}{k}."))
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and math.isfinite(x):
        out[pre[:-1]] = float(x)
    elif isinstance(x, bool):
        out[pre[:-1]] = float(x)
    return out


SKIP = ("approach", "group", "seed", "K", "r", "epochs", "converged", "p3.mono.flags")


def summarise(rows):
    by = defaultdict(dict)
    for r in rows:
        if "error" in r:
            continue
        v = {k: x for k, x in flat({"p3": r["p3"], "idea_people": r["idea_people"], "swap_p0": r["swap_p0"]}).items()
             if not any(k.startswith(s) or k.endswith("." + s) for s in SKIP)}
        by[(r["approach"], r["point"])][(r["group"], r["seed"])] = v
    rep = {}
    for (a, pt), fits in sorted(by.items()):
        if pt == "W1":
            continue
        ref = by.get((a, "W1"), {})
        common = sorted(set(fits) & set(ref))
        groups = sorted({g for g, s in common})
        keys = sorted(set.intersection(*[set(fits[k]) & set(ref[k]) for k in common])) if common else []
        res = {}
        for kk in keys:
            dg = np.array([np.mean([fits[k][kk] - ref[k][kk] for k in common if k[0] == g]) for g in groups])
            w1 = np.array([np.mean([ref[k][kk] for k in common if k[0] == g]) for g in groups])
            sd = float(np.mean([np.std([ref[k][kk] for k in common if k[0] == g], ddof=1) for g in groups]))
            se = float(dg.std(ddof=1) / math.sqrt(len(dg))) if len(dg) > 1 else None
            res[kk] = {"W1": float(w1.mean()), "difference": float(dg.mean()), "se": se,
                       "relative": float(dg.mean() / w1.mean()) if w1.mean() else None, "W1_seed_sd": sd,
                       "n_se": float(abs(dg.mean()) / se) if se else None, "n_seed_sd": float(abs(dg.mean()) / sd) if sd else None}
        rep[f"approach_{a}_{pt}"] = {"pairs": len(common), "readings": res}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="test: only the first N fits")
    a = ap.parse_args()
    load = lambda st: latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, st, "*", "measures.json"))])
    fs = [f for f in load("stage0_pass3") if f["job"]["K"] == K and float(f["job"]["arm"]["r"]) == R]
    fs += [f for f in load("weight_check") if f["job"]["K"] == K and not f["job"]["arm"].get("swap_p")]
    fs += [f for f in load("weight_curve") if not f["job"]["arm"].get("swap_p")]
    dirs = [f["dir"] for f in fs if f["result"]["converged"]]
    if a.limit:
        dirs = dirs[:a.limit]
    print(len(dirs), "fits", flush=True)
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as pool:
            rows = pool.map(task, dirs, chunksize=4)
    else:
        init(); rows = [task(d) for d in dirs]
    rep = {"summary": summarise(rows), "errors": [r for r in rows if "error" in r], "rows": rows}
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    for name, v in rep["summary"].items():
        print(f"\n{name} ({v['pairs']} paired fits): point − W1, mean ± SE over groups (n SE; n W1 seed SD)")
        for kk, x in v["readings"].items():
            print(f"   {kk:60s} W1 {x['W1']:.4f} diff {x['difference']:+.4f} ± {x['se']:.4f} ({x['n_se'] or 0:.1f} SE; {x['n_seed_sd'] or 0:.1f} SD)")
    print("errors:", len(rep["errors"]))


if __name__ == "__main__":
    main()
