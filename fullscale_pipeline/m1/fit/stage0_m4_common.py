# Stage-0 seed stability (S0-M4) re-read on the common set (owner approved 2026-11-14; reported beside the decided reading,
# a changed outcome at any K is flagged, never a reversal by itself).
# Decided reading (stage0_compare): Track A over each loss's own placed entities; Track B over all entities weighted by
# fit 1's mass. Here, per hold-out group, the common set = entities placed under BOTH losses (placed rule of 2026-11-07
# applied to each loss's own seeds, then intersected per facet). Track A = 1 − mean JS distance over the common set;
# Track B = mass-weighted cosine over the common set only (mass of entities outside it set to 0). Chance reference, the
# chance adjustment, Δ = KL − squared error, the noise margin and the classes are as in stage0_compare.
# Usage: python -m fit.stage0_m4_common --root ROOT --stage STAGE --aggregate AGG.json --out OUT.json
import argparse, itertools, json, math, os
from collections import defaultdict
import numpy as np

from . import measures as M
from .aggregate import load_fit, load_UZ, latest_fits, cell_key
from .stage0_compare import classify, rms_margin, mean_pairs, placed_of, N_SPLITS


def track_pair_set(wm1, wm2, keep):
    """Track A and B per facet restricted to the entity set keep[f]; model level = mean over facets."""
    facets = sorted(wm1)
    perm = M.match(wm1, wm2, facets)
    A, B = {}, {}
    for f in facets:
        m = keep[f]
        if not m.any():
            A[f] = B[f] = np.nan; continue
        P1, P2 = wm1[f]["P"][m], wm2[f]["P"][:, perm][m]
        A[f] = float(1 - M._js_distance_e(P1, P2).mean())
        Y1, Y2 = wm1[f]["Y"][m], wm2[f]["Y"][:, perm][m]
        cos = (Y1 * Y2).sum(1) / np.maximum(np.linalg.norm(Y1, axis=1) * np.linalg.norm(Y2, axis=1), 1e-300)
        w = wm1[f]["mass"][m]
        B[f] = float((w * cos).sum() / w.sum()) if w.sum() > 0 else np.nan
    return {"A_model": float(np.nanmean(list(A.values()))), "B_model": float(np.nanmean(list(B.values())))}


def pair_scores_set(wms, keep, rng):
    out = {}
    for i, j in itertools.combinations(range(len(wms)), 2):
        real = track_pair_set(wms[i], wms[j], keep)
        ch = track_pair_set(wms[i], M._shuffled(wms[j], rng), keep)
        out[(i, j)] = {t: (real[t + "_model"] - ch[t + "_model"]) / (1 - ch[t + "_model"]) for t in ("A", "B")}
    return out


def m4_common(byloss, rng):
    groups = sorted(set(byloss[2]) & set(byloss[1]))
    m4 = {t: [] for t in ("A", "B")}; own_all = {}; sizes = {}
    for g in groups:
        if len(byloss[2][g]) < 2 or len(byloss[1][g]) < 2:
            continue
        pl = {b: placed_of(byloss[b][g]) for b in (2, 1)}
        keep = {f: pl[2][f] & pl[1][f] for f in pl[2]}
        sizes[g] = {f: {"common": int(keep[f].sum()), "squared error": int(pl[2][f].sum()), "KL": int(pl[1][f].sum()),
                        "entities": int(len(keep[f]))} for f in keep}
        own = {}
        for b in (2, 1):
            wms = [M.weighted_memberships(*load_UZ(f["dir"]), {r: tuple(v) for r, v in f["result"]["relations"].items()})
                   for f in byloss[b][g]]
            own[b] = pair_scores_set(wms, keep, rng)
        for t in ("A", "B"):
            m4[t].append(mean_pairs(own[1], list(range(len(byloss[1][g]))), track=t)
                         - mean_pairs(own[2], list(range(len(byloss[2][g]))), track=t))
        own_all[g] = own
    out = {}
    for t in ("A", "B"):
        per_loss = []
        for b in (2, 1):
            vals = []
            for _ in range(N_SPLITS):
                diffs = []
                for g, own in own_all.items():
                    n = len(byloss[b][g])
                    if n < 4:
                        continue
                    p = rng.permutation(n); h = n // 2
                    diffs.append(mean_pairs(own[b], list(p[:h]), track=t) - mean_pairs(own[b], list(p[h:2 * h]), track=t))
                if diffs:
                    vals.append(abs(np.mean(diffs)))
            per_loss.append(float(np.percentile(vals, 95) / math.sqrt(2)) if vals else None)
        out[t] = classify(m4[t], rms_margin(per_loss), first="KL more stable", second="squared error more stable")
        out[t]["per_loss_mean"] = {name: float(np.mean([mean_pairs(own[b], list(range(len(byloss[b][g]))), track=t)
                                                        for g, own in own_all.items()]))
                                   for b, name in ((2, "squared error"), (1, "KL"))}
    cl = (out["A"]["class"], out["B"]["class"])
    if "KL more stable better" in cl and "squared error more stable better" not in cl:
        out["verdict"] = "KL more stable"
    elif "squared error more stable better" in cl and "KL more stable better" not in cl:
        out["verdict"] = "squared error more stable"
    elif all(c == "equivalent" for c in cl):
        out["verdict"] = "equivalent"
    else:
        out["verdict"] = "inconclusive"
    out["set_sizes"] = sizes
    return out


def run(root, stage, Ks_by_approach, seed=0):
    rng = np.random.default_rng(seed)
    base = os.path.join(root, stage)
    fits = latest_fits([load_fit(os.path.join(base, d)) for d in sorted(os.listdir(base))
                        if os.path.exists(os.path.join(base, d, "measures.json"))])
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for f in fits:
        j = f["job"]
        if f["result"]["converged"] and not (j.get("terms") or {}):
            by[j["approach"]][j["K"]][j["beta"]][j["group"]].append(f)
    rep = {}
    for a, Ks in Ks_by_approach.items():
        rep[f"approach_{a}"] = {}
        for K in Ks:
            rep[f"approach_{a}"][K] = m4_common(by[a][K], rng)
            r = rep[f"approach_{a}"][K]
            print(a, K, r["verdict"], {t: (round(r[t]["mean"], 3), r[t]["class"], {k: round(v, 3) for k, v in r[t]["per_loss_mean"].items()}) for t in ("A", "B")}, flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--stage", required=True)
    ap.add_argument("--compare", required=True, help="stage0_compare output (candidate K per approach)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cmp_ = json.load(open(a.compare))
    Ks = {int(k.split("_")[1]): [int(K) for K in v["per_K"]] for k, v in cmp_.items()}
    rep = run(a.root, a.stage, Ks)
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


if __name__ == "__main__":
    main()
