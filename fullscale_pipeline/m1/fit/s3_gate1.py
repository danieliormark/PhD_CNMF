# S3 gate 1 on the stage-0 pass-1 fits (decided 2026-11-07, step 5; run 2026-11-16, owner: "Please run S3 gate 1").
# Per approach, loss (at its K*: squared error 20, KL 13), facet and visible-tie band; placed entities only (item (l)).
# Excess spread e_i (ρ = 1) = H(p_i) − the largest relation-mix spread over i's relations (measures.excess_spread), in bits.
# (1) Prevalence beyond noise: per hold-out group, share of entities whose mean e over the 10 seeds exceeds their own
#     seed SD, minus the same count between two halves of the seeds (mean_half1 − mean_half2 > SD; mean over 50 random
#     halvings); passes if the mean over groups of that difference exceeds its SE (SD over groups / √groups).
# (2) Harm to prediction: per-entity held-out AUC (hidden ties vs hidden empty cells of the entity's row/column, relations
#     with hidden cells touching the facet, AUC averaged over its relations, then over seeds); per group, within each band,
#     mean AUC of entities with excess minus without, averaged over bands weighted by entities with excess; passes if
#     the mean over groups is below −δ, δ = 95th percentile over 199 halvings of |mean over groups (stat half 1 − half 2)|
#     ÷ √2 (the stage-0 margin rule).
# Gate passes for a loss if (1) and (2) both pass (child hyperedges and articles first; other facets reported).
# Usage: python -m fit.s3_gate1 --approach A --beta B --out OUT.json
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, load_entity, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
FACETS = ("child", "article", "parent", "author", "organisation", "journal")


def entity_heldout_auc(U, Z, rds, facet):
    """Per entity of `facet`: mean over relations with hidden cells of the AUC of its hidden ties vs its hidden empties."""
    n = U[facet].shape[0]
    acc = np.zeros(n); cnt = np.zeros(n)
    for rd in rds:
        if len(rd.hid_tie_row) == 0:
            continue
        for side in ("row", "col"):
            f = rd.row_facet if side == "row" else rd.col_facet
            if f != facet or (side == "col" and rd.row_facet == rd.col_facet):
                continue
            A, B, Zr = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
            vt = ((A[rd.hid_tie_row] @ Zr) * B[rd.hid_tie_col]).sum(1)
            ve = ((A[rd.hid_empty_row] @ Zr) * B[rd.hid_empty_col]).sum(1)
            gt = rd.hid_tie_row if side == "row" else rd.hid_tie_col
            ge = rd.hid_empty_row if side == "row" else rd.hid_empty_col
            auc, npos, nneg = M._grouped_auc(np.concatenate([gt, ge]).astype(np.int64), np.concatenate([vt, ve]),
                                             np.r_[np.ones(len(vt), bool), np.zeros(len(ve), bool)], n)
            ok = (npos > 0) & (nneg > 0)
            acc[ok] += auc[ok]; cnt[ok] += 1
    out = np.full(n, np.nan); out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out


def band_names(ties):
    return np.array([M.band_of(int(t)) for t in ties])


def harm_stat(exc, auc, placed, bands):
    """Mean over bands (weighted by entities with excess) of mean AUC(excess) − mean AUC(no excess); exc: bool."""
    num = 0.0; w = 0.0
    for b in np.unique(bands):
        m = placed & (bands == b) & np.isfinite(auc)
        e, ne = m & exc, m & ~exc
        if e.sum() and ne.sum():
            num += e.sum() * (auc[e].mean() - auc[ne].mean()); w += e.sum()
    return num / w if w else np.nan


def run(approach, beta, seed=0):
    rng = np.random.default_rng(seed)
    build = D.Build(D.BUILD)
    base = os.path.join(ROOT, "stage0_pass1")
    fits = [f for f in latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(base, "*", "measures.json"))])
            if f["job"]["approach"] == approach and f["job"]["beta"] == beta and f["job"]["K"] == KSTAR[beta]
            and not (f["job"].get("terms") or {}) and f["result"]["converged"]]
    byg = defaultdict(list)
    for f in fits:
        byg[f["job"]["group"]].append(f)
    per = {}
    for g in sorted(byg):
        fs = sorted(byg[g], key=lambda f: f["job"]["seed"])
        rds = D.fit_data(build, approach, group=g)
        ents = [load_entity(f["dir"]) for f in fs]
        E, A = defaultdict(list), defaultdict(list)
        for f in fs:
            U, Z = load_UZ(f["dir"])
            wm = M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()})
            for fa in FACETS:
                if fa in wm:
                    E[fa].append(M.excess_spread(rds, wm, fa)); A[fa].append(entity_heldout_auc(U, Z, rds, fa))
        per[g] = {}
        for fa in E:
            Ex = np.vstack(E[fa]); Au = np.vstack(A[fa])
            placed = M.placed_entities([e[fa]["reproduced_share"] for e in ents], ents[0][fa]["reproduced_share_degree_only"])
            placed &= np.isfinite(Ex).all(0)
            per[g][fa] = {"Ex": Ex, "Au": Au, "placed": placed, "bands": band_names(ents[0][fa]["ties"])}
        print("group", g, "done", flush=True)
    groups = sorted(per)
    out = {}
    for fa in per[groups[0]]:
        res = {"bands": {}}
        bandset = ["all"] + sorted(set(np.unique(per[groups[0]][fa]["bands"])) - {"0"})
        for bname in bandset:
            d_prev, shares, refs = [], [], []
            for g in groups:
                p = per[g][fa]; sel = p["placed"] & ((p["bands"] == bname) if bname != "all" else True)
                Ex = p["Ex"][:, sel]
                if Ex.shape[1] == 0:
                    continue
                sd = Ex.std(0, ddof=1); share = float((Ex.mean(0) > sd).mean())
                n = Ex.shape[0]; ref = []
                for _ in range(50):
                    perm = rng.permutation(n); h = n // 2
                    ref.append(float(((Ex[perm[:h]].mean(0) - Ex[perm[h:2 * h]].mean(0)) > sd).mean()))
                shares.append(share); refs.append(float(np.mean(ref))); d_prev.append(share - np.mean(ref))
            if len(d_prev) < 2:
                continue
            d = np.array(d_prev); se = float(d.std(ddof=1) / math.sqrt(len(d)))
            res["bands"][bname] = {"share_with_excess": float(np.mean(shares)), "halves_reference": float(np.mean(refs)),
                                   "difference": float(d.mean()), "se": se, "prevalence_passes": bool(d.mean() > se),
                                   "placed_entities_per_group": float(np.mean([(per[g][fa]["placed"] & ((per[g][fa]["bands"] == bname) if bname != "all" else True)).sum() for g in groups]))}
        # (2) harm
        stats = []
        for g in groups:
            p = per[g][fa]; Ex, Au = p["Ex"], p["Au"]
            exc = Ex.mean(0) > Ex.std(0, ddof=1)
            stats.append(harm_stat(exc, np.nanmean(Au, 0) if np.isfinite(Au).any() else Au[0], p["placed"], p["bands"]))
        stats = np.array(stats, float)
        vals = []
        for _ in range(199):
            diffs = []
            for g in groups:
                p = per[g][fa]; n = p["Ex"].shape[0]; perm = rng.permutation(n); h = n // 2
                s = []
                for idx in (perm[:h], perm[h:2 * h]):
                    Ex = p["Ex"][idx]; Au = p["Au"][idx]
                    with np.errstate(all="ignore"):
                        s.append(harm_stat(Ex.mean(0) > Ex.std(0, ddof=1), np.nanmean(Au, 0), p["placed"], p["bands"]))
                diffs.append(s[0] - s[1])
            diffs = [x for x in diffs if np.isfinite(x)]
            if diffs:
                vals.append(abs(np.mean(diffs)))
        delta = float(np.percentile(vals, 95) / math.sqrt(2)) if vals else None
        fin = stats[np.isfinite(stats)]
        res["harm"] = {"auc_excess_minus_none": float(fin.mean()) if len(fin) else None, "per_group": stats.tolist(),
                       "delta": delta, "harm_passes": bool(len(fin) and delta is not None and fin.mean() < -delta)}
        res["gate_passes"] = bool(res["bands"].get("all", {}).get("prevalence_passes") and res["harm"]["harm_passes"])
        res["mean_excess_bits_placed"] = float(np.mean([np.nanmean(per[g][fa]["Ex"][:, per[g][fa]["placed"]]) for g in groups]))
        out[fa] = res
        print(approach, beta, fa, json.dumps({k: v for k, v in res.items() if k != "bands"}), json.dumps(res["bands"].get("all")), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", type=int, required=True); ap.add_argument("--beta", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(a.approach, a.beta), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
