# Item (l): sufficient members, measured on stage-0 fits (owner 2026-11-24, RL-371/372; design doc 2026-11-24).
# Decided: weighted membership (rows sum to 1 over K); tiers — core members: membership in k ≥ 0.5; members: ≥ 0.3 (the
# tier "sufficient members" is judged on); tier values checked on data (counts also at 0.2 and 0.4); members counted
# among covered entities (coverage ≥ 0.5, group level). Domains: semantic = child + parent; social = author +
# organisation + journal, reported both without and with articles (owner). Facet requirement (owner): child hyperedges and
# articles must also have sufficient members. Minimum number (owner's rule, robust form; owner 2026-11-24 "use 2.5 as
# main"): a community lacks sufficient members in a domain / facet if it has none, or if log(1 + n) < median − k × 1.4826
# × MAD of log(1 + n) over the communities of the same domain / facet and cell (study, approach, K, r), pooled over hold-out
# groups and seeds (Leys et al. 2013; Rousseeuw & Croux 1993); k = 2.5 main, 2 sensitivity; the verdict counts if it holds
# in the majority of a community's seeds (aligned numbering, principle 3 of 2026-11-06).
# Reported per cell: share of communities lacking sufficient members per domain / facet and in any required one (both
# article treatments, both k, the four q); the robust threshold; diffuse communities (relative presence in a domain ≥ 0.5
# but insufficient members there) and narrow ones (sufficient members but relative presence < 0.1); mean ± SE over groups.
# Saved fits only: the pass-3 candidate cells (W1) of studies A and B.
# Reported reading added 2026-11-24 (owner: "Yes, please add it as a reported reading"; RL-380): membership-weighted tie
# mass — per community and domain / facet, the share of the observed tie weight carried by its members,
# Σ_i p_ik w_i ÷ Σ_i w_i (w_i = the entity's visible tie weight, Σ over its relations of w_r × tie value after Frobenius
# scaling; covered entities), × K as a relative share (1 = fair share, as relative presence); compared with relative
# presence (reconstructed tie weight, view C) per community: Spearman correlation and mean absolute gap. No threshold.
# Usage: python -m fit.sufficient_members --out fit/results/stage0/sufficient_members.json --workers 8
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from . import measures as M
from .aggregate import load_fit, load_UZ, latest_fits, _rds_of, COVERAGE_C

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
STAGES = {"A": "stage0_pass3", "B": "B_stage0_pass3"}
CHOICE = "fit/results/stage0/pass3_choice_{}.json"
QS = (0.2, 0.3, 0.4, 0.5)
Q_MEMBER, Q_CORE = 0.3, 0.5
KS = (2.5, 2.0)
SEM, SOC = ("child", "parent"), ("author", "organisation", "journal")


def one(d):
    f = load_fit(d); j = f["job"]; U, Z = load_UZ(d)
    rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
    rds = _rds_of(j)
    wm = M.weighted_memberships(U, Z, rel); cov = M.entity_coverage(U, Z, rds)
    tw = {f: np.zeros(len(d["P"])) for f, d in wm.items()}
    for rd in rds:
        tw[rd.row_facet] += rd.weight * np.bincount(rd.tie_row, weights=rd.tie_val, minlength=len(tw[rd.row_facet]))
        if rd.col_facet != rd.row_facet:
            tw[rd.col_facet] += rd.weight * np.bincount(rd.tie_col, weights=rd.tie_val, minlength=len(tw[rd.col_facet]))
    counts = {}
    for fa, dd in wm.items():
        c = cov[fa]; keep = np.isfinite(c) & (c >= COVERAGE_C)
        P = dd["P"][keep]
        counts[fa] = {str(q): (P >= q).sum(0).tolist() for q in QS}
        counts[fa]["mass"] = (P * tw[fa][keep][:, None]).sum(0).tolist(); counts[fa]["mass_total"] = float(tw[fa][keep].sum())
    dp = f["measures"]["domain_presence"]
    return {"study": "B" if os.path.basename(os.path.dirname(d)).startswith("B_") else "A", "approach": j["approach"],
            "K": j["K"], "r": float(j["arm"]["r"]), "group": j["group"], "seed": j["seed"], "counts": counts,
            "presence": {"semantic": dp["semantic"]["mean"], "social": dp["social"]["mean"]}}


def units(row, q, with_articles):
    """Member counts per community for each domain and required facet."""
    c = row["counts"]; K = len(next(iter(c.values()))[str(q)])
    add = lambda fs: np.sum([np.array(c[x][str(q)]) for x in fs if x in c], axis=0) if any(x in c for x in fs) else np.zeros(K)
    soc = SOC + (("article",) if with_articles else ())
    return {"semantic": add(SEM), "social": add(soc), "child": add(("child",)), "article": add(("article",))}


def analyse(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["study"], r["approach"], r["K"], r["r"])].append(r)
    out = {}
    for cell, rs in sorted(cells.items()):
        name = f"{cell[0]}_approach_{cell[1]}_K{cell[2]}_r{cell[3]:g}"
        res = {"fits": len(rs)}
        for wa in (False, True):
            for q in QS:
                U = [units(r, q, wa) for r in rs]
                for k in KS:
                    key = f"articles_{'social' if wa else 'uncounted'}|q{q}|k{k}"
                    thr, fail = {}, {}
                    for unit in ("semantic", "social", "child", "article"):
                        x = np.log1p(np.concatenate([u[unit] for u in U]))
                        med = float(np.median(x)); mad = 1.4826 * float(np.median(np.abs(x - med)))
                        cut = med - k * mad
                        thr[unit] = {"median_count": float(np.expm1(med)), "threshold_count": float(max(np.expm1(cut), 0.0))}
                        fail[unit] = [(u[unit] == 0) | (np.log1p(u[unit]) < cut) for u in U]
                    # majority of seeds per (group, community), aligned numbering
                    byg = defaultdict(list)
                    for r, i in zip(rs, range(len(rs))):
                        byg[r["group"]].append(i)
                    share = defaultdict(list); diffuse = defaultdict(list); narrow = defaultdict(list)
                    for g, idx in byg.items():
                        for unit in ("semantic", "social", "child", "article"):
                            F = np.mean([fail[unit][i] for i in idx], 0) > 0.5
                            share[unit].append(float(F.mean()))
                        anyF = np.mean([fail["semantic"][i] | fail["social"][i] | fail["child"][i] | fail["article"][i] for i in idx], 0) > 0.5
                        share["any_required"].append(float(anyF.mean()))
                        for dom in ("semantic", "social"):
                            Pk = np.mean([np.array(rs[i]["presence"][dom]) for i in idx], 0)
                            F = np.mean([fail[dom][i] for i in idx], 0) > 0.5
                            diffuse[dom].append(float((F & (Pk >= 0.5)).mean())); narrow[dom].append(float((~F & (Pk < 0.1)).mean()))
                    se = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
                    res[key] = {"threshold": thr,
                                "lacking_share": {u: {"mean": float(np.mean(v)), "se": se(v)} for u, v in share.items()},
                                "diffuse_share": {u: float(np.mean(v)) for u, v in diffuse.items()},
                                "narrow_share": {u: float(np.mean(v)) for u, v in narrow.items()}}
        out[name] = res
    return out


def member_mass(rows):
    """Relative membership-weighted tie mass per community (× K) per domain / facet, against relative presence."""
    from scipy.stats import spearmanr
    cells = defaultdict(list)
    for r in rows:
        cells[(r["study"], r["approach"], r["K"], r["r"])].append(r)
    out = {}
    for cell, rs in sorted(cells.items()):
        res = {}
        for unit, fs in (("semantic", SEM), ("social", SOC), ("social_with_articles", SOC + ("article",)), ("child", ("child",)),
                         ("article", ("article",))):
            rel, pres_r, pres_g = [], [], []
            for r in rs:
                c = r["counts"]; K = r["K"]
                m = np.sum([np.array(c[f]["mass"]) for f in fs if f in c], axis=0)
                t = sum(c[f]["mass_total"] for f in fs if f in c)
                if t <= 0:
                    continue
                x = m / t * K; rel.append(x)
                dom = "semantic" if unit in ("semantic", "child") else "social"
                p = np.array(r["presence"][dom]); pres_r.append(p)
                pres_g.append((r["group"], x, p))
            if not rel:
                continue
            X = np.concatenate(rel); Pp = np.concatenate(pres_r)
            byg = defaultdict(list)
            for g, x, p in pres_g:
                byg[g].append((float(spearmanr(x, p)[0]) if len(x) > 2 else np.nan, float(np.mean(np.abs(x - p)))))
            rho = [np.nanmean([v[0] for v in vs]) for vs in byg.values()]; gap = [np.mean([v[1] for v in vs]) for vs in byg.values()]
            se = lambda v: float(np.nanstd(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
            res[unit] = {"relative_quantiles": [float(q) for q in np.quantile(X, (0.0, 0.1, 0.5, 0.9, 1.0))],
                         "share_below_0.5": float((X < 0.5).mean()), "share_below_0.1": float((X < 0.1).mean()),
                         "spearman_with_presence": {"mean": float(np.nanmean(rho)), "se": se(rho)},
                         "mean_abs_gap_to_presence": {"mean": float(np.mean(gap)), "se": se(gap)}}
        out[f"{cell[0]}_approach_{cell[1]}_K{cell[2]}_r{cell[3]:g}"] = res
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    dirs = []
    for study, st in STAGES.items():
        ch = json.load(open(CHOICE.format(study)))
        want = {(int(x.split("_")[1]), c["K"], float(c["r"])) for x in ("approach_1", "approach_2") for c in ch[x]["chosen"]}
        for f in latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, st, "*", "measures.json"))]):
            j = f["job"]
            if (j["approach"], j["K"], float(j["arm"]["r"])) in want and f["result"]["converged"]:
                dirs.append(f["dir"])
    print(len(dirs), "fits", flush=True)
    if a.workers > 1:
        with Pool(a.workers) as pool:
            rows = pool.map(one, dirs, chunksize=4)
    else:
        rows = [one(d) for d in dirs]
    rep = {"cells": analyse(rows), "member_mass": member_mass(rows), "rows": rows}
    json.dump(rep, open(a.out, "w"), default=float)
    for name, res in rep["cells"].items():
        m = res["articles_uncounted|q0.3|k2.5"]
        print(name, {u: round(v["mean"], 3) for u, v in m["lacking_share"].items()},
              "thr", {u: round(t["threshold_count"], 1) for u, t in m["threshold"].items()})
    print("done")


if __name__ == "__main__":
    main()
