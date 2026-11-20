# Start test 2 — reading (ticket 107; owner 2026-11-19: links between different communities allowed (option ii);
# "I agree with S1–S3; test 3 levels, 5% 10 and 20%"). Plan, questions and expected outcomes:
# plans/ticket107_start_test2.md (stated before the fits are read). Aligned numbering throughout — run
# `python -m fit.align --stages start_test2` first.
# Arms (KL, K 13, hold-out group 1, seeds 1–10, terms off, same membership start per seed):
#   random   — pass-1 fits (social cores random, semantic cores diagonal);
#   diagonal — start-test fits (every social core diagonal);
#   S1       — diagonal, article×article random (approach 2);
#   S2_<lv>  — every social core with off-diagonal cells at lv × the diagonal (lv 0.05, 0.1, 0.2);
#   S3_<lv>  — as S2, semantic cores too.
# Per fit: start_test_analysis.per_fit (data loss, per-relation loss, held-out existence and value, diagonal rule,
# child×article requirement, partners); closure_check.one (closure ratio per relation); blurring_check.one (crossing
# share, co-membership on crossing ties, spread of the entities at the ends of crossing ties); per relation the
# off-diagonal share of reconstructed tie weight (Σ_{k≠l} part(k,l) ÷ T) and the share of off-diagonal core cells at the
# positivity floor. Per arm: stability (aggregate.group_checks: chance-adjusted Track A / B, gate). Paired by seed
# against the diagonal arm and against the random arm (difference, SE = SD/√n; beyond 2 SE = "higher"/"lower").
# Usage: python -m fit.start_test2_analysis --out fit/results/stage0/start_test2.json [--root R --stage S] (dry run)
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from .start_test_analysis import per_fit as st_per_fit, paired, load as st_load
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits, group_checks
from .closure_check import one as closure_one
from .blurring_check import one as blurring_one

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
FLOOR = 1e-7


def label(j):
    if j.get("random_links"):
        return "S1"
    lv = j.get("offdiag_level")
    if lv:
        return f"{'S3' if j.get('offdiag_semantic') else 'S2'}_{lv:g}"
    return None


def collect(root, stage):
    out = defaultdict(list)
    fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(root, stage, "*", "measures.json"))]
    for f in latest_fits(fs):
        j = f["job"]
        if j["group"] == 0 and j["beta"] == 1 and j["K"] == 13 and label(j):
            out[(j["approach"], label(j))].append(f)
    ref, dia = st_load("stage0_pass1", False), st_load("start_test", True)
    for a in (1, 2):
        out[(a, "random")] = ref[(a, 1)]; out[(a, "diagonal")] = dia[(a, 1)]
    return out


def offdiag(f, rds):
    U, Z = load_UZ(f["dir"]); out = {}
    for name, rd in rds.items():
        P, T = M.parts(U, Z, rd, floor_zero=None); K = P.shape[0]; off = ~np.eye(K, dtype=bool)
        out[name] = {"offdiag_share": float(P[off].sum() / T), "offdiag_cells_at_floor": float((Z[name][off] <= FLOOR * 1.0001).mean())}
    return out


def one(f, rds):
    x, _ = st_per_fit(f, rds)
    x["closure"] = closure_one(f, rds); x["blurring"] = blurring_one(f, rds); x["offdiag"] = offdiag(f, rds)
    return x


def readings(x, rds, weights):
    """Flat dict of the numbers compared between arms."""
    r = {"data_loss": x["data_loss"], "existence": x["existence"], "value": x["value"],
         "anchor_communities_over_half": x["anchor"]["n_communities_over_half"]}
    for n in rds:
        r[f"loss_{n}"] = x["relation_loss"][n]; r[f"weighted_loss_{n}"] = weights[n] * x["relation_loss"][n]
        r[f"closure_ratio_{n}"] = x["closure"][n]["ratio_between"]
        r[f"offdiag_share_{n}"] = x["offdiag"][n]["offdiag_share"]
        r[f"offdiag_at_floor_{n}"] = x["offdiag"][n]["offdiag_cells_at_floor"]
        b = x["blurring"]["relation"][n]
        r[f"crossing_share_{n}"] = b["crossing_weight_share"]; r[f"co_membership_crossing_{n}"] = b["co_membership_crossing"]
        r[f"end_entropy_crossing_{n}"] = b["row_end_entropy_crossing"]; r[f"end_entropy_all_{n}"] = b["row_end_entropy_all"]
        if n in x["rule"]:
            r[f"rule_share_{n}"] = x["rule"][n]["both_share"]
    for fa, v in x["blurring"]["facet"].items():
        r[f"entropy_{fa}"] = v["entropy_bits"]
    return r


def run(root, stage):
    arms = collect(root, stage); build = D.Build(D.BUILD); rep = {}
    for a in (1, 2):
        rds = {rd.name: rd for rd in D.fit_data(build, a, group=0)}; w = {n: rd.weight for n, rd in rds.items()}
        labs = [l for (aa, l) in arms if aa == a]
        per = {}
        for lab in sorted(labs):
            fs = sorted(arms[(a, lab)], key=lambda f: f["job"]["seed"])
            xs = [one(f, rds) for f in fs]
            g, _ = group_checks(fs) if len(fs) >= 2 else ({}, None)
            per[lab] = {"fits": len(fs), "converged": int(sum(x["converged_actual"] for x in xs)),
                        "stability": {k: g["stability"][k] for k in ("A_adjusted", "B_adjusted", "per_facet_B")} if g.get("stability") else None,
                        "rows": [dict(seed=x["seed"], converged=x["converged_actual"], **readings(x, rds, w)) for x in xs]}
        keys = list(per["diagonal"]["rows"][0]) if per.get("diagonal", {}).get("rows") else []
        keys = [k for k in keys if k not in ("seed", "converged")]
        med = {lab: {k: (float(np.median([r[k] for r in v["rows"] if r["converged"] and r[k] is not None]))
                         if any(r["converged"] and r[k] is not None for r in v["rows"]) else None) for k in keys}
               for lab, v in per.items()}
        cmp = {}
        for lab in per:
            if lab in ("random", "diagonal"):
                continue
            cmp[lab] = {}
            for refl in ("diagonal", "random"):
                R = [dict(r) for r in per[refl]["rows"]]; A = [dict(r) for r in per[lab]["rows"]]
                cmp[lab][f"vs_{refl}"] = {k: paired(R, A, lambda x, k=k: x.get(k)) for k in keys}
        rep[f"approach_{a}"] = {"arms": per, "medians": med, "paired": cmp, "weights": w}
        print(f"approach {a} — medians over converged fits (arm: fits/converged)")
        show = ["data_loss", "existence", "value", "anchor_communities_over_half"] + \
               [f"{p}_{n}" for n in rds for p in ("loss", "closure_ratio", "offdiag_share", "co_membership_crossing", "end_entropy_crossing")]
        order = [l for l in ("random", "diagonal", "S1", "S2_0.05", "S2_0.1", "S2_0.2", "S3_0.05", "S3_0.1", "S3_0.2") if l in per]
        print("  " + " " * 44 + "".join(f"{l:>10s}" for l in order))
        print("  " + f"{'fits/converged':44s}" + "".join(f"{str(per[l]['fits'])+'/'+str(per[l]['converged']):>10s}" for l in order))
        for k in show:
            if k in keys:
                print("  " + f"{k:44s}" + "".join(f"{med[l][k]:10.4g}" if med[l][k] is not None else f"{'—':>10s}" for l in order), flush=True)
        for l in order:
            st = per[l]["stability"]
            if st:
                print(f"  stability {l}: Track A adj {st['A_adjusted']:.3f}  Track B adj {st['B_adjusted']:.3f}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=ROOT); ap.add_argument("--stage", default=D.stage("start_test2"))
    a = ap.parse_args()
    json.dump(run(a.root, a.stage), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
