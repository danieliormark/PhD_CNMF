# Diagonal-start test — reading (ticket 107; owner 2026-11-17). Expected outcomes and the reading rule are stated in
# plans/ticket107_start_test.md BEFORE the fits return; this script applies them unchanged.
# Arms (same data, loss, K*, hold-out group 1, seeds 1–10, terms off, same membership start per seed):
#   reference = stage-0 pass-1 fits (hybrid start: semantic cores diagonal, social cores random);
#   diagonal  = start_test fits (hybrid start with every social core started diagonal, init.diagonal_social_links).
# Per fit: convergence; data loss (total, weighted, final epoch) and per-relation loss; held-out two-part score
# (existence, value; decided score) and existence per relation with hidden cells; per social relation the link
# structure (effective number of partner communities — label-free; within share as fitted and after one-to-one
# renumbering of the column facet; main-partner share); presence summaries (mixed share at 0.1, weaker-domain median)
# and detachment per fit (after renumbering single-relation facets). Per arm: stability (aggregate.group_checks:
# gate passes, Track A/B chance-adjusted).
# Paired comparison by seed (both converged): difference diagonal − reference, mean, SE = SD/√n; "better"/"worse" if the
# mean difference is beyond 2 SE in that direction, otherwise "within noise".
# Outcome per social relation (pre-stated): "stays near one partner" if the diagonal arm's median effective number of
# partners is below the midpoint between 1 and the reference median, otherwise "returns to spread". Fit cost: "worse"
# if data loss, existence or value is worse beyond 2 SE (paired). Outcome 1 = stays and not worse; 2 = returns to spread;
# 3 = stays and worse; 4 = relations of one setting differ (1/3 vs 2). KL is the control: "changed" if any paired
# measure (data loss, existence, value, effective partners of any social relation) is beyond 2 SE.
# Usage: python -m fit.start_test_analysis --out results/stage0/start_test.json
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits, group_checks

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
LOSS = {2: "squared_error", 1: "KL"}
SINGLE = {"author_affiliation": "organisation", "article_affiliation": "organisation", "article_journal": "journal"}


def load(stage, diagonal):
    fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, stage, "*", "measures.json"))]
    out = defaultdict(list)
    for f in latest_fits(fs):
        j = f["job"]
        if j["group"] != 0 or j["seed"] not in range(1, 11) or j["K"] != KSTAR[j["beta"]] or (j.get("terms") or {}) \
                or (j.get("arm") or {}) or (j.get("social_link_start") == "diagonal") != diagonal:
            continue
        out[(j["approach"], j["beta"])].append(f)
    return out


def link(U, Z, rd):
    P, T = M.parts(U, Z, rd); P = P / T
    r, c = linear_sum_assignment(-P)
    perm = np.empty(len(c), dtype=int); perm[r] = c
    rw = P.sum(1); Pr = P / np.maximum(rw[:, None], 1e-300)
    ent = -(Pr * np.log(np.clip(Pr, 1e-300, None))).sum(1)
    return {"effective_partners": float((rw * np.exp(ent)).sum() / rw.sum()), "within": float(np.trace(P)),
            "within_renumbered": float(P[r, c].sum()), "main_partner": float(P.max(1).sum())}, P, perm


def per_fit(f, rds):
    m = f["measures"]; r = f["result"]
    out = {"seed": f["job"]["seed"], "converged": bool(r["converged"]), "epochs": r["epochs"],
           "data_loss": float(r["data_loss_history"][-1]), "relation_loss": r["final_relation_loss"],
           "existence": m["heldout_two_part_combined"]["existence"], "value": m["heldout_two_part_combined"]["value"],
           "existence_by_relation": {k: v["ap_within_class"] for k, v in m["heldout_two_part"].items() if v}}
    dp = m["domain_presence"]; S = np.array(dp["semantic"]["mean"]); O = np.array(dp["social"]["mean"])
    share = np.mean([np.array(v["half_credit"]) for v in m["presence"].values()], 0)
    out["mixed_0.1"] = float(share[(S >= 0.1) & (O >= 0.1)].sum())
    out["weaker_median"] = float(np.median(np.minimum(S, O)))
    U, Z = load_UZ(f["dir"])
    U = {k: v.astype(np.float64) for k, v in U.items()}; Z = {k: v.astype(np.float64) for k, v in Z.items()}
    fp = {x: np.array(v) for x, v in m["facet_presence"].items()}
    out["link"] = {}; det = 0
    for name, rd in rds.items():
        if rd.domain != "social":
            continue
        s, P, perm = link(U, Z, rd)
        out["link"][name] = s
        if name in SINGLE:
            K = P.shape[0]; side = K * P.sum(0)
            det += int(((fp[rd.row_facet] >= 0.5) & (side[perm] < 0.1)).sum())
    out["detached_renumbered"] = det
    return out


def paired(ref, arm, get):
    a = {x["seed"]: get(x) for x in arm if x["converged"]}; b = {x["seed"]: get(x) for x in ref if x["converged"]}
    d = np.array([a[s] - b[s] for s in sorted(set(a) & set(b)) if a[s] is not None and b[s] is not None])
    if len(d) < 2:
        return {"n": int(len(d))}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"n": int(len(d)), "mean_difference": float(d.mean()), "se": float(se), "lower_by_2se": bool(d.mean() < -2 * se),
            "higher_by_2se": bool(d.mean() > 2 * se)}


def run():
    ref, arm = load("stage0_pass1", False), load("start_test", True)
    build = D.Build(D.BUILD)
    rep = {}
    for (a, beta) in sorted(arm):
        rds = {rd.name: rd for rd in D.fit_data(build, a, group=0)}
        name = f"approach_{a}_{LOSS[beta]}"
        R = [per_fit(f, rds) for f in sorted(ref[(a, beta)], key=lambda f: f["job"]["seed"])]
        A = [per_fit(f, rds) for f in sorted(arm[(a, beta)], key=lambda f: f["job"]["seed"])]
        st = {}
        for lab, fs in (("reference", ref[(a, beta)]), ("diagonal", arm[(a, beta)])):
            g, _ = group_checks(sorted(fs, key=lambda f: f["job"]["seed"]))
            st[lab] = {"gate_passes": int(sum(x is False for x in g["gate_fails"])), "converged": int(sum(not x for x in g["convergence_fails"])),
                       **({k: g["stability"][k] for k in ("A_adjusted", "B_adjusted")} if "stability" in g else {})}
        cmp = {"data_loss": paired(R, A, lambda x: x["data_loss"]), "existence": paired(R, A, lambda x: x["existence"]),
               "value": paired(R, A, lambda x: x["value"]), "mixed_0.1": paired(R, A, lambda x: x["mixed_0.1"]),
               "weaker_median": paired(R, A, lambda x: x["weaker_median"]),
               "detached_renumbered": paired(R, A, lambda x: x["detached_renumbered"])}
        for r in R[0]["existence_by_relation"]:
            cmp[f"existence_{r}"] = paired(R, A, lambda x, r=r: x["existence_by_relation"].get(r))
        for r in R[0]["relation_loss"]:
            cmp[f"loss_{r}"] = paired(R, A, lambda x, r=r: x["relation_loss"][r])
        links = {}
        for r in R[0]["link"]:
            rv = [x["link"][r] for x in R if x["converged"]]; av = [x["link"][r] for x in A if x["converged"]]
            med = lambda L, k: float(np.median([y[k] for y in L])) if L else None
            ref_p, arm_p = med(rv, "effective_partners"), med(av, "effective_partners")
            links[r] = {"reference": {k: med(rv, k) for k in rv[0]}, "diagonal": {k: med(av, k) for k in av[0]} if av else None,
                        "paired_partners": paired(R, A, lambda x, r=r: x["link"][r]["effective_partners"]),
                        "midpoint": (1 + ref_p) / 2,
                        "reading": None if arm_p is None else ("stays near one partner" if arm_p < (1 + ref_p) / 2 else "returns to spread")}
        worse = any(cmp[k].get(f) for k, f in (("data_loss", "higher_by_2se"), ("existence", "lower_by_2se"), ("value", "lower_by_2se")))
        better = any(cmp[k].get(f) for k, f in (("data_loss", "lower_by_2se"), ("existence", "higher_by_2se"), ("value", "higher_by_2se")))
        if beta == 2:
            per_rel = {r: (2 if v["reading"] == "returns to spread" else (3 if worse else 1)) for r, v in links.items() if v["reading"]}
            kinds = set(per_rel.values())
            outcome = (4 if len(kinds) > 1 else kinds.pop()) if kinds else None
            reading = {"outcome": outcome, "per_relation": per_rel, "fit_worse": worse, "fit_better": better}
        else:
            changed = worse or better or any(v["paired_partners"].get("lower_by_2se") or v["paired_partners"].get("higher_by_2se") for v in links.values())
            reading = {"control_changed": bool(changed), "fit_worse": worse, "fit_better": better}
        rep[name] = {"fits": {"reference": len(R), "diagonal": len(A)}, "stability": st, "paired": cmp, "links": links,
                     "reading": reading, "per_fit": {"reference": R, "diagonal": A}}
        print(name, json.dumps(reading), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
