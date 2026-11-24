# Intermediate weighting check (owner 2026-11-23: "I approve option (a), prepare the intermediate weighting check";
# RL-353): the trade-off curve between the two domains' share of the KL pull. Points per approach (semantic share of the
# pull): W2 0.50 (stage weight_check, mass weights), 0.65 and 0.75 (stage weight_curve), W1 current (0.827 approach 1,
# 0.896 approach 2; stage0_pass3). All at K 5, r 0.3, hold-out groups 1–5 × seeds 1–10. Each fit is paired with its W1 twin
# (same approach, hold-out group, seed); differences are averaged per group, then mean ± SE over the 5 groups (SE =
# hold-out-draw noise, a yardstick, not a p-value); the seed SD of the W1 fits is reported as the second yardstick.
# Readings, per point:
#   held-out, per relation with hidden ties: within-class AP (existence), its degree-only reference and the gain over it;
#     partial Spearman (value) for child×article and child×parent; combined existence with the MAIN setting's relation
#     weights for every point (RL-352: a fit's own weights must not be used across weightings); semantic-equal existence
#     and value (decided cross-approach score); the gain of each point as a share of W1's gain (semantic) and of W2's gain
#     (affiliation relation);
#   seed stability: chance-adjusted Track A and Track B per hold-out group (aggregations, coverage rule);
#   T1 failures, lower domain presence (median over communities), membership spread of articles / authors /
#     organisations (weight_check_analysis.per_fit);
#   swap (hold-out group 1): model co-membership of articles sharing a social entity above the all-pair mean at p = 0
#     (the point's main fits) and p = 1 (stage weight_curve), paired by seed (swap_link_measure.one).
# Usage: python -m fit.weight_curve_analysis --out fit/results/stage0/weight_curve.json > fit/results/stage0/weight_curve.log
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
from collections import defaultdict
import numpy as np
from . import data as D
from .aggregate import load_fit, latest_fits
from .weight_check_analysis import per_fit
from .swap_link_measure import one as swap_one

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
K, R = 5, 0.3
SEM = ("child_article", "parent_child")


def point_of(job):
    arm = job.get("arm") or {}
    if arm.get("semantic_pull") is not None:
        return f"{float(arm['semantic_pull']):.2f}"
    return "W2_0.50" if arm.get("mass_weights") else "W1"


def heldout(f, w_main):
    tp = f["measures"]["heldout_two_part"]; out = {}
    rels = [r for r, v in tp.items() if v]
    for r in rels:
        out[f"AP_{r}"] = tp[r]["ap_within_class"]; out[f"ref_{r}"] = tp[r]["baseline_weights_ap_within_class"]
        out[f"gain_{r}"] = tp[r]["ap_within_class"] - tp[r]["baseline_weights_ap_within_class"]
        if tp[r].get("value_partial_spearman") is not None:
            out[f"value_{r}"] = tp[r]["value_partial_spearman"]
    out["existence_main_weights"] = sum(w_main[r] * out[f"AP_{r}"] for r in rels) / sum(w_main[r] for r in rels)
    out["existence_semantic_equal"] = float(np.mean([out[f"AP_{r}"] for r in SEM]))
    out["gain_semantic_equal"] = float(np.mean([out[f"gain_{r}"] for r in SEM]))
    out["value_semantic_equal"] = float(np.mean([out[f"value_{r}"] for r in SEM]))
    return out


def structure(f, rd_ca):
    """weight_check_analysis.per_fit without its held-out scores (those use the fit's own relation weights, RL-352)."""
    return {k: v for k, v in per_fit(f, rd_ca).items() if not k.startswith("loss_share") and k not in ("converged", "existence", "value")}


def stab(path, want):
    """{(approach, point): {group: (A_adjusted, B_adjusted)}} for the K 5, r 0.3 cells of one aggregation."""
    out = {}
    if not os.path.exists(path):
        return out, None
    agg = json.load(open(path))
    rule = (agg.get("_status") or {}).get("entity_rule")
    for k, v in agg.items():
        if k.startswith("_") or v["cell"]["arm"].get("swap_p") or float(v["cell"]["arm"].get("r", -1)) != R:
            continue
        pt = point_of(v["cell"])
        if pt not in want:
            continue
        e = v["per_K"].get(str(K))
        if e:
            out[(v["cell"]["approach"], pt)] = {g["group"]: (g["stability"]["A_adjusted"], g["stability"]["B_adjusted"])
                                                 for g in e["groups"] if g.get("stability")}
    return out, rule


def run(root):
    load = lambda st: latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(root, st, "*", "measures.json"))])
    fits = [f for f in load("stage0_pass3") if f["job"]["K"] == K and float(f["job"]["arm"]["r"]) == R]
    fits += [f for f in load("weight_check") if f["job"]["K"] == K and not f["job"]["arm"].get("swap_p")]
    curve = load("weight_curve")
    fits += [f for f in curve if not f["job"]["arm"].get("swap_p")]
    fits = [f for f in fits if f["result"]["converged"]]
    by = {(f["job"]["approach"], point_of(f["job"]), f["job"]["group"], f["job"]["seed"]): f for f in fits}
    build = D.Build(D.BUILD)
    rep = {"_points": {}, "_converged": {}}
    rdc = {}
    for a in (1, 2):
        w_main = {rd.name: rd.weight for rd in D.fit_data(build, a, group=0)}
        for kw, pt in (({}, "W1"), ({"mass_weights": True}, "W2_0.50"), ({"mass_weights": True, "semantic_pull": 0.65}, "0.65"),
                       ({"mass_weights": True, "semantic_pull": 0.75}, "0.75")):
            rds = D.fit_data(build, a, group=0, **kw); pull = {rd.name: rd.weight * float(rd.tie_val.sum()) for rd in rds}
            rep["_points"][f"approach_{a}_{pt}"] = sum(v for rd in rds for n, v in pull.items() if n == rd.name and rd.domain == "semantic") / sum(pull.values())
        points = sorted({p for (aa, p, g, s) in by if aa == a}, key=lambda p: rep["_points"][f"approach_{a}_{p}"])
        for pt in points:
            rows = defaultdict(list)
            for (aa, p, g, s), f in by.items():
                if aa != a or p != pt or (a, "W1", g, s) not in by:
                    continue
                t = by[(a, "W1", g, s)]
                if (a, g) not in rdc:
                    rdc[(a, g)] = next(rd for rd in D.fit_data(build, a, group=g) if rd.name == "child_article")
                x = {**heldout(f, w_main), **structure(f, rdc[(a, g)])}
                y = {**heldout(t, w_main), **structure(t, rdc[(a, g)])}
                rows[g].append((x, y))
            rep["_converged"][f"approach_{a}_{pt}"] = sum(len(v) for v in rows.values())
            res = {}
            for kk in rows[next(iter(rows))][0][0]:
                gm = np.array([np.mean([float(x[kk]) for x, y in v]) for v in rows.values()])
                gc = np.array([np.mean([float(y[kk]) for x, y in v]) for v in rows.values()])
                sd = float(np.mean([np.std([float(y[kk]) for x, y in v], ddof=1) for v in rows.values()]))
                d = gm - gc
                res[kk] = {"point": float(gm.mean()), "W1": float(gc.mean()), "difference": float(d.mean()),
                           "se": float(d.std(ddof=1) / math.sqrt(len(d))), "W1_seed_sd": sd, "groups": len(d)}
            rep[f"approach_{a}_{pt}"] = res
    # R-b (owner 2026-11-23, decided before the results; RL-354): per hold-out group, semantic share = point's
    # semantic-equal existence gain over the degree-only reference ÷ W1's; affiliation share = point's affiliation gain over
    # its reference ÷ W2's (the decided wording; corrected 2026-11-24, RL-361 — the first version divided the increments
    # over W1); criterion = min of the two; mean ± SE over groups, and paired differences between points.
    for a in (1, 2):
        aff = "author_affiliation" if a == 1 else "article_affiliation"
        per = defaultdict(lambda: defaultdict(list))
        for (aa, p, g, s), f in by.items():
            if aa != a:
                continue
            tp = f["measures"]["heldout_two_part"]
            gs = float(np.mean([tp[r]["ap_within_class"] - tp[r]["baseline_weights_ap_within_class"] for r in SEM]))
            ga = tp[aff]["ap_within_class"] - tp[aff]["baseline_weights_ap_within_class"]
            per[p][g].append((gs, ga))
        if "W1" not in per or "W2_0.50" not in per:
            continue
        m = {p: {g: np.mean(v, 0) for g, v in d.items()} for p, d in per.items()}
        groups = sorted(m["W1"]); crit = {}
        se = lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))
        for p in m:
            sem = np.array([m[p][g][0] / m["W1"][g][0] for g in groups]); af = np.array([m[p][g][1] / m["W2_0.50"][g][1] for g in groups])
            crit[p] = np.minimum(sem, af)
            rep[f"approach_{a}_{p}"]["_R_b"] = {"semantic_share": float(sem.mean()), "semantic_share_se": se(sem),
                                               "affiliation_share": float(af.mean()), "affiliation_share_se": se(af),
                                               "criterion": float(crit[p].mean()), "criterion_se": se(crit[p])}
        rep[f"_R_b_pairs_approach_{a}"] = {f"{p}-{q}": {"difference": float((crit[p] - crit[q]).mean()), "se": se(crit[p] - crit[q])}
                                          for p in crit for q in crit if p < q}
    # stability
    S, rules = {}, {}
    for st, want in (("stage0_pass3", {"W1"}), ("weight_check", {"W2_0.50"}), ("weight_curve", {"0.65", "0.75"})):
        s_, rules[st] = stab(os.path.join(root, st, "_aggregate_aligned.json"), want); S.update(s_)
    rep["_entity_rule"] = rules
    if len({json.dumps(v, sort_keys=True) for v in rules.values() if v is not None}) == 1:   # one entity rule only
        for (a, pt), gs in S.items():
            ref = S.get((a, "W1"))
            if not ref or pt == "W1":
                continue
            common = sorted(set(gs) & set(ref))
            for i, nm in enumerate(("stability_A_adjusted", "stability_B_adjusted")):
                d = np.array([gs[g][i] - ref[g][i] for g in common])
                rep[f"approach_{a}_{pt}"][nm] = {"point": float(np.mean([gs[g][i] for g in common])), "W1": float(np.mean([ref[g][i] for g in common])),
                                                 "difference": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "groups": len(d)}
    # swap (hold-out group 1)
    sw = defaultdict(dict); cache = {}
    for f in curve:
        j = f["job"]
        if j["arm"].get("swap_p") and f["result"]["converged"]:
            sw[(j["approach"], point_of(j), 1.0)][j["seed"]] = swap_one(f, cache)
    for (a, pt, g, s), f in by.items():
        if g == 0:
            sw[(a, pt, 0.0)][s] = swap_one(f, cache)
    rep["swap"] = {}
    for (a, pt, p), d1 in list(sw.items()):
        if p != 1.0:
            continue
        d0 = sw.get((a, pt, 0.0), {}); seeds = sorted(set(d0) & set(d1))
        if not seeds:
            continue
        out = {}
        for e in d0[seeds[0]]:
            x0 = np.array([d0[s][e]["model_excess"] for s in seeds]); x1 = np.array([d1[s][e]["model_excess"] for s in seeds]); d = x1 - x0
            out[e] = {"p0": float(x0.mean()), "p1": float(x1.mean()), "change": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
                      "relative": float(d.mean() / x0.mean()) if x0.mean() else None, "data_p0": float(np.mean([d0[s][e]["data_excess"] for s in seeds])),
                      "seeds": len(seeds)}
        rep["swap"][f"approach_{a}_{pt}"] = out
    # print
    for a in (1, 2):
        for pt in sorted([k.split(f"approach_{a}_")[1] for k in rep if k.startswith(f"approach_{a}_") and not k.startswith("_")],
                         key=lambda p: rep["_points"][f"approach_{a}_{p}"]):
            res = rep[f"approach_{a}_{pt}"]
            print(f"\napproach {a}  point {pt}  semantic share of pull {rep['_points'][f'approach_{a}_{pt}']:.3f}  "
                  f"pairs {rep['_converged'].get(f'approach_{a}_{pt}')}  (point, W1, diff ± SE over groups, W1 seed SD)")
            for kk, v in res.items():
                if kk == "_R_b":
                    print("   R-b", {k: round(x, 3) for k, x in v.items()})
                else:
                    print(f"   {kk:36s} {v['point']:.4f} {v['W1']:.4f} {v['difference']:+.4f} ± {v['se']:.4f}" + (f"  sd {v['W1_seed_sd']:.4f}" if "W1_seed_sd" in v else ""))
    for k in sorted(k for k in rep if k.startswith("_R_b_pairs")):
        print(k, {p: f"{v['difference']:+.3f}±{v['se']:.3f}" for p, v in rep[k].items()})
    for name, out in sorted(rep["swap"].items()):
        print("swap", name, {e: f"{v['p0']:.4f}→{v['p1']:.4f} ({v['change']:+.4f}±{v['se']:.4f})" for e, v in out.items()})
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    json.dump(run(a.root), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
