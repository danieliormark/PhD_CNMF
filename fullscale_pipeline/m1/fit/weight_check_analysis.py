# W3 check (owner 2026-11-22: "I approve W3, prepare the check"; RL-339/340): KL with mass weights (decided 50 / 50
# domain split holds under KL) against the current Frobenius scaling (semantic 83–90 % of the pull). Each mass-weight fit
# is paired with its pass-3 twin (same approach, K, r, hold-out group, seed, start). Differences are mass − current,
# averaged per hold-out group, then mean ± SE over the 5 groups (SE = hold-out-draw noise, a yardstick, not a p-value).
# Readings per fit (aligned numbering): held-out existence and value (decided two-part score); child×article
# requirement T1 (every community's own cell > half of its child×article tie weight on both sides) and the share of
# fits failing it; share of reconstructed tie weight in communities present in both domains (mixed share, P_k ≥ 0.1 in
# both — the decided swap's earlier reading) and the lower domain presence (median over communities); each relation's
# share of the fitted data loss; membership spread of articles and of persons. Per cell: seed stability (chance-adjusted
# Track A, from the aggregations). Swap (K 13, r 1, hold-out group 1): the swap reading (swap_link_measure.one) at p 0 and
# p 1 under each scaling, paired by seed.
# Usage: python -m fit.weight_check_analysis --out fit/results/stage0/weight_check.json [--root R]
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
from collections import defaultdict
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits
from .start_test_analysis import anchor_guard
from .swap_link_measure import one as swap_one

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def bits(P):
    return -(P * np.log2(np.clip(P, 1e-300, None))).sum(1)


def per_fit(f, rd_ca):
    m = f["measures"]; r = f["result"]
    U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in r["relations"].items()}
    P, _ = M.parts(U, Z, rd_ca); a = anchor_guard(P)
    dp = m["domain_presence"]; S = np.array(dp["semantic"]["mean"]); O = np.array(dp["social"]["mean"])
    share = np.mean([np.array(v["half_credit"]) for v in m["presence"].values()], 0)
    rl = r["final_relation_loss"]; tot = sum(rl.values())
    wm = M.weighted_memberships(U, Z, rel)
    out = {"existence": m["heldout_two_part_combined"]["existence"], "value": m["heldout_two_part_combined"]["value"],
           "T1_fails": a["n_communities_over_half"] < a["K"], "mixed_share": float(share[(S >= 0.1) & (O >= 0.1)].sum()),
           "lower_presence_median": float(np.median(np.minimum(S, O))),
           "article_spread": float(bits(wm["article"]["P"]).mean()), "converged": bool(r["converged"]),
           **{f"loss_share_{k}": v / tot for k, v in rl.items()}}
    for fa in ("author", "organisation"):
        if fa in wm:
            out[f"{fa}_spread"] = float(bits(wm[fa]["P"]).mean())
    return out


def key(j):
    return (j["approach"], j["K"], float(j["arm"]["r"]), j["group"], j["seed"])


def run(root):
    build = D.Build(D.BUILD)
    chk = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "weight_check", "*", "measures.json"))])
    p3 = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass3", "*", "measures.json"))])
    twin = {key(f["job"]): f for f in p3}
    main = [f for f in chk if not f["job"]["arm"].get("swap_p")]
    rdc = {}
    pairs = defaultdict(list)
    for f in main:
        k = key(f["job"]); t = twin.get(k)
        if t is None or not (f["result"]["converged"] and t["result"]["converged"]):
            continue
        a, g = k[0], k[3]
        if (a, g) not in rdc:
            rdc[(a, g)] = next(rd for rd in D.fit_data(build, a, group=g) if rd.name == "child_article")
        pairs[(k[0], k[1], k[2])].append((g, per_fit(f, rdc[(a, g)]), per_fit(t, rdc[(a, g)])))
    rep = {}
    for cell, rows in sorted(pairs.items()):
        keys = [x for x in rows[0][1] if x != "converged"]
        res = {"pairs": len(rows)}
        for kk in keys:
            by = defaultdict(list)
            for g, m_, c_ in rows:
                if m_.get(kk) is not None and c_.get(kk) is not None:
                    by[g].append((float(m_[kk]), float(c_[kk])))
            gm = np.array([np.mean([x[0] for x in v]) for v in by.values()]); gc = np.array([np.mean([x[1] for x in v]) for v in by.values()])
            d = gm - gc
            res[kk] = {"mass": float(gm.mean()), "current": float(gc.mean()), "difference": float(d.mean()),
                       "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}
        rep[f"approach_{cell[0]}_K{cell[1]}_r{cell[2]:g}"] = res
    # stability from the aggregations (chance-adjusted Track A, per group)
    rules = {}
    for nm, path in (("mass", os.path.join(root, "weight_check", "_aggregate_aligned.json")),
                     ("current", os.path.join(ROOT, "stage0_pass3", "_aggregate_aligned.json"))):
        if not os.path.exists(path):
            continue
        agg = json.load(open(path))
        rules[nm] = (agg.get("_status") or {}).get("entity_rule") or {"rule": "placed", "c": None}   # item (l), RL-345
        for k, v in agg.items():
            if k.startswith("_") or v["cell"]["arm"].get("swap_p"):
                continue
            if nm == "mass" and not v["cell"]["arm"].get("mass_weights"):
                continue
            a, r = v["cell"]["approach"], float(v["cell"]["arm"]["r"])
            for K, e in v["per_K"].items():
                name = f"approach_{a}_K{K}_r{r:g}"
                if name in rep:
                    gs = {g["group"]: g["stability"]["A_adjusted"] for g in e["groups"] if g.get("stability")}
                    rep[name].setdefault("_stab", {})[nm] = gs
    same_rule = len({json.dumps(v, sort_keys=True) for v in rules.values()}) == 1
    for name, res in rep.items():
        st = res.pop("_stab", {})
        if "mass" in st and "current" in st and same_rule:   # stability compared only under one entity rule
            gs = sorted(set(st["mass"]) & set(st["current"])); d = np.array([st["mass"][g] - st["current"][g] for g in gs])
            res["stability_A_adjusted"] = {"mass": float(np.mean([st["mass"][g] for g in gs])), "current": float(np.mean([st["current"][g] for g in gs])),
                                           "difference": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}
    rep["_entity_rule"] = rules
    # swap (K 13, r 1, group 1)
    sw = defaultdict(dict); cache = {}
    for f in chk:
        j = f["job"]
        if j["K"] != 13 or j["group"] != 0 or not f["result"]["converged"]:
            continue
        arm = j["arm"]; sc = "mass" if arm.get("mass_weights") else "current"
        sw[(j["approach"], sc, float(arm.get("swap_p", 0.0)))][j["seed"]] = swap_one(f, cache)
    for f in p3:
        j = f["job"]
        if j["K"] == 13 and j["group"] == 0 and float(j["arm"]["r"]) == 1.0 and f["result"]["converged"]:
            sw[(j["approach"], "current", 0.0)][j["seed"]] = swap_one(f, cache)
    rep["swap"] = {}
    for a in (1, 2):
        for sc in ("mass", "current"):
            p0, p1 = sw.get((a, sc, 0.0), {}), sw.get((a, sc, 1.0), {})
            seeds = sorted(set(p0) & set(p1))
            if not seeds:
                continue
            ents = list(p0[seeds[0]])
            out = {}
            for e in ents:
                x0 = np.array([p0[s][e]["model_excess"] for s in seeds]); x1 = np.array([p1[s][e]["model_excess"] for s in seeds])
                d = x1 - x0
                out[e] = {"model_p0": float(x0.mean()), "model_p1": float(x1.mean()), "change": float(d.mean()),
                          "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None,
                          "relative": float(d.mean() / x0.mean()) if x0.mean() else None, "seeds": len(seeds)}
            rep["swap"][f"approach_{a}_{sc}"] = out
    for name, res in rep.items():
        if name in ("swap", "_entity_rule"):
            continue
        print(name, f"({res['pairs']} pairs) — mass − current, mean ± SE over hold-out groups")
        for kk, v in res.items():
            if isinstance(v, dict):
                print(f"   {kk:32s} mass {v['mass']:.4f}  current {v['current']:.4f}  diff {v['difference']:+.4f} ± {v['se'] if v['se'] is not None else float('nan'):.4f}")
    for name, out in rep["swap"].items():
        print("swap", name, {e: f"{v['model_p0']:.4f}→{v['model_p1']:.4f} ({v['relative']:+.2f}, {v['change']:+.4f}±{v['se'] if v['se'] is not None else float('nan'):.4f})" for e, v in out.items()})
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    json.dump(run(a.root), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
