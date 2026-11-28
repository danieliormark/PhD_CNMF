# D1 re-reads from the saved d1_analysis.json, no fits (owner 2026-11-28; RL-432):
#   (1) is an arm's held-out gain Δ related to the size of J1's own gain? — across K (per approach and arm) and across
#       hold-out groups within (approach, K) (Pearson, then pooled after standardising within cell);
#   (2) semantic-equal value classes re-read with δ_value = 10 % of J1's value score (owner: "I agree with using δ_value
#       = 10 % of J1's value score"), Dunnett critical values as in the saved reading;
#   (3) social held-out gain of the approach's affiliation relation: arm − J1 per hold-out group (mean over seeds),
#       mean ± SE over groups (hold-out-draw noise), relative to J1, in J1's seed SD.
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.d1_reread fit/results/stage1/d1_analysis.json fit/results/stage1/d1_reread.json
import json, math, sys
from collections import defaultdict
import numpy as np
from fit import d1_analysis as A

ARMS = ("J2", "J3a", "J3b", "J4", "J5")


def group_means(units, approach, K, arm, f):
    acc = defaultdict(list)
    for key, u in units.items():
        a, k, g = map(int, key.split("|"))
        if a != approach or k != K:
            continue
        for v in u["fits"].values():
            if v["arm"] == arm:
                x = f(v)
                if x is not None:
                    acc[g].append(x)
    return {g: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")) for g, v in acc.items()}


def main(src, dst):
    d = json.load(open(src)); units = d["_units"]; out = {}
    for ap in (1, 2):
        rep = d[f"approach_{ap}"]; o = {"per_K": {}}
        across_K = defaultdict(list); pooled = []
        for K in map(int, rep["per_K"]):
            e = rep["per_K"][str(K)]; x = {}
            gain = lambda v: (v["heldout"]["existence_sem"] - v["heldout"]["reference_sem"]
                              if v["heldout"]["existence_sem"] is not None and v["heldout"]["reference_sem"] is not None else None)
            g1 = group_means(units, ap, K, "J1", gain)
            corr = {}
            for arm in ARMS:
                dg = e["M1_groups"].get(arm) or {}
                gs = sorted(set(g1) & {int(g) for g in dg})
                a1 = np.array([g1[g][0] for g in gs]); dd = np.array([dg[str(g)] for g in gs])
                corr[arm] = float(np.corrcoef(a1, dd)[0, 1]) if len(gs) > 2 else None
                if len(gs) > 2:
                    pooled += list(zip((a1 - a1.mean()) / a1.std(ddof=1), (dd - dd.mean()) / dd.std(ddof=1)))
                across_K[arm].append((e["J1_gain"], e["M1"][arm]["mean"]))
            x["groups_corr_J1gain_vs_delta"] = corr
            # (2) value re-read
            vj1 = group_means(units, ap, K, "J1", lambda v: v["heldout"]["value_sem"])
            dv = A.DELTA_SHARE * float(np.mean([m for m, _ in vj1.values()]))
            vr = e["M1_value_reported"]; c95, c90 = vr["crit_dunnett_95"], vr["crit_dunnett_90"]
            x["value"] = {"J1_value": dv / A.DELTA_SHARE, "delta_value": dv,
                          **{arm: {"mean": vr[arm]["mean"], "se": vr[arm]["se"],
                                   "class": A.classify(vr[arm]["mean"], vr[arm]["se"], c95, c90, dv)[0],
                                   "class_with_existence_delta": vr[arm]["class"]} for arm in ARMS}}
            # (3) social held-out gain with SE
            rels = sorted({r for u in units.values() for v in u["fits"].values() for r in (v["heldout"]["social_gain"] or {})})
            soc = {}
            for r in rels:
                s1 = group_means(units, ap, K, "J1", lambda v, r=r: (v["heldout"]["social_gain"] or {}).get(r))
                if not s1:
                    continue
                sd1 = float(np.nanmean([s for _, s in s1.values()]))
                soc[r] = {"J1": float(np.mean([m for m, _ in s1.values()])), "J1_seed_sd": sd1}
                for arm in ARMS:
                    sx = group_means(units, ap, K, arm, lambda v, r=r: (v["heldout"]["social_gain"] or {}).get(r))
                    gs = sorted(set(s1) & set(sx)); dd = np.array([sx[g][0] - s1[g][0] for g in gs])
                    if len(dd) > 1:
                        m, se = float(dd.mean()), float(dd.std(ddof=1) / math.sqrt(len(dd)))
                        soc[r][arm] = {"diff": m, "se": se, "relative": m / soc[r]["J1"], "in_seed_sd": m / sd1 if sd1 else None,
                                       "groups": len(dd)}
            x["social"] = soc
            o["per_K"][str(K)] = x
        o["across_K_corr_J1gain_vs_delta"] = {arm: float(np.corrcoef(*zip(*v))[0, 1]) for arm, v in across_K.items()}
        P = np.array(pooled); o["groups_corr_pooled"] = {"r": float(np.corrcoef(P[:, 0], P[:, 1])[0, 1]), "n": len(P)}
        out[f"approach_{ap}"] = o
    json.dump(out, open(dst, "w"), indent=1)
    for ap, o in out.items():
        print("==", ap, "across-K corr", {k: round(v, 2) for k, v in o["across_K_corr_J1gain_vs_delta"].items()}, "pooled groups", o["groups_corr_pooled"])
        for K, x in o["per_K"].items():
            print(f" K {K} groups corr", {k: (round(v, 2) if v is not None else None) for k, v in x["groups_corr_J1gain_vs_delta"].items()})
            v = x["value"]; print(f"   value J1 {v['J1_value']:.4f} δ {v['delta_value']:.5f}", {a: (round(v[a]['mean'], 4), round(v[a]['se'], 4), v[a]['class'], v[a]['class_with_existence_delta']) for a in ARMS})
            for r, s in x["social"].items():
                print(f"   social {r} J1 {s['J1']:.5f} (seed sd {s['J1_seed_sd']:.5f})", {a: (round(s[a]['diff'], 5), round(s[a]['se'], 5), round(s[a]['relative'], 2)) for a in ARMS if a in s})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
