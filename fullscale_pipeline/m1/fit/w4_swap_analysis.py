# Check W4 — does the word-similarity term manufacture people–idea grouping? (owner 2026-11-26: "I approve the 70 fits and
# the paired criterion"; RL-405). Per cell (approach, K, chosen r) and seed s (hold-out group 1, seeds 1–10), for each
# social entity type the swap reading (swap_link_measure.one: model co-membership excess of article pairs sharing that
# entity, minus all pairs):
#   R_on,s  = excess(term on, p 1) − excess(term on, p 0)        R_off,s = the same with the term off (r 0)
#   d_s     = R_on,s − R_off,s   (same seed = same swap permutation and same start; the term starts after the warm-up)
# Flatness f_s = −sign(R̄_off) × (R_on,s − R_off,s), R̄_off = mean over seeds: f > 0 means the term-on response is closer
# to 0 than the term-off one (flatter); for a falling response (R̄_off < 0) f_s = d_s. Decided criterion, people entity (approach 1 author,
# approach 2 co-author tie): W4 flag if mean f > 2 SE(f) over seeds; "not testable" if |mean R_off| ≤ 2 SE(R_off).
# Journals and organisations reported beside; data excess at p 1 (must be ≈ 0: checks the swap).
# Sources: p 0 = stage0_pass3 (r 0 and r*); p 1 term off = w4_swap; p 1 term on = w4_swap, weight_curve (W1 arm, K 5 r 0.3)
# or weight_check (current scaling, K 13 r 1.0).
# Usage: python -m fit.w4_swap_analysis --out fit/results/stage0/w4_swap.json
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np
from .aggregate import load_fit, latest_fits
from .swap_link_measure import one as swap_one
from .plan import W4_CELLS

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
PEOPLE = {1: "author", 2: "co-author tie"}


def plain_w1(arm):
    return not arm.get("mass_weights") and arm.get("semantic_pull") is None and not arm.get("child_child") \
        and not arm.get("log_damping") and not arm.get("swap_single_band")


def collect(root):
    want = {(a, K, r) for a, K, rs in W4_CELLS for r in (0.0, rs)}
    out = defaultdict(dict)                       # (approach, K, r, p) -> seed -> fit
    for st in ("stage0_pass3", "w4_swap", "weight_curve", "weight_check"):
        for f in latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(root, st, "*", "measures.json"))]):
            j = f["job"]; arm = j.get("arm") or {}
            if j["group"] != 0 or not f["result"]["converged"] or not plain_w1(arm):
                continue
            key = (j["approach"], j["K"], float(arm.get("r", 0.0)), float(arm.get("swap_p", 0.0)))
            if key[:3] in want and (st == "stage0_pass3") == (key[3] == 0.0):
                out[key].setdefault(j["seed"], f)
    return out


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    fits = collect(a.root); cache = {}; read = {}
    rep = {}
    for ap_, K, r in W4_CELLS:
        keys = {(t, p): (ap_, K, r if t == "on" else 0.0, p) for t in ("on", "off") for p in (0.0, 1.0)}
        seeds = sorted(set.intersection(*[set(fits[k]) for k in keys.values()]))
        vals = {}
        if not seeds:
            rep[f"approach_{ap_}_K{K}_r{r:g}"] = {"seeds": 0, "missing": [f"{t}_p{int(p)}" for (t, p), k in keys.items() if not fits[k]]}
            continue
        for (t, p), k in keys.items():
            for s in seeds:
                d = fits[k][s]["dir"]
                if d not in read:
                    read[d] = swap_one(fits[k][s], cache)
                vals[(t, p, s)] = read[d]
        cell = {"seeds": len(seeds), "sources": {f"{t}_p{int(p)}": sorted({os.path.basename(os.path.dirname(fits[k][s]['dir'])) for s in seeds})
                                                   for (t, p), k in keys.items()}, "entities": {}}
        for e in vals[("on", 0.0, seeds[0])]:
            Ron = [vals[("on", 1.0, s)][e]["model_excess"] - vals[("on", 0.0, s)][e]["model_excess"] for s in seeds]
            Roff = [vals[("off", 1.0, s)][e]["model_excess"] - vals[("off", 0.0, s)][e]["model_excess"] for s in seeds]
            m_off, se_off = ms(Roff); sgn = -1.0 if m_off < 0 else 1.0
            f = [-sgn * (rn - ro) for ro, rn in zip(Roff, Ron)]
            mf, sf = ms(f)
            p0on = ms([vals[("on", 0.0, s)][e]["model_excess"] for s in seeds])[0]
            p0off = ms([vals[("off", 0.0, s)][e]["model_excess"] for s in seeds])[0]
            ent = {"R_on": dict(zip(("mean", "se"), ms(Ron))), "R_off": dict(zip(("mean", "se"), (m_off, se_off))),
                   "R_on_relative": ms(Ron)[0] / p0on if p0on else None, "R_off_relative": m_off / p0off if p0off else None,
                   "flatness": {"mean": mf, "se": sf, "n_se": mf / sf if sf else None},
                   "testable": bool(abs(m_off) > 2 * se_off), "data_excess_p1_on": ms([vals[("on", 1.0, s)][e]["data_excess"] for s in seeds])[0],
                   "data_excess_p0": ms([vals[("on", 0.0, s)][e]["data_excess"] for s in seeds])[0]}
            ent["W4_flag"] = bool(ent["testable"] and mf > 2 * sf)
            ent["decides"] = e == PEOPLE[ap_]
            cell["entities"][e] = ent
        rep[f"approach_{ap_}_K{K}_r{r:g}"] = cell
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    for name, c in rep.items():
        print(f"\n{name} ({c['seeds']} paired seeds)" + (f" missing {c['missing']}" if not c["seeds"] else ""))
        if not c["seeds"]:
            continue
        for e, v in c["entities"].items():
            print(f"  {e:28s} R_off {v['R_off']['mean']:+.4f}±{v['R_off']['se']:.4f}  R_on {v['R_on']['mean']:+.4f}±{v['R_on']['se']:.4f}  "
                  f"flatness {v['flatness']['mean']:+.4f}±{v['flatness']['se']:.4f}  testable {v['testable']}  W4 flag {v['W4_flag']}"
                  f"{'  <- decides' if v['decides'] else ''}  data p1 {v['data_excess_p1_on']:+.4f}")


if __name__ == "__main__":
    main()
