# Checkpoint-3 analysis (ticket 107): the 72 real-data fits (approach 1, hold-out group 1, K = 5 and 20, 3 seeds, terms off).
#   (i)   fraction rule per loss (largest-stable, S2–S4; S1 needs a known truth and is not applicable), 2026-11-10;
#   (ii)  start comparison: plain NNDSVD, hybrid (subsample NNDSVD), random — alignment, held-out prediction, stability;
#   (iii) float32 vs float64 at the provisional fractions;
#   (iv)  timing and the extrapolation to the stage budget.
# Usage: python -m fit.checkpoint3_analysis --root /mnt/hum01-rds/Basov/p91688di/m1_fits --out DIR
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

from . import measures as M
from .aggregate import load_UZ, load_entity

FRACTIONS = (0.01, 0.02, 0.03)
FRACTIONS_BY_BETA = {2: (0.01, 0.02, 0.03), 1: (0.01, 0.02, 0.03, 0.05, 0.075, 0.1)}   # KL extended 2026-11-12
PROV = {2: 0.03, 1: 0.01}


def load(root):
    fits = []
    for d in sorted(glob.glob(os.path.join(root, "checkpoint3", "*", "measures.json"))):
        dd = os.path.dirname(d)
        spec = json.load(open(dd + "/spec.json")); res = json.load(open(dd + "/result.json")); m = json.load(open(d))
        h = res["data_loss_history"]
        ca, pc = m["heldout"]["child_article"], m["heldout"]["parent_child"]
        key = "squared_error" if spec["job"]["beta"] == 2 else "kl_error"
        fits.append(dict(dir=dd, job=spec["job"], res=res, m=m, final=h[-1], low=min(h),
                         ho=m["heldout_combined"]["semantic_only_" + key],
                         cs_ca=ca["captured_share"], cs_pc=pc["captured_share"],
                         auc_ca=ca["auc"], auc_pc=pc["auc"],
                         auc_ca_b=ca["auc_by_band"], within=m["presence"]["child_article"]["within_share"],
                         pmin=min(m["presence"]["child_article"]["presence"]),
                         mono=m["mono_flags"]))
    return fits


def cell(f):
    j = f["job"]
    return (j["beta"], j["K"], j["start"], j["dtype"], j["frac"])


def stats(xs):
    xs = np.array(xs, float)
    return {"mean": float(xs.mean()), "min": float(xs.min()), "max": float(xs.max()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0}


def seed_stability(fs):
    UZ = [load_UZ(f["dir"]) for f in fs]
    rel = {r: tuple(v) for r, v in fs[0]["res"]["relations"].items()}
    wms = [M.weighted_memberships(U, Z, rel) for U, Z in UZ]
    ents = [load_entity(f["dir"]) for f in fs]
    placed = {fa: M.placed_entities([e[fa]["reproduced_share"] for e in ents], ents[0][fa]["reproduced_share_degree_only"]) for fa in ents[0]}
    st = M.stability(wms, placed)
    return {"A_adj": st["A_adjusted"], "B_adj": st["B_adjusted"], "seeds_pass": sum(s["passes"] for s in st["per_seed"]),
            "per_facet_B": st["per_facet_B"]}, wms


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fits = load(a.root)
    cells = defaultdict(list)
    for f in fits:
        cells[cell(f)].append(f)
    table = {}
    wm_cache = {}
    for c, fs in sorted(cells.items()):
        fs.sort(key=lambda f: f["job"]["seed"])
        st, wms = seed_stability(fs)
        wm_cache[c] = wms
        table[c] = {"n": len(fs), "converged": sum(f["res"]["converged"] for f in fs),
                    "epochs": stats([f["res"]["epochs"] for f in fs]), "seconds": stats([f["res"]["seconds_total"] for f in fs]),
                    "final_loss": stats([f["final"] for f in fs]), "s2_gap": max((f["final"] - f["low"]) / f["low"] for f in fs),
                    "heldout": stats([f["ho"] for f in fs]), "captured_ca": stats([f["cs_ca"] for f in fs]),
                    "captured_pc": stats([f["cs_pc"] for f in fs]), "auc_ca": stats([f["auc_ca"] for f in fs]),
                    "auc_pc": stats([f["auc_pc"] for f in fs]), "auc_ca_3to5": stats([f["auc_ca_b"]["3-5"]["auc"] for f in fs if f["auc_ca_b"]["3-5"]["auc"] is not None]),
                    "auc_ca_6p": stats([f["auc_ca_b"]["6+"]["auc"] for f in fs if f["auc_ca_b"]["6+"]["auc"] is not None]),
                    "within_ca": stats([f["within"] for f in fs]), "presence_min_ca": stats([f["pmin"] for f in fs]),
                    "mono_domain": [len(f["mono"]["mono_domain"]) for f in fs], "mono_facet": [len(f["mono"]["mono_facet"]) for f in fs],
                    "stability": st}
    out = {"cells": {json.dumps(k): v for k, v in table.items()}}
    # (i) fraction rule: hybrid start, float64
    rule = {}
    for beta in (2, 1):
        for K in (5, 20):
            rows, ok_below = {}, True
            fracs = [fr for fr in FRACTIONS_BY_BETA[beta] if (beta, K, "nndsvd_sub0.8", "float64", fr) in table]
            for i, fr in enumerate(fracs):
                t = table[(beta, K, "nndsvd_sub0.8", "float64", fr)]
                if i == 0:
                    rows[fr] = {"reference": True}
                    continue
                s = table[(beta, K, "nndsvd_sub0.8", "float64", fracs[i - 1])]
                n = t["n"]
                S2 = t["s2_gap"] <= 1e-3
                S3a = t["final_loss"]["mean"] - s["final_loss"]["mean"] <= s["final_loss"]["max"] - s["final_loss"]["min"]
                S3b = t["heldout"]["mean"] - s["heldout"]["mean"] <= s["heldout"]["sd"] / np.sqrt(n)
                S4 = t["converged"] >= s["converged"]
                passes = bool(S2 and S3a and S3b and S4)
                rows[fr] = {"S2_no_drift": bool(S2), "S3_train_loss": bool(S3a), "S3_heldout_within_1SE": bool(S3b), "S4_convergence": bool(S4),
                            "passes_vs_smaller": passes, "stable": bool(passes and ok_below)}
                ok_below = ok_below and passes
            rule[f"beta{beta}_K{K}"] = rows
    out["fraction_rule"] = rule
    # (iii) float32 vs float64 at the provisional fractions
    f32 = {}
    for beta in (2, 1):
        for K in (5, 20):
            a64 = cells[(beta, K, "nndsvd_sub0.8", "float64", PROV[beta])]; a32 = cells[(beta, K, "nndsvd_sub0.8", "float32", PROV[beta])]
            a64.sort(key=lambda f: f["job"]["seed"]); a32.sort(key=lambda f: f["job"]["seed"])
            agree = [M.track_pair(wm_cache[(beta, K, "nndsvd_sub0.8", "float64", PROV[beta])][i],
                                  wm_cache[(beta, K, "nndsvd_sub0.8", "float32", PROV[beta])][i]) for i in range(3)]
            seedst = table[(beta, K, "nndsvd_sub0.8", "float64", PROV[beta])]["stability"]
            f32[f"beta{beta}_K{K}"] = {
                "epochs_64": [f["res"]["epochs"] for f in a64], "epochs_32": [f["res"]["epochs"] for f in a32],
                "final_loss_rel_diff": [abs(x["final"] - y["final"]) / y["final"] for x, y in zip(a32, a64)],
                "heldout_rel_diff": [abs(x["ho"] - y["ho"]) / y["ho"] for x, y in zip(a32, a64)],
                "within_diff": [x["within"] - y["within"] for x, y in zip(a32, a64)],
                "converged_64": sum(f["res"]["converged"] for f in a64), "converged_32": sum(f["res"]["converged"] for f in a32),
                "seconds_64": [f["res"]["seconds_total"] for f in a64], "seconds_32": [f["res"]["seconds_total"] for f in a32],
                "membership_agreement_32_vs_64": {"A": [x["A_model"] for x in agree], "B": [x["B_model"] for x in agree]},
                "seed_to_seed_for_comparison": {"A_adj": seedst["A_adj"], "B_adj": seedst["B_adj"]}}
    out["float32"] = f32
    json.dump(out, open(os.path.join(a.out, "checkpoint3_analysis.json"), "w"), indent=1, default=float)
    # printed summary
    print("(i) FRACTION RULE (S2 no drift, S3 training loss and held-out within 1 SE, S4 convergence) vs the next smaller fraction")
    for k, rows in rule.items():
        print(" ", k, {fr: ("ref" if r.get("reference") else ("STABLE" if r["stable"] else "fails " + ",".join(n for n in ("S2_no_drift", "S3_train_loss", "S3_heldout_within_1SE", "S4_convergence") if not r[n])))
                       for fr, r in rows.items()})
    print("\n(ii) STARTS (float64; fraction = provisional per loss): mean over 3 seeds")
    print(" β K  start         conv epochs  sec    held-out  capt.CA capt.PC AUC.CA AUC.PC  AUC.CA(3-5)  within  presMin  A_adj  B_adj  seeds-pass")
    for beta in (2, 1):
        for K in (5, 20):
            for st in ("nndsvd", "nndsvd_sub0.8", "random"):
                t = table[(beta, K, st, "float64", PROV[beta])]
                print(f" {beta} {K:2d} {st:13s} {t['converged']}/3 {t['epochs']['mean']:6.0f} {t['seconds']['mean']:6.0f} {t['heldout']['mean']:9.4g} {t['captured_ca']['mean']:7.3f} {t['captured_pc']['mean']:7.3f} "
                      f"{t['auc_ca']['mean']:6.3f} {t['auc_pc']['mean']:6.3f} {t['auc_ca_3to5']['mean']:8.3f}     {t['within_ca']['mean']:6.3f}  {t['presence_min_ca']['mean']:6.3f}  {t['stability']['A_adj']:6.3f} {t['stability']['B_adj']:6.3f}  {t['stability']['seeds_pass']}/3")
    print("\n(iii) FLOAT32 vs FLOAT64 (hybrid start, provisional fraction)")
    for k, v in f32.items():
        print(" ", k, "epochs 64/32", v["epochs_64"], v["epochs_32"], "| final-loss rel.diff max %.1e" % max(v["final_loss_rel_diff"]),
              "| held-out rel.diff max %.1e" % max(v["heldout_rel_diff"]), "| within diff max %.1e" % max(abs(x) for x in v["within_diff"]),
              "| agreement 32 vs 64: A %.4f B %.4f" % (np.mean(v["membership_agreement_32_vs_64"]["A"]), np.mean(v["membership_agreement_32_vs_64"]["B"])),
              "| seed-to-seed adj A %.3f B %.3f" % (v["seed_to_seed_for_comparison"]["A_adj"], v["seed_to_seed_for_comparison"]["B_adj"]),
              "| sec 64 %.0f → 32 %.0f" % (np.mean(v["seconds_64"]), np.mean(v["seconds_32"])))
    print("\n(iv) TIMING (seconds per fit, mean of 3; hybrid, float64)")
    for beta in (2, 1):
        for K in (5, 20):
            print(f"  β{beta} K{K}: " + ", ".join(f"frac {fr}: {table[(beta, K, 'nndsvd_sub0.8', 'float64', fr)]['seconds']['mean']:.0f}s/{table[(beta, K, 'nndsvd_sub0.8', 'float64', fr)]['epochs']['mean']:.0f}ep"
                                                for fr in FRACTIONS_BY_BETA[beta] if (beta, K, 'nndsvd_sub0.8', 'float64', fr) in table))


if __name__ == "__main__":
    main()
