# Two-part held-out criterion (ticket 107, 2026-11-13, option D) on the checkpoint-3 and KL-extension fits, from saved
# factors (no fits): existence = degree-corrected AP (child×article 0.375, child×parent 0.375, author×affiliation 0.25),
# value = partial Spearman among hidden ties (child×article 0.5, child×parent 0.5); fraction rule S2–S4 with S3 as two
# equal guards (each: not worse than the next smaller fraction by more than one SE over seeds).
# Usage: python -m fit.heldout_twopart_analysis --root /mnt/hum01-rds/Basov/p91688di/m1_fits --out DIR
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from . import data as D, measures as M
from .aggregate import load_UZ
from .checkpoint3_analysis import FRACTIONS_BY_BETA

EW = {"child_article": 0.375, "parent_child": 0.375, "author_affiliation": 0.25}
VW = {"child_article": 0.5, "parent_child": 0.5}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    b = D.Build(); rds = {}
    cells = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.root, "checkpoint3", "*", "UZ.npz"))):
        dd = os.path.dirname(d); spec = json.load(open(dd + "/spec.json")); j = spec["job"]
        if j["start"] != "nndsvd_sub0.8" or j["dtype"] != "float64":
            continue
        res = json.load(open(dd + "/result.json"))
        key = (j["approach"], j["group"])
        if key not in rds:
            rds[key] = {r.name: r for r in D.fit_data(b, j["approach"], group=j["group"])}
        U, Z = load_UZ(dd)
        per = {n: M.heldout_two_part(U, Z, rds[key][n], value=n in VW) for n in EW}
        h = res["data_loss_history"]
        cells[(j["beta"], j["K"], j["frac"])].append(dict(
            seed=j["seed"], per=per, existence=sum(EW[n] * per[n]["ap_corrected"] for n in EW),
            value=sum(VW[n] * per[n]["value_partial_spearman"] for n in VW),
            final=h[-1], low=min(h), converged=res["converged"], seconds=res["seconds_total"], epochs=res["epochs"]))
        print(j["beta"], j["K"], j["frac"], j["seed"], round(cells[(j["beta"], j["K"], j["frac"])][-1]["existence"], 4),
              round(cells[(j["beta"], j["K"], j["frac"])][-1]["value"], 4), flush=True)
    st = lambda xs: (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)
    summary, rule = {}, {}
    for (beta, K, fr), fs in sorted(cells.items()):
        e, v = st([f["existence"] for f in fs]), st([f["value"] for f in fs])
        rel = {n: {m: float(np.mean([f["per"][n][m] for f in fs if f["per"][n].get(m) is not None]))
                   for m in fs[0]["per"][n]} for n in EW}
        summary[f"beta{beta}_K{K}_frac{fr}"] = {"n": len(fs), "existence": e, "value": v, "relations": rel,
                                               "seconds": float(np.mean([f["seconds"] for f in fs]))}
    for beta in (2, 1):
        for K in (5, 20):
            fr_list = [fr for fr in FRACTIONS_BY_BETA[beta] if (beta, K, fr) in cells]
            rows, ok_below = {}, True
            for i, fr in enumerate(fr_list):
                if i == 0:
                    rows[fr] = "ref"; continue
                t, s = cells[(beta, K, fr)], cells[(beta, K, fr_list[i - 1])]
                n = len(t)
                tf = np.mean([f["final"] for f in t]); sf = [f["final"] for f in s]
                S2 = max((f["final"] - f["low"]) / f["low"] for f in t) <= 1e-3
                S3a = tf - np.mean(sf) <= max(sf) - min(sf)
                se, te = st([f["existence"] for f in s]), st([f["existence"] for f in t])
                sv, tv = st([f["value"] for f in s]), st([f["value"] for f in t])
                S3e = se[0] - te[0] <= se[1] / np.sqrt(n)
                S3v = sv[0] - tv[0] <= sv[1] / np.sqrt(n)
                S4 = sum(f["converged"] for f in t) >= sum(f["converged"] for f in s)
                fails = [k for k, ok in (("S2", S2), ("S3_train", S3a), ("S3_existence", S3e), ("S3_value", S3v), ("S4", S4)) if not ok]
                ok_below = ok_below and not fails
                rows[fr] = "STABLE" if ok_below else ("fails " + ",".join(fails) if fails else "passes, but a smaller fraction failed")
            rule[f"beta{beta}_K{K}"] = rows
    json.dump({"summary": summary, "fraction_rule": rule}, open(os.path.join(a.out, "heldout_twopart.json"), "w"), indent=1)
    print(json.dumps(rule, indent=1))
    for k, v in summary.items():
        r = v["relations"]
        print(k, "exist %.4f±%.4f value %.4f±%.4f" % (*v["existence"], *v["value"]), "sec %.0f" % v["seconds"],
              "| AP corr ca/pc/aa %.4f %.4f %.4f" % tuple(r[n]["ap_corrected"] for n in EW),
              "| baseline AP corr %.4f %.4f %.4f" % tuple(r[n]["baseline_ap_corrected"] for n in EW),
              "| AP %.4f %.4f %.4f" % tuple(r[n]["ap"] for n in EW), "| base AP %.4f %.4f %.4f" % tuple(r[n]["baseline_ap"] for n in EW),
              "| AUC %.3f %.3f %.3f" % tuple(r[n]["auc"] for n in EW), "| base AUC %.3f %.3f %.3f" % tuple(r[n]["baseline_auc"] for n in EW),
              "| val ps ca/pc %.3f %.3f sp %.3f %.3f base ps %.3f %.3f" % (r["child_article"]["value_partial_spearman"], r["parent_child"]["value_partial_spearman"],
                  r["child_article"]["value_spearman"], r["parent_child"]["value_spearman"],
                  r["child_article"]["baseline_value_partial_spearman"], r["parent_child"]["baseline_value_partial_spearman"]),
              "| ess %.0f %.0f %.0f left-out %.3f %.3f %.3f" % (*(r[n]["correction_ess"] for n in EW), *(r[n]["ties_without_matching_empty"] for n in EW)))


if __name__ == "__main__":
    main()
