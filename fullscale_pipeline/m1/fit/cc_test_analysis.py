# child×child focused test (owner 2026-11-24, RL-377/378): the relation arms (stage cc_test) against the regulariser runs
# (stage0_pass3: r 0 / 0.3 / 1) at approach 1, K 5 and 13, hold-out groups 1–5 × seeds 1–10. Per (cell, group):
#   held-out per relation — gain = within-class AP − degree-only reference (child×article, parent×child), value (partial
#     Spearman), combined existence with the main-setting weights (RL-358);
#   flatness — effective number of communities 2^spread of W-tied and of untied child hyperedges (median over entities);
#   stability — Track A / B (measures.stability, coverage set, chance-adjusted) and the companion readings decided to be
#     reported (RL-375): centred agreement (per entity, Pearson correlation across the K communities of its two matched
#     memberships; flat profiles that differ by noise score ≈ 0) and information-weighted Track A (each entity's 1 − JS
#     agreement weighted by 1 − spread ÷ log2 K, mean of the two seeds); both chance-adjusted against fit 2 with rows
#     shuffled, facet means, model level = mean over facets;
#   community readings — pass3_readings.one_fit (T1, crossing child×article weight, article spread, mono-domain, …) and
#     idea / people coherence (weight_curve_community.idea_people).
# Decision rule (fixed before the results, design doc 2026-11-24): per K, an arm replaces the regulariser (r* = 0.3 at K 5
# and K 13, rule C) if (i) its child×article gain is not below the regulariser's by more than 1 paired SE, (ii) the median
# effective communities of W-tied hyperedges is lower beyond 1 SE, (iii) the companion stability reading (centred
# agreement, model level) is not below by more than 1 SE. Differences paired by hold-out group (mean over seeds); SE over
# the 5 groups (hold-out-draw noise, a yardstick). Among relation-only arms, the best by rule C is reported (peak
# child×article gain; ties within 1 SE → smaller matched r); every arm is judged and reported.
# Usage: python -m fit.cc_test_analysis --out fit/results/stage0/cc_test.json --workers 8
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits, _rds_of, COVERAGE_C
from .align import fit_rds
from . import pass3_readings as P3
from .weight_curve_community import idea_people

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KS = (5, 13)
R_STAR = {5: 0.3, 13: 0.3}
_G = {}


def init():
    P3.setup(D.BUILD, "A")
    W = sp.load_npz(D.Build(D.BUILD).path + "W_child_child.npz").tocsr(); W.eliminate_zeros()
    _G["tied"] = np.diff(W.indptr) > 0


def label(job):
    arm = job.get("arm") or {}
    cc = arm.get("child_child")
    if not cc:
        return f"reg_r{float(arm.get('r', 0)):g}"
    return f"cc_{cc['mode']}_m{cc['matched_r']:g}" + (f"+reg_r{float(arm['r']):g}" if float(arm.get("r", 0)) > 0 else "")


def companion(wms, keep, seed=0):
    """Centred agreement and information-weighted Track A, chance-adjusted, model level (mean over facets)."""
    rng = np.random.default_rng(seed); facets = sorted(wms[0]); K = wms[0][facets[0]]["P"].shape[1]
    res = defaultdict(list)
    for i, j in combinations(range(len(wms)), 2):
        perm = M.match(wms[i], wms[j], facets)
        sh = M._shuffled(wms[j], rng); perm_c = M.match(wms[i], sh, facets)
        cen, inf = {}, {}
        for lab, w2, pm in (("real", wms[j], perm), ("chance", sh, perm_c)):
            ce, iw = [], []
            for f in facets:
                m = keep[f]
                if not m.any():
                    continue
                P1, P2 = wms[i][f]["P"][m], w2[f]["P"][:, pm][m]
                a = P1 - P1.mean(1, keepdims=True); b = P2 - P2.mean(1, keepdims=True)
                den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
                ok = den > 1e-12
                ce.append(float(np.mean(np.where(ok, (a * b).sum(1) / np.maximum(den, 1e-300), 0.0))))
                agr = 1 - M._js_distance_e(P1, P2)
                wt = 1 - 0.5 * (M.entropy_bits(P1) + M.entropy_bits(P2)) / math.log2(K)
                iw.append(float((wt * agr).sum() / max(wt.sum(), 1e-300)))
            cen[lab], inf[lab] = float(np.mean(ce)), float(np.mean(iw))
        res["centred"].append((cen["real"] - cen["chance"]) / (1 - cen["chance"]))
        res["info_weighted_A"].append((inf["real"] - inf["chance"]) / (1 - inf["chance"]))
        res["centred_raw"].append(cen["real"])
    return {k: float(np.mean(v)) for k, v in res.items()}


def task(args):
    lab, K, g, dirs = args
    fits = sorted([f for f in (load_fit(d) for d in dirs) if f["result"]["converged"]], key=lambda f: f["job"]["seed"])
    rel = {k: tuple(v) for k, v in fits[0]["result"]["relations"].items()}
    UZ = [load_UZ(f["dir"]) for f in fits]
    wms = [M.weighted_memberships(U, Z, rel) for U, Z in UZ]
    covs = [M.entity_coverage(U, Z, _rds_of(f["job"])) for (U, Z), f in zip(UZ, fits)]
    keep = M.covered_entities(covs, COVERAGE_C)
    tied = _G["tied"]
    per_fit = []
    cache = {}
    for f, wm in zip(fits, wms):
        tp = f["measures"]["heldout_two_part"]; comb = f["measures"]["heldout_two_part_combined"]
        s = M.entropy_bits(wm["child"]["P"]); live = keep["child"]
        row = {"gain_child_article": tp["child_article"]["ap_within_class"] - tp["child_article"]["baseline_weights_ap_within_class"],
               "gain_parent_child": tp["parent_child"]["ap_within_class"] - tp["parent_child"]["baseline_weights_ap_within_class"],
               "value_child_article": tp["child_article"]["value_partial_spearman"], "value_parent_child": tp["parent_child"]["value_partial_spearman"],
               "existence_main_weights": comb["existence"],
               "eff_comm_tied_median": float(np.median(2 ** s[tied & live])), "eff_comm_untied_median": float(np.median(2 ** s[~tied & live])),
               "eff_comm_tied_p90": float(np.quantile(2 ** s[tied & live], 0.9))}
        rds = {rd.name: rd for rd in fit_rds(f["job"], cache)}
        p3 = P3.one_fit(f, rds)
        row.update({"T1_fails": float(p3["T1_fails"]), "mono_count": float(p3["mono"]["count"]),
                    "crossing_child_article": p3["crossing"]["child_article"]["crossing_weight_share"],
                    "comem_within_child_article": p3["crossing"]["child_article"]["co_membership_within"],
                    "article_spread_bits": p3["mosaic_blend"]["article_entropy_mean"],
                    "lower_presence_median": p3["mono"]["lower_presence_median"]})
        ip = idea_people(f, rds); row.update({"idea_corr": ip["idea_corr_w"], "idea_excess": ip["idea_excess"], "people_corr": ip["people_corr"]})
        sem = sum(rd.weight * float(rd.tie_val.sum()) for rd in rds.values() if rd.domain == "semantic")
        row["semantic_pull_share"] = sem / sum(rd.weight * float(rd.tie_val.sum()) for rd in rds.values())
        per_fit.append(row)
    st = M.stability(wms, keep)
    comp = companion(wms, keep)
    out = {k: float(np.mean([r[k] for r in per_fit])) for k in per_fit[0]}
    out.update({"track_A_adjusted": st["A_adjusted"], "track_B_adjusted": st["B_adjusted"], **comp, "fits": len(fits)})
    return {"label": lab, "K": K, "group": g, "readings": out}


def decide(res):
    by = defaultdict(dict)
    for r in res:
        by[(r["K"], r["label"])][r["group"]] = r["readings"]
    rep = {}
    for K in KS:
        ref = by.get((K, f"reg_r{R_STAR[K]:g}"), {})
        arms = sorted(l for (k, l) in by if k == K and l.startswith("cc_"))
        out = {}

        def diff(lab, key):
            gs = sorted(set(by[(K, lab)]) & set(ref))
            d = np.array([by[(K, lab)][g][key] - ref[g][key] for g in gs])
            return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None
        for lab in sorted(l for (k, l) in by if k == K):
            keys = by[(K, lab)][next(iter(by[(K, lab)]))].keys()
            e = {"mean": {k: float(np.mean([v[k] for v in by[(K, lab)].values()])) for k in keys}}
            if ref and lab != f"reg_r{R_STAR[K]:g}":
                e["vs_regulariser_r_star"] = {k: dict(zip(("difference", "se"), diff(lab, k))) for k in keys if k != "fits"}
                if lab.startswith("cc_"):
                    ca, cs = diff(lab, "gain_child_article"); fl, fs = diff(lab, "eff_comm_tied_median"); sb, ss = diff(lab, "centred")
                    crit = {"i_child_article_not_worse": not (ca < -cs), "ii_less_flat": fl < -fs, "iii_stability_not_worse": not (sb < -ss)}
                    e["decision_rule"] = {**crit, "replaces_regulariser": all(crit.values())}
            out[lab] = e
        rel_only = [l for l in arms if "+reg" not in l]
        if rel_only:
            peak = max(rel_only, key=lambda l: out[l]["mean"]["gain_child_article"])
            out["_best_relation_arm_rule_C"] = peak
        rep[f"K{K}"] = out
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    load = lambda st: latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(a.root, st, "*", "measures.json"))])
    fs = [f for f in load("cc_test")] + [f for f in load("stage0_pass3") if f["job"]["approach"] == 1 and f["job"]["K"] in KS]
    by = defaultdict(list)
    for f in fs:
        j = f["job"]
        if j["approach"] == 1 and j["K"] in KS:
            by[(label(j), j["K"], j["group"])].append(f["dir"])
    tasks = [(l, K, g, ds) for (l, K, g), ds in sorted(by.items())]
    print(len(tasks), "tasks", flush=True)
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as pool:
            res = pool.map(task, tasks, chunksize=1)
    else:
        init(); res = [task(t) for t in tasks]
    rep = {"per_group": res, "decision": decide(res)}
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    for K, out in rep["decision"].items():
        print(f"\n{K}: best relation arm by rule C: {out.get('_best_relation_arm_rule_C')}")
        for lab, e in out.items():
            if lab.startswith("_"):
                continue
            m = e["mean"]
            line = f"  {lab:24s} ca-gain {m['gain_child_article']:.4f} eff-tied {m['eff_comm_tied_median']:.2f} centred {m['centred']:.3f} A {m['track_A_adjusted']:.3f} B {m['track_B_adjusted']:.3f} idea {m['idea_corr']:.3f}"
            if "decision_rule" in e:
                line += f"  rule {e['decision_rule']}"
            print(line)


if __name__ == "__main__":
    main()
