# Stage 0 pass 3: readings reported beside the J1 choice (owner 2026-11-19; reporting rule: cheap readings accepted and
# reported), per cell (approach, K, r) of one study (A: genre hyperedges in; B: without; RL-290). Aligned numbering
# (decided 2026-11-17): run `python -m fit.align --stages <stage>` first. Converged fits only; medians over fits.
# Per fit (existing code, so the numbers are comparable with start test 2 and pass 1):
#   child×article requirement T1 (owner 2026-11-19, "I choose T1 + V1"): a fit fails if any community's own cell holds
#     no more than half of its child×article tie weight on either side (start_test_analysis.anchor_guard); the cell's
#     would-be veto V1 by the counting rule at fraction 0.2 (a group fails if > 20 % of its converged seeds fail; the
#     cell fails if > 20 % of groups or of all fits fail); in stage 0 V1 is reported only (step-11 decision);
#   failing communities (T1): their share of the fit's child×article tie weight, and (study A) the genre share of their
#     child-side tie weight against the corpus's (catch-all community of generic vocabulary, RL-281–283);
#   diagonal rule shares and effective partners per relation (start_test_analysis.per_fit);
#   off-diagonal share of reconstructed tie weight and off-diagonal core cells at the positivity floor, per relation
#     (start_test2_analysis.offdiag; standing reading, 2026-11-19);
#   closure ratio per relation (closure_check.one; article×article is the standing one);
#   co-membership on crossing ties vs within and chance, crossing share (blurring_check.one);
#   mono-domain communities (decided rule: P_k < 0.1 in one domain and ≥ 0.5 in the other; report only in stage 0,
#     owner 2026-11-19): count per fit, lower of the two presences; recurrence in the majority of the group's seeds
#     (Track A/B matching, as mono_domain_pass1) for every instance;
#   mosaic vs blend / article coherence (measures.floor_diagnostics; agreed 2026-11-06 for pass 3): share of child
#     hyperedges with ≥ 2 articles below 0.8 × (between-article spread [+ within]) (ω = 0 / 1), mean shortfall, median
#     own / between / within spread (bits), share with largest membership > 0.9; articles' membership entropy;
#   flattening of W-tied child hyperedges (as pass 2, w_form_diagnostics): R ÷ S, mean largest membership of tied /
#     untied child hyperedges;
#   study A: held-out existence of child×article split by the hidden tie's hyperedge class, genre / substantive
#     (generic_hyperedges, final list RL-289), within-class AP and chance (as generic_role).
# Usage: python -m fit.pass3_readings --stage stage0_pass3 --study A --out fit/results/stage0/pass3_readings.json
#        [--build PATH] [--workers 8]
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits
from .start_test_analysis import per_fit as st_per_fit, anchor_guard
from .start_test2_analysis import offdiag
from .closure_check import one as closure_one
from .blurring_check import one as blurring_one
from .mono_domain_pass1 import flags as mono_flags

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
BUILD_B = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/"
FRACTION = 0.2
_W = {}


def bits(P):
    return -(P * np.log2(np.clip(P, 1e-300, None))).sum(1)


def setup(build_path, study):
    b = D.Build(build_path)
    W = sp.load_npz(b.path + "W_child_child.npz").tocsr(); W.eliminate_zeros()
    d = np.asarray(W.sum(1)).ravel(); coo = W.tocoo()
    _W.update(build=b, study=study, coo=coo, v=coo.data / d[coo.row], d=d, N=int((d > 0).sum()))
    if study == "A":
        from .generic_hyperedges import labels
        _W["lab"] = labels(b)


def t1_profile(U, Z, ca, P, lab):
    """Failing communities of T1: share of the tie weight; (study A) genre share of their child-side tie weight."""
    T = P.sum(); d = np.diag(P); rw, cw = P.sum(1), P.sum(0)
    fail = ~((d / np.maximum(rw, 1e-300) > 0.5) & (d / np.maximum(cw, 1e-300) > 0.5))
    out = {"n_failing": int(fail.sum()), "failing_weight_share": float((0.5 * (rw + cw) / T)[fail].sum())}
    if lab is not None and fail.any():
        A, B, z = U["article"], U["child"], Z["child_article"]
        r, c, v = ca.tie_row, ca.tie_col, ca.tie_val
        AZ = A[r] @ z; xh = (AZ * B[c]).sum(1); resp = AZ * B[c] / np.maximum(xh, 1e-300)[:, None]
        tot = (resp * v[:, None]).sum(0); g = (resp * (v * (lab[c] == "G"))[:, None]).sum(0) / np.maximum(tot, 1e-300)
        out["genre_share_failing"] = g[fail].tolist(); out["genre_share_others_median"] = float(np.median(g[~fail])) if (~fail).any() else None
        out["genre_share_corpus"] = float(v[lab[c] == "G"].sum() / v.sum())
    return out


def genre_split(U, Z, ca, lab):
    A, B, z = U["article"], U["child"], Z["child_article"]
    ht = ((A[ca.hid_tie_row] @ z) * B[ca.hid_tie_col]).sum(1); he = ((A[ca.hid_empty_row] @ z) * B[ca.hid_empty_col]).sum(1)
    n_empty_obs = int(ca.obs_row_mask.sum()) * ca.shape[1] - len(ca.tie_row) - len(ca.hid_tie_row)
    scale = math.floor(0.10 * n_empty_obs) / len(he)
    out = {}
    for k in ("G", "S"):
        kt, ke = lab[ca.hid_tie_col] == k, lab[ca.hid_empty_col] == k
        ap, _, ch = M.within_class_ap(ht, he, ca, scale, keep_t=kt, keep_e=ke)
        out[f"existence_{k}"], out[f"chance_{k}"], out[f"hidden_ties_{k}"] = ap, ch, int(kt.sum())
    return out


def one_fit(f, rds):
    j = f["job"]; x, wm = st_per_fit(f, rds)
    U, Z = load_UZ(f["dir"]); ca = rds["child_article"]
    P, _ = M.parts(U, Z, ca); K = P.shape[0]
    lab = _W.get("lab")
    row = {"approach": j["approach"], "K": K, "r": float(j["arm"]["r"]), "group": j["group"], "seed": j["seed"],
           "converged": bool(f["result"]["converged"]), "epochs": f["result"]["epochs"],
           "data_loss": x["data_loss"], "existence": x["existence"], "value": x["value"],
           "T1_fails": x["anchor"]["n_communities_over_half"] < K, "anchor_within_share": x["anchor"]["within_share"],
           "T1": t1_profile(U, Z, ca, P, lab), "rule_share": {n: v["both_share"] for n, v in x["rule"].items()},
           "partners": x["partners"], "detached": x["detached"]}
    od = offdiag(f, rds); cl = closure_one(f, rds); bl = blurring_one(f, rds)
    row["offdiag_share"] = {n: v["offdiag_share"] for n, v in od.items()}
    row["offdiag_at_floor"] = {n: v["offdiag_cells_at_floor"] for n, v in od.items()}
    row["closure_ratio"] = {n: v["ratio_between"] for n, v in cl.items()}
    row["crossing"] = {n: {k: v[k] for k in ("crossing_weight_share", "co_membership_crossing", "co_membership_within",
                                              "co_membership_chance")} for n, v in bl["relation"].items()}
    dp = f["measures"]["domain_presence"]; S = np.array(dp["semantic"]["mean"]); O = np.array(dp["social"]["mean"])
    fl = mono_flags(S, O)
    row["mono"] = {"count": int(sum(v is not None for v in fl)), "flags": fl, "lower_presence_min": float(np.minimum(S, O).min()),
                   "lower_presence_median": float(np.median(np.minimum(S, O)))}
    fd = M.floor_diagnostics(ca, wm)
    row["mosaic_blend"] = {"n": fd["n"], "below_min_omega0": float((fd["shortfall_omega0"] > 0).mean()),
                           "below_min_omega1": float((fd["shortfall_omega1"] > 0).mean()),
                           "mean_shortfall_omega0": float(fd["shortfall_omega0"].mean()),
                           "mean_shortfall_omega1": float(fd["shortfall_omega1"].mean()),
                           "own_median": float(np.median(fd["own_spread"])), "between_median": float(np.median(fd["between"])),
                           "within_median": float(np.median(fd["within"])), "largest_above_0.9": fd["share_largest_above"]["0.9"],
                           "article_entropy_mean": float(bits(wm["article"]["P"]).mean())}
    Pc = wm["child"]["P"]; coo, v, d, N = _W["coo"], _W["v"], _W["d"], _W["N"]
    R = float((v * ((Pc[coo.row] - Pc[coo.col]) ** 2).sum(1)).sum() / N); tied = d > 0; Pt = Pc[tied]
    Sx = float(2 * ((Pt ** 2).sum(1).mean() - (Pt.mean(0) ** 2).sum()))
    row["flattening"] = {"R_over_S": R / Sx, "largest_share_tied": float(Pt.max(1).mean()), "largest_share_untied": float(Pc[~tied].max(1).mean())}
    if lab is not None:
        row["genre_split"] = genre_split(U, Z, ca, lab)
    return row


def task(args):
    a, g, dirs = args
    rds = {rd.name: rd for rd in D.fit_data(_W["build"], a, group=g)}
    out = []
    for dd in dirs:
        f = load_fit(dd)
        try:
            out.append(one_fit(f, rds))
        except Exception as e:                       # one failing fit must not lose the others; reported
            out.append({"approach": a, "group": g, "seed": f["job"]["seed"], "K": f["job"]["K"], "r": float(f["job"]["arm"]["r"]),
                        "error": repr(e)})
    return out


def recurrence(rows, dirs_of):
    """Mono-domain instances: matched community mono in the same direction in > half of the group's converged seeds."""
    inst = []
    by = defaultdict(list)
    for r in rows:
        if r.get("converged") and "mono" in r:
            by[(r["approach"], r["K"], r["r"], r["group"])].append(r)
    for key, rs in by.items():
        if not any(r["mono"]["count"] for r in rs):
            continue
        rs = sorted(rs, key=lambda r: r["seed"]); wms = []
        for r in rs:
            f = load_fit(dirs_of[(key[0], key[1], key[2], key[3], r["seed"])]); U, Z = load_UZ(f["dir"])
            wms.append(M.weighted_memberships(U, Z, {k: tuple(v) for k, v in f["result"]["relations"].items()}))
        facets = sorted(wms[0])
        for i, ri in enumerate(rs):
            for k, fl in enumerate(ri["mono"]["flags"]):
                if fl is None:
                    continue
                same = 1 + sum(rs[jx]["mono"]["flags"][M.match(wms[i], wms[jx], facets)[k]] == fl for jx in range(len(rs)) if jx != i)
                inst.append({"approach": key[0], "K": key[1], "r": key[2], "group": key[3], "seed": ri["seed"], "k": k,
                             "type": fl, "seeds_same_direction": int(same), "n_seeds": len(rs), "recurs": same > len(rs) / 2})
    return inst


def v1(rows):
    """Would-be veto V1 by the counting rule (fraction 0.2) over the cell's converged fits."""
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(bool(r["T1_fails"]))
    gf = [np.mean(v) > FRACTION for v in by.values() if v]
    allf = [x for v in by.values() for x in v]
    return {"fits_failing": int(sum(allf)), "fits": len(allf), "groups_failing": int(sum(gf)), "groups": len(gf),
            "vetoed": bool(np.mean(gf) > FRACTION or np.mean(allf) > FRACTION) if allf else None}


def med(rows, get):
    v = [get(r) for r in rows]; v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def summarise(rows, inst, study):
    cells = defaultdict(list)
    for r in rows:
        if r.get("converged") and "error" not in r:
            cells[(r["approach"], r["K"], r["r"])].append(r)
    rels = sorted({n for r in rows if "offdiag_share" in r for n in r["offdiag_share"]})
    rep = defaultdict(dict)
    for (a, K, rr), rs in sorted(cells.items()):
        s = {"fits_converged": len(rs), "V1_would_be": v1(rs),
             "T1_failing_communities_per_fit": med(rs, lambda r: r["T1"]["n_failing"]),
             "T1_failing_weight_share_median_over_failing_fits": med([r for r in rs if r["T1_fails"]], lambda r: r["T1"]["failing_weight_share"]),
             "anchor_within_share": med(rs, lambda r: r["anchor_within_share"]),
             "mono_per_fit_mean": float(np.mean([r["mono"]["count"] for r in rs])),
             "mono_recurring": int(sum(x["recurs"] for x in inst if (x["approach"], x["K"], x["r"]) == (a, K, rr))),
             "lower_presence_min_median": med(rs, lambda r: r["mono"]["lower_presence_min"]),
             "existence": med(rs, lambda r: r["existence"]), "value": med(rs, lambda r: r["value"])}
        if study == "A":
            gs = [x for r in rs if r["T1_fails"] for x in r["T1"].get("genre_share_failing", [])]
            s["T1_failing_genre_share_median"] = float(np.median(gs)) if gs else None
            s["genre_share_corpus"] = med(rs, lambda r: r["T1"].get("genre_share_corpus"))
            for k in ("G", "S"):
                s[f"existence_{k}"] = med(rs, lambda r, k=k: r["genre_split"][f"existence_{k}"])
                s[f"chance_{k}"] = med(rs, lambda r, k=k: r["genre_split"][f"chance_{k}"])
        for n in [n for n in rels if any(n in r["offdiag_share"] for r in rs)]:   # relations of this approach
            s[f"offdiag_share_{n}"] = med(rs, lambda r, n=n: r["offdiag_share"].get(n))
            s[f"offdiag_at_floor_{n}"] = med(rs, lambda r, n=n: r["offdiag_at_floor"].get(n))
            s[f"closure_ratio_{n}"] = med(rs, lambda r, n=n: r["closure_ratio"].get(n))
            s[f"co_membership_crossing_{n}"] = med(rs, lambda r, n=n: (r["crossing"].get(n) or {}).get("co_membership_crossing"))
            s[f"co_membership_within_{n}"] = med(rs, lambda r, n=n: (r["crossing"].get(n) or {}).get("co_membership_within"))
            s[f"co_membership_chance_{n}"] = med(rs, lambda r, n=n: (r["crossing"].get(n) or {}).get("co_membership_chance"))
            s[f"crossing_share_{n}"] = med(rs, lambda r, n=n: (r["crossing"].get(n) or {}).get("crossing_weight_share"))
            s[f"rule_share_{n}"] = med(rs, lambda r, n=n: r["rule_share"].get(n))
            s[f"partners_{n}"] = med(rs, lambda r, n=n: r["partners"].get(n))
        for k in ("below_min_omega0", "below_min_omega1", "mean_shortfall_omega0", "own_median", "between_median",
                  "within_median", "largest_above_0.9", "article_entropy_mean"):
            s[f"mosaic_{k}"] = med(rs, lambda r, k=k: r["mosaic_blend"][k])
        for k in ("R_over_S", "largest_share_tied", "largest_share_untied"):
            s[f"flattening_{k}"] = med(rs, lambda r, k=k: r["flattening"][k])
        rep[f"approach_{a}"][f"K{K}_r{rr:g}"] = s
    return dict(rep)


def run(root, stage, build_path, study, workers):
    setup(build_path, study)
    fits = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(root, stage, "*", "measures.json"))])
    groups = defaultdict(list); dirs_of = {}
    for f in fits:
        j = f["job"]; groups[(j["approach"], j["group"])].append(f["dir"])
        dirs_of[(j["approach"], j["K"], float(j["arm"]["r"]), j["group"], j["seed"])] = f["dir"]
    tasks = [(a, g, ds) for (a, g), ds in sorted(groups.items())]
    with Pool(workers, initializer=setup, initargs=(build_path, study)) as pool:
        rows = [r for part in pool.imap_unordered(task, tasks) for r in part]
    inst = recurrence(rows, dirs_of)
    rep = {"_fits": len(rows), "_errors": [r for r in rows if "error" in r], "_mono_instances": inst,
           "cells": summarise(rows, inst, study), "_rows": rows}
    for a, cs in rep["cells"].items():
        print(a)
        for c, s in cs.items():
            line = (f"  {c:10s} n {s['fits_converged']:3d}  T1 fail fits {s['V1_would_be']['fits_failing']:3d}/{s['V1_would_be']['fits']}"
                    f" V1 {s['V1_would_be']['vetoed']}  mono/fit {s['mono_per_fit_mean']:.2f} (recurring {s['mono_recurring']})"
                    f"  offdiag child×article {s.get('offdiag_share_child_article') or 0:.4f}"
                    f"  closure art×art {s.get('closure_ratio_article_article') or float('nan'):.3f}"
                    f"  below min ω0 {s['mosaic_below_min_omega0']:.3f}  R/S {s['flattening_R_over_S']:.3f}")
            if study == "A":
                line += f"  existence S {s['existence_S']:.4f}/{s['chance_S']:.4f} G {s['existence_G']:.4f}/{s['chance_G']:.4f}"
            print(line, flush=True)
    print("errors:", len(rep["_errors"]))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True); ap.add_argument("--study", choices=("A", "B"), required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--root", default=ROOT)
    ap.add_argument("--build", default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build = a.build or (D.BUILD if a.study == "A" else BUILD_B)
    rep = run(a.root, a.stage, build, a.study, a.workers)
    json.dump(rep, open(a.out, "w"), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else
              (int(x) if isinstance(x, np.integer) else (float(x) if isinstance(x, np.floating) else str(x))))


if __name__ == "__main__":
    main()
