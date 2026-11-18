# Equivalence test of the social log-damping arm (ticket 107; owner 2026-11-18: "shall we check first that rescaling and
# w_r work same way?" and "shall we remeasure the positivity floor and adam ε as well as other parameters, so our work
# won't rely on bare expectations?"). Saved fits only; runs on incline. Plan: plans/ticket107_damping_test.md.
# Arms (fit/plan.py damping_test): weight (B: damped data at Frobenius norm 1, w_r × q), data_matched (A: damped data ×
# q, start = B's start with damped cores × q), data (A from the ordinary start). KL, K = 13, group 1, seeds 1–3.
# Comparisons, per approach and seed (all readings use each fit's own numbering; same start → same numbering unless the
# fits drift apart, which the reconstruction reading shows without any numbering):
#   (a) data_matched vs weight — the implementations (expected equal up to q);
#   (b) data vs data_matched   — the start's scale only (the start of the social core matrices is not scaled to data);
#   (c) yardstick: data arm, seed vs seed — the ordinary seed-to-seed spread of each reading;
#   (d) preview, reporting only: data arm vs the undamped start-test fit of the same seed (what the damping changes).
# Readings: final total loss (A's total = B's total if equivalent), epochs, first epoch at which the two total-loss
# paths differ by more than 1e-4 (relative; the stopping threshold); per relation loss_A vs q·loss_B;
# reconstruction on visible ties x̂_A vs q·x̂_B (relative L2 difference; numbering-free); diagonal-rule share per social
# relation and child×article failing communities; weighted memberships (overlap Σ_k min(p_k, p'_k), mean over entities;
# for B both as fitted and with its damped cores × q, i.e. on A's scale); fixed-size settings per fit: share of core
# cells / factor entries at the positivity floor (1e-7), gradient size per parameter block at the end against Adam's
# ε (1e-8), and the smallest block size relative to its start (step-size floor binds below 1e-3).
# Reading: ratio = median difference in (a) ÷ median seed-to-seed difference (c): < 0.1 equivalent in practice;
# 0.1–0.5 drift (check the rules and memberships); > 0.5 not equivalent → find which fixed-size setting causes it.
# Usage: python -m fit.damping_test_analysis --out fit/results/stage0/damping_test.json
import argparse, glob, json, os
from collections import defaultdict
from itertools import combinations
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
FLOOR, EPS = 1e-7, 1e-8
ARMS = {"weight": {"log_damping": "weight"}, "data_matched": {"log_damping": "data", "damping_start": "matched"},
        "data": {"log_damping": "data"}}


def arm_label(job):
    a = job.get("arm") or {}
    return next((k for k, v in ARMS.items() if a == v), None)


def collect():
    """{(approach, arm): {seed: fit}}, arm also 'undamped' (start-test KL fits, K 13, group 1)."""
    out = defaultdict(dict)
    for stage in ("damping_test", "start_test"):
        fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, stage, "*", "measures.json"))]
        for f in latest_fits(fs):
            j = f["job"]
            if j["group"] != 0 or j["beta"] != 1 or j["K"] != 13 or j["seed"] not in (1, 2, 3) or (j.get("terms") or {}):
                continue
            lab = "undamped" if stage == "start_test" and not (j.get("arm") or {}) else arm_label(j)
            if lab:
                out[(j["approach"], lab)][j["seed"]] = f
    return out


def xhat(U, Z, rd):
    return ((U[rd.row_facet][rd.tie_row] @ Z[rd.name]) * U[rd.col_facet][rd.tie_col]).sum(1)


def rule_share(P):
    off = P.copy(); np.fill_diagonal(off, -np.inf); d = np.diag(P)
    ok = (d > off.max(1)) & (d > off.max(0)); h = 0.5 * (P.sum(1) + P.sum(0)) / P.sum()
    return float(h[ok].sum())


def gradient_rms(U, Z, rds):
    """RMS of the training-loss gradient per parameter block at the saved solution (float64), and the share of entries
    whose |gradient| is below 10 ε (there Adam's ε is at least a tenth of the step's denominator)."""
    import torch
    from .loss import RelationTensors, total_loss
    Ut = {f: torch.tensor(u, requires_grad=True) for f, u in U.items()}
    Zt = {r: torch.tensor(z, requires_grad=True) for r, z in Z.items()}
    tot, _ = total_loss(Ut, Zt, [RelationTensors(rd) for rd in rds], 1)
    tot.backward()
    out = {}
    for n, t in [(f"U:{f}", Ut[f]) for f in sorted(Ut)] + [(f"Z:{r}", Zt[r]) for r in sorted(Zt)]:
        g = t.grad.numpy()
        out[n] = {"rms": float(np.sqrt((g ** 2).mean())), "share_below_10eps": float((np.abs(g) < 10 * EPS).mean())}
    return out


def per_fit(f, rds):
    U, Z = load_UZ(f["dir"]); res = f["result"]
    rel = {k: tuple(v) for k, v in res["relations"].items()}
    wm = M.weighted_memberships(U, Z, rel)
    out = {"converged": res["converged"], "epochs": res["epochs"], "total": res["loss_history"][-1], "history": res["loss_history"],
           "relation_loss": res["final_relation_loss"], "xhat": {rd.name: xhat(U, Z, rd) for rd in rds},
           "P": {x: v["P"] for x, v in wm.items()}, "U": U, "Z": Z, "rel": rel}
    rule = {}
    for rd in rds:
        P, _ = M.parts(U, Z, rd)
        if rd.name == "child_article":
            d = np.diag(P)
            ok = (d / np.maximum(P.sum(1), 1e-300) > 0.5) & (d / np.maximum(P.sum(0), 1e-300) > 0.5)
            out["anchor_failing"] = int((~ok).sum())
        elif rd.domain == "social":
            rule[rd.name] = rule_share(P)
    out["diagonal_rule"] = rule
    out["floor"] = {"core": {r: float((z <= FLOOR * 1.0001).mean()) for r, z in Z.items()},
                    "factor": {x: float((u <= FLOOR * 1.0001).mean()) for x, u in U.items()}}
    out["gradient"] = gradient_rms(U, Z, rds)
    logs = res.get("logs") or []
    if len(logs) >= 2:
        first, last = logs[0]["block_rms"], logs[-1]["block_rms"]
        out["block_size_vs_start_min"] = float(min(last[k] / first[k] for k in first if first[k] > 0))
    out["heldout_existence"] = f["measures"]["heldout_two_part_combined"]["existence"]
    return out


def overlap(P, Q):
    return float(np.minimum(P, Q).sum(1).mean())


def compare(x, y, q):
    """Differences of fit x against fit y; y's reconstruction and relation losses are multiplied by q (relation → q)."""
    h1, h2 = np.array(x["history"]), np.array(y["history"]); n = min(len(h1), len(h2))
    dev = np.nonzero(np.abs(h1[:n] - h2[:n]) / np.abs(h2[:n]) > 1e-4)[0]
    r = {"total_rel": abs(x["total"] - y["total"]) / abs(y["total"]), "epochs": x["epochs"] - y["epochs"],
         "first_epoch_total_differs_1e-4": float(dev[0]) if len(dev) else float(n),
         "relation_loss_rel": {}, "xhat_rel": {}, "rule_abs": {}, "membership_overlap": {}}
    for n in x["xhat"]:
        a, b = x["xhat"][n], q.get(n, 1.0) * y["xhat"][n]
        r["xhat_rel"][n] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        la, lb = x["relation_loss"][n], q.get(n, 1.0) * y["relation_loss"][n]
        r["relation_loss_rel"][n] = abs(la - lb) / abs(lb)
    for n in x["diagonal_rule"]:
        r["rule_abs"][n] = abs(x["diagonal_rule"][n] - y["diagonal_rule"][n])
    r["anchor_failing"] = (x["anchor_failing"], y["anchor_failing"])
    r["membership_overlap"] = {fa: overlap(x["P"][fa], y["P"][fa]) for fa in x["P"]}
    if any(v != 1.0 for v in q.values()):        # y read on x's scale: y's damped cores × q
        Zq = {n: z * q.get(n, 1.0) for n, z in y["Z"].items()}
        Pq = M.weighted_memberships(y["U"], Zq, y["rel"])
        r["membership_overlap_on_A_scale"] = {fa: overlap(x["P"][fa], Pq[fa]["P"]) for fa in x["P"]}
    return r


def summarise(rows):
    """Median over pairs of every numeric reading (nested by relation / facet)."""
    if not rows:
        return None
    out = {}
    for k, v in rows[0].items():
        if isinstance(v, dict):
            out[k] = {n: float(np.median([r[k][n] for r in rows])) for n in v}
        elif isinstance(v, tuple):
            out[k] = [r[k] for r in rows]
        else:
            out[k] = float(np.median([r[k] for r in rows]))
    return out


def ratio(a, c):
    """Ratio of (a)'s median difference to the seed-to-seed median, per reading that has both."""
    out = {}
    for k in ("total_rel", "xhat_rel", "relation_loss_rel", "rule_abs"):
        if k not in a or k not in c:
            continue
        if isinstance(a[k], dict):
            out[k] = {n: (a[k][n] / c[k][n] if c[k][n] > 0 else None) for n in a[k]}
        else:
            out[k] = a[k] / c[k] if c[k] > 0 else None
    return out


def verdict(rt):
    vals = [v for v in (list(rt.get("xhat_rel", {}).values()) + [rt.get("total_rel")]) if v is not None]
    if not vals:
        return "not readable"
    m = max(vals)
    return "equivalent in practice" if m < 0.1 else ("drift: check rules and memberships" if m <= 0.5 else
                                                     "not equivalent: find the fixed-size setting")


def run():
    fits = collect(); build = D.Build(D.BUILD); rep = {}
    for a in (1, 2):
        rds = {"data": D.fit_data(build, a, group=0, log_damping="data"),
               "weight": D.fit_data(build, a, group=0, log_damping="weight"), "undamped": D.fit_data(build, a, group=0)}
        q = {rd.name: rd.damping["q"] for rd in rds["data"] if rd.damping}
        R = {lab: {s: per_fit(f, rds["weight" if lab == "weight" else ("undamped" if lab == "undamped" else "data")])
                   for s, f in fits.get((a, lab), {}).items()} for lab in ("weight", "data_matched", "data", "undamped")}
        seeds = sorted(set(R["weight"]) & set(R["data_matched"]))
        A = [compare(R["data_matched"][s], R["weight"][s], q) for s in seeds]
        Bs = [compare(R["data"][s], R["data_matched"][s], {}) for s in sorted(set(R["data"]) & set(R["data_matched"]))]
        C = [compare(R["data"][s], R["data"][t], {}) for s, t in combinations(sorted(R["data"]), 2)]
        prev = {s: {"diagonal_rule": {"damped": R["data"][s]["diagonal_rule"], "undamped": R["undamped"][s]["diagonal_rule"]},
                    "anchor_failing": {"damped": R["data"][s]["anchor_failing"], "undamped": R["undamped"][s]["anchor_failing"]},
                    "heldout_existence": {"damped": R["data"][s]["heldout_existence"], "undamped": R["undamped"][s]["heldout_existence"]}}
                for s in sorted(set(R["data"]) & set(R["undamped"]))}
        sa, sb, sc = summarise(A), summarise(Bs), summarise(C)
        ra = ratio(sa, sc) if sa and sc else None; rb = ratio(sb, sc) if sb and sc else None
        fixed = {lab: {s: {"converged": x["converged"], "epochs": x["epochs"], "floor": x["floor"], "gradient": x["gradient"],
                           "block_size_vs_start_min": x.get("block_size_vs_start_min")} for s, x in R[lab].items()}
                 for lab in ("weight", "data_matched", "data")}
        name = f"approach_{a}"
        rep[name] = {"q": q, "fits": {lab: len(R[lab]) for lab in R},
                     "a_implementations": {"median": sa, "ratio_to_seed_spread": ra, "verdict": verdict(ra) if ra else "not readable"},
                     "b_start_scale": {"median": sb, "ratio_to_seed_spread": rb, "verdict": verdict(rb) if rb else "not readable"},
                     "c_seed_spread": sc, "d_preview": prev, "fixed_size_settings": fixed}
        print(name, "q", {k: round(v, 3) for k, v in q.items()}, flush=True)
        print("  (a) implementations:", rep[name]["a_implementations"]["verdict"], "| (b) start scale:",
              rep[name]["b_start_scale"]["verdict"], flush=True)
        for lab, d in (("(a)", sa), ("(b)", sb), ("(c) seeds", sc)):
            if d:
                print(f"  {lab:10s} total {d['total_rel']:.2e}  x̂ " + " ".join(f"{n} {v:.2e}" for n, v in d["xhat_rel"].items())
                      + "  rule " + " ".join(f"{n} {v:.3f}" for n, v in d["rule_abs"].items())
                      + "  overlap " + " ".join(f"{n} {v:.3f}" for n, v in d["membership_overlap"].items()), flush=True)
                if "membership_overlap_on_A_scale" in d:
                    print("             overlap on A's scale " + " ".join(f"{n} {v:.3f}" for n, v in d["membership_overlap_on_A_scale"].items()), flush=True)
        for lab, fx in fixed.items():
            for s, v in fx.items():
                print(f"  {lab:12s} seed {s}: conv {v['converged']} ep {v['epochs']}  core cells at floor "
                      + " ".join(f"{n} {x:.2f}" for n, x in v["floor"]["core"].items())
                      + "  blocks with ≥1% |grad|<10ε " + str(sum(g["share_below_10eps"] >= 0.01 for g in v["gradient"].values()))
                      + f"  min block size/start {v['block_size_vs_start_min']}", flush=True)
    return rep


def _np(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1, default=_np)


if __name__ == "__main__":
    main()
