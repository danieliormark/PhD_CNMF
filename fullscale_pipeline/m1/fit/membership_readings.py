# Item (l), option (a) (owner 2026-11-23: "I approve option a, prepare the readings to avoid this self-circularity problem
# you raised"). Coverage (c = 0.5) stays the only entity filter; beside it, per entity and per facet, for interpretation:
#   R1 membership agreement across seeds (Track A per entity): for each seed pair of one hold-out group, communities matched
#      as Track A/B (measures.match), agreement = 1 − Jensen–Shannon distance (natural log, as Track A) between the
#      entity's two weighted memberships; per entity the mean over the group's seed pairs. Chance: the same with fit 2's
#      rows shuffled per facet (as Track A's chance), so every entity has a chance mean over the same number of pairs.
#   R2 co-membership agreement across seeds (alignment-free; consensus-matrix idea, Monti et al. 2003, Lancichinetti &
#      Fortunato 2012): co-membership c_ij = Σ_k P_ik P_jk (chance that i and j fall in the same community if each draws
#      one from its own membership); per entity and seed pair, the Pearson correlation between its co-memberships with
#      every entity that has a visible tie (each facet's entities weighted 1 / that facet's count, so each facet has one
#      equal vote; self excluded), in seed s and seed t. Computed exactly through K × K sums. Needs no matching of
#      communities, so a split / merge between seeds does not read as instability of the entity. Chance: fit t shuffled.
#   R3 placement without a seed-variability term: per fit, excess = reproduced share Σmin(x̂, x)/Σx minus the degree-only
#      reference (x̂₀ = row total × column total / total); per entity the mean over seeds, the share of seeds with excess
#      > 0; label "placed" = mean excess > 0. The 2026-11-07 placed rule (margin = the entity's seed SD) kept beside it.
#   R4 circularity check (data splitting, Cox 1975; selection independent of the measured quantity, Kriegeskorte et al.
#      2009): seeds of a group split into two halves (by seed number); placement from one half, agreement (R1, R2) from
#      the seed pairs inside the other half; both directions averaged. Same split read three ways: cross-fitted (above),
#      same seeds (placement and agreement from all seeds, no margin), placed rule (margin, all seeds). Track A of placed
#      vs unplaced entities, chance-adjusted (real − chance)/(1 − chance), per facet and tie band.
#   R5 flags for inspection / text check: an entity's R1 (R2) agreement is "within the chance range" if it is at or below
#      the 95th percentile of the chance means of entities of the same facet and tie band. Shares per facet and the
#      2 × 2 of placed × beyond-chance; R1 within chance but R2 beyond chance = a matching artefact rather than an unstable
#      entity.
# Entities: covered (group-level, RL-345) and with ≥ 1 visible tie. Saved fits only. Cells: the pass-3 candidates of
# pass3_choice_{A,B}.json ("chosen"). Per-entity arrays go to RDS (m1_fits/membership_readings/); summaries to --out.
# r sweep (owner 2026-11-23: "I approve making this short test"; RL-357): --set r_sweep reads every (approach, K, r) of
# study A — pass 3 (r 0 / 0.3 / 1) and the λ calibration (three more r per K) — to separate the word-similarity term's
# strength r from K in the child hyperedges' instability; membership spread (bits, mean over seeds) is added per entity and
# per tie band, to check whether the tie-count differences in agreement follow from how concentrated memberships are.
# Usage: python -m fit.membership_readings --out fit/results/stage0/membership_readings.json [--workers 8]
#        python -m fit.membership_readings --set r_sweep --out fit/results/stage0/membership_readings_r_sweep.json --workers 8
#        [--only A:1:5:0.3:0] (one study:approach:K:r:group, for a test)
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
import numpy as np
from . import measures as M
from .aggregate import load_fit, load_UZ, latest_fits, load_entity, _rds_of, COVERAGE_C

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
STAGES = {"A": "stage0_pass3", "B": "B_stage0_pass3"}
CHOICE = "fit/results/stage0/pass3_choice_{}.json"
OUTDIR = os.path.join(ROOT, "membership_readings")
BANDS = M.TIE_BANDS


def band_masks(ties):
    return {name: (ties >= lo) & (ties <= hi) for lo, hi, name in BANDS}


def comembership_corr(Ps, Pt, live):
    """Per facet and entity: weighted Pearson correlation between its co-memberships with every live entity in fit s and
    in fit t (weights 1/n_live per facet, self excluded). Ps, Pt: {facet: P}; live: {facet: bool}."""
    K = next(iter(Ps.values())).shape[1]
    a_s = np.zeros(K); a_t = np.zeros(K); Bss = np.zeros((K, K)); Btt = np.zeros((K, K)); Bst = np.zeros((K, K)); W = 0.0
    wf = {}
    for g in Ps:
        n = int(live[g].sum())
        if n == 0:
            wf[g] = 0.0; continue
        w = 1.0 / n; wf[g] = w; W += 1.0
        S, T = Ps[g][live[g]], Pt[g][live[g]]
        a_s += w * S.sum(0); a_t += w * T.sum(0)
        Bss += w * S.T @ S; Btt += w * T.T @ T; Bst += w * S.T @ T
    out = {}
    for f in Ps:
        x, y = Ps[f], Pt[f]
        Sx = x @ a_s; Sy = y @ a_t
        Sxx = np.einsum("ik,kl,il->i", x, Bss, x); Syy = np.einsum("ik,kl,il->i", y, Btt, y)
        Sxy = np.einsum("ik,kl,il->i", x, Bst, y)
        Wi = np.full(len(x), W)
        sx = (x * x).sum(1); sy = (y * y).sum(1); w = wf[f] * live[f]
        Sx = Sx - w * sx; Sy = Sy - w * sy; Sxx = Sxx - w * sx ** 2; Syy = Syy - w * sy ** 2; Sxy = Sxy - w * sx * sy
        Wi = Wi - w
        cov = Sxy / Wi - Sx * Sy / Wi ** 2
        vx = Sxx / Wi - (Sx / Wi) ** 2; vy = Syy / Wi - (Sy / Wi) ** 2
        out[f] = cov / np.sqrt(np.maximum(vx * vy, 1e-300))
    return out


def task(args):
    study, a, K, r, g, dirs = args
    fits = sorted([f for f in (load_fit(d) for d in dirs) if f["result"]["converged"]], key=lambda f: f["job"]["seed"])
    n = len(fits)
    ents = [load_entity(f["dir"]) for f in fits]
    facets = sorted(ents[0])
    rel = {k: tuple(v) for k, v in fits[0]["result"]["relations"].items()}
    UZ = [load_UZ(f["dir"]) for f in fits]
    wms = [M.weighted_memberships(U, Z, rel) for U, Z in UZ]
    spread = {fa: np.mean([M.entropy_bits(w[fa]["P"]) for w in wms], 0) for fa in facets}
    covs = [M.entity_coverage(U, Z, _rds_of(f["job"])) for (U, Z), f in zip(UZ, fits)]
    del UZ
    covered = M.covered_entities(covs, COVERAGE_C)
    ties = {fa: ents[0][fa]["ties"].astype(int) for fa in facets}
    live = {fa: ties[fa] > 0 for fa in facets}
    keep = {fa: covered[fa] & live[fa] for fa in facets}
    exc = {fa: np.vstack([e[fa]["reproduced_share"] - e[fa]["reproduced_share_degree_only"] for e in ents]) for fa in facets}
    placed_rule = {fa: M.placed_entities([e[fa]["reproduced_share"] for e in ents], ents[0][fa]["reproduced_share_degree_only"])
                   for fa in facets}
    pairs = list(combinations(range(n), 2))
    rng = np.random.default_rng(1000 * g + K)
    R1 = {fa: np.zeros((len(pairs), len(exc[fa][0])), np.float32) for fa in facets}
    R1c, R2, R2c = ({fa: np.zeros_like(R1[fa]) for fa in facets} for _ in range(3))
    for p, (i, j) in enumerate(pairs):
        perm = M.match(wms[i], wms[j], facets)
        sh = M._shuffled(wms[j], rng)
        perm_c = M.match(wms[i], sh, facets)
        Pi = {fa: wms[i][fa]["P"] for fa in facets}
        for fa in facets:
            R1[fa][p] = 1 - M._js_distance_e(Pi[fa], wms[j][fa]["P"][:, perm])
            R1c[fa][p] = 1 - M._js_distance_e(Pi[fa], sh[fa]["P"][:, perm_c])
        c = comembership_corr(Pi, {fa: wms[j][fa]["P"] for fa in facets}, live)
        cc = comembership_corr(Pi, {fa: sh[fa]["P"] for fa in facets}, live)
        for fa in facets:
            R2[fa][p] = c[fa]; R2c[fa][p] = cc[fa]
    halves = (list(range(0, n // 2)), list(range(n // 2, n)))
    in_half = [np.array([p for p, (i, j) in enumerate(pairs) if i in h and j in h]) for h in halves]
    rep = {"study": study, "approach": a, "K": K, "r": r, "group": g, "n_seeds": n, "facets": {}}
    arrays = {}
    for fa in facets:
        m = keep[fa]; bands = band_masks(ties[fa])
        r1, r1c, r2, r2c = R1[fa].mean(0), R1c[fa].mean(0), R2[fa].mean(0), R2c[fa].mean(0)
        e_mean = exc[fa].mean(0); pos_share = (exc[fa] > 0).mean(0)
        placed_now = e_mean > 0
        # R5 (corrected 2026-11-24, RL-363): an entity is beyond chance if its mean agreement over the seed pairs exceeds
        # its OWN chance mean (the same pairs against shuffled rows). The first version pooled the chance means of a facet
        # and tie band and used their 95th percentile; entities' chance levels differ with how spread their memberships
        # are (a flat membership agrees with anything), so pooling flagged concentrated entities next to flattened ones.
        beyond1 = r1 > r1c; beyond2 = r2 > r2c
        # R4: Track A of placed vs unplaced, three ways
        def split_A(placed, Rr, Rc, mask):
            out = {}
            for lab, sel in (("placed", placed & mask), ("unplaced", ~placed & mask)):
                if sel.sum() == 0:
                    out[lab] = None; continue
                real, ch = float(Rr[sel].mean()), float(Rc[sel].mean())
                out[lab] = {"n": int(sel.sum()), "real": real, "chance": ch, "adjusted": (real - ch) / (1 - ch)}
            return out

        def three_ways(Rr_all, Rc_all, mask):
            cross = []
            for h, other in ((0, 1), (1, 0)):
                pl = exc[fa][halves[h]].mean(0) > 0
                cross.append(split_A(pl, Rr_all[in_half[other]].mean(0), Rc_all[in_half[other]].mean(0), mask))
            cf = {}
            for lab in ("placed", "unplaced"):
                v = [c[lab] for c in cross if c[lab] is not None]
                cf[lab] = None if not v else {k: float(np.mean([x[k] for x in v])) for k in ("n", "real", "chance", "adjusted")}
            return {"cross_fitted": cf, "same_seeds": split_A(placed_now, Rr_all.mean(0), Rc_all.mean(0), mask),
                    "placed_rule": split_A(placed_rule[fa], Rr_all.mean(0), Rc_all.mean(0), mask)}

        split = {"all": {"R1": three_ways(R1[fa], R1c[fa], m), "R2": three_ways(R2[fa], R2c[fa], m)}}
        for b, bm in bands.items():
            if (m & bm).sum() >= 20:
                split[b] = {"R1": three_ways(R1[fa], R1c[fa], m & bm)}
        q = lambda v: [float(x) for x in np.quantile(v, (0.1, 0.25, 0.5, 0.75, 0.9))] if len(v) else None
        rep["facets"][fa] = {
            "n_entities": int(m.sum()), "placed_share": float(placed_now[m].mean()), "placed_rule_share": float(placed_rule[fa][m].mean()),
            "excess_mean": float(e_mean[m].mean()), "excess_quantiles": q(e_mean[m]), "seeds_positive_share_quantiles": q(pos_share[m]),
            "R1": {"mean": float(r1[m].mean()), "chance": float(r1c[m].mean()), "quantiles": q(r1[m]),
                   "adjusted": float((r1[m].mean() - r1c[m].mean()) / (1 - r1c[m].mean())), "beyond_chance_share": float(beyond1[m].mean())},
            "R2": {"mean": float(np.nanmean(r2[m])), "chance": float(np.nanmean(r2c[m])), "quantiles": q(r2[m][np.isfinite(r2[m])]),
                   "beyond_chance_share": float(beyond2[m].mean())},
            "flags_2x2": {f"{'placed' if pl else 'unplaced'}_{'beyond' if bc else 'within'}": float((m & (placed_now == pl) & (beyond1 == bc)).mean() / max(m.mean(), 1e-300))
                          for pl in (True, False) for bc in (True, False)},
            "R1_within_R2_beyond_share": float((m & ~beyond1 & beyond2).sum() / max(m.sum(), 1)),
            "by_band_R1": {b: {"n": int((m & bm).sum()), "mean": float(r1[m & bm].mean()), "chance": float(r1c[m & bm].mean()),
                               "beyond_chance_share": float(beyond1[m & bm].mean()), "placed_share": float(placed_now[m & bm].mean()),
                               "R2_mean": float(np.nanmean(r2[m & bm])), "spread_bits": float(spread[fa][m & bm].mean())}
                           for b, bm in bands.items() if (m & bm).sum() > 0},
            "split": split}
        arrays.update({f"{fa}__{k}": v for k, v in (("R1", r1), ("R1_chance", r1c), ("R2", r2), ("R2_chance", r2c),
                                                    ("excess_mean", e_mean), ("excess_pos_share", pos_share),
                                                    ("coverage_mean", np.nanmean(np.vstack([cv[fa] for cv in covs]), 0) if live[fa].any() else np.full(len(m), np.nan)),
                                                    ("ties", ties[fa]), ("kept", m), ("beyond_R1", beyond1), ("beyond_R2", beyond2),
                                                    ("placed_rule", placed_rule[fa]), ("spread_bits", spread[fa]))})
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUTDIR, f"{study}_a{a}_K{K}_r{r:g}_g{g}.npz"), **arrays)
    return rep


def tasks_for(only=None, which="candidates"):
    out = []
    for study, st in STAGES.items():
        if which == "r_sweep":
            if study != "A":
                continue
            want = None
            fs = latest_fits([load_fit(os.path.dirname(p)) for stg in (st, "lambda_cal")
                              for p in glob.glob(os.path.join(ROOT, stg, "*", "measures.json"))])
            fs = [f for f in fs if not (f["job"].get("arm") or {}).get("swap_p")]
            want = {(f["job"]["approach"], f["job"]["K"], float(f["job"]["arm"]["r"])) for f in fs}
        else:
            ch = json.load(open(CHOICE.format(study)))
            want = {(int(ap.split("_")[1]), c["K"], float(c["r"])) for ap in ("approach_1", "approach_2") for c in ch[ap]["chosen"]}
            fs = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, st, "*", "measures.json"))])
        by = defaultdict(list)
        for f in fs:
            j = f["job"]; r = float((j.get("arm") or {}).get("r", np.nan))
            if (j["approach"], j["K"], r) in want:
                by[(j["approach"], j["K"], r, j["group"])].append(f["dir"])
        out += [(study, a, K, r, g, ds) for (a, K, r, g), ds in sorted(by.items())]
    if only:
        s, a, K, r, g = only.split(":")
        out = [t for t in out if (t[0], t[1], t[2], t[3], t[4]) == (s, int(a), int(K), float(r), int(g))]
    return out


def summarise(res):
    """Per cell: mean and SE over hold-out groups of the facet readings that are numbers."""
    cells = defaultdict(list)
    for r in res:
        cells[f"{r['study']}_approach_{r['approach']}_K{r['K']}_r{r['r']:g}"].append(r)
    out = {}
    for c, rs in sorted(cells.items()):
        def walk(objs):
            if all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in objs):
                v = np.array(objs, float)
                return {"mean": float(np.nanmean(v)), "se": float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())) if np.isfinite(v).sum() > 1 else None}
            if all(isinstance(o, dict) for o in objs):
                keys = set.intersection(*[set(o) for o in objs])
                return {k: walk([o[k] for o in objs]) for k in sorted(keys)}
            if all(isinstance(o, list) for o in objs) and len({len(o) for o in objs}) == 1:
                return [walk([o[i] for o in objs]) for i in range(len(objs[0]))]
            return None
        out[c] = {"groups": len(rs), "facets": walk([r["facets"] for r in rs])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--only")
    ap.add_argument("--set", default="candidates", choices=("candidates", "r_sweep"))
    a = ap.parse_args()
    ts = tasks_for(a.only, getattr(a, "set"))
    print(len(ts), "tasks", flush=True)
    if a.workers > 1:
        with Pool(a.workers) as pool:
            res = pool.map(task, ts, chunksize=1)
    else:
        res = [task(t) for t in ts]
    json.dump({"per_group": res, "summary": summarise(res)}, open(a.out, "w"), indent=1, default=float)
    print("written", a.out)


if __name__ == "__main__":
    main()
