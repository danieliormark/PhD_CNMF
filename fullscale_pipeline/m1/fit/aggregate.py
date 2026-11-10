# F8 (3/3) — aggregation of finished M1 fits (ticket 107). Implements the decided selection rules:
#   cell        = one setting at one K (stage, approach, arm, loss, start, fraction, dtype, terms) over hold-out groups × seeds
#   per group   : placed entities (seed variation, 2026-11-07); per seed: stability gate (Track A/B on weighted memberships,
#                 Track A over placed entities, per seed vs the maximum of its pairs' shuffled chance, 2026-11-09 (C1, E));
#                 ghost veto (no placed entity has k as largest membership); alignment veto (child×article alignment over
#                 placed entities not above its stratified shuffle mean by more than the group's seed SD of the observed
#                 alignment — the decided "not above the stratified shuffle reference beyond seed noise"); convergence
#                 (fourth check, option c); non-converged seeds are left out of the other three checks
#   counting    : a group fails a check if > fraction of its seeds fail; K fails if > fraction of groups fail or > fraction
#                 of all fits fail (fraction 0.2 provisional; also reported at 0.1/0.25/0.5) (2026-11-09)
#   held-out    : a K's score = mean over groups of each group's mean over converged seeds (the loss's own form; the
#                 lowest-training-loss seed's score reported beside it); SE = SD of the group means / √groups; band = every
#                 K with score ≤ best + SE of the best K (2026-11-03)
#   K protocol  : band from all K → exclude K failing the gate, a veto (and convergence → owner, option c) → candidates
#                 (inspection and the text check choose among them) → if none: the largest K below the band (smaller K)
#                 passing everything, flagged → if none: "no K qualifies" with the status of every K above the band,
#                 owner decides (§6.8, 2026-11-09)
#   between     : stability between hold-out groups (lowest-loss converged seed of each group, pairwise, chance-adjusted)
# Usage: python -m fit.aggregate --root ROOT --stage STAGE [--fraction 0.2]
import argparse, json, os
from collections import defaultdict
import numpy as np

from . import measures as M

FRACTIONS = (0.1, 0.2, 0.25, 0.5)
CELL_KEYS = ("stage", "approach", "arm", "beta", "start", "frac", "dtype", "terms", "lr_mode", "max_epochs")


def cell_key(job):
    return json.dumps({k: job.get(k) for k in CELL_KEYS}, sort_keys=True)


def load_fit(d):
    spec = json.load(open(os.path.join(d, "spec.json"))); res = json.load(open(os.path.join(d, "result.json")))
    meas = json.load(open(os.path.join(d, "measures.json")))
    return {"dir": d, "job": spec["job"], "result": res, "measures": meas}


def load_UZ(d):
    z = np.load(os.path.join(d, "UZ.npz"))
    U = {k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("U__")}
    Z = {k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("Z__")}
    return U, Z


def load_entity(d):
    z = np.load(os.path.join(d, "entity.npz"))
    out = defaultdict(dict)
    for k in z.files:
        f, m = k.split("__", 1)
        out[f][m] = z[k]
    return out


# ---------------------------------------------------------------------------------------------- K protocol (pure)
def choose_k(cells):
    """cells: {K: {"score": mean held-out score (lower better), "se": SE of that K's group means,
    "gate_fails": bool, "ghost_fails": bool, "alignment_fails": bool, "convergence_fails": bool}}.
    Returns the band, the candidates, the fallback and flags (§6.8 protocol with step-11 vetoes)."""
    Ks = sorted(cells)
    best = min(Ks, key=lambda k: cells[k]["score"])
    cut = cells[best]["score"] + cells[best]["se"]
    band = [k for k in Ks if cells[k]["score"] <= cut]
    fails = lambda k, conv=True: (cells[k]["gate_fails"] or cells[k]["ghost_fails"] or cells[k]["alignment_fails"]
                                  or (conv and cells[k]["convergence_fails"]))
    cand = [k for k in band if not fails(k)]
    owner = [k for k in band if fails(k) and not fails(k, conv=False)]       # excluded by non-convergence alone
    out = {"best": best, "cutoff": cut, "band": band, "candidates": cand, "convergence_only_to_owner": owner,
           "fallback": None, "no_K_qualifies": False, "above_band_status": None}
    if not cand:
        below = [k for k in Ks if k < min(band) and not fails(k)]
        if below:
            out["fallback"] = max(below)
        else:
            out["no_K_qualifies"] = True
            out["above_band_status"] = {k: {"passes_all": not fails(k)} for k in Ks if k > max(band)}
    return out


# ------------------------------------------------------------------------------------------- one cell, one group
def group_checks(fits, rds_child_article=None, W_has_tie=None, seed=0):
    """fits: the seeds of one cell and one hold-out group (loaded with load_fit). Returns per-seed check results."""
    conv = [bool(f["result"]["converged"]) for f in fits]
    ok = [i for i, c in enumerate(conv) if c]
    out = {"seeds": [f["job"]["seed"] for f in fits], "convergence_fails": [not c for c in conv],
           "gate_fails": [None] * len(fits), "ghost_fails": [None] * len(fits), "alignment_fails": [None] * len(fits)}
    if len(ok) < 2:
        return out, None
    ents = [load_entity(fits[i]["dir"]) for i in ok]
    placed = {f: M.placed_entities([e[f]["reproduced_share"] for e in ents], ents[0][f]["reproduced_share_degree_only"])
              for f in ents[0]}
    UZ = [load_UZ(fits[i]["dir"]) for i in ok]
    rel = {r: tuple(v) for r, v in fits[ok[0]]["result"]["relations"].items()}
    wms = [M.weighted_memberships(U, Z, rel) for U, Z in UZ]
    st = M.stability(wms, placed, seed=seed)
    for j, i in enumerate(ok):
        out["gate_fails"][i] = not st["per_seed"][j]["passes"]
        out["ghost_fails"][i] = len(M.ghosts(wms[j], placed)) > 0
    if rds_child_article is not None:
        al = [M.tie_alignment(rds_child_article, wm["article"]["P"], wm["child"]["P"], placed["article"], placed["child"],
                              strata_extra=W_has_tie, n_shuffles=99, seed=seed) for wm in wms]
        obs = np.array([a["observed"] for a in al]); sd = obs.std(ddof=1)
        for j, i in enumerate(ok):
            out["alignment_fails"][i] = not (al[j]["observed"] - al[j]["shuffled_mean"] > sd)
        out["alignment"] = [{"observed": a["observed"], "shuffled_mean": a["shuffled_mean"], "adjusted": a["adjusted"]} for a in al]
    out["stability"] = {k: v for k, v in st.items() if k != "per_seed"}; out["stability_per_seed"] = st["per_seed"]
    out["placed_share"] = {f: float(m.mean()) for f, m in placed.items()}
    return out, (wms, placed, ok)


def count(cell_groups, key, fraction):
    """The counting rule over groups whose seeds may differ in number (non-converged seeds are not evaluated by the
    gate and vetoes): a group fails if > fraction of its evaluated seeds fail; K fails if > fraction of the groups fail
    or > fraction of all evaluated fits fail (2026-11-09, A–C)."""
    per = [[bool(x) for x in g[key] if x is not None] for g in cell_groups]
    per = [p for p in per if p]
    if not per:
        return None
    g_fail = [np.mean(p) > fraction for p in per]
    allv = np.concatenate([np.array(p, bool) for p in per])
    return {"group_fails": g_fail, "K_fails": bool(np.mean(g_fail) > fraction or allv.mean() > fraction),
            "by_groups": bool(np.mean(g_fail) > fraction), "pooled": bool(allv.mean() > fraction),
            "failures_per_group": [int(sum(p)) for p in per], "evaluated_per_group": [len(p) for p in per],
            "groups_with_any_failure": [i for i, p in enumerate(per) if any(p)]}


def aggregate(root, stage, fraction=0.2, build_path=None):
    from . import data as D
    import scipy.sparse as sp
    base = os.path.join(root, stage)
    fits = [load_fit(os.path.join(base, d)) for d in sorted(os.listdir(base))
            if os.path.exists(os.path.join(base, d, "measures.json"))]
    build = D.Build(build_path or D.BUILD)
    W = sp.load_npz(build.path + "W_child_child.npz")
    W_has_tie = (np.diff(W.tocsr().indptr) > 0).astype(int)
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))     # cell → K → group → fits
    for f in fits:
        cells[cell_key(f["job"])][f["job"]["K"]][f["job"]["group"]].append(f)
    report = {}
    ca_cache = {}
    for ck, byK in cells.items():
        cinfo = json.loads(ck); beta = cinfo["beta"]
        score_key = "squared_error" if beta == 2 else "kl_error"
        perK = {}
        for K, byG in sorted(byK.items()):
            groups, scores, best_seed_scores, between_fits = [], [], [], []
            for g, fs in sorted(byG.items(), key=lambda x: (x[0] is None, x[0])):
                fs = sorted(fs, key=lambda f: f["job"]["seed"])
                if (cinfo["approach"], g) not in ca_cache:
                    ca_cache[(cinfo["approach"], g)] = next(rd for rd in D.fit_data(build, cinfo["approach"], group=g)
                                                            if rd.name == "child_article")
                gc, extra = group_checks(fs, ca_cache[(cinfo["approach"], g)], W_has_tie)
                gc["group"] = g; groups.append(gc)
                conv = [f for f in fs if f["result"]["converged"]]
                if conv:
                    scores.append(np.mean([f["measures"]["heldout_combined"][score_key] for f in conv]))
                    bestf = min(conv, key=lambda f: f["result"]["data_loss_history"][-1])
                    best_seed_scores.append(bestf["measures"]["heldout_combined"][score_key])
                    between_fits.append(bestf)
            G = len(scores)
            entry = {"groups": groups, "n_groups_scored": G,
                     "score": float(np.mean(scores)) if G else None,
                     "se": float(np.std(scores, ddof=1) / np.sqrt(G)) if G > 1 else None,
                     "score_lowest_loss_seed": float(np.mean(best_seed_scores)) if G else None,
                     "epochs_median": float(np.median([f["result"]["epochs"] for g in byG.values() for f in g])),
                     "seconds_median": float(np.median([f["result"]["seconds"] for g in byG.values() for f in g]))}
            for chk in ("gate_fails", "ghost_fails", "alignment_fails", "convergence_fails"):
                entry[chk] = {str(fr): count(groups, chk, fr) for fr in FRACTIONS}
            if len(between_fits) > 1:
                wmsb = []
                for f in between_fits:
                    U, Z = load_UZ(f["dir"]); wmsb.append(M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()}))
                sb = M.stability(wmsb)
                entry["between_groups"] = {k: v for k, v in sb.items() if k != "per_seed"}
            perK[K] = entry
        scored = {K: e for K, e in perK.items() if e["score"] is not None and e["se"] is not None}
        sel = None
        if len(scored) >= 2:
            fr = str(fraction)
            sel = choose_k({K: {"score": e["score"], "se": e["se"],
                                **{c: bool(e[c][fr]["K_fails"]) if e[c][fr] else False
                                   for c in ("gate_fails", "ghost_fails", "alignment_fails", "convergence_fails")}}
                            for K, e in scored.items()})
        report[ck] = {"cell": cinfo, "per_K": perK, "selection": sel}
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--stage", required=True)
    ap.add_argument("--fraction", type=float, default=0.2); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = aggregate(a.root, a.stage, a.fraction)
    out = a.out or os.path.join(a.root, a.stage, "_aggregate.json")
    json.dump(rep, open(out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    print(f"wrote {out} ({len(rep)} cells)")


if __name__ == "__main__":
    main()
