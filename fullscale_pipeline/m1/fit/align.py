# Alignment of community numbering across facets (ticket 107; DECIDED 2026-11-17, owner: "I approve the renumbering",
# "I agree with proposal a"). Read time only: saved fits are never modified (CLAUDE.md §4.20 convention).
# With free core matrices each facet's numbering is free (GLOSSARY "Numbering freedom"; RL-259): renumbering facet f by
# a permutation and reordering every core matrix touching f on f's side leaves every reconstructed value unchanged.
# Alignment fixes one numbering for all facets: starting from articles (reference), each facet is matched to an already
# aligned facet it shares a relation with, by a one-to-one assignment (Hungarian) maximising the tie weight on matching
# numbers (parts, measures.parts). Order (both approaches): children via child×article, parents via child×parent, then
# the social facets (authors via author×article; organisations via author×affiliation or article×affiliation; journals
# via article×journal). Option (a): always applied; the match strength (share of each relation's tie weight on matching
# numbers after alignment) is reported beside every reading.
# Per fit, written next to the fit on RDS as aligned.json: permutations per facet, match strength per relation (as fitted
# and aligned), and the presence readings recomputed under the aligned numbering (presence per relation, domain and facet
# presence, mono flags).
# Usage: python -m fit.align --stages stage0_pass1 swap_bare start_test
import argparse, glob, json, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import measures as M

REFERENCE = "article"


def align(U, Z, rds):
    """Returns aligned U, Z, permutations {facet: perm} (new column k = fitted column perm[k]) and the order used."""
    U = dict(U); Z = dict(Z)
    K = next(iter(U.values())).shape[1]
    perms = {REFERENCE: np.arange(K)}
    order = []
    pending = True
    while pending:
        pending = False
        for rd in sorted(rds, key=lambda r: (0 if r.domain == "semantic" else 1, r.name)):
            a, b = rd.row_facet, rd.col_facet
            if a == b or ((a in perms) == (b in perms)):
                continue
            new = b if a in perms else a
            P, _ = M.parts(U, Z, rd)
            Q = P if new == b else P.T                       # rows: aligned facet's communities; columns: the new facet's
            r, c = linear_sum_assignment(-Q)
            perm = c[np.argsort(r)]
            U[new] = U[new][:, perm]
            for rd2 in rds:
                if rd2.row_facet == new:
                    Z[rd2.name] = Z[rd2.name][perm, :]
                if rd2.col_facet == new:
                    Z[rd2.name] = Z[rd2.name][:, perm]
            perms[new] = perm; order.append((new, rd.name)); pending = True
            break                                            # recompute after each step (parts depend on earlier steps)
    return U, Z, perms, order


def apply(U, Z, relations, perms):
    """Apply stored permutations (from aligned.json) to a fit's U, Z; relations: {name: (row facet, column facet)}."""
    U2 = {f: (u[:, np.asarray(perms[f])] if f in perms else u) for f, u in U.items()}
    Z2 = {}
    for r, z in Z.items():
        a, b = relations[r]
        if a in perms:
            z = z[np.asarray(perms[a]), :]
        if b in perms:
            z = z[:, np.asarray(perms[b])]
        Z2[r] = z
    return U2, Z2


def match_strength(U, Z, rds):
    out = {}
    for rd in rds:
        P, T = M.parts(U, Z, rd)
        out[rd.name] = float(np.trace(P) / T) if T > 0 else None
    return out


def aligned_readings(U, Z, rds):
    tolist = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    Ua, Za, perms, order = align(U, Z, rds)
    pres = M.presence(Ua, Za, rds)
    dom = M.domain_presence(pres); fac = M.facet_presence(pres)
    return {"perms": {f: p.tolist() for f, p in perms.items()}, "order": order,
            "moved_share": {f: float((p != np.arange(len(p))).mean()) for f, p in perms.items()},
            "match_strength_fitted": match_strength(U, Z, rds), "match_strength_aligned": match_strength(Ua, Za, rds),
            "presence": {r: {k: tolist(v[k]) for k in ("half_credit", "presence", "row_side", "col_side", "within_over_half_credit")}
                         | {"domain": v["domain"], "row_facet": v["row_facet"], "col_facet": v["col_facet"]} for r, v in pres.items()},
            "domain_presence": {k: ({kk: tolist(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in dom.items()},
            "facet_presence": {f: v.tolist() for f, v in fac.items()},
            "mono_flags": M.mono_flags(dom, fac)}


def fit_rds(job, cache):
    """The fit's relation data (with its swap, if any), as run_fit built them."""
    from . import data as D
    from .worlds import graded_swap
    arm = job.get("arm") or {}
    key = (job["approach"], job["group"], json.dumps({k: v for k, v in arm.items() if not k.startswith("swap")}, sort_keys=True))
    if key not in cache:
        cache.clear()
        build = D.Build(job.get("build", D.BUILD))
        cache[key] = D.fit_data(build, job["approach"], group=job["group"], journal_multiplier=arm.get("journal_multiplier", 1.0),
                                semantic_split=arm.get("semantic_split"), drop=tuple(arm.get("drop", ())),
                                use_journal=arm.get("use_journal", True), binary=bool(arm.get("binary", False)),
                                log_damping=arm.get("log_damping"), mass_weights=bool(arm.get("mass_weights", False)))
    rds = cache[key]
    if arm.get("swap_p"):
        rds = graded_swap(rds, float(arm["swap_p"]), seed=job["seed"], single_band=bool(arm.get("swap_single_band")))[0]
    return rds


def main():
    from .aggregate import load_UZ
    ap = argparse.ArgumentParser(); ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--root", default="/mnt/hum01-rds/Basov/p91688di/m1_fits"); ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    dirs = []
    for st in a.stages:
        dirs += [os.path.dirname(p) for p in glob.glob(os.path.join(a.root, st, "*", "measures.json"))]
    specs = []
    for d in dirs:
        if not a.redo and os.path.exists(os.path.join(d, "aligned.json")):
            continue
        j = json.load(open(os.path.join(d, "spec.json")))["job"]
        specs.append((d, j))
    specs.sort(key=lambda x: (x[1]["approach"], x[1]["group"], json.dumps(x[1].get("arm") or {}, sort_keys=True)))
    cache = {}
    for n, (d, j) in enumerate(specs):
        rds = fit_rds(j, cache)
        U, Z = load_UZ(d)
        U = {k: v.astype(np.float64) for k, v in U.items()}; Z = {k: v.astype(np.float64) for k, v in Z.items()}
        out = aligned_readings(U, Z, rds)
        tmp = os.path.join(d, "aligned.json.tmp")
        json.dump(out, open(tmp, "w")); os.replace(tmp, os.path.join(d, "aligned.json"))
        if n % 50 == 0:
            print(n, "of", len(specs), flush=True)
    print("done", len(specs), flush=True)


if __name__ == "__main__":
    main()
