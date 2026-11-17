# Link structure of each relation in stage-0 pass 1 at K* (ticket 107; owner questions 2026-11-17 after RL-253).
# Question: why is the off-diagonal share of author×affiliation and article×journal ≈ 1, and why does article×journal
# differ between losses? Candidate explanation: organisations, journals and parents are "single-relation facets" — each
# appears in one relation matrix only — so their community labels are not fixed by the loss: for a permutation Π,
# U_a Z U_bᵀ = U_a (ZΠ)(U_bΠ)ᵀ, and nothing else in the loss involves U_b. The diagonal of such a relation then depends on
# an arbitrary labelling (in practice: the start).
# Per fit and relation, from the tie-weight parts part(k,l) = Z[k,l] a_k b_l (measures.parts):
#   within share as fitted (Σ_k part(k,k) ÷ total);
#   within share after relabelling the column facet's communities one-to-one (Hungarian, maximal diagonal tie weight);
#   main-partner share Σ_k max_l part(k,l) ÷ total (label-free: how much of each row community's weight goes to its
#   single strongest partner community) and the effective number of partners per row community (exp of the entropy of
#   its row of parts, weighted by row weight) — label-free, so comparable across relations and losses;
#   for author×affiliation / article×affiliation / article×journal: detachment (present ≥ 0.5 on the row facet, < 0.1 on
#   the column facet) before and after relabelling, and the share of the community's row-facet membership weight
#   (weighted membership columns, measures.weighted_memberships) carried by entities with no known affiliation /
#   journal (missing rows), detached vs other communities (owner's point: such absence may be a true result).
# Usage: python -m fit.link_structure_pass1 --out results/stage0/link_structure_pass1.json
import argparse, json
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import data as D, measures as M
from .aggregate import load_UZ
from .facet_detachment_pass1 import all_fits, KSTAR, LOSS

SINGLE = {"parent_child", "author_affiliation", "article_affiliation", "article_journal"}   # one side is a single-relation facet
MISSING = {"author_affiliation", "article_affiliation", "article_journal"}


def run():
    fits = [f for f in all_fits() if f["job"]["K"] == KSTAR[f["job"]["beta"]] and f["job"]["seed"] in range(1, 11)]
    build = D.Build(D.BUILD)
    rds_cache = {}
    rows = defaultdict(list)
    for n, f in enumerate(sorted(fits, key=lambda f: (f["job"]["approach"], f["job"]["group"]))):
        j = f["job"]; key = (j["approach"], j["group"])
        if key not in rds_cache:
            rds_cache.clear(); rds_cache[key] = {rd.name: rd for rd in D.fit_data(build, j["approach"], group=j["group"])}
        rds = rds_cache[key]
        U, Z = load_UZ(f["dir"])
        U = {k: v.astype(np.float64) for k, v in U.items()}; Z = {k: v.astype(np.float64) for k, v in Z.items()}
        wm = M.weighted_memberships(U, Z, f["result"]["relations"])
        fp = {x: np.array(v) for x, v in f["measures"]["facet_presence"].items()}
        cell = f"approach_{j['approach']}_{LOSS[j['beta']]}"
        for name, rd in rds.items():
            P, T = M.parts(U, Z, rd)
            P = P / T
            r, c = linear_sum_assignment(-P)
            perm = np.empty(len(c), dtype=int); perm[r] = c
            rw = P.sum(1)
            Pr = P / np.maximum(rw[:, None], 1e-300)
            ent = -(Pr * np.log(np.clip(Pr, 1e-300, None))).sum(1)
            out = {"within": float(np.trace(P)), "within_relabelled": float(P[r, c].sum()),
                   "main_partner": float(P.max(1).sum()),
                   "effective_partners": float((rw * np.exp(ent)).sum() / rw.sum()),
                   "relabel_moves": float((perm != np.arange(len(perm))).mean())}
            if name in MISSING:
                rf, cf = rd.row_facet, rd.col_facet
                K = P.shape[0]
                side_col = K * P.sum(0)                              # column facet's side reading (single-relation facet)
                det0 = (fp[rf] >= 0.5) & (side_col < 0.1)
                det1 = (fp[rf] >= 0.5) & (side_col[perm] < 0.1)
                Y = wm[rf]["Y"]; miss = ~rd.obs_row_mask
                share_missing = Y[miss].sum(0) / np.maximum(Y.sum(0), 1e-300)
                out.update({"detached_as_fitted": int(det0.sum()), "detached_relabelled": int(det1.sum()),
                            "missing_share_detached_relabelled": share_missing[det1].tolist(),
                            "missing_share_others": share_missing[~det1].tolist(),
                            "missing_share_rows": float(miss.mean())})
            rows[(cell, name)].append(out)
        if n % 20 == 0:
            print(n, "of", len(fits), flush=True)
    rep = {}
    for (cell, name), L in rows.items():
        s = {k: float(np.median([x[k] for x in L])) for k in ("within", "within_relabelled", "main_partner", "effective_partners", "relabel_moves")}
        s["fits"] = len(L); s["single_relation_facet"] = name in SINGLE
        if name in MISSING:
            s["detached_as_fitted_per_fit"] = float(np.mean([x["detached_as_fitted"] for x in L]))
            s["detached_relabelled_per_fit"] = float(np.mean([x["detached_relabelled"] for x in L]))
            d = sum((x["missing_share_detached_relabelled"] for x in L), []); o = sum((x["missing_share_others"] for x in L), [])
            s["missing_share_detached_median"] = float(np.median(d)) if d else None
            s["missing_share_others_median"] = float(np.median(o))
            s["missing_share_rows"] = L[0]["missing_share_rows"]
        rep.setdefault(cell, {})[name] = s
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
