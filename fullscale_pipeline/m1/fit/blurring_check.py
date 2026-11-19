# Do memberships blur to make up for missing off-diagonal links? (owner 2026-11-19, check of the caveat to 1b1: "a
# ratio near 1 doesn't prove there is no problem. It may hold only because memberships were blurred to make up for
# the missing off-diagonal links"). Saved fits only: KL, K 13, hold-out group 1, seeds 1–10; random social start (pass
# 1) vs diagonal social start (start test), paired by seed. Under a diagonal core a tie is rebuilt only through
# x̂_ij = Σ_k a_ik z_k b_jk, so a tie between entities with different main communities needs both ends to share
# communities; with off-diagonal links it does not.
# Readings (weighted memberships, each fit's own numbering; the measures are numbering-free):
#   per facet: spread of an entity's memberships — entropy in bits (0 = one community; log2 13 = 3.70 = even) and the
#     largest membership share; means over entities;
#   per relation: co-membership of a tie's two ends, Σ_k p_k q_k, for visible ties whose ends have different main
#     communities ("crossing") and the same main community ("within"), tie-weighted means; chance = the same over
#     random pairs of the two facets' entities (product of the facets' mean memberships, Σ_k p̄_k q̄_k).
# Blurring that compensates would show as: crossing ties' co-membership higher under the diagonal start, and the
# entities at the ends of crossing ties more spread.
# Usage: python -m fit.blurring_check --out fit/results/stage0/aligned/blurring_check.json
import argparse, json, os
import numpy as np
from . import data as D, measures as M
from .aggregate import load_UZ
from .start_test_analysis import load


def entropy_bits(P):
    return -(P * np.log2(np.clip(P, 1e-300, None))).sum(1)


def one(f, rds):
    U, Z = load_UZ(f["dir"])
    wm = M.weighted_memberships(U, Z, {k: tuple(v) for k, v in f["result"]["relations"].items()})
    P = {x: v["P"] for x, v in wm.items()}
    out = {"facet": {x: {"entropy_bits": float(entropy_bits(p).mean()), "largest_share": float(p.max(1).mean())}
                     for x, p in P.items()}, "relation": {}}
    for name, rd in rds.items():
        a, b = rd.row_facet, rd.col_facet
        pa, pb = P[a][rd.tie_row], P[b][rd.tie_col]; v = rd.tie_val
        co = (pa * pb).sum(1); cross = pa.argmax(1) != pb.argmax(1)
        r = {"crossing_weight_share": float(v[cross].sum() / v.sum()),
             "co_membership_crossing": float((co[cross] * v[cross]).sum() / v[cross].sum()) if cross.any() else None,
             "co_membership_within": float((co[~cross] * v[~cross]).sum() / v[~cross].sum()) if (~cross).any() else None,
             "co_membership_chance": float((P[a].mean(0) * P[b].mean(0)).sum())}
        # spread of the entities at the ends of crossing ties vs all tied entities of the same facet (row side)
        rows_cross = np.unique(rd.tie_row[cross]); rows_all = np.unique(rd.tie_row)
        r["row_end_entropy_crossing"] = float(entropy_bits(P[a][rows_cross]).mean()) if len(rows_cross) else None
        r["row_end_entropy_all"] = float(entropy_bits(P[a][rows_all]).mean())
        out["relation"][name] = r
    return out


def run():
    build = D.Build(D.BUILD); ref, arm = load("stage0_pass1", False), load("start_test", True); rep = {}
    for a in (1, 2):
        rds = {rd.name: rd for rd in D.fit_data(build, a, group=0)}
        R = {f["job"]["seed"]: one(f, rds) for f in ref[(a, 1)]}; A = {f["job"]["seed"]: one(f, rds) for f in arm[(a, 1)]}
        s = sorted(set(R) & set(A)); rep[f"approach_{a}"] = {"seeds": s, "random": R, "diagonal": A}
        print(f"approach {a} ({len(s)} paired seeds): random → diagonal start, mean over seeds (paired SE of the change)")
        def show(get, lab):
            x = np.array([get(R[k]) for k in s]); y = np.array([get(A[k]) for k in s]); d = y - x
            print(f"   {lab:55s} {x.mean():.3f} → {y.mean():.3f}  ({d.mean():+.3f}, SE {d.std(ddof=1)/np.sqrt(len(s)):.3f})")
        for x in sorted(R[s[0]]["facet"]):
            show(lambda q: q["facet"][x]["entropy_bits"], f"{x}: membership entropy (bits)")
            show(lambda q: q["facet"][x]["largest_share"], f"{x}: largest membership share")
        for n in rds:
            show(lambda q: q["relation"][n]["crossing_weight_share"], f"{n}: crossing share of tie weight (data)")
            show(lambda q: q["relation"][n]["co_membership_crossing"], f"{n}: co-membership, crossing ties")
            show(lambda q: q["relation"][n]["co_membership_within"], f"{n}: co-membership, within ties")
            show(lambda q: q["relation"][n]["co_membership_chance"], f"{n}: co-membership, chance")
            show(lambda q: q["relation"][n]["row_end_entropy_crossing"], f"{n}: row-end entropy, ends of crossing ties")
            show(lambda q: q["relation"][n]["row_end_entropy_all"], f"{n}: row-end entropy, all tied row entities")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    rep = run()
    json.dump({k: {"seeds": v["seeds"], "random": v["random"], "diagonal": v["diagonal"]} for k, v in rep.items()},
              open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
