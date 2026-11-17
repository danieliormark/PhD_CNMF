# Alignment (ticket 107, decided 2026-11-17): checks on real pass-1 fits, both approaches, both losses (K*).
#  1. reconstructed values at visible ties and hidden cells unchanged by alignment (relative ≤ 1e-12);
#  2. stored permutations (apply) reproduce the aligned U, Z exactly;
#  3. scrambling every non-article facet's numbering (with the core matrices reordered to match) and re-aligning gives
#     the same aligned U, Z — the result does not depend on the fitted numbering;
#  4. match strength after alignment ≥ as fitted for every relation in the alignment chain.
import numpy as np
from fit import measures as M
from fit.align import align, apply, match_strength, fit_rds
from fit.aggregate import load_UZ
from fit.facet_detachment_pass1 import all_fits


def rebuilt(U, Z, rds):
    out = []
    for rd in rds:
        A, B, z = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
        for r, c in ((rd.tie_row, rd.tie_col), (rd.hid_tie_row, rd.hid_tie_col), (rd.hid_empty_row, rd.hid_empty_col)):
            if len(r):
                out.append(((A[r] @ z) * B[c]).sum(1))
    return np.concatenate(out)


def scramble(U, Z, rds, rng):
    K = U["article"].shape[1]
    perms = {f: rng.permutation(K) for f in U if f != "article"}
    rel = {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}
    return apply(U, Z, rel, perms)


fits = all_fits()
rng = np.random.default_rng(0)
cache = {}
for a in (1, 2):
    for beta, K in ((2, 20), (1, 13)):
        f = next(f for f in fits if f["job"]["approach"] == a and f["job"]["beta"] == beta and f["job"]["K"] == K and f["job"]["group"] == 0)
        rds = fit_rds(f["job"], cache)
        U, Z = load_UZ(f["dir"]); U = {k: v.astype(np.float64) for k, v in U.items()}; Z = {k: v.astype(np.float64) for k, v in Z.items()}
        Ua, Za, perms, order = align(U, Z, rds)
        x0, x1 = rebuilt(U, Z, rds), rebuilt(Ua, Za, rds)
        d1 = np.abs(x1 - x0).max() / np.abs(x0).max(); assert d1 < 1e-12, d1
        rel = {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}
        Ub, Zb = apply(U, Z, rel, perms)
        assert all(np.array_equal(Ub[k], Ua[k]) for k in U) and all(np.array_equal(Zb[k], Za[k]) for k in Z)
        Us, Zs = scramble(U, Z, rds, rng)
        Uc, Zc, _, _ = align(Us, Zs, rds)
        assert all(np.allclose(Uc[k], Ua[k]) for k in U) and all(np.allclose(Zc[k], Za[k]) for k in Z), "depends on fitted numbering"
        s0, s1 = match_strength(U, Z, rds), match_strength(Ua, Za, rds)
        chain = {r for _, r in order}
        assert all(s1[r] >= s0[r] - 1e-12 for r in chain)
        print(f"approach {a} {'KL' if beta == 1 else 'squared error'} K {K}: max change {d1:.1e}; order {[o[0] for o in order]}; "
              + "; ".join(f"{r} {s0[r]:.2f}→{s1[r]:.2f}" for r in s0))
print("all checks passed")
