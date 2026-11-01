# Hold-out measurements on the built chunk12v3 matrices (ticket 107, design doc addendum 2026-10-29; owner's permission
# 2026-11-01). Reads a build directory, writes nothing.
#   (1) eligible ties per matrix: both entities have >= 2 ties (an entity with 1 tie is never tested);
#   (2) a random draw of the 10 groups under the per-entity cap: in any one group an entity loses at most
#       max(1, floor(n/3)) of its n ties and never its last one; ties that fit no group are never hidden;
#       what one group (one fit's hidden set, 10%) holds, by the tie count of the row and column entities (2, 3-5, 6+);
#   (3) the cost of the hidden empty cells (10% of the empty cells of the same matrices): their number, the memory to
#       hold their indices, and the time to evaluate the reconstruction on them once (they must be evaluated every epoch
#       to leave them out of a squared-error loss), timed here on random factor matrices.
import json, math, os, sys, time
import numpy as np
import scipy.sparse as sp

D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
miss = json.load(open(D + "missing.json"))
MATS = [("child×article", "child_article", None), ("child×parent", "parent_child", None),
        ("article×affiliation (appr. 2)", "P3_article_affiliation", "P3_article_affiliation_rows"),
        ("author×affiliation (appr. 1)", "P2_author_affiliation", "P2_author_affiliation_rows"),
        ("author×article (appr. 1)", "P1_author_article", None),
        ("article×article (appr. 2)", "P5_article_article", None)]
rng = np.random.default_rng(107)
G = 10

def draw_groups(r, c, nr, nc):
    """Assign eligible ties to one of G groups under the cap; returns group per tie (-1 = never hidden)."""
    cap_r = np.maximum(1, nr // 3); cap_c = np.maximum(1, nc // 3)
    used_r = np.zeros((G, len(nr)), dtype=np.int32); used_c = np.zeros((G, len(nc)), dtype=np.int32)
    out = np.full(len(r), -1)
    for t in rng.permutation(len(r)):
        i, j = r[t], c[t]
        ok = np.nonzero((used_r[:, i] < np.minimum(cap_r[i], nr[i] - 1)) & (used_c[:, j] < np.minimum(cap_c[j], nc[j] - 1)))[0]
        if len(ok):
            g = rng.choice(ok); out[t] = g; used_r[g, i] += 1; used_c[g, j] += 1
    return out

def bucket(n): return "2" if n == 2 else "3-5" if n <= 5 else "6+"

print(f"build: {D}")
hidden_empty = {}
for name, f, mrows in MATS:
    M = sp.load_npz(D + f + ".npz").tocoo()
    if name.startswith("article×article"):            # symmetric: one tie per pair; diagonal is missing
        keep = M.row < M.col; r, c = M.row[keep], M.col[keep]
        nr = np.bincount(np.r_[M.row], minlength=M.shape[0]); nc = nr
    else:
        r, c = M.row, M.col
        nr = np.bincount(r, minlength=M.shape[0]); nc = np.bincount(c, minlength=M.shape[1])
    missing_rows = set(miss.get(mrows, [])) if mrows else set()
    rows_obs = M.shape[0] - len(missing_rows)
    cells = rows_obs * M.shape[1] - (M.shape[0] if name.startswith("article×article") else 0)
    if name.startswith("article×article"): cells //= 2
    ties = len(r); elig = (nr[r] >= 2) & (nc[c] >= 2)
    print(f"\n== {name}: {M.shape[0]:,} × {M.shape[1]:,}, observed cells {cells:,}, ties {ties:,}")
    print(f"   eligible ties (both entities with >= 2 ties): {elig.sum():,} ({elig.mean():.1%})")
    re_, ce_ = r[elig], c[elig]
    grp = draw_groups(re_, ce_, nr, nc) if elig.sum() else np.array([])
    placed = (grp >= 0).sum() if len(grp) else 0
    per_g = np.bincount(grp[grp >= 0], minlength=G) if placed else np.zeros(G, int)
    print(f"   ties placed in one of the 10 groups: {placed:,} ({placed / max(1, elig.sum()):.1%} of eligible); "
          f"per group min {per_g.min():,} max {per_g.max():,} = {per_g.mean() / max(1, ties):.1%} of all ties")
    if placed:
        g0 = grp == 0
        for side, idx, n in (("row", re_, nr), ("column", ce_, nc)):
            ents = np.unique(idx[g0]); b = {}
            for e in ents: b[bucket(n[e])] = b.get(bucket(n[e]), 0) + 1
            alle = {}
            for e in np.unique(idx): alle[bucket(n[e])] = alle.get(bucket(n[e]), 0) + 1
            print(f"   group 1, {side} entities losing a tie, by their tie count: " +
                  ", ".join(f"{k}: {b.get(k, 0):,} of {alle.get(k, 0):,} eligible" for k in ("2", "3-5", "6+")))
    hid_e = int(round(0.1 * (cells - ties)))
    hidden_empty[name] = hid_e
    print(f"   hidden empty cells at 10%: {hid_e:,} (vs hidden ties per group {per_g.mean():,.0f}; ratio {hid_e / max(1, per_g.mean()):,.0f}:1)")

# cost of evaluating the reconstruction on hidden empty cells
import torch
torch.set_num_threads(1)
print("\n== cost of the hidden empty cells (single thread, as in production)")
for K in (4, 5):
    n = 2_000_000
    U = torch.rand(97850, K); V = torch.rand(35983, K); Z = torch.rand(K, K)
    i = torch.randint(0, 97850, (n,)); j = torch.randint(0, 35983, (n,))
    t = time.time()
    for _ in range(3):
        val = ((U[i] @ Z) * V[j]).sum(1); loss = (val ** 2).sum()
    ev = (time.time() - t) / 3
    U.requires_grad_(True); V.requires_grad_(True); Z.requires_grad_(True)
    t = time.time()
    for _ in range(3):
        val = ((U[i] @ Z) * V[j]).sum(1); (val ** 2).sum().backward()
    evb = (time.time() - t) / 3
    per_m = evb / n * 1e6
    print(f"   K={K}: forward {ev / n * 1e9:.0f} ns per cell; forward+backward {per_m:.3f} s per million cells")
    for name, h in hidden_empty.items():
        if h > 1e6:
            print(f"      {name}: {h / 1e6:,.1f} M cells -> {per_m * h / 1e6:,.1f} s per epoch, "
                  f"{per_m * h / 1e6 * 2000 / 3600:,.1f} h per 2,000-epoch fit; indices {h * 8 / 1e9:.2f} GB (int32 pairs)")
