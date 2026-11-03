# Ticket 107, L2 K range (2026-11-03): Horn's parallel analysis adapted to sparse relation matrices — count the
# singular values of each built matrix that exceed the same-rank singular value of degree-preserving shuffles (each tie
# keeps its row and value and receives the column of a randomly permuted tie, so row and column tie counts are kept).
# Result: it fails here (real values below the shuffles), the known effect of very unequal tie counts. Read-only.
import time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
F = {"child×parent": "parent_child", "child×article": "child_article", "author×article": "P1_author_article",
     "author×affiliation": "P2_author_affiliation", "article×affiliation": "P3_article_affiliation",
     "article×article": "P5_article_article", "article×journal": "P4_article_journal"}
M, R = 80, 3
rng = np.random.default_rng(1)
for name, f in F.items():
    t = time.time()
    X = sp.load_npz(D + f + ".npz").tocoo().astype(float)
    k = min(M, min(X.shape) - 1)
    s = np.sort(svds(X.tocsr(), k=k, random_state=0)[1])[::-1]
    null = [np.sort(svds(sp.coo_matrix((X.data, (X.row, rng.permutation(X.col))), shape=X.shape).tocsr(),
                         k=k, random_state=0)[1])[::-1] for _ in range(R)]
    nmax = np.max(null, axis=0); above = s > nmax
    first_fail = int(np.argmin(above)) if not above.all() else k
    print(f"{name:20s} layers above shuffle (consecutive from 1st): {first_fail:3d} (total above: {above.sum()} of {k}); "
          "ratio real/null at 1,5,10,20: " + ", ".join(f"{s[i] / nmax[i]:.2f}" for i in (0, 4, 9, 19))
          + f"  [{time.time() - t:.0f}s]", flush=True)
