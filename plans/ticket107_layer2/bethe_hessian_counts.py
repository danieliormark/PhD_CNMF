# Ticket 107, L2 (K range): number of communities detectable in each relation's tie pattern (ties binarised), by the
# Bethe Hessian count (Saade, Krzakala & Zdeborova 2014; Le & Levina 2015): K = number of negative eigenvalues of
# H(r) = (r^2 - 1) I - r A + D on the bipartite graph (rows and columns as nodes; article×article as is).
# Two choices of r: "moments" r = sqrt(sum d^2 / sum d - 1) (Saade et al.) and "mean" r = sqrt(mean degree) (BHm, Le &
# Levina). Counted by Sylvester's law of inertia: negative pivots of a symmetric factorisation (sparse LU with a
# symmetric ordering and diagonal pivots only); checked against Lanczos (eigsh, smallest algebraic) on three matrices.
# Read-only. Run: python bethe_hessian_counts.py [matrix names]
import sys, time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
F = {"article×journal": "P4_article_journal", "article×article": "P5_article_article",
     "article×affiliation": "P3_article_affiliation", "author×article": "P1_author_article",
     "author×affiliation": "P2_author_affiliation", "child×article": "child_article", "child×parent": "parent_child"}
def bh(name):
    X = (sp.load_npz(D + F[name] + ".npz") != 0).astype(float).tocsr()
    A = X if name == "article×article" else sp.bmat([[None, X], [X.T, None]]).tocsr()
    d = np.asarray(A.sum(1)).ravel(); keep = d > 0; A = A[keep][:, keep]; d = d[keep]
    out = {}
    for rule, r in (("moments", np.sqrt((d ** 2).sum() / d.sum() - 1)), ("mean", np.sqrt(d.mean()))):
        out[rule] = ((r * r - 1) * sp.identity(A.shape[0]) - r * A + sp.diags(d), r)
    return out
for name in sys.argv[1:] or list(F):
    t = time.time()
    for rule, (H, r) in bh(name).items():
        lu = splu(H.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        u = lu.U.diagonal(); sym = (lu.perm_r == lu.perm_c).all()
        print(f"{name:20s} r={rule:7s} nodes {H.shape[0]:7d} r {r:5.2f} negative pivots {int((u < 0).sum()):4d}  "
              f"symmetric permutation {sym}  min|pivot| {np.abs(u).min():.1e}  [{time.time()-t:.0f}s]", flush=True)
