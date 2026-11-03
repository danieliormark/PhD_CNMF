# Ticket 107, L2.1 (2026-11-02). What the first SVD layer of each built matrix is: concentrated group or general level; uncentred vs centred sigma_1.
import json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds, LinearOperator
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
idx = json.load(open(D + "index.json"))
M = [("child×parent", "parent_child"), ("child×article", "child_article"), ("author×article", "P1_author_article"),
     ("author×affiliation", "P2_author_affiliation"), ("article×affiliation", "P3_article_affiliation"),
     ("article×journal", "P4_article_journal"), ("article×article", "P5_article_article")]
def centred_sigma1(X, axis):
    X = X.tocsr().astype(float); n, m = X.shape
    if axis == 0:   # subtract each column's mean over rows
        mu = np.asarray(X.mean(0)).ravel(); one = np.ones(n)
        mv = lambda v: X @ v.ravel() - one * (mu @ v.ravel()); rmv = lambda w: X.T @ w.ravel() - mu * w.ravel().sum()
    else:           # subtract each row's mean over columns
        mu = np.asarray(X.mean(1)).ravel(); one = np.ones(m)
        mv = lambda v: X @ v.ravel() - mu * v.ravel().sum(); rmv = lambda w: X.T @ w.ravel() - one * (mu @ w.ravel())
    L = LinearOperator((n, m), matvec=mv, rmatvec=rmv, dtype=float)
    return svds(L, k=1, random_state=0)[1][0]
for name, f in M:
    X = sp.load_npz(D + f + ".npz").tocsr().astype(float)
    u, s, vt = svds(X, k=1, random_state=0); u = np.abs(u[:, 0]); v = np.abs(vt[0]); s1 = s[0]
    tot = X.multiply(X).sum()
    rn = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); cn = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel())
    eff_r, eff_c = 1 / np.sum(u ** 4), 1 / np.sum(v ** 4)
    live_r, live_c = (rn > 0).sum(), (cn > 0).sum()
    cos_r = u @ rn / np.linalg.norm(rn); cos_c = v @ cn / np.linalg.norm(cn)
    c0, c1 = centred_sigma1(X, 0), centred_sigma1(X, 1)
    print(f"{name:20s} {X.shape[0]:>6}×{X.shape[1]:<6} s1^2/|X|^2 {s1**2/tot:.4f}  layer-1 effective rows {eff_r:8.1f} of {live_r} live, "
          f"cols {eff_c:8.1f} of {live_c}; cos(u1,row norms) {cos_r:.2f}, cos(v1,col norms) {cos_c:.2f}; "
          f"sigma1 uncentred {s1:.3f}, col-centred {c0:.3f} ({c0/s1:.2f}), row-centred {c1:.3f} ({c1/s1:.2f})")
    if name in ("child×parent", "child×article"):   # both stored with child hyperedges as columns
        cols = idx["children"]
        top = np.argsort(-v)[:8]
        print("   top child hyperedges of layer 1:", [sorted(idx["child_atoms"][cols[i]]) for i in top])
