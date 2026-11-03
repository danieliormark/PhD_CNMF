# Ticket 107, L2 (K range), 2026-11-03: checks requested by the external review (ChatGPT) of bethe_hessian_counts.py.
# Read-only. Run from fullscale_pipeline/m1 with tensor_env: python ../../plans/ticket107_layer2/bethe_hessian_checks.py
# Checks requested in the external review of the Bethe Hessian inertia count:
# (A) residuals of the SuperLU factorisation: ||Pr H Pc - L U||/||H|| and ||U - D L^T||/||U||, D = diag(U)
# (B) independent inertia by dense LDL^T with Bunch-Kaufman pivoting (scipy.linalg.ldl; 2x2 blocks handled by their
#     eigenvalues) for graphs up to ~12.2k nodes; (C) counts over a grid of r; (D) shifted counts H +/- delta I.
import sys, time, numpy as np, scipy.sparse as sp, scipy.linalg as la
from scipy.sparse.linalg import splu
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
F = {"article×article": "P5_article_article", "article×journal": "P4_article_journal",
     "article×affiliation": "P3_article_affiliation", "author×article": "P1_author_article",
     "author×affiliation": "P2_author_affiliation", "child×article": "child_article", "child×parent": "parent_child"}
def graph(name):
    X = (sp.load_npz(D + F[name] + ".npz") != 0).astype(float).tocsr()
    A = X if name == "article×article" else sp.bmat([[None, X], [X.T, None]]).tocsr()
    d = np.asarray(A.sum(1)).ravel(); keep = d > 0
    return A[keep][:, keep].tocsr(), d[keep]
def H_of(A, d, r): return ((r * r - 1) * sp.identity(A.shape[0]) - r * A + sp.diags(d)).tocsc()
def lu_count(H, resid=False):
    lu = splu(H, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    u = lu.U.diagonal(); n = int((u < 0).sum())
    if not resid: return n, None
    n_ = H.shape[0]
    Pr = sp.csc_matrix((np.ones(n_), (lu.perm_r, np.arange(n_))), shape=(n_, n_))
    Pc = sp.csc_matrix((np.ones(n_), (np.arange(n_), lu.perm_c)), shape=(n_, n_))
    R = Pr @ H @ Pc - lu.L @ lu.U
    r1 = sp.linalg.norm(R) / sp.linalg.norm(H)
    r2 = sp.linalg.norm(lu.U - sp.diags(u) @ lu.L.T) / sp.linalg.norm(lu.U)
    scale = np.abs(H.diagonal()).max()
    return n, (r1, r2, np.abs(u).min() / scale, (lu.perm_r == lu.perm_c).all())
def ldl_count(H):
    L, Dm, perm = la.ldl(H.toarray(), lower=True)
    return int((la.eigvalsh(Dm) < 0).sum()) if False else int(sum((np.linalg.eigvalsh(Dm[i:i+2, i:i+2]) < 0).sum() if (i + 1 < len(Dm) and Dm[i + 1, i] != 0) else (Dm[i, i] < 0) for i in _blocks(Dm)))
def _blocks(Dm):
    i, n = 0, Dm.shape[0]
    while i < n:
        yield i
        i += 2 if (i + 1 < n and Dm[i + 1, i] != 0) else 1
for name in sys.argv[1:] or list(F):
    t = time.time(); A, d = graph(name)
    rm, rmean = np.sqrt((d ** 2).sum() / d.sum() - 1), np.sqrt(d.mean())
    H = H_of(A, d, rm); n, (r1, r2, rel, sym) = lu_count(H, resid=True)
    line = (f"{name:20s} n={A.shape[0]:6d} r_mom {rm:5.2f} r_mean {rmean:4.2f} | LU count {n:3d}  resid(PHP-LU) {r1:.1e}"
            f"  resid(U-DL^T) {r2:.1e}  min|pivot|/max|diag| {rel:.1e}  same perm {sym}")
    if A.shape[0] <= 12500:
        line += f" | Bunch–Kaufman LDL^T count {ldl_count(H):3d}"
    delta = 1e-6 * np.abs(H.diagonal()).max()
    line += f" | shifted ±δ: {lu_count((H + delta * sp.identity(H.shape[0])).tocsc())[0]}/{lu_count((H - delta * sp.identity(H.shape[0])).tocsc())[0]}"
    grid = np.unique(np.round(np.geomspace(1.1, max(rm, 2.0) * 1.2, 9), 2))
    line += " | r grid: " + " ".join(f"{g:g}:{lu_count(H_of(A, d, g))[0]}" for g in grid)
    print(line + f"  [{time.time()-t:.0f}s]", flush=True)
