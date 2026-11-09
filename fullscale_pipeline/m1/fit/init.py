# F3 — starting values for the M1 fits (ticket 107). Adapted from chunk13v10.initialize_tucker_adapted_nndsvd_and_propagate
# (v9.4; tickets 64, 74, 85): no new decision, the toy method on the trial's facets.
#   1. random base (uniform + 1e-4) for every factor and core matrix, from the fit's own seed;
#   2. NNDSVD (Boutsidis & Gallopoulos 2008) of the semantic anchors stacked on their shared column facet
#      (child×article and child×parent, both with child hyperedges as columns; visible cells only, so a hidden cell can
#      never shape the start); the left vectors are sliced back to articles and parents, the right ones go to children;
#      the anchors' core matrices start as diag(singular values);
#   3. the other facets are filled by degree-corrected propagation through the remaining relations (row-normalised
#      relation × an initialised facet), in a fixed order until every facet is set;
#   4. NNDSVDar-style noise (mean/10 + 1e-4) on every factor matrix (ticket 74).
# The seed changes the random base, the noise, the svds start vector and the padding; the NNDSVD part itself is the same
# for every seed (as in the toy), so seeds start close together — see the checkpoint-2 report.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

ANCHORS = ("child_article", "parent_child")


def _nndsvd(u, s, vt, K, rng):
    k_svd = len(s)
    if k_svd < K:
        pad = K - k_svd
        u = np.hstack([u, rng.random((u.shape[0], pad)) * 0.01])
        vt = np.vstack([vt, rng.random((pad, vt.shape[1])) * 0.01])
        s = np.concatenate([s, rng.random(pad) * 0.01])
    U, V, S = np.zeros((u.shape[0], K)), np.zeros((vt.shape[1], K)), np.zeros(K)
    for j in range(K):
        uj, vj = u[:, j], vt[j, :]
        up, un, vp, vn = np.maximum(uj, 0), np.maximum(-uj, 0), np.maximum(vj, 0), np.maximum(-vj, 0)
        pn = np.linalg.norm(up) * np.linalg.norm(vp)
        nn = np.linalg.norm(un) * np.linalg.norm(vn)
        a, b, sc = (up, vp, pn) if pn >= nn else (un, vn, nn)
        U[:, j] = a / np.linalg.norm(a) if np.linalg.norm(a) > 0 else a
        V[:, j] = b / np.linalg.norm(b) if np.linalg.norm(b) > 0 else b
        S[j] = s[j] * sc
    return U, V, S


def visible_matrix(rd):
    """The visible part of a relation as a sparse matrix (hidden cells and missing rows are zero)."""
    return sp.csr_matrix((rd.tie_val, (rd.tie_row, rd.tie_col)), shape=rd.shape)


def initialise(rds, K, seed):
    """Starting U (facet → array) and Z (relation → array) for the relations rds (data.RelationData list)."""
    rng = np.random.default_rng(seed)
    sizes = {}
    for rd in rds:
        sizes[rd.row_facet] = rd.shape[0]; sizes[rd.col_facet] = rd.shape[1]
    facets = sorted(sizes)
    U = {f: rng.random((sizes[f], K)) + 1e-4 for f in facets}
    Z = {rd.name: rng.random((K, K)) + 1e-4 for rd in sorted(rds, key=lambda r: r.name)}
    by = {rd.name: rd for rd in rds}
    done = set()
    anchors = [a for a in ANCHORS if a in by]
    if anchors:
        col = {by[a].col_facet for a in anchors}
        assert len(col) == 1, "joint anchor SVD needs one shared column facet"
        mats = [visible_matrix(by[a]) for a in anchors]
        J = sp.vstack(mats).tocsc()
        k_svd = min(K, min(J.shape) - 1)
        v0 = rng.random(min(J.shape)) + 0.1
        u, s, vt = scipy.sparse.linalg.svds(J, k=k_svd, v0=v0)
        o = np.argsort(s)[::-1]
        Un, Vn, Sn = _nndsvd(u[:, o], s[o], vt[o, :], K, rng)
        off = 0
        for a, M in zip(anchors, mats):
            U[by[a].row_facet] = Un[off:off + M.shape[0]] + 1e-4
            Z[a] = np.diag(Sn) + 1e-4
            done.add(by[a].row_facet); off += M.shape[0]
        U[col.pop()] = Vn + 1e-4
        done.update(by[a].col_facet for a in anchors)
    todo = set(facets) - done
    while todo:
        progress = False
        for rd in sorted(rds, key=lambda r: r.name):
            if rd.name in anchors:
                continue
            f1, f2 = rd.row_facet, rd.col_facet
            M = visible_matrix(rd)
            if f1 in todo and f2 in done:
                src, tgt, A = f2, f1, M
            elif f2 in todo and f1 in done:
                src, tgt, A = f1, f2, M.T.tocsr()
            else:
                continue
            rs = np.asarray(A.sum(axis=1)).ravel()
            Dinv = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1.0), 0.0)
            P = sp.diags(Dinv) @ A @ U[src]
            n = np.linalg.norm(P, axis=0, keepdims=True); n[n == 0] = 1.0
            U[tgt] = P / n + 1e-4
            todo.discard(tgt); done.add(tgt); progress = True
        if not progress:
            break
    for f in facets:
        U[f] = U[f] + rng.random(U[f].shape) * (U[f].mean() / 10.0 + 1e-4)
    return U, Z
