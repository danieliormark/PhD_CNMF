# F7 — synthetic and rewired data for the M1 checks (ticket 107).
#
# code_check_world (decided 2026-11-08; a code/optimiser check only, never evidence about the method):
#   K* = 6 planted communities on approach 1's facets and relations, smaller entity counts; sparse memberships with pure
#   members in every community (identifiability up to permutation and scaling); entity degree multipliers resampled from
#   the trial's own degrees; one community present only in the semantic domain (its rows and columns are zero in every
#   social core matrix and no social entity belongs to it) and one present only in the child facet (tied to articles and
#   parents of community 0 through between-community core cells); versions: noiseless X = UZUᵀ, Poisson counts around it
#   (for KL), Gaussian noise around it (for squared error); the real mask structure: missing rows at the trial's shares,
#   one hold-out group drawn by the trial's rules (data._draw_tie_groups / _draw_empty_groups).
# graded_swap (decided 2026-11-08/09): a fraction p of articles has its identities permuted, within social-degree bands,
#   on the social side only; the semantic side is untouched (both domains keep their internal structure exactly).
import math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from . import data as D

APPROACH1 = D.APPROACH[1]
WORLD_SIZES = {"article": 300, "child": 1500, "parent": 600, "author": 900, "organisation": 150, "journal": 40}
SEMANTIC_ONLY, CHILD_ONLY = 4, 5


def _trial_degrees(build, approach=1):
    deg = {}
    for rel in D.APPROACH[approach]:
        f, rf, cf = D.RELATIONS[rel][:3]
        M = build.load(f)
        for fac, d in ((rf, np.diff(M.tocsr().indptr)), (cf, np.diff(M.tocsc().indptr))):
            deg[fac] = deg.get(fac, 0) + d
    return {f: d[d > 0] for f, d in deg.items()}


def planted_truth(seed, build=None, K=6, sizes=WORLD_SIZES, child_only=True):
    rng = np.random.default_rng(seed)
    degs = _trial_degrees(build) if build is not None else None
    allowed = {f: list(range(K)) for f in sizes}
    for f in ("author", "organisation", "journal"):
        allowed[f] = [k for k in range(K) if k not in (SEMANTIC_ONLY, CHILD_ONLY)]
    for f in ("article", "parent"):
        allowed[f] = [k for k in range(K) if k != CHILD_ONLY or not child_only]
    U = {}
    for f, n in sizes.items():
        ks = allowed[f]
        prim = np.array([ks[i % len(ks)] for i in range(n)]); rng.shuffle(prim)
        M = np.zeros((n, K))
        M[np.arange(n), prim] = 1.0
        mixed = rng.random(n) < 0.4                           # 60% pure members
        for i in np.nonzero(mixed)[0]:
            other = rng.choice([k for k in ks if k != prim[i]])
            a = rng.uniform(0.55, 0.85)
            M[i, prim[i]] = a; M[i, other] = 1 - a
        theta = rng.choice(degs[f], n) if degs is not None else rng.lognormal(0, 0.7, n)
        theta = np.sqrt(theta / theta.mean())
        U[f] = M * theta[:, None]
    Z = {}
    for rel in APPROACH1:
        _, rf, cf, dom = D.RELATIONS[rel][:4]
        z = np.eye(K)
        z += rng.uniform(0.0, 0.03, (K, K)) * (rng.random((K, K)) < 0.2)
        if dom == "social":
            z[SEMANTIC_ONLY, :] = 0; z[:, SEMANTIC_ONLY] = 0
        z[CHILD_ONLY, :] = 0; z[:, CHILD_ONLY] = 0
        if cf == "child" and child_only == "split":            # identifiable child-only community (2026-11-09):
            z[0 if rel == "child_article" else 2, CHILD_ONLY] = 0.8   # tied to articles of 0 but parents of 2, so it is
                                                               # not an extension of any single community
        elif cf == "child" and child_only:                     # first version: tied to community 0 in both relations
            z[0, CHILD_ONLY] = 0.8                             # (unidentifiable: equals an extension of community 0)
        elif not child_only and dom == "semantic":            # variant: community CHILD_ONLY is an ordinary community
            z[CHILD_ONLY, CHILD_ONLY] = 1.0
        Z[rel] = z
    return U, Z


def _relation_data(name, X, missing_rows, weight, rng, empty_cap):
    rf, cf, dom, hidden = D.RELATIONS[name][1], D.RELATIONS[name][2], D.RELATIONS[name][3], D.RELATIONS[name][4]
    X = X.copy(); X[missing_rows] = 0.0
    r, c = np.nonzero(X)
    v = X[r, c]
    hid = np.zeros(len(r), bool)
    er = ec = np.zeros(0, np.int64)
    if hidden:
        nr = np.bincount(r, minlength=X.shape[0]); nc = np.bincount(c, minlength=X.shape[1])
        elig = (nr[r] >= 2) & (nc[c] >= 2)
        g = np.full(len(r), -1); g[elig] = D._draw_tie_groups(r[elig], c[elig], nr, nc, rng)
        hid = g == 0
        if len(r) < X.shape[0] * X.shape[1] - len(missing_rows) * X.shape[1]:
            old = (D.EMPTY_SHARE, D.EMPTY_CAP)
            D.EMPTY_CAP = empty_cap
            er_, ec_, eg, _, _ = D._draw_empty_groups(X.shape, r, c, np.asarray(missing_rows, np.int64), rng)
            D.EMPTY_SHARE, D.EMPTY_CAP = old
            er, ec = er_[eg == 0], ec_[eg == 0]
    obs = np.ones(X.shape[0], bool); obs[missing_rows] = False
    fro = float(np.sqrt(np.sum(v[~hid] ** 2)))
    v = v / fro
    rd = D.RelationData(name, rf, cf, X.shape, weight, dom, r[~hid], c[~hid], v[~hid], obs, False,
                        r[hid], c[hid], v[hid], er, ec, fro)
    return rd


def code_check_world(version, seed, build=None, K=6, sizes=WORLD_SIZES, poisson_scale=3.0, gauss_sd=0.1, empty_cap=20000,
                     child_only=True):
    """version: 'noiseless' | 'poisson' | 'gaussian'. Returns (rds, truth_U, truth_Z) with truth_Z scaled like the data."""
    U, Z = planted_truth(seed, build, K, sizes, child_only)
    rng = np.random.default_rng(seed + 1000)
    w = D.relation_weights(APPROACH1)
    miss_share = {"author_affiliation": 582 / 10052, "article_journal": 4 / 2136}
    rds, Zs = [], {}
    for rel in APPROACH1:
        rf, cf = D.RELATIONS[rel][1:3]
        X = U[rf] @ Z[rel] @ U[cf].T
        s = 1.0                                               # expected data = s · U Z Uᵀ
        if version == "poisson":
            s = poisson_scale / X[X > 0].mean()
            X = rng.poisson(X * s).astype(float)
        elif version == "gaussian":
            nz = X[X > 0]
            X = X + rng.normal(0.0, gauss_sd * nz.mean(), X.shape)
        n_miss = int(round(miss_share.get(rel, 0.0) * X.shape[0]))
        missing = np.sort(rng.choice(X.shape[0], n_miss, replace=False)) if n_miss else np.zeros(0, np.int64)
        rd = _relation_data(rel, X, missing, w[rel], rng, empty_cap)
        rds.append(rd)
        Zs[rel] = Z[rel] * s / rd.fro                         # the truth on the data's normalised scale
    Ut = {f: np.maximum(u, 1e-7) for f, u in U.items()}
    Zt = {r: np.maximum(z, 1e-7) for r, z in Zs.items()}
    return rds, Ut, Zt


def _weighted_columns(U, Z, relations, f):
    """U_norm[:, k] × column weight w_k(f) (GLOSSARY): zero for a community that carries nothing in facet f, so a
    community absent from a facet does not appear as a uniform unit column."""
    norm = lambda u: u / np.maximum(np.linalg.norm(u, axis=0, keepdims=True), 1e-300)
    Un = {g: norm(U[g]) for g in U}
    s = {g: np.linalg.norm(U[g], axis=0) for g in U}
    w2 = np.zeros(U[f].shape[1])
    for r, (a, b) in relations.items():
        Zs = Z[r] * np.outer(s[a], s[b])
        if a == f:
            w2 += np.diag(Zs @ (Un[b].T @ Un[b]) @ Zs.T)
        if b == f:
            w2 += np.diag(Zs.T @ (Un[a].T @ Un[a]) @ Zs)
    return Un[f] * np.sqrt(np.maximum(w2, 0.0))


def matched_cosines(U_fit, U_true, Z_fit=None, Z_true=None, relations=None, facets=None):
    """Hungarian matching of fitted to planted communities on the membership columns stacked over facets; each facet's
    columns are U_norm × column weight (if Z and relations are given), so absent communities are zero columns.
    Returns per planted community the cosine with its matched fitted column, and the map planted → fitted."""
    facets = facets or sorted(U_true)
    if relations is not None:
        A = np.vstack([_weighted_columns(U_fit, Z_fit, relations, f) for f in facets])
        B = np.vstack([_weighted_columns(U_true, Z_true, relations, f) for f in facets])
    else:
        norm = lambda u: u / np.maximum(np.linalg.norm(u, axis=0, keepdims=True), 1e-300)
        A = np.vstack([norm(U_fit[f]) for f in facets]); B = np.vstack([norm(U_true[f]) for f in facets])
    A = A / np.linalg.norm(A, axis=0, keepdims=True); B = B / np.linalg.norm(B, axis=0, keepdims=True)
    C = B.T @ A                                           # planted × fitted cosines
    r, c = linear_sum_assignment(-C)
    return C[r, c], dict(zip(r.tolist(), c.tolist()))


# ------------------------------------------------------------------------------------------------- graded swap
SOCIAL_BANDS = (0, 1, 2, 4, 8, 16)            # social-degree band lower bounds (0, 1, 2–3, 4–7, 8–15, 16+); an assumption


def graded_swap(rds, p, seed):
    """Permute the identities of a random fraction p of articles among themselves, within social-degree bands, in the
    social relations only (rows/columns of articles, their missing-row marks and hidden cells move with them)."""
    rng = np.random.default_rng(seed)
    soc = [rd for rd in rds if rd.domain == "social" and "article" in (rd.row_facet, rd.col_facet)]
    n = next(rd.shape[0] if rd.row_facet == "article" else rd.shape[1] for rd in soc)
    deg = np.zeros(n)
    for rd in soc:
        for side, idx in (("row", rd.tie_row), ("col", rd.tie_col)):
            if (rd.row_facet if side == "row" else rd.col_facet) == "article":
                deg += np.bincount(idx, minlength=n)
    band = np.searchsorted(np.array(SOCIAL_BANDS), deg, side="right") - 1
    chosen = rng.random(n) < p
    perm = np.arange(n)
    for b in np.unique(band):
        ids = np.nonzero(chosen & (band == b))[0]
        if len(ids) > 1:
            perm[ids] = ids[rng.permutation(len(ids))]
    out = []
    for rd in rds:
        if rd not in soc:
            out.append(rd); continue
        mp = lambda a: perm[a]
        rr = (lambda a: mp(a)) if rd.row_facet == "article" else (lambda a: a)
        cc = (lambda a: mp(a)) if rd.col_facet == "article" else (lambda a: a)
        obs = rd.obs_row_mask
        if rd.row_facet == "article":
            obs = np.empty_like(obs); obs[perm] = rd.obs_row_mask
        out.append(D.RelationData(rd.name, rd.row_facet, rd.col_facet, rd.shape, rd.weight, rd.domain,
                                  rr(rd.tie_row), cc(rd.tie_col), rd.tie_val, obs, rd.diag_masked,
                                  rr(rd.hid_tie_row), cc(rd.hid_tie_col), rd.hid_tie_val,
                                  rr(rd.hid_empty_row), cc(rd.hid_empty_col), rd.fro))
    return out, perm
