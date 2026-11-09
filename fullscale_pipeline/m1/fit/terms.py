# F4 — in-loop terms of the M1 fits (ticket 107). Each term reads the normalised view (U_norm, Z_scaled) and weighted
# memberships; column weights are recomputed every epoch without gradient (GLOSSARY "Column weights"; the full gradient
# otherwise, 2026-10-29 correction). All terms switch on at the warm-up epoch (300 by default, to be re-examined from the
# membership-stability logs; 2026-10-28, 2026-11-06).
#
#   WordSimilarity (λ_child; λ_article for the fallback arm), 2026-10-27/29:
#       R = (1/N) Σ_i Σ_j (W_ij / d_i) ‖p_i − p_j‖²,  p = weighted memberships, d_i = Σ_j W_ij, N = entities with a tie.
#   ConcentrationFloor (step 4, 2026-11-07): for each child hyperedge h, the mix of its articles' memberships, frozen at the
#       warm-up epoch: q_h(a) ∝ x_ah + β_W Σ_j W_hj x_aj (own articles weight 1, articles of W-similar hyperedges weight
#       similarity × β_W), mix_h = Σ_a q_h(a) m_a. Spread (bits) splits exactly: H(mix_h) = within_h + between_h,
#       within_h = Σ_a q_h(a) H(m_a). minimum_h = γ (between_h + ω within_h); applies to hyperedges with ≥ 2 articles in
#       that set; penalty = mean_h max(0, minimum_h − H(p_h))².
#   SpreadCeiling (S3, step 5, 2026-11-07; tested only if its gate passes): for entity i of a facet, its relation mix in
#       relation r = tie-weighted average of the weighted memberships of the entities tied to i in r (frozen at warm-up;
#       for children in child×article the floor's mix with the same β_W); ceiling_i = ρ × the largest relation-mix spread
#       over i's relations; penalty = mean_i max(0, H(p_i) − ceiling_i)².
#   OffDiagonalShare (O3 CANDIDATE, step 2, 2026-11-06 — NOT DECIDED; chosen or not after stage 0): per relation the
#       share of the reconstructed tie weight in between-community parts, Σ_{k≠l} part(k,l) / Σ part, part(k,l) =
#       Z[k,l] a_k b_l (a, b column sums); penalty = Σ_r v_r share_r with v_r the matrix-importance weights (default
#       equal, 1/n_relations).
import numpy as np
import scipy.sparse as sp
import torch

LOG2 = float(np.log(2.0))
COLUMN_CUTOFF = 1e-3          # toy cut-off (ticket 100); its value is part of item (l)


# ---------------------------------------------------------------------------------------------- weighted memberships
def column_weights(Un, Zs, relations, f):
    """w_k(f)² = Σ over relations touching f of Z[k,:](U_gᵀU_g)Z[k,:]ᵀ (rows) or Z[:,k]ᵀ(U_aᵀU_a)Z[:,k] (columns);
    no gradient; columns below COLUMN_CUTOFF × the largest get 0."""
    with torch.no_grad():
        w = torch.zeros(Un[f].shape[1], dtype=Un[f].dtype)
        for r, (a, b) in relations.items():
            z = Zs[r]
            if a == f:
                w = w + torch.diagonal(z @ (Un[b].T @ Un[b]) @ z.T)
            if b == f:
                w = w + torch.diagonal(z.T @ (Un[a].T @ Un[a]) @ z)
        w = torch.sqrt(torch.clamp(w, min=0.0))
        return torch.where(w < COLUMN_CUTOFF * w.max(), torch.zeros_like(w), w)


def weighted_membership(Un, Zs, relations, f, rows=None):
    """Rows of facet f: U_norm × column weights, rescaled to sum to 1 (differentiable in U_norm)."""
    w = column_weights(Un, Zs, relations, f)
    X = Un[f] if rows is None else Un[f][rows]
    X = X * w
    return X / X.sum(1, keepdim=True).clamp(min=1e-300)


def entropy_bits(P):
    """Shannon entropy in bits of each row of P (rows sum to 1)."""
    return -(P * torch.log(P.clamp(min=1e-300))).sum(-1) / LOG2


def _entropy_bits_np(P):
    return -(P * np.log(np.clip(P, 1e-300, None))).sum(-1) / LOG2


def _row_normalise(M):
    M = sp.csr_matrix(M, dtype=np.float64)
    s = np.asarray(M.sum(axis=1)).ravel()
    return sp.diags(np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)) @ M


def _visible(rd):
    return sp.csr_matrix((rd.tie_val, (rd.tie_row, rd.tie_col)), shape=rd.shape)


def article_mixing(rd_child_article, W=None, beta_w=0.0):
    """Children × articles mixing weights q_h(a) ∝ x_ah + β_W Σ_j W_hj x_aj (rows sum to 1) and each row's number of
    articles. rd_child_article: data.RelationData (rows articles, columns children); W: children × children, symmetric."""
    Xt = _visible(rd_child_article).T.tocsr()            # children × articles
    C = Xt if (W is None or beta_w == 0) else (Xt + beta_w * (sp.csr_matrix(W) @ Xt)).tocsr()
    C.eliminate_zeros()
    n_art = np.diff(C.indptr)
    return _row_normalise(C), n_art


# ----------------------------------------------------------------------------------------------------------- terms
class WordSimilarity:
    name = "word_similarity"

    def __init__(self, facet, W):
        W = sp.csr_matrix(W, dtype=np.float64); W.eliminate_zeros()
        d = np.asarray(W.sum(axis=1)).ravel()
        coo = W.tocoo()
        self.facet = facet
        self.i = torch.as_tensor(coo.row, dtype=torch.long); self.j = torch.as_tensor(coo.col, dtype=torch.long)
        self.v = torch.as_tensor(coo.data / d[coo.row], dtype=torch.float64)
        self.N = int((d > 0).sum())
        self.needs_freeze = False

    def value(self, Un, Zs, relations):
        P = weighted_membership(Un, Zs, relations, self.facet)
        diff = P[self.i] - P[self.j]
        return (self.v.to(P.dtype) * (diff ** 2).sum(1)).sum() / self.N


class ConcentrationFloor:
    name = "concentration_floor"

    def __init__(self, rd_child_article, gamma, omega, beta_w=0.0, W=None):
        self.q, self.n_art = article_mixing(rd_child_article, W, beta_w)
        self.gamma, self.omega, self.beta_w = gamma, omega, beta_w
        self.child, self.article = rd_child_article.col_facet, rd_child_article.row_facet
        self.needs_freeze = True
        self.idx = None

    def freeze(self, Un, Zs, relations):
        with torch.no_grad():
            m = weighted_membership(Un, Zs, relations, self.article).double().numpy()
        Hm = _entropy_bits_np(m)
        keep = self.n_art >= 2
        q = self.q[keep]
        mix = q @ m
        within = q @ Hm
        between = np.maximum(_entropy_bits_np(mix) - within, 0.0)
        minimum = self.gamma * (between + self.omega * within)
        self.idx = torch.as_tensor(np.nonzero(keep)[0], dtype=torch.long)
        self.minimum = torch.as_tensor(minimum)
        self.info = {"n_hyperedges": int(keep.sum()), "mean_between_bits": float(between.mean()) if keep.any() else 0.0,
                     "mean_within_bits": float(within.mean()) if keep.any() else 0.0}
        self.between, self.within = between, within

    def value(self, Un, Zs, relations):
        P = weighted_membership(Un, Zs, relations, self.child, rows=self.idx)
        short = torch.relu(self.minimum.to(P.dtype) - entropy_bits(P))
        return (short ** 2).mean()


class SpreadCeiling:
    name = "spread_ceiling"

    def __init__(self, facet, rds, rho, beta_w=0.0, W=None):
        """rds: the fit's data.RelationData list; the relations touching `facet` define its relation mixes."""
        self.facet, self.rho, self.beta_w = facet, rho, beta_w
        self.mix = []                                       # (mixing matrix facet × partner, partner facet)
        for rd in rds:
            if facet not in (rd.row_facet, rd.col_facet):
                continue
            if rd.name == "child_article" and facet == rd.col_facet:
                q, _ = article_mixing(rd, W, beta_w)
                self.mix.append((q, rd.row_facet))
                continue
            M = _visible(rd)
            if rd.row_facet == facet:
                self.mix.append((_row_normalise(M), rd.col_facet))
            if rd.col_facet == facet and rd.row_facet != rd.col_facet:
                self.mix.append((_row_normalise(M.T), rd.row_facet))
        self.needs_freeze = True
        self.idx = None

    def freeze(self, Un, Zs, relations):
        n = Un[self.facet].shape[0]
        best = np.full(n, -np.inf)
        with torch.no_grad():
            for q, partner in self.mix:
                m = weighted_membership(Un, Zs, relations, partner).double().numpy()
                has = np.diff(q.indptr) > 0
                s = np.full(n, -np.inf)
                s[has] = _entropy_bits_np(q[has] @ m)
                best = np.maximum(best, s)
        keep = np.isfinite(best)
        self.idx = torch.as_tensor(np.nonzero(keep)[0], dtype=torch.long)
        self.ceiling = torch.as_tensor(self.rho * best[keep])
        self.info = {"n_entities": int(keep.sum()), "mean_ceiling_bits": float(self.ceiling.mean()) if keep.any() else 0.0}

    def value(self, Un, Zs, relations):
        P = weighted_membership(Un, Zs, relations, self.facet, rows=self.idx)
        excess = torch.relu(entropy_bits(P) - self.ceiling.to(P.dtype))
        return (excess ** 2).mean()


class OffDiagonalShare:
    """O3 CANDIDATE — not decided (chosen or not after stage 0)."""
    name = "off_diagonal_share"

    def __init__(self, relation_names, importance=None):
        self.names = list(relation_names)
        self.v = importance or {r: 1.0 / len(self.names) for r in self.names}
        self.needs_freeze = False

    def value(self, Un, Zs, relations):
        tot = 0.0
        for r in self.names:
            a, b = relations[r]
            P = Zs[r] * torch.outer(Un[a].sum(0), Un[b].sum(0))
            tot = tot + self.v[r] * (1.0 - torch.diagonal(P).sum() / P.sum())
        return tot
