# F1 — masked reconstruction loss for the M1 fits (ticket 107), squared error (β = 2) and generalised KL (β = 1),
# computed without forming the reconstruction X̂ = A Z Bᵀ (child×parent alone has 3.5 billion cells).
#
# Loss cells of relation r (O): observed rows × all columns, minus the diagonal of a same-facet relation
# (article×article), minus the fold's hidden ties and hidden empty cells (design doc 2026-11-01, option B).
#   squared error: Σ_O (x − x̂)² = Σ_vis x² − 2 Σ_vis x x̂ + [Σ_rows x̂² − Σ_diag x̂² − Σ_hidden x̂²]
#       Σ_rows x̂² = tr(Zᵀ (AᵀDA) Z (BᵀB)), D = diagonal 0/1 of observed rows (K×K Gram matrices only);
#   generalised KL: Σ_O [x log(x / x̂) − x + x̂] = Σ_vis [x log x − x − x log x̂] + [Σ_rows x̂ − Σ_diag x̂ − Σ_hidden x̂]
#       Σ_rows x̂ = (1ᵀDA) Z (Bᵀ1).
# "vis" = visible ties (the only cells with x > 0 in O). Both forms are exact; tests compare them with a dense brute
# force. Total training loss = Σ_r w_r · loss_r (L2.2, 2026-11-02).
import numpy as np
import torch


class RelationTensors:
    """Tensor view of a data.RelationData for one fit (built once, reused every epoch)."""
    def __init__(self, rd, dtype=torch.float64):
        L = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)
        F = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)
        self.name, self.row_facet, self.col_facet, self.weight = rd.name, rd.row_facet, rd.col_facet, rd.weight
        self.same_facet = rd.row_facet == rd.col_facet
        self.diag_masked = rd.diag_masked
        self.tie_r, self.tie_c, self.tie_x = L(rd.tie_row), L(rd.tie_col), F(rd.tie_val)
        self.row_mask = F(rd.obs_row_mask.astype(np.float64))
        self.hid_r, self.hid_c = L(rd.hid_row), L(rd.hid_col)
        self.n_hid_ties = len(rd.hid_tie_row)
        self.hid_tie_x = F(rd.hid_tie_val)
        self.sum_x2 = float(np.sum(np.asarray(rd.tie_val) ** 2))
        x = np.asarray(rd.tie_val)
        xp = np.where(x > 0, x, 1.0)                    # KL needs x ≥ 0; the constant is unused for squared error
        self.sum_xlogx_minus_x = float(np.sum(np.where(x > 0, x * np.log(xp), 0.0) - x)) if np.all(x >= 0) else float('nan')
        if self.diag_masked:
            self.diag_idx = L(np.nonzero(rd.obs_row_mask)[0])


def cell_values(A, Z, B, r, c):
    """x̂ at cells (r, c): rows of A Z dotted with rows of B."""
    if len(r) == 0:
        return A.new_zeros(0)
    return ((A[r] @ Z) * B[c]).sum(1)


def relation_loss(A, Z, B, rt, beta):
    """Masked loss of one relation; A = row facet factor (n×K), B = column facet factor (m×K), Z (K×K)."""
    xh_t = cell_values(A, Z, B, rt.tie_r, rt.tie_c)
    xh_h = cell_values(A, Z, B, rt.hid_r, rt.hid_c)
    Am = A * rt.row_mask[:, None]
    if beta == 2:
        GA = Am.T @ Am
        GB = B.T @ B
        full = ((Z.T @ GA @ Z) * GB).sum()
        loss = rt.sum_x2 - 2.0 * (rt.tie_x * xh_t).sum() + full - (xh_h ** 2).sum()
        if rt.diag_masked:
            d = cell_values(A, Z, B, rt.diag_idx, rt.diag_idx)
            loss = loss - (d ** 2).sum()
        return loss
    if beta == 1:
        full = Am.sum(0) @ Z @ B.sum(0)
        loss = rt.sum_xlogx_minus_x - (rt.tie_x * torch.log(xh_t)).sum() + full - xh_h.sum()
        if rt.diag_masked:
            d = cell_values(A, Z, B, rt.diag_idx, rt.diag_idx)
            loss = loss - d.sum()
        return loss
    raise ValueError("beta must be 1 (generalised KL) or 2 (squared error)")


def total_loss(U, Z, rels, beta):
    """Σ_r w_r loss_r and the unweighted per-relation losses. U: facet → factor matrix; Z: relation name → core."""
    parts = {}
    tot = 0.0
    for rt in rels:
        l = relation_loss(U[rt.row_facet], Z[rt.name], U[rt.col_facet], rt, beta)
        parts[rt.name] = l
        tot = tot + rt.weight * l
    return tot, parts


def hidden_predictions(U, Z, rt):
    """x̂ at the hidden ties and hidden empty cells (for the held-out scores of F6)."""
    xh = cell_values(U[rt.row_facet], Z[rt.name], U[rt.col_facet], rt.hid_r, rt.hid_c)
    return xh[:rt.n_hid_ties], xh[rt.n_hid_ties:]
