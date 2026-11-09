# F2 — parameters of the M1 model (ticket 107): one factor matrix per facet, one core matrix per relation.
# Positivity by clamping after every optimiser step (CLAUDE.md §4.8, ticket 74: softplus starved the gradient).
# Normalised view (CLAUDE.md §4.2): U_scales = L2 column norms; U_norm = U / U_scales; Z_scaled = Z ⊙ (s_row s_colᵀ).
# The reconstruction U_norm Z_scaled U_normᵀ equals U Z Uᵀ exactly, so the data loss is computed from the raw
# parameters; the normalised view is what penalty terms and measures read (they must not depend on the scale freedom
# of ticket 79: multiplying column k of U_f by c and dividing the matching rows/columns of every Z touching f by c).
import torch

CLAMP_MIN = 1e-7


class Model(torch.nn.Module):
    def __init__(self, sizes, relations, K, init_U=None, init_Z=None, dtype=torch.float64, generator=None):
        """sizes: facet → number of entities (active facets only); relations: name → (row facet, column facet).
        init_U / init_Z: arrays to start from (F3 builds them); otherwise uniform random in (0, 1)."""
        super().__init__()
        self.K, self.relations = K, dict(relations)
        self.facets = sorted(sizes)                              # sorted: deterministic order (CLAUDE.md §4.9)
        self.U = torch.nn.ParameterDict()
        self.Z = torch.nn.ParameterDict()
        for f in self.facets:
            u = torch.as_tensor(init_U[f], dtype=dtype) if init_U is not None else \
                torch.rand(sizes[f], K, dtype=dtype, generator=generator)
            self.U[f] = torch.nn.Parameter(u.clone())
        for r in sorted(self.relations):
            z = torch.as_tensor(init_Z[r], dtype=dtype) if init_Z is not None else \
                torch.rand(K, K, dtype=dtype, generator=generator)
            self.Z[r] = torch.nn.Parameter(z.clone())
        self.clamp_()

    @torch.no_grad()
    def clamp_(self):
        for p in list(self.U.values()) + list(self.Z.values()):
            p.clamp_(min=CLAMP_MIN)

    def raw(self):
        return dict(self.U), dict(self.Z)

    def normalised(self):
        """U_scales, U_norm, Z_scaled (differentiable)."""
        scales = {f: self.U[f].norm(dim=0) for f in self.facets}
        U_norm = {f: self.U[f] / scales[f] for f in self.facets}
        Z_scaled = {r: self.Z[r] * torch.outer(scales[a], scales[b]) for r, (a, b) in self.relations.items()}
        return scales, U_norm, Z_scaled
