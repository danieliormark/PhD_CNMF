# Ticket 107, L2.1 (2026-11-02): two entropy-based summaries per matrix: (E1) Shannon entropy of the cells,
# p_ij = x_ij^2 / ||X||^2, as an effective number of cells exp(H); (E2) Renyi-2 entropy of the layer strengths,
# p_m = s_m^2 / ||X||^2, as an effective number of layers ||X||^4 / sum s_m^4, with sum s_m^4 = ||X X^T||_F^2
# computed in row blocks (no full SVD needed). Read-only.
import numpy as np, scipy.sparse as sp
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
M = [("child×parent", "parent_child"), ("child×article", "child_article"), ("author×article", "P1_author_article"),
     ("author×affiliation", "P2_author_affiliation"), ("article×affiliation", "P3_article_affiliation"),
     ("article×journal", "P4_article_journal"), ("article×article", "P5_article_article")]
for name, f in M:
    X = sp.load_npz(D + f + ".npz").tocsr().astype(float)
    if X.shape[0] > X.shape[1]: X = X.T.tocsr()
    v = X.data ** 2; T = v.sum(); p = v / T
    Hc = -(p * np.log(p)).sum()
    s4 = 0.0
    for i in range(0, X.shape[0], 2000):
        B = (X[i:i + 2000] @ X.T).toarray(); s4 += (B * B).sum()
    print(f"{name:20s} ties {X.nnz:>7}  effective cells exp(H) {np.exp(Hc):9.0f} ({np.exp(Hc) / X.nnz:.2f} of ties)  "
          f"effective layers {T * T / s4:7.1f}")
