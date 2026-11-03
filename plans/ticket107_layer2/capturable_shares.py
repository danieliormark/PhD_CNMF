# Ticket 107, L2 (2026-11-02): best share of each matrix that K communities could capture if it were fitted alone,
# (s_1^2 + ... + s_K^2) / ||X||_F^2 (Eckart-Young bound; s_m = singular values), and the pull indicator
# weight x share under the decided weights (child×parent, child×article 0.25; each social matrix 1/6). Read-only.
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
F = {"cp": "parent_child", "ca": "child_article", "auart": "P1_author_article", "auaff": "P2_author_affiliation",
     "artaff": "P3_article_affiliation", "art2": "P5_article_article", "j": "P4_article_journal"}
KS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
S = {}
for k, f in F.items():
    X = sp.load_npz(D + f + ".npz").astype(float); T = X.multiply(X).sum()
    s = np.sort(svds(X, k=31, random_state=0, tol=1e-10)[1])[::-1] ** 2
    S[k] = np.cumsum(s) / T
print("best share capturable alone (cp child×parent, ca child×article, auart author×article, auaff author×affiliation,"
      " artaff article×affiliation, art2 article×article, j article×journal)\nK      " + " ".join(f"{k:>7}" for k in F))
for K in KS: print(f"{K:<6} " + " ".join(f"{S[k][K - 1]:7.4f}" for k in F))
print("\npull: semantic | appr1 social excl. journal | appr2 social excl. journal | journal | appr2/appr1 | "
      "semantic share appr1/appr2 | journal share of all, appr1 | ca/cp")
for K in KS:
    sem = .25 * (S["cp"][K-1] + S["ca"][K-1]); a1 = (S["auart"][K-1] + S["auaff"][K-1]) / 6
    a2 = (S["artaff"][K-1] + S["art2"][K-1]) / 6; j = S["j"][K-1] / 6
    print(f"K={K:<3} {sem:.4f} | {a1:.4f} | {a2:.4f} | {j:.4f} | {a2/a1:.2f} | {sem/(sem+a1):.0%}/{sem/(sem+a2):.0%} | "
          f"{j/(sem+a1+j):.0%} | {S['ca'][K-1]/S['cp'][K-1]:.2f}")
