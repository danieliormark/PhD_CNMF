# Ticket 107, L2.1 (2026-11-02): article×journal variants, computed exactly from journal sizes n_j (block structure):
# (i) binary partition: s^2 = n_j; (ii) same-journal article×article, ties 1: per journal (n-1)^2 once and 1 (n-1 times);
# (iii) ties 1/(n-1) (Newman 2001): 1 once and 1/(n-1)^2 (n-1 times); (iv) columns x 1/sqrt(n): 1 per journal;
# (iv') as (iv) with singleton journals missing; (v) column x idf, idf = ln(N/(n+1)) + 1: n * idf^2.
# Share at K = sum of the K largest s^2 / total; effective layers = total^2 / sum s^4. Read-only.
import numpy as np, scipy.sparse as sp
A = sp.load_npz("/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/P4_article_journal.npz").tocsc()
n = np.sort(np.diff(A.indptr))[::-1].astype(float); n = n[n > 0]; m = n[n >= 2]; N = 2136
print(f"journals {len(n)}, with >= 2 articles {len(m)}, singletons {(n == 1).sum()}, sizes top 10 {n[:10].astype(int).tolist()}")
print(f"same-journal pairs {int((n * (n - 1) / 2).sum())}, of which 5 largest journals {int((n[:5] * (n[:5] - 1) / 2).sum())}")
def rep(name, s2, extra=""):
    s2 = np.sort(s2)[::-1]; T = s2.sum()
    print(f"{name:55s} K=4 {s2[:4].sum()/T:.4f}  K=10 {s2[:10].sum()/T:.4f}  effective layers {T*T/(s2**2).sum():7.1f} {extra}")
rep("(i) binary partition (current)", n)
rep("(ii) same-journal article×article, ties 1", np.r_[(m - 1) ** 2, np.repeat(1., (m - 1).sum())])
rep("(iii) same-journal, ties 1/(n-1)", np.r_[np.ones(len(m)), np.concatenate([np.repeat(1 / (k - 1) ** 2, int(k - 1)) for k in m])])
rep("(iv) columns x 1/sqrt(n)", np.ones(len(n)), f"singleton share {(n == 1).sum() / len(n):.2f}")
rep("(iv') as (iv), singleton journals missing", np.ones(len(m)))
w2 = (np.log(N / (n + 1)) + 1) ** 2
rep("(v) column x idf", n * w2, f"singleton share {w2[n == 1].sum() / (n * w2).sum():.3f}")
