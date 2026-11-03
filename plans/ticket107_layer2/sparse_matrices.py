# Ticket 107, L2.4 (2026-11-02): facts about the two sparse matrices. author×article: persons with one article and their
# share of the squared total; capturable share at K = 4 with only persons with 2+ articles; how much of author×affiliation
# and of the article–organisation links through persons rests on single-article persons. article×article: pairs, degree,
# concentration of the squared total. Weight per effective tie = rule weight / exp(cell entropy) (entropy_measures.txt).
# Read-only; prints counts only.
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
D = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/"
P1 = sp.load_npz(D + "P1_author_article.npz").tocsr().astype(float); P2 = sp.load_npz(D + "P2_author_affiliation.npz").tocsr()
deg = np.diff(P1.indptr); v2 = np.asarray(P1.multiply(P1).sum(1)).ravel()
print(f"author×article: persons {(deg > 0).sum()}, with 1 article {(deg == 1).sum()} ({(deg == 1).mean():.3f}), "
      f"their share of the squared total {v2[deg == 1].sum() / v2.sum():.3f}, persons with 2+ {(deg >= 2).sum()}")
for name, X in (("all persons", P1), ("persons with 2+ articles", P1[deg >= 2])):
    T = X.multiply(X).sum(); s = np.sort(svds(X, k=10, random_state=0)[1])[::-1] ** 2
    print(f"  {name}: ties {X.nnz}, share at K = 4 {s[:4].sum() / T:.4f}, K = 10 {s[:10].sum() / T:.4f}")
d2 = np.diff(P2.indptr)
print(f"author×affiliation ties {P2.nnz}, of single-article persons {d2[deg == 1].sum()} ({d2[deg == 1].sum() / P2.nnz:.3f}); "
      f"organisations tied {len(set(P2.indices))}, reachable via 2+-article persons only {len(set(P2[deg >= 2].indices))}; "
      f"article–organisation links via persons {(P1.T @ P2).nnz}, via 2+-article persons only {(P1[deg >= 2].T @ P2[deg >= 2]).nnz}")
P5 = sp.load_npz(D + "P5_article_article.npz").tocsr(); d = np.diff(P5.indptr)
u = sp.triu(P5, 1).tocoo(); x = np.sort(u.data ** 2)[::-1]
print(f"article×article: pairs {len(x)}, tied articles {(d > 0).sum()}, mean degree {d.mean():.2f} ({d[d > 0].mean():.2f} among tied), "
      f"degree 1: {(d == 1).sum()}; top 1% / 10% of pairs hold {x[:len(x) // 100].sum() / x.sum():.3f} / {x[:len(x) // 10].sum() / x.sum():.3f}")
eff = {"child×parent": (0.25, 179081), "child×article": (0.25, 123266), "author×article": (1 / 6, 9065),
       "author×affiliation": (1 / 6, 11351), "article×affiliation": (1 / 6, 3469), "article×journal": (1 / 6, 2132),
       "article×article": (1 / 6, 691)}
base = 0.25 / 179081
print("weight per effective tie (relative to child×parent): " + ", ".join(f"{m} {w / n:.1e} ({w / n / base:.0f}×)" for m, (w, n) in eff.items()))
