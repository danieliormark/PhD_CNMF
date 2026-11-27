# W4 follow-up (RL-416), data only: do same-author article pairs also share a journal / organisation, before and after the
# swap? The swap moves an article's whole social bundle, so the shares should be unchanged. Usage (from m1): PYTHONPATH=. python fit/w4_cochannel.py
import sys; sys.path.insert(0, ".")
import numpy as np, scipy.sparse as sp
from fit import w4_swap_analysis as W
from fit.align import fit_rds
from fit.swap_link_measure import incidence, pair_mask
fits = W.collect(W.ROOT); cache = {}
for key in [(1, 13, 0.0, 0.0), (1, 13, 0.0, 1.0), (2, 13, 0.0, 0.0), (2, 13, 0.0, 1.0)]:
    f = fits[key][min(fits[key])]; rds = {rd.name: rd for rd in fit_rds(f["job"], cache)}
    n = rds["child_article"].shape[0]
    if key[0] == 1:
        Ea = incidence(rds["author_article"], n); A2 = rds["author_affiliation"]
        Eo = sp.csr_matrix((Ea @ sp.csr_matrix((np.ones(len(A2.tie_row)), (A2.tie_row, A2.tie_col)), shape=A2.shape)) > 0, dtype=float)
    else:
        Ea = incidence(rds["article_article"], n); Eo = incidence(rds["article_affiliation"], n)
    Ej = incidence(rds["article_journal"], n)
    r, c = pair_mask(Ea)
    Sj = (Ej @ Ej.T).tocsr(); So = (Eo @ Eo.T).tocsr()
    sj = np.asarray(Sj[r, c]).ravel() > 0; so = np.asarray(So[r, c]).ravel() > 0
    npairs = n * (n - 1) / 2
    base_j = ((Sj > 0).sum() - (Sj.diagonal() > 0).sum()) / 2 / npairs
    base_o = ((So > 0).sum() - (So.diagonal() > 0).sum()) / 2 / npairs
    print(key, "articles", n, "same-author pairs", len(r), f"share also same journal {sj.mean():.3f} (all pairs {base_j:.4f})",
          f"share also same organisation {so.mean():.3f} (all pairs {base_o:.4f})", f"neither {(~sj & ~so).mean():.3f}", flush=True)
