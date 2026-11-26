# Swap measure "which people group with which ideas" (proposed 2026-11-18; owner 2026-11-21: "I approve W-c" — run it
# on the existing bare-swap fits first). The bare swap permutes the social identities (authors, organisations, journals,
# co-author ties) of a fraction p of articles, within social-degree bands; the semantic relations are untouched.
# For article pairs that share a social entity in the data the fit saw (approach 1: an author, a journal, an organisation
# through their authors; approach 2: a co-author tie, an organisation, a journal):
#   (A) data   — Jaccard similarity of the two articles' child-hyperedge sets (visible child×article ties), minus the
#                mean over all article pairs: does the link between people and ideas exist in the data the fit saw?
#                At p = 1 it must fall to about 0 (this also checks the swap).
#   (B) model  — co-membership of the two articles, Σ_k p_k q_k (weighted memberships), minus the mean over all pairs:
#                does the model put a person's articles together?
# Reading (proposal 2026-11-18): (A) clear and (B) falls with (A) → the model groups people's articles as far as they share
# ideas (heterogeneity follows the data); (B) stays high at p = 1 → communities follow people regardless of content;
# (B) ≈ 0 even at p = 0 → articles grouped by ideas alone; (A) weak → the data carry little link to test.
# Saved fits only: KL, K 13, hold-out group 1, seeds 1–10; p = 0 = pass-1 fits; study A and study B (M1_STUDY).
# Usage: [M1_STUDY=B] python -m fit.swap_link_measure --out fit/results/stage0/[B/]swap_link_measure.json
import argparse, glob, json, os
from collections import defaultdict
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits
from .align import fit_rds

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
_W = sp.load_npz(D.Build(D.BUILD).path + "W_child_child.npz").tocsr()
_d = np.asarray(_W.sum(1)).ravel(); _WN = (sp.diags(np.where(_d > 0, 1 / np.maximum(_d, 1e-300), 0)) @ _W).tocsr()


def incidence(rd, n_art):
    """Binary articles × entities matrix from a relation touching articles (visible ties)."""
    if rd.row_facet == "article" and rd.col_facet == "article":
        return sp.csr_matrix((np.ones(len(rd.tie_row)), (rd.tie_row, rd.tie_col)), shape=(n_art, n_art))
    if rd.row_facet == "article":
        return sp.csr_matrix((np.ones(len(rd.tie_row)), (rd.tie_row, rd.tie_col)), shape=(n_art, rd.shape[1]))
    return sp.csr_matrix((np.ones(len(rd.tie_row)), (rd.tie_col, rd.tie_row)), shape=(n_art, rd.shape[0]))


def pair_mask(E):
    """Upper-triangle pairs of articles sharing at least one entity (article×article: the tie itself)."""
    S = (E @ E.T).tocoo() if E.shape[0] != E.shape[1] or (E != E.T).nnz else E.tocoo()
    m = S.row < S.col
    return S.row[m], S.col[m]


def one(f, cache, rds=None):
    # rds: the fit's relation data, if the caller already has them (d1_analysis, 2026-11-26); J1 has no article×journal
    j = f["job"]; rds = {rd.name: rd for rd in (rds if rds is not None else fit_rds(j, cache))}
    ca = rds["child_article"]; n = ca.shape[0]
    X = sp.csr_matrix((np.ones(len(ca.tie_row)), (ca.tie_row, ca.tie_col)), shape=ca.shape)
    inter = (X @ X.T).toarray(); size = np.asarray(X.sum(1)).ravel()
    union = size[:, None] + size[None, :] - inter
    J = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    # W-aware semantic similarity (owner 2026-11-21: "did you account for the W-based similarity?"): each article's
    # child-hyperedge vector is extended by its hyperedges' W neighbours (x + x·Wn, Wn = W row-normalised), cosine
    Xw = X + X @ _WN; nw = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel())
    Jw = (Xw @ Xw.T).toarray() / np.maximum(np.outer(nw, nw), 1e-300)
    U, Z = load_UZ(f["dir"]); P = M.weighted_memberships(U, Z, {k: tuple(v) for k, v in f["result"]["relations"].items()})["article"]["P"]
    C = P @ P.T
    iu = np.triu_indices(n, 1); Jall, Call, Jwall = J[iu].mean(), C[iu].mean(), Jw[iu].mean()
    ent = {}
    if j["approach"] == 1:
        Ea = incidence(rds["author_article"], n); ent["author"] = Ea
        A2 = rds["author_affiliation"]
        Eo = (Ea @ sp.csr_matrix((np.ones(len(A2.tie_row)), (A2.tie_row, A2.tie_col)), shape=A2.shape)) > 0
        ent["organisation (via authors)"] = sp.csr_matrix(Eo, dtype=float)
    else:
        ent["co-author tie"] = incidence(rds["article_article"], n)
        ent["organisation"] = incidence(rds["article_affiliation"], n)
    if "article_journal" in rds:
        ent["journal"] = incidence(rds["article_journal"], n)
    out = {}
    for name, E in ent.items():
        r, c = pair_mask(E)
        out[name] = {"pairs": int(len(r)), "data_excess": float(J[r, c].mean() - Jall), "model_excess": float(C[r, c].mean() - Call),
                     "data_w_excess": float(Jw[r, c].mean() - Jwall),
                     "data_all": float(Jall), "model_all": float(Call)}
    return out


def run():
    stages = [D.stage("stage0_pass1"), D.stage("swap_bare")]
    fs = latest_fits([load_fit(os.path.dirname(p)) for s in stages for p in glob.glob(os.path.join(ROOT, s, "*", "measures.json"))])
    rows = defaultdict(list); cache = {}
    for f in sorted(fs, key=lambda f: (f["job"]["approach"], json.dumps(f["job"].get("arm") or {}, sort_keys=True))):
        j = f["job"]; arm = j.get("arm") or {}
        if j["beta"] != 1 or j["K"] != 13 or j["group"] != 0 or (j.get("terms") or {}) or not f["result"]["converged"]:
            continue
        if arm.get("swap_single_band") or any(k not in ("swap_p",) for k in arm) or j.get("social_link_start"):
            continue
        p = float(arm.get("swap_p", 0.0)); x = one(f, cache); x["_seed"] = j["seed"]; rows[(j["approach"], p)].append(x)
    rep = {}
    for (a, p), rs in sorted(rows.items()):
        ents = [e for e in rs[0] if not e.startswith("_")]
        rep.setdefault(f"approach_{a}", {})[str(p)] = {"fits": len(rs), **{e: {k: float(np.median([x[e][k] for x in rs])) for k in rs[0][e]} for e in ents},
                                                      "per_seed": {str(x["_seed"]): {e: x[e] for e in ents} for x in rs}}
    for a, byp in rep.items():
        print(a, "— median over seeds; excess over all article pairs (data: Jaccard of child-hyperedge sets; model: co-membership)")
        ents = [e for e in next(iter(byp.values())) if e not in ("fits", "per_seed")]
        for e in ents:
            d0, m0 = byp["0.0"][e]["data_excess"], byp["0.0"][e]["model_excess"]
            w0 = byp["0.0"][e]["data_w_excess"]
            print(f"   {e:28s} " + "  ".join(f"p {p}: sem {v[e]['data_excess']:+.4f} ({v[e]['data_excess'] / d0:4.2f}) semW {v[e]['data_w_excess']:+.4f} ({v[e]['data_w_excess'] / w0:4.2f}) model {v[e]['model_excess']:+.4f} ({v[e]['model_excess'] / m0:4.2f})"
                                               for p, v in byp.items()) + f"  [pairs at p 0: {byp['0.0'][e]['pairs']}]", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
