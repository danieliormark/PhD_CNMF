# Same word, different atom type (ticket 107; owner 2026-11-19: "you have individual entries for same words with
# different roles like c or p. Do I understand that same word in different role is treated as separate entities? If so
# we may better check how does it affects the outcome"). In the build an atom's type is part of its identity (design doc
# §1: help/P ≠ help/C; types: P predicate, C concept, M modifier), so use/P and use/C are two child hyperedges, and W
# (child×child word overlap) also compares typed atoms, so nothing in the data links them. No fits; full build (all
# visible and hidden ties, i.e. the matrix before the hold-out).
#   (1) build: child hyperedges whose bare word sets coincide ("typed forms" of one word set); their share of ties, raw
#       usage and weighted values; per word set: articles using it after a merge vs using its most used form.
#   (2) article pairs that share at least one child hyperedge, before vs after merging typed forms (all hyperedges;
#       substantive only; genre only), pairs counted once.
#   (3) G/S classification: typed rule (the word lists apply per type) vs type-blind rule (by bare word), after the
#       owner's review; hyperedges whose class differs, and the class shares under each.
# Usage: python -m fit.typed_forms --out fit/results/stage0/typed_forms.json
import argparse, json
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from . import data as D
from .generic_hyperedges import labels


def pairs_sharing(X):
    """Number of unordered article pairs with at least one shared column of the binary matrix X (articles × columns)."""
    C = (X @ X.T).tocsr(); C.setdiag(0); C.eliminate_zeros()
    return int(C.nnz // 2)


def run():
    b = D.Build(D.BUILD); ids = b.index["children"]; at = b.index["child_atoms"]
    TF = sp.csr_matrix(b.load("raw_child_article_tf")); TF.eliminate_zeros()
    X = (TF > 0).astype(np.int32).tocsc()
    df = np.asarray(X.sum(0)).ravel(); used = df > 0
    lt, lb = labels(b), labels(b, type_blind=True)
    key = [frozenset(a.split("/")[0] for a in at[i]) for i in ids]
    grp = defaultdict(list)
    for j in np.nonzero(used)[0]:
        grp[key[j]].append(j)
    gid = {k: n for n, k in enumerate(grp)}; col = np.array([gid[key[j]] if used[j] else -1 for j in range(len(ids))])
    multi = {k: g for k, g in grp.items() if len(g) > 1}
    inm = np.zeros(len(ids), bool)
    for g in multi.values():
        inm[g] = True
    r, c = TF.nonzero(); tf = np.asarray(TF[r, c]).ravel()
    rep = {"used_hyperedges": int(used.sum()), "bare_word_sets": len(grp), "word_sets_with_several_forms": len(multi),
           "hyperedges_in_them": int(inm.sum()), "share_of_ties": float(inm[c].mean()),
           "share_of_raw_usage": float(tf[inm[c]].sum() / tf.sum())}
    # articles reached by a merge, per word set
    rows = []
    for k, g in multi.items():
        sets = [set(X[:, j].indices) for j in g]; u = set().union(*sets); mx = max(len(s) for s in sets)
        rows.append({"forms": [" ".join(at[ids[j]]) for j in sorted(g, key=lambda j: -df[j])],
                     "df_forms": sorted((int(df[j]) for j in g), reverse=True), "df_merged": len(u),
                     "articles_not_in_main_form": len(u) - mx, "classes_typed": sorted(set(lt[g])),
                     "classes_type_blind": sorted(set(lb[g]))})
    rows.sort(key=lambda x: -x["articles_not_in_main_form"])
    rep["articles_not_in_main_form_total"] = int(sum(x["articles_not_in_main_form"] for x in rows))
    rep["word_sets_top"] = rows[:60]
    # (2) article pairs sharing ≥ 1 hyperedge, before / after merging typed forms
    M = sp.csr_matrix((np.ones(used.sum()), (np.nonzero(used)[0], col[used])), shape=(len(ids), len(grp)))
    Xr = X.tocsr()
    for name, keep_typed in (("all", used), ("substantive_typed_rule", used & (lt == "S")),
                             ("genre_typed_rule", used & (lt == "G"))):
        Xk = Xr[:, np.nonzero(keep_typed)[0]]
        Mk = M[np.nonzero(keep_typed)[0]]
        before = pairs_sharing(Xk); after = pairs_sharing(((Xk @ Mk) > 0).astype(np.int32))
        rep[f"pairs_{name}"] = {"before": before, "after": after, "gain": (after - before) / max(before, 1)}
    n = X.shape[0]; rep["article_pairs_total"] = n * (n - 1) // 2
    # (3) classification typed vs type-blind
    for nm, lab in (("typed", lt), ("type_blind", lb)):
        g = lab == "G"
        rep[f"class_{nm}"] = {"G_hyperedges": int((g & used).sum()), "G_share_of_ties": float(g[c].mean()),
                              "G_share_of_raw_usage": float(tf[g[c]].sum() / tf.sum())}
    diff = np.nonzero(used & (lt != lb))[0]
    rep["class_differs"] = {"hyperedges": int(len(diff)), "share_of_ties": float(np.isin(c, diff).mean()),
                            "examples": [{"hyperedge": " ".join(at[ids[j]]), "articles": int(df[j]), "typed": lt[j],
                                          "type_blind": lb[j]} for j in diff[np.argsort(-df[diff])][:40]]}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    rep = run(); json.dump(rep, open(ap.parse_args().out, "w"), indent=1)
    for k, v in rep.items():
        if k not in ("word_sets_top", "class_differs"):
            print(k, v)
    print("class differs:", {k: v for k, v in rep["class_differs"].items() if k != "examples"})
    for e in rep["class_differs"]["examples"]:
        print("   ", e)
    print("word sets by articles not in the main form:")
    for x in rep["word_sets_top"][:30]:
        print("   ", x["articles_not_in_main_form"], x["df_merged"], x["df_forms"], " | ".join(x["forms"]), x["classes_typed"], x["classes_type_blind"])


if __name__ == "__main__":
    main()
