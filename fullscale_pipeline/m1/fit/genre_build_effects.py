# Study B (build without genre hyperedges; owner 2026-11-19, RL-290/291): what removing the 953 genre child hyperedges
# (final list, RL-289) does to the rest of the build, measured before writing study B's build rules. No fits; the
# builder's own record loading and unit rules (chunk12v3.load_records, semantic_units) are reused, then the genre
# hyperedges are removed from each unit as nested children and as cousins, and the consequences counted:
#   - parent hyperedges (identity = the multiset of its children's identities): left with no child (vanish); identity
#     changed; two or more study-A parents that become one study-B parent (merge);
#   - sentence units left with no parent and no cousin tied to a surviving parent (dropped by the builder's rule);
#     articles left with no unit or with no child×article tie;
#   - child×article: each article's share of ties and of raw usage removed (the partial row normalisation then raises
#     the remaining ties' weights);
#   - W (child×child): ties with a genre end, and substantive hyperedges whose W ties all go to genre hyperedges.
# Usage: python -m fit.genre_build_effects --out fit/results/stage0/genre_build_effects.json
import argparse, collections, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chunk12v3 as B
from . import data as D
from .generic_hyperedges import labels


def run():
    b = D.Build(D.BUILD); ids = b.index["children"]; lab = labels(b)
    genre = {k for k, l in zip(ids, lab) if l == "G"}
    trial = sorted(l.strip() for l in open(B.INPUTS["trial_list"]) if l.strip())
    log = {}; units = B.semantic_units(B.load_records(set(trial)), log)
    rep = {"genre_hyperedges": len(genre), "units_A": len(units)}
    # parents
    occ = collections.Counter(); newkey = {}; vanish_occ = 0; changed = set()
    for u in units:
        for pk, kids in u["parents"]:
            keep = [s for s in kids if B.content_key(s) not in genre]
            nk = B.content_key(B.content_key(s) for s in keep) if keep else None
            newkey[pk] = nk; occ[pk] += 1
            if nk is None: vanish_occ += 1
            elif len(keep) < len(kids): changed.add(pk)
    A = set(newkey); vanished = {p for p, n in newkey.items() if n is None}
    to = collections.defaultdict(set)
    for p, n in newkey.items():
        if n is not None: to[n].add(p)
    merged = {n: s for n, s in to.items() if len(s) > 1}
    rep["parents"] = {"A": len(A), "B": len(to), "vanish": len(vanished), "vanish_occurrences": vanish_occ,
                      "identity_changed_kept": len(changed), "B_parents_from_several_A": len(merged),
                      "A_parents_in_merges": int(sum(len(s) for s in merged.values())),
                      "largest_merge": max((len(s) for s in merged.values()), default=0)}
    # example: parents vanishing (all children genre) and merges
    ex_v = collections.Counter()
    ex_m = []
    for u in units:
        for pk, kids in u["parents"]:
            if pk in vanished: ex_v[" + ".join(" ".join(sorted(s)) for s in kids)] += 1
    rep["parents"]["vanish_examples"] = ex_v.most_common(15)
    # units and articles
    dropped_units = 0; art_units = collections.Counter(); art_units_B = collections.Counter()
    for u in units:
        art_units[u["pmcid"]] += 1
        surv = [pk for pk, _ in u["parents"] if newkey[pk] is not None]
        cous = [(s, [p for p in pks if newkey.get(p) is not None]) for s, pks in u["cousins"] if B.content_key(s) not in genre]
        if not surv and not any(pks for _, pks in cous):
            dropped_units += 1
        else:
            art_units_B[u["pmcid"]] += 1
    rep["units"] = {"dropped": dropped_units, "share": dropped_units / len(units)}
    rep["articles"] = {"A": len(art_units), "left_with_no_unit": len(set(art_units) - set(art_units_B))}
    # child×article
    TF = b.load("raw_child_article_tf").tocsr(); g = np.isin(np.array(ids), list(genre))
    r, c = TF.nonzero(); t = np.asarray(TF[r, c]).ravel(); isg = g[c]
    n_art = TF.shape[0]
    ties = np.bincount(r, minlength=n_art); gties = np.bincount(r, weights=isg, minlength=n_art)
    use = np.bincount(r, weights=t, minlength=n_art); guse = np.bincount(r, weights=t * isg, minlength=n_art)
    sh = gties / np.maximum(ties, 1); su = guse / np.maximum(use, 1)
    rep["child_article"] = {"ties_removed": int(isg.sum()), "share_of_ties": float(isg.mean()),
                            "articles_with_no_tie_left": int(((ties - gties) == 0).sum()),
                            "per_article_share_of_ties_removed": {q: float(np.quantile(sh, p)) for q, p in
                                                                  (("p10", .1), ("median", .5), ("p90", .9), ("max", 1))},
                            "per_article_share_of_raw_usage_removed_median": float(np.median(su))}
    # W
    W = b.load("W_child_child").tocoo(); m = W.row < W.col
    wr, wc = W.row[m], W.col[m]; ge = g[wr] | g[wc]
    deg = np.bincount(np.r_[wr, wc], minlength=len(ids))
    deg_s = np.bincount(np.r_[wr[~ge], wc[~ge]], minlength=len(ids))
    rep["W"] = {"ties": int(m.sum()), "ties_with_genre_end": int(ge.sum()), "share": float(ge.mean()),
                "substantive_hyperedges_with_W_ties": int(((deg > 0) & ~g).sum()),
                "substantive_losing_all_W_ties": int(((deg > 0) & (deg_s == 0) & ~g).sum())}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    rep = run(); json.dump(rep, open(ap.parse_args().out, "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
