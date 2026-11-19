# Generic child hyperedges — step 1: draft classification and review list (ticket 107; owner 2026-11-19: "I approve
# all 3 points": (1) inventory and classification, (2) their role in the existing fits, (3) a robustness arm only if
# 1–2 show they matter). The DRAFT rule below is for the owner's review; the owner's corrections replace it.
# Classes of a child hyperedge (its typed atoms, e.g. "provide/P/en"):
#   F  focal       — every atom is a focal atom (role "focal": the corpus's canonical topic terms, e.g. llm, bert);
#   G  genre       — every atom is in GENRE (vocabulary of scientific writing: reporting / argument verbs, generic
#                    research nouns, auxiliaries, modals, negation, possessives, generic modifiers);
#   ?  ambiguous   — every atom is in GENRE or AMBIGUOUS, at least one in AMBIGUOUS (generic in form but possibly a
#                    method or object of this corpus: train, predict, classify, model, performance, ...);
#   S  substantive — anything else (at least one atom outside these lists, focal atoms combined with others included).
# Writes the review list (hyperedges used by ≥ 50 articles, all visible ties of the trial build) and the shares of
# ties / raw usage / weighted values per class over ALL child hyperedges.
# Usage: python -m fit.generic_hyperedges --review ../../plans/ticket107_generic_hyperedges_review.md
#            --out fit/results/stage0/generic_hyperedges.json
import argparse, json
import numpy as np
from . import data as D

GENRE_P = set("""use be show provide demonstrate include have suggest compare achieve indicate make base find offer
leverage remain develop combine apply integrate explore follow highlight employ improve present represent see introduce
report identify assess utilize enhance perform focus examine propose produce require address support conduct reveal
involve capture ensure test enable create incorporate exhibit aim build consider investigate become rely emerge observe
obtain analyze reflect increase take select result relate design reduce lead give understand exist describe note lack
adopt serve yield associate contain illustrate contribute underscore implement refer extend allow derive appear raise
establish guide validate consist adapt add determine receive begin confirm emphasize correspond know underlie maintain do
engage range seek choose facilitate define calculate call regard deliver draw need grow reach vary limit explain operate
excel differ construct collect compute mean discuss comprise share display summarize seem express expand gain assign run
hold interpret come depend work surpass expect drive evaluate outperform align process power write measure ask""".split())
GENRE_C = set("""study result finding example research researcher work datum potential case question way method approach
analysis paper information author section possible percent hand order extent framework evidence particular high
challenge summary gap literature difference tool addition instance time context limitation important fact process
experiment pattern""".split())
GENRE_M = set("""be have do more well most our recent future present can may would will not this""".split())
AMBIG_P = set("""generate train fine_tune predict classify detect simulate prompt embed pretraine label rate score respond
interact perceive automate extract optimize learn""".split())
AMBIG_C = set("""model system performance ability dataset task feature input output""".split())


def classify(atoms):
    parts = [a.split("/") for a in atoms]
    if all(len(p) > 2 and p[2] == "focal" for p in parts):
        return "F"
    def kind(p):
        w, t = p[0], (p[1][:1] if len(p) > 1 else "")
        if (t == "P" and w in GENRE_P) or (t == "C" and w in GENRE_C) or (t == "M" and w in GENRE_M):
            return "G"
        if (t == "P" and w in AMBIG_P) or (t == "C" and w in AMBIG_C):
            return "?"
        return "S"
    ks = {kind(p) for p in parts}
    if ks == {"G"}:
        return "G"
    if ks <= {"G", "?"}:
        return "?"
    return "S"


def labels(build):
    ids = build.index["children"]; at = build.index["child_atoms"]
    return np.array([classify(at[i]) for i in ids])


def run(review_path):
    build = D.Build(D.BUILD); ids = build.index["children"]; at = build.index["child_atoms"]
    ca = next(rd for rd in D.fit_data(build, 1, group=0) if rd.name == "child_article")
    r, c, v = ca.tie_row, ca.tie_col, ca.tie_val
    tf = np.asarray(build.load("raw_child_article_tf")[r, c]).ravel().astype(float)
    lab = labels(build); df = np.bincount(c, minlength=len(ids))
    rep = {"by_class": {}}
    for k in ("G", "F", "?", "S"):
        m = lab == k; mt = m[c]
        rep["by_class"][k] = {"hyperedges": int((m & (df > 0)).sum()), "ties": float(mt.mean()),
                              "raw_usage": float(tf[mt].sum() / tf.sum()), "weighted": float(v[mt].sum() / v.sum()),
                              "single_word": int((m & (df > 0) & np.array([len(at[i]) == 1 for i in ids])).sum())}
    top = [i for i in np.argsort(-df) if df[i] >= 50]
    rep["review_list"] = [{"hyperedge": " ".join(at[ids[i]]), "articles": int(df[i]), "draft": lab[i]} for i in top]
    with open(review_path, "w") as fh:
        fh.write("# Generic child hyperedges — draft classification for the owner's review (ticket 107, 2026-11-19)\n\n")
        fh.write("Hyperedges used by at least 50 articles (trial build, hold-out group 1, visible ties). Draft class: "
                 "**G** genre of scientific writing, **F** focal topic terms only, **?** ambiguous (generic in form, "
                 "possibly a method or object of this corpus), **S** substantive. Rule and word lists: "
                 "`fullscale_pipeline/m1/fit/generic_hyperedges.py`. To correct: change the class in the last column "
                 "(or name words to move between lists); the rule is then re-run on all hyperedges.\n\n")
        fh.write("| Class | Hyperedges | Single-word | Share of ties | Share of raw usage | Share of weighted values |\n|---|---|---|---|---|---|\n")
        for k, x in rep["by_class"].items():
            fh.write(f"| {k} | {x['hyperedges']} | {x['single_word']} | {x['ties']:.1%} | {x['raw_usage']:.1%} | {x['weighted']:.1%} |\n")
        fh.write("\n| # | Hyperedge (typed atoms) | Articles using it | Draft class | Owner's class |\n|---|---|---|---|---|\n")
        for n, x in enumerate(rep["review_list"], 1):
            fh.write(f"| {n} | `{x['hyperedge']}` | {x['articles']} | {x['draft']} | |\n")
    for k, x in rep["by_class"].items():
        print(k, x)
    from collections import Counter
    print("review list:", len(rep["review_list"]), Counter(x["draft"] for x in rep["review_list"]))
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--review", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(a.review), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
