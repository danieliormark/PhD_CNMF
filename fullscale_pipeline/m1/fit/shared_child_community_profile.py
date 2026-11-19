# Profile of the child communities that fail the child×article requirement at K = 32 (ticket 107; owner 2026-11-19:
# "You can make this analysis, but I'd notice that idf weighting reflects rather our assumptions than reality. In
# "reality" with no weighting applied most frequently used hyperedges ... should have larger values/weights"). RL-281:
# at K = 32 some KL fits have one child community whose hyperedges are used by the articles of most other communities.
# Two readings to separate: (a) a socially large community whose vocabulary others borrow; (b) a catch-all of generic
# journals and generic vocabulary. Saved fits only (pass 1, KL, K = 32, all groups, seeds 1–10), FITTED numbering.
# Per failing community c, beside the median over the other communities of the same fit:
#   - its main hyperedges (largest weighted membership = c): number; their document frequency (visible ties; df) and
#     share with df = 1;
#   - how evenly article communities use its hyperedges: effective number of article communities exp(H) over its
#     child-side column of the parts table, and its own article community's share of that column;
#   - its size read two ways: each visible child×article tie is split among community pairs in proportion to their
#     contribution to the tie's prediction (KL responsibilities); summed with the fit's weighted values (as the parts
#     table) and with the raw term frequencies (no idf, no normalisation) → the community's share of the corpus's
#     weighted and of its raw article–hyperedge usage (child side, article side, half credit);
#   - its articles, journals (number, and size = visible articles per journal), authors / organisations (main members);
#   - same group across fits: Jaccard overlap of the 100 hyperedges with the largest membership in c, between failing
#     communities of different fits, against the same between randomly chosen other communities.
# Usage: python -m fit.shared_child_community_profile --out fit/results/stage0/shared_child_community_profile.json
import argparse, glob, json, os
from itertools import combinations
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def eff_n(w):
    p = w / w.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def profile(f, rds, raw_tf, build):
    U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
    wm = M.weighted_memberships(U, Z, rel); P = {x: v["P"] for x, v in wm.items()}
    ca = rds["child_article"]; A, B, z = U["article"], U["child"], Z["child_article"]; K = z.shape[0]
    parts, T = M.parts(U, Z, ca); d = np.diag(parts)
    fail = ~((d / np.maximum(parts.sum(1), 1e-300) > 0.5) & (d / np.maximum(parts.sum(0), 1e-300) > 0.5))
    if not fail.any():
        return None
    r, c, v = ca.tie_row, ca.tie_col, ca.tie_val; tf = raw_tf
    AZ = A[r] @ z                                              # ties × K: Σ_k A_ik Z_kl, per column community l
    xh = (AZ * B[c]).sum(1)
    col_resp = AZ * B[c] / xh[:, None]                         # share of each tie's prediction through child community l
    ZB = B[c] @ z.T                                            # ties × K: Σ_l Z_kl B_jl, per row community k
    row_resp = A[r] * ZB / xh[:, None]
    df = np.bincount(c, minlength=ca.shape[1]); main_child = P["child"].argmax(1)
    tied = df > 0
    out = []
    jsize = np.bincount(rds["article_journal"].tie_col, minlength=rds["article_journal"].shape[1])
    for k in range(K):
        col = parts[:, k]
        q = {"child_side_weighted": float((v * col_resp[:, k]).sum() / v.sum()),
             "child_side_raw": float((tf * col_resp[:, k]).sum() / tf.sum()),
             "article_side_weighted": float((v * row_resp[:, k]).sum() / v.sum()),
             "article_side_raw": float((tf * row_resp[:, k]).sum() / tf.sum())}
        q["half_credit_weighted"] = 0.5 * (q["child_side_weighted"] + q["article_side_weighted"])
        q["half_credit_raw"] = 0.5 * (q["child_side_raw"] + q["article_side_raw"])
        mh = np.nonzero(tied & (main_child == k))[0]
        mj = np.nonzero(P["journal"].argmax(1) == k)[0]; mj = mj[jsize[mj] > 0]
        q.update({"k": k, "fails": bool(fail[k]), "main_hyperedges": int(len(mh)),
                  "df_median": float(np.median(df[mh])) if len(mh) else None,
                  "df_mean": float(df[mh].mean()) if len(mh) else None,
                  "df1_share": float((df[mh] == 1).mean()) if len(mh) else None,
                  "users_effective": eff_n(col) if col.sum() > 0 else None,
                  "own_share_child_side": float(col[k] / col.sum()) if col.sum() > 0 else None,
                  "main_articles": int((P["article"].argmax(1) == k).sum()),
                  "main_journals": int(len(mj)), "journal_size_median": float(np.median(jsize[mj])) if len(mj) else None,
                  "journal_tie_share": float(jsize[mj].sum() / jsize.sum()),
                  "top100": np.argsort(-P["child"][:, k] * tied)[:100].tolist()})
        for fa in ("author", "organisation"):
            if fa in P:
                q[f"main_{fa}s"] = int((P[fa].argmax(1) == k).sum())
        out.append(q)
    return out


def run():
    build = D.Build(D.BUILD); rep = {}
    raw = build.load("raw_child_article_tf")
    for a in (1, 2):
        fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass1", "*", "measures.json"))]
        fs = [f for f in latest_fits(fs) if f["job"]["beta"] == 1 and f["job"]["K"] == 32 and f["job"]["approach"] == a
              and not f["job"].get("arm") and not (f["job"].get("terms") or {})]
        rows, fails, others = [], [], []
        for g in sorted({f["job"]["group"] for f in fs}):
            rds = {rd.name: rd for rd in D.fit_data(build, a, group=g)}
            tf = np.asarray(raw[rds["child_article"].tie_row, rds["child_article"].tie_col]).ravel().astype(float)
            for f in [f for f in fs if f["job"]["group"] == g]:
                pr = profile(f, rds, tf, build)
                if pr is None:
                    continue
                for q in pr:
                    q.update(group=g, seed=f["job"]["seed"])
                    (fails if q["fails"] else others).append(q)
        keys = ["main_hyperedges", "df_median", "df_mean", "df1_share", "users_effective", "own_share_child_side",
                "child_side_weighted", "child_side_raw", "article_side_weighted", "article_side_raw",
                "half_credit_weighted", "half_credit_raw", "main_articles", "main_journals", "journal_size_median",
                "journal_tie_share", "main_authors", "main_organisations"]
        med = lambda L, k: (float(np.median([q[k] for q in L if q.get(k) is not None])) if any(q.get(k) is not None for q in L) else None)
        rng = np.random.default_rng(0)
        jac = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
        jf = [jac(x["top100"], y["top100"]) for x, y in combinations(fails, 2) if (x["group"], x["seed"]) != (y["group"], y["seed"])]
        jo = []
        for _ in range(500):
            x, y = rng.choice(len(others), 2, replace=False)
            if (others[x]["group"], others[x]["seed"]) != (others[y]["group"], others[y]["seed"]):
                jo.append(jac(others[x]["top100"], others[y]["top100"]))
        rep[f"approach_{a}"] = {"failing_communities": len(fails), "fits_with_failure": len({(q["group"], q["seed"]) for q in fails}),
                                "failing": {k: med(fails, k) for k in keys}, "other_communities_same_fits": {k: med(others, k) for k in keys},
                                "top100_jaccard_failing_across_fits": float(np.median(jf)) if jf else None,
                                "top100_jaccard_other_across_fits": float(np.median(jo)) if jo else None,
                                "per_failing": [{k: q.get(k) for k in keys + ["group", "seed", "k"]} for q in fails]}
        # example hyperedges: the 15 most widely used among the failing community's main hyperedges (first failing fit)
        if fails:
            q = fails[0]; g = q["group"]
            rds = {rd.name: rd for rd in D.fit_data(build, a, group=g)}
            f = next(f for f in fs if f["job"]["group"] == g and f["job"]["seed"] == q["seed"])
            U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
            Pc = M.weighted_memberships(U, Z, rel)["child"]["P"]
            df = np.bincount(rds["child_article"].tie_col, minlength=rds["child_article"].shape[1])
            mh = np.nonzero((Pc.argmax(1) == q["k"]) & (df > 0))[0]
            top = mh[np.argsort(-df[mh])[:15]]
            ids = build.index["children"]
            rep[f"approach_{a}"]["example_hyperedges"] = [{"atoms": build.index["child_atoms"][ids[i]], "df": int(df[i])} for i in top]
        R = rep[f"approach_{a}"]
        print(f"approach {a}: {R['failing_communities']} failing communities in {R['fits_with_failure']} fits; median failing vs other communities of the same fits")
        for k in keys:
            print(f"   {k:26s} {R['failing'][k]!s:>10.10} {R['other_communities_same_fits'][k]!s:>10.10}")
        print(f"   top-100 hyperedge overlap across fits (Jaccard): failing {R['top100_jaccard_failing_across_fits']:.3f}, other {R['top100_jaccard_other_across_fits']:.3f}")
        for e in R.get("example_hyperedges", []):
            print("   e.g. df", e["df"], " ".join(e["atoms"]))
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
