# Soft readings for the K = 32 failing child×article communities (ticket 107; owner 2026-11-19: "under soft partition
# articles can have attributions to different communities and we even discussed some measures useful for soft
# partition"). Companion to shared_child_community_profile.py, whose "main ..." rows are argmax labels. Here every
# entity counts with its weighted membership (GLOSSARY "Measures on membership distributions"):
#   soft size of community c in a facet = Σ_i p_ic (expected number of members);
#   membership-weighted mean document frequency of c's hyperedges = Σ_j p_jc df_j / Σ_j p_jc;
#   membership-weighted mean size of c's journals (visible articles per journal);
#   profile overlap (Σ_k min) between a hyperedge's own membership and the mean membership of the articles using it,
#     averaged over hyperedges with weight p_jc — low = its users sit in other communities than the hyperedge itself;
#   co-membership Σ_k p_k q_k of a child×article tie's two ends, tie-weighted, for ties whose hyperedge has membership
#     in c (weight p_jc), against chance (product of the two facets' mean memberships).
# Usage: python -m fit.shared_child_community_soft --out fit/results/stage0/shared_child_community_soft.json
import argparse, glob, json, os
import numpy as np
import scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def readings(f, rds):
    U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
    P = {x: v["P"] for x, v in M.weighted_memberships(U, Z, rel).items()}
    ca = rds["child_article"]; parts, _ = M.parts(U, Z, ca); d = np.diag(parts)
    fail = ~((d / np.maximum(parts.sum(1), 1e-300) > 0.5) & (d / np.maximum(parts.sum(0), 1e-300) > 0.5))
    if not fail.any():
        return None
    r, c, v = ca.tie_row, ca.tie_col, ca.tie_val
    df = np.bincount(c, minlength=ca.shape[1]).astype(float)
    X = sp.csr_matrix((np.ones_like(v), (c, r)), shape=(ca.shape[1], ca.shape[0]))      # child × article, 1 per tie
    users = X @ P["article"]; users = users / np.maximum(users.sum(1, keepdims=True), 1e-300)  # mean article profile
    ov = np.minimum(users, P["child"]).sum(1)
    co = (P["article"][r] * P["child"][c]).sum(1)
    chance = float((P["article"].mean(0) * P["child"].mean(0)).sum())
    aj = rds["article_journal"]; js = np.bincount(aj.tie_col, minlength=aj.shape[1]).astype(float)
    out = []
    for k in range(Z["child_article"].shape[0]):
        w = P["child"][:, k] * (df > 0); wt = P["child"][c, k] * v; wj = P["journal"][:, k] * (js > 0)
        out.append({"fails": bool(fail[k]), "soft_hyperedges": float(w.sum()), "soft_articles": float(P["article"][:, k].sum()),
                    "soft_journals": float(wj.sum()), "df_weighted_mean": float((w * df).sum() / w.sum()),
                    "journal_size_weighted_mean": float((wj * js).sum() / wj.sum()) if wj.sum() > 0 else None,
                    "user_overlap": float((w * ov).sum() / w.sum()), "co_membership_ties": float((wt * co).sum() / wt.sum()),
                    "co_membership_chance": chance})
    return out


def run():
    build = D.Build(D.BUILD); rep = {}
    fs_all = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass1", "*", "measures.json"))]
    for a in (1, 2):
        fs = [f for f in latest_fits(fs_all) if f["job"]["beta"] == 1 and f["job"]["K"] == 32 and f["job"]["approach"] == a
              and not f["job"].get("arm") and not (f["job"].get("terms") or {})]
        F, O = [], []
        for g in sorted({f["job"]["group"] for f in fs}):
            rds = {rd.name: rd for rd in D.fit_data(build, a, group=g)}
            for f in [f for f in fs if f["job"]["group"] == g]:
                q = readings(f, rds)
                for x in q or []:
                    (F if x["fails"] else O).append(x)
        keys = [k for k in F[0] if k != "fails"]
        med = lambda L, k: float(np.median([x[k] for x in L if x[k] is not None]))
        rep[f"approach_{a}"] = {"failing": {k: med(F, k) for k in keys}, "other": {k: med(O, k) for k in keys}, "n_failing": len(F)}
        print(f"approach {a} ({len(F)} failing communities): median failing vs other communities of the same fits")
        for k in keys:
            print(f"   {k:28s} {rep[f'approach_{a}']['failing'][k]:10.4g} {rep[f'approach_{a}']['other'][k]:10.4g}")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
