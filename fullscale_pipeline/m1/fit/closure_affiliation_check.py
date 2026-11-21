# Author×affiliation closure above 1 (study A pass 3, RL-321: 1.13–1.40, rising with K) — where does the extra crossing
# weight come from? (owner 2026-11-21: "Start with the author×affiliation closure check"). Saved fits only, aligned
# numbering (fit/align.py), study A pass 3, approach 1, candidate cells K 5 r 0.3, K 13 r 1, K 20 r 1 (r*), all 5 groups
# × 10 seeds. Main community of an entity = its largest weighted membership. Closure ratio of a set of persons =
# (reconstructed tie weight from those persons to organisations of another main community ÷ their total reconstructed
# weight, over observed cells minus hidden cells) ÷ (the same share of their observed tie weight) — closure_check.one
# restricted to rows. Readings per person group:
#   - by number of organisations the person is tied to (1, 2, 3+; raw author×affiliation counts, all bylines);
#   - where the reconstructed crossing weight sits: on the person's observed ties vs on empty cells;
#   - membership spread (entropy, bits) of persons and of their organisations;
#   - organisations by size (persons tied): small / medium / large thirds, the same ratio from the organisation side.
# Approach 2 (owner 2026-11-21: "is there any difference in approach 1 vs approach 2 overprediction?"): --approach 2 reads
# article×affiliation (rows = articles; "persons" in the output = articles).
# Usage: python -m fit.closure_affiliation_check --out fit/results/stage0/closure_affiliation_check.json [--approach 2]
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json
from collections import defaultdict
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
CELLS = {(5, 0.3), (13, 1.0), (20, 1.0)}


def bits(P):
    return -(P * np.log2(np.clip(P, 1e-300, None))).sum(1)


def one(f, rd, n_org, org_size_band):
    U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
    wm = M.weighted_memberships(U, Z, rel)
    a, b = rd.row_facet, rd.col_facet; A, B, z = U[a], U[b], Z[rd.name]; K = A.shape[1]
    ma, mb = wm[a]["P"].argmax(1), wm[b]["P"].argmax(1)
    obs = np.nonzero(rd.obs_row_mask)[0]
    Bm = np.zeros((K, K)); np.add.at(Bm, mb, B)                      # per main community m: Σ_j in m B_j
    AZ = A[obs] @ z                                                  # persons × K
    to_main = AZ @ Bm.T                                              # reconstructed weight from person i to orgs of main m
    hid_r, hid_c = rd.hid_row, rd.hid_col                            # hidden cells are not part of the fitted cells
    xh = ((A[hid_r] @ z) * B[hid_c]).sum(1)
    pos = np.full(A.shape[0], -1); pos[obs] = np.arange(len(obs))
    keep = pos[hid_r] >= 0
    np.subtract.at(to_main, (pos[hid_r][keep], mb[hid_c][keep]), xh[keep])
    tot = to_main.sum(1); own = to_main[np.arange(len(obs)), ma[obs]]; rec_cross = tot - own
    # observed ties
    r, c, v = rd.tie_row, rd.tie_col, rd.tie_val
    cross_t = ma[r] != mb[c]
    xt = ((A[r] @ z) * B[c]).sum(1)
    dat_tot = np.bincount(r, weights=v, minlength=A.shape[0])[obs]; dat_cross = np.bincount(r, weights=v * cross_t, minlength=A.shape[0])[obs]
    rec_cross_on_ties = np.bincount(r, weights=xt * cross_t, minlength=A.shape[0])[obs]
    out = {}
    groups = {"1 organisation": n_org[obs] == 1, "2 organisations": n_org[obs] == 2, "3+ organisations": n_org[obs] >= 3,
              "all": np.ones(len(obs), bool)}
    Ha, Hb = bits(wm[a]["P"]), bits(wm[b]["P"])
    for g, m in groups.items():
        if not m.any():
            continue
        d = dat_cross[m].sum() / dat_tot[m].sum(); rcs = rec_cross[m].sum() / tot[m].sum()
        out[g] = {"persons": int(m.sum()), "tie_weight_share": float(dat_tot[m].sum() / dat_tot.sum()),
                  "data_crossing_share": float(d), "recon_crossing_share": float(rcs), "closure_ratio": float(rcs / d) if d > 0 else None,
                  "recon_crossing_on_ties_share": float(rec_cross_on_ties[m].sum() / max(rec_cross[m].sum(), 1e-300)),
                  "person_entropy": float(Ha[obs][m].mean())}
    # organisation side, by size band
    tot_c = np.bincount(c, weights=v, minlength=B.shape[0]); cr_c = np.bincount(c, weights=v * cross_t, minlength=B.shape[0])
    Am = np.zeros((K, K)); np.add.at(Am, ma[obs], A[obs])
    from_main = (Am @ z) @ B.T                                       # K × orgs (hidden cells not removed: small)
    tot_o = from_main.sum(0); own_o = from_main[mb, np.arange(B.shape[0])]
    for band in ("small", "medium", "large"):
        m = (org_size_band == band) & (tot_c > 0)
        d = cr_c[m].sum() / tot_c[m].sum(); rcs = (tot_o[m] - own_o[m]).sum() / tot_o[m].sum()
        out[f"organisations {band}"] = {"organisations": int(m.sum()), "data_crossing_share": float(d), "recon_crossing_share": float(rcs),
                                        "closure_ratio": float(rcs / d) if d > 0 else None, "organisation_entropy": float(Hb[m].mean())}
    return out


def run(approach=1):
    build = D.Build(D.BUILD)
    REL = "author_affiliation" if approach == 1 else "article_affiliation"
    raw = build.load("raw_P2_author_affiliation" if approach == 1 else "raw_P3_article_affiliation").tocsr()
    n_org = np.diff(raw.indptr)
    persons_per_org = np.asarray((raw > 0).sum(0)).ravel()
    q = np.quantile(persons_per_org[persons_per_org > 0], [1 / 3, 2 / 3])
    band = np.where(persons_per_org <= q[0], "small", np.where(persons_per_org <= q[1], "medium", "large"))
    fs = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass3", "*", "measures.json"))])
    fs = [f for f in fs if f["job"]["approach"] == approach and (f["job"]["K"], float(f["job"]["arm"]["r"])) in CELLS and f["result"]["converged"]]
    rds = {}
    res = defaultdict(list)
    for f in sorted(fs, key=lambda f: f["job"]["group"]):
        g = f["job"]["group"]
        if g not in rds:
            rds = {g: next(rd for rd in D.fit_data(build, approach, group=g) if rd.name == REL)}
        res[(f["job"]["K"], float(f["job"]["arm"]["r"]))].append(one(f, rds[g], n_org, band))
    rep = {"org_size_cuts_persons": q.tolist(), "persons_by_n_org": {str(int(k)): int(v) for k, v in zip(*np.unique(np.minimum(n_org, 3), return_counts=True))}}
    for cell, rows in sorted(res.items()):
        keys = rows[0].keys(); summ = {}
        for k in keys:
            summ[k] = {m: float(np.median([r[k][m] for r in rows if k in r and r[k][m] is not None])) for m in rows[0][k]}
        rep[f"K{cell[0]}_r{cell[1]:g}"] = {"fits": len(rows), "median": summ}
        print(f"K {cell[0]} r {cell[1]:g} ({len(rows)} fits) — medians over fits")
        for k, s in summ.items():
            print(f"   {k:22s} " + "  ".join(f"{m} {v:.3f}" if isinstance(v, float) else f"{m} {v}" for m, v in s.items()), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--approach", type=int, default=1)
    a = ap.parse_args()
    json.dump(run(a.approach), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
