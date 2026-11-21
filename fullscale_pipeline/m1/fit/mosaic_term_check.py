# Mosaic vs blend under the word-similarity term (owner 2026-11-21: "Please present the mosaic vs blend point"). S3 (the
# spread ceiling) was dropped on 2026-11-16 because at λ = 0 entities were LESS spread than their relation mixes
# (RL-239). Pass 3 (RL-321) shows child hyperedges' own spread rising with r. This reading separates the term's effect:
# W-tied child hyperedges (the term acts on them) vs untied ones (it does not), at r = 0 / 0.3 / 1, same K, same seeds.
# Saved fits only: study A pass 3, K 5 and 13, both approaches, hold-out group 1, seeds 1–10. Per child hyperedge:
#   own spread H(p) (bits); excess spread = own spread − the largest spread of its relation mixes (measures.excess_spread,
#   ρ = 1: the mix of the memberships of its articles, and of its parents, weighted by tie values) — > 0 means the idea
#   is more mixed than every context it is observed in (a blend the data do not show);
#   held-out existence of child×article hidden ties split by the hyperedge being W-tied or not (within-class AP and its
#   chance, as the decided score) — does spreading help or hurt prediction of where tied ideas are used.
# Usage: python -m fit.mosaic_term_check --out fit/results/stage0/mosaic_term_check.json
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def run():
    build = D.Build(D.BUILD)
    W = sp.load_npz(build.path + "W_child_child.npz").tocsr(); tied = np.diff(W.indptr) > 0
    fs = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass3", "*", "measures.json"))])
    fs = [f for f in fs if f["job"]["group"] == 0 and f["job"]["K"] in (5, 13) and f["result"]["converged"]]
    rep = defaultdict(list)
    for a in (1, 2):
        rds = D.fit_data(build, a, group=0); ca = next(rd for rd in rds if rd.name == "child_article")
        sem = [rd for rd in rds if rd.name in ("child_article", "parent_child")]
        n_empty_obs = int(ca.obs_row_mask.sum()) * ca.shape[1] - len(ca.tie_row) - len(ca.hid_tie_row)
        scale = math.floor(0.10 * n_empty_obs) / len(ca.hid_empty_row)
        df = np.bincount(ca.tie_col, minlength=ca.shape[1])
        for f in [f for f in fs if f["job"]["approach"] == a]:
            U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
            wm = M.weighted_memberships(U, Z, rel)
            exc = M.excess_spread(sem, wm, "child"); own = M.entropy_bits(wm["child"]["P"])
            shared = df >= 2                                      # ideas used by ≥ 2 articles (visible ties)
            row = {}
            for lab, m in (("tied", tied & shared), ("untied", ~tied & shared)):
                e = exc[m & np.isfinite(exc)]
                row[lab] = {"n": int(m.sum()), "own_spread_median": float(np.median(own[m])),
                            "excess_median": float(np.median(e)), "share_excess_positive": float((e > 0).mean()),
                            "share_excess_above_0.5bit": float((e > 0.5).mean())}
            A, B, z = U["article"], U["child"], Z["child_article"]
            ht = ((A[ca.hid_tie_row] @ z) * B[ca.hid_tie_col]).sum(1); he = ((A[ca.hid_empty_row] @ z) * B[ca.hid_empty_col]).sum(1)
            for lab, kt, ke in (("tied", tied[ca.hid_tie_col], tied[ca.hid_empty_col]), ("untied", ~tied[ca.hid_tie_col], ~tied[ca.hid_empty_col])):
                ap, _, ch = M.within_class_ap(ht, he, ca, scale, keep_t=kt, keep_e=ke)
                row[lab].update({"existence": ap, "chance": ch, "hidden_ties": int(kt.sum())})
            rep[(a, f["job"]["K"], float(f["job"]["arm"]["r"]))].append(row)
    out = {}
    for (a, K, r), rows in sorted(rep.items()):
        s = {lab: {k: float(np.median([x[lab][k] for x in rows if x[lab][k] is not None])) for k in rows[0][lab]} for lab in ("tied", "untied")}
        out[f"approach_{a}_K{K}_r{r:g}"] = {"fits": len(rows), "median": s}
        print(f"approach {a} K {K:2d} r {r:<3g} ({len(rows)} fits)  " + " | ".join(
            f"{lab}: n {s[lab]['n']:.0f} own {s[lab]['own_spread_median']:.2f} excess {s[lab]['excess_median']:+.2f} "
            f"(>0 {s[lab]['share_excess_positive']:.2f}, >0.5 {s[lab]['share_excess_above_0.5bit']:.2f}) "
            f"AP {s[lab]['existence']:.4f}/{s[lab]['chance']:.4f}" for lab in ("tied", "untied")), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
