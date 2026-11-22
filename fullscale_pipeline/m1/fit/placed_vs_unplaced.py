# Item (l): do unplaced entities differ from placed ones? (owner 2026-11-22: "please check if the unplaced entities differ
# from the placed entities"). Study A pass 3, approach 1 and 2, K 5 r 0.3 and K 13 r 1, hold-out group 1, seeds 1–10.
# Placed (2026-11-07, measures.placed_entities): mean reproduced share of the entity's observed ties beats the degree-only
# reference by more than its own seed SD. Per facet, placed vs unplaced: number of visible ties (all relations touching
# the facet), membership spread (bits, mean over seeds), share of the facet's tie weight; for articles: child-hyperedge
# ties, authors, journal size; held-out existence of child×article hidden ties split by the article being placed or not
# (within-class AP and its chance, mean over seeds).
# Usage: python -m fit.placed_vs_unplaced --out fit/results/stage0/placed_vs_unplaced.json
import os
os.environ["M1_ALIGNED"] = "1"
import argparse, glob, json, math
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits, load_entity

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def run():
    build = D.Build(D.BUILD); rep = {}
    fs = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass3", "*", "measures.json"))])
    for a in (1, 2):
        rds = D.fit_data(build, a, group=0); ca = next(rd for rd in rds if rd.name == "child_article")
        n_empty_obs = int(ca.obs_row_mask.sum()) * ca.shape[1] - len(ca.tie_row) - len(ca.hid_tie_row)
        scale = math.floor(0.10 * n_empty_obs) / len(ca.hid_empty_row)
        for K, r in ((5, 0.3), (13, 1.0)):
            cell = sorted([f for f in fs if f["job"]["approach"] == a and f["job"]["K"] == K and float(f["job"]["arm"]["r"]) == r
                           and f["job"]["group"] == 0 and f["result"]["converged"]], key=lambda f: f["job"]["seed"])
            ents = [load_entity(f["dir"]) for f in cell]
            placed = {fa: M.placed_entities([e[fa]["reproduced_share"] for e in ents], ents[0][fa]["reproduced_share_degree_only"]) for fa in ents[0]}
            wms = [M.weighted_memberships(*load_UZ(f["dir"]), {k: tuple(v) for k, v in f["result"]["relations"].items()}) for f in cell]
            out = {}
            for fa, pl in placed.items():
                deg = np.zeros(len(pl)); wt = np.zeros(len(pl))
                for rd in rds:
                    for side, idx in (("row", rd.tie_row), ("col", rd.tie_col)):
                        if (rd.row_facet if side == "row" else rd.col_facet) == fa and not (side == "col" and rd.row_facet == rd.col_facet):
                            deg += np.bincount(idx, minlength=len(pl)); wt += np.bincount(idx, weights=rd.weight * rd.tie_val, minlength=len(pl))
                spread = np.mean([M.entropy_bits(w[fa]["P"]) for w in wms], 0)
                act = deg > 0
                row = {}
                for lab, m in (("placed", pl & act), ("unplaced", ~pl & act)):
                    row[lab] = {"n": int(m.sum()), "ties_median": float(np.median(deg[m])) if m.any() else None,
                                "ties_mean": float(deg[m].mean()) if m.any() else None,
                                "share_with_1_tie": float((deg[m] == 1).mean()) if m.any() else None,
                                "spread_median_bits": float(np.median(spread[m])) if m.any() else None,
                                "tie_weight_share": float(wt[m].sum() / wt[act].sum())}
                out[fa] = row
            # held-out child×article by the article being placed
            ap = {"placed": [], "unplaced": []}
            for f in cell:
                U, Z = load_UZ(f["dir"]); A_, B_, z = U["article"], U["child"], Z["child_article"]
                ht = ((A_[ca.hid_tie_row] @ z) * B_[ca.hid_tie_col]).sum(1); he = ((A_[ca.hid_empty_row] @ z) * B_[ca.hid_empty_col]).sum(1)
                for lab, keep in (("placed", placed["article"]), ("unplaced", ~placed["article"])):
                    v, _, ch = M.within_class_ap(ht, he, ca, scale, keep_t=keep[ca.hid_tie_row], keep_e=keep[ca.hid_empty_row])
                    ap[lab].append((v, ch))
            out["heldout_child_article_by_article_placement"] = {lab: {"existence": float(np.mean([x[0] for x in v])), "chance": float(np.mean([x[1] for x in v])),
                                                                       "seed_sd": float(np.std([x[0] for x in v], ddof=1))} for lab, v in ap.items()}
            rep[f"approach_{a}_K{K}_r{r:g}"] = out
            print(f"approach {a} K {K} r {r:g}")
            for fa, row in out.items():
                if fa.startswith("heldout"):
                    print("   held-out child×article, by article:", {k: f"{v['existence']:.4f}/{v['chance']:.4f} (sd {v['seed_sd']:.4f})" for k, v in row.items()}); continue
                print(f"   {fa:12s} " + " | ".join(f"{lab}: n {v['n']}, ties med {v['ties_median']}, 1-tie {v['share_with_1_tie'] if v['share_with_1_tie'] is None else round(v['share_with_1_tie'],2)}, spread {v['spread_median_bits'] if v['spread_median_bits'] is None else round(v['spread_median_bits'],2)} bits, weight share {v['tie_weight_share']:.2f}" for lab, v in row.items()), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
