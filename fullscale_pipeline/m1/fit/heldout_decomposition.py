# Held-out squared error split into its parts (ticket 107, owner's permission 2026-11-11): for the semantic relations
# (child×article, child×parent) of every checkpoint-3 fit, the score is  E = Σ_hidden ties (x − x̂)² + s · Σ_hidden empty x̂²
# (s = the option-B scale: floor(10% of the matrix's observed empty cells) / number of hidden empty cells actually drawn),
# against the zero-prediction error Z = Σ_hidden ties x².  Reported as ratios to Z:  ties part, empty-cell part, total
# (= 1 − captured share), plus the calibration of the hidden ties (Σ x̂ / Σ x) and the mean x̂ on hidden empty cells.
# Usage: python -m fit.heldout_decomposition --root ROOT --out FILE.json
import argparse, glob, json, math, os
import numpy as np
from . import data as D
from .aggregate import load_UZ


def parts(U, Z, rd):
    A, B, Zr = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
    xt = rd.hid_tie_val
    ht = ((A[rd.hid_tie_row] @ Zr) * B[rd.hid_tie_col]).sum(1)
    he = ((A[rd.hid_empty_row] @ Zr) * B[rd.hid_empty_col]).sum(1)
    n_ties_all = len(rd.tie_row) + len(rd.hid_tie_row)
    n_obs = int(rd.obs_row_mask.sum()) * rd.shape[1]
    scale = math.floor(0.10 * (n_obs - n_ties_all)) / len(he)
    zero = float((xt ** 2).sum())
    return {"zero": zero, "ties": float(((xt - ht) ** 2).sum()), "empty": float(scale * (he ** 2).sum()), "scale": scale,
            "calibration": float(ht.sum() / xt.sum()), "mean_empty_pred": float(he.mean()), "mean_tie_value": float(xt.mean()),
            "mean_tie_pred": float(ht.mean()), "stored_total": None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build = D.Build()
    rds = {rd.name: rd for rd in D.fit_data(build, 1, group=0)}
    out = []
    for d in sorted(glob.glob(os.path.join(a.root, "checkpoint3", "*", "measures.json"))):
        dd = os.path.dirname(d)
        job = json.load(open(dd + "/spec.json"))["job"]; m = json.load(open(d))
        U, Z = load_UZ(dd)
        rec = {"job": job, "dir": dd}
        for r in ("child_article", "parent_child"):
            p = parts(U, Z, rds[r])
            p["stored_total"] = m["heldout"][r]["squared_error"]
            rec[r] = p
        out.append(rec)
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} fits")


if __name__ == "__main__":
    main()
