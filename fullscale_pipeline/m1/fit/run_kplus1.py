# K* + 1 fits (decided 2026-11-08 / 2026-11-10): noiseless code-check world fitted at K = 7 (planted 6), squared error
# (fraction 0.03) and KL (0.01), hybrid start, recomputed per-block rate; is the extra community flagged empty by the
# 0.1th-percentile reference (with and without floor-level core cells read as 0), and does the child-only community appear?
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

def run(beta):
    import torch, numpy as np
    torch.set_num_threads(1)
    from fit import data as D, worlds as Wd, measures as M
    from fit.solver import fit
    rds, Ut, Zt = Wd.code_check_world("noiseless", seed=1, build=D.Build(), child_only="split")
    rel = {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}
    r = fit(rds, 7, beta, seed=1, max_epochs=5000, lr=0.03 if beta == 2 else 0.01, lr_mode="recompute", start="nndsvd_sub0.8")
    wm = M.weighted_memberships(r["U"], r["Z"], rel)
    out = {"beta": beta, "converged": r["converged"], "epochs": r["epochs"], "final_data_loss": r["data_loss_history"][-1]}
    for fz in (None, 1e-7):
        pres = M.presence(r["U"], r["Z"], rds, floor_zero=fz)
        em = M.empty_columns(pres, rds, wm)
        out[f"floor_zero={fz}"] = {"presence": {k: np.round(v["presence"], 4).tolist() for k, v in pres.items()},
                                  "empty_p0.1": em["p0.1"]["relations"], "empty_facets_p0.1": em["p0.1"]["facets"],
                                  "empty_all_refs": {ref: em[ref]["relations"] for ref in em}}
    from fit.worlds import matched_cosines
    cos, perm = matched_cosines(r["U"], {f: np.hstack([u, np.full((u.shape[0], 1), 1e-7)]) for f, u in Ut.items()},
                                r["Z"], {k: np.pad(z, ((0, 1), (0, 1)), constant_values=1e-7) for k, z in Zt.items()}, rel)
    out["matched_cosines_planted"] = [round(float(c), 4) for c in cos[:6]]
    out["perm"] = perm
    return out

if __name__ == "__main__":
    o = sys.argv[1]; os.makedirs(o, exist_ok=True)
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run, (2, 1)))
    json.dump(res, open(os.path.join(o, "kplus1.json"), "w"), indent=1, default=str)
    print("done")
