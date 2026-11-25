# D1-M3 restated for KL (owner 2026-11-25: "I approve i, ii and iii"; RL-396): per relation r,
#   c_r = 1 − L_r(fit) ÷ L_r(ref)  (KL deviance explained against the degree-only reference, tie weights; here the exact
#         masked rank-1 KL optimum, which differs from row total × column total ÷ grand total by ≤ 0.1 %),
#   realised share s_r = w_r ΔL_r ÷ Σ_q w_q ΔL_q (ΔL = L(ref) − L(fit)), pull share p_r = w_r L_r(ref) ÷ Σ_q w_q L_q(ref).
# Illustration on saved stage-0 pass-3 fits (main setting = arm J2), study A, groups 1–5 × seeds 1–3.
# Usage: python fit/d1m3_kl_check.py fit/results/stage0/d1m3_kl_check.json   (from fullscale_pipeline/m1, PYTHONPATH=.)
import os, sys, glob, json
os.environ["M1_ALIGNED"] = "0"
import numpy as np, torch
sys.path.insert(0, ".")
from fit.aggregate import load_fit, latest_fits
from fit.align import fit_rds
from fit.loss import RelationTensors, relation_loss

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits/stage0_pass3"
CELLS = {(1, 5, 0.3), (1, 13, 0.3), (2, 5, 0.3), (2, 13, 1.0)}

def rank1_loss(rt, n_rows, n_cols, iters=60):
    m = rt.row_mask.numpy(); tr, tc, tx = rt.tie_r.numpy(), rt.tie_c.numpy(), rt.tie_x.numpy()
    hr, hc = rt.hid_r.numpy(), rt.hid_c.numpy()
    rs = np.bincount(tr, tx, n_rows); cs = np.bincount(tc, tx, n_cols)
    a = rs.copy(); b = cs / max(cs.sum(), 1e-300)
    for _ in range(iters):
        sb = b.sum() - np.bincount(hr, b[hc], n_rows) - (b[:n_rows] * 1.0 if rt.diag_masked else 0)
        a = np.where(m > 0, rs / np.maximum(sb, 1e-300), 0.0)
        sa = (m * a).sum() - np.bincount(hc, a[hr], n_cols) - ((m * a)[:n_cols] if rt.diag_masked else 0)
        b = cs / np.maximum(sa, 1e-300)
    T = lambda v: torch.as_tensor(v[:, None], dtype=torch.float64)
    return float(relation_loss(T(a), torch.ones(1, 1, dtype=torch.float64), T(b), rt, 1))

def main():
    fits = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "*", "measures.json"))])
    fits = [f for f in fits if (f["job"]["approach"], f["job"]["K"], float(f["job"]["arm"]["r"])) in CELLS and f["result"]["converged"]
            and f["job"]["seed"] <= 3]
    print(len(fits), "fits", flush=True)
    cache, ref_cache, rows = {}, {}, []
    for f in sorted(fits, key=lambda f: (f["job"]["approach"], f["job"]["group"])):
        j = f["job"]; rds = fit_rds(j, cache)
        key = (j["approach"], j["group"])
        if key not in ref_cache:
            ref_cache[key] = {}
            for rd in rds:
                rt = RelationTensors(rd)
                nr = len(rd.obs_row_mask); nc = rd.shape[1]
                ref_cache[key][rd.name] = (rank1_loss(rt, nr, nc), rd.weight, float(np.sum(rd.tie_val)))
        fl = f["result"]["final_relation_loss"]
        rows.append({"approach": j["approach"], "K": j["K"], "r": float(j["arm"]["r"]), "group": j["group"], "seed": j["seed"],
                     "rel": {r: {"L_fit": fl[r], "L_ref": v[0], "w": v[1], "mass": v[2]} for r, v in ref_cache[key].items()}})
        print(j["approach"], j["K"], j["group"], j["seed"], {r: round(1 - x["L_fit"] / x["L_ref"], 3) for r, x in rows[-1]["rel"].items()}, flush=True)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
