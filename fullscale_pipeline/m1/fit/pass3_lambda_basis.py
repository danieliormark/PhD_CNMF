# Pass 3: basis of λ_child per K (ticket 107; pass 3 K grid {5, 8, 13, 20}, owner 2026-11-19, RL-296). Same quantity as
# pass 2 (fit/pass2_lambda.py, decided 2026-11-14/16): λ_child = r × (data loss ÷ (R ÷ S)) on the fitted λ = 0 solution,
# R ÷ S = the contrast form of the word-similarity term on the final weighted memberships of child hyperedges; mean over
# the 10 seeds of hold-out group 1. KL only. Sources compared (saved fits only):
#   pass1      — stage-0 pass 1 (hybrid start with random social links), every K of the grid;
#   S3_10      — start test 2, start S3 at 10 % (the decided start), K = 13 only (the only K fitted under that start).
# The ratio S3_10 ÷ pass1 at K = 13 shows how much the start moves the basis.
# Usage: python -m fit.pass3_lambda_basis --out fit/results/stage0/pass3_lambda_basis.json [--build PATH --stages ...]
import argparse, glob, json, os, collections
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def contrast(P, coo, v, d, N):
    diff = P[coo.row] - P[coo.col]
    R = float((v * (diff ** 2).sum(1)).sum() / N)
    Pt = P[d > 0]; S = float(2 * ((Pt ** 2).sum(1).mean() - (Pt.mean(0) ** 2).sum()))
    return R, S


def run(build_path, sources):
    build = D.Build(build_path)
    W = sp.load_npz(build.path + "W_child_child.npz").tocsr(); W.eliminate_zeros()
    d = np.asarray(W.sum(1)).ravel(); coo = W.tocoo(); v = coo.data / d[coo.row]; N = int((d > 0).sum())
    rows = collections.defaultdict(list)
    for src, stage, filt in sources:
        for sd in glob.glob(os.path.join(ROOT, stage, "*", "spec.json")):
            j = json.load(open(sd))["job"]
            if j["beta"] != 1 or j["group"] != 0 or (j.get("terms") or {}) or j.get("arm") or not filt(j):
                continue
            if j.get("build", D.BUILD).rstrip("/") != build.path.rstrip("/"):
                continue
            dd = os.path.dirname(sd)
            if not os.path.exists(dd + "/result.json"):
                continue
            res = json.load(open(dd + "/result.json")); uz = np.load(dd + "/UZ.npz")
            U = {k[3:]: uz[k].astype(np.float64) for k in uz if k.startswith("U__")}
            Z = {k[3:]: uz[k].astype(np.float64) for k in uz if k.startswith("Z__")}
            P = M.weighted_memberships(U, Z, res["relations"])["child"]["P"]
            R, S = contrast(P, coo, v, d, N)
            dl = res["data_loss_history"][-1]; dl = dl[-1] if isinstance(dl, (list, tuple)) else dl
            rows[(src, j["approach"], j["K"])].append((j["seed"], dl, R, S))
    out = {}
    for (src, a, K), x in sorted(rows.items()):
        x = np.array(sorted(x)); basis = x[:, 1] / (x[:, 2] / x[:, 3])
        out[f"{src}|{a}|{K}"] = {"source": src, "approach": a, "K": K, "n": len(x), "data_loss": float(x[:, 1].mean()),
                                 "Rc": float((x[:, 2] / x[:, 3]).mean()),
                                 "basis": float(x[:, 1].mean() / (x[:, 2] / x[:, 3]).mean()),
                                 "basis_seed_cv": float(basis.std(ddof=1) / basis.mean()) if len(x) > 1 else None}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--build", default=D.BUILD)
    a = ap.parse_args()
    pre = "B_" if "nogenre" in a.build else ""
    sources = [("pass1", pre + "stage0_pass1", lambda j: j["K"] in (3, 5, 8, 13, 20)),   # K 3: grid-end extension of pass 3 (RL-307)
               ("S3_10", pre + "start_test2", lambda j: j.get("offdiag_level") == 0.1 and j.get("offdiag_semantic")
                and j.get("social_link_start") == "diagonal")]
    out = run(a.build, sources)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, e in out.items():
        print(f"{e['source']:6s} approach {e['approach']} K {e['K']:2d} n {e['n']:2d}  data loss {e['data_loss']:.1f}  "
              f"R/S {e['Rc']:.4f}  basis {e['basis']:.1f}  (seed CV {e['basis_seed_cv']:.3f})")
    for a_ in (1, 2):
        p, s = out.get(f"pass1|{a_}|13"), out.get(f"S3_10|{a_}|13")
        if p and s:
            print(f"approach {a_} K 13: basis S3_10 ÷ pass1 = {s['basis'] / p['basis']:.3f}")


if __name__ == "__main__":
    main()
