# Stage-0 pass 2: basis of λ_child (owner 2026-11-14, decision a): λ_child = r × (data loss ÷ R) at λ = 0, per loss and
# approach, from the pass-1 fits at each loss's K* (squared error 20, KL 13), hold-out group 1 (index 0), mean over the
# 10 seeds; R = (1/N) Σ_i Σ_j (W_ij/d_i)‖p_i − p_j‖² on the final weighted memberships of child hyperedges.
# Reads saved fits only. Usage: G=0 OUT=fit/results/stage0/pass2_lambda_basis.json python fit/pass2_lambda.py
import json, glob, os, sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1")
from fit import data as D, measures as M
build = D.Build(D.BUILD)
W = sp.load_npz(build.path + "W_child_child.npz").tocsr(); W.eliminate_zeros()
d = np.asarray(W.sum(1)).ravel(); coo = W.tocoo(); v = coo.data / d[coo.row]; N = int((d > 0).sum())
print("child rows", W.shape[0], "with a W tie", N, "share", N / W.shape[0])
Kstar = {2: 20, 1: 13}
rows = []
for sd in glob.glob("/mnt/hum01-rds/Basov/p91688di/m1_fits/stage0_*/*/spec.json"):
    j = json.load(open(sd))["job"]
    if j["group"] != int(os.environ.get("G","0")) or j["K"] != Kstar[j["beta"]] or (j.get("terms") or {}): continue
    dd = os.path.dirname(sd)
    if not os.path.exists(dd + "/result.json"): continue
    res = json.load(open(dd + "/result.json"))
    uz = np.load(dd + "/UZ.npz")
    U = {k[3:]: uz[k].astype(np.float64) for k in uz if k.startswith("U__")}
    Z = {k[3:]: uz[k].astype(np.float64) for k in uz if k.startswith("Z__")}
    P = M.weighted_memberships(U, Z, res["relations"])["child"]["P"]
    diff = P[coo.row] - P[coo.col]
    R = float((v * (diff ** 2).sum(1)).sum() / N)
    Pt = P[d > 0]; S = float(2 * ((Pt ** 2).sum(1).mean() - (Pt.mean(0) ** 2).sum()))   # option B denominator
    top_t = float(P[d > 0].max(1).mean())
    dl = res["data_loss_history"][-1]; dl = dl[-1] if isinstance(dl, (list, tuple)) else dl
    rows.append((j["approach"], j["beta"], j["seed"], dl, R, S, top_t))
import collections
g = collections.defaultdict(list)
for a, b, s, dl, R, S, tt in rows: g[(a, b)].append((dl, R, S, tt))
for k in sorted(g):
    x = np.array(g[k]); print("approach", k[0], "loss", {2: "squared error", 1: "KL"}[k[1]], "n", len(x),
      "data loss mean %.4g sd %.2g" % (x[:, 0].mean(), x[:, 0].std(ddof=1)), "R mean %.4f sd %.4f" % (x[:, 1].mean(), x[:, 1].std(ddof=1)),
      "ratio data/R %.4g" % (x[:, 0].mean() / x[:, 1].mean()))
out = {}
for k in sorted(g):
    x = np.array(g[k]); out[f"{k[0]}_{k[1]}"] = {"data_loss": float(x[:, 0].mean()), "R": float(x[:, 1].mean()), "n": len(x),
        "S": float(x[:, 2].mean()), "Rc": float((x[:, 1] / x[:, 2]).mean()), "largest_share_tied": float(x[:, 3].mean())}
json.dump(out, open(os.environ["OUT"], "w"), indent=1)
