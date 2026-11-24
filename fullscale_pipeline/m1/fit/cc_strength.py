# child×child focused test (owner 2026-11-24, RL-377/378): weight of the child×child relation matched to the
# regulariser's strength r — weight(r) = r × data loss of the λ = 0 solution ÷ the relation's loss on that solution, the
# relation's core fitted with the memberships held fixed (generalised KL, multiplicative updates). Same source as the
# regulariser's basis (pass3_lambda_basis: stage-0 pass 1, λ = 0): approach 1, K 5 and 13, hold-out group 1, seeds 1–3.
# Modes: "full" (every off-diagonal cell fitted) and "ties" (only W's tie cells).
# Usage: python -m fit.cc_strength --out fit/results/stage0/cc_strength.json
import argparse, glob, json, os
import numpy as np, scipy.sparse as sp
from . import data as D
from .aggregate import load_UZ

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"


def fit_core(A, r, c, x, mode, iters=800):
    K = A.shape[1]; s = A.sum(0)
    den = (np.outer(s, s) - A.T @ A) if mode == "full" else (A[r].T @ A[c])
    Z = np.full((K, K), x.sum() / max(float(np.sum(den)), 1e-300))
    for _ in range(iters):
        xh = np.maximum(((A[r] @ Z) * A[c]).sum(1), 1e-300)
        S = sp.csr_matrix((x / xh, (r, c)), shape=(A.shape[0], A.shape[0]))
        Z = Z * (A.T @ (S @ A)) / np.maximum(den, 1e-300)
    xh = np.maximum(((A[r] @ Z) * A[c]).sum(1), 1e-300)
    tot_hat = float(s @ Z @ s - np.einsum("ik,kl,il->", A, Z, A)) if mode == "full" else float(xh.sum())
    return float(np.sum(x * np.log(x / xh) - x) + tot_hat)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    b = D.Build(D.BUILD)
    r, c, v = b.ties("child_child"); m = r != c; r, c, v = r[m], c[m], v[m].astype(float); x = v / np.sqrt(np.sum(v ** 2))
    out = {}
    for K in (5, 13):
        rows = []
        for p in glob.glob(os.path.join(ROOT, "stage0_pass1", "*", "spec.json")):
            j = json.load(open(p))["job"]
            if j["approach"] == 1 and j["K"] == K and j["beta"] == 1 and j["group"] == 0 and j["seed"] in (1, 2, 3) \
                    and not (j.get("terms") or {}) and not j.get("arm"):
                d = os.path.dirname(p); res = json.load(open(os.path.join(d, "result.json")))
                if not res["converged"]:
                    continue
                dl = res["data_loss_history"][-1]; dl = dl[-1] if isinstance(dl, (list, tuple)) else dl
                U, _ = load_UZ(d); A = U["child"]
                rows.append({"seed": j["seed"], "data_loss": float(dl), **{f"loss_{mo}": fit_core(A, r, c, x, mo) for mo in ("full", "ties")}})
        dl = np.mean([q["data_loss"] for q in rows])
        e = {"fits": rows, "data_loss": float(dl)}
        for mo in ("full", "ties"):
            L = np.mean([q[f"loss_{mo}"] for q in rows])
            e[f"loss_{mo}"] = float(L); e[f"weight_{mo}"] = {str(rr): float(rr * dl / L) for rr in (0.3, 1.0)}
        out[str(K)] = e
        print(K, json.dumps({k: v for k, v in e.items() if k != "fits"}), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
