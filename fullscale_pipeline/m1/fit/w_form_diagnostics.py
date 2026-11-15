# Diagnostics of the W-term test (owner 2026-11-15; RL-234): for every fit of a stage (and its λ = 0 reference among the
# pass-1 fits with the same approach, loss, K, hold-out group and seed): R (pairs tied in W), S (mean squared membership
# difference over all pairs of tied child hyperedges), R ÷ S, the mean largest membership share of tied and of untied
# child hyperedges, the share of tied child hyperedges in the most common main community, the number of communities
# that are the main community of > 1% of tied child hyperedges, held-out existence and value (decision score), final
# data loss, convergence and epochs. Reads saved fits only.
# Usage: python -m fit.w_form_diagnostics --root ROOT --stage w_form_test --ref stage0_pass1 --out OUT.json
import argparse, glob, json, os
import numpy as np, scipy.sparse as sp
from . import data as D, measures as M
from .aggregate import load_UZ


def diag(dd, W, coo, v, N, d):
    j = json.load(open(dd + "/spec.json"))["job"]; r = json.load(open(dd + "/result.json"))
    ms = json.load(open(dd + "/measures.json"))
    U, Z = load_UZ(dd)
    P = M.weighted_memberships(U, Z, {k: tuple(x) for k, x in r["relations"].items()})["child"]["P"]
    R = float((v * ((P[coo.row] - P[coo.col]) ** 2).sum(1)).sum() / N)
    tied = d > 0; Pt = P[tied]
    S = float(2 * ((Pt ** 2).sum(1).mean() - (Pt.mean(0) ** 2).sum()))
    share = np.bincount(Pt.argmax(1), minlength=P.shape[1]) / len(Pt)
    tp = ms.get("heldout_two_part_combined") or {}
    return {"job": j, "R": R, "S": S, "R_over_S": R / S, "largest_share_tied": float(Pt.max(1).mean()),
            "largest_share_untied": float(P[~tied].max(1).mean()), "top_main_community_share_tied": float(share.max()),
            "communities_main_for_1pct_tied": int((share > 0.01).sum()), "existence": tp.get("existence"),
            "value": tp.get("value"), "data_loss": r["data_loss_history"][-1], "converged": r["converged"], "epochs": r["epochs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--stage", required=True); ap.add_argument("--ref", default="stage0_pass1"); ap.add_argument("--ref-root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = sp.load_npz(D.Build(D.BUILD).path + "W_child_child.npz").tocsr(); W.eliminate_zeros()
    d = np.asarray(W.sum(1)).ravel(); coo = W.tocoo(); v = coo.data / d[coo.row]; N = int((d > 0).sum())
    key = lambda j: (j["approach"], j["beta"], j["K"], j["group"], j["seed"])
    fits = [os.path.dirname(p) for p in glob.glob(os.path.join(a.root, a.stage, "*", "result.json"))]
    want = {key(json.load(open(f + "/spec.json"))["job"]) for f in fits}
    refs = {}
    for p in glob.glob(os.path.join(a.ref_root or a.root, a.ref, "*", "result.json")):
        f = os.path.dirname(p); j = json.load(open(f + "/spec.json"))["job"]
        if key(j) in want and not (j.get("terms") or {}) and os.path.exists(f + "/measures.json"):
            if key(j) not in refs or j.get("max_epochs", 0) > refs[key(j)][1]:     # a rerun replaces its original
                refs[key(j)] = (f, j.get("max_epochs", 0))
    out = {"reference": [diag(f, W, coo, v, N, d) for f, _ in refs.values()],
           "arms": [diag(f, W, coo, v, N, d) for f in fits if os.path.exists(f + "/measures.json")]}
    json.dump(out, open(a.out, "w"), indent=1)
    for row in out["reference"] + out["arms"]:
        j = row["job"]; arm = j.get("arm") or {"option": "λ=0"}
        print(j["beta"], j["seed"], arm.get("option"), arm.get("r"), {k: (round(x, 3) if isinstance(x, float) else x)
              for k, x in row.items() if k != "job"})


if __name__ == "__main__":
    main()
