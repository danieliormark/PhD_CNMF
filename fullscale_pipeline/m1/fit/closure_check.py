# Is the within-community closure of the diagonal start consistent with the observed ties? (owner, 2026-11-18: "I am
# concerned that the diagonal start may overconcentrate social entities ... I wonder if we can check to which extent such
# a strong within-community closure is consistent with the observed data"). Saved fits only (start-test arms: reference =
# pass 1 random social start; diagonal = start_test), hold-out group 1, seeds 1–10, aligned numbering.
# Each entity is given its main community (largest weighted membership). Per relation and fit, with that one partition:
#   data_between   = observed (visible) tie weight whose two ends have different main communities ÷ all visible tie
#                    weight — how much of the DATA crosses the model's community boundaries;
#   recon_between  = the same share of the RECONSTRUCTED tie weight over the cells the loss sees (observed rows × all
#                    columns, minus hidden cells, minus an entity's cell with itself in article×article);
#   chance_between = 1 − Σ_g a_g b_g (a_g, b_g: shares of visible tie weight on row / column entities of community g) —
#                    the data's between share if the partition were unrelated to the ties;
#   ratio_between  = recon_between ÷ data_between: 1 = the model rebuilds as much cross-community weight as the data
#                    have; < 1 = it closes communities more than the data do (over-closure); > 1 = less;
#   tie ratios     = reconstructed ÷ observed value summed over visible ties with both ends in one community (within) and
#                    with ends in different communities (between): a model that over-closes under-rebuilds the
#                    between ties relative to the within ties.
#   soft_within    = Σ_k part(k,k) ÷ T (parts table, for reference).
# Usage: python -m fit.closure_check --out results/stage0/aligned/closure_check.json
import argparse, json, os
from collections import defaultdict
import numpy as np
os.environ["M1_ALIGNED"] = "1"
from . import data as D, measures as M
from .aggregate import load_UZ
from .start_test_analysis import load, LOSS


def xhat(U, Z, rd, r, c):
    return ((U[rd.row_facet][r] @ Z[rd.name]) * U[rd.col_facet][c]).sum(1)


def one(f, rds):
    U, Z = load_UZ(f["dir"])
    wm = M.weighted_memberships(U, Z, {k: tuple(v) for k, v in f["result"]["relations"].items()})
    main = {x: d["P"].argmax(1) for x, d in wm.items()}
    out = {}
    for name, rd in rds.items():
        a, b = rd.row_facet, rd.col_facet; K = U[a].shape[1]
        g, h, v = main[a][rd.tie_row], main[b][rd.tie_col], rd.tie_val
        same = g == h
        data_between = float(v[~same].sum() / v.sum())
        ag = np.bincount(g, weights=v, minlength=K) / v.sum(); bh = np.bincount(h, weights=v, minlength=K) / v.sum()
        A = np.zeros((K, K)); np.add.at(A, main[a][rd.obs_row_mask], U[a][rd.obs_row_mask])
        B = np.zeros((K, K)); np.add.at(B, main[b], U[b])
        R = A @ Z[name] @ B.T
        hr, hc = rd.hid_row, rd.hid_col
        if len(hr):
            np.add.at(R, (main[a][hr], main[b][hc]), -xhat(U, Z, rd, hr, hc))
        if rd.diag_masked:
            obs = np.nonzero(rd.obs_row_mask)[0]
            np.add.at(R, (main[a][obs], main[b][obs]), -xhat(U, Z, rd, obs, obs))
        recon_between = float(1 - np.trace(R) / R.sum())
        xt = xhat(U, Z, rd, rd.tie_row, rd.tie_col)
        P, T = M.parts(U, Z, rd)
        out[name] = {"data_between": data_between, "recon_between": recon_between, "chance_between": float(1 - (ag * bh).sum()),
                     "ratio_between": recon_between / data_between if data_between > 0 else None,
                     "tie_ratio_within": float(xt[same].sum() / v[same].sum()) if same.any() else None,
                     "tie_ratio_between": float(xt[~same].sum() / v[~same].sum()) if (~same).any() else None,
                     "soft_within": float(np.trace(P) / T)}
    return out


def run():
    ref, arm = load("stage0_pass1", False), load("start_test", True)
    build = D.Build(D.BUILD); rep = {}
    for (a, beta) in sorted(arm):
        rds = {rd.name: rd for rd in D.fit_data(build, a, group=0)}
        name = f"approach_{a}_{LOSS[beta]}"; rep[name] = {}
        for lab, fs in (("reference", ref[(a, beta)]), ("diagonal", arm[(a, beta)])):
            rows = [one(f, rds) for f in fs]
            rep[name][lab] = {"fits": len(rows), "converged": int(sum(f["result"]["converged"] for f in fs)),
                              "relations": {r: {k: (float(np.median([x[r][k] for x in rows if x[r][k] is not None]))
                                                    if any(x[r][k] is not None for x in rows) else None)
                                                for k in rows[0][r]} for r in rows[0]}}
        print(name, flush=True)
        for r in rds:
            q = [rep[name][lab]["relations"][r] for lab in ("reference", "diagonal")]
            fmt = lambda x: "  —  " if x is None else f"{x:.3f}"
            print(f"  {r:20s} data_between {fmt(q[0]['data_between'])} / {fmt(q[1]['data_between'])}  recon_between "
                  f"{fmt(q[0]['recon_between'])} / {fmt(q[1]['recon_between'])}  ratio {fmt(q[0]['ratio_between'])} / "
                  f"{fmt(q[1]['ratio_between'])}  chance {fmt(q[0]['chance_between'])} / {fmt(q[1]['chance_between'])}  "
                  f"tie ratio within/between {fmt(q[0]['tie_ratio_within'])},{fmt(q[0]['tie_ratio_between'])} / "
                  f"{fmt(q[1]['tie_ratio_within'])},{fmt(q[1]['tie_ratio_between'])}  soft_within {fmt(q[0]['soft_within'])} / {fmt(q[1]['soft_within'])}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
