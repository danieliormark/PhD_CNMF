"""Ticket 105, test T5: calibration of LAMBDA_TEMPORAL on the real T2 slice.

Per config (prior = its v9.4 T1 knee model), T2 K and weight, at several seeds:
  - full-data fit: reconstruction (math_loss), convergence, epochs, accepted T1->T2 pairs
    at the freeze, and *persistence*: after matching T1 to the final fit post hoc (the same
    temporal_match_columns, applied to every weight including 0 so the comparison is like
    for like), the share of persisting entities whose T2 main community is the image of
    their T1 main community;
  - seed stability of the full-data fits: Track A/B (production _pair_tracks) over all seed
    pairs, chance-adjusted against row-shuffled pairs (ticket 95 / FINDINGS §32);
  - held-out ties (MetaFac's own criterion: predict data not used in the fit): 10% of the
    ties of every relation with >= MIN_NNZ ties are hidden, never an entity's last tie (so
    live sets and the prior's persisting set are unchanged); after fitting on the rest, each
    hidden tie is scored by its reconstructed value against an equal number of random
    non-ties of the same relation (AUC per relation, mean over relations). The same mask is
    used for every weight, so comparisons are paired.
WEIGHTS includes 1e-6, a control: the term is on (freeze, stopping rule) but has almost no
force, which separates the prior's effect from the change of trajectory alone.

    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE python t5_calibration.py C2 [K ...]
"""
import importlib.util, os, sys, json, itertools, time, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c13v10", f"{E}/chunk13v10.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
import scipy.sparse as sp

CFG = sys.argv[1]
KS = [int(k) for k in sys.argv[2:]] or [2, 3, 4, 5]
WEIGHTS = [float(w) for w in os.environ.get("T5_WEIGHTS", "0,1e-6,0.001,0.003,0.01,0.03,0.1").split(",")]
SEEDS = [int(s) for s in os.environ.get("T5_SEEDS", "1000,1001,1002,1003,1004").split(",")]
LZ = float(os.environ.get("T5_LZ", "0.12"))
HOLD, MIN_NNZ = 0.10, 150

sel = json.load(open(f"{E}/results/v9.4.t1_v2/model_selection.json"))[CFG]
raw1 = m.load_and_validate_data(OUT + "Star_extended_matrices_t1_v2.pkl")
raw2 = m.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
SOC, SEM, ANC = m.get_active_facets(CFG)
FACETS = m.get_required_facets(SOC, SEM, ANC)
prior = m.build_temporal_prior(f"{E}/results/v9.4.t1_v2/{CFG}/K_{sel['K']}/pareto_models/{sel['chosen']['trial']}",
                               raw1, raw2, SOC, SEM)


def make_mask(raw, rng):
    """Hide HOLD of the ties of each relation with >= MIN_NNZ ties, never a row's or column's last tie."""
    train, hidden = dict(raw), {}
    for rel in SOC + SEM:
        X = sp.coo_matrix(raw[rel])
        if X.nnz < MIN_NNZ:
            continue
        rdeg = np.bincount(X.row, minlength=X.shape[0]); cdeg = np.bincount(X.col, minlength=X.shape[1])
        keep = np.ones(X.nnz, bool)
        for t in rng.permutation(X.nnz)[: int(HOLD * X.nnz) * 3]:
            if keep.sum() <= X.nnz - int(HOLD * X.nnz):
                break
            i, j = X.row[t], X.col[t]
            if rdeg[i] > 1 and cdeg[j] > 1:
                keep[t] = False; rdeg[i] -= 1; cdeg[j] -= 1
        train[rel] = sp.csr_matrix((X.data[keep], (X.row[keep], X.col[keep])), shape=X.shape)
        dense = set(zip(X.row.tolist(), X.col.tolist()))
        neg = []
        while len(neg) < (~keep).sum():
            i, j = int(rng.integers(X.shape[0])), int(rng.integers(X.shape[1]))
            if (i, j) not in dense:
                neg.append((i, j))
        hidden[rel] = (np.stack([X.row[~keep], X.col[~keep]], 1), np.array(neg))
    return train, hidden


def fit(raw, K, w, seed):
    m.set_seeds(seed)
    U, Z, d = m.run_inner_solver(raw_data=raw, soc_keys=SOC, sem_keys=SEM, anchor_keys=ANC, K=K,
                                 dimensions=raw["dimensions"],
                                 params={"lambda_z_offdiag": LZ, "lambda_temporal": w},
                                 device=m.DEVICE, seed_function=lambda s=seed: m.set_seeds(s),
                                 inner_epochs=m.EXTENDED_EPOCHS, temporal_prior=prior if w > 0 else None)
    return ({f: u.cpu().numpy().astype(np.float64) for f, u in U.items()},
            {r: z.cpu().numpy().astype(np.float64) for r, z in Z.items()}, d)


def auc(Un, Zn, hidden):
    out = {}
    for rel, (pos, neg) in hidden.items():
        fa, fb = m.RELATION_MAP[rel]
        score = lambda P: np.einsum("ik,kl,il->i", Un[fa][P[:, 0]], Zn[rel], Un[fb][P[:, 1]])
        sp_, sn = score(pos), score(neg)
        out[rel] = float((sp_[:, None] > sn[None, :]).mean() + 0.5 * (sp_[:, None] == sn[None, :]).mean())
    return out


def persistence(Un, Zn):
    """Post-hoc match T1 -> this fit on the persisting entities; share keeping their community."""
    pm = m.build_presence_masks(raw2, SOC, SEM)
    P2, _ = m.compute_weighted_membership(Un, Zn, pm)
    fac = [f for f in sorted(prior["facets"]) if f in P2 and len(prior["facets"][f]["idx"])]
    P1s = [prior["facets"][f]["P"] for f in fac]; P2s = [P2[f][prior["facets"][f]["idx"]] for f in fac]
    acc, rec = m.temporal_match_columns(np.vstack(P1s), P2s)
    mp = dict(acc)
    keep_n = tot = 0
    for A, B in zip(P1s, P2s):
        a = A.argmax(1); b = B.argmax(1)
        ok = np.array([x in mp for x in a])
        tot += ok.sum(); keep_n += sum(mp[x] == y for x, y in zip(a[ok], b[ok]))
    return {"posthoc_pairs": len(acc), "persisting_in_matched": int(tot),
            "share_keeping_community": float(keep_n / tot) if tot else None}


def stability(fits, K):
    rng = np.random.default_rng(1000)
    real, chance = [], []
    for i, j in itertools.combinations(range(len(fits)), 2):
        A, B, _, _ = m._pair_tracks(fits[i], fits[j], FACETS, K)
        sh = {f: fits[j][f][rng.permutation(len(fits[j][f]))] for f in FACETS}
        As, Bs, _, _ = m._pair_tracks(fits[i], sh, FACETS, K)
        real.append((A, B)); chance.append((As, Bs))
    r, c = np.mean(real, 0), np.mean(chance, 0)
    return {"track_A": float(r[0]), "track_B": float(r[1]), "chance_A": float(c[0]), "chance_B": float(c[1]),
            "adj_A": float((r[0] - c[0]) / (1 - c[0])), "adj_B": float((r[1] - c[1]) / (1 - c[1]))}


results = []
t0 = time.time()
train, hidden = make_mask(raw2, np.random.default_rng(2026))
for K in KS:
    for w in WEIGHTS:
        fits, per_seed = [], []
        for seed in SEEDS:
            Un, Zn, d = fit(raw2, K, w, seed)
            Ut, Zt, dt = fit(train, K, w, seed)
            a = auc(Ut, Zt, hidden)
            match = d.get("temporal_match") or {}
            per_seed.append({"seed": seed, "math_loss": d["math_loss"], "converged": d["converged"],
                             "epochs": len(d["loss_history"]), "accepted_at_freeze": len(match.get("pairs", [])),
                             "heldout_auc": a, "heldout_auc_mean": float(np.mean(list(a.values()))),
                             "heldout_converged": dt["converged"], **persistence(Un, Zn)})
            fits.append(Un)
        st = stability(fits, K)
        rec = {"config": CFG, "K": K, "weight": w, "seeds": per_seed, "stability": st}
        results.append(rec)
        ps = [s["share_keeping_community"] for s in per_seed if s["share_keeping_community"] is not None]
        print(f"{CFG} K={K} w={w}: recon {np.mean([s['math_loss'] for s in per_seed]):.4f} | "
              f"held-out AUC {np.mean([s['heldout_auc_mean'] for s in per_seed]):.4f} | "
              f"persistence {np.mean(ps) if ps else float('nan'):.3f} (n seeds {len(ps)}) | "
              f"pairs at freeze {np.mean([s['accepted_at_freeze'] for s in per_seed]):.1f} | "
              f"stability adj A {st['adj_A']:.3f} B {st['adj_B']:.3f} | "
              f"conv {sum(s['converged'] for s in per_seed)}/{len(SEEDS)} | {time.time()-t0:.0f}s", flush=True)
        json.dump(results, open(f"t5_{CFG}.json", "w"), indent=1, default=float)
