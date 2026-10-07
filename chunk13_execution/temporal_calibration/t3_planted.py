"""Ticket 105, test T3: planted two-slice scenarios (CLAUDE.md §4.25, expected-behaviour table).

A controlled two-slice block-model generator with the pipeline's facets and relations
(config C2), so the truth is known and its strength can be set. The FINDINGS §25 generator
rewires the real toy degree sequences, and at that sparsity it barely recovers planted
communities (social at chance), so a test built on it alone could not show anything.

Slices. Per global facet, a T1-live set and a T2-live set with a set overlap (the persisting
entities); articles are slice-specific. Regimes:
  toy   - live counts, overlaps and mean degrees taken from the real T1/T2 toy matrices
          (authors overlap 4 of 253, core atoms 123 of 465, ...)
  dense - same live counts, 50% of each T2 facet persisting, degrees x3
Ties: every live row gets degree max(1, Poisson(mean degree of that relation)); columns are
drawn without replacement among live columns, with a boost for columns of the same planted
community solved so that WITHIN of the ties are within-community; every live column keeps at
least one tie. Values 1.

Truth (K1 = 4 communities in T1, uniform). T2 scenarios:
  growth           persisting keep their community; new entities spread over the same 4
  migration        20% of persisting entities move to another community
  split            community 0 splits into 0 and 4 (persisting members 50/50); K2 = 5
  merge            community 3 merges into 2; K2 = 3
  dissolve_emerge  community 3 ends (its persisting members go to 0-2); community 4 is new and
                   made of new entities only; K2 = 4

Fits (production run_inner_solver, lambda_z_offdiag 0.12, seed 42): T1 at K = 4 without prior;
the prior from that fit; T2 at the true K2 for each weight in LAMBDAS (0 = no prior); for split
and merge also at K = 4. Scored against the truth: Hungarian-matched accuracy of the weighted-
membership argmax, overall and for persisting, new, migrant, split, merged, dissolved and
emerged entities; each accepted T1->T2 pair checked against the scenario's allowed mapping
(false matches); whether the emerged community was matched; and fitted communities made >= 90%
of new entities (a spurious "new" community where the truth has none).

    python t3_planted.py <scenario> <regime> [reps]
"""
import importlib.util, os, sys, json, time, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c13v10", f"{E}/chunk13v10.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

SCENARIO, REGIME = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CFG, K1, LZ, WITHIN = "C2", 4, 0.12, 0.9
LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
GLOBAL = ["auth", "affil", "journ", "parent_he", "core_child_he", "core_atom", "cousin_he", "fringe_atom"]
SOC, SEM, ANC = m.get_active_facets(CFG)

real1 = m.load_and_validate_data(OUT + "Star_extended_matrices_t1_v2.pkl")
real2 = m.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
pm1r = m.build_presence_masks(real1, SOC, SEM)
pm2r = m.build_presence_masks(real2, SOC, SEM)
dims = dict(real1["dimensions"])


def mean_deg(raw, rel):
    X = raw[rel]; d = np.diff(sp.csr_matrix(X).indptr)
    return float(d[d > 0].mean())


DEG = {r: mean_deg(real2, r) for r in SOC + SEM}
if REGIME == "dense":
    DEG = {r: 3 * v for r, v in DEG.items()}


def live_sets(rng):
    L1, L2 = {}, {}
    for f in GLOBAL:
        if f not in pm1r:
            continue
        n1, n2 = int(pm1r[f].sum()), int(pm2r[f].sum())
        nb = int((pm1r[f] & pm2r[f]).sum()) if REGIME == "toy" else int(0.5 * n2)
        nb = min(nb, n1, n2, n1 + n2 - nb)
        perm = rng.permutation(dims[f])
        both, only1 = perm[:nb], perm[nb:n1]
        only2 = perm[n1:n1 + n2 - nb]
        L1[f] = np.sort(np.concatenate([both, only1])); L2[f] = np.sort(np.concatenate([both, only2]))
    return L1, L2


def plant(rng, L1, L2):
    """z1 over T1-live, z2 over T2-live (-1 elsewhere); scenario rules; allowed T1->T2 map."""
    K2 = {"growth": 4, "migration": 4, "split": 5, "merge": 3, "dissolve_emerge": 4}[SCENARIO]
    new_pool = {"growth": [0, 1, 2, 3], "migration": [0, 1, 2, 3], "split": [0, 1, 2, 3, 4],
                "merge": [0, 1, 2], "dissolve_emerge": [0, 1, 2, 4]}[SCENARIO]
    allowed = {"growth": {0: {0}, 1: {1}, 2: {2}, 3: {3}}, "migration": {0: {0}, 1: {1}, 2: {2}, 3: {3}},
               "split": {0: {0, 4}, 1: {1}, 2: {2}, 3: {3}}, "merge": {0: {0}, 1: {1}, 2: {2}, 3: {2}},
               "dissolve_emerge": {0: {0}, 1: {1}, 2: {2}}}[SCENARIO]
    z1, z2, group = {}, {}, {}
    for f in GLOBAL:
        if f not in L1:
            continue
        a = -np.ones(dims[f], int); b = -np.ones(dims[f], int); g = np.array(["none"] * dims[f], dtype=object)
        a[L1[f]] = rng.integers(0, K1, len(L1[f]))
        pers = np.intersect1d(L1[f], L2[f]); new = np.setdiff1d(L2[f], L1[f])
        b[new] = rng.choice(new_pool, len(new)); g[new] = "new"
        zp = a[pers].copy(); gp = np.array(["persisting"] * len(pers), dtype=object)
        if SCENARIO == "migration":
            mv = rng.random(len(pers)) < 0.2
            zp[mv] = (zp[mv] + rng.integers(1, K1, mv.sum())) % K1; gp[mv] = "migrant"
        elif SCENARIO == "split":
            s0 = zp == 0
            zp[s0] = np.where(rng.random(s0.sum()) < 0.5, 0, 4); gp[s0] = "split"
        elif SCENARIO == "merge":
            gp[np.isin(zp, [2, 3])] = "merged"; zp[zp == 3] = 2
        elif SCENARIO == "dissolve_emerge":
            d3 = zp == 3
            zp[d3] = rng.integers(0, 3, d3.sum()); gp[d3] = "dissolved"
            g[new[b[new] == 4]] = "emerged"
        b[pers] = zp; g[pers] = gp
        z1[f], z2[f], group[f] = a, b, g
    z1["art"] = rng.integers(0, K1, dims["art"])
    n_art2 = int(real2["S_Art_Auth"].shape[0])
    z2["art"] = rng.choice(new_pool, n_art2); group["art"] = np.array(["new"] * n_art2, dtype=object)
    return z1, z2, group, K2, allowed


def synth_slice(rng, live, z, n_art):
    """One slice: the config's relations, community-biased, degree from Poisson."""
    out = {}
    for rel in SOC + SEM:
        fa, fb = m.RELATION_MAP[rel]
        na = n_art if fa == "art" else dims[fa]; nb = n_art if fb == "art" else dims[fb]
        rows_live = np.arange(n_art) if fa == "art" else live[fa]
        cols_live = np.arange(n_art) if fb == "art" else live[fb]
        zr, zc = z[fa], z[fb]
        R, C = [], []
        for i in rows_live:
            d = min(max(1, rng.poisson(DEG[rel])), len(cols_live))
            same = zc[cols_live] == zr[i]
            w = np.ones(len(cols_live))
            W, T = same.sum(), len(cols_live)
            if 0 < W < T:
                w[same] = WITHIN * (T - W) / ((1 - WITHIN) * W)
            pick = rng.choice(len(cols_live), size=d, replace=False, p=w / w.sum())
            R += [i] * d; C += list(cols_live[pick])
        R, C = np.array(R), np.array(C)
        orph = np.setdiff1d(cols_live, C)
        if len(orph):                       # every live column keeps a tie
            R = np.concatenate([R, rng.choice(rows_live, len(orph))]); C = np.concatenate([C, orph])
        X = sp.csr_matrix((np.ones(len(R)), (R, C)), shape=(na, nb)); X.data[:] = 1.0
        out[rel] = X
    d = dict(dims); d["art"] = n_art
    out["dimensions"] = d
    return out


def fit(raw, K, prior=None, lam=0.0):
    m.set_seeds(42)
    U, Z, d = m.run_inner_solver(raw_data=raw, soc_keys=SOC, sem_keys=SEM, anchor_keys=ANC, K=K,
                                 dimensions=raw["dimensions"],
                                 params={"lambda_z_offdiag": LZ, "lambda_temporal": lam},
                                 device=m.DEVICE, seed_function=m.set_seeds,
                                 inner_epochs=m.EXTENDED_EPOCHS, temporal_prior=prior)
    Un = {f: u.cpu().numpy().astype(np.float64) for f, u in U.items()}
    Zn = {r: z.cpu().numpy().astype(np.float64) for r, z in Z.items()}
    return Un, Zn, d


def score(Un, Zn, raw, z, group, Ktrue):
    pm = m.build_presence_masks(raw, SOC, SEM)
    P, eng = m.compute_weighted_membership(Un, Zn, pm)
    K = next(iter(P.values())).shape[1]
    cont = np.zeros((Ktrue, K))
    rows = []
    for f in P:
        idx = np.where(pm[f] & eng[f] & (z[f] >= 0))[0]
        a = P[f][idx].argmax(1); t = z[f][idx]
        np.add.at(cont, (t, a), 1)
        rows += [(f, i, int(ti), int(ai), group[f][i] if f in group else "na") for i, ti, ai in zip(idx, t, a)]
    r, c = linear_sum_assignment(-cont)
    fit_to_true = {int(cc): int(rr) for rr, cc in zip(r, c)}
    res = {"accuracy": {}, "fit_to_true": fit_to_true}
    for gname in ["all", "persisting", "new", "migrant", "split", "merged", "dissolved", "emerged"]:
        sel = [(t, a) for (_, _, t, a, g) in rows if gname == "all" or g == gname]
        if sel:
            res["accuracy"][gname] = float(np.mean([fit_to_true.get(a, -1) == t for t, a in sel]))
    # fitted communities made almost only of new entities
    newonly = []
    for k in range(K):
        mem = [g for (_, _, _, a, g) in rows if a == k]
        if len(mem) >= 10:
            share_new = np.mean([g in ("new", "emerged") for g in mem])
            if share_new >= 0.9:
                newonly.append({"fitted": k, "true": fit_to_true.get(k), "share_new": float(share_new), "n": len(mem)})
    res["new_only_communities"] = newonly
    return res


results = []
t0 = time.time()
for rep in range(REPS):
    rng = np.random.default_rng(1000 * rep + sum(map(ord, SCENARIO + REGIME)))  # deterministic
    L1, L2 = live_sets(rng)
    z1, z2, group, K2, allowed = plant(rng, L1, L2)
    n_art1, n_art2 = int(real1["S_Art_Auth"].shape[0]), int(real2["S_Art_Auth"].shape[0])
    raw1 = synth_slice(rng, L1, z1, n_art1)
    raw2 = synth_slice(rng, L2, z2, n_art2)
    U1, Z1, d1 = fit(raw1, K1)
    s1 = score(U1, Z1, raw1, z1, {f: np.array(["t1"] * len(v), dtype=object) for f, v in z1.items()}, K1)
    prior = m.temporal_prior_from_model(U1, Z1, raw1, raw2, SOC, SEM, {"synthetic": SCENARIO})
    Ks = [K2] + ([4] if SCENARIO in ("split", "merge") else [])
    for K in Ks:
        for lam in LAMBDAS:
            U2, Z2, d2 = fit(raw2, K, prior if lam > 0 else None, lam)
            s2 = score(U2, Z2, raw2, z2, group, int(max(v.max() for v in z2.values())) + 1)
            match = d2.get("temporal_match") or {}
            checks = []
            for p in match.get("pairs", []):
                t1_true = s1["fit_to_true"].get(p["t1"]); t2_true = s2["fit_to_true"].get(p["t2"])
                checks.append({"t1_true": t1_true, "t2_true": t2_true,
                               "correct": t1_true in allowed and t2_true in allowed[t1_true], "z": p["z"]})
            emerged_matched = (SCENARIO == "dissolve_emerge"
                               and any(c["t2_true"] == 4 for c in checks))
            rec = {"rep": rep, "K": K, "lambda": lam, "converged": d2["converged"],
                   "epochs": len(d2["loss_history"]), "math_loss": d2["math_loss"],
                   "t1_accuracy": s1["accuracy"]["all"], "accuracy": s2["accuracy"],
                   "accepted": len(checks), "false_matches": sum(not c["correct"] for c in checks),
                   "emerged_matched": emerged_matched, "new_only": s2["new_only_communities"],
                   "persisting_n": prior["source"]["n_persisting"]}
            results.append(rec)
            print(f"{SCENARIO}/{REGIME} rep{rep} K={K} lam={lam}: acc {json.dumps({k: round(v, 3) for k, v in s2['accuracy'].items()})} "
                  f"| T1 acc {s1['accuracy']['all']:.3f} | accepted {len(checks)} false {rec['false_matches']} "
                  f"| emerged matched {emerged_matched} | new-only {len(s2['new_only_communities'])} "
                  f"| conv {d2['converged']} {rec['epochs']} ep | {time.time()-t0:.0f}s", flush=True)
json.dump(results, open(f"t3_{SCENARIO}_{REGIME}.json", "w"), indent=1, default=float)
