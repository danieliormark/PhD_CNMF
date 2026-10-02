"""Facet-level reading of Track A/B (FINDINGS §32). Refits one archived model at the 10 stability
seeds, then for every pair of seeds (and a row-shuffled control pair, as in Section 5) reports per
facet: Track A, Track B, and the share of live entities whose main community is the same in both
fits after matching (plain and weighted by the entity's mass in fit 1), plus the chance-adjusted
score (real - chance) / (1 - chance).

    python facet_stability.py C2 4 trial_0094          (run from this directory, tensor_env)

The module is loaded before numpy is imported (ticket 104). Run with the launcher's arithmetic
settings (ATEN_CPU_CAPABILITY=default, MKL_CBWR=COMPATIBLE, single-threaded) to match CSF.
"""
import importlib.util, json, sys, itertools
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
spec = importlib.util.spec_from_file_location("c13", f"{E}/chunk13v9.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
from scipy.optimize import linear_sum_assignment

c, K, t = sys.argv[1], int(sys.argv[2]), sys.argv[3]
md = json.load(open(f"{E}/results/{m.PIPELINE_VERSION}/{c}/K_{K}/pareto_models/{t}/model_metadata.json"))
raw = m.load_and_validate_data(f"{E}/../outputs/Star_extended_matrices_t1_v2.pkl")
soc, sem, anc = m.get_active_facets(c); facets = m.get_required_facets(soc, sem, anc)
pm = m.build_presence_masks(raw, soc, sem)
fits = []
for s in m.STABILITY_SEEDS:
    m.set_seeds(s)
    U, Z, d = m.run_inner_solver(raw_data=raw, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=K, dimensions=raw["dimensions"],
                                 params=md["hyperparameters"], inner_epochs=m.EXTENDED_EPOCHS, device=m.DEVICE,
                                 seed_function=lambda s=s: m.set_seeds(s))
    if d["converged"]:
        fits.append({f: U[f].cpu().numpy() for f in facets})
print("converged", len(fits), "of", len(m.STABILITY_SEEDS))


def match_B(U1, U2):
    S1 = np.vstack([U1[f] for f in facets]); S2 = np.vstack([U2[f] for f in facets])
    cost = np.array([[1 - S1[:, a] @ S2[:, b] / (np.linalg.norm(S1[:, a]) * np.linalg.norm(S2[:, b]) + 1e-9)
                      for b in range(K)] for a in range(K)])
    return linear_sum_assignment(cost)[1]


rows, rng = [], np.random.default_rng(1000)
for i, j in itertools.combinations(range(len(fits)), 2):
    for shuffled in (False, True):
        U1 = fits[i]
        U2 = {f: (fits[j][f][rng.permutation(len(fits[j][f]))] if shuffled else fits[j][f]) for f in facets}
        A, B, fa, fb = m._pair_tracks(U1, U2, facets, K)
        perm = match_B(U1, U2)
        agree, wagree = {}, {}
        for f in facets:
            live = pm[f] if f in pm else np.ones(len(U1[f]), bool)
            a1 = U1[f][live].argmax(1); a2 = U2[f][live][:, perm].argmax(1); w = U1[f][live].sum(1)
            agree[f] = float((a1 == a2).mean()); wagree[f] = float(((a1 == a2) * w).sum() / w.sum())
        rows.append(dict(shuffled=shuffled, A=A, B=B, fa=fa, fb=fb, agree=agree, wagree=wagree))
json.dump(rows, open(f"facet_stability_{c}_K{K}_{t}.json", "w"), default=float)

adj = lambda r, s: (r - s) / (1 - s)
R = [r for r in rows if not r["shuffled"]]; S = [r for r in rows if r["shuffled"]]
mean = lambda key, X, f=None: float(np.mean([x[key] if f is None else x[key][f] for x in X]))
print(f"model: A {mean('A', R):.3f} (chance {mean('A', S):.3f}, adj {adj(mean('A', R), mean('A', S)):.2f}, SD {np.std([r['A'] for r in R], ddof=1):.3f})"
      f" | B {mean('B', R):.3f} (chance {mean('B', S):.3f}, adj {adj(mean('B', R), mean('B', S)):.2f}, SD {np.std([r['B'] for r in R], ddof=1):.3f})")
for f in facets:
    print(f"{f:14s} A {mean('fa', R, f):.3f}/{mean('fa', S, f):.3f} adj {adj(mean('fa', R, f), mean('fa', S, f)):+.2f} | "
          f"B {mean('fb', R, f):.3f}/{mean('fb', S, f):.3f} adj {adj(mean('fb', R, f), mean('fb', S, f)):+.2f} | "
          f"same main community {mean('agree', R, f):.3f}/{mean('agree', S, f):.3f} adj {adj(mean('agree', R, f), mean('agree', S, f)):+.2f} | "
          f"mass-weighted {mean('wagree', R, f):.3f}/{mean('wagree', S, f):.3f}")
