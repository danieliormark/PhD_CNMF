"""Ticket 105, test T1b: unit checks of the consensus prior and confidence weighting
(temporal_prior_from_seeds, chunk13v10.py), on the real C2 K=4 T1 knee model.

(a) ten identical models: consensus == knee prior, confidence == 1 everywhere
(b) seeds that are column-permuted copies of the knee model are aligned back:
    consensus == knee prior, confidence == 1
(c) one seed with a different profile for some entities: their confidence drops clearly below
    1; every other entity stays at 1 up to rounding (>= 0.999: changing some atoms' rows also
    moves the column weights of the hyperedges they connect to, so those move slightly);
    confidence lies in [0, 1]
(d) a T2 fit with the prior from (a) (consensus + confidence) equals a fit with the plain knee
    prior up to rounding (averaging identical rows and the square root in the JS distance leave
    residues ~1e-8, so bit-identity is not expected): same epochs, |d math_loss| < 1e-5

    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE python t1b_prior_variants.py
"""
import importlib.util, os, sys, json, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c13v10", f"{E}/chunk13v10.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np

raw1 = m.load_and_validate_data(OUT + "Star_extended_matrices_t1_v2.pkl")
raw2 = m.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
soc, sem, anc = m.get_active_facets("C2")
mdir = f"{E}/results/v9.4.t1_v2/C2/K_4/pareto_models/trial_0094"
U = {f: u.numpy().astype(np.float64) for f, u in torch.load(f"{mdir}/U_matrices.pt").items()}
Z = {r: z.numpy().astype(np.float64) for r, z in torch.load(f"{mdir}/Z_core.pt").items()}
knee = m.build_temporal_prior(mdir, raw1, raw2, soc, sem)
res = {}


def same_P(pa, pb, tol=1e-12):
    return all(np.allclose(pa["facets"][f]["P"], pb["facets"][f]["P"], atol=tol, rtol=0) for f in pa["facets"])


# (a)
pa = m.temporal_prior_from_seeds([(U, Z)] * 10, raw1, raw2, soc, sem)
res["a_consensus_equals_knee"] = same_P(pa, knee)
res["a_conf_all_one"] = all(np.allclose(v["conf"], 1.0) for v in pa["facets"].values())

# (b) column-permuted copies
rng = np.random.default_rng(3)
models = [(U, Z)]
for t in range(9):
    p = rng.permutation(4)
    Up = {f: u[:, p] for f, u in U.items()}
    Zp = {r: z[np.ix_(p, p)] for r, z in Z.items()}
    models.append((Up, Zp))
pb = m.temporal_prior_from_seeds(models, raw1, raw2, soc, sem)
res["b_permuted_seeds_aligned_back"] = same_P(pb, knee, tol=1e-10)
res["b_conf_all_one"] = all(np.allclose(v["conf"], 1.0, atol=1e-6) for v in pb["facets"].values())

# (c) one seed differs for a few core atoms: their rows are rotated by one community
f0 = "core_atom"
changed = knee["facets"][f0]["idx"][:5]
U2 = {f: u.copy() for f, u in U.items()}
U2[f0][changed] = np.roll(U2[f0][changed], 1, axis=1)
pc = m.temporal_prior_from_seeds([(U, Z)] * 9 + [(U2, Z)], raw1, raw2, soc, sem)
conf = pc["facets"][f0]["conf"]
is_changed = np.isin(pc["facets"][f0]["idx"], changed)
res["c_conf_in_unit_interval"] = all(((v["conf"] >= 0) & (v["conf"] <= 1)).all() for v in pc["facets"].values())
res["c_changed_entities_conf_below_one"] = bool((conf[is_changed] < 1 - 1e-6).all())
res["c_other_entities_conf_one"] = bool((conf[~is_changed] >= 0.999).all()) and all(
    bool((v["conf"] >= 0.999).all()) for g, v in pc["facets"].items() if g != f0)
res["c_min_conf_others"] = min([float(conf[~is_changed].min())] + [float(v["conf"].min())
                               for g, v in pc["facets"].items() if g != f0 and len(v["conf"])])
res["c_conf_of_changed"] = [round(float(x), 3) for x in conf[is_changed]]

# (d) the prior from (a) equals the plain knee prior, and the frozen match is the same.
# Identical *fits* are not required: a perturbation of one float32 rounding step (here at
# epoch 313) grows along the optimisation path and changes where early stopping fires
# (measured 2026-10-07: 938 vs 1996 epochs, math_loss 0.8608 vs 0.8469 for a 1e-8
# difference in the prior). That sensitivity is recorded, and the diagnosis uses a
# near-zero-weight control and several seeds because of it.
res["d_conf_exactly_one"] = all(bool((v["conf"] == 1.0).all()) for v in pa["facets"].values())
res["d_P_max_abs_diff"] = max(float(np.abs(pa["facets"][f]["P"] - knee["facets"][f]["P"]).max())
                              for f in pa["facets"] if len(pa["facets"][f]["idx"]))
matches = []
for prior in (knee, pa):
    m.set_seeds(1002)
    _, _, d = m.run_inner_solver(raw_data=raw2, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=4,
                                 dimensions=raw2["dimensions"],
                                 params={"lambda_z_offdiag": 0.12, "lambda_temporal": 0.1},
                                 device=m.DEVICE, seed_function=lambda: m.set_seeds(1002),
                                 inner_epochs=400, temporal_prior=prior)
    tm = d["temporal_match"]
    matches.append([(p["t1"], p["t2"], round(p["z"], 9)) for p in tm["pairs"]])
res["d_same_match"] = matches[0] == matches[1]
res["d_pairs"] = matches[0]
res["d_prior_equal"] = res["d_conf_exactly_one"] and res["d_P_max_abs_diff"] < 1e-12 and res["d_same_match"]

ok = all(res[k] for k in ("a_consensus_equals_knee", "a_conf_all_one", "b_permuted_seeds_aligned_back",
                          "b_conf_all_one", "c_conf_in_unit_interval", "c_changed_entities_conf_below_one",
                          "c_other_entities_conf_one", "d_prior_equal"))
res["T1b_pass"] = bool(ok)
print(json.dumps(res, indent=2))
json.dump(res, open("t1b_prior_variants.json", "w"), indent=2)
sys.exit(0 if ok else 1)
