"""Ticket 105, test T0: chunk13v10.py without a prior (or with a prior at weight 0)
must reproduce chunk13v9.py (v9.4) bit for bit, on T1 and on T2.

    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE python t0_regression.py

Reference (CLAUDE.md §6, incline35): v9.4 C1/K=3, lambda_z_offdiag=0.5727904470799616,
seed 42 -> math_loss 0.8629798889160156, 770 epochs. torch is imported before numpy
(ticket 104).
"""
import importlib.util, os, sys, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # module import writes results/<version>/ here


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


v9 = load("c13v9", f"{E}/chunk13v9.py")
v10 = load("c13v10", f"{E}/chunk13v10.py")
import numpy as np

LZ = 0.5727904470799616


def fit(mod, raw, cfg, K, prior=None, extra=None):
    soc, sem, anc = mod.get_active_facets(cfg)
    mod.set_seeds(42)
    params = {"lambda_z_offdiag": LZ, **(extra or {})}
    kw = {"temporal_prior": prior} if prior is not None else {}
    U, Z, d = mod.run_inner_solver(raw_data=raw, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=K,
                                   dimensions=raw["dimensions"], params=params, device=mod.DEVICE,
                                   seed_function=mod.set_seeds, **kw)
    return U, Z, d


def same(a, b):
    (Ua, Za, da), (Ub, Zb, db) = a, b
    eqU = all(torch.equal(Ua[f], Ub[f]) for f in Ua) and Ua.keys() == Ub.keys()
    eqZ = all(torch.equal(Za[r], Zb[r]) for r in Za) and Za.keys() == Zb.keys()
    return eqU and eqZ and da["loss_history"] == db["loss_history"]


ok = True
raw1 = v9.load_and_validate_data(OUT + "Star_extended_matrices_t1_v2.pkl")
raw2 = v9.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
for slice_name, raw in (("T1", raw1), ("T2", raw2)):
    for cfg, K in (("C1", 3), ("C2", 4), ("C6", 4)):
        a = fit(v9, raw, cfg, K)
        b = fit(v10, raw, cfg, K)
        r = same(a, b)
        ok &= r
        line = (f"{slice_name} {cfg} K={K}: v9.4 {a[2]['math_loss']!r} / {len(a[2]['loss_history'])} | "
                f"v10 no prior {b[2]['math_loss']!r} / {len(b[2]['loss_history'])} | identical={r}")
        if slice_name == "T2":
            # a prior at weight 0 must change nothing either
            soc, sem, _ = v10.get_active_facets(cfg)
            sel = __import__("json").load(open(f"{E}/results/v9.4.t1_v2/model_selection.json"))[cfg]
            mdir = f"{E}/results/v9.4.t1_v2/{cfg}/K_{sel['K']}/pareto_models/{sel['chosen']['trial']}"
            prior = v10.build_temporal_prior(mdir, raw1, raw2, soc, sem)
            c = fit(v10, raw, cfg, K, prior=prior, extra={"lambda_temporal": 0.0})
            r0 = same(a, c)
            ok &= r0
            line += f" | v10 prior at weight 0 identical={r0} (temporal_match={c[2]['temporal_match']})"
        print(line, flush=True)

ref = fit(v10, raw1, "C1", 3)[2]
hit = ref["math_loss"] == 0.8629798889160156 and len(ref["loss_history"]) == 770
print(f"C1/K=3 T1 reference: {ref['math_loss']!r} / {len(ref['loss_history'])} epochs; "
      f"matches CLAUDE.md §6 value: {hit}")
ok &= hit
print("T0", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
