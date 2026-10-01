# Ticket 102 calibration grid. One fit per (config, K, lambda_conc, gamma, warmup, seed);
# lambda_z_offdiag per cell = the median-math_loss stability-verified archived v9.2 model
# (same rule as last session's refit_cell.py). Saves U, Z, diagnostics per fit.
import importlib.util, itertools, json, os, sys
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
DATA = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1_v2.pkl"
OUT = os.path.join(HERE, sys.argv[1]); os.makedirs(OUT, exist_ok=True)
STAGE = sys.argv[1]
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6"]; KS = [2, 3, 4, 5]
SEEDS = [1000, 1001, 1002, 1003, 1004]


def cell_lambda(c, K):
    kd = f"{EXE}/results/v9.2.t1_v2/{c}/K_{K}/pareto_models"
    if not os.path.exists(f"{kd}/experiment_manifest.json"):
        return None
    ok = []
    for info in json.load(open(f"{kd}/experiment_manifest.json"))["archived_models"]:
        md = json.load(open(f"{kd}/{info['folder_name']}/model_metadata.json"))
        if "consensus_track_A_unweighted" in md:
            ok.append((md["optuna_math_loss"], info["folder_name"], md["hyperparameters"]["lambda_z_offdiag"]))
    if not ok:
        return None
    ok.sort()
    return ok[len(ok) // 2]


if STAGE == "main":
    SETTINGS = [(0.0, 0.5, 300)] + [(l, g, 300) for l in [0.03, 0.1, 0.3, 1, 3, 10, 30] for g in [0.5, 0.8]]
elif STAGE == "dual":
    SETTINGS = [(0.1, 0.8, 300)]
else:  # warm-up sensitivity, set by argv[2:] as lambda gamma
    l, g = float(sys.argv[2]), float(sys.argv[3])
    SETTINGS = [(l, g, w) for w in [100, 600]]

m = raw = None


def init():
    global m, raw
    os.chdir(OUT)
    spec = importlib.util.spec_from_file_location("c13", os.path.join(HERE, "chunk13v9_conc.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    raw = m.load_and_validate_data(DATA)


def fit(job):
    c, K, trial, lz, lc, g, w, s = job
    tag = f"{c}_K{K}_l{lc}_g{g}_w{w}_s{s}"
    if os.path.exists(f"{OUT}/{tag}.json"):
        return tag
    soc, sem, anc = m.get_active_facets(c)
    m.set_seeds(s)
    U, Z, d = m.run_inner_solver(raw_data=raw, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=K,
                                 dimensions=raw["dimensions"], inner_epochs=8000,
                                 params={"lambda_z_offdiag": lz, "lambda_conc": lc, "conc_gamma": g, "conc_warmup": w, "conc_dual_stop": STAGE == "dual"},
                                 device=m.DEVICE, seed_function=lambda s=s: m.set_seeds(s))
    np.savez_compressed(f"{OUT}/{tag}.npz", **{f"U__{f}": t.cpu().numpy() for f, t in U.items()},
                        **{f"Z__{r}": t.cpu().numpy() for r, t in Z.items()})
    json.dump(dict(config=c, K=K, trial=trial, lambda_z_offdiag=lz, lambda_conc=lc, gamma=g, warmup=w, seed=s,
                   math_loss=d["math_loss"], epochs=len(d["loss_history"]), converged=bool(d["converged"]),
                   conc_loss=d["conc_loss"]), open(f"{OUT}/{tag}.json", "w"))
    return tag


if __name__ == "__main__":
    jobs = []
    for c, K in itertools.product(CONFIGS, KS):
        cl = cell_lambda(c, K)
        if cl is None:
            print("no verified model", c, K); continue
        for (lc, g, w), s in itertools.product(SETTINGS, SEEDS):
            jobs.append((c, K, cl[1], cl[2], lc, g, w, s))
    print(len(jobs), "fits", flush=True)
    with Pool(28, initializer=init) as p:
        for i, t in enumerate(p.imap_unordered(fit, jobs)):
            if i % 100 == 0:
                print(i, t, flush=True)
    print("DONE", flush=True)
