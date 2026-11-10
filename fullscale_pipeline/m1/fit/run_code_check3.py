# Code-check world, round 3 (owner 2026-11-10: hybrid start "if not computationally costly"): NNDSVD on a seed-specific
# 80% subsample of the visible anchor ties (propagation unchanged), same corrected world and settings as round 2
# (per-block steps, fraction 0.01, epoch limit 5,000), 3 seeds × {noiseless squared error, noiseless KL, Poisson KL,
# Gaussian squared error} = 12 fits; also the start diversity between seeds (vs plain NNDSVD).
import argparse, json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ARMS = [("noiseless", 2), ("noiseless", 1), ("poisson", 1), ("gaussian", 2)]

def run(job, max_epochs):
    import torch
    torch.set_num_threads(1)
    from fit import data as D, worlds as Wd
    from fit.solver import fit
    from fit.run_code_check import summarise
    rds, Ut, Zt = Wd.code_check_world(job["world"], seed=1, build=D.Build(), child_only="split")
    r = fit(rds, 6, job["beta"], seed=job["seed"], max_epochs=max_epochs, lr=0.01, lr_mode="relative", start="nndsvd_sub0.8")
    out = summarise(r, rds, Ut, Zt); out["job"] = job
    return out

def start_diversity():
    from fit import data as D, worlds as Wd
    from fit.init import initialise
    rds, _, _ = Wd.code_check_world("noiseless", seed=1, build=D.Build(), child_only="split")
    out = {}
    for name, kw in (("nndsvd", {}), ("nndsvd_sub0.8", {"subsample": 0.8})):
        S = [initialise(rds, 6, s, **kw)[0] for s in (1, 2, 3)]
        d = [np.mean([np.linalg.norm(S[a][f] - S[b][f]) / np.linalg.norm(S[a][f]) for f in S[a]]) for a, b in ((0, 1), (0, 2), (1, 2))]
        out[name] = float(np.mean(d))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--max-epochs", type=int, default=5000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    json.dump(start_diversity(), open(os.path.join(a.out, "start_diversity.json"), "w"))
    jobs = [dict(world=w, beta=b, seed=s) for w, b in ARMS for s in (1, 2, 3)]
    t0 = time.time()
    with ProcessPoolExecutor(12) as ex, open(os.path.join(a.out, "fits.jsonl"), "w") as fh:
        for f in as_completed([ex.submit(run, j, a.max_epochs) for j in jobs]):
            fh.write(json.dumps(f.result()) + "\n"); fh.flush()
    print(f"{len(jobs)} fits, {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
