# Code-check world, round 4 (owner 2026-11-10: "I approve testing the recomputed scale"): per-block learning rate
# recomputed every step from the block's current RMS (Adafactor/LAMB-like; floor 1e-3 × the block's starting RMS),
# fractions 0.01 and 0.03, both losses, hybrid start (subsample NNDSVD 80%), round-2 world, limit 5,000 epochs;
# 4 worlds × 2 fractions × 3 seeds = 24 fits. Also logs each block's RMS over the fit (owner's question: do the losses
# drive the blocks to different scales, i.e. different recomputed rates?).
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ARMS = [("noiseless", 2), ("noiseless", 1), ("poisson", 1), ("gaussian", 2)]

def run(job, max_epochs):
    import torch
    torch.set_num_threads(1)
    from fit import data as D, worlds as Wd
    from fit.solver import fit
    from fit.run_code_check import summarise
    rds, Ut, Zt = Wd.code_check_world(job["world"], seed=1, build=D.Build(), child_only="split")
    r = fit(rds, 6, job["beta"], seed=job["seed"], max_epochs=max_epochs, lr=job["frac"], lr_mode="recompute",
            start="nndsvd_sub0.8")
    out = summarise(r, rds, Ut, Zt); out["job"] = job
    out["block_rms_first"] = r["logs"][0]["block_rms"]; out["block_rms_last"] = r["logs"][-1]["block_rms"]
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--max-epochs", type=int, default=5000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    jobs = [dict(world=w, beta=b, frac=f, seed=s) for w, b in ARMS for f in (0.01, 0.03) for s in (1, 2, 3)]
    t0 = time.time()
    with ProcessPoolExecutor(14) as ex, open(os.path.join(a.out, "fits.jsonl"), "w") as fh:
        for f in as_completed([ex.submit(run, j, a.max_epochs) for j in jobs]):
            fh.write(json.dumps(f.result()) + "\n"); fh.flush()
    print(f"{len(jobs)} fits, {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
