# Code-check world, round 2 (owner 2026-11-09: "I approve adding the random start +O3 arm to the corrected world run").
# Corrected world: the child-only community is tied to articles of community 0 and parents of community 2 (identifiable).
# Per-block step sizes (step = fraction × RMS of the block's starting values), fractions 0.01 and 0.03, epoch limit
# 5,000. Arms: NNDSVD start; random start; random start + O3 (off-diagonal tie-weight share, candidate) at λ 0.1 and 1,
# switched on from the first epoch (no freeze needed). Noisy worlds at fraction 0.01 with a fit started at the truth.
# Usage (from fullscale_pipeline/m1): python -m fit.run_code_check2 --out DIR [--workers 14]
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

def job_list():
    jobs = []
    for beta in (2, 1):
        for frac in (0.01, 0.03):
            for s in (1, 2, 3):
                jobs.append(dict(world="noiseless", beta=beta, frac=frac, start="nndsvd", seed=s, o3=0.0))
                jobs.append(dict(world="noiseless", beta=beta, frac=frac, start="random", seed=s, o3=0.0))
        for lam in (0.1, 1.0):
            for s in (1, 2, 3):
                jobs.append(dict(world="noiseless", beta=beta, frac=0.01, start="random", seed=s, o3=lam))
    for world, beta in (("poisson", 1), ("gaussian", 2)):
        jobs.append(dict(world=world, beta=beta, frac=0.01, start="truth", seed=0, o3=0.0))
        for s in (1, 2, 3):
            for start, o3 in (("nndsvd", 0.0), ("random", 0.0), ("random", 1.0)):
                jobs.append(dict(world=world, beta=beta, frac=0.01, start=start, seed=s, o3=o3))
    return jobs

def run(job, max_epochs):
    import torch
    torch.set_num_threads(1)
    from fit import data as D, worlds as Wd, terms as T
    from fit.solver import fit
    from fit.run_code_check import summarise
    rds, Ut, Zt = Wd.code_check_world(job["world"], seed=1, build=D.Build(), child_only="split")
    terms = [(T.OffDiagonalShare([rd.name for rd in rds]), job["o3"])] if job["o3"] > 0 else None
    kw = dict(init_U=Ut, init_Z=Zt) if job["start"] == "truth" else dict(start=job["start"])
    r = fit(rds, 6, job["beta"], seed=job["seed"], max_epochs=max_epochs, lr=job["frac"], lr_mode="relative",
            terms=terms, warmup=0, **kw)
    out = summarise(r, rds, Ut, Zt)
    out["job"] = job
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-epochs", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = job_list(); t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(os.path.join(a.out, "fits.jsonl"), "w") as fh:
        futs = [ex.submit(run, j, a.max_epochs) for j in jobs]
        for f in as_completed(futs):
            fh.write(json.dumps(f.result()) + "\n"); fh.flush()
    print(f"{len(jobs)} fits, {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
