# F8 (2/3) — job lists for the M1 stages (ticket 107; staged plan plans/ticket107_layer2/test_architecture.txt).
# A job is a dict of every setting of one fit; run_fit.job_id hashes it, so a job list can be re-submitted and finished
# fits are skipped. Lists for later stages are added when their inputs (chosen loss, λ values, J*) are decided.
# Usage: python -m fit.plan checkpoint3 > jobs_checkpoint3.jsonl   (or stage0_pass1 --seeds S)
import argparse, json, sys

PROVISIONAL_FRACTION = {2: 0.03, 1: 0.01}            # largest-stable rule, provisional (2026-11-10)
COARSE_K = (3, 5, 8, 13, 20)


def job(stage, approach, group, K, beta, seed, frac, start="nndsvd_sub0.8", dtype="float64", terms=None, arm=None,
        max_epochs=5000, lr_mode="recompute", warmup=300):
    return {"stage": stage, "approach": approach, "group": group, "K": K, "beta": beta, "seed": seed, "frac": frac,
            "start": start, "dtype": dtype, "terms": terms or {}, "arm": arm or {}, "max_epochs": max_epochs,
            "lr_mode": lr_mode, "warmup": warmup}


def checkpoint3():
    """Checkpoint 3 on CSF3 (decided 2026-11-09/10): hold-out group 0 (the first group), approach 1, K = 5 and 20,
    3 seeds, terms off. (i) fraction check: both losses × 0.01/0.02/0.03 (36 fits; also the timing); (ii) start
    comparison at the provisional fraction: plain NNDSVD and random (24 fits; the hybrid comes from (i));
    (iii) float32 at the provisional fraction (12 fits; float64 counterparts from (i)). 72 fits."""
    out = []
    for beta in (2, 1):
        for K in (5, 20):
            for s in (1, 2, 3):
                for f in (0.01, 0.02, 0.03):
                    out.append(job("checkpoint3", 1, 0, K, beta, s, f))
                for st in ("nndsvd", "random"):
                    out.append(job("checkpoint3", 1, 0, K, beta, s, PROVISIONAL_FRACTION[beta], start=st))
                out.append(job("checkpoint3", 1, 0, K, beta, s, PROVISIONAL_FRACTION[beta], dtype="float32"))
    return out


def stage0_pass1(seeds):
    """Stage 0 pass 1 (decided 2026-11-05/06): both losses × both approaches × coarse K × hold-out groups 1–5 (indices
    0–4) × S seeds, J2, A0, every in-loop term off, hybrid start, fraction per loss (final values from checkpoint 3)."""
    return [job("stage0_pass1", a, g, K, beta, s, PROVISIONAL_FRACTION[beta])
            for a in (1, 2) for beta in (2, 1) for K in COARSE_K for g in range(5) for s in range(1, seeds + 1)]


def rehearsal():
    """CSF3 rehearsal (owner 2026-11-10): 16 tiny real-data fits (K = 3; stopping tolerance 1.0 so every fit stops at
    epoch 21 and counts as converged, letting the aggregation run every check) — approach 1, hold-out groups 0 and 1,
    4 seeds, both losses — plus 1 fit that fails on purpose. Tests: environment and RDS on compute nodes, 8 parallel fits
    per task, outputs, the failure caught and logged, re-submission skipping finished fits, the chained aggregation."""
    out = [dict(job("rehearsal", 1, g, 3, beta, s, PROVISIONAL_FRACTION[beta], max_epochs=40), tol=1.0)
           for beta in (2, 1) for g in (0, 1) for s in (1, 2, 3, 4)]
    out.append(dict(job("rehearsal", 1, 0, 3, 2, 99, 0.03, max_epochs=40), tol=1.0, fail_on_purpose=True))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=("checkpoint3", "stage0_pass1", "rehearsal"))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    jobs = {"checkpoint3": checkpoint3, "rehearsal": rehearsal}.get(a.which, lambda: stage0_pass1(a.seeds))()
    for j in jobs:
        print(json.dumps(j, sort_keys=True))
    print(f"{len(jobs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
