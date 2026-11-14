# F8 (2/3) — job lists for the M1 stages (ticket 107; staged plan plans/ticket107_layer2/test_architecture.txt).
# A job is a dict of every setting of one fit; run_fit.job_id hashes it, so a job list can be re-submitted and finished
# fits are skipped. Lists for later stages are added when their inputs (chosen loss, λ values, J*) are decided.
# Usage: python -m fit.plan checkpoint3 > jobs_checkpoint3.jsonl   (or stage0_pass1 --seeds S)
import argparse, json, os, sys

PROVISIONAL_FRACTION = {2: 0.03, 1: 0.01}            # largest-stable rule, provisional (2026-11-10); kept for checkpoint 3
FRACTION = {2: 0.03, 1: 0.03}                        # DECIDED 2026-11-13 (owner: "I approve option b for the fraction rule/learning rate"): rule from the smallest fraction converging at every K (0.02), RL-218/219
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
    0–4) × S seeds, J2, A0, every in-loop term off, hybrid start, fraction per loss (FRACTION, 2026-11-13), float32 (decided 2026-11-11)."""
    return [job("stage0_pass1", a, g, K, beta, s, FRACTION[beta], dtype="float32")
            for a in (1, 2) for beta in (2, 1) for K in COARSE_K for g in range(5) for s in range(1, seeds + 1)]


def kl_fraction_extension():
    """KL fraction grid extension (owner 2026-11-12: "write the test for KL at 0.05 and 0.1. Maybe also at 0.075"): KL,
    fractions 0.05 / 0.075 / 0.1, K = 5 and 20, 3 seeds, hybrid start, float64 (to compare with checkpoint 3's float64
    reference at 0.03 under the same arithmetic), hold-out group 1, approach 1, terms off. 18 fits; stage name
    "checkpoint3" so the checkpoint-3 analysis reads them with the earlier fits."""
    return [job("checkpoint3", 1, 0, K, 1, s, f) for f in (0.05, 0.075, 0.1) for K in (5, 20) for s in (1, 2, 3)]


TIMING_ARMS = {                                   # one value per term; values do not matter for the cost per epoch
    "off": {},
    "w_regulariser": {"lambda_child": 0.1},
    "floor": {"floor": {"lambda": 0.1, "gamma": 0.8, "omega": 1.0, "beta_w": 0.0}},
    "floor_with_W": {"floor": {"lambda": 0.1, "gamma": 0.8, "omega": 1.0, "beta_w": 0.5}},
    "ceiling": {"ceiling": {"lambda": 0.1, "rho": 1.0, "beta_w": 0.0, "facets": ["child", "article"]}},
    "o3": {"o3": {"lambda": 0.1}},
    "all": {"lambda_child": 0.1, "floor": {"lambda": 0.1, "gamma": 0.8, "omega": 1.0, "beta_w": 0.5},
            "ceiling": {"lambda": 0.1, "rho": 1.0, "beta_w": 0.5, "facets": ["child", "article"]}, "o3": {"lambda": 0.1}},
}


def timing_terms():
    """Timing test with the in-fit terms on (owner 2026-11-13: "I am ready to submit the timing job"): cost per epoch of
    each term, before (epochs < 300, terms off) and after the warm-up, at the stage settings (fraction 0.03, float32,
    hybrid start, approach 1, hold-out group 1, seed 1); both losses × K = 5, 20 × 7 arms = 28 fits, each stopped at
    900 epochs (300 warm-up + 600 with terms). Term values are placeholders: they set the cost, not a calibration."""
    return [dict(job("timing_terms", 1, 0, K, beta, 1, FRACTION[beta], dtype="float32", terms=t, arm={"timing": name},
                     max_epochs=900), tol=0.0)                     # tol 0: no early stop, every fit runs 900 epochs
            for beta in (2, 1) for K in (5, 20) for name, t in TIMING_ARMS.items()]


def stage0_extension(root="/mnt/hum01-rds/Basov/p91688di/m1_fits"):
    """Stage 0 pass 1 extension (owner 2026-11-14, decisions 1 and 2): (i) the pre-registered K = 32 (every loss's band
    reached 20): both approaches × both losses × groups 1–5 × 10 seeds = 200 fits; (ii) the 13 squared-error approach-1
    K = 13 fits that reached the 5,000-epoch ceiling, rerun from the same start. Both with an 8,000-epoch ceiling (a
    ceiling only; aggregate.latest_fits lets a rerun replace its original)."""
    import glob
    out = [job("stage0_pass1", a, g, 32, beta, s, FRACTION[beta], dtype="float32", max_epochs=8000)
           for a in (1, 2) for beta in (2, 1) for g in range(5) for s in range(1, 11)]
    for d in sorted(glob.glob(os.path.join(root, "stage0_pass1", "*", "result.json"))):
        r = json.load(open(d)); j = json.load(open(os.path.join(os.path.dirname(d), "spec.json")))["job"]
        if (j["approach"], j["beta"], j["K"]) == (1, 2, 13) and not r["converged"]:
            out.append(dict(j, max_epochs=8000))
    return out


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
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=("checkpoint3", "stage0_pass1", "rehearsal", "kl_fraction_extension", "timing_terms", "stage0_extension"))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    jobs = {"checkpoint3": checkpoint3, "rehearsal": rehearsal, "kl_fraction_extension": kl_fraction_extension, "timing_terms": timing_terms, "stage0_extension": stage0_extension}.get(a.which, lambda: stage0_pass1(a.seeds))()
    for j in jobs:
        print(json.dumps(j, sort_keys=True))
    print(f"{len(jobs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
