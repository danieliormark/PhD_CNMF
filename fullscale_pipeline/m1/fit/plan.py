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


def stage0_pass2(basis="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/pass2_lambda_basis_v2.json"):
    """Stage 0 pass 2, λ_child per loss (decided 2026-11-06; owner 2026-11-14/16): the contrast form R ÷ S (option B,
    terms.WordSimilarityContrast, adopted 2026-11-16 after the A/B test RL-237); each loss at its own K* (squared error
    20, KL 13), both approaches; λ_child = r × (data loss ÷ (R ÷ S)) from the pass-1 λ = 0 fits (pass2_lambda.py, hold-out
    group 1, mean over 10 seeds); r per loss (2026-11-16): squared error {0.003, 0.01, 0.03}, KL {0.03, 0.1, 0.3};
    groups 1–5 × 10 seeds; fraction 0.03, float32, hybrid start, warm-up 300, 8,000-epoch ceiling. 600 fits. r = 1 for KL
    only if r = 0.3 is chosen (owner told before submission)."""
    b = json.load(open(basis))
    kstar = {2: 20, 1: 13}
    grid = {2: (0.003, 0.01, 0.03), 1: (0.03, 0.1, 0.3)}
    out = []
    for a in (1, 2):
        for beta in (2, 1):
            e = b[f"{a}_{beta}"]
            for r in grid[beta]:
                lam = r * e["data_loss"] / e["Rc"]
                out += [job("stage0_pass2", a, g, kstar[beta], beta, s, FRACTION[beta], dtype="float32",
                            terms={"lambda_child": lam, "w_form": "contrast"}, arm={"r": r}, max_epochs=8000)
                        for g in range(5) for s in range(1, 11)]
    return out


def stage0_pass2_kl_r1(basis="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/pass2_lambda_basis_v2.json"):
    """Highest-grid-value rule for KL (decided 2026-11-14; fired in pass 2, RL-244; owner 2026-11-16: "I approve the KL
    r = 1 extension; prepare the job"): KL at r = 1, everything else as stage0_pass2 (contrast form, K* = 13, both
    approaches, groups 1–5 × 10 seeds, fraction 0.03, float32, hybrid start, warm-up 300, 8,000-epoch ceiling). Stage name
    stage0_pass2, so the λ rule reads it with the other r. 100 fits."""
    b = json.load(open(basis))
    out = []
    for a in (1, 2):
        e = b[f"{a}_1"]
        lam = 1.0 * e["data_loss"] / e["Rc"]
        out += [job("stage0_pass2", a, g, 13, 1, s, FRACTION[1], dtype="float32",
                    terms={"lambda_child": lam, "w_form": "contrast"}, arm={"r": 1.0}, max_epochs=8000)
                for g in range(5) for s in range(1, 11)]
    return out


def stage0_pass2_se_r01(basis="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/pass2_lambda_basis_v2.json"):
    """Squared error at r = 0.1 (owner 2026-11-16: "I approve your proposal b for the values of r"): squared error chose
    its highest grid value r = 0.03 in pass 2 (RL-244); r = 0.1 is fitted, everything else as stage0_pass2 (contrast form,
    K* = 20, both approaches, groups 1–5 × 10 seeds, fraction 0.03, float32, hybrid start, warm-up 300, 8,000-epoch
    ceiling). Stage name stage0_pass2, so the λ rule reads it with the other r. 100 fits."""
    b = json.load(open(basis))
    out = []
    for a in (1, 2):
        e = b[f"{a}_2"]
        lam = 0.1 * e["data_loss"] / e["Rc"]
        out += [job("stage0_pass2", a, g, 20, 2, s, FRACTION[2], dtype="float32",
                    terms={"lambda_child": lam, "w_form": "contrast"}, arm={"r": 0.1}, max_epochs=8000)
                for g in range(5) for s in range(1, 11)]
    return out


def w_form_test(basis="/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/pass2_lambda_basis_v2.json"):
    """Test of the two remedies for the flattening of tied child hyperedges (RL-234; owner 2026-11-15: "test both options
    a and b independently and separately"), each against the pass-1 λ = 0 fits with the same seed and group:
    option A — decided form R, λ = r × data loss ÷ R;
    option B — contrast form R_c = R ÷ S (terms.WordSimilarityContrast), λ = r × data loss ÷ R_c;
    the same grid r ∈ {0.001, 0.003, 0.01, 0.03, 0.1} for both (r = the term's size relative to the data loss at the λ = 0
    solution, so equal r means equal starting strength; 0.1 already flattened tied children under A in 340 epochs).
    Approach 1, each loss at its K* (squared error 20, KL 13), hold-out group 1 (index 0), seeds 1–3, fraction 0.03,
    float32, hybrid start, warm-up 300, 8,000-epoch ceiling. 60 fits. A test, not the pass-2 calibration."""
    b = json.load(open(basis))
    kstar = {2: 20, 1: 13}
    out = []
    for beta in (2, 1):
        e = b[f"1_{beta}"]
        grid = (0.001, 0.003, 0.01, 0.03, 0.1)
        arms = [("A", r, r * e["data_loss"] / e["R"], {}) for r in grid]
        arms += [("B", r, r * e["data_loss"] / e["Rc"], {"w_form": "contrast"}) for r in grid]
        for opt, r, lam, extra in arms:
            out += [job("w_form_test", 1, 0, kstar[beta], beta, s, FRACTION[beta], dtype="float32",
                        terms=dict({"lambda_child": lam}, **extra), arm={"option": opt, "r": r}, max_epochs=8000)
                    for s in (1, 2, 3)]
    return out


def binary_test():
    """Binary test (owner 2026-11-16: "make the binary test first"; corrected reading 2026-11-16): fits on data where every
    visible tie value is 1 (data.fit_data binary=True), all terms off, compared with the pass-1 weighted λ = 0 fits of the
    same seed: held-out existence (does fitting the values cost existence prediction, and how much) and held-out value
    (can a fit that never saw the values rank them as well as the weighted fit). Approach 1, both losses at K*
    (squared error 20, KL 13), hold-out group 1 (index 0), seeds 1–3, fraction 0.03, float32, hybrid start, 8,000-epoch
    ceiling. 6 fits."""
    kstar = {2: 20, 1: 13}
    return [job("binary_test", 1, 0, kstar[beta], beta, s, FRACTION[beta], dtype="float32", arm={"binary": True},
                max_epochs=8000) for beta in (2, 1) for s in (1, 2, 3)]


def swap_bare():
    """Bare graded cross-domain swap (decided 2026-11-08/09; run alongside pass 2, owner 2026-11-16): a fraction p of
    articles has its identities permuted, within social-degree bands, in the social relations only (worlds.graded_swap;
    swap seed = fit seed, so the articles chosen at a smaller p are a subset of those at a larger p). p ∈ {0.25, 0.5,
    0.75, 1}; p = 0 = the pass-1 fits of the same group and seeds. Terms off, each loss at K* (squared error 20, KL 13),
    both approaches, hold-out group 1 (index 0), seeds 1–10, fraction 0.03, float32, hybrid start, 8,000-epoch ceiling.
    160 fits. Read by fit/swap_analysis.py."""
    kstar = {2: 20, 1: 13}
    return [job("swap_bare", a, 0, kstar[beta], beta, s, FRACTION[beta], dtype="float32", arm={"swap_p": p},
                max_epochs=8000)
            for a in (1, 2) for beta in (2, 1) for p in (0.25, 0.5, 0.75, 1.0) for s in range(1, 11)]


def swap_followup():
    """Swap follow-ups at p = 1 (decided 2026-11-08 as conditional on "no change"; owner 2026-11-17: "I actually do not
    object testing in parallel different options"). (i) one band for all articles (no degree bands), each loss at K*;
    (ii) degree bands as before, at the next coarse K above K* (squared error 32, KL 20; their p = 0 references are the
    pass-1 fits of hold-out group 1, seeds 1–10). Both approaches, hold-out group 1, seeds 1–10, terms off, as swap_bare.
    80 fits. Stage "swap_bare", so swap_analysis reads them beside the bare swap."""
    kstar = {2: 20, 1: 13}; kup = {2: 32, 1: 20}
    out = [job("swap_bare", a, 0, kstar[beta], beta, s, FRACTION[beta], dtype="float32",
               arm={"swap_p": 1.0, "swap_single_band": True}, max_epochs=8000)
           for a in (1, 2) for beta in (2, 1) for s in range(1, 11)]
    out += [job("swap_bare", a, 0, kup[beta], beta, s, FRACTION[beta], dtype="float32", arm={"swap_p": 1.0}, max_epochs=8000)
            for a in (1, 2) for beta in (2, 1) for s in range(1, 11)]
    return out


def start_test():
    """Diagonal-start test (owner 2026-11-17: "Please prepare the code and document the expected outcomes"): the hybrid
    start of pass 1 with every social core matrix started diagonal (init.diagonal_social_links; same U and anchor cores
    as the pass-1 fit of the same seed). Each loss at K* (squared error 20, KL 13), both approaches, hold-out group 1
    (index 0), seeds 1–10, terms off, fraction per loss, float32, 8,000-epoch ceiling. 40 fits; reference = the pass-1
    fits of the same group and seeds. Read by fit/start_test_analysis.py (expected outcomes: plans/ticket107_start_test.md)."""
    kstar = {2: 20, 1: 13}
    return [dict(job("start_test", a, 0, kstar[beta], beta, s, FRACTION[beta], dtype="float32", max_epochs=8000),
                 social_link_start="diagonal")
            for a in (1, 2) for beta in (2, 1) for s in range(1, 11)]


def damping_test():
    """Equivalence test of the social log-damping arm (owner 2026-11-18: "shall we check first that rescaling and w_r
    work same way?" and "shall we remeasure the positivity floor and adam ε ... so our work won't rely on bare
    expectations?"). KL, K* = 13, both approaches, hold-out group 1 (index 0), seeds 1–3, terms off, fraction 0.03,
    float32, 8,000-epoch ceiling, diagonal social start (as the start test, decided for KL). Three arms per seed:
      weight        — damped values at Frobenius norm 1, relation weight w_r × q (implementation B);
      data_matched  — damped values × q (the arm, implementation A), start = the weight arm's start with every damped
                      core matrix × q (same point in the fit's terms);
      data          — implementation A from the ordinary start (as the arm would run in stage 1).
    18 fits; the undamped references are the start-test fits of the same seeds. Read by fit/damping_test_analysis.py."""
    arms = {"weight": {"log_damping": "weight"}, "data_matched": {"log_damping": "data", "damping_start": "matched"},
            "data": {"log_damping": "data"}}
    return [dict(job("damping_test", a, 0, 13, 1, s, FRACTION[1], dtype="float32", arm=arm, max_epochs=8000),
                 social_link_start="diagonal")
            for a in (1, 2) for s in (1, 2, 3) for arm in arms.values()]


def start_test2():
    """Start test 2 (owner 2026-11-19: links between different communities allowed (option ii); "I agree with S1–S3;
    test 3 levels, 5% 10 and 20%"). KL, K = 13, hold-out group 1 (index 0), seeds 1–10, terms off, fraction 0.03,
    float32, 8,000-epoch ceiling; hybrid start + diagonal social cores (as the start test), then:
      S1 — article×article keeps the hybrid start's random core (approach 2 only);
      S2 — every social core's off-diagonal cells at 5 / 10 / 20 % of the diagonal (init.offdiagonal_start);
      S3 — as S2, the semantic (anchor) cores too.
    130 fits; references: start-test fits (diagonal) and pass-1 fits (random social start), same seeds. Read by
    fit/start_test2_analysis.py; plan plans/ticket107_start_test2.md."""
    base = lambda a, s, **kw: dict(job("start_test2", a, 0, 13, 1, s, FRACTION[1], dtype="float32", max_epochs=8000),
                                   social_link_start="diagonal", **kw)
    out = [base(2, s, random_links=["article_article"]) for s in range(1, 11)]
    out += [base(a, s, offdiag_level=lv, offdiag_semantic=sem)
            for sem in (False, True) for lv in (0.05, 0.1, 0.2) for a in (1, 2) for s in range(1, 11)]
    return out


BUILD_B = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3_nogenre/"   # study B: no genre hyperedges (RL-293)


def B_stage0_pass1():
    """Study B (build without genre hyperedges; owner 2026-11-19: "I approve repeating the KL runs", RL-291), pass 1:
    the KL jobs of study A's pass 1 and its K = 32 extension with the study-B build — both approaches × K 3, 5, 8, 13,
    20, 32 × hold-out groups 1–5 × 10 seeds = 600 fits; fraction 0.03, float32, hybrid start, terms off; 5,000-epoch
    ceiling (8,000 at K = 32, as in the extension). Stage name B_stage0_pass1 (separate directory)."""
    return [dict(job("B_stage0_pass1", a, g, K, 1, s, FRACTION[1], dtype="float32", max_epochs=8000 if K == 32 else 5000),
                 build=BUILD_B)
            for a in (1, 2) for K in COARSE_K + (32,) for g in range(5) for s in range(1, 11)]


PASS3_K = (5, 8, 13, 20)                        # owner 2026-11-19 (RL-296); K = 32 only if K = 20 comes out best
PASS3_R = (0.0, 0.3, 1.0)                       # owner 2026-11-19 (RL-298)
PASS3_BASIS = {"A": "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/pass3_lambda_basis.json",
               "B": "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/results/stage0/pass3_lambda_basis_B.json"}


def stage0_pass3(study="A", Ks=PASS3_K):
    """Stage 0 pass 3 (owner 2026-11-19): K and λ_child re-chosen under the decided start, one protocol for both studies
    (A: genre hyperedges in; B: without, RL-290/295). K ∈ {5, 8, 13, 20} (RL-296) × r ∈ {0, 0.3, 1} (RL-298), λ_child =
    r × basis of that K and approach (data loss ÷ (R ÷ S) on the pass-1 λ = 0 solution, hold-out group 1, mean of 10
    seeds; pass3_lambda_basis.py, RL-297), contrast form; start S3 at 10 % (hybrid NNDSVD start, social cores diagonal,
    then social and semantic cores with off-diagonal cells at 10 % of √(d_k d_l); RL-280); KL, both approaches, hold-out
    groups 1–5 × seeds 1–10, fraction 0.03, float32, warm-up 300, 8,000-epoch ceiling (sizing approved 2026-11-19).
    Choice by J1 (RL-299): fit/pass3_choice.py. 1,200 fits per study. `lambda_from_r` marks jobs whose λ_child follows
    from (r, K), so the aggregation groups them by r (aggregate.cell_key). Ks: a top-up passes its own K list (C3,
    grid-end rule)."""
    basis = json.load(open(PASS3_BASIS[study]))
    stage = "stage0_pass3" if study == "A" else "B_stage0_pass3"
    out = []
    for a in (1, 2):
        for K in Ks:
            e = basis[f"pass1|{a}|{K}"]
            for r in PASS3_R:
                terms = {"lambda_child": r * e["basis"], "w_form": "contrast"} if r > 0 else {}
                for g in range(5):
                    for s in range(1, 11):
                        j = dict(job(stage, a, g, K, 1, s, FRACTION[1], dtype="float32", terms=terms, arm={"r": r},
                                     max_epochs=8000),
                                 social_link_start="diagonal", offdiag_level=0.1, offdiag_semantic=True, lambda_from_r=True)
                        if study == "B":
                            j["build"] = BUILD_B
                        out.append(j)
    return out


def B_repeats():
    """Study B (no genre hyperedges): repeats of study A's KL runs of stage 0 (owner 2026-11-19: "I approve repeating the
    KL runs", RL-291) at K 13 in both approaches as in A (owner 2026-11-20: "I approve option a, K 13 for both
    approaches", RL-304). Every KL job of A's pass 2 (r 0.03 / 0.1 / 0.3, and r = 1), bare swap and follow-ups,
    diagonal-start test, damping test and start test 2, unchanged except: stage "B_<stage>", build = study B's, and
    λ_child = r × study B's basis at K 13 (data loss ÷ (R ÷ S) on B's pass-1 λ = 0 fits, hold-out group 1, mean of 10
    seeds; pass3_lambda_basis_B.json — the quantity of pass2_lambda.py). 688 fits."""
    basis = json.load(open(PASS3_BASIS["B"]))
    out = []
    for gen in (stage0_pass2, stage0_pass2_kl_r1, swap_bare, swap_followup, start_test, damping_test, start_test2):
        for j in gen():
            if j["beta"] != 1:
                continue
            j = dict(j, stage="B_" + j["stage"], build=BUILD_B)
            if (j.get("terms") or {}).get("lambda_child"):
                j["terms"] = dict(j["terms"], lambda_child=float(j["arm"]["r"]) * basis[f"pass1|{j['approach']}|{j['K']}"]["basis"])
            out.append(j)
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
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=("checkpoint3", "stage0_pass1", "rehearsal", "kl_fraction_extension", "timing_terms", "stage0_extension", "stage0_pass2", "w_form_test", "binary_test", "swap_bare", "stage0_pass2_kl_r1", "stage0_pass2_se_r01", "swap_followup", "start_test", "damping_test", "start_test2", "B_stage0_pass1", "stage0_pass3", "B_stage0_pass3", "stage0_pass3_K3", "B_repeats", "B_stage0_pass3_full"))
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    jobs = {"checkpoint3": checkpoint3, "rehearsal": rehearsal, "kl_fraction_extension": kl_fraction_extension, "timing_terms": timing_terms, "stage0_extension": stage0_extension, "stage0_pass2": stage0_pass2, "w_form_test": w_form_test, "binary_test": binary_test, "swap_bare": swap_bare, "stage0_pass2_kl_r1": stage0_pass2_kl_r1, "stage0_pass2_se_r01": stage0_pass2_se_r01, "swap_followup": swap_followup, "start_test": start_test, "damping_test": damping_test, "start_test2": start_test2, "B_stage0_pass1": B_stage0_pass1, "stage0_pass3": lambda: stage0_pass3("A"), "B_stage0_pass3": lambda: stage0_pass3("B"), "stage0_pass3_K3": lambda: stage0_pass3("A", Ks=(3,)), "B_repeats": B_repeats, "B_stage0_pass3_full": lambda: stage0_pass3("B", Ks=(3,) + PASS3_K)}.get(a.which, lambda: stage0_pass1(a.seeds))()
    for j in jobs:
        print(json.dumps(j, sort_keys=True))
    print(f"{len(jobs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
