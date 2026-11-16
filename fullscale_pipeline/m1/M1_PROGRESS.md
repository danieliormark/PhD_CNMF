# M1 progress tracker — Psychology trial (ticket 107)

What has been run, what is running, and what is still to run, with the purpose of each run. Updated whenever a job is
submitted, finishes, or a decision changes what will run. Details live in the records named in the last column:
`fullscale_pipeline/RUN_LOG.md` (RL-…, what was run and what came out) and `plans/ticket107-m1-chunk12v3-design.md`
(design doc, what was decided and why). The staged plan with fit budgets: `plans/ticket107_layer2/test_architecture.txt`.

**Last updated:** 2026-11-16, late evening.

**Status words:** *Done* = finished and read; *Running* = on CSF3 now; *Submitted* = in the CSF3 queue; *Ready* =
code and job written, waiting for submission; *Waiting for decision* = needs the owner's choice before it can be
written or read; *Not started* = later stage.

**Terms used below.** *Fit* = one factorisation run (one setting, one hold-out group, one seed). *Hold-out group* = one
of 5 draws of hidden ties used to score prediction. *K* = number of communities; *K\** = the best K by held-out
existence in pass 1 (squared error 20, KL 13). *Existence* = held-out within-class AP (are hidden ties ranked above
hidden empty cells); *value* = held-out partial Spearman (are hidden ties ranked by their real strength). *r* = strength
of the word-similarity term (the term's size relative to the data loss at the start). *Approach 1 / 2* = the two social
designs (with an author facet / articles linked to affiliations and to each other by citations).

## Now

| Run | Status | CSF3 job | Fits | Progress |
|---|---|---|---|---|
| Bare graded cross-domain swap | Running | 24804741 | 160 | 14 of 160 finished (2026-11-16, 20:00) |
| Pass 2, KL r = 1 (highest-grid-value rule) | Submitted | (ID once tasks start) | 100 | queued |
| Pass 2, squared error r = 0.1 | Ready (owner submits) | — | 100 | `bash fit/slurm/stage0_pass2_se_r01.sh` |

## 1. Build step (before stage 0) — done

| Run | Status | Fits | Purpose | Outcome | Record |
|---|---|---|---|---|---|
| Matrix build (chunk12v3) and hold-out draw | Done | — | Build the trial's relation matrices and the 5 hold-out groups | psych_trial_v3 | RL-206, RL-207 |
| Code-check world, rounds 1–4 | Done | 118 (incline) | Check that the fit code and optimiser recover planted communities (a code check, never evidence about the method) | Optimiser settings fixed (hybrid start, learning rate recomputed every step) | RL-208–212 |
| CSF3 rehearsal | Done | 17 | Test the CSF3 job machinery (outputs, failure handling, aggregation) | Works | RL-213 |
| Checkpoint 3 | Done | 72 | Learning-rate fraction, start method, float32 precision, time per fit | Hybrid start, float32 | RL-214–215 |
| KL fraction extension | Done | 18 | Same fraction check for KL at larger fractions | Fraction 0.03 for both losses | RL-216–219 |
| Timing test with the in-fit terms on | Done | 28 | Cost per epoch of each term, to size stage 0 | Terms add ≤ 17% | RL-224 |

## 2. Stage 0 — loss comparison (squared error vs KL)

### 2.1 Runs done

| Run | Status | CSF3 job | Fits | Purpose | Outcome | Record |
|---|---|---|---|---|---|---|
| Pass 1 | Done | 24693492 | 1,000 | Both losses × both approaches × K 3–20 × 5 groups × 10 seeds, all terms off: choose K per loss and give the bare loss comparison | Best K: squared error 20, KL 13 (owner confirmed) | RL-225–226 |
| Pass 1 extension | Done | 24719869 | 213 | K = 32 (every loss's band reached 20) and reruns of 13 unconverged squared-error fits | K = 32 never best; K\* unchanged | RL-229–230 |
| W-term A/B test | Done | 24740567 | 60 | The word-similarity term flattened tied child hyperedges (RL-234); compare its decided form (A) with the contrast form (B) | B adopted (owner) | RL-236–237 |
| Pass 2 (first submission) | Cancelled | 24774361 | — | Submitted by accident before the floor and S3 decisions | Cancelled by owner | RL-238 |
| Binary test | Done | 24775031 | 6 | Does fitting tie values cost prediction of where ties are? | No cost under either loss; values help squared error; KL fits them without predicting hidden ones | RL-240–241 |
| Pass 2 | Done | 24789580 | 600 | Strength of the word-similarity term per loss: squared error r 0.003/0.01/0.03, KL r 0.03/0.1/0.3, at K\*, both approaches, 5 groups × 10 seeds | λ rule: squared error r = 0.03, KL r = 0.3 (both the highest grid value). The term leaves existence unchanged and improves value (KL 0.044 → 0.085). Rule re-run on all 600 fits pending (599 read) | RL-244 |

### 2.2 Analyses on saved fits done (no new fits)

| Analysis | Purpose | Outcome | Record |
|---|---|---|---|
| Bare loss comparison | Compare the losses with all terms off | Squared error by the decided rule; flags: KL dominance when cross-scored, KL places far more entities | RL-232–233 |
| S3 gate 1 | Is there excess spread that harms prediction? (decides whether the spread ceiling is tested) | Fails for child hyperedges and articles → S3 dropped (owner) | RL-239 |
| Floor diagnostics | Do shared child hyperedges fall short of their articles' mix? (decides the concentration floor) | Small shortfall → floor inactive (owner) | RL-239 |
| Warm-up from membership logs | When to switch terms on | 300 epochs kept (owner) | RL-231 |
| Mono-domain communities, per fit | How often communities miss one domain | Squared error 0.4–0.65 per fit; KL none | RL-242 |
| Flattening at each r (pass 2) | Reported beside the λ rule (owner: reported only) | Squared error r = 0.03: largest share of tied child hyperedges 0.36 vs 0.61; KL r = 0.3: 0.61 vs 0.88 | RL-244 |

### 2.3 Running or submitted

| Run | Status | CSF3 job | Fits | Purpose | Read by | Record |
|---|---|---|---|---|---|---|
| Bare graded cross-domain swap | Running | 24804741 | 160 | Is heterogeneity found because the data have it, or produced regardless? Social identities of a fraction p = 0.25/0.5/0.75/1 of articles are swapped (p = 0 = pass 1), each loss at K\*, both approaches, group 1, 10 seeds. Feeds the decision on mono-domain communities | `fit/swap_analysis.py` (decided reading rule) | RL-243 |
| Pass 2, KL r = 1 | Submitted | — | 100 | Highest-grid-value rule: KL chose its top value r = 0.3; check whether a stronger term still qualifies | `fit/pass2_lambda_rule.py` after the chained aggregation | RL-245 |
| Pass 2, squared error r = 0.1 | Ready | — | 100 | Squared error also chose its top value r = 0.03 (no rule was decided for it); owner approved fitting r = 0.1. If it qualifies the rule chooses it (flattening reported only) | `fit/pass2_lambda_rule.py` after the chained aggregation | RL-246 |

### 2.4 Still to run or decide

| Item | Status | Fits | Purpose | Blocks |
|---|---|---|---|---|
| Intermediate r values (e.g. 0.055 between 0.03 and 0.1) | Waiting for decision | ≈ 100 per value | Owner asked whether 0.03 → 0.1 is too large a step. Proposed: none in stage 0; finer steps around the stage-0 choice in λ calibration | — |
| What if r = 0.1 (squared error) or r = 1 (KL) is chosen again | Waiting for decision | — | Both would again be the highest grid value | Pass 3 |
| Reading of "stability not worse than at λ = 0" | Waiting for decision | — | No margin was decided; proposed: the decided guard convention (one SE of λ = 0) | Reading KL r = 1 |
| Pass 3 | Not started | ≈ 200–300 (sized when the candidate K are fixed) | The loss verdict: each loss at its own chosen r at the candidate K (decided rule: gate → membership agreement → held-out → stability → squared error by default) | Stage 0 decision |
| Off-diagonal share (pass 1, no fits) | To analyse | — | Feeds the choice between the off-diagonal options O1/O3 (decided after stage 0) | λ calibration |
| Mono-domain: seed recurrence, mass, cause split (no fits) | To analyse | — | Completes the decided stage-0 report on mono-domain communities | Mono-domain decision |
| Conditional swap follow-ups | Only if the swap shows "no change" | ≈ 20–40 per arm | Single degree band at p = 1; larger K (owner's permission) | — |
| Graded swap at a further K | Only if the swap curve is ambiguous | 160 per K | Decided 2026-11-09 | — |

### 2.5 Decisions taken after stage 0 (owner)

Loss (β); O1/O3; response to mono-domain communities; item (l) (weakly engaged entities); stage-1 counting fraction;
empty-column reference.

## 3. Later stages — not started

| Stage | Fits (staged plan, S = 10 seeds) | Purpose |
|---|---|---|
| λ calibration | ≈ 1,580 (+ conditional) | Strength of each kept term per K under the chosen loss; graded swap per kept term |
| Stage 1 | ≈ 6,000 | Journal weighting (J arms) per approach; binding gate, vetoes and convergence |
| Stage 2 | ≈ 1,000 | K protocol with vetoes and fallback; re-checks |
| Stage 3 | ≈ 320 (+ optional swap 80) | Approach comparisons, final refit on all data |
