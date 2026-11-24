# M1 progress tracker — Psychology trial (ticket 107)

What has been run, what is running, and what is still to run, with the purpose of each run. Updated whenever a job is
submitted, finishes, or a decision changes what will run. Details live in the records named in the last column:
`fullscale_pipeline/RUN_LOG.md` (RL-…, what was run and what came out) and `plans/ticket107-m1-chunk12v3-design.md`
(design doc, what was decided and why). The staged plan with fit budgets: `plans/ticket107_layer2/test_architecture.txt`.

**Last updated:** 2026-11-23.

**Status words:** *Done* = finished and read; *Running* = on CSF3 now; *Submitted* = in the CSF3 queue; *Ready* =
code and job written, waiting for submission; *Waiting for decision* = needs the owner's choice before it can be
written or read; *Not started* = later stage.

**Terms used below.** *Fit* = one factorisation run (one setting, one hold-out group, one seed). *Hold-out group* = one
of 5 draws of hidden ties used to score prediction. *K* = number of communities; *K\** = the best K by held-out
existence in pass 1 (squared error 20, KL 13). *Existence* = held-out within-class AP (are hidden ties ranked above
hidden empty cells); *value* = held-out partial Spearman (are hidden ties ranked by their real strength). *r* = strength
of the word-similarity term: λ_child is set so that, evaluated on the fitted λ = 0 solution (pass 1, K\*), the term equals r × the data loss (the weighted reconstruction error on the cells used for fitting). With the term on, the fit makes tied child hyperedges more alike, so the term ends far smaller (about 0.1% of the data loss for squared error, 2–3% for KL). *Approach 1 / 2* = the two social
designs (with an author facet / articles linked to affiliations and to each other by citations).

## Now

| Run | Status | CSF3 job | Fits | Progress |
|---|---|---|---|---|
| Bare graded cross-domain swap | Done, presented 2026-11-17 | 24804741 | 160 | 160/160 (RL-248) |
| Pass 2, KL r = 1 (highest-grid-value rule) | Done | 24807473 | 100 | 100/100; KL r = 1 chosen (RL-247) |
| Pass 2, squared error r = 0.1 | Done | 24825747 | 100 | r = 0.1 outside the existence band; squared error stays at r = 0.03 (RL-250) |

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
| Bare graded cross-domain swap | Done | 24804741 | 160 | Is heterogeneity found because the data have it, or produced regardless? Social identities of a fraction p = 0.25/0.5/0.75/1 of articles are swapped (p = 0 = pass 1), each loss at K\*, both approaches, group 1, 10 seeds. Feeds the decision on mono-domain communities | `fit/swap_analysis.py` (decided reading rule) | RL-243 |
| Pass 2, KL r = 1 | Done | 24807473 | 100 | Highest-grid-value rule: KL chose its top value r = 0.3; check whether a stronger term still qualifies | `fit/pass2_lambda_rule.py` after the chained aggregation | RL-245 |
| Pass 2, squared error r = 0.1 | Done (RL-250: r = 0.03 stays) | 24825747 | 100 | Squared error also chose its top value r = 0.03 (no rule was decided for it); owner approved fitting r = 0.1. If it qualifies the rule chooses it (flattening reported only) | `fit/pass2_lambda_rule.py` after the chained aggregation | RL-246 |

### 2.4 Still to run or decide

| Item | Status | Fits | Purpose | Blocks |
|---|---|---|---|---|
| Intermediate r values (e.g. 0.055 between 0.03 and 0.1) | Decided: not fitted | — | Owner (2026-11-17): none in stage 0; finer steps around the stage-0 choice in λ calibration | — |
| What if the highest grid value is chosen again | Decided 2026-11-17 | — | Stop at the highest fitted r in stage 0 (KL r = 1); λ calibration establishes the upper limit of λ_child (grid above the top value, finer steps) | — |
| Reading of "stability not worse than at λ = 0" | Decided 2026-11-17 | — | One SE of λ = 0 (reference SE), gate must not start failing; paired differences reported | — |
| Study B pass 1 (KL, K 3–32) | Done 2026-11-20 (RL-303): 600/600 converged; candidates approach 1 {5, 8, 13, 20} (best 20 by 0.0001), approach 2 {5, 8, 13} (best 13); existence flat over K 5–32; stability below A; B basis computed | 600 | K choice in study B (repeat of study A's KL pass 1 + K 32) | Study B pass 2 |
| Pass 3, study B | Done 2026-11-21 (RL-317): 1,500/1,500 converged. Option (b) candidates: approach 1 K {5, 8, 20} (r* 0.3 / 0.3 / 0), approach 2 K {5, 20} (r* 0.3 / 0); T1 holds everywhere; K 20 best by existence in approach 1 → RL-296's K 32 condition met, to the owner | 1,500 | Same protocol as A (C3) | A–B comparison |
| λ calibration, study A | **Finished complete 2026-11-23: 1,500 / 1,500, 0 errors; post-processing under the coverage rule (RL-349); not yet reported.** Earlier: finished incomplete 2026-11-22: 1,452 / 1,500 fits (array tasks 28, 32, 36, 39, 41, 42 never ran: no logs; missing approach 1 K 8 r 0.45 ×24, K 5 r 0.8 ×16, K 5 r 0.6 ×8); post-processing 25112303 ran on the incomplete set — resubmitted 2026-11-23: fit array 25132000, post 25132001 (`lambda_cal.sh`, 1,500 fits: r 0.45 / 0.6 / 0.8 or 1.5 / 2 / 3 per K, all coarse K, both approaches; ≈ 1,100–1,300 CPU-h; read with pass 3 by R2; RL-330/333) | 1,500 | λ_child per K for stage 1 | Swap with the term on (W-a, 160 fits); optional text check |
| W3 check: KL with mass weights (50 / 50 under KL) vs current scaling | **Finished complete 2026-11-23: 240 / 240, 0 errors (RL-349); read 2026-11-23 (RL-352: scoring-weight confound corrected; W1 / W2 with the owner).** Earlier: finished incomplete 2026-11-23 (fit array 25115915, post 25115916): 238 / 240 (task 27: Slurm node errors "Unable to resolve infra01"; 2 approach-2 swap fits unfinished) — resubmitted 2026-11-23: fit array 25132002, post 25132003 (`weight_check.sh`, 240 fits, ≈ 150–200 CPU-h; RL-339/340) | 240 | Decide W1 (keep) or W2 (switch) | Possibly pass 3 / λ calibration |
| Intermediate weighting check: semantic share of the KL pull 0.65 / 0.75 (between W2 0.50 and W1 0.83 / 0.90) | Done 2026-11-24: 280 / 280 converged (RL-360); read (RL-361): by R-b approach 2 → 0.65; approach 1 tied (0.65 / 0.75 / W1) — owner to decide | 280 | Map the trade-off between the domains; choose the weighting by a rule fixed before the results | λ calibration (redone if the weighting moves); stage 1 |
| Community-quality readings across weighting points (saved fits) | Ready 2026-11-24 (owner: "Yes, please prepare the job"; RL-362): `weight_curve_community.sh`, 1 job × 8 cores, 400 fits, ≈ 15–30 min; waiting for submission | — | Does the child×article prediction loss at 0.65 make communities more biased (idea vs people coherence, crossing ties, article coherence) | Weighting decisions (approaches 1 and 2) |
| Item (l) remeasurement under the coverage rule (saved fits, no new fits) | Prepared 2026-11-23 (owner: "I propose you start with the task of remeasuring the readings that used the placed rule, now under the coverage rule"; RL-348): `remeasure_coverage.sh`, 5 array tasks × 8 cores — pass-3 aggregation and J1 choice in A and B (stability gate, ghost / alignment vetoes, Track A, λ-rule guard), start tests 1–2 (A, B), stability tests 1–3, S3 gate 1; old outputs kept as `*_placed.*`. Array 25134381 finished 2026-11-23: tasks 0–3 done; task 4 (S3 gate 1) failed on a script bug (fixed) (RL-349), rerun `--array=4` done 2026-11-23: every gate verdict unchanged (RL-351). Results RL-350/351 | — | Do the gate, vetoes, λ rule and candidates change? (no) | Reported (report 1) |
| Item (l) option (a): membership readings (saved fits, no new fits) | Prepared 2026-11-23 (owner: "I approve option a, prepare the readings to avoid this self-circularity problem you raised"; RL-351): `fit/membership_readings.py`, `membership_readings.sh` (1 job × 8 cores, 75 tasks: 15 pass-3 candidate cells × 5 groups; test task 3 min, 1.1 GB). Done 2026-11-23: job 25190974, 75 / 75 tasks, 10 min (RL-354); read (RL-355) | — | How stable and how well placed is each entity's membership; how much of the placed rule's Track A gap was selection | Report; inspection / text-check flags |
| Membership readings r sweep (study A, all r × K, saved fits) | Done 2026-11-24: job 25206589, 300 / 300 tasks (RL-357/360); not yet read | — | Is the child hyperedges' instability caused by r (word-similarity term) or by K; tie-count effect vs membership spread | λ-calibration report (r per K) |
| Study B repeats at K 13 (pass 2, swap + follow-ups, diagonal-start, damping, start test 2) | Done 2026-11-20 (RL-314): all reproduce A's choices — λ rule r = 1; diagonal start better; start test 2 → S3 10 %; damping implementations equivalent; swap no change (one follow-up crosses its margin: approach 2, K 20) | 688 | Every KL run of A's stage 0 repeated on B (RL-291/304) | Study B pass 3 |
| Study B build (without genre hyperedges) | Built 2026-11-19 (`psych_trial_v3_nogenre`, rules decided by owner, RL-293); hold-out: fresh draw, seed 107 (owner, RL-294) — owner 2026-11-19: with and without genre are two full studies, every run of one also run on the other (RL-290) | — | Build rules for parents, W, emptied units. Then the KL runs of stage 0 repeated on B (owner 2026-11-19, RL-291): pass 1 KL + K 32, pass 2 KL + r 1, swap + follow-ups, diagonal-start (KL), damping test, start test 2 — ≈ 1,430 fits | Pass 3 in study B |
| Pass 3, study A | Done 2026-11-20 (RL-307/309/313): 1,500 fits (K 3–20 × r 0 / 0.3 / 1), all converged. Option (b) candidates: approach 1 K {5, 8, 13, 20} (r* 0.3 / 0.3 / 1 / 1), approach 2 K {5, 8, 13} (r* 0.3 / 1 / 1); K 3 outside the band (paired −0.005 ± 0.002) | 1,500 |
| Off-diagonal share (pass 1, no fits) | To analyse | — | Feeds the choice between the off-diagonal options O1/O3 (decided after stage 0) | λ calibration |
| Mono-domain: seed recurrence, mass, cause split (no fits) | Done 2026-11-17 (RL-251) | — | None recurs in the majority of seeds → no mono-domain community by the decided rule; presented with the swap | — |
| Follow-up checks for owner questions: members tie counts, value chance check, semantic held-out across swap (no fits) | Done 2026-11-17 (RL-252) | — | Weak-domain members have ordinary tie counts (corrects earlier inference); semantic held-out flat p = 0 → 1 in all four settings | — |
| Facet detachment in pass 1; link-matrix structure and start (no fits) | Done 2026-11-17 (RL-253) | — | Parents without children never occur (child×parent link diagonal); author/organisation, article/journal, article/organisation detachment common, rises with K, never recurs across seeds; semantic link structure set by the start (diagonal under NNDSVD starts, off-diagonal under random) | — |
| Link structure at K*: labelling vs spread; missing affiliations/journals (no fits) | Done 2026-11-17 (RL-255) | — | KL off-diagonal ≈ labelling of single-relation facets (≈ 1 partner after relabelling); squared error genuine spread (3–5 partners); missing data do not explain detachment | — |
| Swap follow-ups (i) one band, (ii) larger K, at p = 1 | Done 2026-11-18 (RL-268): no change in every comparison (aligned); neither the degree bands nor fixed K explains the bare swap's "no change" | 24848752 | 80 fits: `fit/slurm/swap_followup.sh` (owner: "I actually do not object testing in parallel different options") | — |
| Diagonal-start test (social link matrices started diagonal) | Done 2026-11-18 (RL-269): KL better on every reading with the diagonal start; squared error 3/10 and 0/10 converged at 8,000 epochs (decided reading not possible; supplementary: lower data loss, one partner, but held-out existence lower, more child×article failures, communities separate by domain). Owner 2026-11-18: the 17 unconverged squared-error fits are not continued | 24849355 | 40 fits; `plans/ticket107_start_test.md` | — |
| Log-damping equivalence test (data rescaling vs relation weight; fixed-size settings measured) | Done 2026-11-19 (RL-279): equivalent in practice (A vs B ≤ 0.6 % of seed spread); start scale 26–42 % of seed spread → arm uses the ordinary start | 24899341 | 18 fits; `plans/ticket107_damping_test.md`; submit `bash .../fit/slurm/damping_test.sh` | — |
| Start test 2 (room for links between different communities: S1 / S2 / S3 × 5, 10, 20 %) | Done 2026-11-19 (RL-280): off-diagonal links return to the floor from structured starts; S1 fits worse; owner chose S3 10 % as the start for pass 3 and later; diagonal cores accepted (design doc) | (id not given) | 130 fits; `plans/ticket107_start_test2.md`; submit `bash .../fit/slurm/start_test2.sh` | — |
| Alignment of community numbering (read time, no fits) | Implemented 2026-11-17 (RL-260); swap / mono-domain / detachment re-read (RL-261, RL-262); stage-0 selection re-read done (RL-263) | — | Swap verdict flips: squared error approach 1 → heterogeneity follows the data; KL approach 2 → no change (near miss); no mono-domain community still. K* unchanged (20 / 13); under squared error the social facets' aligned numbering agrees across seeds only at chance level, while each facet's own grouping is reproducible | — |
| Graded swap at a further K | Only if the swap curve is ambiguous | 160 per K | Decided 2026-11-09 | — |

### 2.5 Decisions taken after stage 0 (owner)

Loss (β); O1/O3; response to mono-domain communities; item (l) (weakly engaged entities); stage-1 counting fraction;
empty-column reference.

## 3. Later stages — not started

| Stage | Fits (staged plan, S = 10 seeds) | Purpose |
|---|---|---|
| λ calibration | ≈ 1,580 (+ conditional) | Strength of each kept term per K under the chosen loss; graded swap per kept term |
| Stage 1 | ≈ 6,000 (+ social log-damping arm, decided 2026-11-18) | Journal weighting (J arms) per approach; social log-damping robustness arm (journals, organisations, authors' article counts; not crossed with J arms; implementation A = mass restored in the data, ordinary start as its undamped reference, under the start chosen by start test 2 — decided 2026-11-19, design doc); binding gate, vetoes and convergence |
| Stage 2 | ≈ 1,000 | K protocol with vetoes and fallback; re-checks |
| Stage 3 | ≈ 320 (+ optional swap 80) | Approach comparisons, final refit on all data |
