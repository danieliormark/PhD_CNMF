# Ticket 107 — equivalence test of the social log-damping arm (prepared 2026-11-18)

## 1. Why
The social log-damping arm (decided 2026-11-18) multiplies every tie of an author (author×article), organisation
(affiliation relations) or journal (article×journal) by log(1 + d) / d, d = the entity's visible ties in the fold.
Under the per-relation Frobenius scaling this lowers the relation's total mass and so, under KL, its weight in the loss
(RL-272). Fix (GPT review, RL-273): restore each damped relation's visible mass in the data (implementation A).
Owner 2026-11-18: "shall we check first that rescaling and w_r work same way?" — i.e. whether A behaves like the
alternative B (damped data at Frobenius norm 1, relation weight w_r × q, q = undamped mass ÷ damped mass) — and
"shall we remeasure the positivity floor and adam ε as well as other parameters, so our work won't rely on bare
expectations?". Equal at the optimum in theory; not exactly in M1 because of settings that are fixed numbers, not
proportional to the data: positivity floor 1e-7, Adam's ε 1e-8, step-size floor 1e-3 × the block's start size, and
the social core matrices' start (not scaled to the data). Float32 rounding adds to these.

## 2. Design (fit/plan.py damping_test; 18 fits)
KL, K = 13, approaches 1 and 2, hold-out group 1, seeds 1–3, diagonal social start, terms off, fraction 0.03, float32.
| Arm | Data | Relation weight | Start |
|---|---|---|---|
| weight (B) | damped, Frobenius norm 1 | w_r × q | ordinary |
| data_matched (A) | damped × q (mass restored) | w_r | B's start, damped core matrices × q |
| data (A) | damped × q | w_r | ordinary (as stage 1 would run it) |
q (group 1): author×article 1.007, author×affiliation 1.178, article×affiliation 1.083, article×journal 1.404.
Undamped references: start-test KL fits, same seeds.

## 3. Readings (fit/damping_test_analysis.py)
(a) data_matched vs weight — the implementations; (b) data vs data_matched — the start's scale; (c) yardstick: data
arm seed vs seed; (d) preview, reporting only: data arm vs the undamped fit of the same seed (diagonal rule,
child×article failing communities, held-out existence).
Per pair: final total loss (relative difference), epochs, first epoch the two total-loss paths differ by > 1e-4;
per relation loss A vs q × loss B; reconstruction on visible ties x̂_A vs q × x̂_B (relative L2, numbering-free);
diagonal-rule share per social relation; child×article failing communities; weighted memberships (overlap Σ_k min,
mean over entities; B also read with its damped cores × q). Per fit: share of core cells and factor entries at the
floor; gradient RMS per block and share of entries with |gradient| < 10 ε; smallest block size ÷ its start size.

## 4. Expected outcomes and reading
ratio = median difference in (a) ÷ median seed-to-seed difference in (c), for total loss and reconstruction.
1. ratio < 0.1 → equivalent in practice: the equivalence is measured, not assumed; the wording (approved
   provisionally) stands; implementation A goes into stage 1.
2. 0.1–0.5 → drift: check that rules and memberships agree; if they do, proceed and record the drift.
3. > 0.5 → not equivalent: find the setting responsible (most likely the floor, see §5) before stage 1.
(b) is read the same way: a large start-scale effect would mean the arm's starts must be scaled by q.
Memberships as fitted differ between A and B by design (RL-273); the reading on A's scale is the like-for-like one.

## 5. Checks before submission (2026-11-18, no fits)
- Data step (data.damp_values): masses as RL-272 (article×journal 46.2 undamped; B 32.9); A = q × B to 2e-16, hidden
  ties included; unit test fit/tests/test_damping.py passes.
- Smoke run in memory (approach 1, seed 1, 30 epochs): matched A and B total loss equal to float32 precision
  (epoch 29: 6077.627 vs 6077.622); per relation loss A = q × B to ≤ 1e-5 relative; factor matrices differ by ≤ 6e-6.
- Dry run of the reading script on start-test fits: under the diagonal start 92 % of every core matrix's cells sit at
  the floor (all off-diagonal cells) — the floor is the setting most likely to break exact equivalence; gradient RMS
  per block 0.3–229, entries with |gradient| < 10 ε ≤ 0.05 % (ε not expected to matter); smallest block size 0.28 × its
  start (step-size floor at 1e-3 not reached). Seed-to-seed reconstruction difference on visible ties 0.40–0.67.
- Entity totals (fit/damping_entity_totals.py, results/stage0/damping_entity_totals.{json,log}): mean entity total
  still rises with d in every relation (order kept); mass share of the largest band: journals with ≥ 32 articles
  0.47 → 0.12, organisations with ≥ 32 ties 0.17 → 0.05 (author×affiliation), with ≥ 16 ties 0.11 → 0.04
  (article×affiliation); author×article changes little (83 % of authors have one article; share of authors with ≥ 4
  articles 0.025 → 0.012), team-size bands unchanged; productive authors on large teams: no band near zero relative
  to its size.

## 6. Results (2026-11-19, RL-279)
18/18 converged. (a) A vs B: total loss differs by 5e-6 relative (seed spread 1e-3; ratio ≤ 0.006); loss paths never
differ by > 1e-4; reconstruction ratios ≤ 0.026; rules identical; memberships overlap ≥ 0.998 → outcome 1, equivalent
in practice. Floor binding on all off-diagonal cells in both arms alike; ε and the step-size floor do not act.
(b) Start scale: ratio 0.26–0.42 of seed spread → the arm runs from the same (ordinary) start as its reference.
(d) Preview uninformative under diagonal cores (rules met by construction; existence changes ≤ 0.002).
