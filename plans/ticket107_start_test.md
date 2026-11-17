# Diagonal-start test — design, reading rule and expected outcomes (ticket 107)

Written 2026-11-17, **before any fit of this test has run**, at the owner's request ("Please prepare the code and
document the expected outcomes so we could compare the results with them"). The reading rule below is a proposal for
the owner's confirmation; once confirmed it is applied unchanged by `fit/start_test_analysis.py`. Background: RUN_LOG
RL-253 (facet detachment, link structure), RL-255 (renumbering vs genuine spread).

## 1. The question

Under squared error, a community's ties in the social relation matrices go to about 3–5 **partner communities**
(RL-255). For example, community 4's authors are tied to organisations spread over communities 2, 7, 11 and 15. Under
KL it is about 1 partner, once single-relation facets are renumbered. The owner's decided reading R-A (2026-11-06)
is that a heterogeneous community is the same community number across facets. That needs about one partner.

**Question:** does squared error spread because of **where it starts**, or because **the loss and data favour it**?

## 2. The two arms

| | Reference (pass 1, existing) | Diagonal arm (new) |
|---|---|---|
| Memberships (U) at the start | hybrid start: subsample NNDSVD of the semantic relations, other facets propagated | **identical, bit for bit** for the same seed (tested) |
| Semantic link matrices (child×article, child×parent) at the start | diagonal (singular values) + 1e-4 | identical |
| Social link matrices at the start (author×article, author×affiliation, article×journal; approach 2: article×affiliation, article×article, article×journal) | random uniform (0, 1) + 1e-4: about 1/K of each matrix on the diagonal (measured: 0.04–0.09), so the diagonal gets no preference | **c·I + 1e-4**: diagonal, with c set so that the starting rebuilt values have the same overall size (Frobenius norm over observed rows) as the random matrix they replace (tested: equal to 1e-16) |
| Everything else | squared error K = 20, KL K = 13; both approaches; hold-out group 1; seeds 1–10; no terms; fraction 0.03; float32 | identical; ceiling 8,000 epochs (a ceiling only) |

That is 40 new fits, read against the 40 matching pass-1 fits. Entries off the diagonal start small, but they are free to
grow. Under squared error, child×article grows them from its own diagonal start (1.7–1.9 partners), so the start
does not lock anything in.

## 3. Measures (all already defined; no new measure)

- **Effective number of partner communities**, per social relation. For each row community: exp of the entropy of
  how its rebuilt tie weight divides over the column communities. These values are averaged, weighted by each
  community's tie weight. 1 = one partner; 4 = spread evenly over four. It does not depend on community numbers, so it
  is the **primary** measure. Reported beside it:
  - within share as fitted;
  - within share after one-to-one renumbering of the column facet;
  - main-partner share.
- **Fit:**
  - data loss on the fitted cells (total, weighted; and per relation);
  - held-out two-part score (existence = within-class AP; value = partial Spearman; the decided score);
  - existence per relation with hidden cells.
- **Stability:** gate passes and Track A / B, chance-adjusted, within each arm.
- **Presence:** mixed share at 0.1, weaker-domain median, and detached communities per fit (after renumbering
  single-relation facets).

## 4. Reading rule (proposed, stated before the data)

1. **Paired by seed.** Both arms share the same membership start for each seed. For every measure: difference =
   diagonal − reference, over the seeds where both converged. Mean, and SE = SD ÷ √n. **"Better" / "worse"** means the
   mean difference lies beyond **2 SE** in that direction; otherwise "within noise".
   - Why 2 SE and not 1 SE (the band convention): with 10 paired seeds, a 1-SE line would call about a third of
     no-effect comparisons a difference.
   - The trade-off: a small real cost in fit may be missed. The paired differences are reported in full either way.
2. **Partner reading per social relation.** "Stays near one partner" if the diagonal arm's median effective number of
   partners is below the midpoint between 1 and the reference median; otherwise "returns to spread".
3. **Fit cost.** "Worse" if data loss is higher, or existence or value is lower, beyond 2 SE (paired).
4. **Outcome under squared error** (per relation, then per setting): see §5. **KL is the control:** "changed" if any of
   data loss, existence, value, or any social relation's number of partners differs beyond 2 SE.

Reference values (hold-out group 1, median over seeds; from the analysis script's dry run):

| Setting | Relation | Partners (reference) | Midpoint |
|---|---|---|---|
| Squared error, approach 1 | author×article / author×affiliation / article×journal | 3.00 / 3.46 / 4.93 | 2.00 / 2.23 / 2.97 |
| Squared error, approach 2 | article×affiliation / article×article / article×journal | 4.12 / 3.83 / 4.78 | 2.56 / 2.42 / 2.89 |
| KL, approach 1 | author×article / author×affiliation / article×journal | 1.13 / 1.03 / 1.46 | control |
| KL, approach 2 | article×affiliation / article×article / article×journal | 1.08 / 1.02 / 1.72 | control |

## 5. Expected outcomes, their explanations and consequences

The underlying choice is between two representations of the same reality. Take a university whose researchers belong
to three topic communities:
- **(R1)** its membership is split over the three communities, and the links are diagonal. This is R-A.
- **(R2)** it sits in an organisation community of its own, linked to the three author communities: one-to-many.

| Outcome (squared error) | What it looks like | Explanation | Consequence for decisions |
|---|---|---|---|
| **1. Start artefact** | Stays near one partner; data loss, existence and value not worse (or better) | The spread in pass 1 comes from the random social start; the data do not need R2 | R-A reachable without a penalty → **O1 + a diagonal start for all link matrices** (a start change, not a term). If the fit is *better*, the random start led to poorer solutions. Before pass 3: the squared-error stage-0 fits (passes 1–2, swap) were made with the old start; whether they must be redone is a decision for the owner (see §7). |
| **2. Data / loss favour spread** | Returns to 3–5 partners | Squared error with these data prefers R2 whatever the start | R-A under squared error needs **O3** (penalise the off-diagonal share) at a measured cost in fit, or R-A is revised for social facets. Under the owner's principle (reproduce the data; our assumptions only as far as they are correct), imposing R1 needs evidence that it is correct, not only that it is wanted. Reported beside the pass-3 loss choice: KL meets R-A where squared error does not. |
| **3. Start trap with a cost** | Stays near one partner, but data loss is higher or held-out lower beyond 2 SE | Both representations are reachable; the data fit R2 better | As in 2: holding to R-A costs fit, and the size of the cost is evidence on how much one-to-many social structure the data contain (e.g. universities and journals spanning topics). O3's cost can be read off this arm. |
| **4. Mixed** | Relations differ (e.g. affiliations stay near one partner, journals return to spread) | Social relations differ in structure | A per-relation answer: O3 weighted by matrix (envisaged 2026-11-06) or R-A for some facets only. |

| Outcome (KL, control) | Meaning |
|---|---|
| **Unchanged** (expected) | The KL solution does not depend on how the social links start → the start question concerns squared error only |
| **Changed** | The start shapes KL solutions too → the start becomes a design question for both losses |

**Also read, under every outcome:**
- **Stability.** If one arm's seeds agree more, its solutions are better determined by the data. If both arms reach
  equal fit with different structure, the data cannot decide between R1 and R2.
- **Presence and detachment.** Under R1, community numbers mean the same group across facets, so detachment should
  fall.

**Prior expectation (stated before the data, weak).** Outcome 2 or 3 for squared error is somewhat more likely than
outcome 1, because squared error already spreads child×article from a diagonal start. KL: unchanged.

## 6. What the test cannot show

- **Limited scope:** one hold-out group, one K per loss, one alternative start, no extra terms. A different K or the
  λ_child term could change the balance.
- **Not which representation is true.** It shows only which one the loss reaches from each start, and at what cost in
  fit and held-out prediction. The held-out cells of social relations are few (author×affiliation /
  article×affiliation only), so fit to the social side is judged mainly by data loss.

## 7. How the result feeds decisions, and the order of work

1. **O1 / O3** (§8.2 step 2; to be chosen after stage 0). This test separates start from data in the off-diagonal
   diagnostic.
2. **The start** (hybrid, decided 2026-11-10). Outcome 1 would change it for the social link matrices.
3. **Order of work.** If the start changes, it should change **before pass 3**, which chooses the loss, so pass 3
   does not have to be redone. Whether passes 1–2 and the swap need refitting under the new start would be decided
   then (cost: about 1,200 pass-1 fits).
4. **The loss choice** (pass-3 rule, unchanged). The result is reported beside it.

## 8. Files

- **Start:** `fit/init.py` `diagonal_social_links`.
- **Runner:** `fit/run_fit.py`, job key `social_link_start: "diagonal"`.
- **Jobs:** `fit/plan.py` `start_test` → `fit/slurm/jobs_start_test.jsonl`.
- **Tests:** `fit/tests/test_start_test.py`.
- **Submit (CSF3 login node):** `bash /mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1/fit/slurm/start_test.sh`
  (5 tasks × 8 fits, 24 h).
- **Read (incline):** `python -m fit.start_test_analysis --out fit/results/stage0/start_test.json`.
