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

*[2026-11-18: the effective number of partner communities is no longer the primary measure; the diagonal rule of
§4 point 2 is. The list below is kept as first written.]*

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

## 4. Reading rule — REVISED 2026-11-18 (before any start-test fit was read)

**Owner (2026-11-18):** "I approve your revised reading substituting the midpoint line; importantly for child*article
we should still have >0.5 for within community main diagonal". Background: the owner's statement on the assumption
(design doc, 2026-11-18): the anchor child×article must have a high within-community diagonal (this makes a community
socio-semantic); for the other relations the assumption may be relaxed to "main diagonal > any off-diagonal value";
true communities may still break it (an ongoing split publishing in different journals; an emerging community whose
members are at different universities and rarely collaborate). The midpoint rule of the first version (§4 point 2,
never decided) is withdrawn. All readings use the **aligned numbering** (RL-260).

1. **Paired by seed** (DECIDED 2026-11-17, owner: "I approve the 2se rule"). Both arms share the same membership start
   for each seed. For every measure: difference = diagonal − reference over the seeds where both converged; mean, SE =
   SD ÷ √n; "better" / "worse" beyond **2 SE** in that direction, otherwise "within noise".
2. **Diagonal rule, every relation except child×article** (approach 1: child×parent, author×article,
   author×affiliation, article×journal; approach 2: child×parent, article×affiliation, **article×article**,
   article×journal). Community k meets it in a relation if its own-numbered cell of the parts table, part(k,k), is
   larger than every other cell of **its row and its column** (owner: "I agree that both"). Measure: share of the
   relation's reconstructed tie weight (half credit) in communities that meet it. Row-only and column-only shares and
   counts are reported beside it. article×article (articles linked through shared authors, chunk12v3 P5) is read under
   both losses like the others (owner, 2026-11-18: "If something does not work under one loss, it imho is not an excuse
   not to test under another loss"); its diagonal is fixed by the article numbering, so no renumbering is involved.
3. **Anchor guard, child×article: > 0.5 within the community diagonal** (owner, 2026-11-18). Unit — **pending the
   owner's choice**: (a) per fit, the within share Σ_k part(k,k) ÷ T > 0.5 in every converged fit of the arm; or
   (b) per community, every community's own-numbered cell is more than half of its child×article tie weight (row and
   column). Both are reported.
4. **Fit cost.** "Worse" if data loss is higher, or existence or value is lower, beyond 2 SE (paired).
5. **Outcome per relation, squared error.**
   - **1, start artefact:** the rule share is higher in the diagonal arm beyond 2 SE, the fit is not worse, and the
     anchor guard holds.
   - **3, trade-off:** the rule share is higher beyond 2 SE, but the fit is worse or the anchor guard fails.
   - **2, start not the cause:** otherwise.
   - **4, mixed:** relations of one setting differ.
   **KL is the control:** "changed" if data loss, existence, value or any relation's rule share differs beyond 2 SE.
6. **Failures of the rule, studied in both arms** (owner: "it is interesting to study cases when row or column is not
   beaten"). For every community that fails in a direction: its own share, its main partner (number, share) and its tie
   weight. Per community number, the main partner in every relation is also saved, so that chains can be read (e.g.
   authors of k, writing articles of k, mainly at organisations of community l, publishing in journals of k and l2).
   **Recurrence across seeds:** each facet is matched across two seeds on its own (independent of numbering). A failure
   of community k toward main partner l recurs in another seed if the matched community fails in the same direction
   toward the matched partner. It "recurs" if it does so in more than half of the arm's other seeds. A recurring failure
   is a candidate real practice (a split, a dispersed emerging community) to be inspected on named entities. A
   non-recurring one depends on the seed. Recurrence is necessary, not sufficient, evidence of a real practice.
7. **Also reported:** effective number of partner communities (no longer classifying); stability per arm, with one
   matching across facets and with each facet matched on its own (RL-263); presence and detachment.

**Reference values (pass 1, hold-out group 1, seeds 1–10, aligned numbering; from the dry run, which uses the
reference as both arms; all groups in RUN_LOG RL-264).** Rule share in tie weight; failures per fit; share of failure
weight that recurs across seeds:

| Setting | Relation | Rule share | Failures per fit | Recurring weight |
|---|---|---|---|---|
| Squared error, approach 1 (K 20) | author×article / author×affiliation / article×journal / child×parent | 0.46 / 0.56 / 0.47 / 1.00 | 20 / 26 / 25 / 0 | 0 / 0 / 0 / — |
| Squared error, approach 2 (K 20) | article×affiliation / article×article / article×journal / child×parent | 0.53 / 0.00 / 0.48 / 1.00 | 24 / 39 / 25 / 0 | 0 / 0.01 / 0.11 / — |
| KL, approach 1 (K 13) | author×article / author×affiliation / article×journal / child×parent | 0.96 / 0.99 / 0.90 / 1.00 | 3 / 6 / 3 / 0 | 0 / 0 / 0 / — |
| KL, approach 2 (K 13) | article×affiliation / article×article / article×journal / child×parent | 0.95 / 0.60 / 0.76 / 1.00 | 4 / 14 / 5 / 0 | 0 / 0.51 / 0.03 / — |

Anchor (child×article), reference: within share 0.78 (squared error approach 1), 0.73 (approach 2), 1.00 (KL); 12–13
of 20 communities have more than half of their tie weight on their own number under squared error, all 13 under KL.

## 5. Expected outcomes, their explanations and consequences (revised 2026-11-18)

| Outcome (squared error) | What it looks like | Explanation | Consequence for decisions |
|---|---|---|---|
| **1. Start artefact** | Rule share higher beyond 2 SE; fit not worse; anchor guard holds | The rule failures in pass 1 come from the random social start | **O1 + a diagonal start for all core matrices** (a start change, not a term). Before pass 3: whether the squared-error stage-0 fits must be redone is a decision for the owner (§7). |
| **2. Start not the cause** | Rule share within noise (or lower) | Squared error with these data reaches the same structure from either start | Read the recurrence of failures. Recurring failures are candidate real practices (inspect named entities; no penalty). Non-recurring failures are seed-dependent: **O3** (a term on the off-diagonal parts), or counted against squared error in the loss choice. |
| **3. Trade-off** | Rule share higher beyond 2 SE, but fit worse or anchor guard fails | Both structures are reachable; the data fit the off-diagonal one better | The cost of the rule in fit is measured directly; O3's cost can be read off this arm. |
| **4. Mixed** | Relations differ | Social relations differ in structure | Per-relation answer: O3 weighted by relation, or the rule for some relations only. |

| Outcome (KL, control) | Meaning |
|---|---|
| **Unchanged** (expected) | The KL solution does not depend on how the social cores start |
| **Changed** | The start shapes KL solutions too → the start becomes a design question for both losses |

**Prior expectation (stated before the data, weak).** Squared error: outcome 2 or 3 somewhat more likely than 1, as
before. In pass 1 squared-error failures almost never recur across seeds, so under outcome 2 they would read as
seed-dependent. KL: unchanged; its article×article failures recur in about half their weight, a candidate real pattern
(articles sharing authors across topic communities).

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
- **Read (incline):** `python -m fit.align --stages start_test`, then `python -m fit.start_test_analysis --out
  fit/results/stage0/start_test.json` (needs `ANCHOR_UNIT` set from the owner's choice; `--dry-run` checks the code on
  the reference fits only).
