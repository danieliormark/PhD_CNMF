# D1 — how strongly the journal matrix enters the model: pre-registered measures and decision rule

Ticket 107, layer 2, Psychology trial (build `psych_trial_v2`, 2,136 articles). Written 2026-11-05. Status of each part
is marked **DECIDED (owner, date)**, **PROPOSED** or **OPEN**; nothing here has been run. Numbers in the examples are
either measured (source named) or invented for illustration (said so). Calculations behind the examples:
`d1_examples.py` → `d1_examples.txt` (no data read, no model fitted).

## 1. Why this decision is needed

The model fits several relation matrices with one shared set of K communities. Each matrix enters the loss with a
relation weight; the weights are set by the decided rule (each domain gets half, split equally among its matrices;
design doc addendum 2026-11-02 (2)). The rule balances weights, not what the fit actually does with them.

**The concern with article×journal.** Every article has exactly one journal, so article×journal is a *partition*: a
set of disjoint blocks, one per journal. A partition is very easy to reproduce with a few communities. Measured on the
trial build: K communities fitted to article×journal alone could capture 15% (K = 2) to 64% (K = 30) of it, against
1–17% for every other matrix (`capturable_shares.txt`). Weighted by the rule, journals would account for 64–70% of the
**pull** (defined in §3) at every K. The fear is that the communities then become groupings of journals rather than
groupings that combine meaning (semantic relations) and people (social relations).

**The other side.** Journals are part of the social structure we want to describe: a journal is where a research
community publishes. The owner's measured point (2026-11-02, `journal_checks.txt`): journals that are *popular within
the subset* are more psychology-centred (psychology-primary share 0.11 vs 0.03–0.05), and journals differ in kind (key
journals of a subfield, journals at intersections of subfields, journals at the intersection with fields outside the
subset). Dropping or flattening journal information may lose real structure.

D1 therefore compares several ways of entering the journal matrix and chooses one, by criteria fixed before the fits.

## 2. The arms (what is compared)

All arms: Frobenius scaling (each matrix divided by its Frobenius norm, so ‖X_r‖² = 1); the decided weight rule, with
the social half rescaled to 0.5 when the journal matrix is reduced or removed (DECIDED, owner, 2026-11-02); both
approaches (approach 1: author×article, author×affiliation; approach 2: article×affiliation, article×article; both:
child×parent, child×article and the journal matrix); coarse K = 3, 5, 8, 13, 20; 5 hold-out groups × S seeds.

| Arm | Journal matrix | Its weight | What it represents |
|---|---|---|---|
| J1 | none | — (other social matrices 0.25 each) | the **reference**: communities formed without any journal information |
| J2 | binary (1 if the article is in the journal) | 1/6 | the current decision P4: every journal tie counts the same |
| J3a | binary × 1/4 | 0.056 | journal information kept, its overall strength reduced |
| J3b | binary × 1/8 | 0.029 | as J3a, reduced further |
| J4 | journal column × idf | 1/6 | journals with many articles count less per article (idf = inverse document frequency) |
| J5 | columns × 1/√n_j, singleton-journal ties missing | 1/6 | every journal of ≥ 2 articles carries equal total weight; one-article journals carry none |

*Caveat for reading J4 and J5:* they down-weight journals with many articles — which, in this subset, are the more
psychology-centred ones (§1). A good result for J4/J5 means the fit is better when the core journals count less.

## 3. Terms and notation

| Symbol / term | Meaning |
|---|---|
| r | a relation matrix (e.g. child×article) |
| X_r, X̂_r | the matrix and the model's reconstruction of it |
| w_r | **relation weight**: the multiplier on matrix r's error in the loss (the decided rule) |
| K | number of communities |
| ‖·‖² | sum of squared cells (squared Frobenius norm) |
| **captured share** c_r | 1 − ‖X_r − X̂_r‖² / ‖X_r‖²: the share of matrix r the model reproduces, relative to predicting all zeros. In the hold-out fits computed on the visible cells only. 0 = no better than all zeros; 1 = perfect. |
| **best share at K** b_r(K) | (σ₁² + … + σ_K²)/‖X_r‖², σ the singular values of X_r: the most any K communities could capture of matrix r *fitted alone* (Eckart–Young theorem); an upper bound |
| **pull** | w_r × b_r(K) as a % of the sum over the fit's matrices: where fitting *could* gain most; computed before any fit |
| **realised share** | w_r × c_r as a % of Σ_s w_s c_s: matrix r's part of the weighted fit the model *actually achieved* |
| **influence** | the change in the communities when matrix r is removed or reweighted; measured only by comparing fits (here: D1-M2) |
| m_ik | article i's **weighted membership** in community k (ticket 100): its membership vector, each community weighted by its reconstruction mass, rows summing to 1 |
| n_j | number of trial articles in journal j |
| hold-out group g | one of 5 sets of hidden cells of child×article and child×parent, the same in every arm (the data-sampling unit) |
| seed s | one random start of the fit; S seeds per group (S = 10–20, owner 2026-11-03); seeds only sharpen a group's mean |
| Δ | difference in a score between two arms, per group (§4.1) |
| δ | **noise margin**: the size of difference that seed randomness alone produces (§4.1) |
| **class** | the label a comparison receives under the four pre-registered outcomes (§4.1), as in L2.4 |
| Track A / Track B | the two stability measures (CLAUDE.md §4.24): communities of two fits are matched (Hungarian assignment), then each entity's membership is compared; Track A gives every entity an equal vote (Jensen–Shannon distance), Track B weights entities by their membership mass (cosine) |
| chance-adjusted score | (real − chance)/(1 − chance), chance from row shuffles; 0 = chance agreement, 1 = identical |

## 4. The measures

Every measure compares an arm Jx with J1 under the same hold-out groups. Measures are named with the element prefix
(D1-M1 …) because other elements (L2.4, L2.5) number their own measures from M1.

### 4.1 D1-M1 — held-out prediction

**Definition.** Score of a fit = the weighted squared error on the hidden cells of child×article and child×parent
(the training loss's own form: Σ_r w_r × hidden error of r). Both matrices have weight 0.25 in every arm, so the score is
the same criterion in all arms (L2.6, DECIDED 2026-11-03). Per group g: Δ_g = score(J1) − score(Jx), each arm's score the
mean over its S seeds; **positive Δ = Jx predicts the hidden semantic cells better**. The noise margin δ = 95th
percentile of the absolute per-group difference between two halves of the same arm's seeds, over all splits.

**Classes** (L2.4 definitions): **adds information** — the 95% interval of the mean Δ lies above 0 and the mean Δ ≥ +δ;
**misleads** — the interval lies below 0 and the mean Δ ≤ −δ; **redundant** — the mean Δ is shown to lie within ±δ by two
one-sided tests (Schuirmann 1987; Lakens 2017); **inconclusive** — anything else.

**What it represents.** The journal matrix has no hidden cells (decided 2026-10-29), so this does not test how well
journals are fitted. It tests whether knowing the journals makes the communities better at predicting *meaning* — which
child hyperedges occur in which articles and in which parent statements — that the model has not seen. If journal
information reflects real research communities, it should help; if it imposes groupings that cut across meaning, it
should harm.

**Reported, not classified:** held-out scores of the social matrices with hidden cells. Their weights differ between
arms because of the rescaling (e.g. article×affiliation 0.25 in J1, 1/6 in J2, 0.222 in J3a), so they are not the same
criterion across arms.

### 4.2 D1-M2 — change in article memberships (influence)

**Definition.** Track A and Track B, chance-adjusted, between fits of Jx and J1 in the same hold-out group, averaged
over seed pairs. "Memberships changed" if the score is below the 5th percentile of the within-arm seed-pair scores (two
seeds of the same arm and group), per track.

**What it represents.** Whether adding the journal matrix (in that form) changes *which articles belong together* —
the influence of the matrix on the communities. A matrix can be heavily fitted and still change little (if journals
already coincide with the communities), or be lightly fitted and change a lot.

### 4.3 D1-M3 — realised share of the journal matrix

**Definition.** Realised share of article×journal (§3), mean over groups and seeds, reported beside its pull and beside
the realised share of the semantic domain (child×parent + child×article). Not defined for J1.

**Flag "journal-dominated" — DECIDED (owner, 2026-11-05):** the journal's realised share ≥ the semantic domain's realised
share (the journal alone carries as much of the achieved fit as the whole semantic side, which was given half of the
weight). Alternatives: ≥ 1/3 (twice its weight share of 1/6); report without a flag.

**Worked example (invented numbers)**, approach 1, J2, K = 8:

| | child×parent | child×article | author×article | author×affiliation | article×journal |
|---|---|---|---|---|---|
| w_r | 0.25 | 0.25 | 1/6 | 1/6 | 1/6 |
| c_r | 0.015 | 0.035 | 0.010 | 0.030 | 0.300 |
| w_r × c_r | 0.0038 | 0.0088 | 0.0017 | 0.0050 | 0.0500 |
| realised share | 5.4% | 12.7% | 2.4% | 7.2% | **72.3%** |

Semantic domain 18.1% < journal 72.3% → flagged under the proposed threshold.

**What it represents.** Where the model actually spends its K communities. Unlike pull (an upper bound computed from
each matrix alone, before fitting, identical across fits except for the weight), it reflects the joint fit, in which a
community must serve all matrices at once.

**Limitation.** It rewards how easy a matrix is to fit: a partition is cheap to reproduce, so some excess over the
journal's 1/6 weight share is expected and is not by itself a defect. This is why the flag uses a high bar and why the
flag alone never triggers a change of arm (§5).

**Established use.** Explained variance per block is the standard report in multi-block factorisation: variance
explained per view and factor in MOFA (Argelaguet et al. 2018, *Mol. Syst. Biol.*), joint and individual variance per
block in JIVE (Lock et al. 2013, *Ann. Appl. Stat.*), fit per dataset in coupled matrix–tensor factorisation (Acar,
Kolda & Dunlavy 2011). The weighted percentage is closest to the *contribution of a group* in Multiple Factor Analysis
(Escofier & Pagès), weighted by the group weight and summing to 100%; ours is computed over the whole fit, not per axis.

### 4.4 D1-M4 — journal-shaped communities

**Definition.** For community k and journal j (only journals with ≥ 2 trial articles; **DECIDED, owner, 2026-11-05**):
share_kj = Σ_{i in j} m_ik / Σ_i m_ik, the share of k's article membership mass in journal j; j_k = max_j share_kj. Shuffle
the journal labels among the articles (each journal keeps its n_j, each article its membership), recompute j_k, repeat
(number of shuffles: §4.5). p_k = (1 + number of shuffles with j_k ≥ the real j_k)/(1 + number of shuffles).
- **journal-shaped**: p_k < 0.05 **and** j_k ≥ 0.25;
- **journal-associated**: p_k < 0.05 and j_k < 0.25 (reported as a count, not used in the rule);
- communities whose largest share sits in a **singleton journal** (one trial article: a quarter or more of the
  community's article mass on one article) are reported separately as article-dominated (DECIDED, owner, 2026-11-05).

Per arm: the number of journal-shaped communities, mean over groups and seeds; Jx "has more than J1" if the difference
exceeds the seed-half noise margin of this count (computed as δ in §4.1). Reported split by the journals'
within-subset popularity (groups as in `journal_checks.py`) and with each journal's subfield profile.

**Why the shuffle.** Journals differ greatly in size (largest 184 of 2,136 articles, 8.6%). A community spread evenly
over the corpus already has ~9% of its mass in that journal, and a community concentrated on few articles gets high
shares by chance. The shuffle keeps both facts and breaks only the article–journal link, so it gives the j_k that size
and membership concentration alone produce.

**Why the 0.25 floor.** With many communities and 169 journals, small but real over-representation will be common. The
floor separates "over-represented in a journal" from "built around a journal".

**Worked example** (`d1_examples.txt`: trial journal sizes, invented memberships, 999 shuffles):

| Community | j_k (journal size) | shuffles: mean / 95th pct | p | verdict |
|---|---|---|---|---|
| A spread over the corpus | 0.097 (184) | 0.086 / 0.097 | 0.045 | j_k < 0.25: not journal-shaped |
| B 30% of its mass in one 8-article journal | 0.302 (8) | 0.099 / 0.139 | 0.001 | **journal-shaped** |
| C over-represented in the largest journal | 0.219 (184) | 0.086 / 0.098 | 0.001 | journal-associated |

A shows why the floor also matters statistically: its j_k sits at the chance level and its p falls just under 0.05 —
the floor keeps such borderline cases out.

**What it represents.** Whether communities are organised around single journals. **J1 is the key baseline**: J1 never
sees journals, so its count shows how journal-aligned the communities are from meaning and co-authorship alone. More
journal-shaped communities in Jx than in J1 means the journal matrix imposes journal structure; a similar count means
the alignment was in the data anyway (journals as real research communities).

### 4.5 Number of shuffles — DECIDED (owner, 2026-11-05): 999

The 200 used so far was carried over from earlier chance-level checks (ticket 95, the temporal matching), not derived
for this test. With n shuffles the empirical 95th percentile corresponds to a true quantile with standard error
√(0.05·0.95/n): 0.015 at n = 200 (true quantile ~0.92–0.98), 0.007 at n = 999 (0.936–0.964); the smallest attainable p
is 1/(n + 1) (`d1_examples.txt`). Standard guidance for Monte Carlo tests at the 5% level is about 1,000 (Davison &
Hinkley 1997, ch. 4; Manly 2007); p = (1 + r)/(1 + n) is the recommended estimate (North, Curtis & Sham 2002). The
shuffles need no fitting (one weighted count per community per shuffle), so 999 costs seconds. Timed: 999 shuffles for one fit at K = 20 take 0.6 s (journal sums by `np.add.at`), i.e. about 1 h over 6,000 fits —
small beside the fits; the owner's condition ("as long as it does not take too much computation") is met. The same
change applies to L2.4-M3 (team-shaped communities), for consistency.

**To revisit elsewhere (owner, 2026-11-05):** other shuffle or resampling counts in the pipeline were also set without
a derivation: (i) Track A/B chance level — one row-shuffle per seed pair, so 45 values at 10 seeds (`chunk13v9.py`
`run_dual_track_stability_analysis`; CLAUDE.md §4.24); (ii) the temporal matching's 200 shuffles
(`chunk13v10.py` `TEMPORAL_MATCH_SHUFFLES`, not on the trial path); (iii) the "5th percentile of within-arm seed-pair
scores" used by D1-M2 and L2.4-M2 — at S = 10 there are 45 seed pairs per group, so this percentile rests on few values;
(iv) the noise margin δ from seed halves (its 95th percentile over splits); (v) `parallel_analysis.py`. Each to be
checked against the same Monte Carlo precision argument when the M1 fit code is written.

## 5. Decision rule — PROPOSED, except where marked

**One J* per approach (DECIDED, owner, 2026-11-05).** The rule is applied separately to each approach, giving J*₁ and
J*₂. The K search of stage 2 runs at the approach's own J*; each approach is also fitted at the other's J* (cross-fits),
so the approach comparison (stage 3a) is 2 approaches × 2 J × 2 K and a difference between approaches can be split into
the part due to the journal weighting and the part due to the social matrices. Stage 1 already fits every approach in
every arm at the coarse K, so cross-fits are needed only at K₁, K₂ not on the coarse grid (≈ 200 fits at S = 10; none
if J*₁ = J*₂).

**At which K — DECIDED (owner, 2026-11-05).** D1 is decided after stage 1, which fits the arms only at the coarse
K = 3, 5, 8, 13, 20, before K₁, K₂ are known. Candidate K per approach = the coarse K values inside the one-standard-error
band of any arm; an arm's class = the class it has at the majority of candidates (no majority → inconclusive; one
candidate → its class). Required re-check after stage 2: J1, J* and the runner-up at the final K₁, K₂ on D1-M1, M2, M4;
a contradiction is put to the owner with both results (free when K₁ is a coarse value; otherwise 300 fits at 5 groups,
S = 10). This also covers a final K above 20.

**Rule, per approach:**
1. Exclude arms classified **misleads** (D1-M1).
2. An arm has a **journal problem** if it has more journal-shaped communities than J1 (D1-M4) **and** its memberships
   changed relative to J1 (D1-M2). The D1-M3 flag is reported beside it, as an aggravating reading, never alone.
3. Among the remaining arms: if any **adds information**, take the best held-out score among those (ties within δ → J2
   if among them, else the highest journal realised share). If none adds information, take **J2** (P4, the decided
   default: the ablation tests P4, it does not overturn it) if it remains, else the remaining arm with the highest
   journal realised share (keeps the most journal information).
4. If only J1 remains: J* = J1, and journals receive a profile after the fit (each journal's membership = the mean of
   its articles' memberships): the journal facet is still described but does not shape the communities.
5. The owner decides after the text check (top articles and child hyperedges of journal-shaped communities); the rule
   gives the default and its reasons.

**Worked example (invented)**, approach 2, K = 8, J1 has 1 journal-shaped community:

| Arm | D1-M1 class | D1-M4 journal-shaped | D1-M2 changed? | outcome |
|---|---|---|---|---|
| J2 | redundant | 4 | yes | journal problem → out |
| J3a | redundant | 1 | no | remains |
| J4 | inconclusive | 2 | yes | journal problem → out |
| J5 | inconclusive | 1 | no | remains |
| J3b | redundant | 1 | no | remains |

Nothing adds information and J2 is out, so the default is the remaining arm with the highest journal realised share:
J3a (weight 0.056). Reading: binary journals at full weight create journal-shaped communities without improving
prediction; a quarter of the weight keeps journal information without that effect.

## 6. Multiple comparisons and test power — OPEN

**The problem.** Five arms are compared with one reference. If each comparison is a test at 5% and no arm truly
differs, the chance that at least one is falsely classed *adds information* or *misleads* is ≈ 1 − 0.95⁵ = 23%
(somewhat less, since the comparisons share J1). A false *adds information* would choose an arm; a false *misleads*
would exclude one — both change the decision, so the error rate over the family of five comparisons is what matters.

**Corrections.** *Bonferroni*: each test at 5%/5 = 1%; valid for any set of tests without assumptions; conservative.
*Holm* (step-down Bonferroni): same validity, never less powerful. *Dunnett* (Dunnett 1955): the procedure designed for
"several treatments against one control" under a normal model; it uses the correlation of the comparisons through the
shared control and is therefore less conservative than Bonferroni while controlling the family error exactly; the
standard choice for this design (Hsu 1996, *Multiple Comparisons*; Bretz, Hothorn & Westfall 2010). Its generalisation
with an estimated correlation matrix (multivariate t; Hothorn, Bretz & Westfall 2008) covers unequal variances.

**Power** (probability of detecting an effect that is really there; true effect in units of the standard error of the
mean Δ; `d1_examples.txt`):

| Groups | Error estimate (df) | Critical value: single / Dunnett / Bonferroni | Power at 2 / 3 / 4 / 5 SE: single | Dunnett | Bonferroni |
|---|---|---|---|---|---|
| 5 | each comparison's own SE (4) | 2.78 / 3.98 / 4.60 | 0.34 / 0.62 / 0.84 / 0.95 | 0.15 / 0.34 / 0.58 / 0.78 | 0.10 / 0.25 / 0.45 / 0.65 |
| 5 | pooled block error (20) | 2.09 / 2.73 / 2.85 | 0.48 / 0.81 / 0.97 / 1.00 | 0.26 / 0.61 / 0.88 / 0.98 | 0.23 / 0.57 / 0.86 / 0.98 |
| 10 | own SE (9) | 2.26 / 3.06 / 3.25 | 0.43 / 0.76 / 0.94 / 0.99 | 0.21 / 0.51 / 0.80 / 0.95 | 0.18 / 0.45 / 0.75 / 0.93 |
| 10 | pooled block error (45) | 2.01 / 2.61 / 2.69 | 0.50 / 0.84 / 0.97 / 1.00 | 0.28 / 0.65 / 0.91 / 0.99 | 0.26 / 0.62 / 0.90 / 0.99 |

Readings: (i) with 5 groups and each comparison's own error, any correction leaves little power; (ii) the larger
gains come from the error estimate and the number of groups, not from Dunnett vs Bonferroni (whose difference shrinks
as df grow); (iii) *pooled block error*: all arms share the hold-out groups (a randomised block design), so one error
estimate from all arms, with (groups − 1)(arms − 1) df, is the classical analysis — at the cost of assuming every arm's
differences are equally noisy, checkable on the stage-1 results; (iv) 10 groups make the SE √2 smaller for the same
true effect (3 SE at 5 groups → 4.2 SE at 10): doubling stage 1 (3,000 → 6,000 fits at S = 10), or 10 groups × 5 seeds
at unchanged cost (fewer seeds for δ and the stability gate); (v) the corrected *redundant* test (equivalence) is
stricter in the same way, so more arms come out *inconclusive*; (vi) the 2-of-3-K requirement adds little protection
against chance (the K values share groups and similar models) and is kept for substantive reasons.

**Update 2026-11-05.** An invented example (6 arms × 5 groups) showed the pooled error flattering a noisy arm (J2) and
hiding a small but consistent effect (J3b); unequal noise is expected structurally (arms far from J1 differ more
between groups) and cannot be checked with 5 groups (Hartley's F-max critical value ≈ 25 at 4 df). **DECIDED (owner,
2026-11-05): Dunnett with each comparison's own error; the pooled version is not used.** Uncorrected intervals are
reported beside the Dunnett ones. Proposed, open: all 10 drawn hold-out groups in stage 1.

## 7. Status summary

| Item | Status |
|---|---|
| arms J1–J5, rescaling of the social half | DECIDED (owner, 2026-11-02) |
| D1-M1 score on the semantic matrices only | DECIDED (owner, 2026-11-03, L2.6) |
| singleton journals excluded from D1-M4, reported separately | DECIDED (owner, 2026-11-05) |
| one J* per approach, with cross-fits | DECIDED (owner, 2026-11-05) |
| D1-M1 classes, D1-M2, D1-M4 definitions, decision rule | PROPOSED |
| D1-M3 flag (journal realised share ≥ semantic domain's realised share) | DECIDED (owner, 2026-11-05) |
| 999 shuffles (and for L2.4-M3); other counts to revisit (§4.5) | DECIDED (owner, 2026-11-05) |
| Dunnett's procedure for the five comparisons with J1 | PRELIMINARILY APPROVED (owner, 2026-11-05) |
| K at which classes are read: candidate K = coarse K inside the one-SE band of any arm of that approach; verdict = class at the majority of candidates (no majority → inconclusive); required re-check of J1, J* and the runner-up at the final K₁, K₂ (covers K above 20) | DECIDED (owner, 2026-11-05) |
| Dunnett with each comparison's own error (estimated-correlation version, Hothorn, Bretz & Westfall 2008); the pooled block error is not used (owner: "I wouldn't assume equal noise") | DECIDED (owner, 2026-11-05) |
| 10 hold-out groups in stage 1 (all 10 of the drawn groups) instead of 5; S chosen after one fit is timed | OPEN |
