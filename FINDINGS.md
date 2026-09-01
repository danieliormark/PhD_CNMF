# FINDINGS.md — Chunk 13v9 Analytical Record

> Companion to CLAUDE.md. CLAUDE.md covers architecture, contracts, and the
> defect register — *what the code is*. This file covers *what we have learned
> about the model*: findings, the evidence behind them, hypotheses that were
> refuted, and what remains open.
>
> Read this before proposing changes to the loss function, the metrics, or the
> interpretation of results. Several plausible-sounding ideas here have already
> been tested and failed; several apparent findings turned out to be artifacts.
>
> Third companion file: `SESSION_PROTOCOL.md` — standing conditions for
> diagnostic/patch sessions with Claude Code (default run settings, reporting
> granularity, persistence rules, what's out of scope unless a prompt opens it).

---

## 0. How to read this file

Each finding carries a **status**:

- **Established** — measured, replicated, mechanism understood
- **Supported** — measured, but scope-limited or mechanism unconfirmed
- **Open** — identified, not resolved
- **Refuted** — hypothesis tested and rejected (recorded so it isn't retried)

Where a finding rests on a specific measurement, the numbers are given. Where it
rests on an argument, the argument is stated so it can be checked.

---

## 1. Core tensor Z is non-identifiable

**Status: Established.**

### The problem

For any invertible K×K matrix W:

```
(U[f1]·W) · (W⁻¹·Z·W⁻ᵀ) · (U[f2]·W)ᵀ  =  U[f1]·Z·U[f2]ᵀ
```

because `W·W⁻¹ = I` and `W⁻ᵀ·Wᵀ = I`. The reconstruction is unchanged, so the
data cannot distinguish one (U, Z) pair from the other. This is **rotational
indeterminacy** — a standard property of tri-factorisations without an
orthogonality constraint.

Z would be identifiable if U were orthogonal (`UᵀU = I`). It isn't, and can't be:
soft membership — an author belonging to several fields at once — is the
substantive point of the model.

### Evidence

| Run | λ_z | Anchor Z init | recon_loss | Anchor coherence |
|---|---|---|---|---|
| 2 | 0.0 | diagonal | **0.7593** | ~0.95–1.00 |
| 4 | 0.0 | random dense | **0.7591** | mostly 0.000 |

Four decimals apart in fit; opposite Z structure.

**Independent corroboration from production history.** In 13v6 and earlier,
before SVD initialisation of anchors, anchor Z diagonals were near zero and
non-anchors sat around 0.08–0.8. After SVD init was introduced (13v7+), anchors
read ~0.99. Same corpus, same data — only the initialisation changed.

### Consequence

Any quantity read from Z alone reports where the optimiser landed, not what the
data says. This invalidates:

- Module 3 §3's coherence check, `Z[k,k]/(0.5·(row_sum_k + col_sum_k))`
- The earlier "Latent Structural Volume" measure based on Z's diagonal

Both were reading a gauge.

---

## 2. Off-diagonal Z is substitutable, not absent

**Status: Supported for the toy corpus. Scope-limited.**

### Evidence

Constraining Z to exactly diagonal (off-diagonal frozen at zero, not merely
penalised) — four comparisons:

| Comparison | Slice | Init | Gap |
|---|---|---|---|
| Run 1 → Run 9 | T1 | production | −1.43% (constrained better) |
| Run 15 → Run 16 | T2 | production | −1.80% |
| Run 11 → Run 13 | T1 | fully random | −2.67% |
| Run 17 → Run 18 | T2 | fully random | −1.36% |

All four negative. Convergence also ~2.4× faster (398 epochs vs 961).

The fully-random comparison matters most: Run 11 removed the SVD entirely
(random U, random dense Z including anchors, no propagation). If NNDSVD
orthogonality had been suppressing genuine coupling, freeing the model should
have let unconstrained win. It didn't.

### Important qualification — RETRACTED illustrative example (see §14)

*Original claim (now corrected):* "Off-diagonal Z can carry an entire relation.
In Run 4 (λ_z=0.0), `share_sum` for `M_Cousin_Parent` was exactly 0.0000 — no
within-community mass at all, the relation carried entirely by cross-community
pairings."

**This example was wrong.** §14's permutation test found that
`M_Cousin_Parent`'s Z in that run was — up to scale — a clean **permutation
matrix** (structure score 24.5, diagonal share jumping from 0.000 to 0.989
after re-aligning labels). The relation had normal within-community structure;
the two facets' community labels were simply mismatched. `share_sum = 0.0000`
measured label misalignment, not absent within-community coupling.

**A relation that does show genuine off-diagonal dependence (better example):**
`S_Art_Auth` in the same run — diagonal share 0.592 raw, only 0.620 after
best-case relabelling (structure score 0.043, the lowest in the table). This
stays incoherent even after correcting for permutation, which is what genuine
cross-community coupling looks like.

The general **substitutability** conclusion below does not depend on the
retracted example — it rests on the four diagonal-vs-unconstrained comparisons,
which were not re-examined by the permutation test and stand as before.

### Why it has not been adopted

- Scope is 25–36 articles. Cross-community coupling may exist at 22k and be
  undetectable here (~7 articles per community at K=4 leaves little room for
  cross-community bridges to form).
- Freezing Z reverts the model toward joint NMF, discarding the
  tri-factorisation's stated contribution.

### Why "let Optuna crank λ_z to 1.0" does NOT substitute for freezing

The penalty applies to `Z_scaled[i,j] = Z_pos[i,j] · scale_f1[i] · scale_f2[j]`.
The cheapest way to reduce that product is to shrink `U_scales`, not to zero the
off-diagonal. So the penalty pressures community **sizes** rather than coupling.

Measured at λ_z = 1.0, K=4: mass shares 0.470 / 0.095 / 0.261 / 0.174 (vs
0.24/0.24/0.31/0.21 at λ_z = 0.05), recon_loss worse (0.7873 vs 0.7746),
convergence 68% slower. At K=6 it failed to converge within 2000 epochs.

**λ_z → ∞ does not converge to the diagonal constraint.** These are different
interventions with different effects.

---

## 3. The (R) vs (G) mechanism test was invalid — WITHDRAWN

**Status: Refuted (the test, not necessarily the hypothesis).**

Two explanations were proposed for why the diagonal constraint *improves* fit
rather than merely not hurting:

- **(R) Routing** — off-diagonal `Z[i,j]` lets community *i*'s column in `U[f1]`
  pair with community *j*'s column in `U[f2]`, permitting two communities to have
  identical `U[f1]` columns and still reconstruct distinctly. Closing the route
  forces differentiation.
- **(G) Generic capacity reduction** — removing K²−K parameters per relation
  discourages degenerate solutions independently.

The test measured Spearman correlation between `|c_u off-diagonal|` and
`|Z_scaled off-diagonal|` across community pairs. Result: signs flipped across
runs for five of nine relations, no consistent tilt.

**Why the test was invalid.** It assumed that if routing occurs, the mass must
appear in `Z[i,j]` specifically. But §1 establishes that Z's placement is
arbitrary — in a flat valley, Adam can distribute mass across `Z[i,i]`,
`Z[j,j]`, and `Z[i,j]` freely. Absence of correlation is what
non-identifiability predicts *whether or not* routing occurs.

The test was self-undermining: it assumed Z placement is informative while §1
says it isn't. **Mechanism remains unknown.**

---

## 4. `affil` and `auth` collapse at K ≥ 4

**Status: Established. Mechanism open.**

`c_u[f] = U_norm[f]ᵀ · U_norm[f]` is a K×K cosine-similarity matrix between
community loading profiles on facet *f*. Off-diagonal `[i,j] = 1.0` means
communities *i* and *j* have identical profiles on that facet — they are not
two communities there.

`max |off-diagonal| c_u`, production init, C6, T1:

| Facet | Live (T1) | K=2 | K=3 | K=4 | K=6* |
|---|---|---|---|---|---|
| affil | 31 | 0.0019 | 0.0073 | **0.9859** | **0.9994** |
| auth | 212 | 0.0248 | 0.0327 | **0.9679** | **0.8386** |
| core_atom | 300 | 0.1433 | 0.5599 | 0.6345 | 0.5353 |
| cousin_he | 210 | 0.1766 | 0.1521 | 0.1461 | 0.2058 |
| core_child_he | 166 | 0.1284 | 0.1176 | 0.1420 | 0.1905 |
| fringe_atom | 307 | 0.0658 | 0.2641 | 0.2730 | 0.4303 |
| journ | 14 | 0.0012 | 0.2153 | 0.0105 | 0.0205 |
| art | 25 | 0.0457 | 0.0581 | 0.0475 | 0.1304 |
| parent_he | 63 | 0.3394 | 0.1238 | 0.0979 | 0.1739 |

*K=6 hit the epoch ceiling — flagged, not a converged fixed point.

Sharp transition (0.007 → 0.986 for `affil`), not gradual degradation.

### REFUTED: the facet-size hypothesis

Predicted that collinearity tracks entity count — too few entities to support K
directions. It does not:

- `affil` (31 live) and `auth` (212 live) collapse at the same K despite being 7× apart
- `journ` (14 live, smallest facet) stays clean at every K under production init

### REFUTED: the Phase-2 propagation hypothesis

Predicted that `auth`'s collinearity is manufactured at initialisation, because
Phase 2 projects `auth` from `art` through `S_Art_Auth` — all K columns passing
through the same matrix. Measured at true epoch 0, before any gradient step:

- Production init: all nine facets 0.51–0.76 (`auth` 0.6848, `journ` 0.7635)
- Random init: all nine facets 0.75–0.80 (`auth` 0.7590, `journ` 0.8016)

Neither facet is an outlier at init. Effective rank equals K for every facet
under both. **Collinearity is a training dynamic, not an init artifact.**

### Untested hypothesis

`affil` and `auth` are the only two facets that do not connect directly to
`art`. Every facet with a direct `art` edge stays below 0.44 at all K. `affil`
reaches `art` only via `auth` (two hops), and `auth` is its sole bridge.
Additionally `S_Auth_Affil` is row-stochastic by construction (each author's
affiliation row sums to 1), giving it a dominant near-uniform direction — a
Markov low-pass filter that may wash out the differences between communities.

**Singular value spectra of the three social relations have not been measured.**
That would test it directly.

### Why this matters

`auth` and `affil` are the social domain. The research question is about
communities spanning semantic *and* social structure. If the social side has
duplicate communities at K ≥ 4, the socio-semantic bridge is being measured on a
side that isn't discriminating.

Also: `U_scales[affil][i]` and `[j]` for a collinear pair are computed
independently and both enter the collapse-entropy check as separate masses. They
are one direction counted twice, so the check overstates distinctness.

---

## 5. `U_prob` statistics are contaminated by dead entities

**Status: Established. Affects the objective function.**

### Definitions

An entity is **live** in a slice if it has ≥1 non-zero entry in ≥1 active
relation touching its facet (union across relations —
`build_presence_masks()`, Module 1 §1.8). **Dead** otherwise.

Dead entities arise from chunk12's design: entity index maps for
auth/affil/journ and all semantic facets are **global** across time slices; only
article indices are slice-specific. An author publishing only in 2023–2025 still
occupies a zero column in T1. This is deliberate — identical dimensions across
slices are what will make T1↔T2 community comparison possible.

### The contamination

`U_prob` is L1 row-normalised. A dead entity at the initialisation floor —
`[1e-4, 1e-4, 1e-4, 1e-4]` — normalises to `[0.25, 0.25, 0.25, 0.25]`. Perfectly
uniform, purely by arithmetic. Indistinguishable from a genuinely balanced
entity.

Measured, `U_prob` row-max at λ_l1 = 0:

| Facet | Dead % | All entities (mean/median) | Live only (mean/median) |
|---|---|---|---|
| affil | 53.0% | 0.509 / 0.341 | 0.699 / 0.675 |
| auth | 54.0% | 0.713 / 0.679 | 0.754 / 0.667 |
| core_atom | 53.3% | 0.634 / 0.452 | 0.842 / 0.938 |
| core_child_he | 59.0% | 0.609 / 0.471 | 0.808 / 0.860 |
| cousin_he | 51.7% | 0.605 / 0.356 | **0.873 / 0.968** |
| fringe_atom | 46.8% | 0.613 / 0.527 | 0.831 / 0.893 |
| journ | 30.0% | 0.651 / 0.694 | 0.786 / 0.799 |
| parent_he | 60.6% | 0.642 / 0.507 | 0.851 / 0.918 |
| art | 0.0% | 0.813 / 0.883 | identical (sanity check ✓) |

`cousin_he` median moves 0.356 → 0.968. `art` at 0% dead shows an exact match,
confirming the mask logic.

### Why it matters beyond diagnostics

Module 3 §4 computes `baseline = np.mean(model_probs)` and Part B's monopoly
scan over `U_prob`. Both feed `sociological_penalty` — half the Pareto
objective. Presence masking was implemented (ticket 60) and threaded in; **it is
worth verifying it is applied at every `U_prob` consumption point**, since these
numbers suggest some paths still average over dead rows.

---

## 6. `lambda_l1` targets the wrong axis, and the problem it addresses does not occur

**Status: Established.**

### Three distinct defects, two fixed

| Defect | Status |
|---|---|
| **Scaling** — `sparsity_loss` was a raw sum (~150–172), comparable to recon_loss only because ~11,000 entries were being added | **Fixed** (ticket 73). Now a mean, ≈0.0134. |
| **Range** — `[1e-4, 1.0]` contributed at most 1.7% of recon_loss; inert across its span | **Fixed** (ticket 77). Widened to `[0.75, 74.6]`. |
| **Wrong axis** — the term measures column concentration, the goal is row concentration | **Not fixed. Not fixable by tuning.** |

### The axis problem

`U_norm` columns have unit L2 norm **by construction** (the scale-invariance
step divides each column by its own L2 norm; the removed magnitudes go into
`U_scales` and fold into `Z_scaled`).

For a vector with fixed L2 = 1, its L1 sum ranges from 1 (all mass on one entry)
to √N (perfectly even). So minimising L1 under fixed L2 means **concentrating
within the column**.

| | Fixed | Concentration means |
|---|---|---|
| **Column** (what the term does) | one community | few *entities* carry that community |
| **Row** (what the hairball concern is about) | one entity | few *communities* per entity |

Rescaling the term (ticket 73) and rescaling its weight (ticket 77) are both
multiplications outside the sum. Neither changes which entries are summed or how
they relate. A column-wise quantity stays column-wise.

### Measured consequence — evacuation, not pruning

Sweep over λ_l1 ∈ {0, 0.75, 5, 25, 74.6}, C6/K=4, **live entities only**.

The **L1/L2 ratio per row** is the diagnostic: for a row of length K it ranges
from 1.0 (fully committed to one community) to √K = 2.0 at K=4 (fully smeared).
Scale-invariant, so it measures membership *shape* independent of mass.

| Facet | λ=0 | λ=0.75 | λ=5 | λ=25 | λ=74.6 | Row L1 sum, λ=0→74.6 |
|---|---|---|---|---|---|---|
| affil | 1.35 | 1.37 | 1.62 | 1.82 | 1.83 | 0.179 → 0.129 |
| art | 1.24 | 1.22 | 1.25 | 1.74 | 1.78 | 0.317 → 0.176 |
| auth | 1.26 | **1.03** | 1.34 | 1.68 | 1.59 | 0.082 → 0.028 |
| core_atom | 1.17 | **1.05** | 1.31 | 1.40 | 1.45 | 0.073 → 0.013 |
| core_child_he | 1.23 | 1.18 | 1.47 | 1.58 | **1.87** | 0.131 → 0.024 |
| cousin_he | 1.14 | 1.14 | 1.19 | 1.46 | 1.71 | 0.131 → 0.019 |
| fringe_atom | 1.19 | 1.23 | 1.73 | 1.66 | 1.67 | 0.101 → 0.013 |
| journ | 1.26 | 1.35 | 1.54 | 1.70 | 1.70 | 0.359 → 0.286 |
| parent_he | 1.17 | **1.12** | 1.27 | 1.55 | 1.67 | 0.117 → 0.064 |

Row mass falls 5–7× while shape flattens toward 2.0. Entities are **hollowed
out, not focused**. The apparent "spreading" in `U_prob` is division of near-zero
by near-zero.

`core_child_he` shows a phase transition: 0% of live entities near the smeared
end at λ=25, **97.6% at λ=74.6**. The widened range reaches a regime where a
core semantic facet collapses wholesale.

**Nuance worth preserving:** `auth`, `core_atom`, `parent_he` show a genuine dip
toward 1.0 at λ = 0.75 specifically (bolded above) — real commitment, in three of
nine facets, at the very bottom of the range, reversing by λ = 5. Evidence that a
*correctly targeted* term could work.

### The hairball does not occur

Fraction of live entities with L1/L2 > 1.8 at λ_l1 = 0:

```
affil 0.032   art 0.040   auth 0.000   core_atom 0.000
core_child_he 0.018   cousin_he 0.014   fringe_atom 0.007
journ 0.000   parent_he 0.016
```

Mean ratios 1.14–1.35 — comfortably in the committed half. **The model commits
on its own** from NNDSVD init and reconstruction pressure alone.

### Recommendation

Set `lambda_l1 = 0` and drop it from Optuna's search **for the toy run**. Revisit
at 22k, where more entities per community make smearing more available. The
L1/L2 ratio is already the right detector.

**Consequence to confront:** with λ_l1 gone, Optuna searches one dimension, and
that dimension (`lambda_z_offdiag`) acts on Z, which §1 shows is largely gauge.
Two questionable parameters is a weak outer loop; one is close to none.

### Also noted, untested

`sparsity_loss` averages over **all** entries including dead ones. ~50% of the
term pushes floor values on entities with no data toward zero — pointless work
that dilutes the live signal. If a sparsity term is ever reintroduced, mask it to
live entities.

---

## 7. UDSR — history and why it should not return in its original form

**Status: Decided (not returning). Reasoning recorded.**

**What it was.** Ubiquity-Discounted Sparsity Regularization — a per-entity
weight on the L1 term:

```python
discount = 1.0 / (1.0 + ubiquity_scores[facet])
sparsity_loss += torch.sum(u_mat * discount)
```

Intent: ubiquitous entities face less pressure to concentrate, so they can spread
across communities. `lambda_l1` sets *how much* sparsity matters; UDSR set *to
whom it applies*.

**History.** An early review found the discount direction **inverted** (multiplying
by ubiquity rather than dividing) — fixed. It then vanished entirely in the
rewrite that introduced softplus, joint-anchor SVD, and scale-invariance,
alongside other regressions that were restored. Whether removal was deliberate is
unclear; CLAUDE.md §4.1's recorded rationale is *technical* (discontinuous ops
break autograd) and does not cover UDSR, which is a smooth weighted L1.

**Why it should not return as designed.** It keyed on `idf_global`, which measures
**ubiquity** (appears in many articles). The actual concern is **breadth**:
entities appearing in articles that themselves span several communities. These
differ:

- 50 articles all in one community → ubiquitous, not broad. Should concentrate.
- 3 articles in 3 communities → not ubiquitous, maximally broad. Should spread.

**`parent_he` demonstrates the gap concretely.** All 63 T1-live parent hyperedges
share **one** idf value, because each appears in exactly one article
(`idf = log(N/(len(arts)+1)) + 1`, and `len(arts) = 1` for all). IDF carries zero
information there. Expected by construction — parent hyperedges are per-sentence
structures that essentially never recur.

IDF is a usable ubiquity signal only where entities can recur: meaningful for
`core_atom`/`fringe_atom`, partial for child/cousin hyperedges, useless for
`parent_he`.

**What already addresses breadth.** Module 3 §4 Part A propagates article
community-membership backward through the topology and Part B computes the
Shannon entropy of the resulting distribution — that entropy **is** the breadth
measure. It is second-order, article-derived, and already implemented.

Two limitations: it *scores* rather than *steers* (outer loop, no gradient), and
it depends on `U_prob['art']`, so using it inside the loss would be circular.

A non-circular version would compute breadth from raw matrices alone, before
fitting — but defining "dispersed" without reference to communities is the hard
part.

**Also noted:** IDF *is* still applied, in the data rather than the model —
chunk12's `build_anchor_csr` weights anchor entries by `log1p(tf) × idf`. That
reduces a ubiquitous term's pull on reconstruction but does not shape how its
loadings distribute across communities.

---

## 8. Reconstruction-space coherence metric — proposed, not validated

**Status: Open. First test was mis-designed; corrected test pending.**

### The proposal (originated with Gemini)

Replace the Z-diagonal coherence check with a measurement in **reconstruction
space** — what the model predicts — rather than **parameter space** — the numbers
it stores.

Per relation joining f1 to f2:

```
numerator   = ‖ U[f1] · diag(Z) · U[f2]ᵀ ‖_F
denominator = ‖ U[f1] ·      Z  · U[f2]ᵀ ‖_F
ratio       = numerator / denominator
```

The numerator is the prediction using only within-community pairings; the
denominator is the full prediction. The ratio is the fraction of predicted edge
mass generated within communities — the original design intent.

### Per-community form (what the research question needs)

Three masked variants, all via the trace identity
`‖U1·Z_mask·U2ᵀ‖²_F = Tr(Z_maskᵀ·U1ᵀU1·Z_mask·U2ᵀU2)`:

| Quantity | Mask | Meaning |
|---|---|---|
| `within_k` | only `[k,k]` | k's within-community contribution |
| `involvement_k` | row k and column k | everything k participates in |
| `total` | full Z | the relation's whole prediction |

- `coherence_k = within_k / involvement_k` — k's within-community fraction
- `share_k = within_k / total` — k's share of the relation. **Comparing this
  across relations for fixed k gives the community's coupling profile** (e.g.
  community 2 is author-article heavy, community 3 semantics-heavy).

`Σ_k share_k` will **not** equal 1 — the rank-1 terms are not orthogonal. How far
it departs from 1 measures interference between communities.

### Invariance argument, and its gap

The **denominator is provably invariant** under any W (shown in §1).

The **numerator is not**, in general: `diag(W⁻¹·Z·W⁻ᵀ) ≠ W⁻¹·diag(Z)·W⁻ᵀ`.
Extracting a diagonal does not commute with an arbitrary transformation.

The argument that it holds anyway: non-negativity restricts reachable W to
(a) permutations — relabelling communities, leaving the diagonal intact — and
(b) positive diagonal rescalings, which the scale-invariance step already
canonicalises. **This is asserted, not proven.**

### Why the first test failed — design error, recorded so it isn't repeated

Test A compared the metric across Runs 1/3/4/11, described as "differing only in
gauge." They do not. Their `math_loss` values are 0.7745 / 0.7818 / 0.7592
(and Run 11 at 4.21, unconverged). A gauge transformation preserves
reconstruction *exactly*; different fit means **different local minima**.

So the test measured "does the metric agree across four different solutions" —
a question about non-convexity, not gauge. Alignment cosines were correspondingly
poor: 10 of 12 non-reference community slots below the 0.7 threshold.

**Gauge-invariance is a property of a formula, testable directly** by applying a
known W to saved tensors — no refitting, no confounds.

### What the mis-designed test did reveal

**The new metric tracks the old one closely** — 0.989/0.967, 0.999/0.996,
0.984/0.940, 0.052/0.065, 0.000/0.000 across every row. Either they measure the
same thing (in which case the swap gains nothing), or Z happens to be
near-diagonal in those runs so they coincide. **Partially reconciled by §14:**
for Run 1 (production), permutation contamination is rare (1 of 9 relations,
and that one is a harmless free-label leaf facet — see §14), so the close
agreement there reflects genuine near-diagonal Z, not both metrics being fooled
by the same mislabeling. For Run 3/Run 4, roughly half the relations *were*
mislabeled (§14), so their reported low scores in this table are a mix of
genuine incoherence and uncorrected permutation — not cleanly interpretable
without re-running through §14's diagonal-share-after-realignment correction.

**Anchors remain pinned at ~0.999.** Module 3 §3 evaluates anchors only, so
swapping the metric does *not* fix `coherence_pen` being inert. The real spread
lives in social and deep-semantic relations: `S_Art_Auth` 0.317, `S_Auth_Affil`
0.572, `M_Fringe_Cousin` 0.772.

**Different initialisations find different communities**, not relabelled versions
of the same ones. That is a stability finding about the whole model, independent
of Z — and it is exactly what Module 4 §S5's dual-track consensus exists to
measure. Expect poor consensus when it runs.

---

## 9. Determinism

**Status: Established, partially fixed.**

Two runs with identical seed and configuration diverged qualitatively — one
converged at epoch 1092, the other hit the 2000 ceiling. Cause: multi-threaded
BLAS reductions are non-associative in floating point (`(a+b)+c ≠ a+(b+c)` under
rounding), so summation order varies with thread scheduling. Differences compound
over 1000+ Adam steps and can land either side of the `rel_change < 1e-4`
early-stopping threshold.

`torch.set_num_threads(1)` makes **same-process** repeats bit-identical (8-decimal
match on `math_loss`, zero difference across the whole loss history).
`torch.use_deterministic_algorithms(True)` without `warn_only` was tested and
completes on CPU — unverified on GPU.

**Residual caveat:** cross-process reproducibility is approximate. Same seed,
separate interpreter launches gave 925 vs 976 epochs, `recon_loss` differing by
~3e-4. Losses track closely, so trajectories are not diverging — it is the
early-stopping trigger firing at slightly different points. Relevant for Module 4
§S5, which compares across separate fits.

---

## 10. Open questions, roughly by consequence

1. **Is the reconstruction-space metric gauge-invariant?** Testable directly on
   saved tensors under permutation and positive-diagonal W. **Partially
   answered by §12/§13**: `Z_scaled[k,k]` (the metric's core quantity) is
   proven invariant under the `U_scales`-moving transformation specifically.
   Still open under general rotational W, and needs the §14 permutation
   correction applied before use on any given relation. The originally-planned
   direct permutation/diagonal-W test on saved tensors was never run — the
   session that would have run it instead found the permutation phenomenon by
   a different route (§14), which answers a closely related but not identical
   question.

2. **Why do the new and old coherence metrics agree so closely?** **Answered
   for Run 1** by §14: rare permutation contamination there (1/9 relations, a
   harmless leaf-facet case) means the agreement reflects genuinely
   near-diagonal Z, not both metrics being fooled identically. Not resolved
   for Run 3/Run 4, where roughly half of relations were mislabelled and the
   comparison needs redoing through §14's realignment.

3. **What should Module 3 §3 measure?** Sharpened by §15: it currently
   contributes exactly zero regardless of what it measures, since
   `TARGET_COHERENCE=0.50` is never approached (closest: 0.744). Options
   unchanged (move to non-anchors, retain as a much-lower-floor pathology
   guard, or drop) but now informed by §14 — non-anchor relations in
   production are mostly *consistent* with the anchor labelling (§14's
   facet-agreement check), which is a different property from being
   *coherent* per the diagonal-share metric, so moving the check there isn't
   automatically an improvement.

4. **What causes the `affil`/`auth` collapse at K ≥ 4?** Two hypotheses
   refuted (§4, §11). The topology hypothesis (peripheral chain,
   row-stochastic `S_Auth_Affil`) is still untested. **Not the same
   phenomenon as §14's permutation finding** — collinearity is within-facet
   (two communities identical on one facet), permutation is cross-relation
   (labels not matching between two facets) — but both concern `U`/`Z`
   consistency and may be worth investigating together.

5. **Does off-diagonal Z earn its place at 22k articles?** Unchanged, scope-limited by construction.

6. **Does Optuna have anything meaningful to tune?** **Sharper now.** With
   `lambda_l1` fixed at 0 (§6, decided) and `coherence_pen`/`collapse_pen`
   both identically zero across the tested grid (§15), Optuna's only live
   lever is `lambda_z_offdiag`, acting on a quantity §1 shows is largely
   gauge, against an objective (`semantic_pen`) that varies in a narrow band.
   This is close to "no working outer loop" rather than "a weak one."

7. **Should an orthogonality penalty be added?** Unchanged from before, but
   now doubly motivated: it would also give Optuna a lever that acts on a
   *determined* quantity (`U_norm`, per §12) rather than an undetermined one
   (`U_scales`) or a gauge-dependent one (`Z`) — see open question 6.

8. **Chunk 12 coverage.** Unchanged — `meta_set ∩ edge_set ∩ sqlite_set`
   discard rate not yet measured.

9. **Does the SVD-reference-frame effect (§14) hold at fewer anchors?** Only
   C6 tested. C1 (1 anchor, weak — 63 non-zeros) is the natural next case;
   prediction is weaker self-consistency there.

10. **Does per-community domain balance hold, once measured on a determined
    basis?** Currently unanswerable — see CLAUDE.md §4.18 and FINDINGS §13.
    Two `U_scales`-based measurements of the *same* model gave opposite
    directions (social-dominant vs semantic-dominant); neither is trustworthy.

---

## 11. Hypotheses tested and refuted — do not retry without new evidence

| Hypothesis | Refuted by |
|---|---|
| Facet size explains collinearity | `affil` (31 live) and `auth` (212 live) collapse at the same K; `journ` (14 live) never collapses under production init |
| Phase-2 propagation manufactures `auth` collinearity | At epoch 0, all nine facets sit at 0.51–0.80 under both init strategies; `auth` is not an outlier |
| Routing (R) vs capacity (G) is decidable by c_u↔Z correlation | The test assumes Z placement is informative; §1 says it isn't. Self-undermining |
| Runs 1/3/4/11 differ only in gauge | Different `math_loss` (0.7745/0.7818/0.7592) means different local minima |
| `lambda_l1` controls membership concentration | It controls column concentration; row-max *falls* as λ rises, and row mass evacuates |
| Higher `lambda_z_offdiag` approximates freezing Z diagonal | At λ_z = 1.0 it distorts community mass (0.470/0.095/0.261/0.174) rather than zeroing off-diagonals |
| IDF is a usable breadth signal | `parent_he` has one idf value for all 63 live entities |
| `U_scales` is a trustworthy mass measure | §12: exact algebraic counterexample — scaling a `U_pos` column by *c* with compensating division in `Z_pos` leaves every loss term unchanged while `U_scales` changes by exactly *c*; confirmed empirically (CV up to 1.2 across differently-initialised, comparably-fit runs) |
| Raw `U_scales` correlates positively with facet size (my own error, corrected mid-session) | Measured: negative correlation in 5 of 6 configs — larger facets have *smaller* raw `U_scales`, opposite of the assumption behind dividing by `√N_f` |
| `M_Cousin_Parent`'s `share_sum=0.0000` (Run 4) is evidence off-diagonal Z carries a whole relation | §14: it was a clean label permutation (structure score 24.5); diagonal share is 0.989 after realignment. Retracted as the illustrative example in §2, replaced with `S_Art_Auth` (genuinely low both before and after realignment) |
| Multiplying `U_scales` by `√live` (rather than dividing) is the correct size correction | Tested directly via the concentration ratio `‖U_norm‖₁/√live`: observed values cluster 0.18–0.69, nowhere near 1 (which `×√live` assumes) nor near `1/√N` (which raw/`÷√live` implicitly assumes). Neither guessed exponent matches the data; superseded by §12's finding that the whole quantity is undetermined regardless of exponent |
| Relations sharing a facet in Run 1 disagree on its labelling | The one apparent disagreement (`M_Fringe_Cousin` vs. `M_Cousin_Art`/`M_Cousin_Parent` on `cousin_he`) is a free relabelling on the leaf facet `fringe_atom`, which nothing else constrains — not a real inconsistency (§14) |

---

## 12. `U_scales` is not a mass measure — it is an undetermined free direction

**Status: Established. Supersedes §4.3's original guidance in CLAUDE.md.**

### The argument

`U_pos[f]` is the optimiser's actual parameter for facet *f* (post-clamp, per
§4.8/ticket 74). Each iteration:

```
U_scales[f][k] = ‖U_pos[f][:,k]‖₂
U_norm[f][:,k] = U_pos[f][:,k] / U_scales[f][k]
Z_scaled[i,j]  = Z_pos[i,j] · U_scales[f1][i] · U_scales[f2][j]   (per relation)
```

Take one facet *f*, one community *k*, one constant *c*. Multiply
`U_pos[f][:,k]` by *c*. In every relation where *f* is the row facet, divide
`Z_pos[k, :]` by *c*; where *f* is the column facet, divide `Z_pos[:, k]` by
*c*. Then:

- `U_norm[f][:,k]` is unchanged — the *c* divides straight back out.
- `Z_scaled` is unchanged — the *c* introduced by `U_scales[f][k]` in the
  `scale_matrix` product is cancelled by the compensating division of `Z_pos`.
- `sparsity_loss` (computed on `U_norm`) is unchanged.
- `z_offdiag_loss` (computed on `Z_scaled`) is unchanged.
- `recon_loss` (computed on `U_norm`, `Z_scaled`) is unchanged.
- `U_scales[f][k]` changes by exactly *c*.

Nothing in the loss constrains this direction. `U_scales` can sit at any
positive value the optimiser's trajectory happens to leave it at.

### Test 1 — algebraic, exact

Applied directly to Run 1's saved converged tensors, facet `auth`, community 0,
*c* ∈ {0.5, 2.0, 10.0}:

| c | max Δ‖recon‖_F | Δ sparsity_loss | Δ z_offdiag_loss | `U_scales[auth][0]` ratio |
|---|---|---|---|---|
| 2.0 | 0.0 | 0.0 | 0.0 | 2.000000 |
| 0.5 | 0.0 | 0.0 | 0.0 | 0.500000 |
| 10.0 | 0.0 | 0.0 | 5.25e-10 (float noise) | 9.999998 |

Exact to float precision at every tested *c*. `U_scales` scales by exactly *c*
while every term in the loss is unmoved.

### Test 2 — empirical, across differently-initialised runs

Runs 1, 3, 4 (C6, K=4, all converged, `recon_loss` 0.7747 / 0.7817 / 0.7589 —
comparable fits). Hungarian-aligned to Run 1 as reference (alignment was
already known to be poor for most slots — only Run3 k∈{1,3} and Run4 k=3 clear
the 0.7 cosine bar, consistent with §8/§14's finding that these runs found
substantively different community structure, not just relabelled versions).

CV of `U_scales[f][k]` across the three runs, over all 36 (facet, ref_k)
cells: mean 0.360, median 0.264, min 0.026, max 1.200, 44% of cells below 0.2.

Large and inconsistent — a facet/community pair with near-identical
reconstruction quality across runs can show `U_scales` varying by more than
its own magnitude. Consistent with Test 1's algebraic result.

### What survives and what doesn't

The transformation leaves `U_norm` — and everything computed from it —
completely unchanged. Only the **overall magnitude** of a column is free; its
**shape** is not.

| Determined (safe to use) | Undetermined (do not use for absolute comparison) |
|---|---|
| `U_norm[f][:,k]` — relative proportions across entities | `U_scales[f][k]` |
| `U_prob[f][a,:]` — an entity's membership distribution | Any "facet *f* contributes X to community *k*" claim |
| `c_u[f]` — collinearity between communities on a facet | Cross-facet mass comparisons (`auth` vs `core_atom`) |
| L1/L2 ratio of a row — membership concentration | Cross-community mass comparisons on raw scale |
| `Z_scaled[i,j]` — see §13 | — |

This means §4 (`affil`/`auth` collinearity, via `c_u`), §5 (`U_prob`
contamination), §6 (the L1/L2 sparsity analysis) are **all unaffected** — they
operate entirely on shape quantities. What is affected: any claim in this
document or in production code about *how much mass* a facet or community has
in absolute terms via `U_scales`.

---

## 13. What replaces `U_scales` for mass, and the anchor-handling problem

**Status: Open. The anchor-handling problem below is superseded by a facet-level
reformulation (see the update at the end of this section) — kept here as historical record of
why the relation-level framing was abandoned, not as the current design.**

### Why `Z_scaled[k,k]` is determined where `U_scales` is not

`U_norm` columns have unit L2 norm by construction. So the Frobenius norm of
community *k*'s within-community rank-1 term for a relation joining f1→f2:

```
‖Z_scaled[k,k] · u1_k · u2_kᵀ‖_F = |Z_scaled[k,k]| · ‖u1_k‖₂ · ‖u2_k‖₂
                                  = |Z_scaled[k,k]| · 1 · 1
                                  = |Z_scaled[k,k]|
```

And §12's Test 1 showed `Z_scaled` is exactly unchanged under the
`U_scales`-moving transformation. So `|Z_scaled[k,k]|` is a determined
quantity, comparable across relations because every input matrix is
Frobenius-normalised to `‖X‖²=1`.

This is the same quantity Gemini's reconstruction-space proposal (§8) computes
as `within_k`. Two independent lines of reasoning — the algebraic gauge
argument here and the reconstruction-space coherence proposal — converge on
the same measurement. That convergence is itself evidence worth some weight.

**Scope of this determinacy — not a general fix.** `Z_scaled[k,k]` is
determined under the specific transformation §12 tested (independent per-column
rescaling of `U_pos` with compensation in `Z_pos`). It is **not** shown to be
invariant under general rotational indeterminacy (§1) or under per-relation
permutation (§14). Those remain separately open. A relation whose `Z` is a
disguised permutation (§14) will report a near-zero `Z_scaled[k,k]` that is
just as misleading as `U_scales` was — the permutation correction from §14
must be applied before `Z_scaled[k,k]` is trusted as a mass figure.

### Two attribution problems, both needing a stated convention

**Interference — masses don't sum to the total.** `‖X̂‖²_F =
Tr(Zᵀ·U1ᵀU1·Z·U2ᵀU2)` expands into cross-terms between different community
pairs (`U_norm` columns are not orthogonal — see §4's collinearity findings).
`Σ_k |Z_scaled[k,k]|` is not `‖X̂‖_F`. Measured previously (`share_sum`,
pre-permutation-correction): values from 0.00 to 1.9 across relations. A
stated convention is needed — e.g. `share_k = |Z_scaled[k,k]| / Σ_j
|Z_scaled[j,j]|`, a share of the diagonal total rather than of the full
reconstruction.

**Off-diagonal attribution.** `Z[i,j]` (i≠j) couples two communities; crediting
it to one or the other, or splitting it, is a choice. Half-split
(`|Z[k,k]| + ½Σ_{j≠k}(|Z[k,j]|+|Z[j,k]|)`) is the natural default, but per §14
it must only be applied **after** correcting for per-relation permutation —
otherwise mislabelled within-community structure is counted as cross-community
coupling.

### The anchor double-counting problem

Relation-level mass makes domain attribution cleaner in general (a relation is
unambiguously `S_` or `M_`), but anchor relations (`M_Parent_Art`,
`M_Child_Art`, `M_Cousin_Art`) each touch `art` — the hub shared with the
social relations `S_Art_Auth`/`S_Art_Journ`. Configs with **one** anchor (C1,
C3, C4) credit the hub once; configs with **two** anchors (C2, C5, C6) would
credit it twice under a naive "count every anchor's mass toward its labelled
domain" rule, systematically inflating apparent domain-bridging mass in
dual-anchor configs relative to single-anchor ones — which would corrupt
exactly the cross-config comparison the toy corpus exists to support.

**Candidate fix:** each anchor contributes `0.5 / n_anchors` to each side of
the balance calculation rather than a flat `0.5`, so the total anchor budget
per config is constant regardless of anchor count. (An alternative —
weighting by each anchor's own mass rather than by count — was considered and
rejected for this purpose: it reintroduces the count asymmetry it was meant
to remove, since a config with more anchors has more terms contributing
non-trivial weight.)

### Status of the domain-balance measurements taken so far

All domain-balance numbers computed before this section (using `U_scales`,
in any of its three tested normalisations — raw, `×√live`, direct-L1) rest on
an undetermined quantity and should not be treated as evidence of the true
balance in either direction. See CLAUDE.md §4.18 for the practical
consequence (imbalance direction flipped between social-dominant and
semantic-dominant depending on normalisation, on the *same* fitted model —
consistent with, and now explained by, this section's finding).

### Update: facet-level reformulation supersedes the anchor-handling problem above

The anchor double-counting problem above assumes domain is a property of *relations* — under
that framing, an anchor relation (`M_Child_Art`, touching both `art` and `core_child_he`)
genuinely is ambiguous, and the `0.5/n_anchors` fix was the best available patch. **Domain is
a property of facets, not relations, and there is no such thing as a mixed facet.** `art` is
a bibliographic entity — the same kind of thing as `auth`/`journ`/`affil` — and is
unambiguously social; it only looks mixed because it happens to participate in both `S_` and
`M_` relations. A facet-level formulation never has to answer "is `M_Child_Art` social or
semantic," which is the question that forced the anchor problem in the first place. Verified
directly against `RELATION_MAP` for all 6 configs: `art` has exactly 2 non-anchor relations
(`S_Art_Auth`, `S_Art_Journ`) regardless of anchor count, so a facet-level profile is
structurally comparable across single- and dual-anchor configs without any anchor-count
correction — the `0.5/n_anchors` candidate above is **superseded**, not implemented.

**The mechanism, as designed (not yet implemented in `chunk13v9.py`):** two components,
following the pipeline's own established pattern of an in-loop differentiable pressure term
paired with an outer-loop reality check (`z_offdiag_loss`/`collapse_pen`).

- **In-loop, `U_prob`-based.** Per community `k`: `soc_k`/`sem_k` = a weighted mean, over
  active facets in that domain, of `U_prob[f][:,k]`'s mean across facet `f`'s LIVE entities
  (ticket 60 masking is required — dead entities sit at the clamp floor and normalise to
  ≈uniform `1/K`, which would drag every community toward false balance). `r_k =
  soc_k/(soc_k+sem_k+eps)`, `0.5` = balanced; penalty `mean_k(max(0, |r_k-0.5|-TOL)²)`, `TOL`
  a flexibility band (e.g. `0.10` accepts 40/60–60/40) rather than a strict 50/50 target — this
  is a per-community measure, not a global one: a model with one purely-social and one
  purely-semantic community scores maximal penalty on *both*, even though the two would
  average out to a perfectly balanced *model*.
  **A concrete trap caught before it could be built wrong:** `U_norm`'s columns have unit L2
  norm by construction (`U_scales[f] = ‖col‖₂`, `U_norm = u/U_scales`) — porting v8's
  `binding_penalty` formula (`‖U[:,r]‖₂/√N_f`) onto `U_norm` directly would be differentiable
  and gauge-invariant, and **identically constant**, penalising nothing. `U_prob` (which
  varies per row, unlike a column norm) is the correct substitute.
- **Outer-loop, `Z_scaled`-based**, deliberately a *different* space from the in-loop term —
  an audit sharing the in-loop term's own pathway cannot detect that pathway being satisfied
  degenerately (the Problem 2 lesson, §20). Per relation, production's own
  `_relation_community_share` (§17, `within_k/total`); each relation's share vector is
  credited to *both* facets it touches — a true statement ("this relation contributes to both
  endpoint facets' profile"), not double-counting. **Anchor relations are included**
  (considered and reversed from an initial draft that excluded them): excluding them would
  make the measure blind to exactly the social↔semantic binding a genuinely heterogeneous
  community shows up through, and would violate §4.15's rule against special-casing for
  anchor count. Measured (not assumed) how much this choice matters: mean "anchor
  sensitivity" (reading with vs. without anchors, per community) is modest — T1 0.052, T2
  0.061 — but with a real per-cell outlier at `T1/C5/K4` (0.21), worth a caveat rather than a
  reason to switch.

**Weighting (equal-per-facet vs. live-entity-count) — resolved in favour of entity-count
weighting, reversing an initial lean toward the codebase's equal-weighting convention
(`collapse_mass_share`, `membership_share_all_facets`).** Two arguments, one about
measurement semantics and one about optimisation dynamics:
1. Algebraically, entity weighting is the *only* one of the two that is a true,
   sum-comparable population-mass share — it reduces exactly to pooling every live entity in
   the domain and taking one grand mean of `U_prob[:,k]`, independent of facet boundaries.
   Equal weighting is a different construct: an average of per-facet-*type* proportions,
   which by design discards absolute headcount (a facet with 14 live entities counts as much
   as one with 461). This is the standard proportional-vs-equal-allocation distinction from
   stratified sampling, and the mechanism's actual target — domain-level balance, not
   facet-type representativeness — is the population-level question proportional
   (entity-weighted) allocation is designed to answer.
2. **Decisive for the in-loop term specifically.** The gradient of `soc_k` with respect to one
   entity's `U_prob` row is `1/(|S|·N_f)` under equal weighting — facet-size-dependent, so in
   T1 an individual `auth` entity's leverage over the penalty is roughly 1/23rd an individual
   `journ` entity's — versus a uniform `1/N_domain` for every entity under entity weighting,
   regardless of facet. Under equal weighting, gradient descent would find it cheapest to
   satisfy the penalty by reshuffling small facets (`journ`, `affil`) while leaving `auth`
   (the domain's actual dominant population) essentially untouched — a self-inflicted version
   of the Problem 2 gaming lesson, built into the mechanism's own construction rather than
   discovered by an adversarial search.

**Interference — how much of a relation's reconstructed mass isn't attributable to any single
community — is worth interpreting, not just tracked as noise, and it interacts with the
weighting decision above.** Decomposed exactly for one example (`S_Art_Auth`,
`T1/C6/K4`, verified numerically against the cached fit, not derived): total reconstructed
mass `0.035271` (confirmed equal to `‖U1·Z·U2ᵀ‖²_F` computed directly, not just via the trace
identity); diagonal-squared sum `0.032030`; the gap (`0.003241`, i.e. `Σ share_k = 0.908`,
9.2% interference) splits into raw `Z` off-diagonal magnitude (`0.001906` — genuine
cross-community relational coupling, e.g. authors nominally in one community co-authoring
with another's articles at a real rate) and a remainder (`0.001335`) attributable purely to
`c_u=U_normᵀU_norm` not being the identity — i.e. community-loading collinearity, the same
phenomenon [[chunk13-collinearity-legitimacy]] already treats as potentially legitimate
cross-cutting structure rather than a defect. **This matters more, not less, under entity
weighting**: since `auth` now dominates the social-domain reading (vs. getting 1 vote of 4
under equal weighting), whatever interference level lives specifically in `auth`'s own
relations (`S_Art_Auth`, `S_Auth_Affil`) now propagates into the whole social-side reading
almost directly, rather than being diluted by three other equally-weighted facets.

**Checked (this update): how closely the entity-weighted `soc_k` tracks `auth` alone.**
Recomputed `soc_k` (entity-weighted, `u_prob` space) against `auth`'s own raw per-community
profile directly, across all 22 converged E1 cells: `max_k |soc_k - auth_k|` averages `0.082`
(median `0.085`), topping out at `0.164` (`T2/C2/K3`). `auth` is not literally standing in for
the whole social domain — `art`/`affil`/`journ` (25% of the entity-weighted social pool
combined) still move the reading by a real, bounded amount — but it is close enough that
`auth`'s own community assignment is the dominant driver of the social-domain signal under
this weighting, not one contributor among several. `core_atom` (the largest semantic facet,
642 entities) does **not** play a comparable role on the semantic side — see composition
below; the asymmetry is in facet-size concentration, not (yet, separately) tested for
interference-level differences between the two dominant facets.

### D2 re-tabulation under entity weighting (this update, supersedes the equal-weighted pilot as the basis for `TOL`)

Re-ran E1's full grid (`domain_balance_measurement.json`, same 22 converged cells,
`diagnostic_blocks` 1.12.0 — no new fitting, entity weighting was already computed alongside
equal weighting in the original run, this is a re-tabulation of existing output) with entity
weighting as the primary reading, per the plan's D2/D3 gate.

**Domain composition under entity weighting, first — this shapes everything below.** Social
domain: `auth` is 75.2% (T1) / 74.2% (T2) of the entity-weighted pool; `art`/`affil`/`journ`
split the rest. Semantic domain has no comparable concentration: `core_atom` 28.7%/31.6%,
`fringe_atom` 29.3%/26.4%, `cousin_he` 20.1%/17.6%, `core_child_he` 15.9%/17.9%, `parent_he`
6.0%/6.6% (T1/T2) — the two largest semantic facets are close to tied, and no single one
dominates the way `auth` dominates social. This is exactly the risk the original plan named
("`auth` is ~75% of the social domain, so 'social' effectively becomes 'authors'") — now
measured, not hypothetical, and asymmetric between domains rather than a matched effect.

**`dev_k` distribution, entity-weighted, `u_prob` space (the in-loop-relevant reading):**

| | T1 (n=42) | T2 (n=34) |
|---|---|---|
| mean | 0.133 | 0.126 |
| median | 0.113 | 0.112 |
| p90 | 0.241 | 0.249 |
| max | 0.295 | 0.322 |
| % exceeding 40/60 band (TOL=0.10) | 57.1% | 61.8% |
| % exceeding 35/65 band (TOL=0.15) | 42.9% | 26.5% |
| % exceeding 30/70 band (TOL=0.20) | 26.2% | 14.7% |

`z_scaled` (outer-loop) space shows the same pattern more strongly, and diverges between
slices: mean `dev_k` 0.177 (T1) / 0.261 (T2) under V1 (anchors included), similarly under V2.
T2's `z_scaled` reading is markedly worse than its `u_prob` reading (0.261 vs 0.126 mean) —
a two-space disagreement of a kind not yet explained; flagged for E3 rather than resolved here.

**This is materially larger than the equal-weighted pilot (mean ≈0.08, both slices) —
confirmed, not just anticipated.** Two readings of why, both partially true, stated so `TOL`
isn't set on a misread of the mechanism:
1. **Some of this is a real measurement-fidelity gain.** Equal weighting's mean is an average
   over 4 (social) and 5 (semantic) roughly-independent facet-level signals; averaging across
   independent-ish quantities mechanically shrinks variance regardless of whether the
   underlying imbalance is real. Entity weighting removes that averaging on the social side
   specifically (§ above: `auth` alone drives ~92% of `soc_k`'s movement), so at least part of
   the larger spread is the natural consequence of collapsing from ~4 informative facets to
   ~1 — not evidence that communities are "actually" more imbalanced than the equal-weighted
   numbers suggested.
2. **Some of it is not an artifact.** The direction is no longer symmetric the way the
   equal-weighted pilot's was: under entity weighting (`u_prob`, TOL=0.10, T1+T2 pooled, 76
   communities total), 28 skew semantic, only 17 skew social, 31 sit inside the band — a
   ~1.6:1 lean toward semantic-dominant communities that the equal-weighted reading (26/25/25,
   effectively even) did not show. That directional shift is a genuine consequence of
   reweighting, not just added noise, and is worth carrying forward as a finding in its own
   right: under a true population-mass reading, communities lean semantic more often than they
   lean social on this corpus.

**Concrete cases** (entity-weighted, `u_prob`): most skewed social-leaning is `T2/C2/K4`
community 2 (`r=0.822`, `dev=0.322`); most skewed semantic-leaning is `T1/C2/K3` community 1
(`r=0.222`, `dev=0.278`). Both considerably more extreme than the equal-weighted pilot's top
cases (`r=0.745`/`r=0.304`).

**Practical consequence for `TOL` (D3):** a `TOL=0.10` band, read naively off these numbers,
would flag a majority of communities in both slices (57-62%) — too strict to be a useful
in-loop pressure term if applied unmodified, given point 1 above. A `TOL` in the `0.20-0.25`
range keeps the flagged fraction in the same rough neighbourhood (~15-26%) as the
equal-weighted pilot's `TOL=0.10` reading, but that match is a calibration choice, not a
principled derivation — **`TOL` under entity weighting should be set with point 1's
variance-loss effect explicitly in mind, not by porting a band-width tuned on the
equal-weighted numbers.** Left as the next open call (D3), not resolved by this re-tabulation.

**Design and measurement recorded here; implementation not started.** No code changes to
`chunk13v9.py` for this mechanism exist yet — the in-loop/outer-loop split above, the entity
weighting decision, and this re-tabulation are the design and evidence to build from, not a
description of shipped behaviour. See CLAUDE.md §4.18/ticket 82 for the current status line.

---

## 14. Per-relation label permutation — a third source of Z ambiguity

**Status: Established for the phenomenon. Production (Run 1) largely
unaffected.**

### What was found

Auditing a `share_sum = 0.0000` result for `M_Cousin_Parent` in Run 4 (λ_z=0.0,
random anchor init), the full `Z_scaled` matrix was:

```
        col0      col1      col2      col3
row0  1.61e-08  2.83e-08  6.72e-03  2.25e-01
row1  4.37e-08  7.66e-08  1.64e-01  1.17e-04
row2  2.42e-01  4.35e-08  8.85e-09  4.15e-08
row3  1.31e-08  2.02e-01  2.55e-03  2.19e-08
```

Exactly one dominant entry per row and per column (mapping 0→3, 1→2, 2→0,
3→1), everything else at the `1e-7` clamp floor. This is a **permutation
matrix**, scaled. For a permutation *P*:

```
U[cousin_he] · P · U[parent_he]ᵀ = U[cousin_he] · (U[parent_he] · Pᵀ)ᵀ
```

— ordinary within-community structure, computed after reordering one side's
columns. Not cross-community coupling; the community labels on the two facets
simply don't correspond. `share_sum = 0.0000` measured label mismatch, not
absent within-community structure.

This is distinct from §1's rotational indeterminacy: a *global* gauge
transformation `W` is applied identically to `U` across every relation
simultaneously. A per-relation permutation is local — each relation's `Z` can
independently relabel, because each relation has its own `Z` while `U` is
shared. It is not a gauge transformation on the whole model; it is a
possible failure of internal self-consistency between relations that share a
facet.

### Test: how much off-diagonal "mass" is really permutation?

Per relation, per run: find the permutation *P* (via Hungarian assignment on
`|Z_scaled|`) maximising the diagonal sum; report the diagonal share before
and after, and a structure score (smallest matched entry ÷ largest unmatched
entry — ≫1 means a clean permutation, <1 means genuinely mixed coupling).

| Run | converged | math_loss | relations w/ non-identity best permutation |
|---|---|---|---|
| Run 1 (production) | ✅ | 0.7745 | **1 of 9** (`M_Fringe_Cousin`) |
| Run 3 (random anchor Z, λz=0.05) | ✅ | 0.7819 | 5 of 9 |
| Run 4 (random anchor Z, λz=0.0) | ✅ | 0.7588 | 5 of 9 |
| Run 11 (fully random init) | ❌ ceiling | 4.0849 | 9 of 9 — unreliable, not comparable |

Full Run 1 / Run 4 table (`raw → permuted (structure score)`):

| Relation | Run 1 | Run 4 |
|---|---|---|
| M_Atom_Child | 0.925→0.925 identity (2.96) | 0.952→0.952 identity (4.28) |
| M_Child_Art | 0.994→0.994 identity (57.2) | 0.216→0.978 [1,2,0,3] (11.8) |
| M_Child_Parent | 0.940→0.940 identity (3.10) | 0.905→0.905 identity (2.08) |
| M_Cousin_Art | 0.993→0.993 identity (34.0) | 0.019→0.967 [3,0,1,2] (7.62) |
| M_Cousin_Parent | 0.931→0.931 identity (5.47) | 0.000→0.989 [3,2,0,1] (24.5) |
| M_Fringe_Cousin | 0.507→0.803 [2,1,0,3] (1.50) | 0.463→0.930 [2,1,0,3] (6.39) |
| S_Art_Auth | 0.690→0.690 identity (**0.099**) | 0.592→0.620 [1,0,2,3] (**0.043**) |
| S_Art_Journ | 0.998→0.998 identity (106) | 0.999→0.999 identity (121) |
| S_Auth_Affil | 0.846→0.846 identity (0.32) | 0.584→0.970 [1,0,2,3] (4.96) |

`S_Art_Auth`'s structure score stays **below 1** in both runs (0.099, 0.043) —
low both before and after realignment. This is the genuinely-incoherent case,
not a labelling issue, and is the corrected illustrative example for §2 (the
old `M_Cousin_Parent` example was itself a clean permutation, score 24.5 —
retracted there).

### The one Run 1 "inconsistency" is a free leaf-facet relabelling, not real disagreement

Checking whether relations sharing a facet agree on that facet's labelling
(Run 1): `art`, `auth`, `core_child_he`, `parent_he` — all consistent
(identity everywhere). `cousin_he` — `M_Cousin_Art` and `M_Cousin_Parent` both
independently agree at identity (high structure scores, 34.0 and 5.47);
`M_Fringe_Cousin` alone wants `[2,1,0,3]`.

**This is not genuine disagreement.** `M_Fringe_Cousin` connects `fringe_atom`
→ `cousin_he`, and `fringe_atom` is a **leaf** — it appears in no other
relation. Nothing constrains how `fringe_atom`'s own columns are labelled, so
the permutation is fully absorbed there; it does not require re-permuting
`cousin_he` (which two other relations already pin at identity) and breaks
nothing. Four facets are leaves in this topology (`core_atom`, `fringe_atom`,
`journ`, `affil`); three landed at identity by chance, one didn't — consistent
with nothing constraining the choice on a leaf.

**Conclusion: Run 1 (the configuration that would actually run) has zero
genuine cross-relation labelling inconsistencies.**

### Why production init is more self-consistent — SVD as a reference frame

Anchors get `Z = diag(S_new)` from the SVD; non-anchor relations get random
dense `Z_pos` regardless of init strategy. Yet 6 of 7 non-anchor relations in
Run 1 still landed at identity. The anchors appear to act as a **shared
reference frame** — fixing their labelling propagates through the shared `U`
matrices and pulls the rest of the metagraph into alignment. Runs 3/4, where
anchor `Z` is also randomised, lose that anchor and roughly half the relations
drift into independent (but often still internally clean — see the structure
scores) labellings.

This is a substantive argument for keeping the SVD anchor initialisation
beyond its role in reconstruction quality: it appears to gauge-fix the whole
model, which is why Run 1's metrics are more directly interpretable than
Run 3's or Run 4's.

### Practical implication for interpretation

Within one fitted, production-initialised model, community *k* has a single,
determinate meaning — it is column *k* of each shared `U[f]`. **Interpretation
should be done by describing which entities load on a community, not by its
numeric index** — an index is free to permute on leaf facets, so "the fringe
atoms about X" is a stable description, "fringe community 2" is not
guaranteed to be. For non-leaf facets shared across relations, the index is
pinned by whichever relation constrains it (per the consistency check above),
and cross-relation statements like "cousins from community *i* couple with
parents from community *j*" are directly readable off `Z_scaled[i,j]` — this
was never blocked by the permutation finding, only the *assumption that i=j
by default* was.

### Untested

Only C6 has been examined. Configs with fewer anchors (C1: 1 anchor,
`M_Parent_Art`, 63 non-zeros) are the natural place to check whether the
reference-frame effect weakens — a single, weaker anchor should provide less
gauge-fixing pressure. Not yet run.

---

## 15. `sociological_penalty` is effectively `semantic_pen` alone across the tested grid

**Status: Established.**

`sociological_penalty = collapse_pen + coherence_pen + semantic_pen`. Tested
on converged production fits, C1–C6 × K∈{2,4}, `lambda_l1=0` (per §6),
`lambda_z_offdiag=0.05`. All 12 cells converged, no ceiling hits.

| Config | K | norm. entropy | collapse_pen | weakest_mean | coherence_pen | semantic_pen | sociological_penalty |
|---|---|---|---|---|---|---|---|
| C1 | 2 | 0.809 | 0 | 1.000 | 0 | 0.0925 | 0.0925 |
| C1 | 4 | 0.877 | 0 | 1.000 | 0 | 0.0686 | 0.0686 |
| C2 | 2 | 0.910 | 0 | 0.999 | 0 | 0.0962 | 0.0962 |
| C2 | 4 | 0.995 | 0 | 0.984 | 0 | 0.0669 | 0.0669 |
| C3 | 2 | 0.945 | 0 | 0.850 | 0 | 0.1263 | 0.1263 |
| C3 | 4 | 0.944 | 0 | 1.000 | 0 | 0.0538 | 0.0538 |
| C4 | 2 | 0.995 | 0 | 0.986 | 0 | 0.1322 | 0.1322 |
| C4 | 4 | 0.982 | 0 | **0.744** | 0 | 0.0772 | 0.0772 |
| C5 | 2 | 0.889 | 0 | 0.989 | 0 | 0.1073 | 0.1073 |
| C5 | 4 | 0.965 | 0 | 0.964 | 0 | 0.0651 | 0.0651 |
| C6 | 2 | 0.861 | 0 | 0.998 | 0 | 0.0852 | 0.0852 |
| C6 | 4 | 0.994 | 0 | 0.979 | 0 | 0.0639 | 0.0639 |

`collapse_pen` and `coherence_pen`: **exactly 0.0 in all 12 cells, variance
0.0 exactly.** `sociological_penalty` matches `semantic_pen` bit-for-bit
everywhere. `semantic_pen` itself ranges only 0.0538–0.1322 (variance 0.00059)
— a narrow band even as the sole surviving term.

**The inputs are not constant** — `normalized_entropy` spans 0.808–0.995,
`weakest_mean` spans 0.744–1.000 (lowest at C4/K=4, a single-anchor config).
Both thresholds (`0.60`, `0.50`) simply sit below where any production fit in
this grid lands; closest approach across the whole grid is `weakest_mean =
0.744` against a `0.50` floor. Neither term has fired even once.

**Two different reasons for the same symptom, not one:**

- `coherence_pen`'s input (`weakest_mean`) is gauge-dependent — it reads
  anchor `Z` diagonals, pinned near 0.99 by SVD initialisation regardless of
  data (§1, §14). A high value doesn't establish coherence; it establishes
  that anchors were initialised diagonally. Recalibrating the threshold
  wouldn't fix this — it would just make an untrustworthy quantity fire more
  often.
- `collapse_pen`'s input is mass computed via `U_scales`, shown undetermined
  in §12. Independently, its threshold doesn't express its stated target
  (CLAUDE.md §4.4). Either problem alone would already explain some of the
  silence; both apply.

**Practical consequence:** NSGA-II's second Pareto axis has been `semantic_pen`
alone — itself a narrow-range single quantity averaging Part A ("do semantics
follow their articles") and Part B (the Doxa/monopoly check) — while described
in code and in prior documentation as three independent terms.

**Redesign options, not decided:**

1. Rebuild `collapse_pen` on `Z_scaled`-based mass (§13) with a max-share
   formulation (CLAUDE.md §4.4/ticket 80) instead of entropy.
2. Decide what `coherence_pen` should measure, given anchors are inert by
   construction — candidates discussed in §10 open question 3: move to
   non-anchor relations (untested whether they're any less gauge-dependent —
   §14 suggests non-anchor relations in production are mostly *consistent*
   with anchors, which is a different property from being *coherent*), retain
   as a pathology guard with a much lower floor, or drop it.
3. Split `semantic_pen` into Part A and Part B and report/weight them
   separately — not yet done; unknown whether the 0.054–0.132 range is really
   Part A doing all the work.

---

## 16. Permutation-consistency across the full grid, and a threshold for trusting a relabeling

**Status: Established for the toy corpus. Must be re-derived before the 22k-article run —
see caveat at the end.**

### What this closes

§14 established that C6/K=4 (production init) has zero genuine cross-relation labelling
disagreement, and flagged the rest of the grid as untested — in particular whether C1's
single weak anchor (`M_Parent_Art`, 63 non-zeros) provides enough gauge-fixing pressure to
keep it clean too. This section runs that test: C1–C6 × K∈{2,4}, production settings
(`lambda_l1=0.0`, `lambda_z_offdiag=0.05`), via `diagnostic_blocks.fit_or_load()`
(cached, bit-identical fits — no re-derivation risk from re-fitting).

Scripts: `diagnostic_scripts/permutation_consistency_sweep.py` (the base sweep, produces
`diagnostic_results/permutation_consistency_sweep.json`) and
`diagnostic_scripts/permutation_driver_analysis.py` (the confidence/threshold analysis
below, produces `diagnostic_results/permutation_driver_analysis.json`).

### A bug caught mid-analysis, and the fix it produced

The first version of the sweep flagged a facet as "disagreeing" whenever *any* relation
touching it reported a non-identity permutation. This over-counts: if the relation's other
endpoint is a **leaf** (touches no other relation), that leaf absorbs the permutation for
free — nothing external constrains its labelling, so its choice says nothing about whether
the *shared* facet is actually inconsistent (this is exactly §14's `fringe_atom` /
`M_Fringe_Cousin` finding, generalised into an automated check for the first time here).

**Fix:** a relation only counts as informative about a facet's consistency if the facet at
its *other* end is also non-leaf. Applying this fix changed C6/K=4's own result from
"disagreeing" (wrongly) back to "clean" — exactly reproducing §14's hand-verified
conclusion for that cell. The script asserts this as a standing regression check
(`assert not c6k4["genuine_disagreement_facets"]`) — if this fails after a future edit to
the method, the fix has regressed.

### Raw result: 5 of 12 cells flagged, 16 facet-instances

| Config | K | Anchors | Converged | Genuine disagreement (post-fix) |
|---|---|---|---|---|
| C1 | 2 | 1 | ✅ | none |
| C1 | 4 | 1 | ✅ | art, auth, core_child_he, parent_he |
| C2 | 2/4 | 2 | ✅ | none |
| C3 | 2 | 1 | ✅ | none |
| C3 | 4 | 1 | ✅ | art, auth, core_child_he, cousin_he |
| C4 | 2 | 1 | ✅ | none |
| C4 | 4 | 1 | ❌ ceiling | art, auth, core_child_he, cousin_he (tentative) |
| C5 | 2 | 2 | ✅ | none |
| C5 | 4 | 2 | ✅ | art, auth |
| C6 | 2 | 2 | ✅ | core_child_he, parent_he |
| C6 | 4 | 2 | ✅ | none (cross-check vs §14: PASS) |

C1 and C3 (both single, weak anchors) are clean at K=2 but break at K=4 — partial support
for "weak anchor → less gauge-fixing pressure, worse at higher K." Not a clean rule: C5 (two
anchors, same count as C6) also breaks at K=4, while C2 (also two anchors) stays clean, and
C6 itself flips between K=2 (breaks) and K=4 (clean) — anchor count alone does not predict
this; something about each config's specific topology interacts with K non-monotonically.
Per SESSION_PROTOCOL §C.9, reported and not chased further this session.

### Distinguishing real mislabelling from genuine mixed coupling

The raw disagreement count conflates two different things a non-identity permutation can
mean, and only one of them should ever be "corrected":

- **Mislabelling** — the relation's own diagonal isn't dominant as stored, but *some*
  relabelling makes it so. A pure labelling artifact; correcting it recovers real structure.
- **Genuine mixed coupling** — no relabelling produces a clean diagonal, because the
  relation's entities really do couple across community boundaries (e.g. articles published
  in a journal, or using semantic elements, more associated with another community).
  Nothing to correct here; forcibly diagonalising it would erase real signal.

`structure_score = min(matched) / max(unmatched)`, already computed by the sweep, is exactly
the discriminator: **> 1** means every within-community entry beats every cross-community
entry for that relation (case 1, correctable); **< 1** means at least one cross-community
entry is larger than some within-community entry (case 2, not correctable — do not touch).

Concrete example of each, pulled directly from the cached tensors (C6/K=4):

```
S_Auth_Affil (C4/K=2), stored:        after relabelling [1,0]:
[[0.000  0.238]                       [[0.238  0.000]
 [0.151  0.000]]                       [0.000  0.151]]
diag_share 0.00 -> 1.00, structure_score ~625,000  -- pure mislabelling, correctable

S_Art_Auth (C6/K=4), best permutation IS the stored one (identity), yet:
[[0.0032 0.0114 0.0045 0.0149]
 [0.0039 0.0125 0.0239 0.0006]
 [0.0100 0.0124 0.1048 0.0000]
 [0.0045 0.0218 0.0100 0.1468]]
structure_score = 0.133  -- no relabelling helps; real cross-community coupling
```

### Threshold derivation and sensitivity

`1.0` is the natural boundary — it's where the metric's own claim changes character (every
pair separated, vs. at least one pair not). Checked this isn't just a theoretical nicety:
across all 94 relation-instances in the sweep, scores climb to **0.648** then jump straight
to **1.098** — a real gap in the data, nothing lands between them.

Full sensitivity sweep (`permutation_driver_analysis.py`), re-classifying the 16
disagreement-facet-instances at 16 thresholds from 0.05 to 25:

| Threshold | Real conflicts | Facet-instances |
|---|---|---|
| 0.05 | 10 | (inflated — includes relations with score as low as 0.02, i.e. clearly mixed, wrongly trusted) |
| 0.10 | 4 | |
| 0.30–0.50 | 3 | |
| **0.65–1.50** | **2** | **C4/K4 `core_child_he`, C5/K4 `art` — stable across this entire span** |
| 2.00–10.00 | 1 | C4/K4 drops (its own driving scores, 1.6 vs 2.5, straddle this range) |
| 20.00+ | 0 | even C5/K4 drops (weakest driver `S_Art_Journ`, score 19.8) |

`0.65–1.50` is the widest, flattest plateau in the whole sweep — the chosen threshold (1.0)
sits centrally in it, not near either edge. The one conflict that *is* threshold-fragile
(C4/K4) is also the one non-converged cell — non-convergence and threshold-fragility
pointing the same direction is a second, independent reason to treat it as unreliable rather
than a real finding. **C5/K4's `art` conflict is the most robust result in the sweep** —
stable from 0.65 up to just under 20, a ~30× span.

### Net effect: 16 flagged facet-instances → 1 solid real conflict — RETRACTED, see below

Applying the confidence filter at the derived threshold: **14 of 16 dissolve** — they were
flagged only because a genuinely-mixed (structure_score < 1) relation was in the mix, not
because confidently-labelled relations actually disagree with each other. Only 2 survive,
and one of those is confounded by non-convergence. **C5/K4's `art` facet is the only fully
solid real reference conflict found anywhere in the grid**, driven by: both anchors
(`M_Child_Art`, `M_Cousin_Art`) agreeing with each other at identity (structure scores 16.6,
55.6) against `S_Art_Journ` (structure score 19.8, non-identity).

**RETRACTED (later session, same investigation thread).** This claim came from a script
that re-filtered by confidence without re-applying leaf-exclusion. `S_Art_Journ`'s other
endpoint is `journ` — a leaf in every config (touched only by `S_Art_Journ`) — so per this
section's own leaf-exclusion rule, `S_Art_Journ` should never have counted as a legitimate
dissenting vote against the two anchors regardless of its confidence. Re-run with both
filters combined correctly: **zero confirmed real conflicts survive anywhere in the C1–C6 ×
K∈{2,4} grid.** The corrected, combined-filter check is now the implementation used in
`chunk13v9.py`'s `evaluate_dimensional_collapse` (ticket 82) — see §17.

**On anchor-preference as a tie-break (raised when discussing ticket 82's correction step):**
proposed rule — prefer the child-hyperedge anchor over the cousin-hyperedge anchor when the
two anchors disagree, justified by `core_child_he`'s greater topological centrality (touched
by 3 relations vs `cousin_he`'s 2 in a 2-anchor config) and the `core_`-prefixed naming
convention marking it as the primary semantic backbone. **This grid never actually exercises
that rule** — in every 2-anchor config (C2, C5, C6) and at both K, the two anchors always
agree with each other whenever both are confidently labelled (structure_score > 1). The one
solid conflict found is anchor(s)-vs-non-anchor-relation, not anchor-vs-anchor. The rule
remains reasonable to keep as a documented fallback, but is untested by this data; what the
data *does* support directly is "trust anchor-touching, high-confidence relations over
dissenting non-anchor ones, and trust it more when multiple anchors agree."

### Practical implication for ticket 82

The per-relation correction machinery ticket 82's `Z_scaled`-based mass measure needs is
substantially simpler than the raw 5/12-cells / 16-facet-instance count implied: apply a
Hungarian relabelling only to relations with `structure_score > 1.0` (or the anchor's own
relation, as reference, if a genuine anchor-vs-anchor conflict is ever found); leave
low-structure-score relations untouched, reading their diagonal as-is (real mixed coupling,
not an error). A general graph-consistency solver is not needed for this grid — a one-hop,
confidence-gated correction covers every case found.

### Caveat — must be re-derived at 22k-article scale, not assumed

Same category as tickets 73/77/78's toy-corpus calibrations. The 0.648→1.098 gap and the
5/12 disagreement rate were both measured on this corpus's specific sparsity (anchors as
thin as 63 non-zeros). A denser 22k-article corpus could fill in that gap, shift where a
natural threshold sits, or change how often genuine mixed coupling occurs at all — re-run
`permutation_consistency_sweep.py` and `permutation_driver_analysis.py` against the full
corpus before reusing `1.0` or trusting the disagreement-rate figures above at scale.

---

## 17. Implemented: `evaluate_dimensional_collapse` rewritten on `Z_scaled`, tickets 79/80/82

**Status: Established. Live in `chunk13v9.py`, verified.**

Closes tickets 79 and 80. `evaluate_dimensional_collapse` no longer takes `U_scales_out` —
it computes community mass from `Z_scaled` (relation-level, permutation-corrected) and
penalizes on max-share instead of normalized Shannon entropy. Three decisions were
triangulated externally before implementation; all three are recorded here as the
evidentiary basis, not just asserted.

### Correction criterion: `structure_score > 1.0`, not `chunk13v3.py`/`v4.py`'s
`diagonal_mass/hungarian_mass < 0.7`

Both criteria were implemented and run head-to-head across all 94 relation-instances in the
C1–C6 × K∈{2,4} grid (a fair, apples-to-apples comparison requires asking both tests "is a
*non-trivial* — non-identity — relabelling warranted?"; an earlier draft of this comparison
skipped that guard and produced a spurious 64/94 "disagreement" count, corrected before this
result was recorded). **12 real disagreements.** In 8, `v3`/`v4`'s aggregate-ratio criterion
would have applied a correction to a relation independently identified elsewhere in this
document (§2, §14, §16) as genuinely mixed — `S_Art_Auth` (3×), `M_Fringe_Cousin` (3×),
`S_Auth_Affil` (2×) — because *some* aggregate improvement is available even when the
corrected result still isn't cleanly separated. `structure_score`'s worst-case criterion
(every pair must separate, not just the sum) correctly declines these. In the remaining 4,
`structure_score` catches a clean correction `v3`/`v4`'s fixed threshold narrowly misses —
`C5/K4 S_Art_Journ` is the clearest case: `structure_score=24.4` (unambiguous separation
after correction) but `v3v4_ratio=0.715`, just above the 0.7 cutoff, so it wouldn't correct
despite the fix being available and clean.

### Scope: leaf-unconditional, non-leaf gated on confirmed disagreement — §16's "1 solid
conflict" retracted (see §16 above)

Combined leaf-exclusion and confidence-filtering, applied correctly together (the bug that
produced §16's original "1 conflict" claim is documented and retracted there): **zero
confirmed real non-leaf conflicts survive anywhere in the tested grid.** The implementation
therefore does not include a cross-relation consistency-resolution algorithm — each relation
is corrected independently. This is a documented limitation, not an oversight: if a genuine
non-leaf conflict is found at 22k-article scale, `evaluate_dimensional_collapse` will not
detect or resolve it, and the anchor-preference tie-break proposed earlier in this document
remains an inferred-from-zero-confirmed-cases fallback, not a validated rule.

### Mass/share formula: reconstruction-space (`within_k/total`), not diagonal-sum

Both formulations were computed across the full grid. They agree on 11 of 12 cells but
**flip the verdict on `C6/K=2`** (diagonal-sum: `max_share=0.655`, fires; reconstruction-
space: `max_share=0.592`, does not). Traced to a structural blind spot in the diagonal-sum
formula: it always forces the K diagonal entries to sum to 1, regardless of what fraction of
the relation's real signal they represent. `C6/K=2`'s dominant-looking diagonal entries come
substantially from `M_Child_Parent`, whose `interference = 1 − Σₖ share_recon(k) = 0.83` —
83% of what that relation actually reconstructs is genuinely cross-community. Diagonal-sum
share has no way to see this and gives it a full vote; reconstruction-space share
down-weights it in proportion to how much of the relation is genuinely off-diagonal, without
a separate weighting scheme layered on top. This is the basis for adopting
`within_k/total` as the primary measure rather than treating the two as interchangeable
conventions — one formula catches something the other structurally cannot.

**Relation-weighting scheme, revisited in light of this:** the earlier equal-weight /
structure-weight / recon-quality-weight comparison (this section, weighting-experiment work)
assumed diagonal-sum share and found equal-weight defensible, structure-weight rejected
(self-contradictory with its own use as correction gate), recon-weight architecturally
costlier for no demonstrated benefit. Under reconstruction-space share, the interference term
already performs a version of what that weighting layer was trying to achieve — whether it
makes explicit relation-weighting fully redundant is untested, not assumed; equal-weight is
what's implemented, kept for the same reasons as before (no duplicated machinery, no
correction-gate self-contradiction).

### Implementation and verification

`evaluate_dimensional_collapse(U_final, Z_final, max_share_threshold=0.60,
structure_threshold=1.0)` — new signature (was `U_scales_out, U_final, entropy_threshold,
presence_masks`). `evaluate_complete_solution`'s own external signature (3 call sites:
Optuna objective, §S4 archiver, §S5 stability) was deliberately **not** changed — `U_scales_out`
and `entropy_threshold` remain accepted parameters there but are no longer forwarded to
collapse (vestigial, kept only so no call site needed editing, matching the low-risk-patch
principle CLAUDE.md §3 argues for). The `collapse_score` key in `evaluate_complete_solution`'s
returned dict is unchanged in name, changed in meaning (`max_share`, not
`normalized_entropy`).

Two new helpers, `_hungarian_relabel_relation` and `_relation_community_share`, extracted
as standalone functions per explicit review request (independently testable/auditable, not
inlined). No `presence_masks`/live-entity normalization needed for this computation — every
relation's input is already Frobenius-normalized to `‖X‖²=1` before fitting, which is what
makes relation-level mass comparable across relations without a separate facet-size
correction (this is *why* ticket 60's fix, still correct and still applied elsewhere, is
structurally unnecessary for this specific measure).

Verified post-patch: `py_compile` clean. Real `evaluate_complete_solution()`, called on
cached converged fits across all 12 grid cells, reproduces the independent diagnostic
reimplementation's `max_share` to within 0.02 on all 11 converged cells (consistent with
SESSION_PROTOCOL §F's documented cross-process fit variance, not a discrepancy) — the one
outlier (`C4/K4`, diff 0.109) is explained by that specific cell landing on a different,
non-converged fit this run (`math_loss=0.765392`, matching the ceiling-hit signature seen
earlier in this document, not a new failure mode). Optuna `objective()` smoke-tested
end-to-end (2 trials, `C1`/`K=2`) — both completed, `converged=True`, `user_attrs` populated
correctly, no exceptions anywhere in the call chain.

### Empirical result on the toy corpus

`collapse_pen` now fires on 2 of 12 cells (`C1/K=2`, `C6/K=2`) under reconstruction-space
share — both K=2, both barely over the 0.60 line, consistent with earlier collapse-check
prototyping (see the `collapse_check_zscaled.py` / `collapse_check_weighting_experiment.py`
diagnostic work this thread is built on). `collapse_pen`/`coherence_pen` were previously
`0.0` in all 12 cells (ticket 81) — this is the first time either half of
`sociological_penalty` has fired at all in this pipeline's history.

### Prior art credited

Restores, with a revised criterion and different architectural placement, a mechanism
(`identify_leaf_nodes`, `diagnose_leaf_Z`, `correct_all_leaf_nodes`) present in
`chunk13v3.py`/`chunk13v4.py` and absent from every version since — including the exact same
algorithm (`scipy.optimize.linear_sum_assignment`) applied to the same problem. Unlike the
earlier version, the correction here is **read-time only**, inside `evaluate_complete_solution`
(not a physical mutation of `U`/`Z` after fitting) — so the raw fit stays exactly what the
optimizer produced, recoverable by anyone re-deriving `recon_loss` or auditing the output
directly, and every consumer of `Z_scaled[k,k]`-based mass is structurally required to pass
through the corrected version (same mandatory-single-source-of-truth pattern ticket 69
established for the sociological penalty as a whole) rather than needing to remember to
apply it.

### Toy-corpus caveat

All numeric findings above — the 12 disagreements, the `C6/K=2` flip, the 2/12 firing rate —
are toy-corpus-calibrated, same category as tickets 73/77/78/§16's caveats. Re-derive at
22k-article scale before trusting; do not assume the specific numbers port.

---

## 18. §1's "any invertible W" claim is a bound, not a measured severity — near-separability
check finds strong evidence the fitted model is not actually exposed

**Status: Supported, more strongly than originally reported. Toy-corpus scope. Mechanism
(why) not investigated — this measures whether, not why. Audited and corrected after initial
write-up: Tier 1's result table had two factual errors (now fixed, see the correction
in-place), and Tier 2's original margins (0.5°–3.4°) were an artifact of a flawed monitoring
mask — corrected re-run finds near-total pinning instead, which strengthens rather than
weakens this section's conclusion. See in-place corrections below.**

### The gap in §1 as originally stated

§1's identity — `(U[f1]·W)·(W⁻¹·Z·W⁻ᵀ)·(U[f2]·W)ᵀ = U[f1]·Z·U[f2]ᵀ` for any invertible
`W` — is correct, but it is an **unconstrained** linear-algebra fact. It says nothing about
whether a given non-trivial `W` keeps `U·W` and the transformed `Z` **non-negative**, which
this model requires everywhere (`clamp_(min=1e-7)`, ticket 74). A generic invertible `W`
does not preserve non-negativity. §1's "Established" status is correctly scoped to the math;
it was never checked against whether the freedom is actually *reachable* by a real fitted
solution here. This section is that check.

### Method: near-separability (Tier 1 of a two-tier test; Tier 2 below)

From the NMF identifiability literature (Donoho & Stodden 2003; Arora et al. 2012; a
checkable special case of Huang, Sidiropoulos & Swami 2014's "sufficiently scattered"
condition): if every community has at least one live entity whose membership is nearly pure
(`U_prob` row-max close to 1) in some facet, that is strong evidence against non-trivial
blending being reachable — a blending `W` would have to push that entity's near-zero
entries on other communities negative.

Checked across all 6 configs × K∈{3,4} (12 cells), live entities only
(`build_presence_masks`), at purity thresholds 0.5/0.7/0.85.

**Known confound, addressed directly rather than assumed away:** `evaluate_socio_semantic_
reality`'s Part B ("Doxa hoarding," `MAX_MONOPOLY=0.85`, chunk13v9.py:1160-1211) actively
penalizes `U_prob` row-max > 0.85 for the 4 semantic `target_facets`
(`core_child_he`/`cousin_he`/`core_atom`/`fringe_atom`) — but only when the entity's
*structurally propagated* signal (via topology, not `U_prob` itself) is itself spread across
communities; niche/low-connectivity entities in those facets face no pressure regardless of
purity, and `auth`/`affil`/`journ`/`art` are not touched by Part B at all. Each purity
witness's degree (nnz across every active relation touching that facet) was recorded
alongside its value specifically so an absent witness could be checked against this — a
missing witness among high-degree entities in the 4 confounded facets is expected and
uninformative; a missing witness among low-degree entities, or anywhere in the 4
unconfounded facets, is not explained by Part B and is a real signal.

### Result

| Threshold | Clean facets (`auth`/`affil`/`journ`/`art`) | Confounded facets (4 semantic) |
|---|---|---|
| 0.5 | 56/56 (100%) | 46/48 (96%) |
| 0.7 | 53/56 (95%) | 44/48 (92%) |
| 0.85 (= `MAX_MONOPOLY` itself) | 49/56 (88%) | 41/48 (85%) |

Even at the strict 0.85 bar — the model's own hoarding threshold — 85–88% of every
(config, K, facet) combination has a near-pure witness for *every* community, and a large
share of witnesses hit `max_u_prob = 1.000` exactly. Many witnesses are also high-degree
relative to their own facet (e.g. `art`, the most-connected facet in the topology, is
cleanly assigned in nearly every cell with witness degree 60–80) — stronger evidence than a
low-degree entity being pure by default, since a well-connected entity staying pure despite
pressure from multiple relations is harder to achieve by coincidence.

**Correction (caught in a later audit pass, recorded here rather than silently fixed): the
original text of this subsection stated the 0.85-bar failures were "7 of ~104
confounded-facet cells; none in the clean facets" and named 3 recurring `fringe_atom` cells.
Both claims were wrong** — verified directly against `near_separability_check.json`. The
correct picture:

**Failures at the 0.85 bar: 7 of 48 confounded-facet cells, AND 7 of 56 clean-facet cells
(14 total, not 7).** The "clean" bucket (`SEMANTIC_CONFOUNDED_FACETS`'s complement in the
script) is actually 5 facets, not 4 — it includes `parent_he` alongside `auth`/`affil`/
`journ`/`art`, and **3 of the 7 clean-bucket failures are `parent_he`** (`C2/K3` 0.522,
`C2/K4` 0.824, `C3/K4` 0.650). This matters substantively, not just as an arithmetic fix:
`parent_he` is genuinely outside Part B's `target_facets` (whether to add it is CLAUDE.md
ticket 66, "flagged, not decided") — so by this section's own logic, these are **real,
unconfounded near-separability failures in a semantic facet**, previously missed. The
remaining 4 clean-bucket failures are `auth` (`C2/K4`, `C6/K4`) and `journ` (`C4/K4` —
non-converged, `C6/K3`).

`fringe_atom` is the recurring confounded-bucket facet, but in **4** cells, not 3:
`C1/K4`, `C3/K4`, `C4/K4`, and **`C5/K4`** (previously omitted). Of the 7 confounded-bucket
failures, **4** — not 3 — are not fully explained by the Part B/low-degree confound:
`C1/K4/fringe_atom`, `C3/K4/cousin_he`, `C3/K4/fringe_atom`, `C4/K4/fringe_atom` (the last
non-converged, flag per `SESSION_PROTOCOL` rule 4 accordingly).

### What Tier 1 alone does and doesn't establish

Near-separability is a **sufficient, not necessary** condition — its presence is strong
practical evidence against reachable blending, but its absence in a handful of cells doesn't
prove blending *is* reachable there, and its presence everywhere else doesn't formally prove
it *isn't*. A decisive answer requires Tier 2: a direct feasibility search for a non-trivial
orthogonal `W` (community-mixing rotation, deliberately excluding the already-known/accepted
rescaling freedom of ticket 79) that preserves non-negativity for a real fitted solution.

### Tier 2: direct rotation-feasibility search

For each fitted `(U_final, Z_final)` (= `U_norm`, `Z_scaled` — valid to test directly per
§4.2's equivalence, since non-negativity in this representation implies it in `U_pos`/`Z_pos`
and vice versa, positive column rescaling cannot flip a sign), searched for the largest
Givens rotation `W(θ)` — mixing exactly two communities `i,j`, applied to `U_final[f]` for
every active facet and `W(θ)ᵀ·Z_final[rel]·W(θ)` for every active relation simultaneously
(a shared `W` has to work for every facet/relation at once, since the community axis is
shared across the whole fit) — before any monitored entry goes negative. Checking every
community pair one at a time is a **complete local check**, not a shortcut: the `K(K-1)/2`
pairwise generators span the full tangent space of the orthogonal group at the identity, so
sweeping all pairs covers every possible small rotation, not a subset of them. Coarse scan
(0.5°–90°) then bisection per pair per direction; same grid as Tier 1 (6 configs × K∈{3,4}).

**Methodological history — two corrections, the second reversing the first's conclusion.**
The very first run used an absolute tolerance (reject any monitored entry below −1e-4) and
returned 0.00° margin for all 54 pairs, uniformly — flagged at the time as too clean to
trust. Diagnosis then: many entries sit at the model's own zero-floor (`clamp_(min=1e-7)`)
next to a large entry in the same row, so any nonzero rotation angle nudges the floor entry
past −1e-4 at a vanishingly small angle regardless of whether real structure is rotatable.
**Fix applied at the time**: only monitor entries carrying ≥1% of their column's/relation's
own max value before rotating (`MEANINGFUL_FRAC=0.01`); everything below that threshold was
excluded from the feasibility check. That produced the 0.5°–3.4°-margin result originally
reported here.

**A later audit pass found the `MEANINGFUL_FRAC` fix itself was flawed, in a way that
invalidates the 0.5°–3.4° numbers.** Measured directly (`C6/K4`'s `auth` facet): the
column-relative 1% mask monitored only **18.3%** of entries — averaged across the whole grid,
78–84% of `U` was excluded from every cell's feasibility check. Critically, the excluded
entries are not randomly distributed — they are disproportionately the near-zero entries in
a near-separability witness's *other* communities, which is exactly what a blending rotation
pushes negative first (that's the entire mechanism Tier 1 relies on). A concrete case
confirmed this is not hypothetical: one witness row had `U_prob=0.071` in a non-primary
community (`U_norm=0.00192`, a real, non-floor value, not clamp noise) — excluded from
monitoring solely because it sat below 1% of *that column's* max. The relative mask was
silently exempting the constraint-binding entries from the check whose margin it was supposed
to measure.

**Corrected re-run**: replaced the column-relative mask with an absolute one, calibrated
directly against the model's own clamp floor rather than an arbitrary fraction. Measured the
`U_norm` value distribution first rather than guessing a threshold: literal floor-clamped
entries cluster tightly at ≈1.5e-7 (35% of all entries in the facet checked); a further 44%
sit in `[1e-6, 1e-4)` — real, non-floor values. Set `FLOOR_EXCLUDE=1e-6` (monitors everything
above the literal floor cluster) and tightened `TOL` from 1e-4 to 1e-8 accordingly (the old
1e-4 tolerance was up to 100× larger than many of the now-monitored entries themselves, and
would have let them go substantially negative undetected).

**Result: 53 of 54 pairs are pinned below 0.05°, most at exactly 0.000° in at least one
direction** (max margin found anywhere in the grid: 0.57°, `C2/K4` communities `(1,3)`) —
not the "0.5°–3.4°, bounded but real" freedom previously reported. Checked this isn't a
numerical-precision artifact before trusting it: only 3/54 pairs are pinned at exactly 0.000°
in *both* directions; 33/54 have neither direction at exact zero, and 18/54 show real
directional asymmetry between `+margin` and `−margin` — the pattern is structured, consistent
with genuine constraint-binding, not a uniform floating-point collapse. **The 0.5°–3.4°
figures previously in this document are superseded and should not be used** — they measured
feasibility of a masked subproblem that excluded the entries that actually determine
feasibility, not feasibility of the real system.

**Scope of what this establishes:** a **local** (first-order/infinitesimal) feasibility bound
around the fitted solution — it characterizes the immediate neighborhood, not whether some
large, distant rotation could loop back to a different globally-feasible point (see the
distant-rotation extension below — re-run under the same corrected mask, confirming no such
point exists anywhere tested). Within that scope, the corrected finding **strengthens, not
weakens**, §18's practical
conclusion: genuine local blending freedom is not merely small, it is essentially absent
almost everywhere tested — a materially cleaner result than "narrow but real," and one that
now agrees with, rather than contradicts, the very first (pre-`MEANINGFUL_FRAC`) run's
0.00°-everywhere result, which in retrospect was closer to correct than the "fix" that
superseded it.

**Distant-rotation extension (same method, answers the "harder, different question" above —
folded in here rather than filed as a separate finding, since it's a direct follow-up on the
same search, not a new test).** Worth answering because of a consequence of §1's exact
identity: any rotation, local or distant, that stays feasible has **exactly** the same
reconstruction loss as the original fit — unlike a different-seed/different-basin
disagreement (§1's own Run 2 vs Run 4 example: `recon_loss` 0.7593 vs 0.7591, four decimals
apart, opposite anchor-coherence structure), which loss *can*, if unreliably, distinguish. A
distant feasible rotation of the *same* solution would be invisible to any loss-based
selection, by construction, not merely in practice — worth checking whether one exists before
treating "pick the lowest loss" as a safeguard against this specific failure mode.

Swept the full relevant range (`W(θ+180°)=−W(θ)`, so ±90° covers everything distinct for a
single-pair rotation) at fine resolution for all 54 pairs, checking for a *second*,
disconnected feasible region beyond the local cage. **Methodological self-check applied
before trusting the result, same discipline as the tolerance correction above:** the first
pass used a 1° step and came back with zero islands — but several local margins measured
above are themselves sub-degree, so a 1°-step grid could structurally miss an island of
comparable width. Re-ran at 0.05° (36× finer); same result, and the recovered cage widths at
this resolution independently match the bisection-based margins above (e.g. `C5/K4`
communities `(0,1)`: `−0.65°/+0.75°` here vs `−1.03°/+0.55°` from the separate bisection
search) — two independently-implemented searches agreeing is real cross-validation, not a
repeated bug. **Result: 0 of 54 pairs have more than one feasible region, at either
resolution.** No evidence of a distant, loss-invisible alternate rotation anywhere in the
grid, within the single-pair-rotation family tested.

Scope limits on this extension specifically: only single-pair rotations were swept, one at a
time — a joint rotation moving several community pairs simultaneously (a much
higher-dimensional space) was not searched, so this doesn't rule out a distant feasible point
reachable only by a combined move. And 0.05° resolution, while 36× finer than the first pass,
is still a discrete grid — an island narrower than that step could in principle still be
missed, though something that narrow would carry little practical weight even if found.

**Caveat raised on the audit pass, now resolved by a corrected re-run:** this extension
originally inherited the same flawed `MEANINGFUL_FRAC` mask the bisection search above has
since been corrected for, and its own cross-check ("recovered cage widths… independently
match the bisection-based margins") had only been validated against the now-superseded
0.5°–3.4° numbers, not the corrected ≈0° ones. The caveat predicted the "0 islands" verdict
would plausibly survive the fix anyway, since a stricter, more-monitored feasibility check can
only shrink or preserve a feasible region, not create a new disconnected one out of nothing.

**Re-run under the identical corrected mask (`FLOOR_EXCLUDE=1e-6`, `TOL=1e-8`) confirms the
prediction.** Same design otherwise — 0.05°-step sweep across the full ±90° range, all 54
pairs, all 12 cells. **Result: 0 of 54 pairs show more than one feasible region — every pair
has exactly one contiguous feasible run.** No distant, loss-invisible alternate rotation
exists anywhere in the tested grid, now checked under the same monitoring standard as the
local test above. The run boundaries are, as expected, tighter than the original (superseded)
run reported — mostly within a few hundredths to ~0.15° of zero on the open side, several
pinned at machine precision on both sides — consistent with, and a more precise version of,
the corrected bisection margins above. Scope limits from the original design are unchanged:
only single-pair rotations were swept (a joint multi-pair rotation was not searched), and
0.05° remains a discrete grid resolution.

Diagnostic scripts: `diagnostic_scripts/rotation_island_search.py` (**superseded methodology
— kept on disk for the historical record, not for its numbers**) and
`diagnostic_scripts/rotation_island_search_v2.py` (the corrected, absolute-floor-mask
version — authoritative). Results in `diagnostic_results/rotation_island_search.json`
(superseded) and `diagnostic_results/rotation_island_search_v2.json` (authoritative).

### Practical implication (Tier 1 + Tier 2 combined)

On this corpus, at production settings (`lambda_l1=0.0`, `lambda_z_offdiag=0.05`), the
combined evidence is that Layer 1b's blending freedom is **essentially absent locally, not
merely narrow** — not the unconstrained "any invertible `W`" freedom the raw linear-algebra
identity permits, and this is now a cleaner, stronger conclusion than the "narrow but real"
framing this section originally reported (see the corrected Tier 2 result above — the earlier
0.5°–3.4° margins were an artifact of a flawed monitoring mask, not a real feasibility bound).
The distant-rotation extension, re-run under the same corrected mask, closes the remaining
gap in this conclusion: the absence of freedom is not merely a property of the small local
window checked by the bisection search — no disconnected, loss-invisible alternate rotation
was found anywhere in the full ±90° range either, across all 54 pairs.
The model's substantive outputs (`U_prob` community membership, `Z`-based within/between
coupling) for one specific reported/selected fit are, if anything, *more* trustworthy against
this specific failure mode than previously stated. The ordinary caveat still applies
regardless: cross-fit/cross-seed comparisons still require relabeling correction
(§14/§16/§17) — that is a different problem (§19), not addressed by this section either way.
`fringe_atom` (Tier 1's recurring failure, now confirmed as 4 cells, not 3 — see the Tier 1
correction above) is worth extra scrutiny before relying on its community assignments across
different fits, and the previously-overlooked `parent_he` clean-bucket failures deserve the
same caution despite sitting outside Part B's confound. Toy-corpus-calibrated like every other
finding in this document — re-derive at 22k-article scale before trusting; a denser corpus
could plausibly tighten or loosen these margins in either direction.

Diagnostic scripts: `diagnostic_scripts/near_separability_check.py`,
`diagnostic_scripts/rotation_feasibility_search.py` (**superseded methodology — see the
corrected Tier 2 result above; kept on disk for the historical record, not for its numbers**),
and `diagnostic_scripts/rotation_feasibility_search_v2.py` (the corrected, absolute-floor-mask
version — authoritative for Tier 2's margins). Results in
`diagnostic_results/near_separability_check.json`,
`diagnostic_results/rotation_feasibility_search.json` (superseded), and
`diagnostic_results/rotation_feasibility_search_v2.json` (authoritative).

### Re-derivation on post-08-27-fix data (tickets 86/87 Stage 0e) — verdict held, if anything tighter

All three scripts above (`near_separability_check.py`, `rotation_feasibility_search_v2.py`,
`rotation_island_search_v2.py`) were re-run unmodified on the current post-fix pickles and
compared against the pre-fix (2026-08-19–21) results the numbers above were derived from:

| | old (with-repository) | new (post-fix) |
|---|---|---|
| Tier 1, 0.85-bar failures, clean facets | 7/56 | 3/56 |
| Tier 1, 0.85-bar failures, confounded facets | 7/48 | 7/48 |
| Tier 1, 0.85-bar failures, total | 14 | **10** |
| Tier 2 local, pairs pinned below 0.05° | 53/54 | 53/54 |
| Tier 2 local, max single-directional margin anywhere | 0.57° (`C2/K4`, pair (1,3)) | **0.325°** (`C5/K3`, pair (0,2)) |
| Tier 2 distant, pairs with a second feasible region | 0/54 | 0/54 |

Every number moved the same direction as, or held exactly at, the standing conclusion: Tier
1's purity-witness coverage at the strict bar *improved* (14→10 total failures — more cells
now have a near-pure witness for every community, not fewer); Tier 2's worst-case local
rotation margin *shrank* further (0.57°→0.325° — the already-small blending freedom got
smaller, not larger); Tier 2's distant search still finds zero disconnected feasible regions
anywhere. **Nothing here changes §18's conclusion — this is a confirmation, not a
correction.**

One non-content change worth naming: which single grid cell fails to converge shifted —
old flagged `C4/K4` (Tier 1) / `C3/K4`+`C4/K4` (Tier 2 local); new flags only `C3/K4` in
both. Consistent with this project's already-documented single-seed fit sensitivity (ticket
85), not a new instability — retained in the results, not dropped, per `SESSION_PROTOCOL`
§C.4.

Scripts unchanged (re-run as-is, no code modified). Results:
`diagnostic_results/near_separability_check.json`,
`diagnostic_results/rotation_feasibility_search_v2.json`,
`diagnostic_results/rotation_island_search_v2.json`.

---

## 19. A community's domain-skew "identity" is not reproducible across independently-trained
seeds — confirmed with the pipeline's own §S5 alignment, not just a weaker proxy

**Status: Established. Distinct from §18 — this is problem 1a (relabeling across
independently-trained fits), not 1b (local rotation freedom within one fit); the two should
not be conflated.**

### Question

Does a community's apparent domain character (soc-leaning vs sem-leaning, read off
`Z_scaled`) survive across independent re-fits (different random seeds) of the same
(config, K), or does it reshuffle? Answering this requires *aligning* communities across
seeds first — community index is only meaningful within one fit (§14).

### First pass — weaker alignment, informal

Initially measured with a single-facet (`art`-only) Hungarian match on column-similarity —
the simplest alignment available, not the pipeline's own method. Result: ~45% of
(config, K, community) triples had a sign-consistent domain-skew reading across 5 seeds
(1000/2000/3000/4000/5000), C1–C6 × K∈{3,4}. A concern was raised in discussion: could this
weak alignment be *understating* true stability, making the model look less reproducible
than it really is? Never written up formally — recorded here directly against the corrected
measurement below, per the practice of stating a correction explicitly rather than leaving a
superseded number implicit.

### Corrected measurement — the pipeline's own §S5 alignment

Re-ran the identical design, but aligning seeds with `run_dual_track_stability_analysis`'s
actual method (Module 4 §5): all facets stacked, two independent tracks — Track A (JSD,
probability-space) and Track B (magnitude-weighted cosine) — each matched via
`linear_sum_assignment`. Implemented as *faithful reuse*, not reimplementation:
`diagnostic_blocks.s5_dual_track_alignment` calls `row_normalize`/`col_normalize`/
`row_wise_cosine_similarity` directly off the loaded `chunk13v9.py` module.

**The concern was refuted in the opposite direction from how it was raised.** Stability did
not improve under the stronger method — it was slightly *lower*:

| Method | Stability (K∈{3,4} pooled, C1–C6) |
|---|---|
| `art`-only alignment (informal first pass) | ~45% (19/42) |
| §S5 Track A (JSD) | 38% (16/42) |
| §S5 Track B (weighted cosine) | 40% (17/42) |

**New finding, only visible with two independent tracks:** Track A and Track B — two
different, both-reasonable alignment criteria — agree with each other on only 42/48 (88%) of
seed pairs. In 12% of pairs, the pipeline's own best method disagrees with itself about which
community is which. `C3/K4` is the extreme case (1/4 pairs agree between tracks; one of its 5
seeds also did not converge — discount accordingly, `SESSION_PROTOCOL` rule 4).

**Per-config (Track A), same qualitative pattern as the informal first pass, not improved by
the stronger method:** `C1` 0/7 (0%), `C2` 6/7 (86%), `C3`/`C4`/`C5` ~29% each, `C6` 4/7
(57%). Denser, more anchor-supported topologies (`C2`, `C6`) remain more stable than the
single-anchor `C1` — consistent with §4.15's independent finding about `C1`'s anchor
weakness — but the absolute stability level is not rescued by better alignment anywhere.

### Practical implication

The hypothesis that a weak alignment method was the *cause* of apparently low domain-skew
reproducibility is refuted, not confirmed — if anything, the weaker method appears to have
*overstated* stability, having fewer ways to detect a genuine relabeling than the real
9-facet, two-track method does. **A community's domain-skew identity is not a robust,
reproducible property across independent re-fits of this corpus, even measured by the
pipeline's own best available tool for the job.**

**Calibration against chance, added on a later audit pass (worth stating, since the raw 38%/
40% figures alone don't say how far above "random" that is):** "stable" requires all 5 seeds
to share the reference seed's sign — under a fair-coin null with no real signal at all, the
probability of the 4 non-reference seeds all matching by chance is `(1/2)⁴ = 6.25%`. The
observed 38–40% is **roughly 6× the chance rate**, so domain-skew identity is not
*non-reproducible* in the sense of carrying no signal — there is real, detectable
consistency, just far short of reliable. This doesn't overturn the section's conclusion (both
readings agree the underlying property is not robust enough to lean on across fits), but it
sharpens how the headline number should be read: "far above chance, far below reliable," not
"close to random."

This sharpens, rather than softens, the concern originally raised about interpreting
domain-skew labels as substantive ("this community is coauthorship-driven") across anything
other than one specific reported fit — consistent with, and now measured for, the caveat
already stated throughout this document that cross-fit/cross-seed comparisons require
relabeling correction (§14/§16/§17). Toy-corpus-calibrated like everything else here —
re-derive at 22k-article scale before trusting; a denser corpus, with more entities per
community, could plausibly change this in either direction.

Diagnostic script: `diagnostic_scripts/domain_skew_s5_alignment_reproducibility.py` (reuses
the new `diagnostic_blocks.s5_dual_track_alignment`, `BLOCKS_VERSION` 1.8.0); results in
`diagnostic_results/domain_skew_s5_alignment_reproducibility.json`.

### Calibration: the 38–40% figure is real but overstates severity — continuous consensus is
moderate, not low

The stability figures above are a **binary sign-flip on a derived, sometimes near-zero
quantity** (domain-skew) — a flip from `+0.02` to `−0.02` counts identically to a flip from
`+0.3` to `−0.3`. Extracted the **continuous** similarity scores production's own §S5
actually computes and uses (`mean_js_sim`, `weighted_cos_sim` per facet — not just the
alignment permutation), via a new `diagnostic_blocks.s5_dual_track_consensus`
(`BLOCKS_VERSION` 1.9.0), on the identical fits/pairs.

**Mean global consensus across all 12 cells: 0.667 (Track A) / 0.670 (Track B), range
0.52–0.78.** This is a real, moderate number — not "communities barely correspond" (near 0)
and not "essentially the same solution every time" (near 0.9+). The binary metric's alarm
level was overstated, as suspected; the underlying reproducibility is real but partial.

**The same configs rank the same way under both measures** — reassuring, not contradictory:
`C1` and `C3/K4` (the latter non-converged, discount accordingly) are weakest under both the
sign-flip and the continuous measure; `C2`/`C6` (dual-anchor) are strongest under both. The
two tests disagree only on how alarming to make the headline number, not on which
configurations are more reproducible.

**New signal, cross-validating an independent §18 finding:** per-facet, `art` and the
anchor-adjacent semantic facets (`parent_he`, `core_child_he`, `cousin_he`) consistently
score high (often 0.8–0.99, especially Track B); `fringe_atom` is consistently the weakest
and most erratic facet in nearly every cell (frequently below 0.5), with `core_atom` and
`affil` also often weak. **`fringe_atom` is the same facet §18's Tier 1 near-separability
check independently flagged as its recurring failure** — two unrelated methods converging on
the same weak point is stronger evidence than either alone.

**Practical implication:** a defensible cross-seed robustness statement for a reported model
is "~0.65–0.70 average alignment consensus, weaker for `C1` and `fringe_atom` specifically,
stronger for `C2`/`C6` and anchor-adjacent facets" — moderate and real, not the near-random
picture the binary figure alone would suggest, but not a clean bill of health either.
Toy-corpus-calibrated, same caveat as above.

Diagnostic script: `diagnostic_scripts/domain_skew_s5_continuous_consensus.py`; results in
`diagnostic_results/domain_skew_s5_continuous_consensus.json`.

### Does a "stable core / variable periphery" story explain the pattern? Partially, and not
in the form first proposed

Raised in discussion: the `fringe_atom`/`C1` weakness above suggested a "core entities are
stable, peripheral entities drift" narrative. Flagged, correctly, as a mere association until
tested against the alternatives it's not the only story consistent with the same aggregate
data — a pure facet-level effect, a pure config-level effect, and unstructured noise
(non-convexity/basin-hopping, already documented directly for this pipeline in §1's Run 2 vs
Run 4) are all equally consistent with an aggregate number alone. Two analyses, one free (re-
decomposing already-collected results) and one new (per-entity, reusing the same cached
fits), were run together to separate them.

**Part A — facet and config effects are both real and additive, not competing.** Decomposing
the per-(config, facet) consensus means: `fringe_atom` is weak in **every** config, including
the strongest ones (`C6`: −0.159 below that config's own mean; `C2`: −0.170) — its weakness
does not get rescued by a good topology. `C1` is weak across 7 of 9 facets, including ones
strong elsewhere (`core_child_he`: −0.166 below that facet's own mean). A simple additive
model (grand mean + facet effect + config effect) explains **78%** of the total variance in
the aggregate numbers — facet identity and config identity are both real, independent, and
together account for most of the pattern (residual 22%: interaction or noise).

**Part B — per-entity correlation, computed *within* each config×facet cell (isolating
entity-level effects from facet/config ones by construction), between individual stability
and two candidate "core-ness" measures:**

| Predictor | Similarity measure | Mean ρ | Cells significant positive | Cells significant negative |
|---|---|---|---|---|
| Degree (connectivity) | Track A (JSD) | +0.011 | 23/104 | 14/104 |
| Degree (connectivity) | Track B (cosine) | +0.053 | 27/104 | 13/104 |
| `U_prob` purity (confidence) | Track A (JSD) | +0.177 | 54/104 | 24/104 |
| `U_prob` purity (confidence) | Track B (cosine) | +0.073 | 39/104 | 31/104 |

**Raw connectivity does not predict individual stability** — near-zero mean correlation,
roughly as many cells significantly negative as positive. The "high-degree entities are more
stable" version of the hypothesis is refuted directly, not just unsupported. **Confidence of
assignment (`U_prob` purity) shows a real but partial, inconsistent signal** — a majority of
cells positive under Track A, but a substantial minority (24/104) significantly negative, and
the effect is much weaker under Track B (39 positive vs 31 negative — close to a coin flip).
**Caveat, stated so it isn't over-read:** purity and Track A's similarity measure are both
derived from the same row-normalized distribution, so part of the purity/Track-A correlation
may reflect mathematical kinship between related quantities rather than fully independent
confirmation — the weaker, more independent Track B result is probably the more trustworthy
read of how strong this effect really is.

**Revised conclusion:** the tempting "central core, drifting periphery" story is not what the
data supports. What actually drives cross-seed reproducibility is mostly **which facet and
which config** (additive, 78% of variance) — not an individual entity's centrality. A
confidence-based (not connectivity-based) partial effect exists but is inconsistent rather
than a reliable law. The unexplained residual (Part A's 22%, and the substantial share of
negative per-entity correlations Part B would not predict under a clean core/periphery story)
is consistent with genuine unstructured noise, though this is not directly provable, only the
natural remaining explanation once the others are accounted for.

**Scope explicitly not tested — record before revisiting this question.** "Centrality" here
means exactly two things: raw degree (nnz count across active relations touching a facet) and
`U_prob` row-max ("purity"/confidence). **No graph-theoretic centrality measure was tried** —
not eigenvector centrality (or PageRank/Katz, which would capture an entity's connections to
*well-connected* neighbours, not just raw degree), and not any centrality measure adapted for
the model's actual hyperedge structure (several facets here are hyperedges —
`core_child_he`/`parent_he`/`cousin_he` — for which plain node-degree is a weak proxy; hyper-
graph-specific centrality notions exist and were not tried). Degree and purity were the
cheapest, most directly available proxies given what was already computed elsewhere in this
document — not a considered claim that they are the *right* or only reasonable operationalisation
of "core". A future revisit of this question should try at least eigenvector/Katz centrality
on the relevant relational graphs before concluding the "core/periphery" story is fully closed
rather than just closed for the two proxies tested here.

Diagnostic script: `diagnostic_scripts/core_periphery_stability_test.py`; results in
`diagnostic_results/core_periphery_stability_test.json`.

---

## 20. Problem 2 — is `collapse_pen` (Z_scaled-based) gameable in a way that distorts
model selection? T1: almost never engages. T2: engages regularly, and the exploit pattern
was directly observed

**Status: Partially established, toy corpus only. `collapse_pen`'s near-total inertness is a
T1-slice/K=4-specific finding, NOT a general property of the model — the T2 replication
below corrects this. Scope is still narrower than it first sounds even where it holds — see
"What this does and does not cover" before generalising.**

### The concern, precisely

`collapse_pen` is the only quantity in this pipeline that is (a) derived from `Z_scaled`
and (b) read by Optuna every trial, feeding directly into `sociological_penalty` — one axis
of the Pareto front Module 3/4 use to select hyperparameters (§4.17/§4.20; ticket 81). The
worry is not that `collapse_pen` itself has an independent flaw — it is a straightforward
threshold check on one scalar (`max_share`, the largest community's reconstructed-mass
share). The worry is that `Z_scaled` can be reshaped by the optimizer without genuine
change to the underlying reconstruction (the same class of scale-invariance slack §4.2
already closed once for `z_offdiag_loss`), and that reshaping would show up as a
`collapse_pen` improvement that doesn't correspond to real de-concentration of which
*entities* belong to which community.

### Method: an independent, non-`Z_scaled`-derived cross-check

Built two new `diagnostic_blocks.py` helpers (`BLOCKS_VERSION` 1.9.0→1.10.0):
- `collapse_mass_share(config_id, U_final, Z_final, K)` — **not a reimplementation**, calls
  production's own `_relation_community_share` per active relation and averages exactly as
  `evaluate_dimensional_collapse` does. Self-checked every run: `collapse_mass_share(...).max()`
  matched `evaluate(...)["collapse_score"]` to `0.00e+00` across every fit in this section —
  confirmed identical, not approximated.
- `membership_share_all_facets(config_id, U_final, presence_masks, K)` — pools live-entity
  mean `U_prob` rows across all 9 facets with equal weight (mirroring `collapse_pen`'s own
  equal-weight-per-relation convention, at facet granularity). Never touches `Z_scaled`, so
  it cannot inherit whatever exploit surface lives in the `Z_scaled` pathway.

Both share vectors sum to 1 over K, directly comparable without extra normalisation. `mass_max`
is literally what `collapse_pen`'s threshold check reads; `mem_max` is the independent ground
truth.

### Test 2 — mass vs. membership at production lambda, and a methodological correction found
mid-session

First pass, C1–C6 × **K∈{2,4}** (the grid used elsewhere in this document), production
settings (`lambda_l1=0`, `lambda_z_offdiag=0.05`): solution-level Pearson r=0.956, Spearman
r=0.944; 2/12 cells showed the exploit *pattern* (`mass_max`≤0.60 while `mem_max`>0.60:
`C2/K=2`, `C6/K=2`).

**Correction, requested and applied before drawing conclusions:** K=2 was flagged as too
low-information to trust — with only 2 communities, both share vectors are mechanically
forced to sum to 1 with a single free value per cell, pinning `mass_max`/`mem_max` close
together near 0.5 regardless of whether the two measures are really tracking the same thing.
Re-run on **K∈{3,4}** (the grid used for the rest of this section): solution-level Pearson
r=0.506 (**p=0.093, not significant**), Spearman r=0.748 (p=0.0051); community-level Pearson
r=0.420 (p=0.0056), down from 0.872. **The K=2-inclusive correlation was substantially
inflated by low degrees of freedom, not by a real tight coupling between the two measures.**
At K∈{3,4}, mass and membership are positively, genuinely associated but only moderately —
one is not a safe stand-in for the other, though no gross divergence was found either. 0/12
cells showed the exploit pattern at K∈{3,4} (max values never approached 0.60 on either
measure: mass_max≤0.48, mem_max≤0.56).

### Test 1 — hand-picked `lambda_z_offdiag` sweep (0.0 → 2.0, 40× production)

Same C1–C6 × K∈{3,4} grid, `lambda_l1=0` fixed, `lambda_z_offdiag ∈ {0.0, 0.05, 0.5, 2.0}`
(48 fits; self-check 0.00e+00 throughout). `collapse_pen` fired in only 1/48 fits (barely:
0.0028, `C1/K=4` at `lz=2.0`), confirming §4.17/ticket 81's standing observation that this
term is essentially inert at production-adjacent settings. Raising `lambda_z_offdiag` mostly
*increased* `mass_max` (10/12 cells) rather than decreasing it — mechanically sensible
(suppressing cross-community `Z` coupling concentrates reconstructed mass onto fewer diagonal
terms), and the opposite direction from what a "cheap `collapse_pen` improvement" exploit
would need. The correlation between how much `mass_max` moved and how much `mem_max` moved
(the two *deltas*, per cell) was weak and not significant: Pearson r=0.478 (p=0.116), Spearman
r=0.357 (p=0.255) — the two measures are loosely, not tightly, coupled under pressure, echoing
Test 2's revised reading. One cell (`C3/K=4`) showed the literal exploit shape in its delta
(`mass_max` −0.070, `mem_max` +0.002) but `collapse_pen` was 0.0 both before and after — a
directionless move inside the sub-threshold region, not evidence of dodging an active penalty.

Test 3 (a `U_scales`-shrinking signature check, next on the original priority list) was
**deliberately not run**: `collapse_pen` is built on `Z_scaled` specifically *because* §4.2
already closes the `U_scales`-inflation/`Z_pos`-shrinkage loophole by construction, and the
above two tests show little live pressure for that mechanism to matter in practice regardless.
Low expected value for the effort; skipped by agreement.

### Test 4 — a real Optuna study (decisive for the tuned-range question, not fully for the
search-process question — see caveat below): `collapse_pen` never fires across the actual
search space

Tests 1–2 only show the term is inert in fits *we* chose. This test let production's own
`NSGAIISampler` search `lambda_z_offdiag ~ log-uniform[1e-4, 1.0]` (the real search space,
`lambda_l1=0` fixed per ticket 78) on its own terms — the literature distinction between "a
proxy is hackable in principle" vs. "the actual search process finds and exploits it" (see
this session's Problem-2 literature summary: Skalse, Howe, Krasheninnikov & Krueger, NeurIPS
2022; van Laarhoven 2017 — citations from memory, flag for verification before formal use).

Design: faithfully mirrors `create_optuna_objective`'s real sequence (`run_inner_solver` →
`evaluate_complete_solution`, same lambda handling, same `NSGAIISampler(seed=MASTER_SEED)`) —
called directly against the loaded module, not reimplemented — but written as a parallel
inline copy (not a call to the production factory function itself) so that `U_final`/`Z_final`
could additionally be captured per trial for the independent `mem_max` reading, which
production's real `objective()` never exposes outward. **Module 4 (adaptive grid dispatch,
Pareto extraction, archiving, §S5 stability) was not exercised at all** — in-memory Optuna
storage, flat `n_trials=20` per cell (user-agreed as representative for this diagnostic pass,
not production's `SCOUT_TRIALS=100`). Grid: C1–C6 × K∈{3,4}, T1 slice, 240 trials total,
23.8 min wall time.

**Two limits on how "decisive" this test actually is for the search-process half of the
question, caught on a later audit pass and worth stating plainly rather than dropping:**
1. `NSGAIISampler`'s default `population_size` is **50** (verified against the installed
   `optuna==4.9.0`); at `n_trials=20` per cell, the run never completes even one generation.
   NSGA-II's selection/crossover machinery — the actual "search process" the reward-hacking
   literature distinction is about — never activates; every trial is effectively drawn from
   the sampler's initial random-population phase. This test shows the proxy is not hackable
   *even under pure random search* across the tuned range — a real and useful result — but it
   does not show whether NSGA-II's optimization pressure specifically would (or wouldn't)
   seek out and exploit a gap, which is the stronger claim "the decisive one" implies.
2. Because a fresh `NSGAIISampler(seed=MASTER_SEED)` is constructed once per cell, all 12
   cells draw the **identical 20 `lambda_z_offdiag` values** (confirmed directly from the
   stored records). The 240 trials are 20 distinct lambda draws × 12 cells, not 240
   independent samples — the pooled correlation's reported p-values below (and in the T2
   replication) should be read as optimistic on that account, though the point estimates
   themselves are unaffected. This pairing is actually useful for the T1-vs-T2 comparison
   specifically (identical lambdas, different data — a clean paired design), just not for
   treating n=220 as 220 independent observations.

**`collapse_pen` fired zero times across all 240 real search trials** — not merely rare, but
never once, across the full lambda range Optuna is actually permitted to explore, on any
config/K combination tested. Consequently the exploit pattern also never occurred (0/240),
trivially — there was never a threshold-adjacent case to diverge on. Pooled correlation across
the 220 converged trials: Pearson r=0.220 (p=0.0010), Spearman r=0.201 (p=0.0027) —
statistically significant given the large n, but numerically weak, consistent with Tests 1–2's
moderate-not-strong reading.

**Conclusion for Problem 2, T1 only:** the original worry — a gameable `collapse_pen`
distorting which hyperparameters look best on the Pareto front — has nowhere to operate in
this regime, because `collapse_pen` essentially never becomes an active constraint for the
real search to exploit in the first place. Flagged at the time as scoped, not a general
exoneration — correctly so, per the T2 replication below.

### Test 4 replication on T2 — correction: `collapse_pen` is not universally inert, and the
exploit pattern actually occurred

Identical design (`diagnostic_scripts/collapse_pen_optuna_study_t2.py`), run against the T2
slice (36 articles, 4,158 total non-zero relational entries, vs. T1's 25/3,323 — §11) instead
of T1, to check whether T1's small article count was itself a factor. 240 trials, 24.2 min.

**`collapse_pen` fired 25/240 times — not rare, and concentrated in a clear pattern:** all 25
fires were at **K=3** (zero at K=4, in any config), and almost entirely in two configs —
`C2/K=3` (12 fires) and `C6/K=3` (10 fires), both clustered at **low** `lambda_z_offdiag`
(mostly ≤0.0031, near the search floor — mechanically sensible: with almost no
cross-community-coupling suppression, mass can genuinely concentrate onto one community, and
`collapse_pen` correctly catches it). `C3/K=3` (1 fire) and `C5/K=3` (2 fires) show a
different, mid-range-lambda pattern. **T1's "essentially never fires" conclusion does not
generalise past K=4 or past the T1 slice — it was correct only for the specific regime it
was measured in.**

**The exploit pattern itself occurred — 4/240 trials, all in `C5/K=3`, and this is the
clearest direct evidence across this whole investigation that the originally-hypothesized
mechanism is real, not just theoretical:**

| `lambda_z_offdiag` | `mass_max` | `mem_max` | `collapse_pen` | converged |
|---|---|---|---|---|
| 0.6351 | 0.484 | 0.630 | 0.0000 | True |
| 0.2915 | 0.496 | 0.613 | 0.0000 | True |
| 0.7579 | 0.480 | 0.634 | 0.0000 | True |
| 0.2137 | 0.580 | 0.604 | 0.0000 | True |

All four sit at the **high** end of `lambda_z_offdiag` (0.21–0.76) — the opposite end of the
range from where `collapse_pen` actually fires in this same config. `collapse_pen` reads a
clean 0.0000 while the independent, non-`Z_scaled` membership measure says a single community
holds 60–63% of real entities. This is directly consistent with the mechanism §2 already
confirmed once for `z_offdiag_loss`: raising `lambda_z_offdiag` can move `Z_scaled`'s readout
toward "looks balanced" via the community-resizing route rather than genuine
de-concentration. Narrow — one config, 4/240 trials — but real, and the first time in this
entire investigation (T1 sweep, T1 mass-vs-membership check, T1 Optuna study) that the exploit
pattern actually appeared under real search pressure rather than being absent.

**Stronger than "4/240" alone suggests, checked on a later audit pass:** each cell drew 20
trials from only 19 *distinct* `lambda_z_offdiag` values (one repeat; see the pseudo-
replication caveat above), and the 4 flagged trials sit at exactly the **4 highest** of those
19 draws in `C5/K=3` — verified directly (`[0.2137, 0.2915, 0.6351, 0.7579]`, the top 4 of the
sorted set, exact match to the exploit-flagged lambdas) — i.e. this is **4 of 4** high-lambda
draws showing the pattern, not 4 scattered flukes among 240 trials. That makes the effect look
deterministic within the sampled range, not incidental — a materially stronger claim than the
raw count conveys. The flip side, in the same direction as caveat #2 above: only 4 of the 19
draws exceed `lambda_z_offdiag=0.2`, so the high-lambda regime itself is thinly sampled — this
strengthens the *pattern* (it's not noise) without yet establishing how far into the
high-lambda range it extends.

**Naming caveat, also worth stating explicitly:** "exploit pattern" implies a shared target —
that `mass_max` and `mem_max` are two readings of the same underlying quantity, one gamed and
one honest. They are not: `mass_max` is a relation-mass-weighted, permutation-corrected share;
`mem_max` is an unweighted, equal-per-facet entity fraction (§20's own "Method" section above).
Some divergence between them is expected by construction, not only under gaming pressure —
there is no tested baseline for how much they'd diverge on a fit with *no* lambda-driven
distortion at all. The pattern above (concentrated at the 4 highest lambda draws, absent
elsewhere) is still the right signal to flag, but "divergence pattern, concentrated at high
lambda" is a more precise description than "exploit pattern" until a no-pressure baseline is
established.

Pooled correlation (210 converged trials): Pearson r=0.331 (p<0.0001), Spearman r=0.525
(p<0.0001) — both higher than T1's (Pearson 0.220, Spearman 0.201), so `mass_max`/`mem_max`
track each other somewhat *better* on T2 even though `collapse_pen` engages more — the two
findings (more engagement, more exploit instances, yet also a tighter overall correlation)
are not in tension: a few real threshold-crossing outliers coexist with an overall tighter
relationship, they are not the same statistic.

**Revised conclusion:** `collapse_pen`'s inertness was T1/K=4-specific, not a property of the
model. At K=3 on the larger T2 slice, it is a real, occasionally-active constraint, and the
specific gaming pattern Problem 2 was worried about has now been observed directly (not just
inferred from a bound), concentrated at high `lambda_z_offdiag` in one config. This raises
the priority of Problem 2 relative to the T1-only round — it is not a closed question, and
the next natural step (not yet run) would be a K=3-focused, `C5`-focused deeper sweep to see
how far the exploit pattern extends.

### Incidental finding: `C4/K=4` shows the worst convergence on BOTH slices, `C3/K=4` joins
it on T2 — relevant to the 22k config-selection question, not to Problem 2 itself

**T1** (Test 4): convergence at K=4 varied sharply — `C4/K=4` 9/20 (45%), `C2/K=4` 14/20
(70%), `C3/K=4` 17/20 (85%), `C1/K=4`/`C5/K=4`/`C6/K=4` all 20/20. Breaking down *which*
`lambda_z_offdiag` values failed: `C2/K=4`'s 6 failures and 10 of `C4/K=4`'s 11 failures
cluster at `lambda_z_offdiag`≤0.0031 — near the search floor, almost no cross-community-
coupling regularisation. **The identical near-zero lambda draws** (same `NSGAIISampler` seed
⇒ same sequence across cells) converged without issue on `C1/K=4`, `C5/K=4`, `C6/K=4` — so
this is `C2`/`C4` specifically struggling at low pressure, not a general low-lambda effect.

**T2** (replication, which also fixed the `epochs_run`-capture gap the T1 script had):
convergence at K=4 is markedly worse overall — `C4/K=4` only **3/20 (15%)**, `C3/K=4` only
**7/20 (35%)**, both markedly worse than their T1 figures; `C1/K=4`, `C2/K=4`, `C5/K=4`,
`C6/K=4` all recovered to 20/20. **Every single T2 K=4 failure hit exactly the 2000-epoch
ceiling** (confirmed directly via the now-captured `epochs_run` field), and failure lambdas
spread across nearly the *entire* search range (`C3/K=4`: [0.0001, 0.29]; `C4/K=4`: [0.0001,
0.64]) rather than clustering at the floor as in T1. Combined with T2's much higher average
epoch counts across every cell (572–1975, vs. T1's implied lower range) this looks like a
**different mechanism from T1's** — T2's larger, denser data genuinely needs more epochs
generally and strains the fixed ceiling broadly, not a lambda-specific fragility.

**Combined verdict:** `C4/K=4` is now the weakest cell on *both* slices by a clear margin
(45%→15% converged, T1→T2) — the single most consistent piece of negative evidence against
any config in this document, worth weighing directly against `C4` for the 22k-article run.
`C3/K=4` is weak specifically at scale (T2) but not at T1 — a slice-dependent weakness, not a
uniform one. These are unrelated to `C1`'s separately-known single-anchor weakness (§4.15) —
three different configs, three different mechanisms, not one story.

### Follow-up: does `Z_scaled`'s FULL diagonal/off-diagonal pattern (not just `collapse_pen`'s
one scalar) track independent structure? Yes, moderately — worse for off-diagonal, and the
decline under `lambda_z_offdiag` pressure is confirmed real, not noise

`collapse_pen` reads exactly one scalar summary of `Z_scaled` (`max_share`). Everything above
establishes that model selection is (T1) or is not (T2/K=3) currently being distorted through
that one channel — it says nothing about whether `Z_scaled`'s raw diagonal/off-diagonal
*values*, which is what §1 and this document's community-coupling interpretation actually
depend on for the article, can be reshaped by the same training-dynamics-driven slack,
independent of whether any outer-loop penalty ever notices.

**Method:** an independent, non-`Z_scaled`-derived reading of "how coupled are community k1
and k2 in relation r" — `data_reading[k1,k2] = U_prob_f1[:,k1]ᵀ @ X_r @ U_prob_f2[:,k2]`, i.e.
how much of the relation's *real raw edges* connect an entity plausibly in k1 to one plausibly
in k2, soft-weighted by membership, never touching `Z` — compared against `Z_scaled`'s own
full K×K reading, both normalised to sum to 1 over the grid (diagonal AND off-diagonal, not
just the max).

**A methodological correction happened mid-investigation, worth recording in full since it
overturned the first instinct rather than just refining a number.** The first version of this
test read `Z_scaled` as raw `|Z[k1,k2]|` shares, with no Hungarian permutation correction —
flagged afterward as a possible flaw, since FINDINGS §14 documents that a relation's raw
diagonal can be legitimately scrambled (community index *i* on one facet paired with a
*different* index on the other) even when real within-community coupling is intact. The fix
seemed obvious: extend production's own permutation-correction machinery
(`_hungarian_relabel_relation`) from the diagonal-only reading `_relation_community_share`
already uses to the full K×K matrix (`diagnostic_blocks.relation_share_matrix`, self-checked
bit-for-bit against production's diagonal, 0.00e+00 diff).

**Applying that "fix" made every correlation weaker, not stronger** (pooled Spearman
0.663→0.599; diagonal 0.660→0.562; off-diagonal 0.422→0.343). Working through *why* revealed
the fix was based on a wrong diagnosis: `Z[k1,k2]` is, by the literal construction of the
tri-factorisation (`U1 @ Z @ U2ᵀ`), *already* indexed by `U_prob_f1`'s own column k1 and
`U_prob_f2`'s own column k2 — there is no representational ambiguity to correct for in a
comparison against `data_share`, which uses that exact same native indexing. What
`_hungarian_relabel_relation` actually corrects is a *different* problem: two facets'
community *index numbers* can be arbitrarily offset from each other, which matters for
`collapse_pen`'s aggregate, permutation-invariant within/between-mass accounting (where "these
two clusters are really the same community, just numbered differently" should count as
within), but applying that same relabelling here **breaks** alignment with `data_share`'s
native, unpermuted indexing rather than fixing it. The tool is correct for its designed
purpose (§4.20) and wrong for this comparison. **Conclusion: the original raw (uncorrected)
method was the methodologically appropriate one — reverted to it as the primary reading.**
Recorded here in full, not silently dropped, per this project's standing practice for
self-corrections (§9's spirit, §18/§19's precedent this session).

**Primary result (raw method, confirmed consistent across two independent runs on the same 48
cached T1 fits): pooled Pearson r=0.733, Spearman r=0.660** (n=4,588) — meaningfully stronger
than the max_share-only comparison (community-level Pearson topped out at 0.42–0.51 in Test
2). Split by diagonal vs. off-diagonal:
- Diagonal (within-community, n=1,288): Pearson r=0.613, **Spearman r=0.656**
- Off-diagonal (between-community, n=3,300): Pearson r=0.630, **Spearman r=0.416**

Spearman (rank correlation, less dominated by a few large values) shows off-diagonal terms
tracking the independent reading noticeably worse than diagonal terms. **Practically:
within-community coupling readings from `Z_scaled` are more trustworthy than between-community
coupling readings** — worth an explicit caveat wherever the article leans on `Z`'s off-diagonal
specifically to claim a between-community relationship.

**Under increasing `lambda_z_offdiag` pressure (0.0→2.0), correspondence declines mildly but
consistently:** Spearman falls 0.693→0.677→0.638→0.595 (Pearson: 0.765→0.737→0.747→0.723) — a
decline of 0.098.

**Calibration check (the piece that turns "a trend" into "a confirmed effect"):** is that
decline bigger than ordinary fit-to-fit noise, or could two random re-fits at the *same*
lambda show a swing that large anyway? Tested directly by re-running the identical
correlation across 5 independent seeds (1000/2000/3000/4000/5000, reusing FINDINGS §19's seed
list) at **fixed** production `lambda_z_offdiag=0.05`, same C1–C6 × K∈{3,4} grid (60 fits, one
non-converged and excluded — `C3/K=4`/seed=2000, consistent with §20's convergence findings
above). Seed-to-seed Spearman spread at fixed lambda: **0.035** (range [0.682, 0.717]).
**The lambda-driven decline (0.098) is 2.8× the seed-to-seed baseline spread (0.035).** The
erosion under `lambda_z_offdiag` pressure is a real, lambda-driven effect, not something
ordinary seed variability would produce on its own.

**Net effect on the open question:** `Z_scaled`'s full diagonal/off-diagonal pattern is
meaningfully grounded in real structure overall — more so than `collapse_pen`'s single scalar
suggested — but the off-diagonal (between-community) values specifically are the weaker link,
and their correspondence to independent structure erodes under `lambda_z_offdiag` pressure by
a margin now confirmed to exceed ordinary noise. This is exactly the training-dynamics-driven
mechanism this test was built to check for, now with a calibrated baseline behind it rather
than an uncalibrated trend. Still distinct from §18's territory (which tested a *different*
mechanism — static rotational freedom within one fit — and found it narrow); this section's
mechanism and §18's are related but not the same question, and neither fully substitutes for
the other. **This test used T1 only**; not yet replicated on T2, where §20's other findings
show `collapse_pen`-relevant dynamics differ materially — a natural next step if firmer
grounding is needed before leaning on this in the article.

Diagnostic scripts: `diagnostic_scripts/collapse_pen_mass_vs_membership_check.py` (Test 2),
`diagnostic_scripts/collapse_pen_exploitability_sweep.py` (Test 1),
`diagnostic_scripts/collapse_pen_optuna_study.py` (Test 4, T1),
`diagnostic_scripts/collapse_pen_optuna_study_t2.py` (Test 4, T2 replication),
`diagnostic_scripts/z_scaled_diagonal_offdiagonal_gaming_check.py` (diagonal/off-diagonal,
first pass, raw method, T1), `diagnostic_scripts/z_scaled_offdiag_calibration_test.py`
(permutation-corrected re-run + seed-baseline, corrected method),
`diagnostic_scripts/z_scaled_offdiag_calibration_test_raw.py` (seed-baseline re-run on the
raw method, the primary reading). Results in
`diagnostic_results/collapse_pen_mass_vs_membership_check.json`,
`diagnostic_results/collapse_pen_exploitability_sweep.json`,
`diagnostic_results/collapse_pen_optuna_study.json`,
`diagnostic_results/collapse_pen_optuna_study_t2.json`,
`diagnostic_results/z_scaled_diagonal_offdiagonal_gaming_check.json`,
`diagnostic_results/z_scaled_offdiag_calibration_test.json`,
`diagnostic_results/z_scaled_offdiag_calibration_test_raw.json`.

---

## 21. Domain-balance noise floor (ticket 82, `TOL` decided), and evidence for extending
chunk12's relation weighting (ticket 84, not started)

**Status: `TOL` (D3) decided at `0.15` for the toy corpus, informed by a genuine measured
noise floor, not a guess. A separate, larger finding fell out of measuring that floor: the
5 currently-unweighted "grammar" relations in chunk12 show degree concentration comparable
to or worse than the one relation (`S_Art_Auth`) that already got hub-damping for exactly
this reason — proposed as ticket 84, design not started.**

### D2's entity-weighted spread inflation, made exact (not just observed)

FINDINGS §13's "Update" subsection measured that `dev_k`'s spread grows substantially under
entity weighting (mean `dev_k` ≈0.08 equal-weighted → ≈0.13 entity-weighted) and flagged that
part of this might be a mechanical consequence of `auth` dominating the social-domain
average, not "more real imbalance." That mechanism is now derived exactly, not just
suspected.

`soc_k = Σ_f w_f · p_f[k]`, `Σw_f = 1`. Model each facet's contribution to `soc_k` as a
shared community-level signal plus an idiosyncratic per-facet deviation, roughly uncorrelated
across facets with comparable variance. The idiosyncratic-noise contribution to `Var(soc_k)`
is then `σ_e² · Σ_f w_f²` — and `Σ_f w_f²` (the Herfindahl concentration index of the weight
vector; `1/Σw_f²` is Kish's "effective sample size" in survey statistics) is minimized by
equal weights and grows as weight concentrates on one facet, a pure convexity fact with no
free parameters:

| | equal `Σw²` (`n_eff`) | entity `Σw²` (`n_eff`) | predicted SD ratio `√(entity/equal)` |
|---|---|---|---|
| Social, T1 | 0.250 (4.0) | 0.588 (1.70) | 1.53× |
| Social, T2 | 0.250 (4.0) | 0.577 (1.73) | 1.52× |
| Semantic, T1 | 0.200 (5.0) | 0.238 (4.21) | 1.09× |
| Semantic, T2 | 0.200 (5.0) | 0.237 (4.23) | 1.09× |

Social collapses from an effective 4 facets to ~1.7 (`auth`'s ~75% share); semantic barely
moves (no facet exceeds ~32%) — an asymmetry predictable from facet headcounts alone, before
looking at any `r_k` outcome. Checked directly against the actual `sd(r_k)`:

| | equal | entity | observed ratio | predicted ratio |
|---|---|---|---|---|
| T1 | 0.0967 | 0.1519 | 1.571× | 1.533× |
| T2 | 0.0977 | 0.1417 | 1.450× | 1.519× |

A parameter-free prediction from facet headcounts alone lands within 2-5% of the observed
spread increase — essentially all of the *width* increase is this mechanical effect, not new
information about the corpus.

### The mean/median shift toward semantic is the same mechanism, not a second mystery

FINDINGS §13's update also reported `mean(r_k)` dropping from ≈0.494 to ≈0.465 under entity
weighting (both slices) and flagged this as unexplained — a hypothesis that `auth` is simply
*less decisively assigned* (lower `U_prob` row-max) than the small social facets was tested
and **refuted**: `auth`'s median row-max (0.999) is if anything higher than `journ`'s (0.934)
or `affil`'s (0.920), not lower.

The actual mechanism: `Σ_k soc_k = Σ_f w_f Σ_k p_f[k] = Σ_f w_f = 1` for *any* weighting
scheme (each facet's own profile sums to 1 over communities, and the weights sum to 1), so
`mean_k(soc_k) = 1/K` is an algebraic invariant, identical under equal and entity weighting
— confirmed directly: `mean(soc_k)` = 0.2857 (T1) / 0.2941 (T2) under *both* weightings,
while `var(soc_k)` roughly doubles (0.0140→0.0336 T1, 2.39×; 0.0158→0.0408 T2, 2.58×) and a
real fraction of communities are pushed under 0.10 under entity weighting only (14.3% T1,
5.9% T2, vs. 0% under equal weighting).

`r_k = soc_k/(soc_k+sem_k)` is concave and increasing in `soc_k` for fixed `sem_k` (Jensen's
inequality territory): a variable with the *same mean* but *higher variance*, passed through
a concave function, has a *lower* expected output. Pushing `soc_k` toward zero drives `r_k`
sharply toward zero (denominator stays ≈`sem_k`); pushing `soc_k` up by the same amount only
nudges `r_k` toward 1 with diminishing returns (bounded above). So spreading `soc_k` out more
widely around the same mean — exactly what entity weighting's Herfindahl effect does —
mechanically lowers `mean(r_k)`, with no separate cause needed. The earlier "unexplained,
needs its own investigation" framing is withdrawn; this is the same mechanism as the spread
increase above, read through one additional, well-known nonlinearity.

### Absolute noise floor — multi-seed reproducibility of `r_k`

The spread/mean analysis above compares two *weighting schemes* against each other; it says
nothing about how reliable a *single fit's* `r_k` reading is in absolute terms — the question
needed to set `TOL` so it exceeds genuine noise rather than firing on communities that are
not actually imbalanced, just unluckily seeded.

**Method** (`diagnostic_scripts/domain_balance_seed_noise.py`): for each (config, K, slice)
cell, refit at 5 seeds (`MASTER_SEED` + 4 offsets) — same data, same hyperparameters
(`lambda_l1=0`, `lambda_z_offdiag=0.05`), only the RNG differs, which perturbs the NNDSVDar
init noise (ticket 74) and which of the model's many local optima (§9) the optimizer lands
in. Community labels are not consistent across independent fits (label-switching), so each
non-reference seed is realigned onto the reference seed's labeling via
`s5_dual_track_alignment`'s Track A (JSD/probability-space Hungarian assignment — the same
space `u_prob`'s domain-balance reading uses) — **faithful reuse of production's own §S5
stability-analysis alignment code**, not a new alignment invented for this test. `r_k`
(entity-weighted, `u_prob` space — D2's decided primary reading) is recomputed under each
seed's aligned labeling; the across-seed standard deviation per community is the noise floor.
Smoke-tested first: `C1` (documented weakest, §4.15, single-anchor) showed much larger
seed-to-seed swings than `C6` (dual-anchor, strong) in a 2-seed pilot, confirming the
alignment step is doing real work, not producing arbitrary label-mismatch noise.

**Full grid (C1-C6 × K∈{3,4} × T1/T2, 5 seeds each; 76 usable community-cells, 2 cells
skipped for reference-seed non-convergence — `T2/C3/K4`, `T2/C4/K4`, consistent with the
non-convergence already known for those cells):**

| | mean | median | p90 | max |
|---|---|---|---|---|
| noise SD(`r_k`) | 0.134 | **0.144** | 0.192 | 0.229 |

Paired directly against each community's own `dev_k` (from FINDINGS §13's entity-weighted
E1 numbers, same reference seed): only **46%** of communities have `dev_k` exceeding their
own noise SD at all; only **21%** exceed 1.5×; only **13%** exceed 2×. The noise floor's
median (0.144) is larger than the signal's median (`dev_k`=0.112) — on this toy corpus, a
typical community's apparent domain imbalance is not clearly distinguishable from what a
different random seed alone would produce. Strongest genuine cases: `T1/C1/K4` community 1
(`dev_k`=0.108, noise SD=0.024, ratio 4.5×), `T2/C2/K4` community 2 (`dev_k`=0.322, noise
SD=0.141, ratio 2.3×). Weakest: `T1/C3/K3` community 2 (`dev_k`=0.009, noise SD=0.140, ratio
0.06 — indistinguishable from noise).

**Per-configuration breakdown** (pooling K and slice per config):

| Config | mean | median | max |
|---|---|---|---|
| C1 | 0.114 | 0.137 | 0.193 |
| C2 | 0.135 | 0.135 | 0.193 |
| C3 | 0.105 | 0.113 | 0.159 |
| C4 | 0.156 | 0.163 | 0.178 |
| C5 | 0.151 | 0.164 | 0.229 |
| C6 | 0.144 | 0.148 | 0.216 |

No config is an outlier severe enough to justify exclusion — the spread (0.105-0.156 mean)
is real but modest. Notably, `C1` (documented weakest overall, §4.15) is the **second-lowest**
noise config here, not the highest; dual-anchor configs (C2/C5/C6, mean 0.143/median 0.152)
are if anything slightly noisier than single-anchor ones (C1/C3/C4, mean 0.123/median 0.140)
— the opposite of what §4.15's known weakness would predict. Whatever drives domain-balance
noise specifically is evidently a different, more cell-specific property than the
hypervolume/semantic-axis weakness already documented for C1, not simply "config strength."

**Decision: `TOL = 0.15` for the toy corpus** (D3, closes the open item in §4.18/ticket 82),
sitting just above the overall noise median. Caveat: several cells (`C4`/`C5` T1 medians run
0.16-0.17) have their own local noise floor at or slightly above 0.15, so some false-positive
risk remains concentrated there specifically — expected given noise is not uniform across the
grid, not a reason to reconsider the choice. **Explicitly a toy-corpus value, flagged for
re-estimation on the 22k-article corpus**, matching the pattern already used for `lambda_l1`'s
bounds (ticket 77).

### Re-derivation on post-08-27-fix data (tickets 86/87 Stage 0e) — verdict held, `TOL` kept at `0.15`

The measurement above used with-repository data (regenerated 2026-08-25/26, before the
journal-resolution repository-exclusion fix — `plans/ticket86-87-ghosts-and-relation-concentration.md`
Context section). Both `domain_balance_seed_noise.py` and `domain_balance_measurement.py`
were re-run unmodified on the current post-fix pickles and re-paired community-by-community
(same method as above, same 24-cell grid, now fully converged — 84/84 community-cells usable
vs. 76/84 before, the two previously-non-convergent cells, `T2/C3/K4` and `T2/C4/K4`, now
fitting cleanly):

| | old (with-repository) | new (post-fix) |
|---|---|---|
| noise SD(`r_k`): mean / median / p90 / max | 0.134 / 0.144 / 0.192 / 0.229 | 0.119 / 0.120 / 0.172 / 0.234 |
| `dev_k` (entity-weighted `u_prob`): median | 0.112 | 0.093 |
| % communities with `dev_k` > 1× own noise SD | 46% | 36% (30/84) |
| % exceeding 1.5× | 21% | 15.5% (13/84) |
| % exceeding 2× | 13% | 4.8% (4/84) |
| % with `dev_k` > `TOL`(0.15) outright | — | 19% (16/84) |

Both the noise floor and the signal shrank by comparable proportions on the post-fix data,
so the qualitative relationship is unchanged — median noise SD (0.120) still exceeds median
`dev_k` (0.093) — but the fraction of communities whose apparent imbalance survives its own
seed-noise floor dropped at every multiple tested (46%→36% at 1×, 21%→15.5% at 1.5×,
13%→4.8% at 2×). If the repository-exclusion fix moved this story at all, it moved it toward
"more of the apparent domain-balance signal is noise," not less — reinforcing, not
undermining, the original caution against reading `dev_k` at face value.

Per-config means (pooled across K/slice, for comparison against the original per-config
table above) stayed in the same relative order — C3/C5 lowest (`dev_mean` 0.068/0.087 vs.
`noiseSD_mean` 0.110/0.104), C1/C6 closest to their own noise floor (0.117/0.125,
0.120/0.138) — no config changed which side of the noise floor it sits on.

**Decision: `TOL = 0.15` kept, not re-tuned.** The margin between `TOL` and the noise median
widened (old ratio `0.15/0.144` ≈1.04×, new ratio `0.15/0.120` ≈1.25×) purely because the
noise floor fell while `TOL` stayed fixed — nothing in this re-run argues for lowering it.
Raising it toward the new p90 (≈0.17) was considered and rejected: D3's own rule was "set
`TOL` just above the noise *median*," and switching to a stricter percentile now, immediately
after a data refresh happened to move the estimate, would be fitting the threshold to this
particular measurement rather than applying the standing rule consistently — exactly the
"chasing noise" pattern the seed-noise-floor discipline exists to prevent. The term this
threshold gates is also currently inert in production (`lambda_domain_balance` frozen at
`0.0`, not in Optuna's search space, §4.18/FINDINGS §22), so there is no live training
consequence to a marginal retune. **Would revisit if** a future re-measurement showed the
noise median climbing *above* `0.15` (a real erasure of the margin), not merely moving
around below it. Re-derivation with a larger, statistically confident sample remains
reserved for the 22k-article rebuild, per the original caveat above.

Script/results unchanged (re-run as-is, no code modified):
`diagnostic_scripts/domain_balance_seed_noise.py`,
`diagnostic_scripts/domain_balance_measurement.py`. Results:
`diagnostic_results/domain_balance_seed_noise.json`,
`diagnostic_results/domain_balance_measurement.json`.

**Single-fit vs. multi-fit reliability — why more seeds don't just fix this for the in-loop
term.** Averaging several aligned seeds' `r_k` readings would give a materially more reliable
*point estimate* (standard averaging reduces error with more independent readings) — exactly
the "stability/consensus across seeds" idea already named as a candidate model-quality check
in §10. But the in-loop penalty must compute `r_k` from the model's *current* parameters at
every training epoch to produce a gradient; there is only one parameter set being trained at
any moment, so "average over 5 seeds" is inherently a post-hoc, multi-run operation that
cannot be the signal an in-loop term trains against. Multi-seed averaging remains available
for the *outer-loop* audit (a more reliable final `r_k` for a chosen config, after Optuna has
already picked it) but does not solve the in-loop term's reliability problem. A further
limit: averaging only recovers a truer answer if the different seeds are noisy variations
around one underlying structure; if instead they are landing in substantively different,
comparably-good local optima (§9's non-convexity), more seeds sharpen the *estimate of that
uncertainty* rather than eliminating it.

### chunk12 relation weighting: a correction, and evidence for extending it (ticket 84)

**Correction to this session's own earlier characterization.** Not all `M_*` relations are
"unweighted binary" — verified directly against `chunk12.py`, not from memory. The 5
*grammar* relations (`M_Atom_Child`, `M_Child_Parent`, `M_Fringe_Cousin`, `M_Cousin_Parent`,
`M_Cousin_Child`) are genuinely unweighted (`build_grammar_csr`, weight ≡ 1). The 3 *anchor*
relations (`M_Parent_Art`, `M_Child_Art`, `M_Cousin_Art`) carry real tf-idf weighting
(`build_anchor_csr`: `weight = log1p(tf) · idf[hyperedge]`, `idf = log(N_articles/doc_freq)+1`
computed per semantic hyperedge from global article-reference frequency — chunk12.py:258,
315-321). This weighting is not merely an initialization nudge: the idf-weighted matrix *is*
the training target at every epoch (Frobenius normalization only rescales overall magnitude,
not the relative idf pattern within the matrix), so it shapes `U_final`/`U_prob` continuously
through training, not just at the SVD-init stage. Scope is narrow, though — only 4 of 9
facets (`parent_he`, `core_child_he`, `cousin_he`, `art`) ever touch an idf-weighted
relation; the other 5 facets have zero direct idf influence. This is directly related to,
but a narrower question than, the existing "UDSR" register entry's Test 2 (idf-vs-row-max
correlation, found weak/inconsistent) — that test checked concentration only, not whether
idf shapes the loadings themselves.

**Degree-concentration evidence** (row/col `max÷mean` ratio, both slices, checked directly
against the raw relation matrices, not asserted):

| Relation | row max÷mean (T1/T2) |
|---|---|
| `S_Art_Auth` (**already damped**, the reference case) | 7.5× / 2.8× |
| `M_Atom_Child` | **13.0× / 16.6×** |
| `M_Child_Parent` | 7.1× / 11.2× |
| `M_Fringe_Cousin` | 6.4× / 7.4× |
| `M_Cousin_Parent` | 7.9× / 5.1× |
| `M_Cousin_Child` | 5.3× / 5.2× (row); 5.3× / 12.2× (col) |
| `S_Art_Journ` | 1.0× / 1.0× (row, trivial — one journal per article); **5.8× / 6.5×** (col) |

All 5 grammar relations show degree concentration comparable to or worse than `S_Art_Auth`'s
own already-damped case — `M_Atom_Child` is the worst in the entire topology, more skewed
than the relation that motivated hub-damping in the first place, and currently receives none.
`S_Art_Journ`'s column skew (journals) is literally a document-frequency phenomenon — the
same thing the anchor idf formula already measures, just for a different facet. This connects
directly to §9's already-documented, currently-unresolved limitation: *"Ubiquitous semantic
elements that should load ~1/K across all communities are structurally disfavoured... NMF's
winner-takes-all tendency plus the `z_offdiag` penalty... both push against it."* Extending
weighting to these relations is a concrete, evidence-backed candidate fix for that named
problem, not a new idea introduced without basis.

**What the existing `S_Art_Auth` log-damping actually does, verified exactly against the real
(already-damped) matrix — two different effects, not one:**

```python
damped_mass = log2(1 + degree)
weight(r, c) = count(r, c) * damped_mass / degree
```

Confirmed to match the real matrix to 4 decimal places (`article 17`, degree=65,
`actual_mass=6.0444` = `predicted=log2(66)=6.0444`, exactly). Two readings of "higher-degree
still more emphasized, but the gap shrinks," which behave *oppositely*:

- **Total row mass** (an article's overall presence/influence): matches that description
  exactly. Degree 65 vs. degree 6: raw ratio 10.8×, post-damping mass ratio **2.15×** — still
  monotonically more for the higher-degree article, but the gap compresses hard (log vs.
  linear).
- **Per-tie weight** (what one specific author's own connection to that article is worth):
  **reverses**, does not just compress. Degree-65 article's per-tie weight: 0.093. Degree-6
  article's: 0.468 — each tie in the smaller article counts **5×** more individually. The
  mega-collaboration's *total* presence is still elevated, but each of its 65 co-authors is
  pulled toward that article's community far more weakly than a co-author on a small paper
  would be.

**This per-tie redistribution — not just the total-mass compression — is the actual
mechanism relevant to the stated design goal (avoiding mono-article communities built around
one high-author-count article):** without it, one 65-author mega-collaboration could pull all
65 co-authors strongly toward a single community based on that one paper; the damping
specifically prevents that by weakening each individual tie inside a high-degree article,
not merely by capping the article's aggregate footprint. Any extension of this mechanism to
the grammar relations should preserve this property deliberately, not just replicate the
total-mass compression.

**Recommendation as originally written (superseded by execution, kept for history):** extend
the existing anchor idf formula to `S_Art_Journ` (same document-frequency logic, low risk, no
new mechanism). Extend `S_Art_Auth`-style degree log-damping to the 5 grammar relations,
`M_Atom_Child` first as the clearest case. **Real cost, not to be underestimated:**
`chunk12.py` is the data-generation layer underneath this entire session's cached fits and
every empirical number recorded above and in §13/§17 — changing it invalidates that cache and
requires re-deriving those numbers, a substantially larger blast radius than any
`chunk13v9.py`-level change made this session.

**EXECUTED (ticket 84, CLAUDE.md §8) — both recommendations above were checked against code
and found to need correction before implementation, not simply followed.** `S_Art_Journ`
was **deferred entirely** — the anchor idf formula keys on the row, journals are the column,
and the required `dfreq_global['journ']` was never collected; "low risk, no new mechanism"
was wrong. Scope for the grammar relations was **not** "all 5" — split by row-entity type
(D8): `M_Atom_Child`/`M_Fringe_Cousin` damped on this section's ubiquity argument;
`M_Child_Parent`/`M_Cousin_Parent` left untouched (recurrence there plausibly reflects
genuine synthesis, not noise); `M_Cousin_Child` damped on a separate, bespoke justification
(context genericness for interpretation, not ubiquity — checked against real
`dummy_cousin`/`cousin_he` data). Phase 1 (read-only gate): pass, all metrics. Phase 3
(paired-seed fit validation, real model fits, not just static-matrix checks): **genuinely
mixed** — 3 of 6 target cells moved the predicted direction, 3 didn't, 2 of 6 clear the seed
noise floor with confidence (one each direction) — consistent with this section's and §21's
own standing finding that the noise floor here is comparable to the effect size, not evidence
the mechanism is wrong. **One open item, named rather than left implicit:** `fringe_atom`/T2
is the single wrong-direction cell that clears the noise floor with confidence, and it is
*not* explained by "T1 has fewer live entities" (checked directly — T1 does have fewer
entities and more wrong-direction cells overall, but this specific confident anomaly sits in
T2, the *better*-resourced slice). Flagged for a targeted re-check at 22k scale, not just a
re-run of the aggregate test. **D5 (§9 justification): user-confirmed, struck** — the
`U_prob` L1-row-normalization argument above holds; this only removes the claim that ticket
84 addresses §9, which remains its own separately unresolved limitation. Phase 4: a new
versioned file `chunk12v2.py` (not an in-place edit —
`chunk12.py` untouched) produces this data; null-rebuild verified clean against `chunk12.py`'s
existing output (maps identical, untouched relations byte-identical, damped relations match
the transform exactly). **Not promoted to production** — held pending the 22k-article corpus,
where the noise floor is expected to shrink enough to actually adjudicate the mixed Phase-3
result. Full record: `PhD_CNMF/plans/ticket84-grammar-hub-downweighting.md`; CLAUDE.md §8
ticket 84 and §4.21 carry the condensed version.

### `cousin_he`/T2's precision follow-up, resolved (tickets 86/87 Stage 0e)

Of the "2 of 6 clear the seed noise floor with confidence" above, `cousin_he`/T2's
confirmation was the weaker of the two: it rested on only the original 5-seed noise
estimate, on the thinnest sample in the whole test (`M_Cousin_Child` is active in only 2 of
6 configs). `grammar_damping_cousin_he_more_seeds.py` exists to extend that specifically
(5→15 seeds, C3/C4 only, same aggregation method, no new metric) — the plan file recorded
it as having "died silently mid-run... not relaunched," which turns out to be stale: a
completed run was already sitting in the pre-fix cache (dated 2026-08-26, never folded back
into the plan text — a documentation gap, not a live problem).

Re-run on current (post-08-27-fix) data and compared against that pre-fix completed run:

| | T1 Δρ | T1 exceeds 2×SD (15 seeds)? | T2 Δρ | T2 exceeds 2×SD (15 seeds)? |
|---|---|---|---|---|
| old (pre-fix, 5-seed SD said "yes") | +0.030 | No | −0.218 | **No** (15-seed SD widened to 0.128, ratio fell to 1.7×) |
| new (post-fix) | +0.028 | No | −0.242 | **Yes** (15-seed SD 0.113, ratio ≈2.15×) |

`T1` is unchanged — small, wrong-direction, well inside noise either way (ratio ≈0.27), same
conclusion as before. `T2` is where the extension mattered: on the old data, going from 5 to
15 seeds actually pulled the result *below* the 2×SD bar (the "confirmed" status in the
corrected table rested on an optimistically small 5-seed SD estimate). On the post-fix data,
the 15-seed SD is tighter, not wider, and the effect itself is slightly larger, so it clears
2×SD comfortably. **Net: the Phase 3 headline tally is unchanged (still 3/6 correct
direction, 2/6 confirmed, one each direction), but `cousin_he`/T2's confirmation is now the
best-supported result in the grid rather than the most fragile one** — the specific
precision gap this script was built to close is closed. Does not change the standing
disposition (hold Phase 4, don't chase further on the toy corpus, revisit at 22k scale).

Script unchanged (re-run as-is): `diagnostic_scripts/grammar_damping_cousin_he_more_seeds.py`.
Result: `diagnostic_results/grammar_damping_cousin_he_more_seeds.json`.

Diagnostic scripts (no `chunk13v9.py` or `chunk12.py` changes made):
`diagnostic_scripts/domain_balance_seed_noise.py`. Results:
`diagnostic_results/domain_balance_seed_noise.json`,
`diagnostic_results/domain_balance_seed_noise.log`.

---

## 22. Ticket 82 E2 — domain-balance mechanism implemented (in-loop + outer-loop),
weight fixed at 0.0 for the toy corpus

**Status: E2 is DONE — both halves of the settled design (§18/§21) are now real code in
`chunk13v9.py`, not diagnostic-only.** The in-loop weight (`lambda_domain_balance`) is fixed
at `0.0`, matching `lambda_l1`'s own resolution (ticket 78) — mechanism fully wired and
observable, not stubbed out, but not actively steering training on this corpus. This section
records why, with the evidence that led there.

### Why the outer-loop element exists, and why it is `U_prob`-based

D1-D3's original design (§18/§21) specified an in-loop, differentiable penalty only, with an
outer-loop mirror left as a "maybe" for a later decision. That decision was forced by a
concrete question: since the penalty's weight is structurally a hyperparameter (like
`lambda_z_offdiag`), if it is ever made Optuna-tunable, Optuna needs an independent signal to
judge whether a given weight actually achieved balance — the same role `collapse_pen`/
`coherence_pen`/`semantic_pen` already play for other properties Module 2's loss doesn't
directly enforce. That signal is the outer-loop element.

**It must not be built the way an earlier, separate exploratory script in this codebase
built a structurally similar idea.** Independently of this ticket, `diagnostic_blocks.py`
already contained `make_domain_balance_penalty_torch` and a `domain_penalty_exploitability_*`
test family (undated, not previously written into CLAUDE.md/FINDINGS.md) probing whether a
`Z_scaled`-based domain-balance term is exploitable via the same undetermined free scaling
direction `U_scales` that already forced `collapse_pen`'s rewrite (tickets 79/80). It is:
measured directly (`domain_penalty_exploitability_grid.json`, 12-cell grid, weight 0 vs 10),
the term's own optimized proxy (`mass_tvd`, built from raw `Z_scaled` diagonals) "improved"
30-40x in every cell, while the real membership-based imbalance (`mem_tvd`, `U_prob`-based)
got **worse** in 3 of 12 cells (C3/K4: mass_tvd −0.390 but mem_tvd **+0.298**; C6/K2: mass_tvd
−0.133, mem_tvd **+0.268**; C2/K2: mass_tvd −0.009, mem_tvd **+0.188**) — the optimizer was
gaming the `Z_scaled`-based proxy, not fixing real imbalance. Root cause: `Z_scaled = Z_pos ×
U_scales_f1 × U_scales_f2`, and ticket 79 already proved `U_scales` is an undetermined free
gauge direction (scaling a `U_pos` column by any `c`, compensated in `Z_pos`, leaves
`recon_loss`/`U_norm`/`Z_scaled` unchanged) — pooling raw `Z_scaled` diagonals **across
different facets**, as any `Z_scaled`-based domain-balance sum must, lets the optimizer
manipulate each relation's independently-gameable `U_scales` product without touching which
entities actually belong to which community.

`U_norm` (and `U_prob`, a pure function of it) is **proven algebraically immune** to that
exact transformation — confirmed, not just argued: `evaluate_domain_balance`'s output was
cross-checked against the already-validated diagnostic pipeline
(`facet_membership_profile`+`domain_balance_r_k`) on a real fit and matched to ~8 significant
figures. This is why both the in-loop and outer-loop elements of E2 are built on `U_prob`,
not `Z_scaled` — the design choice §18 already leaned toward, now with a concrete,
measured reason rather than a general preference.

### The in-loop weight: measured, not guessed, and found not to clear this corpus's noise

Before fixing the weight, its natural scale was measured on real baseline (weight=0) fits —
`mean_k(max(0, |r_k-0.5|-TOL)²)`, TOL=0.15 (D3) — the same first step ticket 77 used for
`lambda_l1` (measure `mean_sparsity` on a real fit before back-solving a range). It does not
transfer as cleanly: because the formula is a dead-band (zero below TOL, not a smoothly
positive quantity like `mean_sparsity`), **5 of 12 baseline fits landed with an EXACT 0.0**
raw penalty (already balanced within the ±0.15 band), and the largest measured value was
small (≈0.0043) — a `lambda_l1`-style range derivation (dividing by the smallest measured
value) is undefined here (division by zero) and, even anchored on the largest value, would
have suggested weights far more aggressive than what training could actually tolerate:
`weight=5` implies only ≈2.7% of `math_loss`'s own scale for even the worst baseline case
measured, yet caused real non-convergence (`diagnostic_scripts/domain_balance_inloop_sweep.py`,
`_seedcheck.py`/`_seedcheck_v2.py`) — down to 1 of 5 seeds converging in one v2 cell. Unlike
`lambda_l1` (ticket 77: monotonic loss cost, zero convergence failures across its full tested
range), this term's failure mode is convergence instability itself, at a magnitude a
loss-scale-only derivation would not have flagged as risky.

**Full evidence, both `chunk12` and `chunk12v2` data (this session's explicit requirement —
domain-balance testing must not be v1-only), C3/K4 and C6/K4 pilot cells throughout:**

- **Single-seed pilot** (`domain_balance_inloop_sweep.py`, weight ∈ {0,1,5,20,50}): looked
  promising on C6/K4 (clean, roughly monotonic `mean_dev_k` reduction) — **this did not
  replicate under multi-seed testing** and is flagged here specifically as a caution against
  trusting single-seed diagnostic results, consistent with this project's standing practice
  (ticket 84 Phase 3's "pair the seeds" rule).
- **5-seed paired check at weight=5** (`domain_balance_inloop_seedcheck.py` [v1, T1] /
  `_v2.py` [v2, T1_v2+T2_v2]), 6 (cell × data) combinations: **0 of 6 cleared 2×the paired
  seed-noise SD.** 4 improved (small), 2 worsened (small) — direction split, no confident
  signal either way. Convergence reliability at weight=5 ranged from 5/5 down to **1/5** (v2,
  `T2_v2/C6/K4`).
- **Smaller-weight + higher-exponent follow-up**
  (`domain_balance_inloop_power_and_weight_sweep.py`, 3 seeds, weight ∈ {0.5,1,2} at power=2
  and weight ∈ {200,400} at power=4 — the power=4 weights rescaled so their push matches
  power=2's at a typical small excess, per the worked gradient comparison below), 20 (cell ×
  setting) combinations across 4 (cell × data) conditions: **1 of 20 cleared 2×SD**
  (`T1_v2/C3/K4`, power=2/weight=2 and power=4/weight=200, both improving). **One cell
  (`T1/C3/K4`) worsened at every one of its 5 tested settings** — small each time, but
  consistent in direction, not noise-cancelling. Convergence was much better than weight=5's
  near-collapse but still imperfect (occasional 2/3 seeds converging).
- **Exponent (power) comparison, worked exactly, not assumed:** the derivative of
  `excess^p` is `p·excess^(p-1)`; within this problem's actual range (`excess` ≤ 0.35, since
  `r_k∈[0,1]` and TOL=0.15), a higher power (4) gives a **smaller** gradient than power=2 at
  every point in range at a FIXED weight (crossover would need excess>0.71, impossible here)
  — "higher power = stronger penalty" is false in this range unless the weight is also
  rescaled up. A lower power (1.5) does the reverse — **stronger** than power=2 everywhere,
  and disproportionately so at small `excess` (≈3.4× stronger at excess=0.05, vs ≈1.4×
  stronger at excess=0.30) — reasoned through but not empirically tested this session (the
  measured excess values in this corpus are small, so a lower power was expected, on this
  reasoning, to worsen rather than fix the convergence problem — not verified).

**Decision: `lambda_domain_balance` defaults to `0.0`** (read via `params.get(...)`, same
pattern as `lambda_l1`), **not added to Optuna's search space**. The mechanism is fully
implemented and its raw (unweighted) value is exposed in diagnostics
(`raw_domain_balance_penalty`) for observability, exactly mirroring `raw_sparsity_loss`'s
role after ticket 78. Explicitly a toy-corpus decision — revisit at 22k scale, where a larger
corpus may finally let a weight (and possibly a non-default TOL or exponent) be adjudicated
rather than guessed or abandoned for lack of resolution, the same standing caveat already
attached to `TOL`, `log2`, and `lambda_l1`'s own bounds.

### The outer-loop element is active now, independent of the in-loop weight

`evaluate_domain_balance` (Module 3) is called from `evaluate_complete_solution` and its
output (`domain_balance_pen`) is summed into `sociological_penalty` **unconditionally** —
this is a real, immediate change to Optuna's second Pareto axis, not gated behind
`lambda_domain_balance`. This mirrors `collapse_pen`/`coherence_pen`, neither of which has an
in-loop counterpart either: a pure outer-loop reality check can meaningfully differentiate
trials on a property even when nothing in the training loss is pushing toward it directly —
trials whose architecture/hyperparameters happen to produce more balanced communities will
now score better on this axis, before any decision is made about whether to also add active
in-loop pressure.

### A separate, pre-existing bug found during verification (unrelated to this ticket)

While bit-exact-verifying the in-loop addition (confirming `lambda_domain_balance=0.0`
produces identical training to before), repeated runs of the *unmodified-by-this-ticket*
`fit_production` path gave different `math_loss` values **across separate process launches**
(same seed, same code) — though bit-identical **within** one process. Traced to
`initialize_tucker_adapted_nndsvd_and_propagate` (chunk13v9.py, unrelated to this ticket)
iterating `active_facets` — a raw Python `set()` — directly (lines ~351, ~524), which
CLAUDE.md §4.9 already names as exactly this hazard ("Never iterate a raw `set()`... Module 4
Section 5... unordered iteration produces non-deterministic tensor geometry") for a different
function. Confirmed the cause precisely: fixing `PYTHONHASHSEED` across separate process
launches makes results bit-identical again. **Not fixed here** — pre-existing, outside this
ticket's scope, flagged in CLAUDE.md's defect register (ticket 85) for a separate decision.

### Files

- `chunk13v9.py`: `FACET_DOMAIN`, `DOMAIN_BALANCE_TOL` (Module 1); `lambda_domain_balance`
  read-in, static presence-mask/facet setup, per-epoch differentiable term, `total_loss`
  wiring, `raw_domain_balance_penalty` diagnostic (Module 2, `run_inner_solver`);
  `evaluate_domain_balance` (new, Module 3) wired into `evaluate_complete_solution`.
- `diagnostic_blocks.py`: `make_inloop_domain_balance_penalty_torch` (the validated
  `U_prob`-based candidate, ported into production above), `fit_instrumented`'s
  `extra_loss_fn` extended to accept `U_norm` (BLOCKS_VERSION 1.14.0).
- New diagnostic scripts (all in `chunk13_execution/diagnostic_scripts/`):
  `domain_balance_raw_penalty_measure.py`, `domain_balance_inloop_sweep.py`,
  `domain_balance_inloop_seedcheck.py` / `_v2.py`,
  `domain_balance_inloop_power_and_weight_sweep.py`, plus the earlier
  `domain_balance_measurement_v2.py` / `domain_balance_v1_v2_noise_compare.py` /
  `domain_balance_seed_noise_v2.py` used to establish `chunk12v2` needed covering at all.
  Results in `diagnostic_results/` under matching names.

---

## 23. Ticket 86 — "ghost communities": low-mass communities are real, measured, and mostly
traced to a structural limit in `S_Art_Auth`, not to a single universal cause

**Status: Measurement phase. Definition settled (user, external discussion), validity of the
measurement basis established, phenomenon confirmed present and structural, one root-cause
candidate (author-facet collinearity) confirmed in some configs and ruled out in others,
another (a mega-author-count "PALM-style" paper driving it) tested and not supported, a
partial remedy (excluding affiliations) piloted with a config-dependent result. No mechanism
designed or implemented — this ticket is diagnostic-only so far, deliberately, matching the
sequencing already used for tickets 82/84 (measure before designing).**

### Definition (user's, from a discussion external to this project's own record)

A community is a candidate "ghost" primarily by **very low share of the model's total
reconstructed mass** — the defining property. Domain skew (mono-domain) and facet
concentration (mono-/oligo-facet) are **secondary, descriptive properties of a low-mass
community, not part of the definition itself** — a community can be low-mass while still
spanning both domains and several facets, and this session's own measurements below show
that happens more often than not on this corpus.

### Choosing and validating the mass measure

The natural candidate, `community_share` — each community's share of the model's total
reconstruction-space mass — already exists as an unexposed intermediate inside
`evaluate_dimensional_collapse` (§4.4/§17): it is the full length-K vector `collapse_pen`'s
`max_share` is drawn from, built by averaging `_relation_community_share`'s per-relation share
vector across every active relation, then renormalising to sum to 1. Two validity questions
were checked directly, not assumed, before trusting it:

**Gauge-invariance (ticket 79's transformation).** `community_share` is a function only of
`U_norm` and `Z_scaled`, both already proven exactly invariant under ticket 79's one confirmed
gauge freedom (rescale one `U_pos` community-column by `c`, divide the matching `Z_pos`
row/column by `c` in every relation touching that facet). Verified empirically, not just
inherited from the proof: reconstructed `U_pos`/`Z_pos` from cached `U_norm`/`Z_scaled`/
`U_scales` (round-trip exact to float32 precision, max diff 3e-8–1.2e-7), then ran a
three-way comparison — baseline, an **uncompensated** version (scale `U_pos` only, don't
touch `Z_pos` — a deliberately broken control) and the real **compensated** version — on 4
(facet, community, scaling-constant) cases across `C3/K6` and `C6/K4`, `T1`. Uncompensated
moved `community_share` by 0.06–0.24 (confirming the test has real teeth); compensated moved
it by at most 4.5e-8 — the same order as the round-trip's own floating-point noise, i.e.
**exactly invariant, not merely close.** Script: `ghost_test1_gauge_invariance.py`.

**Dead-entity contamination.** `_relation_community_share`'s Gram-matrix step
(`c_u = U_norm.T @ U_norm`) sums over every row of a facet, live and dead (chunk12's
global-indexing artifact, §11 — up to 39–61% of a facet's rows in T1). Checked directly on
4 cells (`C3`/`C6` × `T1`/`T2`, K=4): zeroing dead rows via `build_presence_masks` before
recomputing `community_share` changed it by **exactly 0.0** in all 4 cells, explained by
magnitude, not merely observed — dead-row `U_norm` values average `4e-8`–`7.5e-6` across
every facet tested, versus live-row means of `0.014`–`0.12` (5+ orders of magnitude apart;
dead rows never receive gradient under `lambda_l1=lambda_domain_balance=0`, so they stay
pinned near the init clamp floor). Script: `ghost_test2_dead_entity_check.py`.

**Conclusion: `community_share` is safe to use as the mass measure**, for these two specific
concerns. Two older, separately-documented limitations of anything built on `Z_scaled` are
**not** closed by this and remain exactly as open as before: general rotational
indeterminacy (§1, partially addressed by §18 below but not at every K tested here) and
per-relation permutation beyond what the Hungarian correction trusts (§14, same scope gap).

### Main grid measurement (Test 3)

C1–C6 × K∈{2,3,4,5,6} × T1/T2 (single seed, production settings `lambda_l1=0`,
`lambda_z_offdiag=0.05`, `lambda_domain_balance=0`) — 60 cells, 53/60 converged (7 hit the
epoch ceiling, all at K≥4, disclosed not discarded). Repeated on `chunk12v2` data
(T1_v2/T2_v2, same grid) per this session's standing requirement that domain/community-
structure testing not be v1-only — 60 more cells, 51/60 converged. Reported against 4
thresholds: flat 1%, flat 2% (the user's own original candidate), and two `K`-relative
alternatives, `0.25/K` and `0.5/K` (a quarter/half of a community's "fair share" `1/K`).

**The flat-percentage threshold never fires, at any K, in either data version — 0 of 300
community-slots (v1) below 1% or 2%.** The `K`-relative thresholds do fire, and rise with K:
at `0.5/K`, 0/12 cells flagged at K=2, rising to 7/12 at K=6 (v1); `0.25/K` only fires at
K=5/6. **Multiple ghosts in the same cell, confirmed:** 3 cells (v1) have 2 communities
simultaneously below `0.5/K` — `T1/C1/K6` (0.063, 0.068 — plus 4 others 0.167–0.321),
`T1/C3/K6` (0.044, 0.078), `T2/C3/K6` (0.046, 0.073) — a single `min_share` reading would
have hidden the second ghost in each. v2 reproduced `C1/K6` and `C3/K6` at nearly identical
magnitudes (`T1_v2/C1/K6`: 0.073, 0.074; `T1_v2/C3/K6`: 0.067, 0.078) and surfaced one new
multi-ghost cell v1 didn't have (`T2_v2/C4/K5`: 0.052, 0.100). Not purely a non-convergence
artifact: 12 of the 15 v1 cells flagged at `0.5/K` converged normally.
Scripts: `ghost_test3_share_distribution.py`, `ghost_test3v2_share_distribution.py`.

### Deep dive on the 3 originally-flagged cells — per-community, not aggregated

For each cell's 2 lowest-share communities: `Z_scaled` diagonal per relation (the
gauge-safe, permutation-corrected within-community reconstruction strength), entity-weighted
domain balance (`r_k`, ticket 82's own machinery), and facet dominance (entity-count-scaled
share of the community's membership mass, by facet).

**Domain balance:** 5 of the 6 communities examined sit within the existing ±0.15 tolerance
band (`TOL`, ticket 82 D3) — `dev_k` 0.027–0.135. One exception: `T2/C3/K6`'s second ghost
(`dev_k=0.160`) is the only one of the six to actually exceed `TOL`, and both of that cell's
ghosts lean semantic (`r_k` 0.34–0.37).

**Facet dominance:** in the two `T1` cells, both flagged communities draw from 4+ facets
spanning both domains, none exceeding ~28–34% of the community's mass — **not mono-facet**.
`T2/C3/K6` is more concentrated (fringe_atom+core_atom+core_child_he+cousin_he = 82–89% of
each ghost's mass, social facets combined only 9–12%) — the one cell where the secondary,
descriptive properties (domain skew, facet concentration) actually coincide with low mass;
the other two show low mass **without** either secondary property, directly illustrating why
the user's definition treats them as secondary, not defining.
Script: `ghost_suspicious_cells_investigation.py`.

### `S_Art_Auth` structural concentration — universal, not ghost-specific, present at every K

The diagonal of `S_Art_Auth`'s `Z_scaled` (the within-community reconstruction strength for
article-author ties) concentrates on far fewer than K communities, **in every one of the 6
configs, at every K from 2 to 6, in both slices, in both data versions** — not a property of
the 3 originally-flagged cells specifically. Measured as top-N share of the diagonal against
its fair share `N/K` (mean across all 24 (config × slice) cells per K, v1):

| K | top-1 (fair) | top-2 (fair) | top-3 (fair) |
|---|---|---|---|
| 2 | 0.83 (0.50) | 1.00 (1.00) | 1.00 (1.00) |
| 3 | 0.70 (0.33) | 0.97 (0.67) | 1.00 (1.00) |
| 4 | 0.64 (0.25) | 0.89 (0.50) | 0.99 (0.75) |
| 5 | 0.56 (0.20) | 0.83 (0.40) | 0.94 (0.60) |
| 6 | 0.52 (0.17) | 0.77 (0.33) | 0.93 (0.50) |

Not an artifact of choosing N=2 (the value used earlier in this investigation before it was
challenged and rechecked at N=1 and N=3) — the pattern holds at every N tested. `v2` matches
`v1` closely at every K (K=6: top-1/2/3 = 0.49/0.75/0.90 vs v1's 0.52/0.77/0.93), expected
since `chunk12v2` never touches `S_Art_Auth`, and a useful stability check on the measurement
itself. Per-config spread at K=6 is real: `C2`/`C6` reach top-1=0.97 (one community holds
nearly the whole relation's diagonal mass); `C4`/`C5` are comparatively less extreme (top-1
0.31–0.34) but still 2–3× their fair share (0.167).

**Working explanation, not yet a proven mechanism:** `S_Art_Auth` has only 218 (T1) / 258 (T2)
non-zero ties total (§11), spread across up to 6 communities — genuinely little data per
community to differentiate 6 separate author-article patterns from. This is offered as the
most likely cause of the structural ceiling, not as a demonstrated one; no direct test of
"more data would fix this" was run (would require a differently-sized corpus).
Script: `ghost_article_degree_vs_winner_loading.py`'s companion concentration pass is folded
into `ghost_auth_collinearity_and_concentration.py`.

### `auth`/`affil` collinearity, extended past FINDINGS §4's original K≤4 ceiling — confirms
one config, rules out another, reproduces identically across data versions

FINDINGS §4 established `auth`/`affil` community-profile collinearity (`c_u` off-diagonal
cosine-similarity) emerges sharply at K≥4, previously only trustworthy up to K=4 (one K=6
example there hit the epoch ceiling). Extended cleanly to K=5/6 here (all cells below
converged) and checked directly against the flagged ghost pairs, both data versions:

| Cell | `cu_auth[k1,k2]` | Mutual best match? | Verdict |
|---|---|---|---|
| `T1/C1/K6` | 0.699 | Yes (k1↔k2) | Collinearity **confirmed** |
| `T1_v2/C1/K6` | 0.820 | Yes (k1↔k2) | Collinearity **confirmed**, same config/K, v2 |
| `T1/C3/K6` | 0.011 | No | Collinearity **ruled out** |
| `T1_v2/C3/K6` | 0.009 | No | Collinearity **ruled out**, reproduces v1 |
| `T2/C3/K6` | 0.006 | No | Collinearity **ruled out** |
| `T2_v2/C4/K5` (new v2 ghost) | 0.011 | No | Collinearity **ruled out** |

**Config-specific, not universal, and reproduces identically between v1 and v2** — the same
config (`C1`) shows strong, mutual collinearity in both data versions at nearly the same
magnitude; the same config (`C3`) shows none in either. This means the same overall symptom
(low share) has at least two distinct underlying causes across configs, not one universal
explanation. Script: `ghost_auth_collinearity_and_concentration.py`.

### Article-side reading — ghosts are not content-starved; the deficit is specifically on
the author side

Checked directly (not inferred) whether the article (`art` facet) embedding is thin for
flagged ghosts, given `S_Art_Journ` (sharing the same `art` embedding) was often strong for
the same communities where `S_Art_Auth` was weak. Measured each flagged community's article-
side membership sum, ranked against the other communities in the same cell, across all 6
flagged cells (3 v1 + 3 v2):

In 4 of 6 cells, at least one flagged community's article-side sum is in the top 2 of the
whole fit (often the single highest) — `T1/C1/K6` idx4, `T1/C3/K6` idx3 (highest of all 6,
5 of 25 live articles >0.5 membership), `T2/C3/K6` idx5, `T1_v2/C1/K6` idx4 (highest of all
6). One cell inverts (`T2_v2/C4/K5`: one ghost strong, rank 2 of 5; the other the single
weakest article-side community in that fit). One cell (`T1_v2/C3/K6`) shows both merely
middling. **General conclusion, corrected from an earlier over-generalisation based on only
3 cells: low overall share does not reliably mean low article-side content — most flagged
communities are genuinely well-anchored to real articles — but this is not a universal rule
across all 6.** Script: `ghost_article_side_reading.py`.

### Article degree vs. winning-community loading — the "mega-author paper" hypothesis tested,
not supported

Hypothesis: articles with unusually many co-authors ("PALM-style" papers) disproportionately
determine which 2 communities dominate `S_Art_Auth`'s diagonal (the "winners"). Tested by
correlating each live article's raw co-author count (row `nnz` in the **undamped-sparsity-
pattern** `S_Art_Auth` matrix — damping changes tie weight, not which ties exist, so this is
degree-independent of ticket 63's log-damping) against its combined `U_prob` membership on
the 2 winning communities, all 6 configs, K=6, both slices (12 cells).

**Not supported.** 10 of 12 Spearman correlations are weak and not significant (`p>0.05`).
The one significant result (`T2/C3/K6`, ρ=−0.446, p=0.006) runs **opposite** to the
hypothesis — higher-degree articles have *lower* winner-community membership there, not
higher. The single highest-degree article per slice (65 co-authors, T1; 20, T2) behaves
inconsistently across configs — winner-pair membership ranges from 0.000 (`T2/C2/K6`,
`T2/C3/K6`) to 1.000 (`T2/C5/K6`) depending purely on which config's topology it sits in.
No general "mega-author papers drive the winners" pattern found.
Script: `ghost_article_degree_vs_winner_loading.py`.

### Affiliation-removal pilot — config-dependent, matching the collinearity split exactly

User proposal: since affiliations could in principle be re-attached at a post-hoc analysis
stage rather than fit jointly, does excluding `S_Auth_Affil` (hence the `affil` facet)
entirely from the fit change the picture? Piloted (not a full grid) via a local `soc_keys`
filter — `fit_instrumented`'s body copied verbatim with one line changed
(`soc_keys = [k for k in soc_keys if k != 'S_Auth_Affil']`), no change to `chunk13v9.py` —
on `C1/K6/T1` and `C3/K6/T1`, single seed:

| Cell | Metric | With `affil` | Without `affil` |
|---|---|---|---|
| C1/K6 | `cu_auth` max off-diagonal | 0.699 | **0.505** |
| C1/K6 | `S_Art_Auth` top-2 share | 0.818 | **0.519** (4 of 6 communities now get real diagonal mass, was 2) |
| C1/K6 | lowest community share | 0.063 | 0.059 |
| C3/K6 | `S_Art_Auth` top-2 share | 0.963 | 0.975 (no improvement) |
| C3/K6 | lowest community share | 0.044 (community index 3) | 0.039 (**same index, 3**) |

**Removing affiliations measurably helps `C1` and does essentially nothing for `C3`** — the
same config split the collinearity check found, now reproduced by an independent
intervention (removing a whole relation, not just measuring). `C3`'s ghost persists at
nearly the same magnitude, in the same community index, despite the architectural change —
evidence something intrinsic to that community (or a limitation not related to `affil`) is
responsible there. Single-seed pilot on 2 cells, not yet a full grid; user has requested the
full C1–C6 × K∈{2..6} × both data versions grid as a follow-up, not yet run.
Script: `ghost_no_affil_ablation.py`.

### Loss-weight sensitivity — outer-loop penalties structurally cannot confound; `lambda_z_offdiag` does, materially

Checked whether any of this investigation's findings could be an artifact of the specific
loss weights used throughout (`lambda_l1=0`, `lambda_z_offdiag=0.05`,
`lambda_domain_balance=0`), rather than a property of the topology/data.

**Outer-loop penalties (`collapse_pen`, `coherence_pen`, `semantic_pen`,
`domain_balance_pen`) cannot be a confound, by construction, not just probably** — per §4.1's
Epistemic Boundary, they are computed strictly post-hoc in Module 3 and never appear inside
`run_inner_solver`'s loss; there is no code path by which they could have shaped the fitted
`U`/`Z` values this investigation measured.

**`lambda_z_offdiag` does matter, materially, for the collinearity finding specifically.**
Refit `C1/K6` and `C3/K6` (T1, single seed) at `λz∈{0.0, 0.05 (used throughout), 0.5}`:

| Cell | λz | `S_Art_Auth` top1/2/3 | `cu_auth` max off-diag | lowest community share |
|---|---|---|---|---|
| C1/K6 | 0.0 | 0.49/0.96/0.98 | **0.263** | 0.079 |
| C1/K6 | 0.05 | 0.41/0.82/0.97 | 0.699 | 0.063 |
| C1/K6 | 0.5 | 0.46/0.91/0.95 | 0.865 | 0.032 |
| C3/K6 | 0.0 | 0.53/0.92/1.00 (not converged) | **0.055** | 0.077 |
| C3/K6 | 0.05 | 0.87/0.96/0.98 | 0.973 | 0.044 |
| C3/K6 | 0.5 | 0.81/0.93/0.97 | 0.796 | 0.055 |

**Two separate conclusions, not one.** The `S_Art_Auth` concentration itself (top-1/2/3
share) stays elevated well above fair share at every tested `λz` in both cells — that part of
the finding is not a `λz` artifact. But `auth`-facet collinearity rises sharply with `λz` in
both cells — at `λz=0` it drops to 0.263 (C1) and 0.055 (C3), both far below the
production-setting values reported above as "confirmed"/"ruled out." **The qualitative
ordering survives at every tested `λz`** (C1 more collinear than C3 throughout) but **the
magnitude reported earlier in this section is inflated by the off-diagonal penalty itself,
not purely a topology-intrinsic property** — worth this explicit correction rather than
letting the earlier, uncaveated numbers stand as the full picture. Single-seed, 2-cell check,
not a full sweep; `C3`'s `λz=0` run did not converge, weakening confidence in that one data
point specifically. Script: `ghost_lambda_z_sensitivity.py`.

### Connection to already-existing rotational/permutation work — real scope gaps, not yet closed

FINDINGS §18 already established, via a two-tier test (near-separability + a direct local-
**and-distant** rotation-feasibility search, both re-run once under a corrected monitoring
mask), that genuine community-blending freedom is essentially absent — not merely narrow —
for this model. **That evidence covers K∈{3,4}, T1 only.** This session's grid (K∈{2,3,4,5,6},
T1/T2, both data versions) is materially wider. Not yet extended; a real, named gap, not an
oversight to gloss over. Similarly, FINDINGS §14's permutation-conflict audit (zero genuine
cross-relation conflicts found) covers the same K∈{2,4} scope and has not been re-run at
K=5/6, where more communities mean more possible permutations and a higher chance of a
genuine conflict — the exact boundary condition FINDINGS §14 itself already flagged as
untested.

### What remains open, not yet done

- Full affiliation-removal grid (C1–C6 × K∈{2..6} × T1/T2/T1_v2/T2_v2) — user-requested,
  scoped (~120 fits, cheap individually), not yet run.
- Rotation-feasibility (§18) and permutation-conflict (§14) checks, both scoped to K∈{3,4}/
  T1 only — not yet extended to this ticket's wider K/slice/data-version grid.
- Seed-noise floor for `community_share`/`min_share` (the planned "Test 4") — not started;
  needed before any specific cell's "ghost" verdict can be checked against ordinary
  seed-to-seed variation, the same discipline already applied to `TOL` (ticket 82 D3).
- No mechanism (in-loop or outer-loop) has been designed or proposed — this ticket remains
  diagnostic-only throughout, matching the sequencing the user set at its start ("we need to
  measure first").
- `lambda_z_offdiag` sensitivity checked on only 2 cells, single seed — the qualitative
  C1-vs-C3 split held up, but the magnitudes at production settings should be read with the
  inflation above in mind until a wider sweep exists.

### Files

- New diagnostic scripts (all `chunk13_execution/diagnostic_scripts/`):
  `ghost_test1_gauge_invariance.py`, `ghost_test2_dead_entity_check.py`,
  `ghost_test3_share_distribution.py`, `ghost_test3v2_share_distribution.py`,
  `ghost_suspicious_cells_investigation.py`, `ghost_auth_collinearity_and_concentration.py`,
  `ghost_article_side_reading.py`, `ghost_article_degree_vs_winner_loading.py`,
  `ghost_no_affil_ablation.py`, `ghost_lambda_z_sensitivity.py`. Results in
  `diagnostic_results/` under matching names.
- No changes to `chunk13v9.py` or `diagnostic_blocks.py` — every measurement above reused
  existing, already-validated functions (`_relation_community_share`,
  `_hungarian_relabel_relation`, `facet_membership_profile`, `domain_balance_r_k`,
  `build_presence_masks`, `fit_or_load`/`fit_instrumented`) or, for the affiliation and
  `lambda_z_offdiag` pilots, a local fork of `fit_instrumented`'s body confined to the
  diagnostic script itself.

---

## 24. Ticket 84 follow-up — journal-venue resolution false negatives found and mostly
fixed; the fix reaches no article in this toy corpus's fitted data; root cause of that
traced through the full ingestion pipeline; one process-safety incident, fully recovered

**Status: journal-resolution logic corrected and re-run; `chunk12.py`/`chunk12v2.py`
rebuilt; verified byte-identical to before — the correction reaches no article that is
actually part of this toy corpus's 61-article valid cross-section. Root cause traced to
postprocessing, not to text availability or graphbrain parsing. One destructive incident
during this investigation, recovered from a NAS snapshot, no data lost.**

### False negatives found in the original D4 resolution (ticket 84, `journal_repo_resolution.py`/`_v2.py`)

D4 (CLAUDE.md §4.21) resolved 10 of 59 originally repository-hosted articles to a real
venue via a live OpenAlex title search, leaving 49 "UNRESOLVED, confirmed no non-repository
candidate." Six well-known machine-learning papers among those 49 were checked directly
against the literature and against OpenAlex's own data: ALBERT (ICLR 2020), T5 (JMLR), PaLM
(JMLR), InstructGPT (NeurIPS 2022), BLIP-2 (ICML 2023), and "Judging LLM-as-a-Judge" /
MT-Bench (NeurIPS 2023 Datasets & Benchmarks track). **All six have a real, non-repository
venue.** Only LLaMA, checked as a control, is a genuine true negative — arXiv-only, no
peer-reviewed venue exists for it.

**Root cause, confirmed against a specific record, not inferred:** the original resolver's
live-search step checked only a candidate work's `primary_location.type`, never its full
`locations` array. T5's own OpenAlex record (`W2981852735`) lists four locations — three
arXiv entries and one `Journal of Machine Learning Research` entry — but `primary_location`
happens to be arXiv, so the resolver discarded a candidate whose correct venue was present,
unchecked, in the same record.

### Fix built and its actual yield

`journal_repo_resolution_v3.py` (new): a `select_best_location()` function inspects a
candidate's full `locations` list and ranks by a source-type hierarchy — `journal` >
`conference` > `book series` > `ebook platform` > anything else non-repository, `repository`
excluded unconditionally — applied the same way across all three resolution paths
(same-record locations, cached related-works, live title search; the original code had three
separate, inconsistent pieces of this check, and the bug above was confirmed in only one).

**Yield: 1 of the 49 originally-unresolved articles was actually fixable this way — T5, now
resolved to JMLR.** The other five checked papers (ALBERT, PaLM, InstructGPT, BLIP-2,
MT-Bench) remain correctly unresolved: their own OpenAlex records were checked directly and
genuinely list only arXiv, with no non-repository location merged in — a gap in OpenAlex's
own data for these specific records, not something a script fix can close.

**Two further defects found and fixed during implementation, both caught before reaching
production data:**
1. The repository-exclusion check originally recognized only the 4 IDs named in D4 (arXiv,
   bioRxiv, ChemRxiv, Research Square). Two more sources already present in this corpus's
   own `journal_id` values — PubMed and DROPS (Schloss Dagstuhl's open archive) — are also
   OpenAlex `type == "repository"` sources, never excluded since neither is one of the 4
   named IDs. Fixed: excludes anything OpenAlex classifies as `type == "repository"`, not
   only the 4 named IDs.
2. One title in `articles.csv` (the T5 article) contains a literal two-character sequence —
   a backslash followed by the letter "n" — where the source text should have a space or
   line break, not an actual newline character. This caused the first attempt at the fix to
   still report T5 unresolved: the live search treated the literal `\n` as text to match,
   and no genuine title contains it. Confirmed directly via `repr()` inspection; also
   confirmed this is the only title in the 121-row corpus with this specific corruption
   (a full regex scan found no others). Fixed by stripping literal backslash-escape
   sequences before constructing any search query or similarity comparison.

### `venue_type` enrichment (user-requested)

`articles.csv` gained a `venue_type` column for all 121 rows, populated via one OpenAlex
Sources lookup per unique `journal_id` (46 unique journals, not per article) — resolves to
`journal` (54 rows), `repository` (48, after the fix), `conference` (7), `book series` (1),
or nothing (9 rows with no journal recorded). The two newly-found repository sources
(PubMed, DROPS) were found this way, incidentally, on articles that were never part of the
original 59-article trigger set.

A persisted reference, `toy_large/known_repository_sources.json`, records the 6 confirmed
repository sources found so far (id, display name, first-seen context) — checked first by
`journal_repo_resolution_v3.py` before any live Sources lookup (confirmed directly: zero
network calls for anything already listed), and appended to automatically whenever a new
`type == "repository"` source is encountered. Purpose: avoid re-discovering the same
repositories via the API a second time at the 22k-article corpus. **Explicitly scoped to
what this toy corpus has encountered, not a claim about the complete universe of OpenAlex
repository-type sources** — an early presentation of this list risked being read as the
latter and was corrected directly when the user asked.

**Raised by the user, not investigated:** whether OpenAlex's `type` field itself has false
negatives — specifically, whether something that functions as a repository in practice (a
reference-management or citation-aggregation tool was named as the motivating example) could
be classified `type == "other"`/`"metadata"` rather than `"repository"`, which the current
hierarchy would currently accept as a valid low-priority venue rather than exclude. Checked
for the one specific named example — not present anywhere in this corpus's data, and no
record of it having been raised earlier in this investigation either. The general concern is
real and not addressed by the current design; left open.

### `chunk12.py`/`chunk12v2.py` updated and rebuilt — no effect on the fitted data

Both files' `REPO_JOURNAL_IDS` (previously a hardcoded 4-ID literal, identical in both
files) now load from `known_repository_sources.json` at runtime instead — one shared,
growing source of truth, matching how D4's original fix already touched both files
identically.

Both regenerated. **Full null-rebuild verification: every relation matrix, every entity
map, and `journal_meta` came out byte-identical to the pre-update pickles, in both the
regular and `_v2` versions, both slices.** This was initially reported as an unexpected
result — the working assumption going in was that `S_Art_Journ`'s tie count would move (T5
gaining a tie, the PubMed/DROPS-hosted articles losing theirs) — and the expectation was
corrected, not the finding: none of the three affected articles are part of this corpus's
61-article valid cross-section (`meta_set ∩ edge_set ∩ sqlite_set`, `chunk12.py`'s own
validity filter) — each fails the `sqlite_set` membership check specifically (absent from
`corpus_curated.sqlite`), independent of journal. The fixes are real and correctly in
place; they simply have nothing to reach in this corpus today.

### Why these three articles are absent from `corpus_curated.sqlite` — traced through the full pipeline

User-requested follow-up: check abstract availability for T5, the DROPS-hosted article
("HISTORIAE..."), and the PubMed-hosted article ("Understanding the Natural Language of
DNA...").

| Stage | File | Result |
|---|---|---|
| Raw abstract text | `corpus_text.csv` | Present for all 3 — 1028 / 955 / 1182 characters |
| Graphbrain parsing | `corpus_raw.sqlite` | Successfully parsed, all 3 — real `source`-typed sentence edges (6 / 7 / 4 sentences respectively) |
| Parse error log | `parse_errors.log` | No entries for any of the 3 |
| Postprocessing / curation | `corpus_curated.sqlite` | Absent, all 3 |

Traced the exact mechanism, not just the outcome: re-ran `postprocessing_4h.py`'s own
`extract_dual_matrices()` against every one of the 17 raw parsed sentences belonging to
these 3 articles (isolated the function definitions from the file's execution code, spaCy
lemmatizer stubbed to surface-form passthrough — the same technique already used earlier
this project on the sibling `7.5postprocessing_4hbased_correct.py`; does not change which
atoms land in which structure). **Every one of the 17 sentences returned `None` — no
sentence in any of the three abstracts contains the predicate-argument structure
`extract_dual_matrices` requires before it will write anything to the curated database.**
Not a parsing failure and not a text-availability gap — a structural filter that these
three abstracts' sentences uniformly fail to clear.

**Version mismatch found, not yet acted on:** `corpus_curated.sqlite` (mtime Jun 15 23:02)
was built by `postprocessing_4h.py` (mtime Jun 15 22:19, ~43 min earlier). A corrected
version of the same extraction logic, `7.5postprocessing_4hbased_correct.py` — previously
used only for isolated verification on a single sampled sentence (the D8/`M_Cousin_Child`
structural-verification note in the ticket84 plan file) — is dated Jun 16 22:51, roughly a
day *after* the curated database was built. **The corrected logic has never been used to
rebuild the production curated database.** Whether its corrections would change the outcome
for these 3 articles, or for others, has not been tested — flagged for whoever decides
which pipeline stages to re-run.

### Process-safety incident (2026-09-01) — data destroyed and fully recovered, no loss

While isolating `postprocessing_4h.py`'s function definitions for the test above, the
file's source was `exec()`'d up to its `# EXECUTION` marker on the assumption that
everything before that marker was side-effect-free function/constant definitions. **It is
not** — a module-level operation before that marker purges `corpus_curated.sqlite` (visible
from its own printed message, "Previous curated matrix purged," missed at the time). The
61.9MB curated database was destroyed; a subsequent verification call against the
now-missing path silently recreated an empty 20KB file at the same path via
`graphbrain.hgraph()`'s auto-create behaviour, compounding the loss until it was noticed
directly.

**Fully recovered, verified byte-for-byte, no data lost.**
`/mnt/hum01-home01/p91688di/tensor_data_staging/` sits on a NAS mount with hourly (24h) and
daily (28-day) Isilon snapshots (already documented as available, ticket84 plan item 6, for
a different file). The daily snapshot `Reynolds_HUM_NFS_28days_2026-08-04_00-30-00` held the
exact pre-incident file (61,935,616 bytes, matching to the byte). Restored via a plain `cp`
from the read-only `.snapshot/` path (run by the user directly, after the harness's own
permission classifier declined to run the copy from this session). Verified post-restore:
177 `source_core` edges, 66 distinct articles — matching the pre-incident state exactly, and
the three target articles confirmed still absent, consistent with the finding above and
undisturbed by the incident.

**Standing precaution added as a result** (`SESSION_PROTOCOL.md` §E): read a script's
entire file for side effects before executing any part of it, specifically naming both
postprocessing scripts.

### Files

- **New:** `chunk13_execution/diagnostic_scripts/journal_repo_resolution_v3.py` (fix +
  `venue_type` enrichment).
- **New:** `toy_large/known_repository_sources.json` (persisted repository-source cache, 6
  entries).
- **Modified:** `toy_large/chunk12.py`, `toy_large/chunk12v2.py` (`REPO_JOURNAL_IDS` now
  loaded from the JSON file, both identically); `tensor_data_staging/articles.csv` (T5's
  `journal_id`/`journal_name` corrected; `venue_type` column added for all 121 rows).
- **Regenerated:** all 6 production pickles (`Star_extended_matrices_t1/t2[.pkl]`,
  `Star_epistemic_decoders_global[.pkl]`, and their `_v2` counterparts) — verified
  byte-identical to their pre-update state; pre-update copies kept at
  `outputs/pre_v3_backup/` pending confirmation they can be removed.
- **Recovered, unmodified in content:** `toy_large/corpus_curated.sqlite` (restored from a
  NAS snapshot after the incident above; its content is unrelated to, and unaffected by,
  any of this session's other changes).
- **Not yet done:** whether/how to re-run preprocessing or postprocessing to recover these 3
  (or other, unsurveyed) articles is left to the user's decision; whether
  `7.5postprocessing_4hbased_correct.py`'s corrections would change today's postprocessing
  outcome for any article has not been tested; the broader `type=="other"` false-negative
  concern above has not been investigated beyond the one named example.

---

