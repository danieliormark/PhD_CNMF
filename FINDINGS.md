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

**Status: Open. Candidate identified, not yet implemented.**

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
distant-rotation extension below, itself now in need of a corrected re-run under the same
absolute-floor mask before its own numbers should be fully trusted — flagged, not yet done).
Within that scope, the corrected finding **strengthens, not weakens**, §18's practical
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

**Caveat added on the later audit pass: this extension inherits the same flawed
`MEANINGFUL_FRAC` mask the bisection search above has since been corrected for, and its
own cross-check ("recovered cage widths… independently match the bisection-based margins")
validated against the now-superseded 0.5°–3.4° numbers, not the corrected ≈0° ones.** The
"0 islands" conclusion is not necessarily wrong — a stricter, more-monitored feasibility
check can only shrink or preserve a feasible region, not create a new disconnected one out of
nothing, so this result plausibly survives the fix — but it has not been re-run under the
corrected absolute-floor mask, and should be before being relied on with the same confidence
as the corrected bisection result above.

Diagnostic script: `diagnostic_scripts/rotation_island_search.py`, results in
`diagnostic_results/rotation_island_search.json`. **Not yet re-run under the corrected mask —
see caveat above.**

### Practical implication (Tier 1 + Tier 2 combined)

On this corpus, at production settings (`lambda_l1=0.0`, `lambda_z_offdiag=0.05`), the
combined evidence is that Layer 1b's blending freedom is **essentially absent locally, not
merely narrow** — not the unconstrained "any invertible `W`" freedom the raw linear-algebra
identity permits, and this is now a cleaner, stronger conclusion than the "narrow but real"
framing this section originally reported (see the corrected Tier 2 result above — the earlier
0.5°–3.4° margins were an artifact of a flawed monitoring mask, not a real feasibility bound).
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

