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

**Status: Supported. Toy-corpus scope. Mechanism (why) not investigated — this measures
whether, not why.**

### The gap in §1 as originally stated

§1's identity — `(U[f1]·W)·(W⁻¹·Z·W⁻ᵀ)·(U[f2]·W)ᵀ = U[f1]·Z·U[f2]ᵀ` for any invertible
`W` — is correct, but it is an **unconstrained** linear-algebra fact. It says nothing about
whether a given non-trivial `W` keeps `U·W` and the transformed `Z` **non-negative**, which
this model requires everywhere (`clamp_(min=1e-7)`, ticket 74). A generic invertible `W`
does not preserve non-negativity. §1's "Established" status is correctly scoped to the math;
it was never checked against whether the freedom is actually *reachable* by a real fitted
solution here. This section is that check.

### Method: near-separability (Tier 1 of a two-tier test; Tier 2 not yet run)

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

**Failures at the 0.85 bar** (7 of ~104 confounded-facet cells; none in the clean facets):
`fringe_atom` is the recurring facet (`C1/K4`, `C3/K4`, `C4/K4`), 3 of these not explained by
the Part B/low-degree confound. `C4/K4` did not converge (flag per `SESSION_PROTOCOL` rule
4) — its failure should not be read as a genuine finding without a converged re-fit.

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

**Methodological correction, kept here since it matters for reusing this method later.** The
first run used an absolute tolerance (reject any monitored entry below −1e-4) and returned
0.00° margin for all 54 pairs, uniformly — too clean to trust. Diagnosis: many entries sit at
the model's own zero-floor (`clamp_(min=1e-7)`) next to a large entry in the same row: any
nonzero rotation angle nudges the floor entry by roughly `angle × (the large entry's size)`,
crossing −1e-4 at a vanishingly small angle regardless of whether real, meaningfully-supported
structure is rotatable. The test was re-detecting "this model has structural zeros"
(expected, uninteresting) rather than answering the intended question. **Fixed**: only
monitor entries that carried meaningful mass *before* rotating (≥1% of that column's/
relation's own max value, computed once from the un-rotated `U`/`Z`); entries below that are
excluded from the feasibility check entirely. Re-run below is the corrected version; the
original 0.00°-everywhere run is void and not used anywhere in this document.

**Result:** all 54 community pairs, across every cell, have a real, non-zero, but **bounded**
local margin — roughly 0.5°–3.4°, no pair pinned at 0° and no pair free out to the full 90°
search range. Practical scale: `sin(3°) ≈ 0.05`, so even the loosest pairs found allow on the
order of a few percent of mass to be locally redistributed between two communities before
non-negativity breaks — not a wholesale relabeling-scale swap. Per-cell tightest-pair margin
ranged `C5/K4` (0.55°, most constrained) to `C6/K4` (0.95°, least constrained); `C3/K4` and
`C4/K4` did not converge and their numbers (0.60° and 0.65° respectively, within the same
range as everything else) should be read with that caveat per `SESSION_PROTOCOL` rule 4.

**Scope of what this establishes:** a **local** (first-order/infinitesimal) feasibility
bound around the fitted solution — it characterizes the immediate neighborhood, not whether
some large, distant rotation could loop back to a different globally-feasible point; that is
a harder, different question, not attempted here. Within that scope, it is a direct,
constructive answer (stronger than Tier 1's indirect evidence): genuine local blending
freedom exists everywhere tested, but it is small and bounded, not large or unbounded,
anywhere in the grid.

### Practical implication (Tier 1 + Tier 2 combined)

On this corpus, at production settings (`lambda_l1=0.0`, `lambda_z_offdiag=0.05`), the
combined evidence is that Layer 1b's blending freedom is real but narrow — not the
unconstrained "any invertible `W`" freedom the raw linear-algebra identity permits, and
nowhere near large enough to plausibly explain a full community swap or make the model's
substantive outputs (`U_prob` community membership, `Z`-based within/between coupling) for
one specific reported/selected fit untrustworthy at the level of interpretation this project
needs. The ordinary caveat still applies regardless: cross-fit/cross-seed comparisons still
require relabeling correction (§14/§16/§17). `fringe_atom` (Tier 1's recurring failure) is
worth extra scrutiny before relying on its community assignments across different fits.
Toy-corpus-calibrated like every other finding in this document — re-derive at 22k-article
scale before trusting; a denser corpus could plausibly tighten or loosen these margins in
either direction.

Diagnostic scripts: `diagnostic_scripts/near_separability_check.py` and
`diagnostic_scripts/rotation_feasibility_search.py`; results in
`diagnostic_results/near_separability_check.json` and
`diagnostic_results/rotation_feasibility_search.json`.

---

