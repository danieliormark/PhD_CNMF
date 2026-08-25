# CLAUDE.md — Chunk 13v9: Collective NMF for Heterogeneous Epistemic Communities

> Read this file in full before editing `chunk13v9.py`. It encodes design decisions
> that look like bugs but are not, and it records the open defect register.
>
> **Companion files, read alongside this one:** `FINDINGS.md` — the analytical record
> (what has been measured about the model, refuted hypotheses, open questions; read before
> proposing changes to the loss function, metrics, or interpretation of results) and
> `SESSION_PROTOCOL.md` — standing conditions for diagnostic/patch sessions (standard run
> settings, reporting rules, what's out of scope by default) so prompts don't need to
> re-state them each time.

---

## 1. Research Purpose

This pipeline runs a **collective non-negative matrix tri-factorisation (NMTF)** over a
star-topology metagraph to discover **heterogeneous epistemic communities** in a
scientometric corpus.

A community is heterogeneous when it spans multiple *facets* — semantic hyperedge levels
(atoms, child hyperedges, parent hyperedges, cousin hyperedges), articles, authors,
journals, and affiliations — and both *domains* (semantic and social). The central research
problem is that unconstrained NMF produces communities dominated by a **single facet**
(e.g. only parent hyperedges) or a **single domain** (e.g. only social relations). Most of
the regularisation machinery in this codebase exists to prevent that.

An agent that optimises reconstruction loss alone will silently undo the sociological
constraints. **Do not treat reconstruction fidelity as the sole objective.**

---

## 2. Pipeline Architecture

Single file, four logical modules, executed top to bottom. There is **no package
structure** — modules are navigational sections, not importable units.

| Module | Owns | Key outputs |
|---|---|---|
| **1. Static Architecture** | Topology, `RELATION_MAP`, all domain constants, data loading, dimension inference, presence masks | `soc_keys`, `sem_keys`, `anchor_keys`, `dimensions`, `clean_data`, `build_presence_masks()` |
| **2. Inner Solver** | PyTorch Adam PGD factorisation, NNDSVD init, multi-hop propagation | `U_final`, `Z_final`, `diagnostics` |
| **3. Meta-Evaluator** | Sociological penalties (collapse, coherence, socio-semantic), Optuna objective | `sociological_penalty`, `objective` callable |
| **4. Master Execution** | Adaptive grid dispatch, Pareto extraction, archiving, stability analysis | `.db` studies, `.pt` tensors, JSON manifests |

Data flow: `Module 1 (load) → Module 2 (fit) → Module 3 (evaluate) → Module 4 (orchestrate)`

`create_optuna_objective`'s `objective(trial)` (Module 3 §5.2) calls `evaluate_complete_solution()`
(Module 3 §5.1) directly rather than re-running the evaluation sequence inline — this is the
*only* place Module 4's archiver (§S4) and stability analysis (§S5) and the Optuna loop all
compute the sociological penalty, closing the loop required by §4.13.

---

## 3. Function Contracts

These boundaries have broken repeatedly. **Verify signatures before changing any of them.**

```python
get_active_facets(config_id)
    -> (soc_keys: list, sem_keys: list, anchor_keys: list)

get_required_facets(soc_keys, sem_keys, anchor_keys=None)
    -> sorted list of FACET names (not relation names)

load_and_validate_data(filepath)
    -> clean_data: dict of {relation_key: scipy_sparse} PLUS key 'dimensions'
       (caller must extract: dimensions = clean_data.get('dimensions'))

build_presence_masks(matrices, soc_keys, sem_keys)
    -> dict[facet -> bool ndarray], True where that entity has >=1 non-zero
       entry in ANY active relation touching its facet (union, not any single
       relation). Must be called fresh per config/slice from the matrices
       actually loaded — never cached or hardcoded. See §11.

run_inner_solver(raw_data, soc_keys, sem_keys, anchor_keys, K,
                 dimensions, params, device, seed_function, ...)
    -> (U_final: dict[facet -> torch.Tensor],
        Z_final: dict[relation -> torch.Tensor],
        diagnostics: dict)

diagnostics keys:
    "math_loss"          float   pure reconstruction loss, pre-regularisation
    "internal_soc_loss"  float   lambda-weighted L1 + Z-offdiag terms
    "loss_history"       list    per-epoch total_loss
    "U_scales"           dict    facet -> np.ndarray shape (K,)
    "converged"          bool    early-stopping flag

evaluate_complete_solution(...)
    -> dict with EXACTLY these keys (all native Python float, see ticket 68):
       "collapse_pen", "collapse_score", "coherence_pen",
       "weakest_coherence", "semantic_pen", "sociological_penalty"
    Builds presence_masks internally via build_presence_masks() and threads
    them into evaluate_dimensional_collapse / evaluate_socio_semantic_reality.
    This is the single source of truth — call it, don't reimplement its
    sequence at a new call site (see ticket 69).

create_optuna_objective(...) -> objective(trial)
    objective returns TUPLE (pure_recon_loss, sociological_penalty)
    Study MUST be created with directions=["minimize", "minimize"]
    objective() internally calls evaluate_complete_solution() — do not
    re-inline the evaluation sequence (ticket 69).
    Sets trial.set_user_attr("converged", bool) and ("epochs_run", int)
    (ticket 75) — a trial that hits the epoch ceiling still RETURNS its
    actual (pure_recon_loss, sociological_penalty), it is only FLAGGED, not
    pruned or penalised. Any downstream consumer of study.best_trials must
    filter on user_attrs["converged"] itself — it is not automatic.
```

---

## 4. Locked Design Decisions

Each of these was reached deliberately. **Do not revert without discussion.**

### 4.1 The Epistemic Boundary (separation of concerns)

Module 2 computes **only differentiable mathematical loss** — Frobenius reconstruction,
L1 sparsity, squared Z off-diagonal penalty. It is **prohibited** from computing
sociological metrics.

*Rationale:* sociological constraints (entropy collapse, topological coherence,
socio-semantic reality) require discontinuous operations — thresholding, boolean masking,
Hungarian assignment — that break autograd graphs. These live strictly in the outer loop
(Modules 3–4) and operate on **detached** tensors.

### 4.2 Penalise `Z_scaled`, never `Z_pos`

The model is scale-invariant: `U_norm @ Z_scaled @ U_norm.T == U_pos @ Z_pos @ U_pos.T`.
Penalising `Z_pos` opens a loophole — Adam inflates `U_raw` and shrinks `Z_pos`, driving
the penalty to zero while the actual cross-community coupling in `Z_scaled` is unchanged.
Penalising `Z_scaled` closes it.

*Known consequence:* effective penalty strength per community pair varies as
`(scale_f1[k1] * scale_f2[k2])²`. This is accepted, not a bug. Monitor it empirically.

### 4.3 REVISED — `U_norm` is useless as a mass measure, and so is `U_scales` (ticket 79)

**This section previously ended with "use `U_scales` for any mass or volume calculation."
That guidance is wrong and has been retracted.** `U_scales` is not a trustworthy mass
measure — it is an undetermined free direction in the loss, proven algebraically and
confirmed empirically. Full derivation and test results: FINDINGS §12.

Summary: multiply column *k* of `U_pos[f]` by any constant *c*, divide the corresponding
row/column of `Z_pos` by *c* in every relation touching *f* — `U_norm`, `Z_scaled`,
`recon_loss`, `sparsity_loss`, and `z_offdiag_loss` are all provably unchanged (verified to
float precision), while `U_scales[f][k]` changes by exactly *c*. Nothing in the current loss
constrains it. Empirically, `U_scales` for the same facet/community varies by a coefficient
of variation up to 1.2 across differently-initialised runs with near-identical `recon_loss`.

**What remains true and usable:** everything about a column's *shape* — `U_norm`, `U_prob`,
the `c_u` collinearity matrix, the L1/L2 ratio — is unaffected by this transformation and
stays valid. Only *facet-level absolute mass* is unrecoverable. See FINDINGS §12 for the
determined/undetermined split.

**What to use instead for community mass:** `|Z_scaled[k,k]|` — the Frobenius norm of
community *k*'s within-community rank-1 term for a given *relation* (not facet). This
quantity is provably unchanged under the same transformation that moves `U_scales` (it is
not, however, invariant under general rotational indeterminacy — §1 — or under per-relation
permutation — FINDINGS §14 — both of which remain separately open). It is comparable across
relations because every input matrix is Frobenius-normalised to `‖X‖²=1`. Mass becomes
relation-level, not facet-level; see FINDINGS §13 for how domain balance and the collapse
check need to be re-derived on this basis, and for how to handle anchor relations (which
touch both domains) without double-counting the `art` hub.

### 4.4 REVISED — Collapse check formula rests on an undetermined quantity, and its
threshold does not express its own stated target (ticket 79, ticket 80)

The formula as implemented: community mass = `Σ_f (U_scales[f][k] / √N_f)`, normalised
across K, then normalised Shannon entropy, penalised below `ENTROPY_THRESHOLD = 0.60`.

**Two independent problems, not yet fixed in code:**

1. **The mass input is `U_scales`, which §4.3 above shows is undetermined.** Measured
   directly: computing this same mass three ways (raw `U_scales`, `×√N_f`, and the direct
   L1 sum `U_scales × ‖U_norm‖₁`) gave three qualitatively different domain-balance
   readings for the same converged model (FINDINGS §13). The collapse check's *entropy
   verdict specifically* happened to be insensitive to which of the three was used
   (entropy stayed ≥0.93 under all three, in all 6 configs) — but that robustness is
   coincidental to this particular use, not evidence the underlying quantity is sound.

2. **The stated target ("no single community should hold more than ~0.6 of total mass")
   does not correspond to `ENTROPY_THRESHOLD = 0.60`.** Normalised entropy is not a
   monotone function of max-share, and the correspondence is K-dependent:
   - At K=2, a 60% max-share distribution has entropy ≈0.97; the 0.60 threshold instead
     fires only around 86% max-share.
   - At K=4, 60% max-share gives entropy ≈0.80; the 0.60 threshold fires around 75% max-share.
   - Two distributions with identical max-share (e.g. (0.6,0.13,0.13,0.13) vs
     (0.6,0.4,0,0), both max-share 0.6) give entropy 0.80 vs 0.49 — the check would treat
     them oppositely despite equal dominance by the stated criterion.
   A direct max-share formulation (`penalty on max(mass_k)/Σmass_k` exceeding 0.60,
   analogous to `MAX_MONOPOLY`'s formula) would express the target without the K-dependence
   or the non-monotonicity. Not yet implemented — the mass-input problem (point 1) should be
   resolved first, since a corrected threshold applied to an undetermined input gains nothing.

**IMPLEMENTED (tickets 79/80, FINDINGS §17).** `evaluate_dimensional_collapse` no longer
takes `U_scales_out` at all — it computes mass from `Z_scaled` (relation-level, corrected
for label permutation where confidently warranted — see ticket 82) and penalizes on
`max_share` directly, the max-share reformulation named below as the candidate fix. Both
points below are historical record of the diagnosis; the code they describe is superseded.

**Empirical consequence, measured across the full grid before this fix (ticket 81, see
below):** `collapse_pen` was **exactly 0.0 in all 12 tested cells** (C1–C6 × K∈{2,4},
production settings, `lambda_l1=0`, `lambda_z_offdiag=0.05`) — entropy never dropped below
0.808 anywhere in the grid, nowhere near the 0.60 floor regardless of which of the two
problems above was responsible. See ticket 81. **Post-fix (FINDINGS §17): `collapse_pen`
now fires on 2 of 12 cells** (`C1/K=2`, `C6/K=2`, both barely over the max-share ceiling) —
the first time either half of `sociological_penalty` has fired at all in this pipeline's
history.

*Ticket 60 (superseded detail, kept for history):* `N_f` was changed to the **live** entity
count (`build_presence_masks()`) rather than the raw facet dimension. That fix is still
correct and still applied — it addressed a real distortion (dead entities from the other
time slice deflating a facet's apparent mass) — but it is a correction to a formula whose
core input (`U_scales`) has since been shown to be undetermined regardless. Kept in place;
superseded as "the fix" by the larger finding above.

### 4.5 Penalties are squared, not linear

`penalty = (max(0, threshold - actual) / threshold) ** 2`

Lenient on minor threshold violations, strict on major ones. This is a deliberate deviation
from the original linear plan.

### 4.6 K is fixed per Optuna study

`pure_recon_loss` decreases monotonically with K. Allowing Optuna to tune K would bias it
toward large K for capacity reasons alone. One study per K; comparison across K happens via
hypervolume in Module 4's scout phase.

*Ticket 59 update:* `K_LIST` is `[2, 3, 4, 5, 6]`, not `[5, 8, 10, 12, 15]`. T1 has ~3,300
total non-zero observations (§11); K=15 gave ~19 free parameters per observation, and C1's
sole anchor `M_Parent_Art` (160×25, 63 non-zeros) asked `svds` for more components
(`k_svd=min(15,24)=15`) than the matrix's rank supports, with `k_svd == K` so the padding
branch never triggered.

### 4.7 Optuna tunes physics only

Optuna draws `lambda_l1` and `lambda_z_offdiag`. It is **barred** from tuning
`ENTROPY_THRESHOLD`, `TARGET_COHERENCE`, or `MAX_MONOPOLY` — otherwise a trial can "succeed"
by moving the goalposts rather than finding better structure.

### 4.8 REVERSED — clamp projection, not softplus reparameterisation (ticket 74)

**This section previously locked in the opposite decision.** It said: parameters live
unconstrained in ℝ as `U_raw`/`Z_raw`, positivity comes from `F.softplus()` in the forward
pass, because that preserves Adam's momentum, which hard clamping (`tensor.clamp_(min=1e-7)`)
was assumed to destroy. **That assumption was tried, measured, and failed for this loss
surface.** Recording the finding here — not just flipping the line — so nobody reintroduces
softplus on the same reasoning without re-running the test below first.

**What went wrong:** softplus's gradient is `sigmoid(raw)` — strictly less than 1
everywhere, and vanishingly small for the small positive values this pipeline's
NNDSVD/multi-hop-propagation init actually produces (most facets start with 70-90% of
entries at `raw ≈ -9.2`, where `sigmoid(-9.2) ≈ 1e-4`). That attenuation dominated whatever
momentum-coherence benefit motivated the original choice. Diagnosed on C6/K=4/lr=0.01/
λ_l1=0.01/λ_z=0.05 (§8 ticket 74):

- `U`'s community loadings never differentiated — the median row's max-community loading
  (`auth` facet, `U_prob` row-max) sat at *exactly* the uniform-over-K baseline (0.25) at
  epoch 0 and only crept to 0.262 after 400 epochs.
- `recon_loss` dropped from init (~9.8-10.2) toward a "near-zero-overlap" trivial floor
  (~0.96) and stalled there — not from vanishing gradients in aggregate (epoch-0 tensor-level
  grad norms were 0.14-1.4, healthy) but because the *individual* floor-heavy entries
  couldn't move.
- Isolated against two other suspects first, both ruled out: the (separately real and
  separately fixed, ticket 73) unnormalized L1 term — λ_l1=0 alone only reached recon_loss
  0.956, barely off the floor; and missing NNDSVDar stabilization noise (v7 had a
  `noise = np.random.rand(*U.shape) * (avg/10 + 1e-4)` lift that v9 had dropped) — restoring
  it broke initial symmetry (median row-max jumped to 0.43 at epoch 0) but the *training-time*
  differentiation still didn't happen (0.429→0.428→0.427 over 400 epochs), and recon_loss was
  unchanged (0.961 vs 0.956).
- **Decisive test:** re-ran the identical loop — same init (including the NNDSVDar noise),
  same scale-invariance normalization, same mean-normalized L1, same joint-anchor SVD — with
  the *only* change being parameters held directly positive (`U_pos = U_raw`, no softplus)
  and `tensor.clamp_(min=1e-7)` after every `optimizer.step()`, matching v7/v8:

  | | softplus (was current) | clamp (v7/v8-style) |
  |---|---|---|
  | recon_loss @ epoch 0 | 10.20 | 10.19 |
  | recon_loss @ epoch 100 | 1.46 | 0.92 |
  | recon_loss @ epoch 399 | 0.96 (plateaued) | **0.80 — still descending** |
  | `auth` `U_prob` row-max p50 @ epoch 0 | 0.429 | 0.433 |
  | `auth` `U_prob` row-max p50 @ epoch 399 | 0.427 (frozen) | **0.539 (moved during training)** |

  Normalization held constant, both criteria for "softplus is the cause" were met
  (recon_loss meaningfully below the softplus plateau, and p50 actually moving during
  training, not just at init). Confirmed via a second, independent path: monkeypatching
  `inverse_softplus_np` to identity and reusing the real (fixed) init function directly,
  then re-running the real `run_inner_solver` end-to-end for cross-check — same result to
  three decimal places.

**Current implementation:** `initialize_tucker_adapted_nndsvd_and_propagate` returns
positive-valued leaf tensors directly (no `inverse_softplus_np` conversion).
`run_inner_solver` sets `U_pos = U_raw`, `Z_pos = Z_raw` (identity — no `F.softplus`), and
clamps every `U_raw`/`Z_raw` tensor to `min=1e-7` in-place, both immediately after init and
after every `optimizer.step()`. `inverse_softplus_np()` itself is left defined (dead code,
harmless) in case a future investigation wants it back — but do not wire it back in without
re-running this test.

**Kept, verified independent of this question:** the joint-anchor SVD (ticket 64), the
restored NNDSVDar stabilization noise (ticket 74, still belongs regardless of positivity
scheme), the mean-normalized L1 (ticket 73), and the forward-pass scale-invariance
normalization (§4.2-§4.4) — none of these were touched by the softplus→clamp change, and
the decisive test explicitly held all of them constant to isolate the one variable.

### 4.9 Deterministic facet sorting (stacking guardrail)

Facet lists must come from `get_required_facets()`, which returns `sorted(list(...))`.
**Never iterate a raw `set()`.** Module 4 Section 5 vertically stacks facet tensors for
Hungarian alignment and Jensen–Shannon divergence; unordered iteration produces
non-deterministic tensor geometry across seeds and catastrophic alignment failure.

### 4.10 Dynamic geometric discovery (self-healing topology)

Module 1 derives `dimensions` by querying `.shape` on the incoming SciPy matrices, with
assertion checks for cross-relation consistency. Do **not** hardcode dimensions or trust
upstream metadata dictionaries.

*Ticket 63 confirmation:* this is not a fallback — it is the **only** live code path.
chunk12's output pickles never contain a `'dimensions'` key at all (see §11). Do not add
logic that assumes the key might be present; do not "simplify" by removing the
auto-discovery branch.

### 4.11 `CORE_THRESHOLD` is retired

Section 4A originally thresholded articles at 0.75 to identify "core articles". It now uses
**continuous probabilistic message passing** — `W_art = U_prob['art'][:, k]` propagated
backward through the active topology. The constant is intentionally absent.

### 4.12 Multi-objective Pareto, not scalarised meta-loss

The objective returns `(pure_recon_loss, sociological_penalty)` as a 2-tuple, optimised by
`NSGAIISampler`. Lambda weights (`LAMBDA_COLLAPSE` etc.) were removed from the aggregation —
the three penalties sum unweighted into `sociological_penalty`. See open ticket 35.

### 4.13 Post-hoc sociological recomputation (the reproducibility Δ)

Module 4's extraction (S4) and stability (S5) sections must **recompute** the sociological
penalty by passing freshly generated `U_final` and `Z_final` through
`evaluate_complete_solution()`. They must never reuse Optuna's archived `trial.values[1]`
for a newly generated seed.

*Rationale:* Module 2 returns **only mathematical** diagnostics — per §4.1, it computes no
sociological metrics at all. `diagnostics` has no sociological key. If Module 4 falls back
on a cached Optuna score for a fresh seed, the reproducibility Δ and the sociological
variance both evaluate to exactly `0.0`, and the stability analysis reports perfect
sociological reproducibility for every model — a failure that looks like success.

**Closed (ticket 52):** S5 now captures `Z_final` (was discarded as `_`) and calls
`evaluate_complete_solution()` per seed, reading `evaluation["sociological_penalty"]`.

**Closed (ticket 69):** the Optuna objective itself (§5.2) was found to independently
re-derive the same penalty via an inlined copy of §5.1's sequence, and was doing so on raw
torch tensors instead of the detached numpy arrays `evaluate_complete_solution()` converts
internally — a `TypeError` on every trial once upstream bugs stopped masking it. `objective()`
now calls `evaluate_complete_solution()` directly. One evaluation sequence, one conversion
path, shared by the Optuna loop, §S4, and §S5 — the divergence risk this section warns about
is now structurally impossible, not just policed by convention.

### 4.14 Hypervolume must operate on a Pareto-filtered front

Current implementation uses `optuna._hypervolume.compute_hypervolume(pts, ref_point)`, fed
from `study.best_trials` — which is already Optuna's Pareto front, so filtering is satisfied
by construction and no action is needed today.

**Conditional guard:** if the hypervolume routine is ever replaced with a hand-rolled
sweep-line (e.g. to escape the private-API dependency), that implementation **must** perform
strict Pareto extraction before computing area. A sweep across unfiltered trial history
integrates dominated points into the coordinate boundaries and corrupts the result.

**Closed (ticket 47 regression):** the call site had regressed to the pre-4.9 `WFG` class
pattern — `compute_hypervolume().compute(pts, ref_point)` — which calls the function with
zero arguments before chaining `.compute(...)` on the result. `compute_hypervolume` is a
plain function (verified via `inspect.signature` against the installed `optuna==4.9.0`:
`compute_hypervolume(loss_vals, reference_point, assume_pareto=False) -> float`). Fixed to
the direct call shown above, matching this section.

### 4.15 C1's single-anchor weakness is a finding, not a bug

C1's sole anchor (`M_Parent_Art`, 63 non-zeros) roots its entire semantic initialisation
tree at one weak SVD, and it will underperform C2/C5/C6 in scout-phase hypervolume. **Do
not** add special-casing, extra regularisation, or init tricks to compensate. Comparing
configurations and eliminating weak ones on this toy corpus, before scaling to the full
22k-article corpus, is the point of the C1–C6 grid — not something to engineer around. See
§9.

### 4.16 Strict determinism (no `warn_only`) + forced single-threading (tickets 43, 76)

`set_seeds()` now calls `torch.set_num_threads(1)` before anything else, and calls
`torch.use_deterministic_algorithms(True)` **without** `warn_only=True`. Both changed
together, in the same patch session, verified separately:

- **Single-threading** (ticket 76) is the confirmed fix for run-to-run non-determinism at a
  fixed `MASTER_SEED` — traced to non-associative multi-threaded BLAS/OMP reduction order,
  not to anything in this pipeline's own code. Repeated identical-seed runs are bit-identical
  once single-threaded (zero max diff across the full loss history).
- **Strict `use_deterministic_algorithms`** (ticket 43) was tested independently — a real
  C6/K=4/50-epoch fit completed with no exception — before being adopted. Do not assume this
  generalizes to GPU: this environment has no CUDA (`DEVICE` is always `cpu` on the
  login/incline nodes used for development), so only the CPU op path has been exercised. If a
  future compute-node GPU run raises `RuntimeError: ... does not have a deterministic
  implementation`, that is a real, not-yet-seen failure mode — revert to `warn_only=True` for
  GPU runs specifically rather than assuming the CPU test covers it.

`submit_chunk13v9.sh`'s thread-binding exports (`OMP_NUM_THREADS=1` etc.) must stay in sync
with `set_num_threads(1)` — an environment that requests more threads than the code will use
is harmless, but one that requests fewer than the code assumes is not tested and shouldn't be
assumed safe.

---

### 4.17 `sociological_penalty` is currently one term wearing three names (ticket 81)

`sociological_penalty = collapse_pen + coherence_pen + semantic_pen`. Measured across the
full grid (C1–C6 × K∈{2,4}, production settings): `collapse_pen` and `coherence_pen` are
**exactly 0.0 in all 12 cells**. `sociological_penalty == semantic_pen` bit-for-bit
everywhere tested.

This is not because the underlying quantities are constant — `normalized_entropy` varies
0.808–0.995 across cells, and `weakest_mean` (coherence's input) varies 0.744–1.000 — it is
because both thresholds (`ENTROPY_THRESHOLD=0.60`, `TARGET_COHERENCE=0.50`) sit well below
where any production fit in this grid ever lands. Closest approach to either floor across
all 12 cells: `weakest_mean=0.744` vs a `0.50` floor. Neither term has been observed to fire
even once.

**Practical consequence:** Optuna's second Pareto axis has effectively been `semantic_pen`
alone (range 0.054–0.132 across the grid — itself a narrow band) for as long as this grid has
been run. `coherence_pen` is additionally reading `weakest_mean`, which §1/FINDINGS §1
establishes is gauge-dependent (anchor Z diagonals, pinned by SVD initialisation) — so even
if its threshold were recalibrated to fire, what it would be firing on is not yet trustworthy.
`collapse_pen`'s input (`U_scales`-based mass) is undetermined per §4.3 above.

**Not yet acted on.** Redesigning this is a decision, not a bug fix — see FINDINGS §15 for
the full analysis and the options under consideration (relation-level mass via `Z_scaled`,
splitting `semantic_pen`'s Part A/B to see if the axis is really just Part A, a max-share
reformulation of collapse per §4.4).

### 4.18 Per-community domain balance has no enforcement mechanism (open, ticket 82)

The 50/50 `alpha` weighting in Module 2 (§ recon_loss construction) balances the *global*
contribution of the social and semantic domains to reconstruction loss. It does **not**
constrain any individual community — a solution where community 1 is entirely semantic and
community 2 entirely social satisfies the global 50/50 split perfectly while producing
communities that are not heterogeneous at all, which contradicts §1's stated research goal.

v8 had a per-community mechanism, `binding_penalty`, computed inside Module 2 (differentiable,
not an outer-loop score):
```python
soc_vol = sum(‖U[f][:,r]‖₂ / √N_f  for f in exclusive_soc)
sem_vol = sum(‖U[f][:,r]‖₂ / √N_f  for f in exclusive_sem)
binding_penalty += (soc_vol/num_soc - sem_vol/num_sem) ** 2
```
This was lost in the rewrite that introduced softplus (§4.8) and was not restored alongside
the other regressions from that rewrite (Frobenius normalisation, alpha weighting,
initialisation). **v9 currently has no per-community domain-balance mechanism at all.**

**No longer blocked — unblocked by recognizing domain as a FACET property, not a relation
property (design session, not yet implemented in code; full reasoning in FINDINGS §13's
update).** v8's relation-level framing (and the naive `U_scales/√N_f` port of it) forced an
ill-posed question for anchor relations: is `M_Child_Art` "social" or "semantic"? It touches
`art` (a bibliographic entity) and `core_child_he` (a semantic hyperedge) simultaneously.
**There is no such thing as a mixed facet, though** — `art` is unambiguously social (same
kind of entity as `auth`/`journ`/`affil`); it only *looks* ambiguous because it participates
in both `S_` and `M_` relations. A facet-level formulation never has to answer the ill-posed
relation-level question, and **supersedes the `0.5/n_anchors` candidate fix** — verified
directly (all 6 configs, `RELATION_MAP`): `art` has exactly 2 non-anchor relations
(`S_Art_Auth`, `S_Art_Journ`) in *every* config regardless of anchor count, so a facet-level
profile is structurally comparable across single- and dual-anchor configs without any
anchor-count correction at all.

**Design settled, both mechanisms following the existing `z_offdiag_loss`(in-loop)/
`collapse_pen`(outer-loop) architectural pattern:**
- **In-loop (differentiable):** per community `k`, `soc_k`/`sem_k` = weighted mean over active
  facets in that domain of `U_prob[f][:,k]`'s mean over LIVE entities (ticket 60 masking
  required — dead entities sit at the clamp floor and normalize to ≈uniform `1/K`, diluting
  every community toward the null); `r_k = soc_k/(soc_k+sem_k+eps)`; penalty
  `mean_k(max(0, |r_k-0.5| - TOL)²)`. **Verified trap, not a hypothetical:** `U_norm`'s columns
  have unit L2 norm by construction (`U_scales[f] = ‖col‖₂`, `U_norm = u/U_scales`, chunk13v9.py
  ~589-591) — porting v8's `‖U[:,r]‖₂` formula onto `U_norm` directly would be gauge-safe and
  differentiable, but **identically constant (≡1)**, penalizing nothing. `U_prob` (which varies
  per row) is the correct substitute, not `U_norm` column norms.
- **Outer-loop (`Z_scaled`-based, mirrors `collapse_pen`'s construction):** per relation,
  production's own `_relation_community_share` (§17); each relation's share vector credited to
  *both* facets it touches (a true statement, not double-counting); anchor relations are
  **included** (reverses an earlier draft of this design) — excluding them would make the
  measure blind to exactly the social↔semantic binding a heterogeneous community shows up
  through, and would violate §4.15's "do not special-case for anchor count" rule. Anchor
  sensitivity (reading with vs. without anchors) is measured, not assumed away — see below.

**Weighting decision (equal-per-facet vs. live-entity-count) — resolved in favor of
entity-count weighting**, reversing an initial lean toward the codebase's equal-weighting
convention. Two independent arguments converged: (1) only entity weighting is mathematically
a true population-mass share — it is algebraically identical to pooling every live entity in
the domain and taking one grand mean of `U_prob[:,k]`, whereas equal-per-facet weighting
discards absolute headcount by construction (an average of per-facet-type proportions, not a
sum-comparable mass); this is the standard "proportional vs. equal allocation" distinction
from stratified sampling, and the mechanism's target (domain-level balance) is the
population-level question proportional allocation answers. (2) **Decisive for the in-loop term
specifically:** the gradient of `soc_k` w.r.t. one entity's `U_prob` row is `1/N_f` under equal
weighting (facet-size-dependent — in T1, an `auth` entity's leverage is ~1/23rd a `journ`
entity's) but a uniform `1/N_domain` under entity weighting, regardless of facet. Equal
weighting would let the optimizer satisfy the penalty cheaply by reshuffling small facets while
leaving `auth` (the dominant facet by population) essentially untouched — a self-inflicted
version of the Problem 2 gaming lesson, not a hypothetical.

**E1 measurement (diagnostic-only, `diagnostic_scripts/domain_balance_measurement.py`,
`diagnostic_blocks.py` 1.12.0) answers the "unanswerable" question below — GO.** On the
determined basis (`u_prob`, equal-weighted as measured — entity-weighted reruns not yet
re-tabulated post-decision): mean `dev_k` (deviation from `r_k=0.5`) ≈0.08 in both T1 and T2,
median ≈0.07 — most communities sit within a generous band, consistent with some imbalance
being expected from topology alone. But **no consistent direction** (T1/T2 combined: 26
communities skew social, 25 semantic, 25 balanced — unlike the pre-fix numbers below, which
disagreed in direction only because they were both measuring an undetermined quantity) and a
genuine tail: ~31-32% of communities exceed a 40/60 band, ~12-14% exceed 35/65, with concrete
cases up to `r=0.745` (`T2/C2/K4` community 2). Entity weighting (not yet re-run as primary)
showed materially larger deviations in the equal-weighted pilot (mean ≈0.13 vs 0.08) —
expected, given `auth`'s dominance under that weighting; the correct next step is re-running
E1's full grid under entity weighting as primary before setting `TOL`, not reusing the
equal-weighted numbers above.

### 4.19 Sparsity term (`lambda_l1`) fixed to 0.0 for the toy corpus (ticket 78)

See ticket 78 in §8 and FINDINGS §6 for the full evidence. Summary: the term as implemented
penalises *column* concentration (few entities per community) rather than *row* concentration
(few communities per entity, the actual "hairball" concern), and at every tested weight in its
(now-superseded) widened range it evacuated row mass rather than concentrating it, while the
hairball it exists to prevent does not occur at `lambda_l1=0` on this corpus. Fixed at `0.0`,
removed from Optuna's search space, `sparsity_loss` itself still computed and exposed in
diagnostics for observability. **Explicitly a toy-corpus decision** — re-test at 22k articles
using the L1/L2 per-row ratio (not `U_prob` row-max) as the detector.

---

### 4.20 Permutation correction is read-time, inside `evaluate_complete_solution` —
not a physical mutation of `U`/`Z` (tickets 79/80/82, FINDINGS §17)

`_hungarian_relabel_relation` and `_relation_community_share` (Module 3 §2) correct a
relation's diagonal reading for mass-computation purposes only, at evaluation time. They do
**not** mutate `U_final`/`Z_final` in place.

*Prior art, and why this session diverged from it:* `chunk13v3.py`/`chunk13v4.py` had a
working mechanism (`identify_leaf_nodes`, `diagnose_leaf_Z`, `correct_all_leaf_nodes`) that
physically relabeled `U`'s columns and `Z`'s axis right after fitting, so every downstream
consumer saw the corrected labeling automatically. That approach was considered and rejected
for two reasons: (1) it alters the saved tensors from what the optimizer actually produced —
anyone later re-deriving `recon_loss`, or running the FINDINGS §8 reconstruction-space
coherence metric, or auditing the raw fit for any other reason, would need to separately know
a correction was silently applied upstream; (2) it was called immediately after fitting,
straddling §4.1's locked boundary — Hungarian assignment is a discontinuous operation
required to live in the outer loop (Module 3/4), not inside Module 2's fitting stage.

*The alternative risk — "every future consumer must remember to apply the correction" — is
closed structurally, not by convention.* `evaluate_complete_solution()` is already, per
ticket 69, the mandatory single entry point every consumer (Optuna's objective, the §S4
archiver, the §S5 stability analysis) must call rather than reimplement. The permutation
correction lives inside it (via `evaluate_dimensional_collapse`), so no caller has a
legitimate path to `Z_scaled[k,k]`-based mass without passing through the function that
corrects it first — the same guarantee that already prevents the sociological-penalty
computation from silently diverging across call sites.

**Do not revert to physical mutation without re-opening this discussion.** If a future need
arises for every consumer to see corrected labels without calling
`evaluate_complete_solution` (e.g. a raw diagnostic dump), add a *separate*, explicitly-named
function for that — do not fold it back into the fitting stage.

## 5. Namespace Gotcha

**Relation keys and facet names are different namespaces.** This has caused two separate bugs.

- Relation keys: `'S_Art_Auth'`, `'M_Child_Art'`, `'M_Cousin_Parent'` — keys of `raw_data` and `RELATION_MAP`
- Facet names: `'art'`, `'auth'`, `'core_child_he'`, `'parent_he'` — keys of `dimensions`, `U_final`, `U_prob`

`RELATION_MAP` is the **only** bridge between them. Never compare one against the other
directly; route through `get_required_facets()`.

---

## 6. Environment

```
Conda env : tensor_env
Path      : /mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env
Python    : 3.10
torch     : 2.11.0+cu130    (CUDA unavailable on login/incline nodes; available on compute)
optuna    : 4.9.0
numpy     : 2.2.5
scipy     : 1.15.3
```

**Optuna 4.9 note:** `optuna._hypervolume.WFG` **no longer exists**. Use
`from optuna._hypervolume import compute_hypervolume` and call
`compute_hypervolume(pts, ref_point)`. Verified correct against a hand-computed 2D case, and
against `inspect.signature` on the installed package (§4.14).

`EXPECTED_OPTUNA_VERSION` (Module 4 §S2/3) is still hardcoded to `"4.1.0"`, so every run
prints a spurious HPC WARNING about a version mismatch against the real `4.9.0` environment.
Cosmetic only — left open, see ticket 71.

**JSON / numpy-scalar note (ticket 68):** `evaluate_dimensional_collapse`,
`evaluate_topological_coherence`, and `evaluate_socio_semantic_reality` now explicitly cast
their return values to native Python `float` at their own `return` statements — numpy
scalars (float32/float64) are not JSON-serializable via stdlib `json`, which is what
Optuna's `trial.set_user_attr(...)` uses under the hood (`TypeError: Object of type float32
is not JSON serializable`). `float(np.nan)` is safe here: `np.nan` is already a native
Python `float`, and stdlib `json.dumps` accepts it by default, emitting the bare literal
`NaN` — which `json.loads` reads back fine, but **is not valid strict JSON** (RFC 8259
disallows `NaN`/`Infinity`). This has not broken anything in this codebase because nothing
here re-parses these JSON blobs with a strict parser, but treat it as a latent trap if any
of the `.json` manifests this pipeline writes (`model_metadata.json`,
`scout_methodology_report.json`, `master_dual_track_stability_report.json`) are ever
consumed by non-Python tooling — a strict JSON parser (many are, outside Python) will reject
a bare `NaN` token.

**Data path:**
`/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1.pkl`
(T2 slice: `Star_extended_matrices_t2.pkl`, same directory — not yet wired into the pipeline,
see §10.)

**SLURM submission pattern** — `submit_chunk13v9.sh` now exists on disk (adapted from v8's;
none had been created for v9 before ticket 76). **Single-threaded, matching `set_seeds()`'s
`torch.set_num_threads(1)`** (ticket 76, §4.16) — do not restore `=4` here without also
reverting the code:

```bash
#!/bin/bash --login
#SBATCH --job-name=chunk13_v9
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /mnt/hum01-home01/p91688di/miniconda3/etc/profile.d/conda.sh
conda activate tensor_env
export LD_LIBRARY_PATH=/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python chunk13v9.py
```

**Compute estimate (ticket 76):** C1-C6 × K=[2,3,4] × `SCOUT_TRIALS=DEEP_DIVE_TRIALS=100`
(the constants actually in code today — not the 20 used in an earlier planning estimate),
single-threaded: ≈**4.2 hours** baseline (3000 total trials × ~5.03s/trial, timed directly on
C6), **4.7-5 hours** with a buffer for ticket-75 non-convergence top-up. See ticket 76 in §8
for the full derivation.

---

## 7. Conventions

- **Single `.py` file.** No `import` statements between modules — Module 1's definitions are
  in scope for everything textually below them.
- **Module 1 owns all constants and topology.** If a constant or topology definition appears
  twice, delete the later copy.
- **Deviations from the original plan are acceptable** when they are genuine improvements
  (squared penalties, continuous message passing, Pareto multi-objective). Note them in a
  comment so future readers know they were deliberate.
- Imports belong at the **top** of the file, not scattered at section boundaries. (Still
  violated in several places — tickets 23/36, left as non-critical.)

---

## 8. Open Defect Register

### Parse-blocking (discovered this session — the file could not import at all)

| # | Location | Issue | Status |
|---|---|---|---|
| 70 | M1 §1.7, `load_and_validate_data` | `if missing:` body not indented under the `if` — `SyntaxError`, blocked the entire file from parsing | **Closed.** Indented. |
| — | M2 §2.2, `run_inner_solver` | `else:` misaligned with its `if key in raw_data:` (was at 12 spaces, needed 8), body under it also misaligned | **Closed** (was already flagged as "Pending" at the bottom of this section; fixed alongside 70/49). |
| 49 | M4 §S4, `extract_and_archive_pareto_front` | `diagnostics.update(evaluation)` and the reproducibility-check block sat outside the `try:` scope (dedented to column 0) but referenced names defined inside it — `SyntaxError` | **Closed.** Re-indented inside `try:`. |

### Critical — runtime failure

| # | Location | Issue | Status |
|---|---|---|---|
| 52 | M4 §S5 | `diagnostics.get("soc_penalty", 0.0)` — key never existed; `Z_final` discarded as `_`. See §4.13. | **Closed.** `Z_final` captured; calls `evaluate_complete_solution()` per seed via ticket 69's consolidation. |
| 58 | M4 §S2/S4/S5 | Three separate `if __name__ == "__main__":` blocks; §S4/§S5 pointed at the placeholder path `"./data/lean_matrices.pkl"` | **Closed.** Consolidated into one block at end of file (line 2084), real data path, all three phases chained. |
| 67 | M3 §5.2, `objective()` | Never passed `K` to `run_inner_solver` — only buried `'K': K_fixed` inside the `params` dict, which the function doesn't read. `TypeError: missing 1 required positional argument: 'K'` on every trial. Found via the post-Batch-C smoke test, not previously registered. | **Closed.** `K=K_fixed` passed explicitly. |
| 68 | M3 §2/§3/§4 evaluator returns | `evaluate_dimensional_collapse`, `evaluate_topological_coherence`, `evaluate_socio_semantic_reality` returned numpy scalars; `trial.set_user_attr(...)` → stdlib `json.dumps` → `TypeError: Object of type float32 is not JSON serializable` on every trial. Found via smoke test. | **Closed.** Cast to native `float` at each function's own `return`, not just in the aggregator — see §6 JSON note. |
| 69 | M3 §5.2, `objective()` | Reimplemented §5.1's evaluation sequence inline instead of calling `evaluate_complete_solution()`, on raw (non-numpy) tensors. Root cause of ticket 68 being reachable and of the original 5-value-unpack regression (see ticket 31/32 history). | **Closed.** `objective()` now calls `evaluate_complete_solution()` directly (§4.13, §3). |
| 83 | M2, `run_inner_solver` | `lambda_l1 = params['lambda_l1']` (line 522) is a hard `[]` lookup. Ticket 78 fixed `lambda_l1` at `0.0` and removed it from `trial.suggest_float` — it is recorded only via `trial.set_user_attr`, which does **not** enter `trial.params`. The Optuna objective itself never hits this (it builds its own `hyperparams` dict with `'lambda_l1'` explicit, §5.2), but §S4's archiver (`params=trial.params`) and §S5's stability loop (`params=metadata["hyperparameters"]`, itself a serialization of `trial.params`) both pass it straight through — both raise `KeyError` on every call. A production run would complete the full ~4-5h Optuna search (§6, ticket 76) and only crash at the archiving stage. Found while mapping the file for the ticket 82 domain-balance design work, unrelated to it. | **Closed.** `params.get('lambda_l1', 0.0)`, matching the existing `lambda_z_offdiag` readout on the next line. Verified directly: reproduced the exact `params` shape `trial.params` has post-ticket-78 (`{'lambda_z_offdiag': ...}`, no `'lambda_l1'` key) and confirmed `run_inner_solver` no longer raises. |

### Silent incorrect computation

| # | Location | Issue | Status |
|---|---|---|---|
| 25 | M3 §S4 | Dead-community guard appended `1.0` for all four target facets, including facets absent from `U_prob` — systematically inflated `Penalty_A` for configs with fewer active facets | **Closed.** Only appends for facets actually in `U_prob`. |
| 54 | M4 §S5 | Described: `break` on first seed failure leaves statistics computed over a partial sample with no warning; `np.std([x], ddof=1)` returns `NaN` when only one seed succeeded | **STALE — verified, left as-is.** Current code aborts and `continue`s to the next trial on any failed seed (does not fall through to partial-sample stats), and guards `ddof=1` with `len(...) > 1 else 0.0`. Description no longer matches the code. |
| 57 | M1 / M4 §1 | Described: `PIPELINE_VERSION` defined twice with different values | **STALE — verified, left as-is.** Only one assignment exists (`"v9.1.time_slice_1"`); `"v4.2_pareto"` is a comment string in Module 4's header, not a second assignment. |
| 60 | M3 §2/§4, M1 §1.8 | **New.** Entity index maps are GLOBAL across chunk12 time slices; only `art` is slice-specific (§11). A slice's matrices are padded with structurally-zero rows/cols for entities belonging to the other slice — at least 53–61% of authors/parent-hyperedges/etc. in T1 (§11). Collapse mass and monopoly/baseline calculations averaged over ALL rows including dead ones. | **Closed.** `build_presence_masks()` added (M1 §1.8); threaded into `evaluate_dimensional_collapse` (live-count normalization) and `evaluate_socio_semantic_reality` Parts A/B (baseline + monopoly scan restricted to live rows). |
| 61 | M2, `run_inner_solver` trace loss | **New.** `relation_recon` hardcoded `||X||² = 1.0` (valid only because X is Frobenius-normalized to unit norm) — wrong for an empty relation (true `||X||² = 0`), giving a permanent unremovable loss floor. Dormant today (no T1/T2 relation is empty) but live for the full corpus. | **Closed.** `target_norm_sq` is `0.0` for relations with `fro_norm <= 0`, `1.0` otherwise; one-time warning per empty relation. |
| 64 | M2, `initialize_tucker_adapted_nndsvd_and_propagate` | **New — MULTI-ANCHOR OVERWRITE, most important fix this session.** Every anchor shares the same column facet (`'art'`); the per-anchor SVD loop assigned `U_np['art']` independently per anchor, so the second anchor silently overwrote the first's article embedding (affects C2, C5, C6 — all dual-anchor). | **Closed.** Restored the v7 approach: all active anchors vstacked into one joint SVD (valid since they share the column space), left singular vectors sliced back per anchor by row offset, shared right singular vectors assigned to `'art'` once. Smoke-tested on C6 specifically (dual-anchor) — passed. |

### Conditional — environment or ordering dependent

| # | Location | Issue | Status |
|---|---|---|---|
| 22 | M3 §S3 | Returns `(0.0, np.nan)` when no anchors active — confirm `np.nan` cannot propagate into the Optuna objective value | **Resolved, no fix needed.** Verified: `coherence_pen` (`0.0`, not the nan) is what feeds `sociological_penalty`; the nan (`weakest_mean`) is only ever stored as a `trial.set_user_attr(...)` for logging. Confirmed safe against the installed json/Optuna stack (§6). |
| 26 | M3 §S4 | `scipy.sparse.dot()` may return `np.matrix` on some versions — `sum(child_paths)` and `len()` behave incorrectly on matrices | **Open, unchanged.** Not independently reproducible without a scipy version where this regresses; left as environment-conditional per original register. |
| 50 | M4 §S4 | `dimensions` may be `None`; S2 has a guard, S4 does not | **Closed.** `if dimensions is None: raise KeyError(...)` added to S4, matching S2. |
| 53 | M4 §S4 + §S5 | `NumpyEncoder` redefined in S5 without the `torch.Tensor` case S4 added — later definition wins, breaking tensor serialisation | **Closed.** Deduped; kept the S4 definition (includes the `torch.Tensor` case). |
| 41 | M4 §S1 | `OPTUNA_JOURNAL_PATH` parent directory never created | **Open, unchanged, confirmed inert.** Grepped: `OPTUNA_JOURNAL_PATH` is never referenced anywhere else in the file — dead config, so the missing-directory issue can't currently bite. Worth deleting or wiring in, not urgent. |

### Non-critical — deviation or maintenance risk

| # | Location | Issue | Status |
|---|---|---|---|
| 23 | M3 | `import` statements at the bottom of sections rather than file top | **Open — deliberate, do not touch.** |
| 27 | M3 §S4 | Part B applies `max_monopoly` to all semantic nodes, not just 95th-percentile elites — undocumented methodological change from plan | **Open — deliberate, do not touch.** |
| 35 | M3 §S5 | Lambda weights removed from penalty aggregation — undocumented | **Open — deliberate, do not touch (§4.12).** |
| 36 | M3 §S5 | `optuna` not imported at top of Module 3 | **Open — deliberate, do not touch.** |
| 43 | M4 §S1 | `torch.use_deterministic_algorithms(True, warn_only=True)` — determinism not guaranteed on GPU | **Closed.** Tested `torch.use_deterministic_algorithms(True)` (no `warn_only`) against a real C6/K=4/50-epoch fit — completed with no exception. `warn_only` removed from `set_seeds()`. **Caveat:** this environment has no CUDA (login/incline node, `DEVICE=cpu` always) — only the CPU path is verified; GPU determinism on a compute node with actual CUDA ops is still unconfirmed. See §4.16. |
| 46 | M4 §S3 | `len(study.trials)` counts pruned and failed trials — Deep Dive phase may under-run | **Closed, subsumed by ticket 75.** Both Scout and Deep Dive now loop on a COMPLETED-AND-CONVERGED count (`t.state == TrialState.COMPLETE and t.user_attrs.get("converged", False)`) rather than raw `len(study.trials)`, with a `target*3` safety-valve attempt cap that prints a warning instead of looping forever. |
| 48 | M4 §S4 | `evaluate_complete_solution` called twice identically — redundant compute | **Closed.** Duplicate call removed. |
| 66 | M3 §S4 target_facets | Whether to add `parent_he` to `target_facets` | **Flagged, not decided. Do not touch.** |
| 71 | M4 §S2/3 | `EXPECTED_OPTUNA_VERSION` hardcoded to `"4.1.0"`, prints a spurious mismatch warning every run against the real `4.9.0` environment | **Open, cosmetic only.** New this session; not fixed. |
| 72 | M1 §1.2, `INNER_EPOCHS`/`LEARNING_RATE` | Module-level constants were **dead** — never referenced anywhere else in the file; every `run_inner_solver` call site relied on the function's own hardcoded defaults, which happened to numerically match the constants but weren't driven by them. | **Closed.** `run_inner_solver`'s defaults are now `inner_epochs=INNER_EPOCHS, learning_rate=LEARNING_RATE` — one place to change the setting, all three call sites (Optuna objective, §S4 archiver, §S5 stability) inherit it. `INNER_EPOCHS` also raised 400→2000 as a **ceiling** (not a target — the relative early-stopping check decides actual per-run length) as part of ticket 74's fix — see §4.8 and ticket 74. |
| 73 | M2, `run_inner_solver`, §B Sociological Reality | `sparsity_loss = torch.sum(u_mat)` over `U_norm`'s unit-L2 columns was structurally ~150–172 regardless of what the model learned (≈√N per column × 9 facets × K=4) — v8 normalized this (`l1_penalty / (global_dims[f] * K)`); v9 had dropped it. | **Closed and verified.** Now a mean over all `U_norm` entries (`sparsity_loss / num_sparsity_entries`). **Recomputed post-ticket-74 (real, converged fit — C6, K=4, current code): `mean_sparsity = 0.0134`** (`total_sum=149.59`, `num_entries=11164`, epoch 955, `math_loss=0.7747`). Landed close to the old stale 0.0153 figure despite the earlier model not actually fitting — coincidence, not validation of the earlier number; treat both as independent measurements. Per-facet mean ranges 0.0085 (`core_atom`) to 0.079 (`art`) — full table in `diagnostic_results/fix3_sparsity_measure.json`. **Implied `lambda_l1` range for `lambda_l1 * mean_sparsity` to land in `[0.01, 1.0]` (a plausible order-of-magnitude window against `recon_loss` O(0.7-10)): `[0.75, 74.6]`.** Current `suggest_float('lambda_l1', 1e-4, 1.0, log=True)` sits entirely below this window — its upper bound (1.0) is roughly the *lower* bound of the useful range, confirming the original diagnosis that the whole search range is inert. **Range left unchanged in code** — bounds are the user's call, not set this session. |
| 74 | M2, `run_inner_solver` — softplus reparameterisation | **Root cause found and fixed.** Split from ticket 73. Even with L1 fully removed or correctly normalized, `recon_loss` on C6/K=4/lr=0.01/400 epochs plateaued at ~0.956-0.961 and the `auth` facet's median `U_prob` row-max loading never moved off the uniform-over-K baseline (0.25) across 400 epochs of training. Root cause: **softplus's gradient (`sigmoid(raw)`) vanishes for the small positive values this pipeline's init produces** (70-90% of most facets' entries start at `raw ≈ -9.2`, `sigmoid ≈ 1e-4`), starving the bulk of `U` of any meaningful gradient regardless of learning rate. Confirmed via a controlled test (softplus vs v7/v8-style `tensor.clamp_(min=1e-7)` positivity, every other component held identical): clamping broke through the plateau (recon_loss 0.80 and still descending vs 0.96 plateaued) and the median row-max moved during training (0.25→0.54 vs 0.25→0.26 frozen). Two other suspects were tested and ruled out along the way: the (separately real) unnormalized-L1 term (ticket 73) accounted for only ~4% of the gap; the missing NNDSVDar stabilization noise (restored, see §4.15/init function) broke initial symmetry but did not restore training-time differentiation on its own. Full writeup and numbers: §4.8 (rewritten, not just flipped, so this reasoning doesn't get silently reintroduced). **Fixed:** softplus removed from `initialize_tucker_adapted_nndsvd_and_propagate` and `run_inner_solver`; parameters held directly positive with `clamp_(min=1e-7)` after every `optimizer.step()`, matching v7/v8. | **Closed.** |
| 75 | M3 §5.2 `objective()`; M4 §S2/S3, §S4 | **New — the standing "converged flag unread" item, now fixed.** `objective()` never read `diagnostics["converged"]`; a trial hitting the 2000-epoch ceiling returned a plausible `recon_loss` and entered the Pareto front indistinguishably from a converged trial. Demonstrated three times in the diagnostic investigation (Run 8 at K=6, Run 11, and Part M's K=6 sweep). | **Closed.** Design: **flag, don't prune** — a pruned trial teaches `NSGAIISampler` nothing, so it would keep resampling the region that caused non-convergence instead of steering away; flagging also keeps the data queryable (does non-convergence correlate with `lambda_z_offdiag`, `K`, or config?) and is reversible. `objective()` now sets `trial.set_user_attr("converged", ...)` and `"epochs_run"` right after `run_inner_solver` returns, and prints a visible `[!] Trial N did not converge` line; the actual `(pure_recon_loss, sociological_penalty)` values are still returned unmodified — no artificial penalty substituted. Filtered at all three downstream consumers: (i) M4 §S2/S3's `valid_pts`/hypervolume calc now uses `converged_trials = [t for t in study.best_trials if t.user_attrs.get("converged", False)]`, with a warning printed if filtering empties the front; (ii) M4 §S4's `pareto_trials` (the archiver — most consequential of the three, since archived models get stability-tested) filters identically before sorting/sampling; (iii) M4 §S5's stability loop was verified consistent, not touched — it already raises `RuntimeError("Silent optimizer failure...")` per seed on `not diagnostics.get("converged", True)`, which the flag doesn't contradict. Trial budget top-up (also closes ticket 46): Scout and Deep Dive both replaced their fixed-`n_trials` `study.optimize()` call with a loop that runs one trial at a time until a **completed-and-converged** count reaches the target, capped at `target*3` attempts with a printed warning if the cap is hit. `methodology_report[config_id][K]` gained `"trials_converged"` / `"trials_not_converged"` fields. Smoke-tested (3-trial in-process study, C6/K=4): all `user_attrs` populated correctly, filtering behaves as designed. |
| 76 | M4 §S1 `set_seeds()` | **New.** Non-determinism at a fixed `MASTER_SEED` traced (prior diagnostic round) to multi-threaded BLAS/OMP reduction non-associativity — confirmed fixed by `torch.set_num_threads(1)` (bit-identical repeats, zero max diff across the full loss history). Not yet wired into the source. | **Closed.** `torch.set_num_threads(1)` added as the first line of `set_seeds()` (not only at module level) so it's reasserted on every call — every real entry point (`objective()`, §S4 archiver, §S5 stability loop) calls `set_seeds()` first thing, guaranteeing the setting is active before any tensor op in each of them. `submit_chunk13v9.sh` created (adapted from v8's, none existed for v9 before) with `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` / `VECLIB_MAXIMUM_THREADS=1` / `NUMEXPR_NUM_THREADS=1`, matching the code. §6's embedded SLURM snippet updated to match. **Compute re-estimate, single-threaded, post-fix:** timed C6/K=2,3,4 directly (worst-case topology, dual-anchor) — 4.57s/5.27s/5.27s per full trial (solver+eval) at 867/994/988 epochs to convergence. For C1–C6 × K=[2,3,4] at the code's actual `SCOUT_TRIALS=DEEP_DIVE_TRIALS=100` (not 20 — flagging this: 20 was an earlier planning-stage figure, the constants in code today are 100/100): Scout = 6×3×100 = 1800 trials, Deep Dive top-up = 6 configs × `top_k_to_keep=max(2,int(0.4×3))=2` × 100 additional trials = 1200 trials, 3000 trials total × ~5.03s avg ≈ **4.2 hours**, before FIX 1's non-convergence top-up. With a ~15% buffer for that (Run 8-style high-`lambda_z_offdiag` non-convergence, low-frequency per Part M) ≈ **4.7-5 hours**. Notably not a 3-4x slowdown vs, the earlier multi-threaded estimate — at this problem's matrix sizes, thread-spawn overhead apparently offset most of the parallel benefit anyway; that earlier ~4h figure also predates the ticket-74 epoch-ceiling raise to 2000, so the two aren't a clean before/after regardless. `submit_chunk13v9.sh`'s `--time` set to `08:00:00` for headroom. |
| 78 | M3 §5.2, `create_optuna_objective` — `lambda_l1` | Ticket 77's widened range `[0.75, 74.6]` was tested (Tests 1/3/4/5, diagnostic session) and found to evacuate row mass rather than concentrate it, at real `recon_loss` cost, while the hairball it exists to prevent does not occur at `lambda_l1=0` (FINDINGS §6). | **Closed.** `lambda_l1` fixed at `0.0`, `suggest_float` call removed from `objective()`. `sparsity_loss` itself is still computed in `run_inner_solver` (unchanged) — only its weight in `total_loss` is now always zero. Added `diagnostics["raw_sparsity_loss"]` (unweighted) so it stays observable even though it no longer affects training, and `trial.set_user_attr("lambda_l1_fixed", True)` for visibility in Optuna's own records. Explicitly a **toy-corpus decision**: comment at the call site states it must be re-tested at 22k articles (more entities per community may make smearing more available there), and names the L1/L2 ratio per live entity (not `U_prob` row-max) as the detector to use for that retest. |
| 77 | M3 §5.2, `create_optuna_objective` — `lambda_l1` search range | Range `suggest_float('lambda_l1', 1e-4, 1.0, log=True)` was measured (ticket 73) to be entirely inert: `lambda_l1 * mean_sparsity` never exceeded 0.0134 against `recon_loss` O(0.7-10). | **Superseded by ticket 78** (`lambda_l1` fixed at 0.0, not tuned at all). The range-widening and Test 1 measurement below are kept as the evidentiary record for *why* ticket 78 happened, not as the current state of the code. New range `[0.75, 74.6]`, derived from the measured `mean_sparsity ≈ 0.0134` (C6/K=4/T1, ticket 73's post-normalization figure) so that `lambda_l1 * mean_sparsity ∈ [~0.01, ~1.0]`. Comment at the call site records the derivation and flags that it **must be re-measured before the 22k-article run** — `mean_sparsity` is a mean over unit-L2-norm columns and scales roughly as `1/sqrt(facet size)` (§4.3), so a bound calibrated on this toy corpus's facet sizes will be wrong by an order of magnitude or more at full scale. **Test 1** (sweep `lambda_l1 ∈ {0.0, 0.75, 5.0, 25.0, 74.6}`, `lambda_z_offdiag=0.05`, C6/K=4/T1, all 5 converged, no ceiling hits, single-threaded determinism re-confirmed bit-identical): the new range is **not inert** — `mean_sparsity` falls monotonically (0.0135→0.0098→0.0064→0.0041→0.0034) and `recon_loss` degrades monotonically (0.7746→0.7797→0.8060→0.8968→0.9282), so `lambda_l1` now has a real, monotonic cost. **But it does not do what the naive reading of "L1 sparsity" suggests**: mean `U_prob` row-max (per-entity membership concentration) does **not** rise with `lambda_l1` — across most facets it *falls*, sometimes sharply (`fringe_atom` mean row-max: 0.83 at λ=0 → 0.44 at λ=5, staying low through λ=74.6; `affil` 0.69→0.42; `parent_he` 0.85→0.53). Root cause: the L1 penalty operates on `U_norm`'s unit-**column**-norm entries (sparse *columns* = few entities per community), which is a different sparsity notion from row-max (sparse *rows* = one community per entity) — pushing one down does not push the other up, and empirically pushes it down too. `mass_share` across the 4 communities stayed roughly balanced (no collapse) at every tested `lambda_l1`. Full per-facet, per-λ numbers (no aggregation) in `diagnostic_results/test1_lambda_l1_sweep.json`. **Practical implication:** raising `lambda_l1` toward the top of its new range buys *less* per-entity concentration, not more, while still costing `recon_loss` — Optuna may not find much use for the top of this range on the concentration axis specifically (it may still help via the `sociological_penalty`'s other terms, not tested here). |
| — | (Test 2, informational only — no ticket, no code change; see below) | Whether the pipeline "hoards" ubiquitous semantics without a ubiquity-discounted L1 term. | **Investigated, inconclusive by design (small sample), see write-up below the table.** |

### Open — pending decision, not yet code-changed

These are analytical findings from diagnostic sessions after ticket 78. No source changes
have been made for any of them; each requires a design decision before implementation.
Full evidence for all four is in FINDINGS.md §12–§16.

| # | Location | Issue | Status |
|---|---|---|---|
| 79 | M2, `U_scales` construction | **U_scales is an undetermined free direction in the loss, not a mass measure.** Proven algebraically (scale a `U_pos` column by *c*, compensate in the corresponding `Z_pos` rows/cols, and `recon_loss`/`sparsity_loss`/`z_offdiag_loss`/`U_norm`/`Z_scaled` are all unchanged to float precision while `U_scales` changes by exactly *c*) and confirmed empirically (CV up to 1.2 for the same facet/community across differently-initialised runs with near-identical `recon_loss`). Invalidates the "use `U_scales` for mass" guidance in §4.3 (old version), §4.4's collapse formula, and every domain-balance calculation attempted on `U_scales`. See FINDINGS §12. | **Closed.** `evaluate_dimensional_collapse` rewritten to use `|Z_scaled[k, π(k)]|` (relation-level, permutation-corrected) via reconstruction-space share (`within_k/total`, FINDINGS §8) — see FINDINGS §17. `U_scales` no longer passed into collapse at all. |
| 80 | M3 §2, `evaluate_dimensional_collapse` | `ENTROPY_THRESHOLD=0.60` does not correspond to the stated target ("no community >0.6 mass"); the correspondence is also K-dependent and non-monotone (two distributions with identical max-share can give very different entropy). See §4.4 above for the numbers. | **Closed.** Direct max-share penalty implemented, ceiling-shape ported verbatim from `evaluate_socio_semantic_reality`'s `MAX_MONOPOLY` penalty (this file, Part B), threshold `MAX_SHARE_THRESHOLD=0.60` (same numeric value as `ENTROPY_THRESHOLD`, reused as the pre-existing target, not re-derived). See FINDINGS §17. |
| 81 | M3 §5.1, `evaluate_complete_solution` — aggregation | `collapse_pen` and `coherence_pen` are exactly `0.0` in all 12 tested cells (C1–C6 × K∈{2,4}, production settings). `sociological_penalty == semantic_pen` bit-for-bit throughout. Not because inputs are constant (`normalized_entropy` 0.808–0.995, `weakest_mean` 0.744–1.000) but because both thresholds sit below where any production fit in this grid lands. See §4.17. | **Partially superseded.** `collapse_pen` now fires on 2/12 cells post-ticket-79/80 fix (FINDINGS §17) — no longer always 0.0. `coherence_pen` is untouched by this session's work and remains exactly 0.0 in all 12 cells; still open. |
| 82 | M2, per-community domain balance | No mechanism currently constrains any individual community's semantic/social mix; only the global 50/50 reconstruction-loss weighting exists. v8's `binding_penalty` (per-community, differentiable) was lost in the softplus-era rewrite and not restored. See §4.18. | **Open — design settled (facet-level, entity-weighted, dual in-loop/outer-loop; §4.18), implementation not yet started.** The relation-level anchor-double-counting problem this row previously described is **superseded**, not solved in place — a facet-level formulation (domain is a facet property, not a relation property) makes it not arise: verified `art` has exactly 2 non-anchor relations in every config regardless of anchor count, so no anchor-count correction is needed. E1 (`domain_balance_measurement.py`) answers the prior "currently unanswerable" question in §4.18 below: real per-community imbalance exists on a determined basis, no consistent direction, moderate typical magnitude with a genuine tail (~31% of communities outside a 40/60 band). Not yet implemented in `chunk13v9.py`; E1 needs re-tabulating under entity weighting (the now-decided primary basis) before `TOL`/thresholds are set (E1's own D2/D3). **Permutation-correction groundwork corrected and shipped:** FINDINGS §16's original "1 solid real conflict" (C5/K4 `art`) was itself a bug — the confidence re-filter that produced it forgot to re-apply leaf-exclusion, wrongly counting `S_Art_Journ` (touches the leaf facet `journ`) as a dissenting vote. Corrected: **zero confirmed real non-leaf conflicts anywhere in the grid.** `evaluate_dimensional_collapse` therefore corrects each relation independently and does not implement cross-relation conflict resolution — documented limitation, see FINDINGS §17. Criterion chosen (`structure_score > 1.0`) verified head-to-head against `chunk13v3.py`/`v4.py`'s original `diagonal_mass/hungarian_mass < 0.7` — 12/94 disagreements, 8 of them cases the older criterion would have wrongly "corrected" a genuinely mixed relation. **Toy-corpus calibrated throughout — re-derive at 22k-article scale before reuse.** |

### Recently closed (verify before trusting)

| # | Fix |
|---|---|
| 2 | `from sympy import python` — **had regressed** (was still present, `sympy` confirmed not installed → `ModuleNotFoundError`). Now actually deleted. Also removed the separate `from ...chunk13v8 import U_prob` (dead: `U_prob` is always computed locally via `compute_probability_distributions()`); this was a distinct import from ticket 2's, tolerated during earlier fix passes but now removed entirely since nothing needs it. |
| 47 | `WFG` → `compute_hypervolume` — **had regressed** to the old class-call pattern; see §4.14. |
| 30/31/32/51 | Completed `evaluate_complete_solution` body; deleted orphaned module-level call blocks in M3 S5 and M4 S5; corrected `params=` keyword and 3-value unpack. **Ticket 31/32's unpack bug had regressed a second time** at the §5.2 `objective()` call site (5-value unpack of a 3-value return) — subsumed by ticket 69's consolidation, which removes the unpack entirely. |
| 44 | Facet-name comparison via `get_required_facets()` |
| 8/39/55 | `RELATION_MAP` and topology consolidated into Module 1 |
| 38 | Wrong C2 topology deleted with `CONFIG_SWITCHBOARD` |
| 40 | Config-specific `anchor_keys` now used instead of `ALL_POSSIBLE_ANCHORS` |
| 42/5/10 | Single `load_and_validate_data` in Module 1 |
| 59 | `K_LIST` changed to `[2, 3, 4, 5, 6]` — see §4.6. |
| 63 | Confirmed (not a code change): mega-hub damping for article-author ties lives upstream in chunk12's `build_log_damped_row_csr` (chunk12.py:291-307), applied to `S_Art_Auth` specifically (chunk12.py:328). 13v9 does not need its own damping logic. See §11. |
| 65 | `CONFIG_IDS` includes `'C6'`; `get_active_facets('C6')` returns valid `sem_keys=['M_Atom_Child','M_Child_Art','M_Cousin_Art','M_Fringe_Cousin','M_Cousin_Parent','M_Child_Parent']`, `anchor_keys=['M_Child_Art','M_Cousin_Art']`, all present in `RELATION_MAP`. Smoke-tested end-to-end (K=3, 2 trials, 5 epochs) — passed, exercised the ticket 64 fix specifically since C6 is dual-anchor. |

---

### "UDSR" investigation (Test 2) — historical-record correction + findings

A prompt this session referred to a "ubiquity-discounted L1" mechanism ("UDSR") as
something "we removed" from the pipeline. **That framing does not match what's on disk.**
Grepped every `chunk13*.py` version (v2 through v9, v8.1, `chunk13soc1.py`,
`chunk13_metafac.py`) for `UDSR`, `ubiquity`/`discount`, and `idf` — the only near-hit is
`chunk13.v8.1.py:27`, `DECODER_PATH = os.path.join(WORKSPACE_TOY,
"outputs/Star_epistemic_decoders_global.pkl")` — a path constant that is **defined and never
referenced again anywhere in that 207-line file**. `chunk12.py:360`'s comment
(`# [!] EXPORTED: 13v8 will use this for the Doxa Tax Exemption`) confirms the *intent*
existed upstream, but no version of chunk13 — including v8.1 itself — ever implements an
idf-weighted or ubiquity-discounted L1 term. **This was planned and stubbed, not built and
removed.** `chunk13v9.py` has zero references to `idf`, `decoders`, or any pickle other than
the T1/T2 matrices pickles — confirmed both by grep and by a runtime check against the loaded
module source in Test 2's script.

**Test 2 results** (using the `lambda_l1=0.75` and `lambda_l1=74.6` runs from ticket 77's
Test 1; `idf_global` loaded from `Star_epistemic_decoders_global.pkl`, keyed exactly like
`U_prob` — `{facet: {entity_index: idf_value}}` — for the 5 semantic facets):

| facet | ρ(idf, row-max) @ λ=0.75 | ρ @ λ=74.6 | n live w/ idf |
|---|---|---|---|
| `core_atom` | −0.177 (p=0.002) | +0.029 (p=0.61, n.s.) | 300 |
| `core_child_he` | +0.168 (p=0.031) | +0.075 (p=0.34, n.s.) | 166 |
| `cousin_he` | +0.258 (p=0.0002) | −0.108 (p=0.12, n.s.) | 210 |
| `fringe_atom` | −0.279 (p<0.0001) | −0.090 (p=0.12, n.s.) | 307 |
| `parent_he` | undefined (constant input) | undefined (constant input) | 63 |

`parent_he`: all 63 T1-live entities share the *exact same* `idf_global` value
(4.4177 — confirmed directly, not a script bug: the facet only has 2 unique idf values across
all 160 global entities, and by chance/construction all T1-live ones fall in the same bucket).
Spearman is undefined here, not weakly correlated — flagging so it isn't misread either way.

**Reading, per the prompt's own framing:** the correlations are weak, inconsistent in sign
across facets, and **none are significant at λ=74.6** (all four |ρ| < 0.11, p > 0.1). At
λ=0.75 two are nominally significant but small (|ρ| < 0.28) and **opposite in sign between
facets** (`core_atom`/`fringe_atom` negative, `core_child_he`/`cousin_he` positive) — i.e. in
some facets ubiquitous entities skew toward *higher* row-max, in others toward *lower*, with
no consistent direction. This doesn't cleanly satisfy either of the prompt's two READ
branches ("ubiquitous=low, niche=high, vindicated" vs "ubiquitous=high, hoarding") — the data
doesn't show a clean pattern either way at this sample size (n≈14-307 per facet, only 5
semantic facets, single K/config/slice). **The correlation shift λ=0.75→74.6 is toward zero
almost everywhere** (`core_atom` +0.21, `fringe_atom` +0.19, `core_child_he` −0.09,
`cousin_he` −0.37) — consistent with ticket 77's Test 1 finding that high `lambda_l1`
flattens row-max broadly rather than selectively; a term that flattens indiscriminately can't
selectively fix hoarding even if hoarding existed. **This does not resolve whether UDSR-style
weighting is needed** — it wasn't tested (no such term exists to test), and the existing L1 +
Doxa-check-only architecture is neither vindicated nor indicted by this data. Full per-entity,
per-facet numbers (20 most ubiquitous / 20 most niche, no aggregation) in
`diagnostic_results/test2_udsr_ubiquity.json`.

---

## 9. Known Limitations

- **Non-convexity.** CNMF has many local minima. Current mitigation is NNDSVD initialisation
  plus fixed seeds. Multi-start, lambda annealing, and BCD-style Z updates are discussed but
  not implemented.
- **Shared semantics.** Ubiquitous semantic elements that should load ~1/K across all
  communities are structurally disfavoured — NMF's winner-takes-all tendency plus the
  `z_offdiag` penalty (which suppresses the cross-community coupling that would naturally
  carry shared-element signal) both push against it. Unresolved.
- **Config C1** has a single anchor (`M_Parent_Art`, 63 non-zeros — see §11), so its entire
  semantic initialisation tree roots at one weak SVD. Expect weaker initialisation quality
  and weaker scout-phase hypervolume than C2/C5/C6 (all dual-anchor). **This is the point of
  the comparison, not a bug to fix** — see §4.15. Do not add special-casing for C1.

---

## 10. Planned Extensions (not yet implemented)

Recorded as intent, not as contracts. Current code does **not** implement any of this —
`PIPELINE_VERSION` marks this as `time_slice_1` only. T2 data (`Star_extended_matrices_t2.pkl`)
exists on disk (§11) but nothing in the pipeline currently loads it.

### Temporal slices (T1 → T2)

When the chronological loop is built, the output of slice T1 will be passed into the T2
solver as `U_prior`, and the temporal term will be a **differentiable distance metric**
against that prior, moderated by a tunable inertia parameter.

*Rationale:* a soft prior, not a hard constraint. Forcing structural immutability between
slices would prevent detection of field emergence and dissolution — the substantive
phenomenon of interest. The prior models institutional memory, so that observed shifts in
the field are data-driven rather than artifacts of stochastic initialisation. This matches
the treatment of every other constraint in this pipeline (soft penalties, never projections).

**Alignment caveat — resolve before implementing.** Community *k* in T1 and community *k* in
T2 are not the same community unless explicitly matched. A temporal penalty computed against
an unaligned `U_prior` would penalise arbitrary label permutations rather than genuine
structural drift, producing a meaningless inertia signal. Module 4 Section 5 already
contains Hungarian alignment (`scipy.optimize.linear_sum_assignment` over JSD and cosine
cost matrices); reuse that machinery to align T1 → T2 before computing any temporal
distance.

**Presence masks are already in place for this.** `build_presence_masks()` (§11, ticket 60)
identifies live-vs-padded entities per slice; any T1↔T2 alignment or comparison must restrict
to entities live in *both* slices, not just live in one.

### Model-quality evaluation for the main (22k-article) corpus

Currently the only model-quality signals are `math_loss` (reconstruction fit) and the three
`sociological_penalty` terms (`collapse_pen`, `coherence_pen`, `semantic_pen` — reality
checks the model is penalised against, Module 3). Both are already in use; neither tests
whether the discovered structure *generalises* or is *repeatable* across runs, which are
different questions from "does it fit" or "does it pass the reality checks." Proposed for
the full corpus, not the toy corpus — diagnostic-only, no production wiring implied by
listing it here:

1. **Held-out link prediction / cross-validation.** Mask a subset of observed (nonzero)
   entries per relation, fit on the rest, evaluate reconstruction on what was held out.
   Standard in the matrix-factorisation/recommender-systems literature; the only measure of
   the three considered that tests generalisation rather than in-sample fit. **Caveat to
   resolve before implementing, not after:** naive uniform-fraction masking is unsafe on
   sparse relations — `M_Parent_Art`'s 63 non-zeros (T1, §11) is the concrete toy-corpus
   example of a relation too sparse to mask without risking destabilising the fit entirely.
   Masking design must be relation-aware (skip or use a much smaller fraction on sparse
   relations) rather than a blanket k-fold. Re-check relation-level sparsity at 22k-article
   scale before fixing a masking fraction — density may differ substantially from the toy
   corpus's.

2. **Stability / consensus across seeds (cophenetic correlation or a simpler pairwise
   co-membership agreement).** Standard in the NMF-for-clustering literature (Brunet et al.
   2004 popularised cophenetic correlation specifically for NMF rank/K selection) — refit the
   same (config, K) cell across multiple seeds and measure how consistently entities get
   assigned to the same community, rather than how well any single fit reconstructs the data.
   Tests a different question than either loss or the reality checks: whether the *same*
   structure is found repeatably, not just a well-fitting one. Directly buildable with
   existing diagnostic tooling (`fit_or_load(..., seed=...)` already supports this) — no new
   fitting machinery required, only the consensus/agreement computation on top of it. This is
   also the belated implementation of "Test c: co-membership consensus," named early in this
   project's Problem-2 diagnostic agenda and never built.

**Considered and rejected: an R²/explained-variance analog.** Since every relation is already
Frobenius-normalised to `‖X‖²=1` (§4.2), a per-relation "fraction of variance explained"
number is almost free to compute (`relation_recon`, already computed per-relation inside
`run_inner_solver` before being summed into pooled `math_loss` — not currently exposed
per-relation, but derivable diagnostically without touching production). Rejected as a
*model-quality* metric (as opposed to a harmless relabelling) for three reasons: (a) no
K-adjustment — raw reconstruction loss decreases monotonically with K "for capacity reasons
alone" (§4.6), which is exactly why K-comparison already goes through hypervolume rather than
raw loss; an R² number would silently reintroduce that trap if ever used across K without the
same guard. (b) No calibrated baseline exists for what residual level counts as "good" on
sparse (~98% zero) binary relational data under a non-negative rank-K constraint — importing
continuous-regression R² intuitions (0.3 "decent," 0.7 "strong") would likely misread a
reasonable low-rank fit as a poor one. (c) It adds no information beyond what `math_loss`
already provides — it is a rescaling, not a new signal, unlike (1) and (2) above.

**Status: proposal only, not scheduled, not built.** No code exists for either (1) or (2).
Revisit when model-quality evaluation for the 22k-article run is actually being planned.

---

## 11. chunk12 Facts (upstream data generator)

Established from reading chunk12.py directly (`tensor_data_staging/toy_large/chunk12.py`,
`build_temporal_metagraph()`), not from 13v9. Relevant because 13v9 must not "fix" any of
this — it is deliberate upstream behavior that 13v9's own design already assumes.

### Global-vs-slice-specific entity indexing

chunk12 builds T1 and T2 from **one corpus** with two disjoint article-year windows
(T1: 2019–2022, T2: 2023–2025). Per its own comment (chunk12.py:76-78):

> Persistent domains are GLOBAL; Transient domain (art) is SLICE-SPECIFIC

Concretely: `get_idx(domain, key)` assigns permanent global integer IDs shared across both
slices for `auth`, `affil`, `journ`, `parent_he`, `core_child_he`, `core_atom`, `cousin_he`,
`fringe_atom`. `get_art_idx(art_id, is_t1)` assigns slice-local sequential IDs separately for
T1 and T2 articles. This means every non-`art` facet has **identical dimensions** across T1
and T2 — this is deliberate and is what will make T1↔T2 community comparison possible later
(§10). **Do not "fix" this to give each slice its own compact indexing** — that would break
the entire premise of the planned temporal extension.

**Consequence:** an entity (author, hyperedge, etc.) that only appears in T2 still occupies a
row/column in T1's matrices, filled with zeros — and vice versa. See live-entity fractions
below; this is severe (as low as 39% live in T1). Ticket 60 (`build_presence_masks`) exists
because of this.

### No `'dimensions'` key in the pickles

`compile_slice()` (chunk12.py:322-343) returns a plain `dict[relation_key -> scipy_sparse]`
with no `'dimensions'` entry, and neither pickle-dump call (chunk12.py:366, 370) adds one
after the fact. **Module 1's auto-discovery branch in `load_and_validate_data` is the live
code path on every run, not a fallback** — see §4.10. Do not add logic that assumes a
`'dimensions'` key might be present in the payload.

### Measured non-zero counts (both slices)

Via direct inspection of `Star_extended_matrices_t1.pkl` / `_t2.pkl`:

| Relation | T1 shape | T1 nnz | T2 shape | T2 nnz |
|---|---|---|---|---|
| `S_Art_Journ` | (25, 20) | 24 | (36, 20) | 35 |
| `S_Art_Auth` | (25, 461) | 218 | (36, 461) | 258 |
| `S_Auth_Affil` | (461, 66) | 95 | (461, 66) | 97 |
| `M_Atom_Child` | (642, 405) | 552 | (642, 405) | 898 |
| `M_Child_Parent` | (405, 160) | 211 | (405, 160) | 330 |
| `M_Fringe_Cousin` | (577, 435) | 481 | (577, 435) | 521 |
| `M_Cousin_Parent` | (435, 160) | 267 | (435, 160) | 304 |
| `M_Cousin_Child` | (435, 405) | 958 | (435, 405) | 1003 |
| `M_Parent_Art` | (160, 25) | **63** | (160, 36) | 98 |
| `M_Child_Art` | (405, 25) | 193 | (405, 36) | 313 |
| `M_Cousin_Art` | (435, 25) | 261 | (435, 36) | 301 |
| **TOTAL** | | **3,323** | | **4,158** |

(`auth`, `affil`, `journ`, and all semantic facets share the same dimension across T1/T2 as
documented above — only `art`'s dimension differs, 25 vs 36.)

### Measured live-entity fractions (union across active relations, ticket 60)

| Facet | T1 live / total | T1 % | T2 live / total | T2 % |
|---|---|---|---|---|
| `art` | 25/25 | 100.0% | 36/36 | 100.0% |
| `journ` | 14/20 | 70.0% | 12/20 | 60.0% |
| `affil` | 31/66 | 47.0% | 40/66 | 60.6% |
| `auth` | 212/461 | 46.0% | 253/461 | 54.9% |
| `core_atom` | 300/642 | 46.7% | 465/642 | 72.4% |
| `core_child_he` | 166/405 | 41.0% | 263/405 | 64.9% |
| `parent_he` | 63/160 | **39.4%** | 97/160 | 60.6% |
| `cousin_he` | 210/435 | 48.3% | 259/435 | 59.5% |
| `fringe_atom` | 307/577 | 53.2% | 388/577 | 67.2% |

T1 is consistently sparser (later time window, T2, has had more corpus growth) — every
non-`art` facet is 39–70% live in T1 vs 55–72% live in T2. This is the concrete magnitude
behind ticket 60's fix and CLAUDE.md's earlier claim ("at most 47% of authors and 39% of
parent hyperedges are live" in T1 — now measured exactly: 46.0% and 39.4% respectively).

### Mega-hub damping lives upstream, not in 13v9 (ticket 63)

chunk12's `build_log_damped_row_csr()` (chunk12.py:291-307) caps total row mass at
`log2(1 + degree)` instead of letting it scale linearly with degree, and is applied
specifically to `S_Art_Auth` (chunk12.py:328, `# [!] UPDATED: Apply logarithmic dampening to
Article-Author ties`) — the article→author tie matrix. `S_Auth_Affil` instead uses plain
row-stochastic normalization (`build_row_stochastic_csr`); grammar relations
(`M_Atom_Child` etc.) are unweighted binary. **13v9 must not reimplement hub damping** —
the article-author weights it receives are already log-damped by the time they reach
`load_and_validate_data`.
