# Ticket 84 — hub down-weighting for grammar relations (chunk12 relation weighting)

> **STATUS: UNAPPROVED WORKING PLAN — NOT A SETTLED RECORD.** Written by Opus in a design
> pass, then independently triangulated by Sonnet (see the "Sonnet triangulation" section
> near the end) — both passes' claims were checked against code/data, not just written down.
> Nothing in it has been executed: no `chunk12.py` change, no cache change, no diagnostic
> script written, no data regenerated. Unlike `CLAUDE.md` and `FINDINGS.md` — durable records
> of what has actually been measured and decided — this document becomes obsolete once
> executed or rejected. **Do not cite it as established** until D5 is confirmed and the
> corresponding `CLAUDE.md §4.21`/`FINDINGS.md §21` corrections are made.
>
> Supersedes this file's previous contents (the ticket-82 domain-balance plan), whose
> settled design, D1–D3 decisions and results are now recorded permanently in
> CLAUDE.md §4.18 / §8 ticket 82 and FINDINGS.md §13 + §21. Ticket 82's remaining work
> (E2, the in-loop term) is unaffected by this plan.

## Terminology

**Hub down-weighting** (the codebase calls it "damping", chunk12.py:291
`build_log_damped_row_csr`, *"Remedy for Mega-Hubs: Limits total row mass to
log2(1+degree)"*): reducing the weight of each connection belonging to an entity that has
very many connections, so that entity's influence grows sub-linearly rather than in direct
proportion to its connection count. **Degree** = an entity's number of connections in a
given relation. **Per-tie weight** = what one individual connection is worth, as opposed to
an entity's total.

---

## Context

CLAUDE.md §4.21 proposed extending chunk12's relation weighting to the 5 currently-unweighted
"grammar" relations (`M_Atom_Child`, `M_Child_Parent`, `M_Fringe_Cousin`, `M_Cousin_Parent`,
`M_Cousin_Child`) and to `S_Art_Journ`. Motivating goal, stated by the user: **a single
high-degree entity should not dominate a community** — a many-author article should not, on
the strength of that one article, pull all its co-authors into one community.

The design pass asked a prior question first: **is this already handled by the pipeline's
existing normalisations?** Measured answer — it is handled in one place and not in another,
and that split is what this ticket is actually about.

**Already handled — at initialisation, more precisely at the propagation stage that follows
it.** `initialize_tucker_adapted_nndsvd_and_propagate` (chunk13v9.py:312) has two distinct
stages, worth naming precisely rather than lumping under "initialisation": **Step 2** runs a
real NNDSVD (proper truncated SVD) on **anchor relations only** — this is the actual
initialisation, and it is where the whole facet tree is rooted; **Step 3** then radiates
outward from those anchor-seeded facets via the *grammar* relations, using a cheaper
row/col-normalise-and-project step (a "Degree-Corrected Multi-Hop Phase 2 Propagation" per
the function's own docstring), not a second SVD. The row-normalisation issue below lives
entirely in Step 3, not in the anchor SVD itself.

`chunk13v9.py:435-439` row-normalises each matrix during that Step-3 propagation, with the
literal comment *"Row-Normalize M to prevent Hub Dominance"*. This **exactly annihilates**
any per-row rescaling: `diags(1/(s·orig))·(s·M) = diags(1/orig)·M`.

**Verified independently by a second model (Sonnet), not just asserted by the first (Opus)**
— traced the actual control flow of Step 3's propagation loop for all six configs (which
branch, `f1`/row-normalised/cancelled vs. `f2`/column-normalised/survives, each grammar
relation takes, given each config's real anchor set and active relation list):

| Relation | Cancelled (`f1`) | Survives (`f2`) |
|---|---|---|
| `M_Atom_Child` — worst-skew relation | **C1, C2, C3, C4, C5, C6 — all six** | never |
| `M_Fringe_Cousin` | **C1, C2, C3, C4, C5, C6 — all six** | never |
| `M_Child_Parent` | C1 | **C2, C3** |
| `M_Cousin_Parent` | C1 | **C6** |
| `M_Cousin_Child` | C3, C4 | never |

`M_Atom_Child`'s "cancelled in all six configs" claim holds exactly. Two corrections to an
earlier pass of this table: `M_Child_Parent` survives via `f2` in **both** C2 and C3 (not
C3 alone), and does **not** survive in C6 — in C6 it is not used for initialisation at all
(both its facets are already initialised via `M_Cousin_Parent` by the time it's reached);
only `M_Cousin_Parent` survives there. Also newly noted: `M_Fringe_Cousin` is cancelled in
**all six** configs too, not just `M_Atom_Child` — strengthening, not weakening, the
"acts almost entirely through the loss, not init" scope limit already stated below. Neither
correction changes any of the decisions D1–D8 or the execution phases — they were
illustrative context, not load-bearing.

**Not handled — in the training loss.** Frobenius normalisation (`chunk13v9.py:543-551`) is a
single global scalar per matrix; it does not equalise rows. Measured share of squared row mass
entering the loss, held by the highest-degree rows:

| Relation | top-1% rows (now → down-weighted) | top-5% rows (now → down-weighted) |
|---|---|---|
| `S_Art_Auth` — **already down-weighted** | (29.8%) → **1.8%** actual | (29.8%) → **1.8%** actual |
| `M_Atom_Child` | **12.0%** → 0.9% | **26.1%** → 5.7% |
| `M_Child_Parent` | 4.3% → 0.7% | 17.1% → 6.0% |
| `M_Fringe_Cousin` | 5.8% → 1.1% | 21.0% → 5.9% |
| `M_Cousin_Parent` | 5.6% → 1.2% | 16.9% → 6.1% |
| `M_Cousin_Child` | 4.8% → 0.7% | 17.5% → 3.9% |

(T1; T2 same pattern. For `S_Art_Auth` the parenthesised figure is the counterfactual *without*
its existing down-weighting; **1.8% is where it actually sits today**.)

**So the ticket is a consistency question, not a novel-problem question:** grammar relations
currently let their top-5% of rows carry 13–29% of the fitted mass, against the **1.8–2.2%**
standard already applied to `S_Art_Auth`. Down-weighting brings them to 3.9–6.3%, comparable
to that accepted standard. *Caveat, stated not hidden:* `S_Art_Auth` has only 25/36 rows, so
its top-1% and top-5% are the same single row — the percentile comparison is coarse. The gap's
magnitude (roughly 6–15×) far exceeds that imprecision, but the comparison is not exact.

**Intended outcome:** grammar relations receive the same loss-level hub treatment article-author
ties already receive, decided on deterministic evidence, with no chunk12 rebuild unless and
until that evidence passes.

---

## Decisions settled in this pass

| # | Decision | Resolution |
|---|---|---|
| D1 | Formula | **Reuse `build_log_damped_row_csr` verbatim** (chunk12.py:291-306) by passing a Counter of 1s. Since `count=1`, it reduces exactly to `w = log2(1+degree)/degree`. Not an analogy to the precedent — bit-for-bit the same computation on a different input. `S_Art_Auth`'s own Counter is already all-1s (its mass 6.044 = log2(66) exactly). **Zero new formula, zero new tunable constant.** |
| D2 | idf vs degree | **Degree. Reject idf**, despite idf being available for all 5 grammar row axes (`dfreq_global` covers all 5 semantic facets; `core_atom`/`fringe_atom` idf is computed and currently never used). Reasons: degree and idf are strongly **negatively** correlated (Spearman −0.39 to −0.73, all p<1e-6) — they measure the same ubiquity on inverted scales, so applying both double-discounts it; idf resolution here is very coarse (2–17 distinct values per facet); the per-tie property the goal requires is a degree property. |
| D3 | Damping strength | **`log2` verbatim, not tuned.** §4.21 asked for the constant to be "picked by evidence" — but with idf resolution of 2–17 values and a noise floor exceeding the signal (FINDINGS §21), **this corpus cannot adjudicate a damping exponent.** That is the argument for zero new tunables, stronger than "matches precedent". Flag for re-derivation at 22k, as `TOL` and `lambda_l1` bounds already are. |
| D4 | `S_Art_Journ` | **Defer — §4.21 is wrong that this is "low risk, no new mechanism."** Verified: it needs (a) **new** df collection (no `dfreq_global['journ']` write exists) and (b) a **column-keyed** idf variant (`build_anchor_csr` keys `idf_dict.get(r)` on the *row*; journals are the column axis). Additionally every row has degree exactly 1, so after Frobenius normalisation column-idf is arithmetically equivalent to reweighting whole *articles* by their journal's rarity — a substantive modelling claim ("articles in rare journals matter more") unrelated to this ticket's goal. Correct §4.21's text. |
| D5 | §9 justification | **Strike it.** `U_prob` is L1 row-normalised (`chunk13v9.py:711-734`), discarding a row's scale — so down-weighting a hub changes its influence on *other* entities but **not its own loading shape**, which is what §9 is about. Less gradient pressure on a hub row also leaves it determined *more* by init noise, arguably mildly *against* §9. And §9's effect has never been demonstrated (Test 2: weak, sign-inconsistent, mostly non-significant). Do not justify a data-layer change by an undemonstrated limitation the mechanism would not fix. **User decision pending — see Open Question below.** |
| D6 | Axis | **Row only**, matching precedent. Document that `M_Cousin_Child` has column skew in T2 (12.2×) left unaddressed. |
| D7 | Slice basis | **Per-slice degree** (user decision). Recorded tradeoff: grammar-relation rows are *global* entities present in both slices, so the same atom can carry different weights in T1 vs T2. `S_Art_Auth`'s precedent does **not** cover this — its rows are articles, which exist in exactly one slice, so no article ever has two degrees. This introduces a T1/T2 asymmetry into matrices that were previously pure structure. **Flag as a documented limitation to revisit when the T1→T2 temporal extension (§10) is built**, since weighting drift could there be mistaken for structural drift. |
| D8 | Scope | **All 5 grammar relations, with Tier-0 metrics reported per relation.** They encode the same kind of thing (ontological containment); treating them differently on toy-corpus skew would be the special-casing §4.15 warns against and would need re-deriving at 22k regardless. Per-relation reporting keeps anomalies visible — notably `M_Cousin_Child`, which uniquely has **no degree-1 rows**, so it is affected everywhere rather than only in the tail. |

### The honest scope limit, stated up front

Because init already annihilates row rescaling for `M_Atom_Child` in every config, and because
`U_prob` discards row scale, this change acts **almost entirely through the loss** — as a
relative mass transfer from hub rows toward degree-1 rows. With 65–88% of grammar rows at
degree exactly 1, that is a **broad reallocation across most of the matrix, not a tail-only
fix**. This is a reason for careful measurement, not reassurance.

---

## Execution — read-only first, chunk12 last

The transform is a pure function of the *existing* matrices (degree = row `nnz`, recoverable
from the built CSR). So the scientific question can be answered **without regenerating
anything**, decoupling it from the blast radius entirely.

### Phase 1 — Tier 0, read-only, no writes anywhere (**GATE**)

New `diagnostic_scripts/grammar_damping_tier0.py`. Loads existing T1/T2 pickles via
`diagnostic_blocks.load_slice`, applies the transform **in memory**, computes:

1. **Per-tie weight ratio** at degree p90 vs p10, per relation, both slices. Bar: **≥3×**
   (reference: `S_Art_Auth`'s measured 5×). This is the property §4.21 explicitly requires.
2. **Share of `‖X‖²_F` held by (a) top-1% rows by degree, (b) all degree-1 rows**, before/after.
   Gate on (b) as the fragmentation guard — ceiling set after first measurement, not guessed.
3. **Row max÷mean concentration** — FINDINGS §21's table, recomputed before/after.
4. **Effective-connectivity check** (the real fragmentation risk — topology is unchanged, no
   edge is removed, so a plain connectivity test would trivially pass): Fiedler value of the
   normalised Laplacian of the `core_child_he` projection `XᵀX` under `M_Atom_Child`, plus
   best-2-cut conductance, before/after. Also the direct mechanism metric: **per child
   hyperedge, the fraction of incoming atom mass supplied by shared (degree≥2) vs unique
   (degree-1) atoms** — down-weighting raises the unique-atom share, which *is* the
   fragmentation mechanism made numeric.
5. **Bit-identity assertion**: the 3 anchor and 3 social matrices must be byte-identical
   before/after. Cheapest, strongest regression test available.

**If Tier 0 fails, the ticket ends here with nothing regenerated and nothing to revert.**

### Phase 2 — cache namespacing (prerequisite for any fitting)

`diagnostic_blocks.py:418-433` `_cache_key` hashes `chunk13v9.py` source but **not the input
data** — 379 cached fits (52.6 MB) would be silently reused against changed matrices, printing
`[cache hit]` as success. **Namespace, do not invalidate**: add the input-data hash to the
cache *filename* so both generations coexist and are legible — the old fits are *needed* for
Phase 3's paired comparison. Hash canonicalised `(indices, indptr, data)` per matrix, **not**
pickle bytes (pickle is not guaranteed byte-stable).

### Phase 3 — Tier 1, fit-level, on a new pickle path (still no chunk12 change)

Write the transformed matrices to a **new** pickle path; point the loader at it. chunk12 is
still untouched.

**Mandatory: pair the seeds.** Fit old and new data at the *same* 5 seeds and compare paired
differences. Much of FINDINGS §21's 0.144 SD is which local optimum a seed lands in — a common
component pairing removes. This is the single largest lever against the noise floor.

- **Primary:** ρ(row degree, `U_prob` row-max) per facet over live entities (166–307 per facet
  — aggregates far more stable than per-community readings). Noise floor: run §21's exact
  5-seed + Track-A alignment protocol on **unchanged** data, take across-seed SD of ρ. Bar:
  correct sign (toward less positive) in **≥4 of 5 facets across both slices**, with
  |Δρ| > 2·SD_paired in ≥2. **Direction consistency across facets is the real test** —
  magnitudes on this corpus will not carry.
- **Report only, do not gate:** pooled `dev_k` across the full 76-cell grid, paired Wilcoxon.
- **Explicitly ruled out as gates:** `recon_loss` and convergence rate. Each relation is
  Frobenius-normalised to a *different* object before and after, so the two losses are not
  comparable; down-weighting makes the matrix more uniform, i.e. easier, so a lower loss is a
  tautology. Keep both as health checks (did it converge, same order of magnitude) only.
- **Pre-registered:** no domain-balance improvement is required, per D5's mechanism argument.

### Phase 4 — port into chunk12, last

Only once the answer is known. Back up **all four** output pickles first — `_t1.pkl`,
`_t2.pkl`, **and `Star_epistemic_decoders_global.pkl`** (chunk12.py:352-364; contains `maps`,
`maps_t1_art`, `maps_t2_art`, `idf_global`). There are **no backups and no git history** for
any of them (`tensor_data_staging` is a git repo with zero commits). ~280 KB total.

Then edit `compile_slice` (chunk12.py:333-337) to route the 5 grammar relations through
`build_log_damped_row_csr` with a Counter-of-1s, and **null-rebuild**: re-run chunk12 and
verify output matches the Phase-3 transformed pickles. **Compare `maps` FIRST, then matrices** —
chunk12's entity indices depend on `graphbrain hg.search()` iteration order, never verified
stable; a pure index permutation yields "different" matrices that are equivalent, while
identical matrices under different maps would be a silent catastrophe.

---

## Files

- **New:** `chunk13_execution/diagnostic_scripts/grammar_damping_tier0.py` (Phase 1, read-only)
- **New:** a Phase-3 paired-comparison script, reusing `domain_balance_seed_noise.py`'s 5-seed +
  `s5_dual_track_alignment` Track-A protocol verbatim
- `chunk13_execution/diagnostic_blocks.py` — `_cache_key` namespacing (Phase 2)
- `tensor_data_staging/toy_large/chunk12.py` — `compile_slice` only (Phase 4)
- `PhD_CNMF/CLAUDE.md` — §4.21 (correct the `S_Art_Journ` and §9 claims), §8 ticket 84,
  §11 if any nnz figures move; `FINDINGS.md` — new subsection under §21

## Reuse (do not reimplement)

`build_log_damped_row_csr` (chunk12.py:291), `load_slice`/`fit_or_load`/`presence_masks`
(diagnostic_blocks.py), `s5_dual_track_alignment` + `apply_seed_permutation` (the seed-alignment
protocol validated in FINDINGS §21), `facet_membership_profile`, `compute_probability_distributions`.

## Verification

- **Liveness is provably unchanged** — verified: `build_presence_masks` (chunk13v9.py:244-277)
  uses `mat.getnnz(...)`, never values; `log2(1+d)/d > 0` for all `d≥1`; chunk12 never calls
  `eliminate_zeros`. No presence-mask regression is possible. (Assert it anyway in Phase 1.)
- Phase 1 metrics 1–5 above, per relation, both slices, no aggregation across slices.
- Phase 3 paired-seed protocol with its own measured noise floor per metric.
- Phase 4 null rebuild: `maps` first, then matrices.
- All results persisted with their scripts per SESSION_PROTOCOL §D; T1/T2 never pooled.

## What this ticket does NOT claim

- Not a fix for §9 (shared/ubiquitous semantics) — see D5.
- Not an initialisation change — annihilated there for `M_Atom_Child` in all configs.
- Not a demonstrated improvement in per-community domain balance — the noise floor
  (FINDINGS §21) makes that undemonstrable on this corpus, and D5 argues none is expected.
- Not a tuned damping strength — `log2` verbatim, re-derive at 22k.

---

## Sonnet triangulation (independent second-model review)

Requested by the user specifically to cross-check this plan before any implementation. Method:
independently re-verified the load-bearing claims against code and data rather than reviewing
the prose alone — traced the actual initialisation/propagation control flow for all 6 configs
(see the corrected table under Context above), re-derived the `dfreq_global`/anchor-idf
row-vs-column-keying facts behind D4, and re-checked the `U_prob` L1-row-normalisation
argument behind D5.

**Verdict: the central mechanical claim holds** (`M_Atom_Child` cancelled at init in all six
configs — confirmed, not just trusted), **with two factual corrections applied above**
(`M_Child_Parent`'s C2/C3/C6 survival pattern; `M_Fringe_Cousin` also fully cancelled). D1–D4,
D6, D8 independently re-confirmed as stated. **D5 — recommend striking §9 as a justification**
(the L1-row-normalisation argument is correct: scaling a row doesn't change its direction, and
`U_prob` discards exactly that scale, so this mechanism cannot produce a §9-style effect).
Sequencing (read-only Tier 0 before touching `chunk12.py`) endorsed as the right order given
the missing cache-hash coverage, absent backups, and unverified `graphbrain` determinism found
during the design pass.

**Net recommendation: proceed to Phase 1 (the read-only Tier-0 script).** It costs nothing to
run, commits to nothing, and the gate is set before any data is touched — the open D5 question
below does not need to block starting it, since Tier 0's metrics don't depend on the §9
question either way.

---

## Open question for the user before implementation

**D5 (§9)** — two independent design/review passes (Opus's plan, Sonnet's triangulation) now
agree the evidence says to strike §9 as a justification for this ticket. Recorded as a
recommendation, not yet a user-confirmed decision. Confirming it also means correcting §4.21
and FINDINGS §21 in `CLAUDE.md`/`FINDINGS.md`, where the link to §9 is currently asserted as
fact — not yet done, since this plan has not been approved for implementation.
