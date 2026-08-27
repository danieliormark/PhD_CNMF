# Ticket 84 — hub down-weighting for grammar relations (chunk12 relation weighting)

> **STATUS: PHASES 1-4 EXECUTED ON THE TOY CORPUS. `chunk12v2.py` exists, verified clean,
> but is NOT the production default** (`chunk12.py`/`'T1'`/`'T2'` untouched and unchanged;
> `chunk12v2.py`/`'T1_v2'`/`'T2_v2'` are new, additive artifacts). Written by Opus in a
> design pass, triangulated by Sonnet, then Phases 1 (Tier-0 gate, PASS), 2 (cache
> namespacing), 3 (paired-seed fit validation, MIXED — 3/6 correct direction, 3/6 wrong, one
> confirmed non-noise anomaly in each direction, corrected mid-session for a real
> cross-config-pooling bug), and 4 (`chunk12v2.py` created, null-rebuild verified clean) were
> all executed this session. **D5 (§9 justification): RESOLVED, user-confirmed — struck.**
> See its own section below. **User decision on Phase 3's mixed result: held, not chased
> further on the toy corpus, revisit at 22k scale** — the noise floor is comparable to the
> effect size throughout (consistent with FINDINGS §21's standing finding); no evidence the
> mechanism is actively harmful, only that it isn't cleanly demonstrable at this scale. **Do
> not cite the Phase 3 result as settled** — it is an honestly-reported mixed outcome on an
> admittedly small, noisy corpus, not a validated finding either way.
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
| D5 | §9 justification | **Struck — RESOLVED, user-confirmed.** `U_prob` is L1 row-normalised (`chunk13v9.py:711-734`), discarding a row's scale — so down-weighting a hub changes its influence on *other* entities but **not its own loading shape**, which is what §9 is about. Less gradient pressure on a hub row also leaves it determined *more* by init noise, arguably mildly *against* §9. And §9's effect has never been demonstrated (Test 2: weak, sign-inconsistent, mostly non-significant). Do not justify a data-layer change by an undemonstrated limitation the mechanism would not fix. §9 itself remains open/unresolved as its own pipeline limitation — only the *link* to this ticket is removed. |
| D6 | Axis | **Row only**, matching precedent. Document that `M_Cousin_Child` has column skew in T2 (12.2×) left unaddressed. |
| D7 | Slice basis | **Per-slice degree** (user decision). Recorded tradeoff: grammar-relation rows are *global* entities present in both slices — confirmed directly at `chunk12.py:77-95` (`get_idx` assigns one permanent integer per string the first time it's seen, across T1 and T2 together; only `art` gets slice-local indices via a separate `get_art_idx`) — so the same atom/hyperedge string can carry different weights in T1 vs T2. `S_Art_Auth`'s precedent does **not** cover this — its rows are articles, which exist in exactly one slice, so no article ever has two degrees. This introduces a T1/T2 asymmetry into matrices that were previously pure structure. **Flag as a documented limitation to revisit when the T1→T2 temporal extension (§10) is built**, since weighting drift could there be mistaken for structural drift. |
| D8 | Scope | **Not uniform — split by what the row entity *is*, revised from the original "all 5 uniformly."** See "Scope split" subsection immediately below for the full argument and data. |

### D8 revised: scope split by row-entity type, not "all 5 uniformly"

The original D8 treated all 5 relations as encoding "the same kind of thing" (ontological
containment) and applied one formula uniformly to avoid special-casing. Discussion with the
user surfaced a real structural distinction the original framing missed: the row entity
differs by *type* across these relations, and that type difference tracks a real difference
in what "high degree" means substantively — not just numerically.

| Group | Relations | Row entity | Decision |
|---|---|---|---|
| A — atom-rooted | `M_Atom_Child`, `M_Fringe_Cousin` | lexical atom (a term/word) | **Apply** `log2(1+d)/d` — same argument as D1-D3: degree here is a direct ubiquity signal, structurally identical to co-authorship's hub effect. Strongest measured hub signature after `S_Art_Auth`: max/median ratio 5.17×/6.34× (`M_Atom_Child`, T1/T2) and 2.89×/2.89× (`M_Fringe_Cousin`), against the precedent's own 5.03×/1.95×. |
| B — hyperedge-rooted, untouched | `M_Child_Parent`, `M_Cousin_Parent` | already-composed child/cousin hyperedge | **Leave undamped.** Row entity is a composed unit, not a raw lexical item — recurrence across 2-14 parents (only 11-15% of rows even reach degree 2) plausibly reflects genuine cross-cutting synthesis (the same sub-claim feeding multiple larger arguments), not noise. Weaker measured signature too (2.71×/3.58×, 2.89×/2.14×) — near or below the precedent, not clearly a hub problem. This is the user's point 3: several co-authors ≠ several shared words, and by the same logic, several shared child-hyperedges ≠ automatic ubiquity noise. |
| — | `M_Cousin_Child` | cousin hyperedge (same type as Group B) but with a **distinct, bespoke justification** | **Apply**, same formula, but NOT for the Group-A "ubiquity" reason — see below. |

#### `M_Cousin_Child`: a different, substantively-grounded justification

`M_Cousin_Child`'s row entity (`cousin_he`) is the same *type* as Group B's, so it doesn't
automatically inherit Group A's ubiquity argument. It gets its own, checked against real data:

The interpretation stage (per the user, who works with colleagues on this) focuses on
**focal/child hyperedges**; cousin hyperedges exist only to **clarify the context** a child
hyperedge was used in. Row degree in `M_Cousin_Child` (how many distinct child hyperedges a
given cousin has been observed alongside, corpus-wide) is therefore not a ubiquity proxy —
it is a **direct measurement of context genericness**: a cousin recurring across 24 different
children is doing essentially no disambiguating work for any one of them; one recurring
across only 2 is doing real, specific work.

Checked against both slices (live rows, i.e. degree ≥1; `M_Cousin_Child`'s degree floor is 2,
not 1 — no monodegree rows exist here):

| | lowest-degree cousins (deg=2, "specific") | highest-degree cousins (deg=20-24, "generic") |
|---|---|---|
| T1 | `effectiveness`; `art dataset method`; `capability generalization ... pretrain`; `experiment` | `train`, `propose`, `show`, `name`, `learn`, `develop` |
| T2 | `benchmark dataset first proteinlmbench`; `manually`; `choice multiple question`; `944` | `present`, `use`, `study`, `predict`, `can`, `enable` |

The high-degree tail is boilerplate reporting verbs ("we train/propose/show/present..."); the
low-degree end is specific technical content (dataset names, numeric quantities, domain
terms). This is structural, not incidental: **`dummy_cousin`** (the predicate-headed
periphery type, as opposed to **`cousin_he`**, the noun/modifier-phrase periphery type — see
the `7.5postprocessing_4hbased_correct.py` verification below) is 29-31% of all live cousins
overall, but **52-60% of the top decile by degree**, both slices.

Consequences for the design, given this is a different mechanism than Group A's:
- **Row-only (D6) is now substantively justified for this relation**, not just by
  consistency: the column axis (`core_child_he`) is where interpretation happens and should
  keep receiving whatever context volume it gets; the ticket's job is only to discount the
  *tellers* that aren't discriminating, not the *listeners*.
- The degree-floor-of-2 (no monodegree rows) makes the "still distinguishable from
  monodegree entities" property (the user's condition 4) non-binding by construction, but the
  same monotonicity holds: mass(2)=log2(3)=1.585, mass(24)=log2(25)=4.64 — total mass still
  strictly favors high-degree rows, while **per-tie weight inverts by ~4.1×** (0.79 vs 0.19),
  so a specific cousin's individual tie counts roughly 4× a generic cousin's.
- **New Phase-1 diagnostic, specific to this relation**: report the `dummy_cousin` vs
  `cousin_he` split of the down-weighted mass, before/after. The effect will land
  disproportionately on predicate-type cousins by construction (the 52-60% figure above) —
  a predictable, named consequence, not something to discover as a surprise later.

#### Structural verification of the row/facet semantics (not just the weighting argument)

Traced `7.5postprocessing_4hbased_correct.py`'s `extract_dual_matrices`/`sweep_periphery`
against one real sampled sentence (article `W2911489562`, BERT/language-model context) with
the destructive purge-on-import and the spaCy lemmatizer removed/stubbed (surface-form
passthrough only — does not change which atoms land in which structure). Confirms the user's
description exactly: a `parent_he` circuit = `dummy_sibling` (governing predicate) +
`focal_he` (target-anchored) + `sibling_he` (co-arguments, same parent) — all folded
indiscriminately into `core_child_he` (`chunk12.py:199-203`, `link[2][1:]`). `cousin_he` /
`dummy_cousin` come from `sweep_periphery`, walking branches outside the core parent→focal
spine but under a shared higher ancestor — "same grandparent, different branch," per the
user's description, not "same immediate parent."

#### Why the group split does not distort relative influence between relations

Checked directly against `chunk13v9.py`, prompted by the question of whether a relation with
higher typical row degree (like `M_Cousin_Child`) would out-compete lower-degree relations in
the fit purely by virtue of degree: **no**. Two mechanisms already neutralize this before any
per-relation weighting choice: **Frobenius normalization** (`:543-544`) rescales *every*
relation to unit Frobenius norm regardless of density/degree, and **`w_sem`** (`:558`) splits
a fixed 0.5 budget equally across whichever semantic relations are active in a config
(`chunk13v9.py:112-136`, 4-6 relations depending on config) — `M_Cousin_Child` gets exactly
the same per-relation vote as any other. Higher degree only redistributes influence *within*
a relation's own fixed vote (the effect this ticket targets), not *between* relations. The one
genuine cross-relation channel — `core_child_he` is a shared facet (column of `M_Atom_Child`,
row of `M_Child_Parent`, column of `M_Cousin_Child`) with one shared column-scale gauge
(`:597-606`) — is the same pre-existing column-skew limitation D6 already documents, not
something this decision changes.

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

**Scope, per the D8 revision above: only `M_Atom_Child`, `M_Fringe_Cousin`, `M_Cousin_Child`
get the transform.** `M_Child_Parent`/`M_Cousin_Parent` are reported for context (current-state
stats only, no "after" — they are not modified).

New `diagnostic_scripts/grammar_damping_tier0.py`. Loads existing T1/T2 pickles via
`diagnostic_blocks.load_slice`, applies the transform **in memory**, computes:

1. **Per-tie weight ratio, corrected definition.** The original draft of this metric said
   "p90 vs p10" with a "≥3×" bar derived from `S_Art_Auth`'s **max-vs-median** ratio (5.03×
   T1 / 1.95× T2) — two different quantities, caught before Phase 1 was written. Corrected
   metric: **max-vs-median per-tie ratio**, matching how the bar was actually derived. Before
   damping this ratio is trivially 1.0× for these matrices (all grammar ties are binary
   counts, so every tie is worth the same regardless of degree); after damping it's
   `w(median)/w(max)`. Bar: **at or above the precedent's own weaker slice (1.95×)** — not a
   fixed "3×", since the precedent itself doesn't clear that in both slices.
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
6. **`M_Cousin_Child`-specific: `dummy_cousin` vs `cousin_he` share of down-weighted mass**,
   before/after — see the D8 revision above. Named in advance as a predictable, structural
   consequence of this relation's bespoke justification, not a side effect to discover later.

**If Tier 0 fails, the ticket ends here with nothing regenerated and nothing to revert.**

#### Phase 1 — EXECUTED. Gate: PASS.

Run: `chunk13_execution/diagnostic_scripts/grammar_damping_tier0.py`, both slices, no
`chunk12.py` change, no cache write, no fit run. Full output:
`diagnostic_results/grammar_damping_tier0.json`.

- **Metric 1 (corrected):** all 3 damped relations clear the bar (≥1.95×) both slices —
  `M_Atom_Child` 5.17×/6.34×, `M_Fringe_Cousin` 2.89×/2.89×, `M_Cousin_Child` 3.45×/3.04×.
- **Metric 2:** top-1% mass share compresses as expected (e.g. `M_Atom_Child` 12.0%→0.9%
  T1). **Fragmentation-guard ceiling (deferred in the original plan to "after first
  measurement") is now set from this run**: post-damping, degree-1 rows' mass share is
  consistently **91-96% of their own population share** (`M_Atom_Child` T1 70.3% of rows →
  64.9% of mass, ratio 0.923; T2 ratio 0.915; `M_Fringe_Cousin` T1 ratio 0.936, T2 ratio
  0.956) — i.e. damping brings them toward, but not past, proportional representation in all
  4 measured cells. **Proposed ceiling: mass share ≤ population share (ratio ≤ 1.0)**,
  measured with headroom in every case so far.
- **Metric 3:** row max/mean concentration falls sharply for all 3 (e.g. `M_Atom_Child`
  13.04→3.53 T1, 16.57→3.76 T2).
- **Metric 4:** the original spectral sub-metric was degenerate as first written — the
  `core_child_he` projection under `M_Atom_Child` is **already disconnected into 3 (T1) / 4
  (T2) components before any transform**, which forces a normalized-Laplacian Fiedler value
  of exactly 0 regardless of damping (a disconnected graph's 2nd-smallest eigenvalue is 0 by
  construction) — caught before reporting a false pass. Fixed: restricted to the largest
  component (110/114 T1, 200/208 T2 — the disconnection itself is minor). Within it: Fiedler
  value falls in both slices (0.175→0.156 T1, 0.180→0.138 T2 — damping loosens connectivity
  slightly, as expected: hub atoms were literally the shared connective tissue). Conductance
  **disagrees in direction between slices** (T1: 0.131→0.199, better-mixed; T2: 0.272→0.218,
  more separable) — per SESSION_PROTOCOL rule 8, reported as-is, not reconciled; sample is 2
  slices, not enough to call a trend either way. **The direct mechanism metric is the one
  that matters and it's unambiguous**: mean shared-atom mass fraction per child hyperedge
  falls in both slices (0.531→0.480 T1, 0.630→0.575 T2) — i.e. unique-atom share rises ~5
  points both times, the fragmentation mechanism confirmed numerically, consistent direction.
- **Metric 5:** all 6 anchor/social matrices byte-identical before/after, both slices.
- **Metric 6:** confirms the prediction made when `M_Cousin_Child`'s justification was
  written, not just re-derives it — `dummy_cousin` mass share falls after damping in both
  slices (37.2%→31.4% T1, 37.7%→33.7% T2), i.e. the transform measurably shifts
  `M_Cousin_Child`'s mass toward `cousin_he` (noun/modifier-phrase) structures and away from
  `dummy_cousin` (generic predicate) structures, as the substantive argument predicted.

**Gate verdict: PASS.** Proceeding to Phase 2 (cache namespacing) is next — not yet done,
since it touches `diagnostic_blocks.py`, shared infrastructure used by other diagnostics, and
warrants a check-in before editing rather than being folded into this same read-only pass.

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

#### Phase 2 — EXECUTED.

`diagnostic_blocks.py` `_cache_key` now includes `_raw_data_hash(slice_name)` — a content
hash over each relation's canonical `(indices, indptr, data)`, sorted by relation key, **not**
pickle bytes (unstable across versions) and **not** the file path (unchanged when a file is
regenerated with new content). Made visible in the filename itself
(`{config}_K{K}_{slice}_{path}_data{10-char hash}_{12-char digest}`), not just buried in the
opaque digest, per the plan's own "namespace... and legible" instruction. `BLOCKS_VERSION`
bumped `1.12.0`→`1.13.0`.

**Verified additive, not destructive:** all 379 pre-existing cache files
(`diagnostic_results/tensors/*.npz`) remain on disk, byte-untouched, under their original
names — none match the new `_data<hash>_` pattern. A fit request against unchanged data now
computes a *different* filename than before this change (the naming scheme itself changed),
so it will `[cache miss]` once and regenerate under the new scheme — a one-time cost, not
invalidation of the old files, which stay available under their original names for Phase 3's
paired before/after comparison. Smoke-tested: `_raw_data_hash('T1')` ≠ `_raw_data_hash('T2')`
(distinct matrices), deterministic across repeated calls (memoized), and present verbatim in
a sample `_cache_key()` output.

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

#### Phase 3 — EXECUTED. Verdict: MIXED / NOT A CLEAN PASS — flagged, not rounded up.

Run: `grammar_damping_phase3_seed_pairs.py`, 240 fits (6 configs × K∈{3,4} × 2 slices × 5
seeds × {baseline, damped}), 230/240 converged (T1 3/120 skipped, T2 7/120 skipped — not
gated, per plan, but recorded). Side pickle, additive `SLICE_PATHS` entries, `chunk12.py`
untouched. Full output: `diagnostic_results/grammar_damping_phase3_seed_pairs.json`.

**Correction to the primary metric's scope, made before running (disclosed, not silent):**
the original bar ("≥4 of 5 facets") predates the D8 scope split. Re-derived which facets can
show real evidence at all: `core_atom`/`fringe_atom`/`cousin_he` (their own row-degree was
actually changed by this ticket) as positive-evidence targets; `core_child_he` (row-degree in
the untouched `M_Child_Parent`) as a **negative control**, expected to show no shift;
`parent_he` **excluded** — verified it is never a row entity in any active relation, so no
row-degree exists for it to test. Revised bar: correct direction in ≥2 of 6 (facet, slice)
target cells, |Δρ| > 2×seed-SD in ≥1. Also: alignment machinery (`s5_dual_track_alignment`)
was **not needed** for this metric and deliberately not used — `U_prob` row-max is provably
invariant to which column index a community gets relabelled to, unlike `r_k`.

**Measured (ρ = degree vs. `U_prob` row-max, mean across configs/K within each seed, then
mean/SD across 5 seeds):**

| Facet | Slice | baseline ρ | damped ρ | Δρ | direction | vs. 2×SD |
|---|---|---|---|---|---|---|
| `core_atom` | T1 | −0.367 | −0.302 | +0.066 | **wrong** | — |
| `fringe_atom` | T1 | −0.307 | −0.332 | −0.025 | correct | — |
| `cousin_he` | T1 | +0.014 | +0.035 | +0.021 | **wrong** | — |
| `core_child_he` (control) | T1 | −0.114 | −0.110 | +0.004 | n/a | negligible, as predicted |
| `core_atom` | T2 | −0.336 | −0.438 | −0.102 | correct | — |
| `fringe_atom` | T2 | −0.289 | −0.239 | +0.051 | **wrong** | **exceeds 2×SD** |
| `cousin_he` | T2 | −0.138 | −0.278 | −0.140 | correct | — |
| `core_child_he` (control) | T2 | −0.162 | −0.158 | +0.004 | n/a | negligible, as predicted |

**Mechanical bar result: 3/6 correct direction (≥2 required), 1/6 exceeds 2×SD (≥1
required) → PASS by the letter of the threshold.** Not reported as a clean pass regardless,
for three reasons, per SESSION_PROTOCOL rule 7/8 (report what was measured, don't resolve
ambiguity, don't reconcile disagreement):

1. **3/6 is a coin flip, not the direction consistency the plan itself said was "the real
   test" on this corpus.** A bar built to require barely more than chance is a weak bar —
   worth naming as a limitation of my own threshold design, not just of the data.
2. **The one cell that clears the noise floor with statistical confidence moved the wrong
   way**: `fringe_atom`/T2 exceeds 2×SD in the direction opposite the prediction. That is a
   more informative fact than the 3-2 headline split — it says the one place this corpus can
   actually distinguish signal from noise, it disagrees with the hypothesis.
3. **Baseline ρ for `core_atom`/`fringe_atom` is already substantially negative in both
   slices (−0.29 to −0.37) before any damping.** The ticket's original framing assumed
   high-degree entities get pulled *toward* one dominant community (a positive ρ to correct).
   That is not what baseline shows here — if anything the reverse tendency already holds.
   Pushing an already-negative ρ further negative (as happened in 3 of the 4 `core_atom`/
   `fringe_atom` cells) is a real, measured effect, but whether it is a *correction* or an
   *overcorrection* relative to the ticket's original goal is not obvious from this number
   alone, and is a substantive question, not a statistical one.

**What is clean:** the negative control behaved exactly as predicted in both slices
(Δρ ≈ 0.004, negligible) — the transform is not producing spurious shifts in facets it never
touched, which is the specificity check this control existed for.

**Not decided here — left for the user.** Per SESSION_PROTOCOL rule 7, this is reported, not
resolved: whether a 50/50 direction split with the one confident move pointing the wrong way
is sufficient evidence to proceed to Phase 4, or whether it should hold the ticket at "mixed,
inconclusive on this corpus" pending the 22k-scale re-derivation already flagged throughout
this plan (D3, D7, D8).

#### Phase 3 re-analysis (user-prompted methodology check; no re-fitting — same 230 cached
fits, re-read at finer granularity) — one real bug found, headline verdict otherwise holds.

User questions surfaced a genuine flaw, not just a theoretical risk: `cousin_he`'s pooled
number silently averaged across all 6 configs, but `M_Cousin_Child` (the relation actually
damped) is only active in 2 of them (C3, C4) — the other 4 have **no causal path** for the
transform to affect that facet at all. Restricting to C3/C4 only: T1 stays wrong-direction
(+0.033 pooled → +0.017 restricted, both weak), **T2's correct-direction result gets larger,
not smaller** (−0.070 pooled → −0.224 restricted) but now rests on only 3-4 config/K cells,
not 11 — a more fragile estimate, cutting both ways. `core_atom`/`fringe_atom` don't have
this problem (`M_Atom_Child`/`M_Fringe_Cousin` are active in all 6 configs) — checked
directly: same-sign in 10-12 of 12 config/K cells for both, so pooling was reasonably sound
there specifically, not merely assumed.

Second measure checked (N_eff, `1/Σp²`, reused verbatim from `toy_large.ipynb` Chunk 14D's
`calc_neff` — not invented): agrees with row-max as a near-exact mirror image in 22-23 of 23
cells. **Choice of concentration measure does not change the conclusion on this corpus** —
row-max was an adequate proxy here, though N_eff remains the more theoretically complete
choice (uses the full loading vector, not just the top entry) for future work.

Label-switching: confirmed empirically, not just argued — a fixed entity's argmax community
index genuinely changes across seeds (1→0→2...), while row-max (which never reads community
identity, only the largest value) needs no alignment to remain comparable, by construction.

**Net effect on the verdict: unchanged in headline count, sharper in substance.** Scripts:
`grammar_damping_phase3_reanalysis.py` (no new fits — reused the cache from the prior run).
Full output: `diagnostic_results/grammar_damping_phase3_reanalysis.json`.

#### Corrected table (SUPERSEDES the Phase 3 table above) — same aggregation method as
originally reported (average row-max per entity across configs within a seed, then
correlate, then mean/SD across the 5 seeds), each row now restricted to the configs where
that facet's own relation is actually active — no re-fitting, same 230 cached fits, re-read.

| Facet | Slice | baseline ρ | damped ρ | Δρ | Direction | Configs used |
|---|---|---|---|---|---|---|
| `core_atom` | T1 | −0.367 | −0.302 | +0.066 | wrong | all 6 (unchanged — always was) |
| `fringe_atom` | T1 | −0.307 | −0.332 | −0.025 | correct | all 6 (unchanged — always was) |
| `cousin_he` | T1 | **+0.117** | **+0.161** | **+0.045** | wrong, more clearly | C3, C4 only (`M_Cousin_Child` active) |
| `core_child_he` (control) | T1 | −0.124 | −0.101 | +0.023 | negligible, ≈predicted | C1, C2, C3, C6 (`M_Child_Parent` active) |
| `core_atom` | T2 | −0.336 | −0.438 | −0.102 | correct | all 6 |
| `fringe_atom` | T2 | −0.289 | −0.239 | +0.051 | wrong — exceeds 2×SD | all 6 |
| `cousin_he` | T2 | **+0.136** | **−0.110** | **−0.245** | correct — **now also exceeds 2×SD** | C3, C4 only |
| `core_child_he` (control) | T2 | −0.138 | −0.156 | −0.018 | negligible, ≈predicted | C1, C2, C3, C6 |

`core_atom`/`fringe_atom` are byte-for-byte the numbers already reported — they were always
computed over all 6 configs (`M_Atom_Child`/`M_Fringe_Cousin` are active everywhere), so the
bug never touched them; confirmed by rerunning, not assumed. `core_child_he` (control) shifts
modestly but stays small, consistent with "no shift expected." `cousin_he` changes
substantially: **T2's baseline correlation flips sign** (−0.138 pooled → **+0.136**
restricted-to-causally-relevant-configs) — in the 2 configs that actually use the damped
relation, high-degree cousins genuinely were more concentrated before damping (matching the
ticket's original hypothesis), and damping reverses that, now clearing the noise floor.

**Bar tally under the corrected table: still 3/6 correct direction (unchanged), but now 2/6
exceed 2×SD (was 1/6) — one in each direction** (`fringe_atom`/T2 confidently wrong,
`cousin_he`/T2 confidently correct). Not a resolution toward "pass" — a sharper demonstration
that this corpus produces genuinely conflicting, not merely noisy, signal. The original
"mixed, not a clean pass, left for the user" verdict stands, on firmer ground than before.

Per-slice tally, precisely (differs from an even split — worth stating exactly, not loosely):
T1 = 1 correct / 2 wrong / 1 negligible-control; T2 = 2 correct / 1 wrong / 1
negligible-control. Pooled: 3 correct / 3 wrong across the 6 target cells — an exact tie.

#### DISPOSITION (user decision, this session): hold Phase 4, do not chase further on the
toy corpus. Revisit with 22k-scale data. One named open item, not swept under the rug.

Considered and rejected as not essential right now: tracing individual wrong-direction
entities' training trajectories to distinguish between candidate mechanisms (baseline-premise
mismatch, reduced-gradient-pressure/init-determined placement, small-corpus noise). Useful for
eventual mechanistic understanding, not required for this decision.

**"T1 is undersampled" — checked, holds partially, not completely.** T1 has measurably fewer
live entities than T2 across every relevant facet (already-documented asymmetry, not new:
`core_atom` 300 vs 465, `core_child_he` 166 vs 263, `fringe_atom` 307 vs 388, `cousin_he` 210
vs 259 — 63-84% of T2's counts), and T1 does show more wrong-direction cells (2 vs T2's 1),
consistent with the sparser slice being noisier. **But it does not fully explain the pattern**:
the one wrong-direction result that actually clears the noise floor (`fringe_atom`/T2,
exceeds 2×SD) sits in T2 — the *better-resourced* slice, not the thin one. If undersampling
were the whole story, T2 should be the cleaner slice; it isn't, quite. Flagged as the one open
item to specifically re-check at 22k scale, not just re-run the same aggregate test.

**Not disastrous, checked against the noise floor rather than asserted:** of the 3
wrong-direction cells, 2 (`core_atom`/T1 +0.066, `cousin_he`/T1 +0.045) do **not** clear 2×SD
— statistically indistinguishable from ordinary seed noise, plausibly just noise. Only
`fringe_atom`/T2 is a confirmed, non-noise anomaly, and it's small in absolute size. No
evidence the mechanism is actively backfiring — evidence it isn't cleanly demonstrable on
this corpus, which is a different (and expected, per FINDINGS §21's standing noise-floor
finding) conclusion.

**A targeted attempt to shrink `cousin_he`'s noise floor specifically (extending its 5 seeds
to 15, C3/C4 only — the thinnest sample in the whole test) was launched but died silently
mid-run** (`grammar_damping_cousin_he_more_seeds.py`) — completed T1/baseline (15/15 seeds)
then stopped with no error, no traceback, right at the start of the damped-data pass. Checked
for OOM/disk causes, found none; likely an HPC login-node resource policy invisible to this
user, not a script bug. **Not relaunched** — consistent with the decision to stop chasing
precision on the toy corpus rather than pursue it further right now.

### Phase 4 — new versioned file, not an in-place edit (REVISED — user proposal, safer than
the original "edit + backup" design, adopted)

**`chunk12.py` is left completely untouched — not even renamed.** Verified directly: nothing
in this codebase imports `chunk12.py` or references its filename in any live code path (only
two hits outside comments, both dead commented-out lines in old `chunk13v2.py`/
`chunk13_metafac.py`). Everything downstream — `diagnostic_blocks.SLICE_PATHS`,
`chunk13v9.py` — consumes chunk12's *output pickles* by path, never the source file itself.
So there is nothing to gain from renaming it and a (small but nonzero) chance of a mistake in
doing so — leaving it alone is strictly safer than a symbolic "legacy" rename.

**New file: `chunk12v2.py`** — a copy of `chunk12.py`, matching this codebase's own existing
convention (`chunk13v2.py` through `chunk13v9.py` already coexist side by side; this isn't a
new pattern). `compile_slice` is edited in the copy only, routing the 3 target relations
(D8: `M_Atom_Child`, `M_Fringe_Cousin`, `M_Cousin_Child` — **not** `M_Child_Parent`/
`M_Cousin_Parent`, Group B, left as `build_grammar_csr`) through `build_log_damped_row_csr`
with a Counter-of-1s.

**`chunk12v2.py` writes to NEW, distinctly-named output pickles** —
`Star_extended_matrices_t1_v2.pkl`, `_t2_v2.pkl`, `Star_epistemic_decoders_global_v2.pkl`,
same `outputs/` directory, `_v2` suffix — **not** overwriting the current production files.
This is a stronger guarantee than "backed up before overwriting": the production pickles are
never touched at all, not even transiently, so there is nothing to restore if anything goes
wrong. Making the new data usable is a small, purely additive config change — new
`diagnostic_blocks.SLICE_PATHS` entries (`'T1_v2'`/`'T2_v2'`) — the exact same pattern already
proven out in Phases 1-3 for the side-pickle `_damped` entries, now backed by a real,
versioned, git-trackable source file instead of a diagnostic script's transform. Whether
`chunk12v2`'s output ever becomes the new default (`'T1'`/`'T2'` repointed to it) is a
**separate, later, deliberate decision** — explicitly not forced by this step.

**What does NOT change under this revision**: running `chunk12v2.py` still means re-deriving
*everything* from the curated `graphbrain` database from scratch — articles, authors, all
coordinate extraction — not just patching 3 matrices in an existing pickle; that is how
`chunk12.py` has always worked (source-derived, never cached). So the **null-rebuild
verification is unchanged and still essential**: compare `maps` FIRST between `chunk12v2`'s
fresh output and `chunk12`'s existing output (`graphbrain hg.search()` iteration-order
stability has never been verified — a pure index permutation would make "different" matrices
actually equivalent, while identical matrices under different maps would be a silent
catastrophe), THEN compare matrices — `M_Child_Parent`/`M_Cousin_Parent` and every social/
anchor relation should be byte-identical to `chunk12.py`'s output; the 3 target relations
should match `log_damp_row(chunk12's output)` exactly, the same transform already validated
in Phases 1 and 3. This is, if anything, a *more* rigorous test than the original in-place
plan — it validates the whole pipeline end-to-end, not a post-hoc patch of an existing pickle.

**EXECUTED.** `chunk12v2.py` created (copy of `chunk12.py`, only `compile_slice` and the 3
output paths changed — full diff is exactly the intended minimal change, checked before
running). Ran in **16.7s** (`real 0m16.712s`) — a data pipeline over `graphbrain` SQLite, not
a model fit, much faster than the up-to-2-minute estimate given beforehand. Dimensions
matched production exactly on first run (`art=25/36, auth=461, affil=66, journ=20,
parent_he=160, core_child_he=405, core_atom=642, cousin_he=435, fringe_atom=577`).

**Null-rebuild verification — both stages pass:**
- **Maps (checked first, as required):** every one of the 8 `maps` domains, `maps_t1_art`,
  `maps_t2_art`, and `idf_global` — **byte-identical** between `chunk12.py`'s existing output
  and `chunk12v2.py`'s fresh run. `graphbrain`'s `hg.search()` iteration-order risk (flagged
  as never-verified throughout this plan) did **not** materialize in this comparison.
- **Matrices, both slices:** all 8 untouched relations (`S_Art_Journ`, `S_Art_Auth`,
  `S_Auth_Affil`, `M_Child_Parent`, `M_Cousin_Parent`, `M_Parent_Art`, `M_Child_Art`,
  `M_Cousin_Art`) **byte-identical**, confirming Group B and all social/anchor relations were
  genuinely untouched. All 3 damped relations (`M_Atom_Child`, `M_Fringe_Cousin`,
  `M_Cousin_Child`) match `log_damp_row(chunk12's original)` exactly — same topology, same
  transformed values — in both slices.

**Production pickles (`Star_extended_matrices_t1.pkl`/`_t2.pkl`,
`Star_epistemic_decoders_global.pkl`) were never touched.** New files only:
`chunk12v2.py`, `Star_extended_matrices_t1_v2.pkl`/`_t2_v2.pkl`,
`Star_epistemic_decoders_global_v2.pkl`.

**Not yet done**: registering `'T1_v2'`/`'T2_v2'` in `diagnostic_blocks.SLICE_PATHS` (needed
to actually fit anything against this data), and the `CLAUDE.md`/`FINDINGS.md` documentation
updates the "Files" list names. Both are small, additive, low-risk next steps — not started
without a separate go-ahead, consistent with this session's pacing.

---

## Files

- **New:** `chunk13_execution/diagnostic_scripts/grammar_damping_tier0.py` (Phase 1, read-only)
- **New:** a Phase-3 paired-comparison script, reusing `domain_balance_seed_noise.py`'s 5-seed +
  `s5_dual_track_alignment` Track-A protocol verbatim
- `chunk13_execution/diagnostic_blocks.py` — `_cache_key` namespacing (Phase 2, done); new
  `'T1_v2'`/`'T2_v2'` `SLICE_PATHS` entries (Phase 4, additive)
- **New:** `tensor_data_staging/toy_large/chunk12v2.py` (Phase 4) — copy of `chunk12.py`,
  `compile_slice` edited only. `chunk12.py` itself is untouched, not renamed.
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

## D5 — RESOLVED (user confirmed): §9 struck as a justification

**Settled.** User: "You can work with this justification as you deem pertinent. If the
consensus is to remove, then remove it." Both independent passes (Opus's design, Sonnet's
triangulation) agreed on removal; no dissent. `CLAUDE.md §4.21`/`§8` and `FINDINGS.md §21`
corrected accordingly — the §9 *link* is removed from this ticket's justification. **§9
itself is untouched and remains open** in `CLAUDE.md §9` (Known Limitations) — this only
removes the claim that ticket 84 addresses or was validated against it; §9 stands as its own,
separately unresolved pipeline limitation, independent of this ticket's outcome.

---

## Deferred work items — pick up here after context compaction

Not started, not scheduled for this session — recorded here as one list, rather than
scattered across D3/D4/D6/D7's individual mentions, specifically so a future session doesn't
have to hunt for them. None of these block anything currently in progress.

1. **`S_Art_Journ` (D4) — RESOLVED, but not via the idf route originally proposed.**
   User-driven investigation (post-compaction) found the real problem wasn't degree-skew
   needing an idf/damping formula at all: row degree (articles per journal-tie) is
   structurally capped at 1 by construction (article→journal is many-to-one), so a
   row-keyed formula would be mathematically inert; the actual skew was **column**-side
   (journal degree) and traced to exactly two non-selective preprint repositories
   (arXiv, bioRxiv), later widened to four (+ChemRxiv, +Research Square) once found already
   present in this same data. Verified empirically: removing just those IDs flattens the
   remaining degree distribution to ~uniform in this toy corpus (no residual power-law tail),
   ruling out smooth damping as the right tool (nothing to damp along) in favor of resolution
   + exclusion. **Implemented:**
   - `toy_large.ipynb` cell 9 (ingestion): each repository-hosted article is now resolved
     against OpenAlex (same-record `locations` → cached `related_works` → live title search,
     in that cost order) to its real venue where one exists; `journal_id`/`journal_name` are
     overwritten if resolved, left as the repository label otherwise (never blanked — the
     raw per-article record stays the accurate historical fact). Disambiguation on a
     multi-candidate title match uses ORCID first, then normalized author-name overlap —
     explicitly never OpenAlex's own internal `author.id` (user instruction — not assumed
     stable across how the two sides of a match were sourced).
   - `chunk12.py`/`chunk12v2.py` (`compile_slice`): whatever is *still* a repository ID after
     that resolution attempt has its `S_Art_Journ` tie dropped entirely (structural zero, not
     damped) — "unknown true venue" is treated the same as "no venue recorded," matching the
     pipeline's pre-existing missing-journal convention. Caught and fixed a real latent bug
     surfaced by this change: `get_art_idx`'s first-seen index allocation was gated on the
     journal tie being recorded, so widening the skip condition reshuffled unrelated
     articles' indices as a side effect — fixed by calling `get_art_idx` unconditionally,
     gating only the coordinate itself.
   - Live resolution run (authenticated OpenAlex tier — `mailto`+`api_key`, university
     credentials, env-var only, never committed): of 59 originally repository-hosted
     articles in the 121-row source corpus, **10 resolved** to a real venue (e.g. ELECTRA→
     ICLR via the free same-record path; Nucleotide Transformer→Nature Methods,
     GENA-LM/RNA-alignment paper→Nucleic Acids Research, etc. via live search), **49
     confirmed no non-repository venue exists** (mostly still-unpublished/unlinked preprints
     — ALBERT, PaLM, LLaMA, InstructGPT). Result reproduced identically across two runs
     (anonymous-pool attempt before hitting a 429, then the authenticated re-run) — same 10
     resolved, same 49 unresolved both times, so the split is a real property of OpenAlex's
     data, not an artifact of rate-limiting.
   - Verified end-to-end: `journ` dimension 20→20 net (4 repositories dropped, 5 distinct
     real venues added, 2 reused existing indices); every non-`S_Art_Journ` relation and
     every non-`journ` map byte-identical to pre-change; `chunk12.py`/`chunk12v2.py` remain
     idempotent and mutually consistent (null-rebuild discipline held) after the change.
   - **Not addressed by this fix, left as-is:** the idf/anchor-formula route from the
     original D4 framing was never built and is now moot — the resolution+exclusion
     mechanism replaces it, not extends it.
2. **`M_Cousin_Child`'s column skew (D6) — INVESTIGATED, decided: leave undamped.**
   Two-part investigation (user-driven, post-compaction), both confirmed against real data
   rather than assumed:
   - **Is the concentration a parsing artifact?** No — verified directly. The T2 outlier
     (`(focal_he llms/C/en)`, column-degree 84 vs. a median of 5) sits correctly in
     `maps['core_child_he']` (the child facet); zero occurrences of "llms" exist in
     `maps['cousin_he']`, ruling out a `sweep_periphery` misclassification. Per the user:
     the corpus was deliberately built by selecting sentences *because* they contain focal
     words (llm, bert, etc.), so this facet's degree concentration is a designed property of
     the sampling, not an artifact — the earlier idf-based "generic anchor term" reading was
     a misattribution, corrected once this context was supplied.
   - **Does that (real, deliberate) concentration distort the fit anyway?** Tested directly —
     `cousin_child_column_test.py`, paired baseline (`chunk12v2`, column undamped) vs.
     treatment (column additionally damped on top of the existing row damping), C3/C4 only
     (the only configs where `M_Cousin_Child` is active), K∈{3,4}, both slices, 5 seeds.
     Target: ρ(`core_child_he` column-degree in `M_Cousin_Child`, `U_prob` row-max). Result:
     weak positive baseline correlation both slices (0.13–0.16, SD 50–75% of the point
     estimate); treatment shift did not clear 2×SD in either slice (T1 moved the predicted
     direction but short of the bar; T2 moved the opposite direction, also short). Negative
     control (`core_child_he` row-degree in the untouched `M_Child_Parent`) stayed near zero
     both ways, confirming the weak target signal is specific to this relation, not a
     facet-wide artifact. Entity-level spot check on "llms" itself (T2, all 4 config/K
     cells): already strongly concentrated at baseline (row-max 0.60–0.77 of 1.0) regardless
     of treatment; post-damping direction inconsistent across cells (2 up, 2 down).
   - **Decision (user): leave `M_Cousin_Child`'s column axis undamped.** No demonstrated
     fitting distortion clears this corpus's noise floor, and the underlying concentration is
     real, deliberately-constructed structure — matching Group B's own standard (leave real
     structure alone absent clear evidence of harm), and avoiding added model complexity the
     evidence doesn't call for. Same "held, not chased further on this corpus, revisit at
     22k scale" posture already applied elsewhere in this ticket, for the same reason (noise
     floor comparable to or exceeding the candidate signal).
3. **Damping strength re-derivation (D3), and the infrastructure gap behind it.** `log2` is
   fixed, not tuned — this toy corpus's idf resolution (2-17 values) and noise floor
   (FINDINGS §21) rule out adjudicating an exponent. Re-derive at 22k scale. **Prerequisite
   gap, not previously stated this plainly:** legitimately tuning this parameter needs a
   held-out validation criterion (per the tuning-vs-p-hacking discussion this session —
   tuning against `recon_loss` is degenerate here, since more damping mechanically makes the
   matrix easier to reconstruct; tuning against a domain-balance-style metric would be
   p-hacking given the noise floor). `CLAUDE.md §10` lists held-out link prediction as
   "proposal only, not scheduled, not built." That gap needs closing before D3 can move past
   "fixed by default," independent of corpus scale.
4. **Per-slice degree basis (D7).** The same global entity can carry two different damping
   weights in T1 vs. T2 (grammar-relation rows are global entities, unlike `S_Art_Auth`'s
   articles, which exist in exactly one slice). Flagged to revisit specifically when the
   T1→T2 temporal extension (`CLAUDE.md §10`) is built, since weighting drift could then be
   mistaken for genuine structural drift between slices.
5. **Overall v1-(`chunk12.py`)-vs-v2-(`chunk12v2.py`) model-quality comparison — deferred by
   explicit user decision, do NOT rush this.** Phase 3 only tested the narrow mechanism
   claim (degree-vs-row-max correlation); the user separately wants a broader "which overall
   gives better results" comparison, and flagged correctly that it likely isn't
   one-dimensional — v2 could plausibly be better on one axis (e.g. domain balance) and worse
   on another (e.g. coherence), and that needs to be *reported* as a genuine multi-axis
   result, not collapsed into one verdict. A draft script exists as a starting point —
   `diagnostic_scripts/grammar_damping_v1_vs_v2_quality.py` (written, NOT run, not even
   smoke-tested yet) — reusing `evaluate()`/`domain_balance_r_k()` to compare
   `collapse_pen`/`coherence_pen`/`semantic_pen`/`sociological_penalty` and entity-weighted
   `dev_k`, with `recon_loss`/convergence explicitly reported as health-checks only (not a
   comparison signal — damping mechanically makes the target matrix easier to reconstruct,
   so a lower `recon_loss` on v2 would be a tautology, not evidence of better structure).
   **Before running it:** design pass first — decide what "better" means across possibly
   conflicting axes, and check per-config consistency before trusting any pooled number
   (exactly the aggregation pitfall the Phase 3 reanalysis caught this session — don't skip
   that check a second time by rushing this one).
6. **`chunk12v2.py` has no real version-control protection.** Checked directly: the `.git`
   root covering `tensor_data_staging` is actually `/mnt/hum01-home01/p91688di` — the whole
   home directory, not a repo scoped to this project — with ~198,000 pending file changes
   already staged (including unrelated files like cached OAuth credentials), never
   committed. `chunk12v2.py` shows as untracked in it. Not touched this session — that
   `.git` state is not safe to commit into casually. If addressed later: likely a clean,
   narrowly-scoped new repo for just the relevant subdirectory, not a commit into the
   existing one, and should not be attempted without explicit user go-ahead given what else
   is sitting in that index.
