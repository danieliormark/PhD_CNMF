# Tickets 86 + 87 — "ghost communities" (A) and per-relation concentration (B)

> **FIRST ACTION ON APPROVAL — commit and push this plan, then STOP for triangulation.**
> Copy this file to `PhD_CNMF/plans/ticket86-87-ghosts-and-relation-concentration.md`, commit it
> alone, and push to `origin/main`. Then pause — do not begin Stage 0. The user wants to read and
> triangulate the plan (as was done for ticket 84) before any work starts.
>
> Two notes for that commit: (1) `danieliormark/PhD_CNMF` is **public** (`"private": false`, verified
> via the unauthenticated GitHub API) — pushing plan files there is already established practice
> (`plans/ticket84-grammar-hub-downweighting.md` is on `origin/main`), so this follows the existing
> pattern rather than making a new exposure decision, but it is stated here so the choice is
> deliberate. (2) `CLAUDE.md` and `FINDINGS.md` currently have **uncommitted** modifications — the
> ticket-86 documentation written before the last compaction. They are deliberately left out of this
> commit; whether to commit them is a separate decision for the user.

> Supersedes this file's previous contents (the ticket-84 grammar-damping plan, Phases 1–4
> executed; its settled record lives in `CLAUDE.md` §4.21/§8 ticket 84 and `FINDINGS.md` §21).
> Ticket 84's **deferred items are preserved verbatim at the end of this file** — they are not
> part of this plan and are not all recorded elsewhere.

---

## Context

Two problems are on the table. The user's framing: **(A) ghost communities** — communities
holding very low share of the model's total reconstructed mass — and **(B) community-level
duopoly** — a relation's within-community mass concentrating on ~2 communities regardless of K.

### The vision: A and B are one phenomenon at two aggregation levels, and B is currently mismeasured

`community_share_k` (the ghost measure) is, by construction, the mean across active relations of
`_relation_community_share`, whose per-relation vector is `Z[k,π(k)]² / ‖U1·Z·U2ᵀ‖²_F`. So:

> **A is a deterministic function of B plus one thing nobody has measured: whether relations
> agree on *which* community wins.**

If every relation is a monopoly but a *different* community wins each, there are no ghosts — every
community owns something. If every relation is a monopoly and the *same* community keeps winning,
you get one giant plus K−1 ghosts. Per-relation concentration alone cannot predict ghosts; the
**winner-overlap across relations** is the missing half, and it has never been computed.

The existing evidence already shows the two dissociate. The affiliation-removal pilot cut C1/K6/T1's
`S_Art_Auth` top-2 share from 0.818 → 0.519 (4 of 6 communities gained real diagonal mass, up from 2)
— a large improvement in B — while the lowest community share went 0.063 → 0.059, i.e. **A did not
improve at all**. Treating them as one problem with one cause would be a mistake.

### B is not specific to `S_Art_Auth`, and `S_Art_Auth` is not the worst case

Every result recorded so far measures B only on `S_Art_Auth`. Recomputing across all relations for
the three flagged cells, using `Z²` (the quantity the mass measure actually uses, not `|Z|`):

`n_eff = 1/Σp²` over `p_k = Z[k,k]²/ΣZ[k,k]²`; K=6, so a fair spread is 6.0.

| Relation | T1/C1/K6 | T1/C3/K6 | T2/C3/K6 |
|---|---|---|---|
| `M_Atom_Child` | **1.00** | 1.10 | **1.00** |
| `M_Child_Parent` | 1.07 | 3.90 | 1.00 |
| `S_Auth_Affil` | 1.67 | 1.00 | 1.05 |
| `M_Fringe_Cousin` | 1.53 | 1.28 | 1.00 |
| `M_Cousin_Child` / `M_Cousin_Parent` | 2.61 | 2.27 | 1.94 |
| `S_Art_Auth` | 2.26 | 1.02 | 2.20 |
| `M_Parent_Art` / `M_Child_Art` | **5.70** | **5.17** | 3.21 |
| `S_Art_Journ` | **5.34** | **5.36** | 4.43 |

`M_Atom_Child` is a literal **monopoly** (n_eff = 1.00 — one community takes everything), worse than
`S_Art_Auth` anywhere. It is also the relation ticket 84 independently identified as having the worst
hub skew in the topology. Meanwhile three relations spread near-fairly. **B is a spectrum across
relations, not a property of one relation**, and the relation the whole investigation focused on sits
in the middle of it.

### The likely reason B looks the way it does — and why the current measurement can't tell

`_hungarian_relabel_relation` refuses to trust a relation's labelling when `structure_score ≤ 1.0`
(some cross-community entry is as large as some within-community entry — genuine mixing, not a
labelling artifact). Across the 60-cell v1 grid:

| Relation | cells where labelling is *trusted* |
|---|---|
| `M_Cousin_Art` | 30/30 |
| `M_Child_Art` | 39/50 |
| `S_Art_Journ` | 33/60 |
| `M_Atom_Child` | 21/60 |
| **`S_Art_Auth`** | **12/60** |
| `M_Cousin_Child` | 3/20 |

The relations that spread are the ones judged structurally clean; the relations that "duopolize" are
the ones judged **genuinely mixed**. That raises a specific, testable alternative to the current
story: *the duopoly may largely be an artifact of reading the diagonal of a relation whose mass is
genuinely off-diagonal.* `interference = 1 − Σ_k share_k` settles it directly, and **has never been
computed for this ticket** — even though `_relation_community_share`'s own docstring names escaping
exactly this blind spot as the reason reconstruction-space share was adopted over diagonal-sum share.

Raw-data spectra point the same way: `S_Auth_Affil`/`S_Art_Auth`/`M_Atom_Child` have effective ranks
of 19/24/42 with `σ₂/σ₁` of 0.89/0.83/0.89 — **nearly flat spectra with no dominant low-rank
structure**. A rank-6 model fitting a flat spectrum has little to latch onto, which would make *which*
community wins weakly determined — and therefore seed-unstable.

### Three problems with the existing evidence that must be fixed before any of it is built on

An audit of all 10 `ghost_*` scripts and a check of the fit cache turned up the following. These are
not stylistic; each one weakens a conclusion currently written into `CLAUDE.md` §4.22 / `FINDINGS.md` §23.

1. **Four of ten scripts read the raw `np.diag(Z)` while selecting communities via the
   permutation-corrected `Z[k,π(k)]`** (`ghost_suspicious_cells_investigation.py:68`,
   `ghost_auth_collinearity_and_concentration.py:91`, `ghost_article_degree_vs_winner_loading.py:44`,
   `ghost_lambda_z_sensitivity.py:55`). `CLAUDE.md` §4.20 designates the corrected reading as the only
   legitimate `Z_scaled` mass path. Measured impact: the correction actually fires (trusted *and*
   non-identity) in 5/60 v1 and 6/60 v2 cells for `S_Art_Auth`, and up to 23% of cells for
   `S_Auth_Affil` — so roughly 1 cell in 10 in the headline duopoly table read the wrong entries.
2. **The affiliation-removal ablation is confounded by loss reweighting, unmentioned.** Dropping
   `S_Auth_Affil` takes the social relation count 3 → 2, so `w_soc = 0.5/n_soc` goes 0.1667 → 0.25 —
   **`S_Art_Auth`'s weight in `recon_loss` rises 50%**. Its two headline outcomes are "`S_Art_Auth`
   resolution improved" and "collinearity fell." Both are exactly what a 50% weight increase on
   `S_Art_Auth` produces on its own. As written, "removing affil helps C1" and "up-weighting
   `S_Art_Auth` helps C1" are indistinguishable.
3. **Every ghost result is single-seed (`MASTER_SEED=42`) with no noise floor.** The promised "Test 4"
   does not exist on disk. Ticket 82's precedent for a comparable per-community quantity is that the
   noise floor *exceeded the signal* (median 0.144 vs 0.112) — so the flagship claims ("3 cells have
   two simultaneous ghosts", "7/12 cells at K=6") are currently unprotected against ordinary run-to-run
   variation.

Lesser but real: `c_u` collinearity is computed over all rows including ~54% dead `auth` rows in T1
— **checked at Stage 0f below: not contaminated** (max shift 6.0e-8 across 16 facet-cell
measurements, explained by dead-row magnitudes sitting 4-6 orders below live-row values); the
`S_Art_Auth` Herfindahl metric renormalizes the K diagonal entries to sum to 1, reintroducing
precisely the blind spot §17 rejected diagonal-sum share for; `community_share_vector` is
redefined verbatim in six scripts; the flagged-ghost cell indices are hardcoded in two
hand-maintained copies with three v2 entries no script in the set produces.

### Data-provenance finding (user-confirmed as needing action)

`S_Art_Journ` now has **13 non-zero ties in both slices** (`CLAUDE.md` §11 still records 24/35).
**12 of 25 T1 articles and 23 of 36 T2 articles have no journal tie at all.** This is the intended
consequence of the repository-exclusion decision (deferred item 1 / D4) — the user confirms the
intent and expects higher `S_Art_Journ` density at 22k scale. It matters here because `S_Art_Journ`
is one of the three relations whose diagonal spreads across nearly all K, so it carries substantial
weight in the ghost measure while now containing almost no information.

The v1 pickles were regenerated 2026-08-27 02:01; the cache data-hash changed `bd43f14414 →
af5b6c3f77` (T1) and `ac159922af → cf13a41c4b` (T2). **200 cached fits sit under the two pre-fix
hashes.** Per the user's instruction ("if any recent test used data version with repository, we need
to redo such test"), everything dated **2026-08-26 or earlier that used `T1`/`T2`** was computed on
with-repository data:

| Result | What depends on it |
|---|---|
| `domain_balance_seed_noise.json` (08-25) | **`TOL=0.15`** — ticket 82 D3, FINDINGS §21's noise floor |
| `grammar_damping_{tier0,phase3_seed_pairs,phase3_reanalysis,cousin_he_more_seeds}.json` (08-26) | **All of ticket 84's Phase 1/3 verdict** |
| `near_separability_check.json`, `rotation_feasibility_search_v2.json`, `rotation_island_search_v2.json` (08-19…24) | **FINDINGS §18** (rotational indeterminacy, both tiers) |
| `collapse_pen_*`, `z_scaled_*` (08-20/21) | FINDINGS §20 gameability |
| `domain_balance_measurement.json` (08-24), `mass_mem_argmax_decomposition.json` (08-23) | FINDINGS §13 E1 |
| `relation_singular_spectrum.json` (08-19) | FINDINGS §4's spectrum evidence |

All 10 `ghost_*` results (08-29) and the 08-27/08-28 `domain_balance_*` work used **post-fix** data
and are unaffected. `T1_v2`/`T2_v2` were regenerated at the same time and their cache is current.

---

## Plan

Sequenced so that nothing expensive runs before the cheap thing that could invalidate it. Stages 1
and 3 need **zero new fits** (all 120 grid cells are cached under current data hashes).

### Stage 0 — repairs, before any new interpretation

**0a. Centralize the mass measure.** Move `community_share_vector` into `diagnostic_blocks.py` beside
the existing `collapse_mass_share` (line ~1191), and have every `ghost_*` script import it. Six
verbatim copies currently exist and all agree; the risk is structural drift, which
`diagnostic_blocks.py`'s own header (lines 8–17) exists to prevent.

**0b. Fix the raw-diagonal reads.** In the four scripts named above, replace `np.diag(Z)` with the
permutation-corrected `Z[k, π(k)]` obtained from `chunk13v9._hungarian_relabel_relation`, and record
`trusted`/`structure_score` alongside every reported value. Re-run and diff against the stored JSON —
report which conclusions move and which don't, rather than silently regenerating.

**0f. Validate `c_u` against dead-row contamination — DONE, result: not contaminated.**
`community_share`'s dead-row robustness was checked directly (ghost_test2_dead_entity_check.py);
`c_u` (`U_norm_facet.T @ U_norm_facet`, the Gram matrix `ghost_auth_collinearity_and_
concentration.py` uses for the C1-specific auth/affil collinearity claim CLAUDE.md §4.22 rests
on) is a different computation and had never been put through the same test. Scoped to the
specific claim under test rather than the full grid: C1 and C3 (the two configs §4.22
contrasts), K=6 (the K that story was extended to), both slices, both data versions — 8 cells,
16 facet-cell measurements, all from cached fits, no new fitting.
`diagnostic_scripts/stage0f_cu_dead_entity_check.py` → `diagnostic_results/stage0f_cu_dead_entity_check.json`.
**Result: `c_u` is not contaminated.** Zeroing dead rows before the Gram-matrix step moved
`c_u`'s off-diagonal entries by at most 6.0e-8 in every one of the 16 measurements — the same
order as float32 round-off, and the same order `community_share`'s own compensated-case check
found (§4.22: "at most 4.5e-8"). Explained by magnitude, not just observed as small: dead-row
`|U_norm|` values sit 4-6 orders of magnitude below live-row values in every cell (e.g. `auth`,
C1/K6/T1: mean live 1.65e-2 vs mean dead 3.4e-6; `affil`, C1/K6/T1: mean live 4.81e-2 vs mean
dead 1.1e-7). Two of the 8 cells did not converge (C1/T2, C1/T2_v2, C3/T2_v2 — 3 of 8, flagged
per `SESSION_PROTOCOL` §C.4, not excluded) — the robustness result held in the non-converged
cells too, so it isn't an artifact of only checking converged fits.
**Conclusion: the §4.22 C1-vs-C3 auth/affil collinearity claim does not need correcting on
this account.** The "lesser but real" item above is closed for `c_u` specifically; the sibling
Herfindahl-renormalization issue in the same note is unaffected by this check and is handled
separately at Stage 1 (`n_eff` computed on `_relation_community_share`'s reconstruction-space
`share_k`, not the raw or corrected diagonal alone, with `interference` reported alongside so a
mixed relation isn't misread as a clean duopoly).

**0c. Re-run the affiliation ablation without the confound.** Three arms, same seed, same K, same
slice, `PYTHONHASHSEED` fixed (open ticket 85 makes this fit non-reproducible across process launches
otherwise): (i) baseline computed *in-script*, not diffed by hand against another JSON; (ii) affil
removed **with `w_soc` pinned at `0.5/3`** so `S_Art_Auth`'s loss weight is unchanged; (iii) affil
kept but `S_Art_Auth`'s weight multiplied by 1.5 — the control that isolates the confound. If (iii)
reproduces (ii)'s improvement, the affiliation story collapses and `CLAUDE.md` §4.22 needs correcting.
Prefer adding a `soc_keys_override` parameter to `fit_instrumented` over the current fourth hand-copy
of the training loop.

**0d. Correct `CLAUDE.md` §11's nnz table** for `S_Art_Journ` (24→13 T1, 35→13 T2) and note the
articles-without-journal counts.

**0e. Re-run the pre-fix-data results listed in the table above.** Highest priority first, because
downstream decisions rest on them: `domain_balance_seed_noise.py` (TOL=0.15's basis) → ticket 84's
Phase 1/3 → FINDINGS §18's two tiers. Scripts all exist and are re-runnable as-is; only the data
underneath changed. Report each as "re-derived on post-fix data: verdict held / verdict moved."

### Stage 1 — incidence, properly measured (no new fits)

**New script `diagnostic_scripts/relation_mass_decomposition.py`.** For all 120 cached cells
(C1–C6 × K∈{2..6} × T1/T2/T1_v2/T2_v2), per active relation, compute and store:

- the full `share_k` vector exactly as `_relation_community_share` returns it (permutation-corrected);
- `Σ_k share_k` and **`interference = 1 − Σ_k share_k`** — the quantity that decides whether B is real
  concentration or an artifact of reading the diagonal of a mixed relation;
- `n_eff = 1/Σp²` over the normalized within-mass — the duopoly measure, replacing the raw-diagonal
  Herfindahl;
- `argmax_k share_k` (the winner), plus `trusted` and `structure_score`.

Then derive the three things that answer the actual questions:

1. **Duopoly incidence table** — `n_eff` by relation × K × config × slice × data version, against the
   fair value K. Answers "how widespread is B, and in which relations", for all 11 relations rather
   than one.
2. **Winner-overlap matrix** — per cell, how many distinct communities win at least one relation, and
   the relation×community win matrix. This is the missing half of the A↔B link.
3. **Additive attribution** — decompose each community's `community_share` into per-relation
   contributions. For every flagged ghost, this names *which relations* made it a ghost, and tests
   directly whether the aggregate is dominated by `S_Art_Journ` (now 13 ties) and the anchors.

### Stage 2 — is any of it above the noise? (new fits; user chose full scope)

**New script `diagnostic_scripts/ghost_seed_noise_floor.py`** — the missing "Test 4". Reuse
`domain_balance_seed_noise.py`'s protocol verbatim: 5 seeds (`MASTER_SEED` + 4 offsets), realign
non-reference seeds with production's own `s5_dual_track_alignment` Track A, facets ordered via
`get_required_facets()` per §4.9. Grid per the user's choice: **K∈{4,5,6} × C1–C6 × all four
slice-versions × 5 seeds ≈ 360 fits** (~30–60 min single-threaded). Fix `PYTHONHASHSEED` (ticket 85).

Report the across-seed SD of `min_share`, of the full `community_share` vector, and of per-relation
`n_eff`, paired community-by-community against the measured signal — the same presentation FINDINGS
§21 used, so the two are directly comparable. Record non-convergence, do not silently drop it.

**This stage gates everything downstream.** If ghost incidence does not clear its own noise floor, no
mechanism should be built, and Stage 4 reduces to 4a alone.

### Stage 3 — why relations differ so much (no fits)

**Extend the existing `diagnostic_scripts/relation_singular_spectrum.py`** (do not write a new one)
from 3 relations / T1 to **all 11 relations × all 4 slice-versions**, keeping its existing effective-
rank-participation-ratio and spectral-entropy measures. Add, on the live bipartite graph of each
relation: connected components, largest-component fraction, and mean degree per side (piloted already —
T1 `M_Atom_Child` has a 74% giant component and n_eff 1.00; `S_Art_Journ` is 11 near-isolated
components and n_eff 5.3).

Then regress observed `n_eff` (Stage 1) on these fit-independent structural quantities. This separates
**"the relation genuinely contains only ~2 blocks"** (B is a correct reading; not a model defect) from
**"the model failed to find blocks that are there"** (B is a defect worth a mechanism). It also closes
FINDINGS §4's own named-but-never-run test ("singular value spectra of the three social relations have
not been measured") — now at wider scope.

### Stage 4 — candidate solutions, in ascending order of commitment

**4a. Treat it as model selection, not a defect (free, decide first).** Ghost incidence rises cleanly
with K — 0/12 cells at K=2 to 7/12 at K=6 on the `0.5/K` threshold — and T1 at K≤3 is near-fair in
every config. A ghost at K=6 may be the model correctly reporting that a sixth community does not
exist. Test: does ghost-free K agree with the K chosen by hypervolume (§4.6) and by the seed-stability
/ consensus measure §10 already proposes? If they agree, the answer is "pick K by this criterion" —
no new loss term, no new tunable, and it advances §10's model-quality work at the same time.

**4b. Outer-loop min-share floor.** The symmetric counterpart of the existing `max_share`/`collapse_pen`
ceiling: `evaluate_dimensional_collapse` already computes the whole `community_share` vector and
currently discards everything but the max. Port ticket 80's penalty shape verbatim, as it was ported
from `MAX_MONOPOLY`. Threshold must come from Stage 2's noise floor, exactly as `TOL=0.15` came from
§21's — not guessed. Note honestly: this is outer-loop only, so per §4.1 it reports and reshapes
Optuna's Pareto axis but creates no gradient.

**4c. Check whether an in-loop term is even possible (free, folds into Stage 1).** §22's measured
lesson is that a `Z_scaled`-based penalty was gamed 30–40× while the real quantity worsened in 3 of 12
cells, so an in-loop term would have to be built on `U_prob` — a *different* quantity from the ghost
definition. Correlate `community_share` against per-community `U_prob` mass across all 120 cells. If
they don't track each other, an in-loop term cannot target the ghost definition and 4b is the only
honest mechanism; say so rather than building a proxy that optimizes something else.

**4d. Relation re-weighting inside `community_share` (gated on Stages 1+3).** Only if Stage 1 shows
the aggregate is dominated by low-information relations. Any proposal must engage §17's stated reasons
for rejecting nnz-weighting (revives the scaling problem Frobenius normalization removes) and
structure-score-weighting (self-contradictory with its own use as the correction gate). Weighting by
Stage 3's measured structural capacity is the variant those objections don't obviously cover — but this
changes a production measurement, so it needs the evidence first.

---

## Verification

- **Stage 0a/0b:** re-run each corrected script and diff its JSON against the stored version; report
  every conclusion that moves. `verify_agreement()` (`diagnostic_blocks.py:1432`) for any solver-path change.
- **Stage 0c:** arm (iii) is the verification — if up-weighting `S_Art_Auth` alone reproduces the
  ablation's effect, the finding is the confound, not the affiliation.
- **Stage 0e:** each re-run is its own verification; the check is whether the recorded verdict survives.
- **Stage 1:** assert the recomputed `community_share` reproduces `ghost_test3*`'s stored
  `community_share_sorted` for all 120 cells (same function, same cached fits — must match to float
  precision). Assert per-relation `Σ_k share_k + interference == 1`.
- **Stage 2:** confirm the reference seed reproduces the existing single-seed result exactly before
  trusting the other four.
- **Stage 3:** spectrum shape must be identical between raw and Frobenius-normalized matrices (the
  existing script already asserts this — normalization is one scalar).
- Every stage: T1/T2 never pooled; v1/v2 never pooled; non-convergence recorded, not dropped; results
  persisted next to their scripts per `SESSION_PROTOCOL` §D.

## Files

- **Modify:** `chunk13_execution/diagnostic_blocks.py` (host `community_share_vector`; add
  `soc_keys_override` to `fit_instrumented`; bump `BLOCKS_VERSION` — note this invalidates cache keys,
  so bump once, at the start).
- **Modify:** the 4 raw-diagonal scripts; `ghost_no_affil_ablation.py` (rewrite around
  `fit_instrumented`); extend `relation_singular_spectrum.py`.
- **New:** `diagnostic_scripts/relation_mass_decomposition.py`, `diagnostic_scripts/ghost_seed_noise_floor.py`.
- **Docs:** `CLAUDE.md` §11 (nnz table), §4.22 + `FINDINGS.md` §23 (corrections from Stage 0),
  new `FINDINGS.md` section for B; a new ticket 87 row in §8 for the per-relation concentration finding.
- **Unchanged:** `chunk13v9.py`. No production change is proposed until Stage 4 is decided.

## Reuse (do not reimplement)

`_relation_community_share`, `_hungarian_relabel_relation`, `evaluate_dimensional_collapse`
(`chunk13v9.py`); `fit_or_load`, `load_slice`, `presence_masks`, `facet_membership_profile`,
`domain_balance_r_k`, `s5_dual_track_alignment`, `apply_seed_permutation`, `verify_agreement`
(`diagnostic_blocks.py`); `domain_balance_seed_noise.py`'s 5-seed protocol; `relation_singular_spectrum.py`.

## What this plan does NOT claim

- Not that B is a defect — Stage 3 exists precisely because it may be a correct reading of the data.
- Not that ghosts are real above noise — Stage 2 exists to find out, and gates Stage 4.
- Not that removing affiliations helps C1 — Stage 0c tests whether that finding survives its confound.
- Not a fix for anything at 22k scale; every threshold here is toy-corpus calibrated.

---
---

# PRESERVED — deferred items carried over from the previous (ticket 84) plan file

Not part of the plan above. Kept because these are not all recorded in `CLAUDE.md`/`FINDINGS.md`.

1. **`S_Art_Journ` (D4) — RESOLVED**, via OpenAlex venue resolution + repository exclusion rather than
   the originally-proposed idf route (row-keyed idf is inert here; article→journal is many-to-one).
   Implemented in `toy_large.ipynb` cell 9 and `chunk12.py`/`chunk12v2.py`. Of 59 repository-hosted
   articles, 10 resolved to a real venue, 49 confirmed to have none; reproduced identically across two
   runs. Fixed a latent `get_art_idx` first-seen-index bug surfaced by the change. **See this plan's
   Context — the downstream consequence (13 ties, 12/25 and 23/36 articles with no journal) was not
   assessed at the time.**
2. **`M_Cousin_Child` column skew (D6) — INVESTIGATED, decided: leave undamped.** The T2 outlier
   (`(focal_he llms/C/en)`, column-degree 84 vs median 5) is correctly classified and reflects the
   corpus's deliberate focal-word sampling, not a parsing artifact. Paired treatment test (C3/C4,
   K∈{3,4}, both slices, 5 seeds) did not clear 2×SD in either slice; negative control stayed near zero.
3. **Damping strength re-derivation (D3)** — `log2` stays fixed. Blocked on held-out validation
   infrastructure that doesn't exist (`CLAUDE.md` §10 is proposal only). Re-derive at 22k.
4. **Per-slice degree basis (D7)** — design settled, not coded: **volume-relative damping** (rescale
   degree by slice edge-share before `log2(1+d)/d`), rejecting cumulative cross-slice pooling as
   diluting real decline/rise signal. Within-slice rank preservation proven. **Proven to be a
   mathematical no-op with only one slice**, so correctly gated on the T1→T2 temporal extension
   (§10). Open sub-question: what the reference volume `D_ref` should be once 3 slices exist.
5. **v1-vs-v2 overall model-quality comparison** — re-sequenced by user decision: do not start until
   the pipeline's own open tickets (82, `lambda_l1`/ghost communities, "etc.") are resolved first.
   Draft `diagnostic_scripts/grammar_damping_v1_vs_v2_quality.py` exists, written but never run.
6. **`chunk12v2.py` version control** — urgency downgraded; NAS provides hourly (24h) and daily (~28d)
   snapshots plus SyncIQ replication, verified by recovering the file from three snapshot generations.
   If ever pursued: **local git only** — `PhD_CNMF` on GitHub is confirmed **public**. The home-directory
   `.git` (~198k pending items, including a private SSH key at `.ssh/id_github`) stays untouched.
7. **Degree-vs-frequency for grammar relations — leave binary (set-based) as-is.** 91–93% of
   (atom, child) pairs occur exactly once (T1 504/552, T2 833/898); repeats are dominated by the
   degenerate single-word-wrapper case. Flagged for revisit at 22k.
8. **Vocabulary-size (log/sqrt) degree normalization** — distinct from item 4, not gated on the
   temporal extension, but deferred to 22k for a data-availability reason: 300–642-entity facets cannot
   distinguish `log` from `sqrt` from no transform.
