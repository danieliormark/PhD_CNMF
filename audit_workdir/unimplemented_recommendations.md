# Unimplemented recommendations (audit deliverable 1b)

Compiled from `triage.py`'s mechanical "planned/not-implemented" phrase mining (34 raw hits —
`audit_workdir/triage_report.json` → `planned_not_implemented_mentions`) plus the items
already on record in `plans/ticket86-87-ghosts-and-relation-concentration.md`. Every raw hit
was read in its surrounding context and classified as **OPEN** (no code exists, nothing on
record rejecting it — listed below), **HISTORICAL** (true when written, since implemented —
excluded), or **REJECTED** (considered and explicitly decided against — excluded from the open
list, kept in its own section for completeness). This classification step is the one piece of
this deliverable that needed real reading rather than a pattern match; everything else below
traces to a specific citation.

## Genuinely open

### Ticket 82 — domain-balance mechanism, E3 follow-up
1. **Planted-structure test for `max_share`/`collapse_pen`**, the parallel of the test that
   produced E3's negative result for `dev_k` — proposed, not run. Direct next step for deciding
   whether an outer-loop veto on obviously-broken communities is buildable at all, since `dev_k`
   was shown unable to support one. *(CLAUDE.md §4.18 E3; FINDINGS §25)*
2. **Redesigning `sociological_penalty`'s term composition** — effectively `semantic_pen` alone
   plus a term of doubtful signal (per E3) across most of the grid. Candidate directions named,
   none decided: relation-level mass via `Z_scaled`, splitting `semantic_pen`'s Part A/B, the
   max-share reformulation. *(CLAUDE.md §4.17; FINDINGS §15)*

### Tickets 86/87 — ghost communities / relation concentration
3. **No mechanism (in-loop or outer-loop) designed** for low-mass "ghost" communities —
   diagnostic-only throughout, deliberately, matching how tickets 82/84 were sequenced.
   *(CLAUDE.md §4.22, §8 ticket 86; FINDINGS §23)*
4. The full affiliation-removal ablation grid (all 6 configs × K∈{2..6} × both data versions) —
   only a 2-cell pilot exists.
5. A seed-noise floor for `community_share` itself ("Test 4") — needed before any single
   cell's ghost verdict can be trusted against ordinary run-to-run variation, the same
   discipline `TOL` went through for ticket 82.
6. Extending the rotation-feasibility (§18) and permutation-conflict (FINDINGS §14) checks to
   this ticket's wider K/slice/data-version grid.
7. A motif-preserving synthetic-data generator — the planted-null test's ties are degree/
   weight-preserving but not motif-preserving (real co-authorship cliques aren't reproduced).
8. Recovery-on-real-data via seed-stability — multi-seed refits of the real corpus (not
   synthetic) to estimate community-recovery confidence directly.

### Ticket 84 — chunk12 relation-weighting follow-up
9. `S_Art_Journ`'s column-keyed idf weighting — deferred entirely (needs a new
   `dfreq_global['journ']` collection, not currently gathered). *(CLAUDE.md §4.21)*
10. A version mismatch found in `corpus_curated.sqlite` (mtime Jun 15 23:02) during the D4
    follow-up — found, not yet acted on. *(FINDINGS §24)*
11. Whether to re-run preprocessing/postprocessing to recover the 3 articles whose sentences
    failed the core-structure filter — explicitly left open. *(FINDINGS §24)*

### CLAUDE.md §10 — Planned Extensions (both explicitly "proposal only, not scheduled, not built")
12. **Temporal slices T1→T2** — full chronological-loop/inertia-prior design settled, zero
    code written.
13. **Model-quality evaluation for the 22k-article corpus** — held-out link prediction (needs
    relation-aware masking, not blanket k-fold) and seed-stability/cophenetic consensus
    (buildable directly on existing `fit_or_load`, no new fitting code needed) — neither built.

### Smaller, localized items
14. §18: "the next natural step... would be a K=3-focused, C5-focused deeper sweep" — named,
    not run.
15. Whether the narrow 0.054–0.132 `semantic_pen` range across the grid is a real ceiling or
    an artifact of something else — flagged as unknown, not separately investigated.

### Borrowed from literature review (added 2026-09-24, Tang et al. 2025 NJP 27 013007)
16. ℓ2,∞ membership error (+ mean row error) for planted-structure tests. *(CLAUDE.md §10)*
17. Matched cosine similarity between membership matrices, for seed-stability and cross-data
    comparisons. *(CLAUDE.md §10)*
18. Partial adoption of a Dirichlet(α) mixed-membership draw inside the existing
    degree-preserving planted generator. *(CLAUDE.md §10)*

## Explicitly rejected (kept for completeness — not pending action)
- Physical mutation of `U`/`Z` for permutation correction, in favor of read-time correction.
  *(CLAUDE.md §4.20)*
- An R²/explained-variance analog as a model-quality metric. *(CLAUDE.md §10)*
- Raising `TOL` toward the post-fix p90 (~0.17) instead of keeping it at 0.15. *(FINDINGS §21)*

## Refresh check (2026-09-22) — silent-implementation verification

Every item above was checked directly against the current `chunk13v9.py` and
`diagnostic_scripts/` (grep/read, not re-derived from the docs). **Result: all 15 remain
open — none were silently built since the recommending sentence was written.** Evidence per
item, briefly:

- **1, 2, 3, 15**: `chunk13v9.py`'s five `evaluate_*` functions and `sociological_penalty`'s
  4-term construction (line 1521) are byte-identical to what's already on record — no new
  ghost-mechanism function, no `max_share`-targeted planted test (`domain_balance_planted_null.py`
  measures `domain_balance_r_k`/`dev_k` only, confirmed by reading its imports and call site —
  a different quantity from `max_share`/`collapse_pen`), no term redesign, no further
  investigation of the `semantic_pen` range.
- **4**: `ghost_no_affil_ablation.py`'s `CELLS` is still `[("C1", 6, "T1"), ("C3", 6, "T1")]` —
  the same 2-cell pilot.
- **5**: no script matching a `community_share` seed-noise-floor design exists (checked by
  name and by content).
- **6**: `rotation_feasibility_search_v2.py`/`rotation_island_search_v2.py` are still
  `K_LIST = [3, 4]` — not extended to tickets 86/87's wider K∈{2..6} grid.
- **7**: no motif-preserving generator exists (grepped for "motif" across all 73 scripts — zero
  hits).
- **8**: a *related* tool exists (`domain_skew_seed_reproducibility.py`) but it tests whether a
  community's domain-skew *character* reproduces across seeds — not recovery confidence in the
  ticket 86/87 sense. The specific recommendation remains unbuilt; worth knowing the adjacent
  tool exists if this is picked up later.
- **9, 10, 11**: no `dfreq_global['journ']` in `chunk12.py`/`chunk12v2.py`; no resolution of the
  `corpus_curated.sqlite` mtime mismatch found; the 3-articles question is still open per
  `CLAUDE.md` §4.23's own current text.
- **12, 13**: no `U_prior`/temporal term in `chunk13v9.py`; no held-out-link-prediction or
  cophenetic-consensus script anywhere in `diagnostic_scripts/`.
- **14**: no K=3/C5-focused follow-up sweep found.

## Notes on method, and what this deliverable does NOT yet do
- A few FINDINGS.md hits (e.g. line 839's "the mechanism, as designed, not yet implemented")
  were excluded as HISTORICAL — that specific passage describes the pre-E2 design, since
  superseded by ticket 82 E2's actual implementation — but this call was made from limited
  surrounding context (the mechanical mining tool prints one line, not the full section), not
  a full re-read of FINDINGS.md. Treat items 1-15 above as reasonably solid; treat the
  exclusions as lower-confidence and worth a second look if precision matters for a specific one.
- **Silent-implementation check: done** (see the Refresh check section above, 2026-09-22) — all
  15 confirmed still open against current code. Re-run this check again if picking this list up
  much later, since it's only current as of that date.
