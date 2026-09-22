# Audit status — CLAUDE.md / FINDINGS.md vs. chunk13v9.py / diagnostic infrastructure

**Read this file first if resuming this work in a new session.** It exists so a session with
no memory of the conversation that produced it can pick this up without re-deriving anything
below. After reading it, re-run `python3 audit_workdir/triage.py` — it is deterministic and
cheap (no LLM calls); re-running it regenerates `triage_report.json`/`.md` from the live
files, so those two outputs are always reproducible and never need to be trusted as a stale
snapshot.

## Why this exists

The user asked for an audit: does information in `CLAUDE.md`, `chunk13v9.py`, the diagnostic
infrastructure (`diagnostic_blocks.py` + `diagnostic_scripts/`), and `FINDINGS.md` match each
other; harmonize what doesn't; list recommendations that were decided but never built (1b);
and separately, compress `CLAUDE.md`/`FINDINGS.md` where they overlap, since they're external
memory Claude has to load every session.

**A first attempt at this (in an earlier session) ran out of the session's token budget.**
Root cause, confirmed by the numbers: a 12-agent parallel extraction fan-out (each agent doing
full-file reads + exhaustive claim extraction) burned ~1.64M tokens on Stage A alone, covering
only 3 of 31 tickets, with the most expensive stage (2 Opus-high adjudication agents over the
full ledger) not yet started. **Root cause is structural, not just "too many agents were
launched": parallel subagents don't share context or cache — each pays its own full cost of
loading and reasoning about the same files, so fan-out multiplies total token spend rather
than amortizing it.**

## Redesign, agreed with the user (2026-09-22)

1. **Script-first triage, before any LLM claim-extraction/adjudication.** Push everything
   pattern-matchable (symbol existence, citation-graph resolution, numeric-constant
   cross-checks, raw-diagonal usage, import structure, "not yet implemented" status language)
   into deterministic Python (`ast` + `regex`), at zero LLM cost. Reserve an LLM call only for
   the small subset a script flags as genuinely ambiguous.
2. **Model tiering for whatever LLM work follows:** Sonnet pinned explicitly (never rely on
   model inheritance — that ambiguity was itself a contributor to the last session's cost) for
   everything short of final cross-source judgment; Opus at **medium** effort (not high)
   reserved for that final adjudication only, and only on a batch the user has specifically
   flagged as worth it.
3. **`chunk13v9.py`/`diagnostic_blocks.py` mirrored into the `PhD_CNMF` GitHub repo**
   (`chunk13_execution/`, commit `ab4bbfe`), with a re-sync convention added to
   `SESSION_PROTOCOL.md` §D, so a session or agent without the `tensor_data_staging` mount can
   still read current production code.
4. **Ownership clarified (user, 2026-09-22): `diagnostic_blocks.py`/`diagnostic_scripts/` are
   not production code the docs need to describe accurately — they're Claude's own analysis
   tooling.** Decisions about them (consolidate, delete, keep for future robustness checks) are
   Claude's to make directly, not something to write up as a doc discrepancy for the user to
   adjudicate.

## What's built and validated (`audit_workdir/triage.py`)

Runs in seconds, zero LLM tokens, full scope (all 31 tickets, all 73 scripts, both docs — no
pilot/batch scoping needed since nothing here costs tokens). Two rounds of tightening already
applied after inspecting raw output for false positives (see script comments for the specific
fixes: Module-internal `§X.Y` code labels vs. real markdown headings; numeric proximity vs.
real assignment; comment lines and matrix-construction idioms in the raw-diagonal check).

**Clean / high-confidence results:**
- Citation graph: **0 broken** out of 324 real `§N` cross-references, either direction.
- Numeric constants: **0 mismatches** between `chunk13v9.py`'s actual values and either doc's
  citations.

**New finding an earlier (expensive) LLM pass missed:** `chunk13v9.py` has **4** duplicate
top-level definitions, not 2 — `DEVICE` (15/1686), `MASTER_SEED` (90/1691), in addition to the
already-known `TARGET_COHERENCE` (70/1696) and `ENTROPY_THRESHOLD` (44/1695). All four are
harmless (identical value both times) but violate the file's own §7 "delete the later copy"
rule. **Not yet decided what to do about this** — candidate for either a doc note (§8 register)
or an actual code cleanup; the user has not been asked yet.

**Small, bounded lists, not yet escalated to any model:**
- 8 raw-diagonal script sites (`triage_report.json` → `escalate_ambiguous_raw_diagonal`) —
  mostly already independently verified by the earlier expensive LLM pass (including the real
  `ghost_no_affil_ablation.py` gap); worth a final confirmation pass, not a fresh investigation.
- 8 scripts importing nothing from `diagnostic_blocks.py` (`scripts_not_importing_diagnostic_blocks`)
  — 3 already explained (legacy, pre-date `diagnostic_blocks.py`: `permutation_test.py`,
  `uscales_determinacy.py`, `domain_balance_v1_v2_noise_compare.py` is pure JSON aggregation by
  design); 5 unverified (`journal_repo_resolution.py`/`_v2`/`_v3`, `partB_term_variance.py`,
  `permutation_driver_analysis.py`).
- 64 of 73 scripts missing the `SESSION_PROTOCOL` `# WHAT:`/`# OUT:` header — compliance debt,
  not urgent.
- 1 script (`grammar_damping_v1_vs_v2_quality.py`) with no result JSON — matches the ticket
  86-87 plan file's own note that it was written but never run.
- **34 "planned/not-implemented" status-language hits** (`planned_not_implemented_mentions`,
  19 in CLAUDE.md, 15 in FINDINGS.md) — the raw material for the audit's 1b deliverable. Mostly
  self-evident status statements (e.g. "Status: proposal only, not scheduled, not built"); a
  few are already `[HISTORICAL]`-superseded and shouldn't be double-counted as open.

**Explicitly NOT good enough yet, flagged rather than presented as a finding:** the
"unresolved backtick-quoted symbol" check (`unresolved_backtick_symbols`) returns 210 items,
dominated by false positives (local variable names never captured as top-level symbols; names
correctly belonging to `chunk12.py`/`toy_large.ipynb`/old `chunk13v3.py`/`chunk13v4.py`, none
of which this script scans). **User declined to invest in sharpening this further for now**
(2026-09-22) — the cost/benefit didn't clearly favor it given the other checks already
surfaced the real findings. Revisit only if a "does the doc reference anything that doesn't
exist anywhere" guarantee is specifically wanted later.

## Pending decisions / next actions (none started yet)

1. **Diagnostic-script cleanup** (delegated to Claude per point 4 above, not yet executed):
   - `ghost_test3_share_distribution.py` and `ghost_test3v2_share_distribution.py` still inline
     the old per-community-share loop instead of importing `community_share_vector` from
     `diagnostic_blocks.py`, despite that function's own docstring claiming the migration from
     all 7 original copies is complete. Fix: switch both to import it, verify numerically
     identical output on at least one cached fit before treating the fix as safe (mirrors the
     "confirmed identical before this move" verification the other 5 already got).
   - Review the 5 unverified non-importing scripts above; decide keep-as-is (legacy/deliberate)
     vs. delete (redundant with something centralized) vs. migrate.
2. **The semantic adjudication layer** (real LLM cost, not yet scoped or started): the
   mechanical triage above catches structural/pattern-level drift, not "does this specific
   claim in §4.18/§25 actually match what the planted-null test found." The 2-stage
   ledger-then-adjudicate design from the crashed session is retired per the redesign above —
   any future pass here should be a small number of direct "read the section + the code it
   describes + the result JSON it cites, report mismatches" calls, Sonnet-pinned, escalating to
   Opus-medium only for genuinely cross-source judgment calls. **Not yet scoped — needs a
   decision on which section(s) to start with and whether to proceed now or later.**
3. **The 4 duplicate top-level constants** — decide whether to note in CLAUDE.md §8 or actually
   delete the redundant later copies in `chunk13v9.py` (the user's call, since this is
   production code, not diagnostic tooling).

## Files in this directory

- `triage.py` — the mechanical triage tool. Re-run any time; it's idempotent and reads live
  files, never a cached/stale copy.
- `triage_report.json` / `triage_report.md` — its output. Regenerate rather than hand-edit.
- `AUDIT_STATUS.md` — this file. Update it whenever a pending item above gets resolved or a new
  one is opened, so it stays the single source of truth for "where this stands."
- `merge_ledger.py`, `extract_results_index.py`, `ledger/` — **artifacts of the retired
  two-stage design from the crashed session.** Kept for reference (the `results_index.json`
  extraction script in particular is still a reasonable pattern if the semantic-adjudication
  layer ever needs it), but not part of the current plan. Do not build on `ledger/`'s claim
  files without re-reading this section first — they reflect the abandoned pilot scope
  (tickets 82/86/87 only), not the full-scope mechanical results above.
