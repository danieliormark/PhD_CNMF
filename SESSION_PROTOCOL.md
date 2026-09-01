# SESSION_PROTOCOL.md — Standing conditions for chunk13v9 work

> Referenced by prompts as "standard diagnostic conditions" or "standard patch
> conditions". Read once per session; do not re-state these in each prompt.
>
> Companion to CLAUDE.md (architecture, contracts, defect register) and
> FINDINGS.md (analytical results, refuted hypotheses, open questions).

---

## A. Session types

Every prompt declares one:

- **DIAGNOSTIC** — measure only. `chunk13v9.py` is not modified. Scratch scripts
  only. Findings inform a decision the user has not yet made.
- **PATCH** — `chunk13v9.py` may be modified, but only as specified. Nothing
  outside the stated scope.
- **PATCH + VERIFY** — patch, then measure the effect of the patch.

If a prompt does not declare a type, ask before starting.

---

## B. Standard run settings

Unless a prompt overrides them:

```
config           C6 (maximal topology, exercises the most relations)
K                4
slice            T1 (Star_extended_matrices_t1.pkl)
lr               0.01
positivity       clamping (NOT softplus — see FINDINGS §6 / CLAUDE §4.8)
epoch ceiling    2000, with the existing relative early stop
seed             MASTER_SEED
threading        torch.set_num_threads(1)
```

**`lambda_l1` and `lambda_z_offdiag` have no default.** Both are under active
research (§G) — a documented default would quietly become the de facto setting
for decisions nobody made deliberately, which is how `lambda_l1 = 0.01` ended
up unremarked in the gauge-invariance runs. **If a prompt calls for a fit and
does not state both lambdas, stop and ask rather than assume.**

Data paths:
```
T1  /mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1.pkl
T2  /mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t2.pkl
decoders  .../Star_epistemic_decoders_global.pkl   (contains idf_global; nothing in 13v9 loads it)
```

---

## C. Reporting rules

1. **Per facet and per relation. Never aggregate across them.** Aggregation has
   twice hidden the finding — the `journ` collinearity was obscured by a mean,
   and per-relation permutation structure was invisible in per-run summaries.

2. **Distributions, not point estimates**, wherever a distribution is
   meaningful: min / p10 / p25 / p50 / p75 / p90 / max / mean.

3. **Full matrices when a matrix is the object of study.** Print the K×K, don't
   summarise it.

4. **Flag any run that hits the epoch ceiling.** Do not report its numbers as
   converged. State its `math_loss` so comparability can be judged.

5. **Live vs all entities.** Any statistic derived from `U_prob` or `U_norm`
   rows must be reported over live entities (via `build_presence_masks()`),
   and separately over all entities where the comparison is informative.
   ~40–60% of entities per facet are structurally dead — see FINDINGS §5.

6. **State assumptions explicitly before building on them.** If a prompt asserts
   that two runs are comparable, or that a quantity is invariant, verify it and
   say so. A mis-stated premise has cost a full test cycle at least once
   (FINDINGS §8).

7. **Do not resolve ambiguity by adding runs.** Report what was measured and let
   the user decide.

8. **If K and configuration disagree on a conclusion, that is the finding.**
   Report both rather than reconciling.

9. **Scope-adjacent discoveries: report, don't chase.** A session diagnosing one
   question sometimes surfaces something in scope-adjacent territory it wasn't
   looking for — the permutation finding emerged this way, from an audit aimed
   at something else. When that happens: report it clearly, do not act on it,
   and do not expand the session's scope to pursue it further. This is implied
   by rule 7 for the specific case of ambiguity, but applies more generally —
   an unplanned finding is not licence to start a new investigation inside the
   current one. Surface it, let the user decide whether it becomes the next
   session's subject.

10. **Define new terms/entities at first use; mind Occam's razor.** Any process
    noun introduced into a plan or explanation that isn't already standard in
    this codebase or the literature (e.g. "gate", "damping" as informal slang,
    a named phase or stage) must be defined in one plain sentence the first
    time it's used — not left for the reader to infer from context. Before
    proposing a new mechanism, parameter, or pipeline stage, check whether an
    existing one already does the job (ticket 84's design pass found this
    twice: the §9 justification and the `S_Art_Journ` "low risk" claim both
    turned out to name work an existing mechanism already did, or didn't do,
    contrary to the initial framing). Prefer the smaller addition — reusing an
    existing formula/function over inventing a new one, a config flag over a
    new file, no new tunable over a tuned one — unless the simpler option is
    checked and found genuinely insufficient, not just assumed to be.

11. **Plain, precise language over metaphor or slang — extends rule 10 from
    naming to prose generally.** Explain a mechanism by saying exactly what it
    does, with the actual numbers, rather than reaching for a figurative label.
    Concrete precedent: "resolving-power ceiling" was used to describe why
    `S_Art_Auth`'s diagonal concentrates on few communities, flagged as
    obscure, and corrected to a literal description — what the diagonal
    values mean, what "resolving" a community concretely means, tied to the
    actual tie counts (218–258) driving it. When a plain-language explanation
    and a named-concept explanation would convey the same thing, use the
    plain one; do not introduce a new label for something a sentence can
    already say directly.

---

## D. Persistence

- Numeric results go to `diagnostic_results/` under `chunk13_execution` — **not**
  the scratchpad. The scratchpad has reset mid-session more than once, losing
  scripts and results.
- **Save `U_final`, `Z_final`, `U_scales`, and `diagnostics` for every run,
  unconditionally** — not just runs judged "likely to be re-examined." That
  judgement isn't reliably knowable in advance: Run 3 and Run 11 were discarded
  as unlikely to matter and became necessary two prompts later. A few MB
  against a full re-fit is not a close trade.
- **Every results file carries its own full run configuration.** Alongside
  whatever the file's specific numbers are, record: config, K, slice, lr,
  `lambda_l1`, `lambda_z_offdiag`, seed, epochs run, and the `converged` flag.
  This applies retrospectively as a going-forward practice, not just to new
  finding types — a results file re-read months later must be self-describing;
  it should never depend on the prompt history to know what it was measuring.
- Name files after the test, e.g. `permutation_test.json`.
- **Scripts persist alongside their results, in `chunk13_execution/diagnostic_scripts/`**,
  name-matched to their output (`partB_term_variance.py` ↔ `partB_term_variance.json`).
  Rule of thumb: **if a script writes to `diagnostic_results/`, the script is
  itself a result.** A results file records *what* was measured (its
  `_run_config` block); only the script records *how* — which normalisation,
  which masking, which of several defensible measures. Lose the script and the
  JSON's numbers stop being interpretable. The scratchpad is for genuinely
  throwaway probes only: it reset again on 2026-08-15 and took
  `threeway_mass_check.py`, `step1_domain_balance.py`, and
  `uscales_normalization_check.py` with it, while their result files survived.
  Rewriting a lost script is not merely expensive — a reconstruction that
  differs subtly produces numbers that silently aren't comparable to the old
  ones.
- **Every `.py` file carries a 2–3 line header annotation**: `# WHAT:` (one
  sentence on the question it answers) and `# OUT:` (the results file it writes,
  plus any setting that dates it — e.g. `permutation_test.py` ran at
  `lambda_l1=0.01`, before ticket 78 fixed it at 0.0). This is what lets a later
  session — or a fresh chat with no context — tell at a glance which file it
  needs, without opening and reading each one.
- **Fits are cached.** `diagnostic_blocks.fit_or_load()` writes every fit to
  `diagnostic_results/tensors/`, keyed on a hash of everything that can change
  the numbers — config, K, slice, both lambdas, seed, init mode, epoch ceiling,
  lr, `BLOCKS_VERSION`, and a content hash of `chunk13v9.py` itself. Editing the
  solver therefore invalidates every cached fit automatically; a stale fit can
  never be silently returned against changed code. Prefer `fit_or_load()` to
  re-fitting: it makes re-analysing the *same* fits under a different measure
  cheap, which is what keeps competing measures comparable to each other.

---

## E. Verification expectations

- `py_compile` after every edit in a PATCH session.
- Any reimplementation of the training loop in a scratch script must be checked
  against real `run_inner_solver` on an identical config — report the agreement
  in epochs and `math_loss`. **`diagnostic_blocks.fit_instrumented()` discharges
  this once for all scripts that import it** (`python diagnostic_blocks.py
  --verify`, recorded in `diagnostic_results/blocks_agreement_check.json`; last
  run bit-identical on C6/K=4). Re-run it after any edit to that function or to
  `run_inner_solver`. A script that hand-rolls its own loop instead still owes
  the check itself.
- Any fast path (trace identities, masked computations) must be checked against
  a dense or hand-computed equivalent on at least one case — report the
  agreement.
- **Read a script's entire file for side effects before executing any part of
  it — not just the section intended to use.** A module-level operation can
  run before, or independent of, a `# EXECUTION`-style marker or an
  `if __name__ == "__main__":` guard; slicing a file at a comment boundary and
  assuming everything above it is side-effect-free is not verification.
  Concrete precedent: `toy_large/postprocessing_4h.py` and
  `toy_large/7.5postprocessing_4hbased_correct.py` (upstream of `chunk12.py`)
  both purge `corpus_curated.sqlite` as a module-level side effect, before
  their own `# EXECUTION` marker — executing only the portion believed to be
  "just function definitions" destroyed the curated database (2026-09-01,
  recovered from a NAS snapshot, no data lost). Do not execute any part of
  either file again without first reading it in full; the same caution
  applies to any script not already known to be side-effect-free.

---

## F. Determinism

- `torch.set_num_threads(1)` makes **same-process** repeats bit-identical.
- **Cross-process** reproducibility is approximate: same seed, separate
  interpreter launches have given 925 vs 976 epochs with `recon_loss` differing
  by ~3e-4. Trajectories are not diverging — the early-stopping trigger fires at
  slightly different points. Report such differences; do not treat them as
  failures.

---

## G. Out of scope unless a prompt explicitly opens them

Do not modify, and do not "helpfully" adjust:

- `lambda_z_offdiag`, its range, or the Z off-diagonal penalty
- `lambda_l1`, its range, or the sparsity term
- Z structure (diagonal constraint, initialisation)
- Module 3 §3's coherence check
- Trial counts (`SCOUT_TRIALS`, `DEEP_DIVE_TRIALS`, `K_LIST`, `CONFIG_IDS`)
- The 50/50 domain alpha weighting
- Anything in CLAUDE.md §4 (locked design decisions)

Each of these is under active discussion. Changing one silently invalidates
measurements taken against it. See C.9 for what to do when a session surfaces
something in this territory without being asked to.

---

## H. Documentation

- **FINDINGS.md and CLAUDE.md are written to only on an explicit, separate
  instruction** — never implied by a request for a report, an audit, a summary,
  or by having produced a finding worth recording. A report is chat output by
  default, full stop. When writing to either file *is* requested, the specific
  section(s) to write should be named. These two files are the source of truth
  for future sessions and for a fresh chat with no other context; anything
  drifting into them unreviewed propagates silently from then on.
- When writing is requested: close or update tickets in CLAUDE.md's register
  for anything fixed; add analytical findings to FINDINGS.md, including
  **refuted** hypotheses — §11 exists so that failed ideas are not retried.
- If a finding contradicts something already recorded, say so explicitly rather
  than adding a second, conflicting entry.
