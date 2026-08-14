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

---

## D. Persistence

- Numeric results go to `diagnostic_results/` under `chunk13_execution` — **not**
  the scratchpad. The scratchpad has reset mid-session more than once, losing
  scripts and results.
- Save converged tensors (`.npz`) for any run likely to be re-examined.
- Name files after the test, e.g. `permutation_test.json`.

---

## E. Verification expectations

- `py_compile` after every edit in a PATCH session.
- Any reimplementation of the training loop in a scratch script must be checked
  against real `run_inner_solver` on an identical config — report the agreement
  in epochs and `math_loss`.
- Any fast path (trace identities, masked computations) must be checked against
  a dense or hand-computed equivalent on at least one case — report the
  agreement.

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
measurements taken against it.

---

## H. Documentation

- Close or update tickets in CLAUDE.md's register for anything fixed.
- Add analytical findings to FINDINGS.md, including **refuted** hypotheses —
  §11 exists so that failed ideas are not retried.
- If a finding contradicts something already recorded, say so explicitly rather
  than adding a second, conflicting entry.
- Ask before writing to either file if the session was DIAGNOSTIC; the user may
  want to review first.
