# Escalation batch — semantic adjudication layer, 2026-09-22/23 overnight run

One item across all four batches. **RESOLVED 2026-09-23** — user reviewed E-1, did not recall what "Run 8"/"Part M" referred to, and chose option 2 (soften the citation). Applied to `CLAUDE.md`'s ticket-75 row directly. Kept below for the record.

---

### E-1 — ticket 75's register row cites two untraceable evidence labels

**Reason:** candidate documentation gap, not a code bug — the fix itself (B5-07) is
independently and fully verified against the live code; only two of its three cited
supporting demonstrations don't resolve to anything on disk.

**Claim in conflict:**
- `CLAUDE.md` §8, ticket 75's row: "Demonstrated three times in the diagnostic investigation
  (Run 8 at K=6, Run 11, and Part M's K=6 sweep)."

**What was checked:** grepped `FINDINGS.md`, `diagnostic_scripts/`, and `diagnostic_results/`
exhaustively for "Run 8" and "Part M" under those exact names.
- **"Run 11"** — resolves cleanly. Documented in `FINDINGS.md` §1's Test-A table and §14's
  permutation table as a fully-random-init fit that hit the epoch ceiling (`math_loss`
  ~4.08–4.21). Corroborates the claim.
- **"Run 8"** — does not appear under that name anywhere. `FINDINGS.md`'s own Run-numbering
  has gaps (5, 6, 7, 8, 10, 12, 14 all missing), consistent with a larger private run history
  existing outside the document, of which only a curated subset got tabulated — so this
  plausibly refers to a real but never-written-up run, not a fabrication.
- **"Part M's K=6 sweep"** — does not appear under that name anywhere, and has no analogue in
  `FINDINGS.md`'s own section numbering (§0–§25) or `chunk13v9.py`'s M1–M4 module-naming
  convention. More concerning than "Run 8" — no obvious explanation for where this label came
  from. `FINDINGS.md` §21 does report an epoch-ceiling hit at K≥4 in a wider grid, which is a
  plausible-but-unconfirmed candidate for what "Part M" might be pointing at.

**Resolution applied (2026-09-23):** option 2 — trimmed to name only "Run 11," with a dated bracket note in place explaining what was removed and why (original wording preserved in git history). See `CLAUDE.md` §8's ticket-75 row.

**Options (as originally presented):**
1. **Locate/formalize the missing evidence** — if "Run 8" and "Part M" refer to real
   investigation you remember, add a short note to `FINDINGS.md` naming what they actually are,
   so the citation resolves for a future reader. *(Recommended if you recall what they were.)*
2. **Soften the register row's citation** to name only what's traceable ("Run 11" plus a
   general "and other runs during the diagnostic investigation"), dropping the two specific
   labels that don't resolve.
3. **Leave as-is** — if you're confident the labels are correct and just under-documented
   elsewhere, no action needed; this note stays on record either way.

Either way, **the fix's mechanism itself is not in question** — `objective()` setting
`converged`/`epochs_run` user_attrs, and all three downstream filter points (Scout/Deep-Dive
hypervolume calc, the §S4 archiver, the §S5 stability loop's `RuntimeError` guard), were all
independently re-verified directly against the live code tonight (item B5-07).
