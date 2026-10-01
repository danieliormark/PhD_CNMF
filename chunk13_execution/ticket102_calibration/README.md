# Ticket 102 calibration (2026-10-02), record only

Scripts behind FINDINGS §31. They ran from the session scratchpad (paths inside
point there) against a scratch copy of chunk13v9.py built by `make_conc.py`;
the term is now in chunk13v9.py itself (v9.4), so these are not needed to run
the pipeline. Fit outputs were not kept.

1. `make_conc.py` builds the scratch solver (term + optional two-part stopping test).
2. `grid.py main|warm|dual` runs the fits (main grid, warm-up check, corrected stopping).
3. `analyze.py <stage>` computes per-fit graph measures and seed-pair stability.
4. `aggregate.py` pairs every setting with the same-seed baseline.
5. `text_check.py` runs the independent abstract-text check.

Note (ticket 104): these scripts import numpy before loading the solver, so their
fits are not bit-identical to pipeline fits.
