# Tests of the bare graded swap on the real trial data (ticket 107, 2026-11-16). Plain script: python -m fit.tests.test_swap_bare OUT
import sys, json, os
import numpy as np
from fit import data as D, worlds as Wd
from fit.run_fit import run_one
fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); fails.append(name) if not ok else None
build = D.Build(D.BUILD)
for a in (1, 2):
    rds = D.fit_data(build, a, group=0)
    soc = [rd for rd in rds if rd.domain == "social" and "article" in (rd.row_facet, rd.col_facet)]
    sels = {}
    for p in (0.25, 0.5, 0.75, 1.0):
        sw, perm = Wd.graded_swap(rds, p, seed=3)
        sels[p] = perm != np.arange(len(perm))
        check(f"a{a} p={p}: semantic relations untouched (same objects)", all(x is y for x, y in zip(rds, sw) if x.domain == "semantic"))
        check(f"a{a} p={p}: perm is a permutation", np.array_equal(np.sort(perm), np.arange(len(perm))))
        ok = True
        for x, y in zip(rds, sw):
            if x.domain != "social": continue
            for side in ("row", "col"):
                ok &= np.array_equal(np.sort(np.bincount(getattr(x, f"tie_{side}"))), np.sort(np.bincount(getattr(y, f"tie_{side}"))))
            ok &= len(x.hid_tie_row) == len(y.hid_tie_row) and len(x.hid_empty_row) == len(y.hid_empty_row)
            ok &= np.array_equal(x.tie_val, y.tie_val) and np.isclose(x.fro, y.fro)
        check(f"a{a} p={p}: social relations keep degree sequences, hidden-cell counts, values, norm", ok)
        # each moved article lands on an article of the same social-degree band
        n = len(perm); deg = np.zeros(n)
        for rd in soc:
            for side, idx in (("row", rd.tie_row), ("col", rd.tie_col)):
                if (rd.row_facet if side == "row" else rd.col_facet) == "article":
                    deg += np.bincount(idx, minlength=n)
        band = np.searchsorted(np.array(Wd.SOCIAL_BANDS), deg, side="right") - 1
        check(f"a{a} p={p}: swaps stay within social-degree bands", np.array_equal(band[perm], band))
        # an article's social ties now sit at its new index: author×article / article×journal etc. moved consistently
        ok = True
        for x, y in zip(rds, sw):
            if x not in soc: continue
            if x.row_facet == "article": ok &= np.array_equal(perm[x.tie_row], y.tie_row)
            if x.col_facet == "article": ok &= np.array_equal(perm[x.tie_col], y.tie_col)
        check(f"a{a} p={p}: every social relation uses the same permutation", ok)
        print(f"     a{a} p={p}: share of articles moved {sels[p].mean():.3f}")
    check(f"a{a}: moved share rises with p", all(sels[p].mean() <= sels[q].mean() for p, q in ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0))))
# end-to-end: tiny fit through run_fit's hook
out = sys.argv[1]
job = {"K": 3, "approach": 1, "arm": {"swap_p": 0.5}, "beta": 2, "dtype": "float32", "frac": 0.03, "group": 0,
       "lr_mode": "recompute", "max_epochs": 30, "seed": 1, "stage": "swap_test", "start": "nndsvd_sub0.8", "terms": {},
       "warmup": 10, "tol": 1.0}
jid, st = run_one(job, out)
d = os.path.join(out, "swap_test", jid)
spec = json.load(open(os.path.join(d, "spec.json")))
check("run_fit: swap recorded in spec and permutation saved", spec["swap"]["p"] == 0.5 and os.path.exists(os.path.join(d, "swap_perm.npy")))
check("run_fit: measures written", os.path.exists(os.path.join(d, "measures.json")))
_, perm = Wd.graded_swap(D.fit_data(build, 1, group=0), 0.5, seed=1)
check("run_fit: saved permutation = graded_swap(seed = fit seed)", np.array_equal(np.load(os.path.join(d, "swap_perm.npy")), perm))
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
