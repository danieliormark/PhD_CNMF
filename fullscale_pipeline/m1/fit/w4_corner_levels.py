# W4 reporting extra (RL-415): the four corner levels (term off/on × swap p 0/1) of the swap reading per cell and entity,
# paired by seed; plus excess ÷ all-pairs mean co-membership (scale-free) and the W-aware data link. Reads saved fits only.
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.w4_corner_levels fit/results/stage0/w4_corner_levels.json
import json, math, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, ".")
from fit import w4_swap_analysis as W
from fit.swap_link_measure import one
from fit.plan import W4_CELLS
fits = W.collect(W.ROOT)
jobs = []
for ap_, K, r in W4_CELLS:
    keys = {(t, p): (ap_, K, r if t == "on" else 0.0, p) for t in ("on", "off") for p in (0.0, 1.0)}
    seeds = sorted(set.intersection(*[set(fits[k]) for k in keys.values()]))
    for (t, p), k in keys.items():
        for s in seeds:
            jobs.append(((ap_, K, r, t, p, s), fits[k][s]))
def run(j):
    key, f = j
    return key, {e: (v["model_excess"], v["data_excess"], v["model_all"], v["data_w_excess"], v["data_all"], v["pairs"]) for e, v in one(f, {}).items()}
with Pool(12) as P:
    res = dict(P.map(run, jobs))
out = {}
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))))
for ap_, K, r in W4_CELLS:
    seeds = sorted({k[5] for k in res if k[:3] == (ap_, K, r)})
    c = {}
    for e in res[(ap_, K, r, "on", 0.0, seeds[0])]:
        g = lambda t, p, i=0: np.array([res[(ap_, K, r, t, p, s)][e][i] for s in seeds])
        c[e] = {"off_p0": ms(g("off", 0.0)), "off_p1": ms(g("off", 1.0)), "on_p0": ms(g("on", 0.0)), "on_p1": ms(g("on", 1.0)),
                "on_minus_off_p0": ms(g("on", 0.0) - g("off", 0.0)), "on_minus_off_p1": ms(g("on", 1.0) - g("off", 1.0)),
                "survives_off": float(g("off", 1.0).mean() / g("off", 0.0).mean()), "survives_on": float(g("on", 1.0).mean() / g("on", 0.0).mean()),
                "data_p0": ms(g("on", 0.0, 1)), "data_p1": ms(g("on", 1.0, 1)),
                "model_all": {f"{t}_p{int(p)}": ms(g(t, p, 2)) for t in ("off", "on") for p in (0.0, 1.0)},
                "ratio": {f"{t}_p{int(p)}": ms(g(t, p) / g(t, p, 2)) for t in ("off", "on") for p in (0.0, 1.0)},
                "ratio_on_minus_off_p1": ms(g("on", 1.0) / g("on", 1.0, 2) - g("off", 1.0) / g("off", 1.0, 2)),
                "ratio_on_minus_off_p0": ms(g("on", 0.0) / g("on", 0.0, 2) - g("off", 0.0) / g("off", 0.0, 2)),
                "dataW_p0": ms(g("on", 0.0, 3)), "dataW_p1": ms(g("on", 1.0, 3)), "data_all": ms(g("on", 0.0, 4)), "pairs": ms(g("on", 0.0, 5))}
    out[f"A{ap_} K{K} r{r:g}"] = c
json.dump(out, open(sys.argv[1], "w"), indent=1)
for cell, c in out.items():
    for e, v in c.items():
        f = lambda x: f"{x[0]:+.4f}±{x[1]:.4f}"
        print(f"{cell:14s} {e:26s} off p0 {f(v['off_p0'])} p1 {f(v['off_p1'])} | on p0 {f(v['on_p0'])} p1 {f(v['on_p1'])} | "
              f"all off/on p1 {v['model_all']['off_p1'][0]:.4f}/{v['model_all']['on_p1'][0]:.4f} ratio off p0 {v['ratio']['off_p0'][0]:.3f} p1 {v['ratio']['off_p1'][0]:.3f} on p0 {v['ratio']['on_p0'][0]:.3f} p1 {v['ratio']['on_p1'][0]:.3f} r_on−off p1 {f(v['ratio_on_minus_off_p1'])} | dataW p0 {v['dataW_p0'][0]:+.4f} p1 {v['dataW_p1'][0]:+.4f} all {v['data_all'][0]:.4f} pairs {v['pairs'][0]:.0f} | "
              f"on−off p0 {f(v['on_minus_off_p0'])} p1 {f(v['on_minus_off_p1'])} | survives off {v['survives_off']:.2f} on {v['survives_on']:.2f} | data p0 {v['data_p0'][0]:+.4f} p1 {v['data_p1'][0]:+.4f}")
