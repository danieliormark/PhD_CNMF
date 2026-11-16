# Floor diagnostic on pass-1 fits at K* (hold-out group 1): share of child hyperedges with ≥ 2 articles whose own spread is below the floor minimum γ(between + ω within), γ = 0.8, ω ∈ {0, 1}; mean shortfall (bits); share with largest membership > 0.9. Run from fullscale_pipeline/m1.
import sys, glob, os, json, numpy as np
sys.path.insert(0, ".")
from fit import data as D, measures as M
from fit.aggregate import load_fit, load_UZ, latest_fits
b = D.Build(D.BUILD); KS = {2: 20, 1: 13}
fits = [f for f in latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob("/mnt/hum01-rds/Basov/p91688di/m1_fits/stage0_pass1/*/measures.json")])
        if f["job"]["group"] == 0 and f["job"]["K"] == KS[f["job"]["beta"]] and not f["job"].get("terms") and f["result"]["converged"]]
cache = {}
res = {}
for f in fits:
    a, be = f["job"]["approach"], f["job"]["beta"]
    if a not in cache:
        cache[a] = next(rd for rd in D.fit_data(b, a, group=0) if rd.name == "child_article")
    U, Z = load_UZ(f["dir"])
    wm = M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()})
    fd = M.floor_diagnostics(cache[a], wm)
    res.setdefault((a, be), []).append([ (fd["shortfall_omega0"] > 0).mean(), (fd["shortfall_omega1"] > 0).mean(),
        fd["shortfall_omega0"].mean(), fd["shortfall_omega1"].mean(), fd["share_largest_above"]["0.9"], np.median(fd["between"]), np.median(fd["own_spread"]), K := U["child"].shape[1]])
for k, v in sorted(res.items()):
    x = np.array(v); print(k, "n", len(x), "below min (ω=0) %.3f (ω=1) %.3f | mean shortfall bits ω0 %.3f ω1 %.3f | >0.9 %.3f | median between %.2f own %.2f | log2K %.2f" % (*x[:, :7].mean(0), np.log2(x[0, 7])))
