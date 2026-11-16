# Follow-up to the binary test (RL-240; owner 2026-11-16 asked how confident "KL effectively ignores the tie values" is).
# Per loss, hold-out group 1, seeds 1–3: (a) membership agreement (Track A over all entities and Track B, chance-adjusted)
# between the binary and weighted fit of the same seed, against weighted–weighted and binary–binary pairs of different
# seeds; (b) training value: partial Spearman of x̂ with the real tie values at VISIBLE ties (child×article, parent×child),
# controlling for the two entities' visible tie counts and their product, for weighted and binary fits; (c) the spread of
# the visible tie values themselves (coefficient of variation; share of ties at the relation's most common value).
import glob, json, os, itertools, numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits
ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"; KS = {2: 20, 1: 13}
rng = np.random.default_rng(0)
build = D.Build(D.BUILD)
rds = {rd.name: rd for rd in D.fit_data(build, 1, group=0)}
for name in ("child_article", "parent_child", "author_article", "author_affiliation"):
    v = rds[name].tie_val; u, c = np.unique(np.round(v / v.mean(), 6), return_counts=True)
    print("values", name, "ties", len(v), "CV %.3f" % (v.std() / v.mean()), "share at most common value %.3f" % (c.max() / len(v)), "distinct", len(u))
def fits(stage, beta):
    out = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, stage, "*", "measures.json"))]
    out = [f for f in latest_fits(out) if f["job"]["approach"] == 1 and f["job"]["beta"] == beta and f["job"]["K"] == KS[beta]
           and f["job"]["group"] == 0 and f["job"]["seed"] in (1, 2, 3) and not (f["job"].get("terms") or {})]
    return {f["job"]["seed"]: f for f in out}
def wmem(f):
    U, Z = load_UZ(f["dir"]); return U, Z, M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()})
def adj(w1, w2):
    real = M.track_pair(w1, w2); ch = M.track_pair(w1, M._shuffled(w2, rng))
    return tuple((real[t] - ch[t]) / (1 - ch[t]) for t in ("A_model", "B_model"))
def train_value(U, Z, rd):
    x = rd.tie_val; xh = ((U[rd.row_facet][rd.tie_row] @ Z[rd.name]) * U[rd.col_facet][rd.tie_col]).sum(1)
    nr = np.bincount(rd.tie_row, minlength=rd.shape[0])[rd.tie_row]; nc = np.bincount(rd.tie_col, minlength=rd.shape[1])[rd.tie_col]
    return M.partial_spearman(x, xh, [nr, nc, nr.astype(float) * nc])
for beta, lname in ((2, "squared error"), (1, "KL")):
    W, B = fits("stage0_pass1", beta), fits("binary_test", beta)
    mw = {s: wmem(f) for s, f in W.items()}; mb = {s: wmem(f) for s, f in B.items()}
    same = [adj(mw[s][2], mb[s][2]) for s in mw if s in mb]
    ww = [adj(mw[a][2], mw[b][2]) for a, b in itertools.combinations(sorted(mw), 2)]
    bb = [adj(mb[a][2], mb[b][2]) for a, b in itertools.combinations(sorted(mb), 2)]
    f = lambda L: "A %.3f B %.3f" % tuple(np.mean(L, 0))
    print(lname, "| binary vs weighted, same seed:", f(same), "| weighted vs weighted:", f(ww), "| binary vs binary:", f(bb))
    for rel in ("child_article", "parent_child"):
        tw = [train_value(mw[s][0], mw[s][1], rds[rel]) for s in mw]; tb = [train_value(mb[s][0], mb[s][1], rds[rel]) for s in mb]
        print("   training value", rel, "weighted %.3f binary %.3f" % (np.mean(tw), np.mean(tb)))
