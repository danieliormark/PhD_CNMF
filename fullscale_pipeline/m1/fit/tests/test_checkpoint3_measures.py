# Checkpoint 3 tests, measures module (F6), ticket 107. Hand-computed examples, dense brute force, and the reporting-code
# check against the planted truth (decided 2026-11-08). Run from fullscale_pipeline/m1:
#   python -m fit.tests.test_checkpoint3_measures
import sys
import numpy as np
import scipy.sparse as sp

from fit import data as D, measures as M, worlds as Wd

FAILS = []


def check(name, got, want=True, tol=None):
    ok = (got == want) if tol is None else abs(got - want) <= tol * max(1.0, abs(want))
    print(("PASS " if ok else "FAIL ") + f"{name}: {got}" + ("" if tol is None and want is True else f" (want {want})"))
    if not ok:
        FAILS.append(name)


def rd_from_dense(name, rf, cf, X, obs=None, hid=None, empties=(), diag=False, dom="social"):
    n, m = X.shape
    obs = np.ones(n, bool) if obs is None else obs
    r, c = np.nonzero(X)
    h = np.zeros(len(r), bool) if hid is None else hid
    er = np.array([e[0] for e in empties], dtype=np.int64); ec = np.array([e[1] for e in empties], dtype=np.int64)
    return D.RelationData(name, rf, cf, X.shape, 1.0, dom, r[~h], c[~h], X[r, c][~h], obs, diag, r[h], c[h], X[r, c][h], er, ec, 1.0)


# ----------------------------------------------------------------------------- 1. parts, side readings, presence
Ua = np.array([[2.0, 0.0], [0.0, 1.0]]); Ub = np.array([[1.0, 0.0], [0.0, 3.0]]); Z = np.array([[0.8, 0.1], [0.2, 0.5]])
rd = rd_from_dense("t", "a", "b", np.ones((2, 2)))
P, T = M.parts({"a": Ua, "b": Ub}, {"t": Z}, rd)
check("parts: part(k,l) = Z a_k b_l (hand: [[1.6,0.6],[0.2,1.5]])", bool(np.allclose(P, [[1.6, 0.6], [0.2, 1.5]])))
check("parts add up to the reconstructed tie weight Σ X̂", T, float((Ua @ Z @ Ub.T).sum()), 1e-12)
pr = M.presence({"a": Ua, "b": Ub}, {"t": Z}, [rd])["t"]
check("half credit = mean of the side readings (hand: 2.0/3.9, 1.9/3.9)", bool(np.allclose(pr["half_credit"], [2.0 / 3.9, 1.9 / 3.9])))
check("relative presence = K × half credit; sums to K", float(pr["presence"].sum()), 2.0, 1e-12)
check("within share = trace / T (hand: 3.1/3.9)", pr["within_share"], 3.1 / 3.9, 1e-12)

# scale freedom: presence and weighted memberships unchanged
rng = np.random.default_rng(0)
U = {"a": rng.random((6, 3)), "b": rng.random((5, 3)), "c": rng.random((4, 3))}
Zd = {"ab": rng.random((3, 3)), "bc": rng.random((3, 3))}
rel = {"ab": ("a", "b"), "bc": ("b", "c")}
rds_s = [rd_from_dense("ab", "a", "b", rng.random((6, 5))), rd_from_dense("bc", "b", "c", rng.random((5, 4)))]
p0 = M.presence(U, Zd, rds_s); w0 = M.weighted_memberships(U, Zd, rel)
cv = np.array([2.0, 0.3, 5.0])
U2 = dict(U); U2["b"] = U["b"] * cv; Z2 = {"ab": Zd["ab"] / cv[None, :], "bc": Zd["bc"] / cv[:, None]}
p1 = M.presence(U2, Z2, rds_s); w1 = M.weighted_memberships(U2, Z2, rel)
check("presence unchanged under the scale freedom", max(float(np.abs(p0[r]["presence"] - p1[r]["presence"]).max()) for r in p0), 0.0, 1e-12)
check("weighted memberships unchanged under the scale freedom", max(float(np.abs(w0[f]["P"] - w1[f]["P"]).max()) for f in w0), 0.0, 1e-12)

# ----------------------------------------------------------------------- 2. mono-domain worked example (2026-11-07)
fake = {"author_article": {"presence": np.array([2.0, 0.9, 0.012]), "domain": "social", "row_facet": "author", "col_facet": "article"},
        "author_affiliation": {"presence": np.array([2.1, 0.8, 0.03]), "domain": "social", "row_facet": "author", "col_facet": "organisation"},
        "article_journal": {"presence": np.array([2.08, 0.94, 0.15]), "domain": "social", "row_facet": "article", "col_facet": "journal"},
        "child_article": {"presence": np.array([1.4, 0.9, 0.90]), "domain": "semantic", "row_facet": "article", "col_facet": "child"},
        "parent_child": {"presence": np.array([1.3, 0.9, 0.60]), "domain": "semantic", "row_facet": "parent", "col_facet": "child"}}
dom = M.domain_presence(fake)
check("worked example: social presence of community 3 = 0.064", float(dom["social"]["mean"][2]), 0.064, 1e-9)
check("worked example: semantic presence of community 3 = 0.75", float(dom["semantic"]["mean"][2]), 0.75, 1e-12)
fl = M.mono_flags(dom, {"x": np.array([1.0, 1.0, 1.0])})
check("worked example: community 3 mono-domain (semantic only), 1 and 2 not", fl["mono_domain"] == [(2, "semantic only")])
fl2 = M.mono_flags(dom, {"x": np.ones(3)}, chance_relations=list(dom["social"]["relations"]))
check("mono-domain not flagged when the low domain is fitted only at chance", fl2["mono_domain"] == [])
fl3 = M.mono_flags(dom, {"child": np.array([0.2, 0.6, 1.0]), "article": np.array([0.05, 0.6, 1.0]), "parent": np.array([0.01, 0.7, 1.0])})
check("mono-facet: present (≥ 0.5) in exactly one facet, < 0.1 elsewhere", fl3["mono_facet"] == [])
fl4 = M.mono_flags(dom, {"child": np.array([0.9, 0.6, 1.0]), "article": np.array([0.05, 0.6, 1.0]), "parent": np.array([0.01, 0.7, 1.0])})
check("mono-facet: community 1 only in child hyperedges", fl4["mono_facet"] == [(0, "child")])

# --------------------------------------------------------------------------- 3. entity measures vs dense brute force
rng = np.random.default_rng(3)
n, m, K = 9, 7, 3
X = sp.random(n, m, density=0.4, random_state=3).toarray()
obs = np.ones(n, bool); obs[[2]] = False; X[2] = 0
r_, c_ = np.nonzero(X); hid = np.zeros(len(r_), bool); hid[[0, 3]] = True
emp = [(i, j) for i in range(n) for j in range(m) if obs[i] and X[i, j] == 0][:4]
rd3 = rd_from_dense("t", "a", "b", X, obs, hid, emp)
A = rng.random((n, K)) + 0.1; B = rng.random((m, K)) + 0.1; Zr = rng.random((K, K)) + 0.1
em = M.entity_measures({"a": A, "b": B}, {"t": Zr}, [rd3])
Xh = A @ Zr @ B.T
train = np.zeros((n, m), bool); train[obs] = True; train[r_[hid], c_[hid]] = False
for (i, j) in emp: train[i, j] = False
Xv = X.copy(); Xv[r_[hid], c_[hid]] = 0
for side, f, ax in (("rows", "a", 1), ("columns", "b", 0)):
    sx = (Xv * train).sum(ax); smin = (np.minimum(Xh, Xv) * (Xv > 0) * train).sum(ax); shat = (Xh * train).sum(ax)
    sq = (((Xv - Xh) ** 2) * train).sum(ax)
    ok = sx > 0
    check(f"reproduced share = brute force ({side})", float(np.abs(em[f]["reproduced_share"][ok] - smin[ok] / sx[ok]).max()), 0.0, 1e-12)
    check(f"on-target share = brute force ({side})", float(np.abs(em[f]["on_target_share"][ok] - smin[ok] / shat[ok]).max()), 0.0, 1e-12)
    check(f"coverage = brute force ({side})", float(np.abs(em[f]["coverage"][ok] - shat[ok] / sx[ok]).max()), 0.0, 1e-12)
    check(f"squared-error accuracy = brute force ({side})", float(np.abs(em[f]["accuracy_squared_error"][ok] - sq[ok] / sx[ok]).max()), 0.0, 1e-10)
    klc = np.where(Xv > 0, Xv * np.log(np.where(Xv > 0, Xv, 1) / Xh) - Xv, 0) + Xh
    check(f"KL accuracy = brute force ({side})", float(np.abs(em[f]["accuracy_kl"][ok] - (klc * train).sum(ax)[ok] / sx[ok]).max()), 0.0, 1e-10)
rt = Xv.sum(1); ct = Xv.sum(0); X0 = np.outer(rt, ct) / Xv.sum()
check("degree-only reference = brute force (rows)", float(np.abs(em["a"]["reproduced_share_degree_only"][rt > 0] -
      ((np.minimum(X0, Xv) * (Xv > 0)).sum(1) / np.maximum(rt, 1e-300))[rt > 0]).max()), 0.0, 1e-12)
rep = [em["a"]["reproduced_share"] + d for d in (0.0, 0.01, -0.01)]
check("placed: mean − degree-only beyond seed SD", bool(np.array_equal(M.placed_entities(rep, em["a"]["reproduced_share_degree_only"]),
      (np.vstack(rep).mean(0) - em["a"]["reproduced_share_degree_only"]) > np.vstack(rep).std(0, ddof=1))))

# --------------------------------------------------------------------------------------------- 4. AUC helpers
gid = np.repeat(np.arange(5), 8); val = rng.random(40); pos = rng.random(40) < 0.4
auc, npos, nneg = M._grouped_auc(gid, val, pos, 5)
brute = []
for g in range(5):
    v, p = val[gid == g], pos[gid == g]
    if p.any() and (~p).any():
        brute.append(np.mean([(a > b) + 0.5 * (a == b) for a in v[p] for b in v[~p]]))
    else:
        brute.append(np.nan)
check("grouped AUC = pairwise brute force", float(np.nanmax(np.abs(auc - np.array(brute)))), 0.0, 1e-12)
ho = M.heldout({"a": A, "b": B}, {"t": Zr}, rd3)
ht = Xh[r_[hid], c_[hid]]; he = np.array([Xh[i, j] for i, j in emp])
n_empty_obs = int(obs.sum()) * m - len(r_)
scale = np.floor(0.1 * n_empty_obs) / len(emp)
check("held-out squared error with empty cells scaled (option B)", ho["squared_error"], float(((X[r_[hid], c_[hid]] - ht) ** 2).sum() + scale * (he ** 2).sum()), 1e-12)
check("captured share = 1 − error / error of predicting 0", ho["captured_share"], 1 - ho["squared_error"] / float((X[r_[hid], c_[hid]] ** 2).sum()), 1e-12)
check("hidden-cell AUC = pairwise brute force", ho["auc"], float(np.mean([(a > b) + 0.5 * (a == b) for a in ht for b in he])), 1e-12)

# ---------------------------------------------------------------------------------------------- 5. tie alignment
Xa = np.eye(4)
rda = rd_from_dense("t", "a", "b", Xa)
Pp = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
ta = M.tie_alignment(rda, Pp, Pp, n_shuffles=99, seed=1)
check("tie alignment: aligned pure memberships → 1", ta["observed"], 1.0, 1e-12)
tu = M.tie_alignment(rda, np.full((4, 2), 0.5), np.full((4, 2), 0.5), n_shuffles=9)
check("tie alignment: uniform memberships → 1/K", tu["observed"], 0.5, 1e-12)
check("tie alignment: shuffled reference below observed for aligned memberships", ta["shuffled_mean"] < 1.0)

# ------------------------------------------------------------------------------------------------- 6. Track A/B
wm1 = M.weighted_memberships(U, Zd, rel)
pmt = np.array([2, 0, 1])
U3 = {f: u[:, pmt] for f, u in U.items()}; Z3 = {r: z[np.ix_(pmt, pmt)] for r, z in Zd.items()}
wm3 = M.weighted_memberships(U3, Z3, rel)
tp = M.track_pair(wm1, wm3)
check("Track A = 1 for the same fit with communities relabelled", tp["A_model"], 1.0, 1e-6)
check("Track B = 1 for the same fit with communities relabelled", tp["B_model"], 1.0, 1e-9)
st = M.stability([wm1, wm3, wm1], seed=2)
check("stability: identical fits beat their shuffled chance in every seed", all(s["passes"] for s in st["per_seed"]))

# ---------------------------------------------------------------------------------------------- 7. counting rule
F = np.zeros((10, 10), bool); F[0, :3] = True; F[1, :3] = True
cr = M.count_rule(F)
check("counting rule: 2 of 10 groups with 3/10 failing → groups fail, K passes (2/10 ≤ 20%)", cr["group_fails"][:2] == [True, True] and not cr["K_fails"])
F[2, :3] = True
check("counting rule: 3 of 10 groups fail → K fails", M.count_rule(F)["K_fails"])
F = np.zeros((10, 10), bool); F[:, :2] = True; F[0, :] = True; F[1, :] = True
check("counting rule: pooled 36% failing → K fails although only 2 groups fail", M.count_rule(F)["K_fails"] and M.count_rule(F)["pooled"])

# -------------------------------------------------- 8. reporting code against the planted truth (2026-11-08)
rds, Ut, Zt = Wd.code_check_world("noiseless", seed=1, child_only="split")
rel = {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}
pres = M.presence(Ut, Zt, rds, floor_zero=None)
dom = M.domain_presence(pres); fac = M.facet_presence(pres); fl = M.mono_flags(dom, fac)
print("     planted: domain presence semantic", np.round(dom["semantic"]["mean"], 3), "social", np.round(dom["social"]["mean"], 3))
print("     planted: facet presence", {f: np.round(v, 2).tolist() for f, v in fac.items()})
check("planted semantic-only community flagged mono-domain (semantic only)", (Wd.SEMANTIC_ONLY, "semantic only") in fl["mono_domain"])
check("planted child-only community flagged mono-facet (child)", (Wd.CHILD_ONLY, "child") in fl["mono_facet"])
from fit.solver import fit
res = fit(rds, 6, 2, seed=0, max_epochs=200, lr=0.03, lr_mode="recompute", init_U=Ut, init_Z=Zt)
pres_f = M.presence(res["U"], res["Z"], rds)
check("fit started at the truth: presence equals the planted presence", max(float(np.abs(pres_f[r]["presence"] - pres[r]["presence"]).max()) for r in pres), 0.0, 1e-2)
fl_f = M.mono_flags(M.domain_presence(pres_f), M.facet_presence(pres_f))
check("fit started at the truth: same mono-domain and mono-facet flags", fl_f == fl)
emc = M.empty_columns(pres, rds, M.weighted_memberships(Ut, Zt, rel))
soc = [r for r in rel if D.RELATIONS[r][3] == "social"]
tw = {r: float(pres[r]["half_credit"][Wd.SEMANTIC_ONLY] * pres[r]["T"]) for r in soc}
print("     KNOWN ISSUE (to the owner): semantic-only community's tie weight in social relations, core cells at the 1e-7 floor:",
      {r: f"{v:.1e}" for r, v in tw.items()}, "vs 0.1th-percentile tie", {r: f"{np.percentile(next(x for x in rds if x.name == r).tie_val, 0.1):.1e}" for r in soc})
pres0 = M.presence(Ut, Zt, rds, floor_zero=1e-7)
emc0 = M.empty_columns(pres0, rds, M.weighted_memberships(Ut, Zt, rel))
check("planted, floor-level core cells read as 0: semantic-only community empty in every social relation",
      all(Wd.SEMANTIC_ONLY in emc0["p0.1"]["relations"][r] for r in soc))
check("planted: no community empty in child×article", emc["p0.1"]["relations"]["child_article"] == [])

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
