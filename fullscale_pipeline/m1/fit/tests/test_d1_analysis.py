# Known-answer tests of the pure parts of fit/d1_analysis.py (run directly: python -m fit.tests.test_d1_analysis).
import math
import numpy as np
from scipy import stats
from fit import d1_analysis as D


def test_holm():
    adj = D.holm([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005·4 = 0.02, 0.01·3 = 0.03, 0.03·2 = 0.06, 0.04·1 = 0.04 → running max 0.02, 0.03, 0.06, 0.06
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02]), adj
    assert D.holm([None, 0.2]) == [None, 0.2]


def test_mvt_crit():
    assert abs(D.mvt_crit([[1.0]], 9, 0.95) - stats.t.ppf(0.975, 9)) < 1e-12
    R = np.full((5, 5), 0.5); np.fill_diagonal(R, 1)
    c = D.mvt_crit(R, 9, 0.95)
    assert abs(c - 3.06) < 0.04, c                          # pre-registration §6: 10 groups, own SE, Dunnett 3.06
    c_ind = D.mvt_crit(np.eye(5), 10_000, 0.95)
    assert abs(c_ind - stats.norm.ppf(1 - (1 - 0.95 ** 0.2) / 2)) < 0.02, c_ind     # Šidák for independent tests


def test_classify():
    assert D.classify(0.02, 0.005, 3.0, 2.5, 0.01)[0] == "adds information"
    assert D.classify(0.008, 0.002, 3.0, 2.5, 0.01)[0] == "inconclusive"       # > 0 but below δ, interval crosses δ
    assert D.classify(-0.02, 0.005, 3.0, 2.5, 0.01)[0] == "misleads"
    assert D.classify(0.001, 0.002, 3.0, 2.5, 0.01)[0] == "redundant"          # 90 % interval ±0.005 inside ±0.01
    assert D.classify(0.001, 0.005, 3.0, 2.5, 0.01)[0] == "inconclusive"


def test_compare_family():
    rng = np.random.default_rng(1)
    G = 10
    D_ = np.column_stack([0.03 + 0.005 * rng.standard_normal(G), 0.0002 * rng.standard_normal(G),
                          -0.03 + 0.005 * rng.standard_normal(G)])
    out = D.compare_family(D_, ["a", "b", "c"], 0.01, [(0.05, 0.005)])
    assert out["a"]["class"] == "adds information" and out["b"]["class"] == "redundant" and out["c"]["class"] == "misleads", out
    assert out["crit_dunnett_95"] > out["crit_t_95"]


def test_majority():
    assert D.majority(["redundant", "redundant", "misleads"]) == "redundant"
    assert D.majority(["redundant", "misleads"]) == "inconclusive"
    assert D.majority(["misleads"]) == "misleads"
    assert D.majority([]) is None


def test_half_split():
    g = np.arange(10.0)
    m = D.half_split_margin(g)
    vals = []
    import itertools
    for A in itertools.combinations(range(10), 5):
        if 0 in A:
            B = [i for i in range(10) if i not in A]
            vals.append(abs(g[list(A)].mean() - g[B].mean()))
    assert len(vals) == 126 and abs(m - np.percentile(vals, 95) / math.sqrt(2)) < 1e-12


def test_permutation_counts():
    # Jx has 1 more community per fit in every group: the observed statistic is the largest relabelling → p ≈ 1/(1+n)
    st = [D.label_stats_diff([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]) for _ in range(10)]
    assert len(st[0]) == 252 and abs(st[0].mean()) < 1e-12
    p, _ = D.perm_test(st, 1.0, "greater", n=9999)
    assert p < 0.001, p
    st0 = [D.label_stats_diff([1, 0, 1, 0, 1], [0, 1, 0, 1, 0]) for _ in range(10)]
    p0, _ = D.perm_test(st0, -0.2, "greater", n=9999)
    assert p0 > 0.5, p0


def test_permutation_agreement():
    n = 10; S = np.full((n, n), 0.5)
    S[:5, :5] = 0.9; S[5:, 5:] = 0.9
    st = D.label_stats_agreement(S, 5)
    obs = 0.5 - 0.9
    assert abs(st.min() - obs) < 1e-12                       # the true labelling has the lowest cross − within
    p, _ = D.perm_test([st] * 10, obs, "less", n=9999)
    assert p < 0.001, p
    p1, _ = D.perm_test([D.label_stats_agreement(np.full((n, n), 0.7), 5)] * 10, 0.0, "less", n=999)
    assert p1 == 1.0


def test_band():
    b = D.one_se_band({3: [0.10, 0.11, 0.12], 5: [0.12, 0.13, 0.14], 8: [0.125, 0.126, 0.127], 13: [0.05, 0.06, 0.07]})
    assert b["best"] == 5 and b["band"] == [5, 8], b


def test_loss_shares():
    out = D.loss_shares({"child_article": 8.0, "parent_child": 9.0, "article_journal": 1.0},
                        {"child_article": 10.0, "parent_child": 10.0, "article_journal": 2.0},
                        {"child_article": 0.25, "parent_child": 0.25, "article_journal": 1 / 6})
    # w ΔL: 0.5, 0.25, 1/6 → total 0.91667
    assert abs(out["s"]["child_article"] - 0.5 / 0.916666) < 1e-5 and abs(out["c"]["article_journal"] - 0.5) < 1e-12
    assert abs(out["s_semantic"] - 0.75 / 0.916666) < 1e-5 and out["flag"] is False


def test_journal_shaped():
    rng = np.random.default_rng(0)
    lab = np.repeat(np.arange(10), 20)                     # 10 journals × 20 articles
    P = np.full((200, 3), 1 / 3)
    P[lab == 0] = [0.9, 0.05, 0.05]                        # community 0 built from journal 0
    P[lab != 0] = [0.02, 0.49, 0.49]
    r = D.journal_shaped(P, lab, 10, rng, n_sh=199)
    assert r["journal_shaped"] == 1 and r["shaped_detail"][0]["journal"] == 0 and r["shaped_detail"][0]["k"] == 0, r
    assert r["shaped_detail"][0]["share_ge_0.5"] == 1.0
    r2 = D.journal_shaped(np.full((200, 3), 1 / 3), lab, 10, rng, n_sh=199)
    assert r2["journal_shaped"] == 0


def test_clearly_worse_all_pairs():
    rng = np.random.default_rng(2)
    z = lambda: rng.standard_normal(10)
    # clear gap ≫ noise and ≥ δ → the lower arm is out
    r = D.clearly_worse_all_pairs({"a": dict(enumerate(0.02 + 0.001 * z())), "b": dict(enumerate(0.001 * z()))}, 0.002)
    assert r["out"] == ["b"] and r["sign_patterns"] == 1024, r
    # gap clear but below δ → none out
    r = D.clearly_worse_all_pairs({"a": dict(enumerate(0.001 + 0.0001 * z())), "b": dict(enumerate(0.0001 * z()))}, 0.002)
    assert r["out"] == [], r
    # two arms: the sign-flip critical value is close to the t quantile (df 9, 2.26)
    r = D.clearly_worse_all_pairs({"a": dict(enumerate(z())), "b": dict(enumerate(z()))}, 0.0)
    assert 1.9 < r["crit"] < 2.6, r["crit"]
    # equal arms: false 'out' rate near 5 % over many draws (all pairs, 3 arms)
    hits = sum(bool(D.clearly_worse_all_pairs({k: dict(enumerate(z())) for k in "abc"}, 0.0)["out"]) for _ in range(400))
    assert hits / 400 < 0.09, hits


def _rep(per, sd=0.002):
    """per: {K: {arm: (class, mean, problem, s_journal)}}; per-group Δ_g = mean + noise (sd) over 10 groups"""
    rng = np.random.default_rng(4)
    return {"per_K": {str(K): {"delta": 0.002,
                               "M1_groups": {x: {str(g): m + sd * z for g, z in enumerate(rng.standard_normal(10))}
                                             for x, (_, m, _, _) in a.items()},
                               "M1": {x: {"class": c, "mean": m, "class_uncorrected": c} for x, (c, m, _, _) in a.items()},
                               "problem": {x: {"problem": p} for x, (_, _, p, _) in a.items()},
                               "M3": {x: {"s_journal": s, "flag": False} for x, (_, _, _, s) in a.items()}}
                      for K, a in per.items()}}


def test_decide():
    # adds information: J3a and J4, within δ of each other, J2 misleads → highest s_journal among the tied (J4)
    per = {5: {"J2": ("misleads", -0.01, False, 0.02), "J3a": ("adds information", 0.010, False, 0.005),
               "J4": ("adds information", 0.0095, False, 0.015), "J5": ("redundant", 0.0, False, 0.01)}}
    arms, d = D.decide(_rep(per), [5])
    assert d["J_star"] == "J4", d
    # C4: J3a clearly better than J4 (gap 0.010 ≫ δ, small noise) → J3a alone, although J4 has the higher s_journal
    per = {5: {"J3a": ("adds information", 0.020, False, 0.005), "J4": ("adds information", 0.010, False, 0.015)}}
    arms, d = D.decide(_rep(per, sd=0.0005), [5])
    assert d["J_star"] == "J3a" and d["step3_C4"]["tied"] == ["J3a"], d
    # none adds information, J2 remains → J2
    per = {5: {"J2": ("redundant", 0.0, False, 0.02), "J3a": ("redundant", 0.0, False, 0.005)}}
    assert D.decide(_rep(per), [5])[1]["J_star"] == "J2"
    # J2 has a journal problem at 2 of 3 candidate K → excluded; remaining J3a / J5 → highest s_journal (J5)
    per = {K: {"J2": ("redundant", 0.0, K != 13, 0.02), "J3a": ("redundant", 0.0, False, 0.005),
               "J5": ("inconclusive", 0.0, K == 5, 0.01)} for K in (5, 8, 13)}
    arms, d = D.decide(_rep(per), [5, 8, 13])
    assert arms["J2"]["excluded_journal_problem"] and not arms["J5"]["excluded_journal_problem"] and d["J_star"] == "J5", (arms, d)
    # a problem at exactly half of the candidate K excludes (owner: "at least half")
    per = {K: {"J2": ("redundant", 0.0, K == 5, 0.02), "J3a": ("redundant", 0.0, False, 0.005)} for K in (5, 8)}
    assert D.decide(_rep(per), [5, 8])[1]["J_star"] == "J3a"
    # every arm out → J1
    per = {5: {"J2": ("misleads", -0.01, False, 0.02), "J3a": ("redundant", 0.0, True, 0.005)}}
    assert D.decide(_rep(per), [5])[1]["J_star"] == "J1"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
