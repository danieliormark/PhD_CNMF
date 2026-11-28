# Tests of fit/d1_extra.py (RL-429): pair summaries, second-best J*, re-check job list. Run: PYTHONPATH=. python -m fit.tests.test_d1_extra
import numpy as np
from fit import d1_extra as X, d1_analysis as D
from fit.tests.test_d1_analysis import _rep


def test_summarise_pairs():
    rng = np.random.default_rng(0); rows = []
    for g in range(10):
        for s in range(1, 6):
            rows.append({"approach": 1, "K": 5, "group": g, "seed": s, "arm": "J1", "v": 0.10 + 0.001 * rng.standard_normal()})
            rows.append({"approach": 1, "K": 5, "group": g, "seed": s, "arm": "J2", "v": 0.12 + 0.001 * rng.standard_normal()})
    out = X.summarise_pairs(rows, ("v",))["approach_1|K_5"]
    d = out["pairs"]["J2 vs J1"]["v"]
    assert abs(d["diff"] - 0.02) < 0.002 and d["groups"] == 10 and d["se"] < 0.001, d
    assert abs(d["relative"] - 0.2) < 0.03 and d["in_ref_seed_sd"] > 10, d
    assert "J3a vs J1" not in out["pairs"]                       # absent arm: no pair


def test_second_best():
    per = {5: {"J2": ("redundant", 0.0, False, 0.02), "J3a": ("redundant", 0.0, False, 0.005), "J5": ("redundant", 0.0, False, 0.01)}}
    rep = _rep(per); rep["candidate_K"] = [5]
    rep["arms"], rep["decision"] = D.decide(rep, [5])
    assert rep["decision"]["J_star"] == "J2"
    assert X.second_best(rep) == "J5"                           # without J2: highest s_journal among the rest
    assert D.COMPARED == ("J2", "J3a", "J3b", "J4", "J5")       # the module constant is restored
    rep["decision"]["J_star"] = "J1"
    assert X.second_best(rep) == "J2"                           # edge case (owner 2026-11-28): J* = J1 → J2


def test_recheck_jobs():
    jobs = X.recheck_jobs(1, 32, ["J1", "J2"], 1.5)
    assert len(jobs) == 2 * 10 * 5 and {j["K"] for j in jobs} == {32} and {j["stage"] for j in jobs} == {"stage2_recheck"}
    assert all(j["terms"]["lambda_child"] == 1.5 and j["dtype"] == "float32" and j["max_epochs"] == 8000 for j in jobs)
    assert {j["arm"]["j"] for j in jobs} == {"J1", "J2"} and all(j["arm"].get("use_journal") is False for j in jobs if j["arm"]["j"] == "J1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
