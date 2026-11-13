# Stage-0 comparison tests (ticket 107): classes, the half-split margin, the decision rule (hand cases), and a smoke run
# on the checkpoint-3 fits (one hold-out group → every class inconclusive by construction; checks that all paths run).
# Run from fullscale_pipeline/m1: python -m fit.tests.test_stage0_compare
import json, math
import numpy as np
from fit import stage0_compare as S

FAILS = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: FAILS.append(name)

c = S.classify([0.05, 0.06, 0.055, 0.052, 0.058], 0.01)
check("classify: clear positive Δ beyond δ → KL better", c["class"] == "KL better")
c = S.classify([0.001, -0.001, 0.0005, -0.0004, 0.0002], 0.01)
check("classify: Δ near 0, 90% interval within ±δ → equivalent", c["class"] == "equivalent")
c = S.classify([0.005, 0.006, 0.0055, 0.0052, 0.0058], 0.01)
check("classify: interval > 0 but inside ±δ → equivalent (TOST), not 'better'", c["class"] == "equivalent")
check("classify: one group → inconclusive", S.classify([0.5], 0.01)["class"] == "inconclusive")
rng = np.random.default_rng(0)
per = {b: {g: list(rng.normal(0, 1, 10)) for g in range(5)} for b in (2, 1)}
d = S.half_margin(per, lambda x: float(np.mean(x)), rng, n_splits=999)
# theory: half-mean difference SD = sqrt(2/5) per group, mean over 5 groups → sqrt(2/25); 95th pct of |N| ≈ 1.96 SD; ÷ √2
check("half margin ≈ 1.96·√(2/25)/√2 = 0.392 (±15%)", abs(d - 1.96 * math.sqrt(2 / 25) / math.sqrt(2)) < 0.06)

def res(m1="inconclusive", m2="inconclusive", m4="inconclusive", m3="no reading (inconclusive)"):
    return {"S0-M1 membership agreement": {"verdict": m1}, "S0-M2 held-out (existence and value)": {"verdict": m2, "reported": {}},
            "S0-M4 seed stability": {"verdict": m4}, "S0-M3 cross-scoring": {"reading": m3}, "per-entity fit (training AUC)": {}}
G = lambda f2=False, f1=False: {2: f2, 1: f1}
r = S.decide({5: res(m1="same", m2="KL better")}, {5: G()})
check("rule: S0-M1 same → squared error even if S0-M2 favours KL", r["loss"] == "squared error")
r = S.decide({5: res(m1="different", m2="KL better"), 8: res(m2="KL better"), 13: res()}, {5: G(), 8: G(), 13: G()})
check("rule: KL wins 2 of 3 candidate K → KL", r["loss"] == "KL")
r = S.decide({5: res(m2="KL better"), 8: res()}, {5: G(), 8: G()})
check("rule: 1 of 2 → tie → squared error (default)", r["loss"] == "squared error")
r = S.decide({5: res(m2="equivalent", m4="KL more stable")}, {5: G()})
check("rule: S0-M2 equivalent → the more stable loss (KL)", r["loss"] == "KL")
r = S.decide({5: res(m2="KL better"), 8: res(m2="KL better")}, {5: G(f1=True), 8: G(f1=True)})
check("rule: KL fails the gate at the majority of K → squared error", r["loss"] == "squared error" and "gate" in r["reason"])
r = S.decide({5: res(), 8: res()}, {5: G(True, True), 8: G(True, True)})
check("rule: both fail the gate → owner", r["loss"] is None)
r = S.decide({5: res(m2="KL better", m3="dominance: squared error")}, {5: G()})
check("rule: cross-scoring dominance against the choice is flagged, not reversed", r["loss"] == "KL" and any("against" in f for f in r["flags"]))

# smoke run on real fits (checkpoint 3: approach 1, group 0, K 5 and 20, 3 seeds per loss at fraction 0.03, hybrid, float64)
rep = S.run("/mnt/hum01-rds/Basov/p91688di/m1_fits", "checkpoint3", None,
            {"frac": 0.03, "start": "nndsvd_sub0.8", "dtype": "float64"})
a1 = rep.get("approach_1", {})
check("smoke: both K compared", sorted(a1.get("per_K", {})) == [5, 20])
k5 = a1["per_K"][5]
check("smoke: S0-M1–M4 and per-entity fit present", all(k in k5 for k in ("S0-M1 membership agreement", "S0-M2 held-out (existence and value)",
      "S0-M3 cross-scoring", "S0-M4 seed stability", "per-entity fit (training AUC)")))
check("smoke: one group → classes inconclusive, decision falls to the default", a1["decision"]["loss"] == "squared error")
print("     S0-M1 D (A, B) at K=5:", [k5["S0-M1 membership agreement"][t]["mean"] for t in ("A", "B")],
      "| S0-M4 Δ:", [k5["S0-M4 seed stability"][t]["mean"] for t in ("A", "B")])
print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
