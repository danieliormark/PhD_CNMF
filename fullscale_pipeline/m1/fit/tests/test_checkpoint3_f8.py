# F8 tests (ticket 107): the K protocol (hand cases), the counting rule with unequal seed counts, and an end-to-end dry
# run on the real build (tiny: K = 3, 8 epochs, 2 groups × 3 seeds) through run_fit and aggregate.
# Run from fullscale_pipeline/m1: python -m fit.tests.test_checkpoint3_f8
import json, os, shutil, sys, tempfile
import numpy as np
from fit import aggregate as A, plan as Pl
from fit.run_fit import run_one, job_id

FAILS = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: FAILS.append(name)

# existence = 1 − the 2026-11-09 example's error score (higher better since 2026-11-13); value equal unless given
c = lambda s, se=0.01, v=0.1, vse=0.01, **f: {"existence": 1 - s, "se_existence": se, "value": v, "se_value": vse, "gate_fails": f.get("g", False), "ghost_fails": f.get("h", False),
                             "alignment_fails": f.get("a", False), "convergence_fails": f.get("c", False)}
# example of 2026-11-09: band {8, 13} (cut-off 0.819), both fail the gate → fallback 5
cells = {3: c(0.900), 5: c(0.860), 8: c(0.817, g=True), 13: c(0.814, se=0.005, g=True), 20: c(0.825)}
r = A.choose_k(cells)
check("band = K within one SE of the best (example: {8, 13})", r["band"] == [8, 13])
check("no candidate in the band → fallback to the largest smaller K that passes (5)", r["candidates"] == [] and r["fallback"] == 5)
cells[5]["ghost_fails"] = True; cells[3]["alignment_fails"] = True
r = A.choose_k(cells)
check("nothing below the band passes → no K qualifies, status above the band reported", r["no_K_qualifies"] and r["above_band_status"] == {20: {"passes_all": True}})
cells = {3: c(0.9), 5: c(0.815, se=0.01), 8: c(0.812, se=0.01, c=True)}
r = A.choose_k(cells)
check("K failing by non-convergence alone goes to the owner, not silently excluded", r["convergence_only_to_owner"] == [8] and r["candidates"] == [5])
cells = {3: c(0.9), 5: c(0.812, v=0.10), 8: c(0.815, v=0.085)}
r = A.choose_k(cells)
check("rule b: best K 5, band {5, 8} by existence; 8 removed by value (0.085 < 0.10 − 0.01); candidate 5",
      r["band"] == [5, 8] and r["removed_by_value"] == [8] and r["candidates"] == [5])
cells = {3: c(0.9), 5: c(0.812, v=0.10), 8: c(0.815, v=0.095)}
check("rule b: a value drop within one SE of the best K does not remove it", A.choose_k(cells)["candidates"] == [5, 8])
ref = {"existence": 0.09, "se_existence": 0.002, "value": 0.08, "se_value": 0.003}
check("guard: passes when both parts are within one SE", A.two_part_guard(ref, {"existence": 0.089, "value": 0.078})["passes"])
g_ = A.two_part_guard(ref, {"existence": 0.095, "value": 0.074})
check("guard: fails on value alone (0.006 > 0.003) even when existence improves", not g_["passes"] and g_["existence_ok"] and not g_["value_ok"])
groups = [{"x": [True, True, True, False, False, False, False, False, False, False]}, {"x": [True] + [False] * 7 + [None, None]}]
cr = A.count(groups, "x", 0.2)
check("counting: group 1 fails (3/10 > 20%), group 2 passes (1/8 evaluated); K fails by groups (1/2 > 20%)",
      cr["group_fails"] == [True, False] and cr["evaluated_per_group"] == [10, 8] and cr["K_fails"])

# end-to-end dry run on the real build
tmp = tempfile.mkdtemp(dir="/tmp/claude-766719/-mnt-hum01-home01-p91688di/c37b6c8f-0a98-41fc-8ef0-f898a4724f86/scratchpad")
jobs = [Pl.job("dryrun", 1, g, 3, 2, s, 0.03, max_epochs=8) for g in (0, 1) for s in (1, 2, 3)]
for j in jobs:
    run_one(j, tmp)
d0 = os.path.join(tmp, "dryrun", job_id(jobs[0]))
check("run_fit writes spec, result, UZ, measures, entity", all(os.path.exists(os.path.join(d0, f)) for f in ("spec.json", "result.json", "UZ.npz", "measures.json", "entity.npz")))
check("re-running a finished job is skipped", run_one(jobs[0], tmp)[1] == "exists")
m = json.load(open(os.path.join(d0, "measures.json")))
check("measures contain held-out scores for the four hidden-cell relations' approach-1 subset", set(m["heldout"]) == {"child_article", "parent_child", "author_affiliation"})
for j in jobs:                                         # mark converged so the checks run (8 epochs never converge)
    p = os.path.join(tmp, "dryrun", job_id(j), "result.json"); r_ = json.load(open(p)); r_["converged"] = True; json.dump(r_, open(p, "w"))
rep = A.aggregate(tmp, "dryrun")
e = next(iter(rep.values()))["per_K"][3]
check("aggregate: two groups scored, SE computed", e["n_groups_scored"] == 2 and e["se"] is not None)
check("aggregate: two-part scores and SEs computed", all(e[k] is not None for k in ("existence", "se_existence", "value", "se_value")))
check("measures: two-part combined score stored", set(m["heldout_two_part_combined"]) == {"existence", "value", "existence_semantic_equal", "value_semantic_equal"})
check("aggregate: every check counted at fraction 0.2", all(e[k]["0.2"] is not None for k in ("gate_fails", "ghost_fails", "alignment_fails", "convergence_fails")))
check("aggregate: between-group stability computed", "between_groups" in e)
print("     dry-run cell:", {k: e[k]["0.2"]["K_fails"] for k in ("gate_fails", "ghost_fails", "alignment_fails")}, "existence", round(e["existence"], 4), "value", round(e["value"], 4))
shutil.rmtree(tmp)
print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
