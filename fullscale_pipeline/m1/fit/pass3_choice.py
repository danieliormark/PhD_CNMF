# Stage 0 pass 3: choice of λ_child and K by J1 (owner 2026-11-19: "I approve J1 option", RL-299), per approach, for
# one study (A: genre hyperedges in; B: without; RL-290/295). Reads the stage's aggregation (aligned numbering, decided
# 2026-11-17; aggregate.py with M1_ALIGNED=1 → _aggregate_aligned.json).
#   1. at each K, the λ rule (pass2_lambda_rule.rule, decided 2026-11-14/16/17) over r ∈ {0, 0.3, 1}: existence band
#      within one SE of the best r, value guard (more than the best r's value SE below → removed), largest remaining r
#      whose stability is not worse than at λ = 0 (gate not failing at 0.2 where λ = 0 passes; Track A / B chance-adjusted
#      not below λ = 0's by more than λ = 0's SE over groups) → r*(K);
#   2. the K protocol (aggregate.choose_k: band by existence, value guard, gate / ghost / alignment / convergence vetoes
#      by the counting rule at fraction 0.2, fallback) over the cells (K, r*(K));
#   3. grid-end rule: flagged when the candidates reach either end of the fitted grid (not fitted here).
# Option (b) (owner 2026-11-20, "I approve option b"): in step 2 existence defines the band and value removes no K (rule b's
# value guard is computed beside it for comparison); candidates = band K passing the gate, ghost, alignment and
# convergence checks (and V1 with --v1 acting); inspection and the text check choose among them.
# Reported beside the choice: the full (K, r) table per approach (existence, value, SEs, stability A / B, every check);
# with --readings, the pass-3 readings per cell (T1 / would-be V1, mono-domain, off-diagonal share, ...).
# Usage: python -m fit.pass3_choice --agg ROOT/stage0_pass3/_aggregate_aligned.json --out OUT.json [--readings R.json]
import argparse, json, os
import numpy as np
from .pass2_lambda_rule import rule, rule_r2, rule_r2c, stab
from .aggregate import choose_k

FRACTION = "0.2"
LAMBDA_RULE = "R2C"       # owner 2026-11-24 ("I agree with option C", RL-373); "R2" (2026-11-21) / "rule_b" = earlier rules
CA_GAIN = {}              # {(approach, K, r): {group: child×article gain}} for R2C, read from the fits' measures
BAND_AT = "best_over_r"   # B2 (owner 2026-11-21); "r_star" = B1 (earlier reading), set by --band-at
V1_ACTS = False      # set by --v1 acting (stage 0: V1 reported only, owner 2026-11-19 "I choose T1 + V1", step-11 decision)
CHECKS = ("gate_fails", "ghost_fails", "alignment_fails", "convergence_fails")


def ca_gains(stage_dirs):
    """Held-out child×article gain (within-class AP − degree-only reference) per (approach, K, r) and hold-out group, mean
    over converged seeds, from the fits' saved measures (R2C, RL-373)."""
    import glob, os
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    for sd in stage_dirs:
        for p in glob.glob(os.path.join(sd, "*", "measures.json")):
            d = os.path.dirname(p); j = json.load(open(os.path.join(d, "spec.json")))["job"]
            if (j.get("arm") or {}).get("swap_p") or not json.load(open(os.path.join(d, "result.json")))["converged"]:
                continue
            tp = (json.load(open(p)).get("heldout_two_part") or {}).get("child_article")
            if tp:
                acc[(j["approach"], j["K"], float(j["arm"]["r"]))][j["group"]].append(tp["ap_within_class"] - tp["baseline_weights_ap_within_class"])
    return {k: {g: float(np.mean(v)) for g, v in byg.items()} for k, byg in acc.items()}


def k_fails(e, c):
    x = (e.get(c) or {}).get(FRACTION)
    return bool(x["K_fails"]) if x else False


def run(agg, readings=None):
    rep = {"status": agg.get("_status", {}).get("fits_finished"), "entity_rule": agg.get("_status", {}).get("entity_rule")}
    cells = [v for k, v in agg.items() if not k.startswith("_")]
    for a in sorted({c["cell"]["approach"] for c in cells}):
        byr = {float(c["cell"]["arm"]["r"]): c["per_K"] for c in cells if c["cell"]["approach"] == a}
        Ks = sorted({int(K) for pk in byr.values() for K in pk}, key=int)
        lam, table = {}, []
        for K in Ks:
            cands = {r: pk[str(K)] for r, pk in byr.items() if str(K) in pk and pk[str(K)].get("existence") is not None
                     and pk[str(K)].get("se_existence") is not None}
            if 0.0 not in cands:
                lam[K] = {"error": "no λ = 0 cell at this K"}; continue
            if LAMBDA_RULE == "R2C":
                lam[K] = rule_r2c(cands, {r: CA_GAIN.get((a, K, r), {}) for r in cands})
            else:
                lam[K] = rule_r2(cands) if LAMBDA_RULE == "R2" else rule(cands)   # R2: owner 2026-11-21
            for r, e in sorted(cands.items()):
                s = stab(e)
                table.append({"K": K, "r": r, "existence": e["existence"], "se_existence": e["se_existence"],
                              "value": e["value"], "se_value": e["se_value"],
                              "existence_lowest_loss_seed": e.get("existence_lowest_loss_seed"),
                              "value_lowest_loss_seed": e.get("value_lowest_loss_seed"),
                              "track_A_adjusted": s["A_adjusted"][0], "track_B_adjusted": s["B_adjusted"][0],
                              "groups_scored": e["n_groups_scored"], "epochs_median": e.get("epochs_median"),
                              **{c: k_fails(e, c) for c in CHECKS}, "chosen_r_at_K": r == lam[K]["chosen_r"]})
        chosen = {K: lam[K]["chosen_r"] for K in Ks if "chosen_r" in lam[K]}
        # B2 (owner 2026-11-21, "I approve B2"): a K's place in the band is judged on its best existence over r (the r
        # that admits it); the K is carried forward at r*(K) from the λ rule, and the gate and vetoes are those of the
        # carried cell (K, r*(K)). BAND_AT = "r_star" reproduces the earlier J1 reading (B1).
        def band_r(K):
            if BAND_AT == "r_star":
                return chosen[K]
            ok = [r for r in byr if str(K) in byr[r] and byr[r][str(K)].get("existence") is not None]
            return max(ok, key=lambda r: byr[r][str(K)]["existence"])
        admit = {K: band_r(K) for K in chosen}
        cellsK = {K: {"existence": byr[admit[K]][str(K)]["existence"], "se_existence": byr[admit[K]][str(K)]["se_existence"],
                      "value": byr[r][str(K)]["value"], "se_value": byr[r][str(K)]["se_value"],
                      **{c: k_fails(byr[r][str(K)], c) for c in CHECKS}} for K, r in chosen.items()}
        # K protocol, option (b) (owner 2026-11-20): existence band, value reported only; V1 (T1 by the counting rule)
        # acting or reported per --v1; rule b (value guard) kept beside it for comparison
        rc = (readings or {}).get("cells", {}).get(f"approach_{a}", {})
        v1 = {K: bool((rc.get(f"K{K}_r{chosen[K]:g}", {}).get("V1_would_be") or {}).get("vetoed")) for K in chosen}
        if V1_ACTS:
            for K in cellsK:
                cellsK[K]["v1_fails"] = v1[K]
        sel = choose_k(cellsK, value_guard=False) if len(cellsK) >= 2 else None
        sel_rule_b = choose_k({K: {k: v for k, v in c.items() if k != "v1_fails"} for K, c in cellsK.items()}) if len(cellsK) >= 2 else None
        out = {"lambda_rule_per_K": lam, "r_star": chosen, "band_r": admit, "band_at": BAND_AT, "K_protocol": sel, "K_protocol_rule_b": sel_rule_b,
               "V1_would_be": v1, "V1_acts": V1_ACTS, "table": table}
        if sel:
            # grid-end rule (stage 0, 2026-11-19: "extended if the new K* falls at the grid's end"; RL-296 for the top).
            # Under option (b) the candidates form a set: each end of the fitted grid is checked on it.
            cand = sel["candidates"] or ([sel["fallback"]] if sel["fallback"] else [])
            acts = []
            if cand and min(cand) == min(Ks):
                acts.append("fit K = 3 (lowest coarse K)" if min(Ks) == 5 else
                            "K = 3 is a candidate: K = 2 belongs to stage 2's refinement (2 added if the band reaches 3)" if min(Ks) == 3
                            else f"candidates reach the lowest grid value K = {min(Ks)}: extend downwards")
            if cand and max(cand) == max(Ks):
                acts.append("candidates reach K = 20: RL-296 adds K = 32 only if the best K is 20 — owner to confirm under option (b)"
                            if max(Ks) == 20 else f"candidates reach the highest grid value K = {max(Ks)}")
            out["grid_end"] = {"candidates": cand, "best_by_existence": sel["best"], "lowest_K_fitted": min(Ks),
                               "highest_K_fitted": max(Ks), "actions": acts}
            out["chosen"] = [{"K": K, "r": chosen[K]} for K in sel["candidates"]] or (
                [{"K": sel["fallback"], "r": chosen[sel["fallback"]], "fallback": True}] if sel["fallback"] else [])
        if readings:
            out["readings"] = rc
        rep[f"approach_{a}"] = out
        print(f"approach {a}: r*(K) = {chosen}; K protocol (option b, band at {BAND_AT}: r {admit}, V1 {'acting' if V1_ACTS else 'reported'}) — best by existence "
              f"{sel and sel['best']}, band {sel and sel['band']}, removed by V1 {sel and sel['removed_by_v1']}, "
              f"candidates {sel and sel['candidates']}, fallback {sel and sel['fallback']}; V1 would-be {v1}; "
              f"[rule b for comparison: removed by value {sel_rule_b and sel_rule_b['removed_by_value']}, candidates "
              f"{sel_rule_b and sel_rule_b['candidates']}]; grid end: {out.get('grid_end')}")
        print(f"   {'K':>3} {'r':>4} {'existence':>16} {'value':>16} {'A adj':>7} {'B adj':>7}  gate ghost align conv  r*(K)")
        for t in table:
            print(f"   {t['K']:>3} {t['r']:>4} {t['existence']:.4f} ± {t['se_existence']:.4f} {t['value']:.4f} ± {t['se_value']:.4f}"
                  f" {t['track_A_adjusted']:7.3f} {t['track_B_adjusted']:7.3f}  {int(t['gate_fails']):>4} {int(t['ghost_fails']):>5}"
                  f" {int(t['alignment_fails']):>5} {int(t['convergence_fails']):>4}  {'*' if t['chosen_r_at_K'] else ''}")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--agg", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--readings", default=None, nargs="+"); ap.add_argument("--v1", choices=("reported", "acting"), default="reported")
    ap.add_argument("--band-at", choices=("best_over_r", "r_star"), default="best_over_r")
    ap.add_argument("--lambda-rule", choices=("R2C", "R2", "rule_b"), default="R2C")
    a = ap.parse_args()
    global V1_ACTS, BAND_AT, LAMBDA_RULE
    V1_ACTS = a.v1 == "acting"; BAND_AT = a.band_at; LAMBDA_RULE = a.lambda_rule
    # several aggregations / readings (λ calibration, 2026-11-21: pass 3's r 0 / 0.3 / 1 with the new r of λ calibration):
    # cells are merged; each r must come from one stage only
    agg = {"_status": {"fits_finished": 0}}
    rules = set()
    for i, path in enumerate(a.agg):
        d = json.load(open(path))
        # item (l) entity set (RL-345): aggregations made under different rules are not merged; older files = placed rule
        rules.add(json.dumps((d.get("_status") or {}).get("entity_rule") or {"rule": "placed", "c": None}, sort_keys=True))
        agg["_status"]["fits_finished"] += (d.get("_status") or {}).get("fits_finished", 0)
        agg.update({f"{i}|{k}": v for k, v in d.items() if not k.startswith("_")})
    assert len(rules) == 1, f"aggregations use different entity rules: {sorted(rules)}"
    agg["_status"]["entity_rule"] = json.loads(rules.pop())
    rs = [(v["cell"]["approach"], float(v["cell"]["arm"]["r"])) for k, v in agg.items() if not k.startswith("_")]
    assert len(rs) == len(set(rs)), "an (approach, r) cell appears in more than one aggregation"
    readings = None
    if a.readings:
        readings = {"cells": {}}
        for path in a.readings:
            for ap_, cells in json.load(open(path))["cells"].items():
                readings["cells"].setdefault(ap_, {}).update(cells)
    if LAMBDA_RULE == "R2C":
        CA_GAIN.update(ca_gains([os.path.dirname(p) for p in a.agg]))
    rep = run(agg, readings)
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


if __name__ == "__main__":
    main()
