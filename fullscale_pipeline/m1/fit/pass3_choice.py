# Stage 0 pass 3: choice of λ_child and K by J1 (owner 2026-11-19: "I approve J1 option", RL-299), per approach, for
# one study (A: genre hyperedges in; B: without; RL-290/295). Reads the stage's aggregation (aligned numbering, decided
# 2026-11-17; aggregate.py with M1_ALIGNED=1 → _aggregate_aligned.json).
#   1. at each K, the λ rule (pass2_lambda_rule.rule, decided 2026-11-14/16/17) over r ∈ {0, 0.3, 1}: existence band
#      within one SE of the best r, value guard (more than the best r's value SE below → removed), largest remaining r
#      whose stability is not worse than at λ = 0 (gate not failing at 0.2 where λ = 0 passes; Track A / B chance-adjusted
#      not below λ = 0's by more than λ = 0's SE over groups) → r*(K);
#   2. the K protocol (aggregate.choose_k: band by existence, value guard, gate / ghost / alignment / convergence vetoes
#      by the counting rule at fraction 0.2, fallback) over the cells (K, r*(K));
#   3. grid-end rule (RL-296): if the best K is 20, K = 32 is to be fitted (flagged, not fitted here).
# Reported beside the choice: the full (K, r) table per approach (existence, value, SEs, stability A / B, every check);
# with --readings, the pass-3 readings per cell (T1 / would-be V1, mono-domain, off-diagonal share, ...).
# Usage: python -m fit.pass3_choice --agg ROOT/stage0_pass3/_aggregate_aligned.json --out OUT.json [--readings R.json]
import argparse, json
import numpy as np
from .pass2_lambda_rule import rule, stab
from .aggregate import choose_k

FRACTION = "0.2"
CHECKS = ("gate_fails", "ghost_fails", "alignment_fails", "convergence_fails")


def k_fails(e, c):
    x = (e.get(c) or {}).get(FRACTION)
    return bool(x["K_fails"]) if x else False


def run(agg, readings=None):
    rep = {"status": agg.get("_status", {}).get("fits_finished")}
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
            lam[K] = rule(cands)
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
        cellsK = {K: {"existence": byr[r][str(K)]["existence"], "se_existence": byr[r][str(K)]["se_existence"],
                      "value": byr[r][str(K)]["value"], "se_value": byr[r][str(K)]["se_value"],
                      **{c: k_fails(byr[r][str(K)], c) for c in CHECKS}} for K, r in chosen.items()}
        sel = choose_k(cellsK) if len(cellsK) >= 2 else None
        out = {"lambda_rule_per_K": lam, "r_star": chosen, "K_protocol": sel, "table": table}
        if sel:
            out["grid_end"] = {"best_is_highest_K": sel["best"] == max(Ks),
                               "action": "fit K = 32 (grid-end rule, RL-296)" if sel["best"] == max(Ks) == 20 else None}
            out["chosen"] = [{"K": K, "r": chosen[K]} for K in sel["candidates"]] or (
                [{"K": sel["fallback"], "r": chosen[sel["fallback"]], "fallback": True}] if sel["fallback"] else [])
        if readings:
            out["readings"] = {k: v for k, v in readings.get(f"approach_{a}", {}).items()}
        rep[f"approach_{a}"] = out
        print(f"approach {a}: r*(K) = {chosen}; K protocol — best {sel and sel['best']}, band {sel and sel['band']}, "
              f"removed by value {sel and sel['removed_by_value']}, candidates {sel and sel['candidates']}, "
              f"fallback {sel and sel['fallback']}; grid end: {out.get('grid_end')}")
        print(f"   {'K':>3} {'r':>4} {'existence':>16} {'value':>16} {'A adj':>7} {'B adj':>7}  gate ghost align conv  r*(K)")
        for t in table:
            print(f"   {t['K']:>3} {t['r']:>4} {t['existence']:.4f} ± {t['se_existence']:.4f} {t['value']:.4f} ± {t['se_value']:.4f}"
                  f" {t['track_A_adjusted']:7.3f} {t['track_B_adjusted']:7.3f}  {int(t['gate_fails']):>4} {int(t['ghost_fails']):>5}"
                  f" {int(t['alignment_fails']):>5} {int(t['convergence_fails']):>4}  {'*' if t['chosen_r_at_K'] else ''}")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--agg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--readings", default=None)
    a = ap.parse_args()
    rep = run(json.load(open(a.agg)), json.load(open(a.readings)) if a.readings else None)
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


if __name__ == "__main__":
    main()
