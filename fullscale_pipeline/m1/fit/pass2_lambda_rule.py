# Stage-0 pass 2: the λ_child rule per loss and approach (decided 2026-11-14, rule b; largest λ kept 2026-11-16).
# Candidates: λ = 0 (pass-1 aggregate at K*) and the pass-2 grid (squared error r 0.003/0.01/0.03, KL 0.03/0.1/0.3).
# 1. best = highest held-out existence (within-class AP, combined; mean over hold-out groups of the group's seed mean);
#    band = candidates with existence ≥ best − SE(best) (SE = SD of group means / √groups).
# 2. a band candidate is removed if its value (partial Spearman) < value(best) − SE_value(best).
# 3. the largest remaining λ whose stability is not worse than at λ = 0 is chosen. "Not worse" (DECIDED 2026-11-17,
#    owner: "I agree with the proposal ii"): the stability gate does not fail at fraction 0.2 when λ = 0 passes it, and
#    neither Track A nor Track B (chance-adjusted, mean over groups) is below λ = 0's by more than λ = 0's SE over groups
#    (reference SE, as the band, the value guard and C1). Paired per-group differences are reported, not used.
# Highest-grid-value rule (2026-11-14): KL at r = 0.3 → fit r = 1 (owner told first). No lowest-grid-value rule (2026-11-16).
# Usage: python -m fit.pass2_lambda_rule --pass1 P1.json --pass2 P2.json --out OUT.json
import argparse, json, math
import numpy as np

KSTAR = {2: 20, 1: 13}


def stab(entry):
    out = {}
    for t in ("A_adjusted", "B_adjusted"):
        v = [g["stability"][t] for g in entry["groups"] if g.get("stability")]
        out[t] = (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, len(v))
    out["gate_fails_0.2"] = bool((entry.get("gate_fails") or {}).get("0.2", {}) and entry["gate_fails"]["0.2"]["K_fails"])
    return out


def rule(cands):
    """cands: {r: entry}; r = 0.0 is λ = 0."""
    best = max(cands, key=lambda r: cands[r]["existence"])
    b = cands[best]
    band = [r for r in cands if cands[r]["existence"] >= b["existence"] - b["se_existence"]]
    kept = [r for r in band if not (cands[r]["value"] is not None and b["value"] is not None
                                    and cands[r]["value"] < b["value"] - b["se_value"])]
    s0 = stab(cands[0.0])
    st = {}
    for r in cands:
        s = stab(cands[r])
        worse = {t: s[t][0] < s0[t][0] - (s0[t][1] or 0.0) for t in ("A_adjusted", "B_adjusted")}
        gate_worse = s["gate_fails_0.2"] and not s0["gate_fails_0.2"]
        st[r] = {"stability": s, "worse": worse, "gate_worse": gate_worse,
                 "passes": not (gate_worse or any(worse.values()))}
    ok = [r for r in kept if st[r]["passes"]]
    chosen = max(ok) if ok else 0.0
    return {"best_by_existence": best, "band": sorted(band), "kept_after_value": sorted(kept),
            "passing_stability": sorted(ok), "chosen_r": chosen, "stability": st,
            "scores": {r: {k: cands[r][k] for k in ("existence", "se_existence", "value", "se_value", "n_groups_scored")}
                       for r in sorted(cands)}}


def _nonchild_by_group(entry):
    """Per hold-out group: mean of the per-facet Track B (seed pairs) over facets other than child hyperedges."""
    return {g["group"]: float(np.mean([v for f, v in g["stability"]["per_facet_B"].items() if f != "child"]))
            for g in entry["groups"] if g.get("stability")}


def rule_r2(cands):
    """λ rule R2 (owner 2026-11-21: "I approve R2 with value reported only"): (1) band = r values whose existence is within
    one SE of the best r's; value is reported, removes nothing; (2) the stability guard as before (gate not failing where
    λ = 0 passes; Track A / B not below λ = 0's by more than λ = 0's SE); (3) among the band r passing the guard, the most
    stable on the facets other than child hyperedges (where the term forces agreement) is chosen if its paired difference
    (by hold-out group) to the largest such r exceeds 2 SE; otherwise the largest r."""
    base = rule(cands)
    best = max(cands, key=lambda r: cands[r]["existence"]); b = cands[best]
    band = sorted(r for r in cands if cands[r]["existence"] >= b["existence"] - b["se_existence"])
    ok = [r for r in band if base["stability"][r]["passes"]]
    out = {"rule": "R2", "best_by_existence": best, "band": band, "passing_stability": ok, "scores": base["scores"],
           "stability": base["stability"], "value_reported": {r: cands[r]["value"] for r in band}}
    if not ok:
        out["chosen_r"] = 0.0; return out
    top = max(ok); nc = {r: _nonchild_by_group(cands[r]) for r in ok}
    mean = {r: float(np.mean(list(v.values()))) for r, v in nc.items()}
    cand = max(ok, key=lambda r: mean[r]); out["nonchild_stability"] = mean
    if cand != top:
        gs = sorted(set(nc[cand]) & set(nc[top])); d = np.array([nc[cand][g] - nc[top][g] for g in gs])
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
        out["paired_vs_largest"] = {"difference": float(d.mean()), "se": se}
        out["chosen_r"] = cand if d.mean() > 2 * se else top
    else:
        out["chosen_r"] = top
    return out


def rule_r2c(cands, ca_gain):
    """λ rule R2, tie-break variant C (owner 2026-11-24: "I agree with option C"; RL-373): band and stability guard as R2;
    among the band r passing the guard, the r with the highest held-out child×article gain (within-class AP minus the
    degree-only reference, mean over hold-out groups); every r whose gain is not below that peak by more than 1 paired SE
    (by hold-out group) is tied with it; the smallest tied r is chosen (least flattening of word-tied child hyperedges,
    RL-364). ca_gain: {r: {group: gain}}. Reported beside it: the amended choice (recommendation, not decided) — never r 0
    when r 0 is tied with an r > 0 (then the smallest tied r > 0), and R2's own choice."""
    out = rule_r2(cands); out["rule"] = "R2C"; out["chosen_r_R2"] = out["chosen_r"]
    ok = out["passing_stability"]
    if not ok:
        return out
    mean = {r: float(np.mean(list(ca_gain[r].values()))) for r in ok}
    peak = max(ok, key=lambda r: mean[r])

    def below(r):
        gs = sorted(set(ca_gain[peak]) & set(ca_gain[r])); d = np.array([ca_gain[peak][g] - ca_gain[r][g] for g in gs])
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
        return float(d.mean()) > se
    tied = sorted(r for r in ok if r == peak or not below(r))
    out.update({"child_article_gain": mean, "peak_child_article_gain": peak, "tied_with_peak": tied, "chosen_r": tied[0]})
    pos = [r for r in tied if r > 0]
    out["chosen_r_amended"] = pos[0] if tied[0] == 0.0 and pos else tied[0]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pass1", required=True); ap.add_argument("--pass2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p1 = json.load(open(a.pass1)); p2 = json.load(open(a.pass2))
    rep = {}
    for appr in (1, 2):
        for beta in (2, 1):
            c1 = next((v for k, v in p1.items() if not k.startswith("_") and v["cell"]["approach"] == appr and v["cell"]["beta"] == beta), None)
            if c1 is None:                                  # study B: KL only
                continue
            cands = {0.0: c1["per_K"][str(KSTAR[beta])]}
            for k, v in p2.items():
                if k.startswith("_") or v["cell"]["approach"] != appr or v["cell"]["beta"] != beta:
                    continue
                cands[float(v["cell"]["arm"]["r"])] = v["per_K"][str(KSTAR[beta])]
            name = f"approach_{appr}_{'KL' if beta == 1 else 'squared_error'}"
            rep[name] = rule(cands)
            if beta == 1 and rep[name]["chosen_r"] == 0.3:
                rep[name]["highest_grid_value_rule"] = "r = 0.3 chosen: fit r = 1 (tell the owner first)"
            r = rep[name]
            print(name, "chosen r =", r["chosen_r"], "| best by existence", r["best_by_existence"], "| band", r["band"],
                  "| after value", r["kept_after_value"], "| stability ok", r["passing_stability"])
            for rr, s in r["scores"].items():
                sb = r["stability"][rr]
                print(f"   r={rr:<6} existence {s['existence']:.4f} ± {s['se_existence']:.4f}  value {s['value']:.4f} ± {s['se_value']:.4f}"
                      f"  A {sb['stability']['A_adjusted'][0]:.3f}  B {sb['stability']['B_adjusted'][0]:.3f}  gate fails {sb['stability']['gate_fails_0.2']}"
                      f"  stability passes {sb['passes']}  groups {s['n_groups_scored']}")
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
