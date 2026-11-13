# Stage-0 loss comparison (ticket 107): squared error (β = 2) vs generalised KL (β = 1), per approach, at the candidate K,
# by the pre-registered measures and decision rule (plans/ticket107-m1-chunk12v3-design.md, 2026-11-05/06, S0-M2 revised
# 2026-11-13 option B). All Δ are KL minus squared error unless stated; classes over hold-out groups:
#   first better   : 95% interval of the mean Δ above 0 and mean Δ ≥ δ        (here "KL better" / "KL more stable" ...)
#   second better  : 95% interval below 0 and mean Δ ≤ −δ
#   equivalent     : two one-sided tests at 5% (the 90% interval within ±δ)
#   inconclusive   : otherwise (also with fewer than 2 groups)
#   δ (noise margin): per loss, 95th percentile over 999 random splits of its seeds into halves (within every group) of
#                     |mean over groups of (half₁ − half₂)|, divided by √2 (2026-11-05); the two losses' margins combined
#                     as their root-mean-square (2026-11-13). S0-M1: the decided joint split (below).
# Measures:
#   S0-M1 membership agreement: D_g = W̄_g (mean chance-adjusted Track score over within-loss seed pairs, both losses
#         pooled) − C̄_g (over cross-loss pairs); "same" (equivalent) needs both tracks, "different" (D ≥ δ, interval > 0)
#         either track; δ from halves of W̄.
#   S0-M2 held-out (option B): existence (within-class AP) and value (partial Spearman), measures.combined_two_part with
#         the approach's weights; KL better = KL better on existence and value not favouring squared error.
#   S0-M3 cross-scoring: held-out squared error and held-out KL of every fit (lower better), Δ = score(squared-error
#         fit) − score(KL fit) per score; split / dominance / weak support.
#   S0-M4 seed stability: W̄ per loss and track; Δ = W̄_KL − W̄_squared error; more stable if one track says so and the
#         other does not say the opposite.
#   per-entity fit: training AUC of child×article rows and columns by tie-count band (2, 3–5, 6+), compared as above.
# Decision rule (2026-11-06): (1) gate — a loss failing the stability gate at the majority of candidate K is excluded
# (both → owner); (2) S0-M1 same → squared error; (3) S0-M2 winner; (4) S0-M4 more stable; else squared error;
# majority over candidate K, a tie → squared error (default); (5) flags, never reversals.
# Details (2026-11-13): S0-M1's margin splits both losses' seeds jointly (as decided 2026-11-05); candidate K = union of
# both losses' one-SE bands (decided 2026-11-05), all commonly scored K if none; Track A's placed entities pooled over
# both losses' seeds of the group (S0-M1, owner approved) or per loss (S0-M4); the per-entity-fit flag classifies bands
# 3–5 and 6+ separately (owner approved); periphery flag = sparser entity with 1 visible tie (2 ties, one hidden: the
# measurable periphery; owner 2026-11-13); margins of S0-M2/M3/M4 and the per-entity fit = RMS of the two losses' own
# half-split margins (owner 2026-11-13).
# Usage: python -m fit.stage0_compare --root ROOT --stage STAGE --aggregate AGG.json [--filter JSON] --out OUT.json
import argparse, itertools, json, math, os
from collections import defaultdict
import numpy as np
from scipy import stats

from . import measures as M
from .aggregate import load_fit, load_UZ, load_entity, cell_key

N_SPLITS = 999
LOSS_NAME = {2: "squared error", 1: "KL"}


# ------------------------------------------------------------------------------------------------ classes and margins
def classify(deltas, delta, first="KL", second="squared error"):
    """deltas: Δ per hold-out group; delta: noise margin. Returns class, mean, 95% and 90% intervals."""
    d = np.asarray([x for x in deltas if x is not None and np.isfinite(x)], float)
    out = {"mean": float(d.mean()) if len(d) else None, "groups": int(len(d)), "delta": delta}
    if len(d) < 2 or delta is None:
        out.update({"class": "inconclusive", "ci95": None, "ci90": None}); return out
    se = d.std(ddof=1) / math.sqrt(len(d)); m = d.mean()
    t95, t90 = stats.t.ppf(0.975, len(d) - 1), stats.t.ppf(0.95, len(d) - 1)
    ci95, ci90 = (m - t95 * se, m + t95 * se), (m - t90 * se, m + t90 * se)
    if ci95[0] > 0 and m >= delta:
        c = first + " better"
    elif ci95[1] < 0 and m <= -delta:
        c = second + " better"
    elif ci90[0] > -delta and ci90[1] < delta:
        c = "equivalent"
    else:
        c = "inconclusive"
    out.update({"class": c, "ci95": [float(x) for x in ci95], "ci90": [float(x) for x in ci90]})
    return out


def half_margin(per_loss, stat, rng, n_splits=N_SPLITS):
    """per_loss: {loss: {group: list of per-seed items}}; stat(items) → number (mean of a measure, or W̄ over the pairs
    inside the half). δ = 95th percentile of |mean_g(stat(half₁) − stat(half₂))| over splits, losses pooled, ÷ √2."""
    vals = []
    for loss, byg in per_loss.items():
        for _ in range(n_splits):
            diffs = []
            for g, items in byg.items():
                if len(items) < 2:
                    continue
                p = rng.permutation(len(items)); h = len(items) // 2
                a, b = stat([items[i] for i in p[:h]]), stat([items[i] for i in p[h:2 * h]])
                if a is not None and b is not None:
                    diffs.append(a - b)
            if diffs:
                vals.append(abs(np.mean(diffs)))
    return float(np.percentile(vals, 95) / math.sqrt(2)) if vals else None


# ----------------------------------------------------------------------------------------------- pairwise Track scores
def rms_margin(margins):
    """Margin for Δ = mean(loss 1) − mean(loss 2) (owner 2026-11-13): root-mean-square of the two losses' own half-split
    margins. Each δ_loss ≈ 1.96·√(2·V_loss) (V = noise variance of that loss's mean), so √((δ₁² + δ₂²)/2) =
    1.96·√(V₁ + V₂), the spread of a difference of two independent means (standard error of a difference,
    √(SE₁² + SE₂²)) — exact also when the losses differ in noise."""
    m = [x for x in margins if x is not None]
    return float(math.sqrt(sum(x * x for x in m) / len(m))) if len(m) == 2 else None


def loss_margin(per, stat, rng):
    """RMS of each loss's own half-split margin (half_margin applied to one loss at a time)."""
    return rms_margin([half_margin({b: per[b]}, stat, rng) for b in per])


def pair_scores(wms, placed, rng):
    """Chance-adjusted Track A and B for every pair of the given fits' weighted memberships."""
    out = {}
    for i, j in itertools.combinations(range(len(wms)), 2):
        real = M.track_pair(wms[i], wms[j], placed)
        ch = M.track_pair(wms[i], M._shuffled(wms[j], rng), placed)
        out[(i, j)] = {t: (real[t + "_model"] - ch[t + "_model"]) / (1 - ch[t + "_model"]) for t in ("A", "B")}
    return out


def mean_pairs(pairs, idx_a, idx_b=None, track="A"):
    """Mean over pairs within idx_a (idx_b None) or between idx_a and idx_b."""
    if idx_b is None:
        v = [pairs[tuple(sorted(p))][track] for p in itertools.combinations(idx_a, 2)]
    else:
        v = [pairs[tuple(sorted((i, j)))][track] for i in idx_a for j in idx_b]
    return float(np.mean(v)) if v else None


def placed_of(fits):
    """Track A's entity set for one group (2026-11-07 rule applied to the given fits): placed if the mean reproduced share
    over the fits minus the degree-only share exceeds the entity's SD over the fits. S0-M1 passes both losses' seeds
    pooled (owner approved 2026-11-13; one entity set for within-loss and cross-loss pairs, so W̄ − C̄ compares like
    with like and neither loss defines it; limitations: the SD includes the between-loss difference, so an entity the
    losses reproduce differently is less likely to count as placed, and entities placed under one loss only can drop
    out — Track B, over all entities, covers them); S0-M4 passes one loss's seeds."""
    ents = [load_entity(f["dir"]) for f in fits]
    return {fa: M.placed_entities([e[fa]["reproduced_share"] for e in ents], ents[0][fa]["reproduced_share_degree_only"])
            for fa in ents[0]}


# ------------------------------------------------------------------------------------------------- one approach, one K
def compare_K(byloss, rng):
    """byloss: {beta: {group: [converged fits]}} for one approach and K. Returns S0-M1–M4 and the per-entity fit."""
    groups = sorted(set(byloss[2]) & set(byloss[1]))
    res = {"groups": groups}
    # ---- S0-M2 (option B) and S0-M3: per-fit numbers
    getm = lambda f, *ks: _get(f["measures"], ks)
    m2 = {}
    for part in ("existence", "value"):
        per = {b: {g: [getm(f, "heldout_two_part_combined", part) for f in byloss[b][g]] for g in groups} for b in (2, 1)}
        d = [_mean(per[1][g]) - _mean(per[2][g]) for g in groups if _mean(per[1][g]) is not None and _mean(per[2][g]) is not None]
        m2[part] = classify(d, loss_margin(per, _mean, rng))
    rep = {}
    for rel in ("child_article", "parent_child", "author_affiliation", "article_affiliation"):
        for key in ("1", "2", "3+"):
            per = {b: {g: [getm(f, "heldout_two_part", rel, "ap_within_class_by_sparser_ties", key) for f in byloss[b][g]] for g in groups} for b in (2, 1)}
            if any(v is not None for b in per for g in per[b] for v in per[b][g]):
                d = [_mean(per[1][g]) - _mean(per[2][g]) for g in groups if _mean(per[1][g]) is not None and _mean(per[2][g]) is not None]
                rep[f"{rel}_ap_within_class_sparser_{key}"] = classify(d, loss_margin(per, _mean, rng))
        for band in ("2", "3-5", "6+"):
            per = {b: {g: [getm(f, "heldout", rel, "auc_by_band", band, "auc") for f in byloss[b][g]] for g in groups} for b in (2, 1)}
            if any(v is not None for b in per for g in per[b] for v in per[b][g]):
                rep[f"{rel}_auc_band_{band}"] = {LOSS_NAME[b]: _mean([_mean(per[b][g]) for g in groups]) for b in (2, 1)}
        for key in ("auc", "ap", "baseline_auc", "baseline_ap", "baseline_ap_within_class", "value_spearman"):
            per = {b: [getm(f, "heldout_two_part", rel, key) for g in groups for f in byloss[b][g]] for b in (2, 1)}
            if any(v is not None for b in per for v in per[b]):
                rep[f"{rel}_{key}"] = {LOSS_NAME[b]: _mean(per[b]) for b in (2, 1)}
    m2["reported"] = rep
    kl_e, kl_v = m2["existence"]["class"], m2["value"]["class"]
    m2["verdict"] = ("KL better" if kl_e == "KL better" and kl_v != "squared error better" else
                     "squared error better" if kl_e == "squared error better" and kl_v != "KL better" else
                     ("equivalent" if kl_e == "equivalent" else "inconclusive"))
    res["S0-M2 held-out (existence and value)"] = m2
    m3 = {}
    for score in ("squared_error", "kl_error"):
        per = {b: {g: [getm(f, "heldout_combined", score) for f in byloss[b][g]] for g in groups} for b in (2, 1)}
        d = [_mean(per[2][g]) - _mean(per[1][g]) for g in groups]      # lower is better: Δ < 0 → squared-error fit better
        c = classify(d, loss_margin(per, _mean, rng), first="KL", second="squared error")
        m3[score] = c          # Δ = score(squared-error fit) − score(KL fit) > 0 means the KL fit scores lower = better
    wins = {s: ("KL" if m3[s]["class"] == "KL better" else "squared error" if m3[s]["class"] == "squared error better" else None) for s in m3}
    eqs = {s: m3[s]["class"] == "equivalent" for s in m3}
    if wins["squared_error"] and wins["squared_error"] == wins["kl_error"]:
        m3["reading"] = "dominance: " + wins["squared_error"]
    elif wins["squared_error"] == "squared error" and wins["kl_error"] == "KL":
        m3["reading"] = "split (each wins its own score): no weight"
    elif any(wins.values()) and any(eqs.values()):
        m3["reading"] = "weak support: " + next(w for w in wins.values() if w)
    else:
        m3["reading"] = "no reading (inconclusive)"
    res["S0-M3 cross-scoring"] = m3
    # ---- per-entity fit (loss-neutral training AUC by band)
    pe = {}
    for side in ("row", "col"):
        for band in ("2", "3-5", "6+"):
            per = {b: {g: [getm(f, "training_auc_by_band", f"child_article_{side}", band, "mean") for f in byloss[b][g]] for g in groups} for b in (2, 1)}
            d = [_mean(per[1][g]) - _mean(per[2][g]) for g in groups if _mean(per[1][g]) is not None and _mean(per[2][g]) is not None]
            pe[f"child_article_{side}_band_{band}"] = classify(d, loss_margin(per, _mean, rng))
    res["per-entity fit (training AUC)"] = pe
    # ---- S0-M1 and S0-M4: Track scores over seed pairs
    m1 = {t: [] for t in ("A", "B")}; m4 = {t: [] for t in ("A", "B")}
    halves1 = {}; halves4 = {}
    for g in groups:
        fs = byloss[2][g] + byloss[1][g]; n2 = len(byloss[2][g])
        if len(byloss[2][g]) < 2 or len(byloss[1][g]) < 2:
            continue
        wms = [M.weighted_memberships(*load_UZ(f["dir"]), {r: tuple(v) for r, v in f["result"]["relations"].items()}) for f in fs]
        i2, i1 = list(range(n2)), list(range(n2, len(fs)))
        pooled = pair_scores(wms, placed_of(fs), rng)
        own = {b: pair_scores([wms[i] for i in idx], placed_of([fs[i] for i in idx]), rng) for b, idx in ((2, i2), (1, i1))}
        for t in ("A", "B"):
            W = np.mean([mean_pairs(pooled, i2, track=t), mean_pairs(pooled, i1, track=t)])
            m1[t].append(W - mean_pairs(pooled, i2, i1, track=t))
            m4[t].append(mean_pairs(own[1], list(range(len(i1))), track=t) - mean_pairs(own[2], list(range(len(i2))), track=t))
        halves1[g] = (pooled, i2, i1); halves4[g] = own
    out1, out4 = {}, {}
    for t in ("A", "B"):
        # δ for S0-M1 as decided (2026-11-05): each loss's seeds split into halves; half h = that half of both losses;
        # W̄_half = mean over within-loss pairs inside the half (both losses pooled); |mean_g(W̄_half1 − W̄_half2)|
        vals = []
        for _ in range(N_SPLITS):
            diffs = []
            for g, (pooled, i2, i1) in halves1.items():
                if len(i2) < 4 or len(i1) < 4:
                    continue
                p2, p1 = rng.permutation(i2), rng.permutation(i1); h2, h1 = len(i2) // 2, len(i1) // 2
                def wbar(a2, a1):
                    v = [pooled[tuple(sorted(p))][t] for idx in (a2, a1) for p in itertools.combinations(idx, 2)]
                    return float(np.mean(v))
                diffs.append(wbar(list(p2[:h2]), list(p1[:h1])) - wbar(list(p2[h2:2 * h2]), list(p1[h1:2 * h1])))
            if diffs:
                vals.append(abs(np.mean(diffs)))
        d1 = float(np.percentile(vals, 95) / math.sqrt(2)) if vals else None
        out1[t] = classify(m1[t], d1, first="different", second="(negative D)")
        per_loss = []
        for b in (2, 1):
            vals = []
            for _ in range(N_SPLITS):
                diffs = []
                for g, own in halves4.items():
                    n = len(byloss[b][g])
                    if n < 4:
                        continue
                    p = rng.permutation(n); h = n // 2
                    diffs.append(mean_pairs(own[b], list(p[:h]), track=t) - mean_pairs(own[b], list(p[h:2 * h]), track=t))
                if diffs:
                    vals.append(abs(np.mean(diffs)))
            per_loss.append(float(np.percentile(vals, 95) / math.sqrt(2)) if vals else None)
        d4 = rms_margin(per_loss)
        out4[t] = classify(m4[t], d4, first="KL more stable", second="squared error more stable")
    out1["verdict"] = ("same" if out1["A"]["class"] == "equivalent" and out1["B"]["class"] == "equivalent" else
                       "different" if "different better" in (out1["A"]["class"], out1["B"]["class"]) else "inconclusive")
    cl = (out4["A"]["class"], out4["B"]["class"])
    if "KL more stable better" in cl and "squared error more stable better" not in cl:
        out4["verdict"] = "KL more stable"
    elif "squared error more stable better" in cl and "KL more stable better" not in cl:
        out4["verdict"] = "squared error more stable"
    elif all(c == "equivalent" for c in cl):
        out4["verdict"] = "equivalent"
    else:
        out4["verdict"] = "inconclusive"
    res["S0-M1 membership agreement"] = out1
    res["S0-M4 seed stability"] = out4
    return res


def decide_K(res, gate_fail):
    """Steps 2–4 at one K (the gate, step 1, is applied over the candidate K in decide()). Returns loss and the step."""
    if res["S0-M1 membership agreement"]["verdict"] == "same":
        return 2, "S0-M1 same → squared error"
    v = res["S0-M2 held-out (existence and value)"]["verdict"]
    if v == "KL better":
        return 1, "S0-M2 KL better"
    if v == "squared error better":
        return 2, "S0-M2 squared error better"
    s = res["S0-M4 seed stability"]["verdict"]
    if s == "KL more stable":
        return 1, "S0-M4 KL more stable"
    if s == "squared error more stable":
        return 2, "S0-M4 squared error more stable"
    return 2, "default (no step decides) → squared error"


def decide(perK, gate):
    """perK: {K: compare_K result}; gate: {K: {2: fails, 1: fails}}. Majority over candidate K (tie → squared error)."""
    Ks = sorted(perK)
    fails = {b: sum(gate[K][b] for K in Ks) for b in (2, 1)}
    excluded = [b for b in (2, 1) if fails[b] > len(Ks) / 2]
    out = {"candidate_K": Ks, "gate_fails_count": {LOSS_NAME[b]: fails[b] for b in (2, 1)}, "per_K": {}, "flags": []}
    if len(excluded) == 2:
        out.update({"loss": None, "reason": "both losses fail the stability gate at the majority of candidate K → owner"}); return out
    for K in Ks:
        b, why = decide_K(perK[K], gate[K])
        out["per_K"][K] = {"loss": LOSS_NAME[b], "step": why}
    if excluded:
        keep = 1 if excluded == [2] else 2
        out.update({"loss": LOSS_NAME[keep], "reason": f"{LOSS_NAME[excluded[0]]} fails the stability gate at the majority of candidate K"})
    else:
        votes = [1 if out["per_K"][K]["loss"] == "KL" else 2 for K in Ks]
        n1 = votes.count(1)
        chosen = 1 if n1 > len(votes) / 2 else 2
        out.update({"loss": LOSS_NAME[chosen], "reason": f"majority over candidate K ({n1} of {len(votes)} for KL; tie → squared error)"})
    chosen = 1 if out["loss"] == "KL" else 2
    for K in Ks:
        r = perK[K]
        reading = r["S0-M3 cross-scoring"]["reading"]
        if reading.startswith("dominance") and LOSS_NAME[chosen] not in reading:
            out["flags"].append(f"K={K}: cross-scoring dominance against the choice ({reading})")
        elif reading.startswith("dominance"):
            out["flags"].append(f"K={K}: cross-scoring dominance in favour (strengthens; reported)")
        for key, c in r["per-entity fit (training AUC)"].items():
            if key.endswith(("3-5", "6+")) and c["class"] == ("squared error better" if chosen == 1 else "KL better"):
                out["flags"].append(f"K={K}: chosen loss's per-entity fit worse beyond δ on {key}")
        if chosen == 1:
            rep = r["S0-M2 held-out (existence and value)"]["reported"]
            three = [c["class"] for k, c in rep.items() if k.endswith("_sparser_3+") and isinstance(c, dict) and "class" in c]
            periph = [c["class"] for k, c in rep.items() if k.endswith("_sparser_1") and isinstance(c, dict) and "class" in c]
            if "KL better" in periph and "KL better" not in three:
                out["flags"].append(f"K={K}: KL's advantage lies only in the measurable periphery (sparser entity with 1 visible tie = 2 ties, one hidden): owner's hypothesis")
    return out


# --------------------------------------------------------------------------------------------------------- helpers
def _get(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def run(root, stage, aggregate_path, filt=None, seed=0, fraction="0.2"):
    rng = np.random.default_rng(seed)
    base = os.path.join(root, stage)
    agg = json.load(open(aggregate_path)) if aggregate_path else {}
    fits = [load_fit(os.path.join(base, d)) for d in sorted(os.listdir(base)) if os.path.exists(os.path.join(base, d, "measures.json"))]
    if filt:
        fits = [f for f in fits if all(f["job"].get(k) == v for k, v in filt.items())]
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))   # approach → K → beta → group
    cellk = {}
    for f in fits:
        if f["result"]["converged"]:
            j = f["job"]; by[j["approach"]][j["K"]][j["beta"]][j["group"]].append(f); cellk[(j["approach"], j["beta"])] = cell_key(j)
    report = {}
    for a, byK in sorted(by.items()):
        common = [K for K in sorted(byK) if 2 in byK[K] and 1 in byK[K]]
        # candidate K = coarse K inside the one-SE band of either loss (decided 2026-11-05: both losses compared at each
        # loss's band K, verdict by majority); the vetoes enter through the gate (step 1), not here
        cand = set()
        for b in (2, 1):
            sel = (agg.get(cellk.get((a, b)), {}) or {}).get("selection") or {}
            cand |= set(sel.get("band") or [])
        cand = sorted(k for k in cand if k in common) or common
        perK = {K: compare_K(byK[K], rng) for K in cand}
        gate = {}
        for K in cand:
            gate[K] = {}
            for b in (2, 1):
                e = (agg.get(cellk.get((a, b)), {}) or {}).get("per_K", {}).get(str(K), {})
                gf = (e.get("gate_fails") or {}).get(fraction)
                gate[K][b] = bool(gf["K_fails"]) if gf else False
        report[f"approach_{a}"] = {"per_K": perK, "decision": decide(perK, gate)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--stage", required=True)
    ap.add_argument("--aggregate", default=None); ap.add_argument("--filter", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = run(a.root, a.stage, a.aggregate, json.loads(a.filter) if a.filter else None)
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    for k, v in rep.items():
        print(k, json.dumps(v["decision"], indent=1, default=str))


if __name__ == "__main__":
    main()
