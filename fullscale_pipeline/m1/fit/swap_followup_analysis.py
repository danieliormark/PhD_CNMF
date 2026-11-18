# Swap follow-ups (ticket 107; prepared RL-254, submitted RL-256, read 2026-11-18). Saved fits only, runs on incline.
# Background: the bare graded swap (RL-248, aligned RL-261) gave "no change" in most settings. Two explanations of
# "no change" were to be separated:
#   (1) the swap keeps a link between people and ideas through the social-degree bands (articles exchange their social
#       ties only with articles of the same social-degree band, and degree band and semantic community are linked);
#   (2) fixed K: at K* the model has no spare communities to separate the domains.
# Arms, all at p = 1 (social side fully swapped), hold-out group 1, seeds 1–10, terms off:
#   (i)  one band for all articles (swap_single_band) at K*           — removes explanation 1;
#   (ii) degree bands at the next coarse K above K* (SE 32, KL 20)     — tests explanation 2.
# Comparisons (decided swap measures and margin, step 1 of the decided rule; only two p values, so no reversal check):
#   (i)  one band p = 1  vs  p = 0 at K*  (pass-1 fits);  beside it: banded p = 1 vs p = 0 at K* (the bare swap);
#        and one band vs banded at p = 1 (does removing the bands add to the effect?);
#   (ii) banded p = 1 at the larger K  vs  p = 0 at the larger K (pass-1 fits).
# Verdict per comparison as in swap_analysis.read: any measure beyond its margin in the opposite direction →
# "conflicting"; else any counting effect → "heterogeneity follows the data"; else "no change". weaker_median counts only
# if ratio_median also has an effect. Margin = √((δ_a² + δ_b²)/2), δ from 199 random halvings of each arm's seeds.
# Reported beside (owner 2026-11-18, reporting only): child×article requirement per community (failing communities per
# fit; fits in which every community passes) and the diagonal rule share per relation (tie weight in communities whose
# own-numbered cell exceeds every other cell of its row and column). Presence readings and the two added readings follow
# the numbering set by M1_ALIGNED (run once with M1_ALIGNED=1 and once without).
# Usage: [M1_ALIGNED=1] python -m fit.swap_followup_analysis --out results/stage0/[aligned/]swap_followup.json
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np
from . import data as D, measures as M, worlds as Wd
from .aggregate import load_fit, load_UZ, latest_fits, ALIGNED
from .swap_analysis import attribution, social_band, cramers_v, margin, DIRECTION, REPORTED

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
KBIG = {2: 32, 1: 20}
LOSS = {2: "squared_error", 1: "KL"}


def rule_share(P):
    """Owner's diagonal rule (as start_test_analysis.diagonal_rule, which is not imported because that module switches
    on the aligned numbering): tie weight (half credit) in communities whose part(k,k) exceeds every other cell of its
    row and column, ÷ the relation's total."""
    off = P.copy(); np.fill_diagonal(off, -np.inf); d = np.diag(P)
    ok = (d > off.max(1)) & (d > off.max(0)); h = 0.5 * (P.sum(1) + P.sum(0)) / P.sum()
    return float(h[ok].sum())


def collect():
    """{(approach, beta): {arm: [fits]}}; arms: p0_Kstar, banded_Kstar, oneband_Kstar, p0_Kbig, banded_Kbig."""
    out = defaultdict(lambda: defaultdict(list))
    for stage in ("stage0_pass1", "swap_bare"):
        fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, stage, "*", "measures.json"))]
        for f in latest_fits(fs):
            j = f["job"]; arm = j.get("arm") or {}
            if j["group"] != 0 or j["seed"] not in range(1, 11) or (j.get("terms") or {}) or not f["result"]["converged"]:
                continue
            if j.get("social_link_start"):
                continue
            p = float(arm.get("swap_p", 0.0)); K = j["K"]; b = j["beta"]
            if stage == "stage0_pass1" and (arm or p != 0.0):
                continue
            size = "Kstar" if K == KSTAR[b] else ("Kbig" if K == KBIG[b] else None)
            if size is None:
                continue
            if p == 0.0:
                lab = f"p0_{size}"
            elif p == 1.0:
                lab = ("oneband_" if arm.get("swap_single_band") else "banded_") + size
            else:
                continue
            out[(j["approach"], b)][lab].append(f)
    return out


def per_fit(f, rds_base, weights, rng):
    m = f["measures"]; arm = f["job"].get("arm") or {}
    dp = m["domain_presence"]; S = np.array(dp["semantic"]["mean"]); O = np.array(dp["social"]["mean"])
    share = np.mean([np.array(v["half_credit"]) for v in m["presence"].values()], 0)
    lo, hi = np.minimum(S, O), np.maximum(S, O)
    two = m["heldout_two_part"]; dom = {rd.name: rd.domain for rd in rds_base}
    ex = {}
    for Dn in ("social", "semantic"):
        sub = {r: v for r, v in two.items() if dom.get(r) == Dn and v is not None}
        ex[Dn] = M.combined_two_part(sub, weights)["existence"] if sub else None
    out = {"seed": f["job"]["seed"], "mono_domain": len(m["mono_flags"]["mono_domain"]),
           "mixed_0.1": float(share[(S >= 0.1) & (O >= 0.1)].sum()), "mixed_0.5": float(share[(S >= 0.5) & (O >= 0.5)].sum()),
           "weaker_median": float(np.median(lo)), "ratio_median": float(np.median(lo / np.maximum(hi, 1e-300))),
           "existence_social": ex["social"], "existence_semantic": ex["semantic"]}
    p = float(arm.get("swap_p", 0.0))
    rds = Wd.graded_swap(rds_base, p, seed=f["job"]["seed"], single_band=bool(arm.get("swap_single_band")))[0] if p > 0 else rds_base
    U, Z = load_UZ(f["dir"])
    acc = attribution(U, Z, rds)
    s, t = acc["social"], acc["semantic"]
    both = (s.sum(1) > 0) & (t.sum(1) > 0)
    sn = s[both] / s[both].sum(1, keepdims=True); tn = t[both] / t[both].sum(1, keepdims=True)
    out["overlap"] = float(np.minimum(sn, tn).sum(1).mean())
    band = social_band(rds)[both]; main = tn.argmax(1)
    out["band_link"] = float(cramers_v(band, main) - np.mean([cramers_v(rng.permutation(band), main) for _ in range(100)]))
    rule = {}
    for rd in rds:
        P, _ = M.parts(U, Z, rd)
        if rd.name == "child_article":
            d = np.diag(P); ok = (d / np.maximum(P.sum(1), 1e-300) > 0.5) & (d / np.maximum(P.sum(0), 1e-300) > 0.5)
            out["anchor_failing"] = int((~ok).sum())
        else:
            rule[rd.name] = rule_share(P)
    out["diagonal_rule"] = rule
    return out


def compare(A, B, rng):
    """Step 1 of the decided rule for arm B against arm A (B − A)."""
    st = {}
    for k, sign in DIRECTION.items():
        a = [r[k] for r in A]; b = [r[k] for r in B]
        if not a or not b:
            st[k] = {"status": "missing"}; continue
        eff = float(np.mean(b) - np.mean(a)); da, db = margin(a, rng), margin(b, rng)
        d = math.sqrt(((da or 0) ** 2 + (db or 0) ** 2) / 2)
        s = "opposite" if sign * eff < -d else ("effect" if sign * eff > d else "none")
        st[k] = {"status": s, "effect": eff, "margin": d}
    if st["weaker_median"]["status"] == "effect" and st["ratio_median"]["status"] != "effect":
        st["weaker_median"]["status"] += " (not counted: ratio unchanged, change is size)"
    sts = [v["status"] for v in st.values()]
    verdict = ("incomplete" if "missing" in sts else "conflicting" if "opposite" in sts
               else "heterogeneity follows the data" if "effect" in sts else "no change")
    return {"verdict": verdict, "measures": st}


def run(seed=0):
    rng = np.random.default_rng(seed)
    build = D.Build(D.BUILD); arms = collect(); rep = {}
    for a in (1, 2):
        rds_base = D.fit_data(build, a, group=0); weights = {rd.name: rd.weight for rd in rds_base}
        for b in (2, 1):
            rows = {lab: [per_fit(f, rds_base, weights, rng) for f in sorted(fs, key=lambda f: f["job"]["seed"])]
                    for lab, fs in arms[(a, b)].items()}
            keys = list(DIRECTION) + list(REPORTED) + ["anchor_failing"]
            means = {lab: {k: (float(np.mean([r[k] for r in R if r[k] is not None])) if R else None) for k in keys} for lab, R in rows.items()}
            for lab, R in rows.items():
                means[lab]["anchor_all_pass_fits"] = int(sum(r["anchor_failing"] == 0 for r in R))
                means[lab]["diagonal_rule"] = {rel: float(np.median([r["diagonal_rule"][rel] for r in R])) for rel in R[0]["diagonal_rule"]} if R else None
                means[lab]["fits"] = len(R)
            g = rows.get
            cmp = {"oneband_vs_p0_Kstar": compare(g("p0_Kstar", []), g("oneband_Kstar", []), rng),
                   "banded_vs_p0_Kstar": compare(g("p0_Kstar", []), g("banded_Kstar", []), rng),
                   "oneband_vs_banded_Kstar": compare(g("banded_Kstar", []), g("oneband_Kstar", []), rng),
                   "banded_vs_p0_Kbig": compare(g("p0_Kbig", []), g("banded_Kbig", []), rng)}
            name = f"approach_{a}_{LOSS[b]}"
            rep[name] = {"K": {"Kstar": KSTAR[b], "Kbig": KBIG[b]}, "means": means, "comparisons": cmp, "per_fit": rows}
            print(name, {k: v["verdict"] for k, v in cmp.items()}, flush=True)
    rep["_numbering"] = "aligned" if ALIGNED else "fitted"
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
