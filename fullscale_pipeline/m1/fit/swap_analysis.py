# Bare graded cross-domain swap — reading (ticket 107; decided 2026-11-08/09/16). Saved fits only, runs on incline.
# p = 0: stage0_pass1 fits (hold-out group 1 = index 0, K*, terms off, seeds 1–10); p > 0: stage swap_bare.
# Per fit (counted per fit and averaged over seeds, owner 2026-11-16):
#   mono_domain      number of mono-domain communities (rule 2026-11-07)                       expected to rise with p
#   mixed_0.1        share of rebuilt tie weight (half credit, mean over relation matrices) in communities with
#                    P_k ≥ 0.1 in both domains (DECIDED 2026-11-16)                             expected to fall
#   weaker_median    median over communities of min(P_k(social), P_k(semantic)) (reported)     expected to fall
#   ratio_median     median over communities of min ÷ max of the two presences (reported)      expected to fall
#   reported only:   mixed_0.5; held-out existence (within-class AP) per domain; article overlap (below)
# Reading rule (DECIDED 2026-11-16, corrections a–d): margin δ for comparing two p values = √((δ_p² + δ_q²)/2), each
# δ = 95th percentile over 199 random halvings of the seeds of |mean(half 1) − mean(half 2)| ÷ √2. Step 1: effect of a
# measure = p = 1 vs p = 0 beyond δ in the expected direction; "opposite" = beyond δ the other way. Step 2: a step between
# adjacent p values reversing beyond its δ is a reversal. A measure gives the verdict if it has an effect and no reversal;
# with a reversal it still counts if the effect is ≥ 2δ (flagged), otherwise it is ambiguous. weaker_median counts only
# if ratio_median also has an effect (otherwise the change is size). Verdict: any opposite → "conflicting"; else any
# counting effect → "heterogeneity follows the data"; else any ambiguous → "ambiguous"; else "no change" (read as:
# presence cannot show that heterogeneity follows the data — not as proof that the loss produces it).
# Diagnostics for "no change" (explanation 1: link kept through the social-degree bands; explanation 2: fixed K):
#   overlap          per article, Σ_k min(s_ak, m_ak), s_a / m_a = share of the rebuilt weight on the article's visible
#                    social / semantic ties that runs through community k (article-side reading); mean over articles
#                    with both kinds of ties. 1 = the same communities rebuild its people and its ideas.
#   band_link        Cramér's V between an article's social-degree band and its main semantic community (argmax m_a),
#                    minus the mean over 100 band shuffles; the size of the link the swap cannot remove.
# Usage: python -m fit.swap_analysis --out results/stage0/swap_bare.json
import argparse, glob, json, math, os
from collections import defaultdict
import numpy as np
from . import data as D, measures as M, worlds as Wd
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
PS = (0.0, 0.25, 0.5, 0.75, 1.0)
DIRECTION = {"mono_domain": +1, "mixed_0.1": -1, "weaker_median": -1, "ratio_median": -1}
REPORTED = ("mixed_0.5", "existence_social", "existence_semantic", "overlap", "band_link")


def fits_for(approach, beta):
    out = defaultdict(list)
    for stage in ("stage0_pass1", "swap_bare"):
        fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, stage, "*", "measures.json"))]
        for f in latest_fits(fs):
            j = f["job"]
            if (j["approach"], j["beta"], j["K"], j["group"]) != (approach, beta, KSTAR[beta], 0) or (j.get("terms") or {}):
                continue
            if j["seed"] not in range(1, 11) or not f["result"]["converged"]:
                continue
            p = float((j.get("arm") or {}).get("swap_p", 0.0))
            if stage == "stage0_pass1" and p != 0.0:
                continue
            out[p].append(f)
    return out


def attribution(U, Z, rds):
    """Per article: K-vector of rebuilt weight on its visible social ties and on its visible semantic ties, by the
    article-side community (row side if articles are rows, column side if columns; both for article×article)."""
    n = U["article"].shape[0]; K = U["article"].shape[1]
    acc = {"social": np.zeros((n, K)), "semantic": np.zeros((n, K))}
    for rd in rds:
        A, B, Zr = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
        for side in ("row", "col"):
            if (rd.row_facet if side == "row" else rd.col_facet) != "article":
                continue
            if side == "row":
                C = A[rd.tie_row] * (B[rd.tie_col] @ Zr.T); idx = rd.tie_row
            else:
                C = B[rd.tie_col] * (A[rd.tie_row] @ Zr); idx = rd.tie_col
            tot = C.sum(1, keepdims=True)
            C = np.where(tot > 0, C / np.maximum(tot, 1e-300), 0.0) * rd.tie_val[:, None]
            np.add.at(acc[rd.domain], idx, C)
    return acc


def social_band(rds):
    soc = [rd for rd in rds if rd.domain == "social" and "article" in (rd.row_facet, rd.col_facet)]
    n = next(rd.shape[0] if rd.row_facet == "article" else rd.shape[1] for rd in soc)
    deg = np.zeros(n)
    for rd in soc:
        for side, idx in (("row", rd.tie_row), ("col", rd.tie_col)):
            if (rd.row_facet if side == "row" else rd.col_facet) == "article":
                deg += np.bincount(idx, minlength=n)
    return np.searchsorted(np.array(Wd.SOCIAL_BANDS), deg, side="right") - 1


def cramers_v(x, y):
    xs, xi = np.unique(x, return_inverse=True); ys, yi = np.unique(y, return_inverse=True)
    T = np.zeros((len(xs), len(ys))); np.add.at(T, (xi, yi), 1)
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    chi2 = ((T - E) ** 2 / np.maximum(E, 1e-300)).sum()
    k = min(T.shape) - 1
    return math.sqrt(chi2 / T.sum() / k) if k > 0 else 0.0


def per_fit(f, rds_base, weights, rng):
    m = f["measures"]
    dp = m["domain_presence"]; S = np.array(dp["semantic"]["mean"]); O = np.array(dp["social"]["mean"])
    share = np.mean([np.array(v["half_credit"]) for v in m["presence"].values()], 0)
    lo, hi = np.minimum(S, O), np.maximum(S, O)
    two = m["heldout_two_part"]
    dom = {rd.name: rd.domain for rd in rds_base}
    ex = {}
    for Dn in ("social", "semantic"):
        sub = {r: v for r, v in two.items() if dom.get(r) == Dn and v is not None}
        ex[Dn] = M.combined_two_part(sub, weights)["existence"] if sub else None
    out = {"mono_domain": len(m["mono_flags"]["mono_domain"]),
           "mixed_0.1": float(share[(S >= 0.1) & (O >= 0.1)].sum()), "mixed_0.5": float(share[(S >= 0.5) & (O >= 0.5)].sum()),
           "weaker_median": float(np.median(lo)), "ratio_median": float(np.median(lo / np.maximum(hi, 1e-300))),
           "existence_social": ex["social"], "existence_semantic": ex["semantic"]}
    p = float((f["job"].get("arm") or {}).get("swap_p", 0.0))
    rds = Wd.graded_swap(rds_base, p, seed=f["job"]["seed"])[0] if p > 0 else rds_base
    U, Z = load_UZ(f["dir"])
    acc = attribution(U, Z, rds)
    s, t = acc["social"], acc["semantic"]
    both = (s.sum(1) > 0) & (t.sum(1) > 0)
    sn = s[both] / s[both].sum(1, keepdims=True); tn = t[both] / t[both].sum(1, keepdims=True)
    out["overlap"] = float(np.minimum(sn, tn).sum(1).mean())
    band = social_band(rds)[both]; main = tn.argmax(1)
    v = cramers_v(band, main)
    out["band_link"] = float(v - np.mean([cramers_v(rng.permutation(band), main) for _ in range(100)]))
    return out


def margin(vals, rng, n_split=199):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) < 4:
        return None
    h = len(v) // 2
    d = [abs(v[p[:h]].mean() - v[p[h:2 * h]].mean()) for p in (rng.permutation(len(v)) for _ in range(n_split))]
    return float(np.percentile(d, 95) / math.sqrt(2))


def read(curves, deltas):
    """Decided reading rule for one loss × approach. curves[measure][p] = mean over seeds; deltas[measure][p] = δ_p."""
    status = {}
    rms = lambda a, b: math.sqrt((a ** 2 + b ** 2) / 2) if a is not None and b is not None else None
    for meas, sign in DIRECTION.items():
        c, dl = curves[meas], deltas[meas]
        if c.get(0.0) is None or c.get(1.0) is None:
            status[meas] = {"status": "missing"}; continue
        eff = c[1.0] - c[0.0]; d01 = rms(dl[0.0], dl[1.0]) or 0.0
        rev = []
        for a, b in zip(PS, PS[1:]):
            if c.get(a) is None or c.get(b) is None:
                continue
            step = c[b] - c[a]; dab = rms(dl[a], dl[b]) or 0.0
            if sign * step < -dab:
                rev.append([a, b, step, dab])
        if sign * eff < -d01:
            st = "opposite"
        elif sign * eff > d01:
            st = "effect" if not rev else ("effect, reversal flagged" if sign * eff >= 2 * d01 else "ambiguous")
        else:
            st = "none"
        status[meas] = {"status": st, "effect": eff, "margin": d01, "reversals": rev,
                        "steps": [c[b] - c[a] if c.get(a) is not None and c.get(b) is not None else None for a, b in zip(PS, PS[1:])]}
    if status.get("weaker_median", {}).get("status", "").startswith("effect") and not status.get("ratio_median", {}).get("status", "").startswith("effect"):
        status["weaker_median"]["status"] += " (not counted: ratio unchanged, change is size)"
    sts = [v["status"] for v in status.values()]
    if any(s == "missing" for s in sts):
        verdict = "incomplete (p = 0 or p = 1 has no converged fit)"
    elif any(s == "opposite" for s in sts):
        verdict = "conflicting"
    elif any(s in ("effect", "effect, reversal flagged") for s in sts):
        verdict = "heterogeneity follows the data"
    elif any(s == "ambiguous" for s in sts):
        verdict = "ambiguous"
    else:
        verdict = "no change"
    return {"verdict": verdict, "measures": status}


def run(seed=0):
    rng = np.random.default_rng(seed)
    build = D.Build(D.BUILD)
    rep = {}
    for a in (1, 2):
        rds_base = D.fit_data(build, a, group=0)
        weights = {rd.name: rd.weight for rd in rds_base}
        for beta in (2, 1):
            fp = fits_for(a, beta)
            rows = {p: [per_fit(f, rds_base, weights, rng) for f in sorted(fp.get(p, []), key=lambda f: f["job"]["seed"])] for p in PS}
            keys = list(DIRECTION) + list(REPORTED)
            curves = {k: {p: (float(np.mean([r[k] for r in rows[p] if r[k] is not None])) if rows[p] else None) for p in PS} for k in keys}
            deltas = {k: {p: margin([r[k] for r in rows[p]], rng) for p in PS} for k in keys}
            name = f"approach_{a}_{'KL' if beta == 1 else 'squared_error'}"
            rep[name] = {"fits_per_p": {p: len(rows[p]) for p in PS}, "curves": curves, "margins": deltas,
                         "reading": read(curves, deltas), "per_fit": rows}
            print(name, rep[name]["fits_per_p"], rep[name]["reading"]["verdict"], flush=True)
            for k in keys:
                print(f"   {k:20s}", " ".join("   —  " if curves[k][p] is None else f"{curves[k][p]:6.3f}" for p in PS),
                      "|", rep[name]["reading"]["measures"].get(k, {}).get("status", "reported"), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(), open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))


if __name__ == "__main__":
    main()
