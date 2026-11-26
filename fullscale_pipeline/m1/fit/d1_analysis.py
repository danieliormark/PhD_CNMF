# D1 — journal weighting: the pre-registered analysis of stage 1 (ticket 107; D1 pre-registration
# plans/ticket107_layer2/D1_journal_preregistration.md §2–§8; design doc 2026-11-25/26; RUN_LOG RL-383–404).
# Written before any stage-1 result is read (owner 2026-11-26: "Yes, please start writing the D1 analysis code").
#
# Unit of computation: one (approach, K, hold-out group) with every arm's fits (5 seeds each); converged fits only.
# Arms (plan.STAGE1_ARMS): J1 (no journal matrix, the reference), J2 (binary), J3a / J3b (× 1/4, × 1/8), J4 (idf), J5
# (1/n_j), both J4/J5 with the journal matrix's total restored to J2's (RL-408); robustness arms J2_logdamp, J1_s065, J2_s065.
#
# Candidate K (decided 2026-11-05; stage-0 reading 2026-11-13): per approach, the union over the six journal arms of each
# arm's one-SE band of the decided held-out score (existence, main-setting weights, RL-358): K with group-mean score ≥
# best − SE(best), SE = SD of the 10 group means / √10. Vetoes act through the K protocol later, not here.
#
# D1-M1 (§8): score = semantic-equal held-out existence (within-class AP, mean of child×article and child×parent);
#   Δ_g = mean over Jx's seeds − mean over J1's seeds, per group; δ = 10 % of J1's gain over the degree-only reference
#   (tie weights) — mean over groups of J1's (score − reference) — per approach and K. Intervals: Dunnett with each
#   comparison's own error (owner 2026-11-05): critical values of max |T| over the five arms from a multivariate t with
#   G − 1 df and the correlation of the arms' Δ_g over groups (estimated, Hothorn, Bretz & Westfall 2008); uncorrected t
#   intervals beside. Classes: adds information (95 % interval > 0 and mean ≥ δ), misleads (mirror), redundant
#   (simultaneous 90 % interval inside ±δ: two one-sided tests), inconclusive. Reported: classes at δ 5 % / 20 %, the
#   hold-out half-split margin (J1's group means, all 126 splits of 10 groups, 95th percentile ÷ √2), semantic-equal value
#   (partial Spearman) classified the same way, social held-out gains (within-class AP − reference) per arm.
#   Across candidate K: the class at the majority of candidate K (none → inconclusive; one candidate → its class).
# D1-M2 (RL-394/395): per group the fits of J1 and Jx; agreement of a fit pair = Track A on covered articles (1 − JS
#   distance of weighted memberships after Hungarian matching on the facets both arms share), chance-adjusted
#   (real − chance)/(1 − chance), chance = fit 2's rows shuffled; statistic = mean over groups of (mean cross-arm − mean
#   within-arm agreement); null: arm labels reshuffled within groups (all label splits of each group enumerated, 9,999
#   random combinations across groups), one-sided; Holm across the five arms per approach and K (RL-397). "Changed" =
#   Holm p < 0.05 and drop (within − cross) ≥ 5 % of within (10 % reported). Track A decides; Track B (mass-weighted
#   cosine), model-level Track A/B and co-membership agreement (R2, membership_readings) reported.
# D1-M3 (RL-396): c_r = 1 − L_r(fit) ÷ L_r(ref), L = generalised KL on the fitted cells, ref = degree-only reference
#   (row total × column total ÷ grand total of visible tie values); realised share s_r = w_r ΔL_r ÷ Σ w ΔL; pull share
#   p_r = w_r L_r(ref) ÷ Σ w L(ref). Flag: s_journal ≥ s_semantic (child×article + child×parent); never acts alone.
# D1-M4 (§4.4, RL-397/399/400): per fit, journal-shaped communities = p_k < 0.05 (999 journal-label shuffles among the
#   articles) and j_k ≥ 0.25, j_k = the largest share of community k's article membership mass in one journal (journals ≥ 2
#   trial articles; covered articles with a journal); Δ̄ = mean over groups of (mean count Jx − mean count J1); within-group
#   permutation test, one-sided, Holm across the five arms; "more" = Holm p < 0.05 and Δ̄ ≥ 0.2 (0.5 reported). Reported:
#   journal-associated count, singleton-journal (article-dominated) count, mass reading (share of article membership in
#   journal-shaped communities, Jx − J1), detection limit of Δ̄ (90 % power, one-sided α 0.01 / 0.05) from the observed
#   noise, community size quantiles, per journal-shaped community its journal's articles' memberships (shares ≥ 0.3 /
#   ≥ 0.5 / < 0.1; contribution from member articles).
# Article set for M2 and M4 (owner, RL-410 (a)): per comparison (A, B) and (approach, K, group) — articles covered
#   (coverage ≥ 0.5, RL-345) on average over the fits of A and B only; M4 further restricts it to articles with a journal.
#   Communities of two fits are matched on the facets all arms share, the journal facet excluded in every comparison
#   (RL-410 (b)).
# Decision rule (RL-402), per approach: (1) exclude arms classed misleads; (2) journal problem at a candidate K = M4
#   "more" and M2 "changed"; excluded at ≥ half of the candidate K; reported: share and exact K, per K the share of Jx fits
#   above J1's group mean and the number of groups with Δ_g ≥ 0.2 and drop_g ≥ 5 %; M3 flag beside; (3) among the rest:
#   any adds information → best mean Δ over the candidate K (arms within the mean δ of the best tie → J2 if among them,
#   else highest s_journal averaged over the candidate K); none → J2 if remaining, else highest s_journal; (4) only J1 left
#   → J1 with a post-hoc journal profile; (5) the owner decides after the text check.
# Robustness readings (RL-403/404), per approach and K: J2_logdamp vs J2, J1_s065 vs J1, J2_s065 vs J2 — D1-M1 class
#   (single comparison: uncorrected interval, δ = 10 % of the reference arm's gain), D1-M2 test (5 % / 10 %; Holm across the three
#   pairs per approach and K, uncorrected beside — owner, RL-411), D1-M4 test,
#   one-SE band of each arm, social readings (the approach's affiliation relation held-out gain; people coherence =
#   co-membership excess of article pairs sharing an author / a co-author tie, swap_link_measure.one at p = 0), each as
#   R − A ± SE over groups; D1 contrast J2 vs J1 at 0.65 beside the same at W1. Dominance of 0.65 (RL-404): strictly
#   inferior = J2_s065 vs J2 misleads at the majority of candidate K and no social reading better by > 1 SE; superior =
#   mirror. Flags for the owner: a robustness arm moving the one-SE band, or memberships changed (≥ 5 %) at a candidate K.
#
# Usage (from fullscale_pipeline/m1, PYTHONPATH=.):
#   python -m fit.d1_analysis --stage stage1 --out fit/results/stage1/d1_analysis.json --workers 8
#   [M1_STUDY=B] python -m fit.d1_analysis --stage B_stage1 --build <BUILD_B> ...
import os
os.environ.setdefault("M1_ALIGNED", "0")      # every reading here is free of the community numbering
import argparse, itertools, json, math, zlib
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
JOURNAL_ARMS = ("J1", "J2", "J3a", "J3b", "J4", "J5")
COMPARED = ("J2", "J3a", "J3b", "J4", "J5")                 # each vs J1 (Dunnett / Holm family per approach and K)
ROBUST = (("J2_logdamp", "J2"), ("J1_s065", "J1"), ("J2_s065", "J2"))
FAMILIES = tuple(("J1", x) for x in COMPARED) + tuple((a, r) for r, a in ROBUST) + (("J1_s065", "J2_s065"),)
SEM = ("child_article", "parent_child")
AFFIL = {1: "author_affiliation", 2: "article_affiliation"}
PEOPLE = {1: "author", 2: "co-author tie"}
DELTA_SHARE, DELTA_REPORTED = 0.10, (0.05, 0.20)
M2_MIN, M2_REPORTED = 0.05, 0.10
M4_MIN, M4_REPORTED = 0.2, 0.5
ALPHA = 0.05
N_ARM_SHUFFLES = 9_999
N_COMM_SHUFFLES = 999
COVERAGE_C = 0.5


# =============================================================================================== pure statistics
def holm(p):
    """Holm (1979) step-down adjusted p-values, same order as p (None kept as None)."""
    idx = [i for i, x in enumerate(p) if x is not None]
    out = [None] * len(p)
    m = len(idx); run = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: p[i])):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def mvt_crit(R, df, q, n=200_000, seed=107):
    """q-quantile of max_i |T_i|, T multivariate t (df) with correlation R (Dunnett-type simultaneous critical value)."""
    R = np.atleast_2d(np.asarray(R, float)); k = R.shape[0]
    if k == 1:
        return float(stats.t.ppf(0.5 + q / 2, df))
    w, V = np.linalg.eigh(R); w = np.clip(w, 0, None)
    L = V * np.sqrt(w)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k)) @ L.T
    s = np.sqrt(rng.chisquare(df, size=n) / df)
    return float(np.quantile(np.abs(z / s[:, None]).max(1), q))


def classify(m, se, c95, c90, delta):
    """Four classes (L2.4 / D1-M1) from the mean, its SE, the 95 % and 90 % critical values and the margin δ."""
    lo95, hi95, lo90, hi90 = m - c95 * se, m + c95 * se, m - c90 * se, m + c90 * se
    if lo95 > 0 and m >= delta:
        c = "adds information"
    elif hi95 < 0 and m <= -delta:
        c = "misleads"
    elif lo90 > -delta and hi90 < delta:
        c = "redundant"
    else:
        c = "inconclusive"
    return c, [lo95, hi95], [lo90, hi90]


def compare_family(D, names, delta, deltas_reported=()):
    """D: G × k array of per-group differences (rows = groups common to all k comparisons). Dunnett-type simultaneous
    intervals with each comparison's own SE and the estimated correlation; uncorrected intervals beside."""
    D = np.asarray(D, float); G, k = D.shape
    out = {"groups": int(G)}
    if G < 3:
        return {**out, **{n: {"class": "inconclusive", "reason": "fewer than 3 groups"} for n in names}}
    m = D.mean(0); sd = D.std(0, ddof=1); se = sd / math.sqrt(G)
    ok = sd > 0
    R = np.corrcoef(D[:, ok].T) if ok.sum() > 1 else np.ones((max(ok.sum(), 1),) * 2)
    c95 = mvt_crit(R, G - 1, 0.95) if ok.any() else float("nan")
    c90 = mvt_crit(R, G - 1, 0.90) if ok.any() else float("nan")
    t95, t90 = stats.t.ppf(0.975, G - 1), stats.t.ppf(0.95, G - 1)
    out.update({"crit_dunnett_95": c95, "crit_dunnett_90": c90, "crit_t_95": float(t95), "crit_t_90": float(t90),
                "correlation": R.tolist() if ok.sum() > 1 else None})
    for i, n in enumerate(names):
        e = {"mean": float(m[i]), "se": float(se[i]), "delta": delta}
        if se[i] == 0:      # no spread over groups: the interval is the point
            c = classify(m[i], 0.0, 0, 0, delta)[0]
            e.update({"class": c, "class_uncorrected": c})
        else:
            e["class"], e["ci95"], e["ci90"] = classify(m[i], se[i], c95, c90, delta)
            e["class_uncorrected"], e["ci95_uncorrected"], e["ci90_uncorrected"] = classify(m[i], se[i], t95, t90, delta)
        e["class_at"] = {str(f): (classify(m[i], se[i], c95 if se[i] else 0, c90 if se[i] else 0, d)[0])
                         for f, d in deltas_reported}
        out[n] = e
    return out


def majority(classes):
    """Class at the majority (> half) of candidate K; one candidate → its class; no majority → inconclusive."""
    classes = [c for c in classes if c is not None]
    if not classes:
        return None
    top, n = max(((c, classes.count(c)) for c in set(classes)), key=lambda x: x[1])
    return top if n > len(classes) / 2 else "inconclusive"


def half_split_margin(group_means):
    """Hold-out half-split margin: 95th percentile over all splits of the groups into two halves of |mean difference|,
    ÷ √2 (§8, reported beside δ)."""
    g = np.asarray(group_means, float); G = len(g); h = G // 2
    if G < 4:
        return None
    vals = []
    for A in itertools.combinations(range(G), h):
        if 0 not in A:          # each unordered split once
            continue
        B = [i for i in range(G) if i not in A][:h]
        vals.append(abs(g[list(A)].mean() - g[B].mean()))
    return float(np.percentile(vals, 95) / math.sqrt(2))


def one_se_band(per_K):
    """per_K: {K: list of group means}. Band = K with mean ≥ best mean − SE(best) (aggregate.choose_k's band)."""
    s = {K: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None)
         for K, v in per_K.items() if len(v)}
    s = {K: x for K, x in s.items() if x[1] is not None}
    if not s:
        return {"band": [], "scores": {}}
    best = max(s, key=lambda K: s[K][0]); cut = s[best][0] - s[best][1]
    return {"band": sorted(K for K in s if s[K][0] >= cut), "best": best, "cutoff": cut,
            "scores": {str(K): {"mean": m, "se": e} for K, (m, e) in s.items()}}


def _splits(n, n_ref):
    return [np.isin(np.arange(n), c) for c in itertools.combinations(range(n), n_ref)]


def label_stats_diff(ref_vals, x_vals):
    """All relabellings of one group (n_ref of the fits called 'reference'): stat = mean(x) − mean(ref)."""
    v = np.asarray(list(ref_vals) + list(x_vals), float); n_ref = len(ref_vals)
    return np.array([v[~L].mean() - v[L].mean() for L in _splits(len(v), n_ref)])


def label_stats_agreement(S, n_ref):
    """S: n × n symmetric agreement of fit pairs (fits 0..n_ref−1 = reference arm). For every relabelling: mean cross-arm
    agreement − mean within-arm agreement (within pairs of both arms pooled)."""
    n = S.shape[0]; iu = np.triu_indices(n, 1)
    out = []
    for L in _splits(n, n_ref):
        same = (L[:, None] == L[None, :])[iu]
        out.append(S[iu][~same].mean() - S[iu][same].mean())
    return np.array(out)


def perm_test(group_stats, observed, alternative, n=N_ARM_SHUFFLES, seed=0):
    """group_stats: per group the statistic under every relabelling (enumerated); null = mean over groups of a random
    relabelling of each group; p = (1 + #null at least as extreme) / (1 + n) (North, Curtis & Sham 2002)."""
    rng = np.random.default_rng(seed)
    null = np.zeros(n)
    for s in group_stats:
        null += s[rng.integers(0, len(s), n)]
    null /= len(group_stats)
    hits = (null >= observed - 1e-12).sum() if alternative == "greater" else (null <= observed + 1e-12).sum()
    return float((1 + hits) / (1 + n)), {"null_mean": float(null.mean()), "null_p05": float(np.percentile(null, 5)),
                                         "null_p95": float(np.percentile(null, 95))}


def detection_limit(var_of_mean, power=0.9):
    """Smallest true Δ̄ detected with the given power by a one-sided z test (normal approximation), α 0.01 and 0.05."""
    se = math.sqrt(max(var_of_mean, 0.0))
    return {str(a): float((stats.norm.ppf(1 - a) + stats.norm.ppf(power)) * se) for a in (0.01, 0.05)}


# ======================================================================================== per-fit readings
def journal_shaped(P, lab, J, rng, n_sh=N_COMM_SHUFFLES):
    """D1-M4 community-level test (§4.4), extended with the RL-399/400 readings. P: covered articles with a journal ×
    K weighted memberships; lab: journal index per article (0..J−1)."""
    import scipy.sparse as sp
    n, K = P.shape
    size = np.bincount(lab, minlength=J); ok = size >= 2
    colmass = np.maximum(P.sum(0), 1e-300)

    def shares(l):
        return np.asarray((sp.csr_matrix((np.ones(n), (l, np.arange(n))), shape=(J, n)) @ P)) / colmass

    def jk(l):
        return np.where(ok[:, None], shares(l), -1).max(0)
    S = shares(lab); Sm = np.where(ok[:, None], S, -1)
    real, jreal, arg = Sm.max(0), Sm.argmax(0), S.argmax(0)      # arg: largest share over all journals (singletons too)
    ge = np.zeros(K)
    for _ in range(n_sh):
        ge += jk(rng.permutation(lab)) >= real
    p = (1 + ge) / (1 + n_sh)
    shaped = (p < ALPHA) & (real >= 0.25)
    detail = []
    for k in np.nonzero(shaped)[0]:
        m = P[lab == jreal[k], k]
        detail.append({"k": int(k), "j_k": float(real[k]), "journal": int(jreal[k]), "journal_articles": int(len(m)),
                       "size": float(P[:, k].sum() / n), "share_ge_0.3": float((m >= 0.3).mean()),
                       "share_ge_0.5": float((m >= 0.5).mean()), "share_lt_0.1": float((m < 0.1).mean()),
                       "contribution_from_members": float(m[m >= 0.3].sum() / max(m.sum(), 1e-300))})
    sizes = P.sum(0) / n
    return {"journal_shaped": int(shaped.sum()), "associated": int(((p < ALPHA) & (real < 0.25)).sum()),
            "singleton_top": int((~ok[arg]).sum()), "mass_in_shaped": float(P[:, shaped].sum() / n),
            "size_quantiles": [float(x) for x in np.quantile(sizes, [0, 0.1, 0.5, 0.9, 1.0])],
            "jk_max": float(real.max()), "shaped_detail": detail}


def degree_only_loss(rd):
    """Generalised KL of the degree-only reference (tie weights: row total × column total ÷ grand total of the visible
    tie values) on the relation's fitted cells (RL-396), via the training loss's own masked computation."""
    import torch
    from .loss import RelationTensors, relation_loss
    rt = RelationTensors(rd)
    n_rows, n_cols = len(rd.obs_row_mask), rd.shape[1]
    rs = np.bincount(rd.tie_row, rd.tie_val, n_rows); cs = np.bincount(rd.tie_col, rd.tie_val, n_cols)
    T = lambda v: torch.as_tensor(v[:, None], dtype=torch.float64)
    return float(relation_loss(T(rs), torch.ones(1, 1, dtype=torch.float64), T(cs / max(cs.sum(), 1e-300)), rt, 1))


def loss_shares(final_loss, ref_loss, weights):
    """D1-M3 per fit: c_r, realised share s_r, pull share p_r, semantic and journal totals, flag."""
    rels = [r for r in final_loss if r in ref_loss]
    dL = {r: ref_loss[r] - final_loss[r] for r in rels}
    den_s = sum(weights[r] * dL[r] for r in rels); den_p = sum(weights[r] * ref_loss[r] for r in rels)
    out = {"c": {r: 1 - final_loss[r] / ref_loss[r] for r in rels},
           "s": {r: weights[r] * dL[r] / den_s for r in rels}, "p": {r: weights[r] * ref_loss[r] / den_p for r in rels}}
    out["s_semantic"] = sum(out["s"].get(r, 0.0) for r in SEM)
    out["s_journal"] = out["s"].get("article_journal")
    out["p_journal"] = out["p"].get("article_journal")
    out["flag"] = None if out["s_journal"] is None else bool(out["s_journal"] >= out["s_semantic"])
    return out


def heldout_readings(meas):
    tp = meas.get("heldout_two_part") or {}
    comb = meas.get("heldout_two_part_combined") or {}
    ref = [tp[r]["baseline_weights_ap_within_class"] for r in SEM if tp.get(r)]
    out = {"existence_sem": comb.get("existence_semantic_equal"), "value_sem": comb.get("value_semantic_equal"),
           "existence_main": comb.get("existence"),
           "reference_sem": float(np.mean(ref)) if len(ref) == len(SEM) else None,
           "social_gain": {r: v["ap_within_class"] - v["baseline_weights_ap_within_class"]
                           for r, v in tp.items() if r not in SEM and v and v.get("ap_within_class") is not None}}
    return out


# ============================================================================================ per unit (parallel)
_G = {}


def _arm_of(f):
    return (f["job"].get("arm") or {}).get("j")


def unit(key):
    """All readings of one (approach, K, group): per fit (held-out, M3, M4, people coherence) and per fit pair (M2)."""
    from . import measures as M
    from .aggregate import load_UZ, _rds_of
    from .swap_link_measure import one as swap_one
    approach, K, group = key
    byarm = _G["units"][key]
    rng = np.random.default_rng([approach, K, group, 107])
    fits = [(a, f) for a in sorted(byarm) for f in byarm[a]]
    ids = [os.path.basename(f["dir"]) for _, f in fits]
    rels = [{r: tuple(v) for r, v in f["result"]["relations"].items()} for _, f in fits]
    UZ = [load_UZ(f["dir"]) for _, f in fits]
    rdss = [_rds_of(f["job"]) for _, f in fits]
    wms = [M.weighted_memberships(U, Z, rel) for (U, Z), rel in zip(UZ, rels)]
    covs = [M.entity_coverage(U, Z, rds) for (U, Z), rds in zip(UZ, rdss)]
    # facets matched on: those every arm has, journal excluded in every comparison (owner, RL-410 (b))
    common = sorted(set.intersection(*[set(w) for w in wms]) - {"journal"})
    jour = _G["journal_of_article"]
    # live entities for co-membership (≥ 1 visible tie in a shared relation), from the first fit's data
    live = {f: np.zeros(len(wms[0][f]["P"]), bool) for f in common}
    for rd in rdss[0]:
        if rd.row_facet in live and rd.col_facet in live:
            live[rd.row_facet][rd.tie_row] = True; live[rd.col_facet][rd.tie_col] = True
    per_fit = {}
    ref_cache = {}
    for i, (a, f) in enumerate(fits):
        rd_by = {rd.name: rd for rd in rdss[i]}
        dkey = json.dumps({k: v for k, v in (f["job"].get("arm") or {}).items() if k not in ("r", "j")}, sort_keys=True)
        if dkey not in ref_cache:
            ref_cache[dkey] = {r: degree_only_loss(rd) for r, rd in rd_by.items()}
        m3 = loss_shares(f["result"]["final_relation_loss"], ref_cache[dkey], {r: rd.weight for r, rd in rd_by.items()})
        coh = None
        if _G["coherence"]:
            c = swap_one(f, {}, rds=rdss[i])
            coh = {e: v["model_excess"] for e, v in c.items()}
        per_fit[ids[i]] = {"arm": a, "seed": f["job"]["seed"], "epochs": f["result"]["epochs"],
                           "heldout": heldout_readings(f["measures"]), "m3": m3, "coherence": coh}
    # D1-M2 and D1-M4 per comparison (owner, RL-410 (a)): the article set of comparison (A, B) = articles covered on
    # average over the fits of A and B only; M4 counts and M2 fit pairs are computed on that set
    fam_out = {}
    for A, B in FAMILIES:
        if A not in byarm or B not in byarm:
            continue
        idx = [i for i, (a, _) in enumerate(fits) if a in (A, B)]
        rng = np.random.default_rng([approach, K, group, 107, _seed(A, B)])
        covered = M.covered_entities([{f: covs[i][f] for f in common} for i in idx], COVERAGE_C)
        keep = covered["article"]
        kj = keep & (jour >= 0)
        _, lab = np.unique(jour[kj], return_inverse=True); J = int(lab.max()) + 1 if kj.any() else 0
        m4 = {ids[i]: (journal_shaped(wms[i]["article"]["P"][kj], lab, J, rng) if J else None) for i in idx}
        placed = {f: covered[f] for f in common}
        pairs = {}
        for i, j in itertools.combinations(idx, 2):
            w1 = {f: wms[i][f] for f in common}; w2 = {f: wms[j][f] for f in common}
            w2s = M._shuffled(w2, rng)
            real = M.track_pair(w1, w2, placed, common); ch = M.track_pair(w1, w2s, placed, common)
            adj = lambda r, c: (r - c) / (1 - c) if c < 1 else float("nan")
            Ps = {f: w1[f]["P"] for f in common}
            cr = _comem(Ps, {f: w2[f]["P"] for f in common}, live)
            cc = _comem(Ps, {f: w2s[f]["P"] for f in common}, live)
            r2 = float(np.nanmean(cr[keep])); r2c = float(np.nanmean(cc[keep]))
            pairs[f"{ids[i]}|{ids[j]}"] = {
                "A_article": adj(real["A"]["article"], ch["A"]["article"]), "B_article": adj(real["B"]["article"], ch["B"]["article"]),
                "A_model": adj(real["A_model"], ch["A_model"]), "B_model": adj(real["B_model"], ch["B_model"]),
                "R2_article": adj(r2, r2c), "A_article_raw": real["A"]["article"], "A_article_chance": ch["A"]["article"]}
        fam_out[_fam(A, B)] = {"m4": m4, "pairs": pairs, "covered_articles": int(keep.sum()),
                               "covered_with_journal": int(kj.sum()),
                               "journals_ge2": int((np.bincount(lab, minlength=J) >= 2).sum()) if J else 0}
    return key, {"fits": per_fit, "fam": fam_out, "order": ids, "common_facets": common}


def _fam(A, B):
    return "|".join(sorted((A, B)))


def _comem(Ps, Pt, live, f="article"):
    """membership_readings.comembership_corr for one facet only (same sums over every facet's live entities; the
    per-entity part computed for facet f alone — the other facets' rows are not read by D1-M2)."""
    K = next(iter(Ps.values())).shape[1]
    a_s = np.zeros(K); a_t = np.zeros(K); Bss = np.zeros((K, K)); Btt = np.zeros((K, K)); Bst = np.zeros((K, K)); W = 0.0
    wf = {}
    for g in Ps:
        n = int(live[g].sum())
        if n == 0:
            wf[g] = 0.0; continue
        w = 1.0 / n; wf[g] = w; W += 1.0
        S, T = Ps[g][live[g]], Pt[g][live[g]]
        a_s += w * S.sum(0); a_t += w * T.sum(0)
        Bss += w * S.T @ S; Btt += w * T.T @ T; Bst += w * S.T @ T
    x, y = Ps[f], Pt[f]
    Sx = x @ a_s; Sy = y @ a_t
    Sxx = np.einsum("ik,kl,il->i", x, Bss, x); Syy = np.einsum("ik,kl,il->i", y, Btt, y)
    Sxy = np.einsum("ik,kl,il->i", x, Bst, y)
    sx = (x * x).sum(1); sy = (y * y).sum(1); w = wf[f] * live[f]
    Sx = Sx - w * sx; Sy = Sy - w * sy; Sxx = Sxx - w * sx ** 2; Syy = Syy - w * sy ** 2; Sxy = Sxy - w * sx * sy
    Wi = np.full(len(x), W) - w
    cov = Sxy / Wi - Sx * Sy / Wi ** 2
    vx = Sxx / Wi - (Sx / Wi) ** 2; vy = Syy / Wi - (Sy / Wi) ** 2
    return cov / np.sqrt(np.maximum(vx * vy, 1e-300))


# ================================================================================================ summaries
def _seed(*parts):
    """Stable seed (Python's string hash changes between processes)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def _gm(vals):
    return float(np.mean(vals)) if len(vals) else None


def summarise(units, approach):
    """Everything per approach from the unit readings: bands, candidate K, M1–M4 per K, robustness, decision."""
    Ks = sorted({k[1] for k in units if k[0] == approach})
    groups_of = lambda K: sorted(g for (a, k, g) in units if a == approach and k == K)
    fit_vals = lambda K, g, arm, fn: [fn(v) for v in units[(approach, K, g)]["fits"].values() if v["arm"] == arm]

    def gmeans(K, arm, fn):
        out = {}
        for g in groups_of(K):
            v = [x for x in fit_vals(K, g, arm, fn) if x is not None]
            if v:
                out[g] = float(np.mean(v))
        return out
    ex = lambda v: v["heldout"]["existence_sem"]
    # ---- bands and candidate K
    arms_present = sorted({v["arm"] for (a, _, _), u in units.items() if a == approach for v in u["fits"].values()})
    bands = {arm: one_se_band({K: list(gmeans(K, arm, lambda v: v["heldout"]["existence_main"]).values()) for K in Ks})
             for arm in arms_present}
    cand = sorted(set().union(*[set(bands[a]["band"]) for a in JOURNAL_ARMS if a in bands])) if bands else []
    rep = {"K": Ks, "bands": bands, "candidate_K": cand, "per_K": {}}
    for K in Ks:
        e = {}
        j1 = gmeans(K, "J1", ex); j1ref = gmeans(K, "J1", lambda v: v["heldout"]["reference_sem"])
        gain = [j1[g] - j1ref[g] for g in j1 if g in j1ref]
        delta = DELTA_SHARE * float(np.mean(gain)) if gain else None
        e["delta"] = delta; e["J1_gain"] = _gm(gain); e["half_split_margin"] = half_split_margin(list(j1.values()))
        # ---- D1-M1 (Dunnett family over the arms present)
        fam = [x for x in COMPARED if gmeans(K, x, ex)]
        if delta is not None and fam:
            per = {x: gmeans(K, x, ex) for x in fam}
            G = sorted(set(j1).intersection(*[set(per[x]) for x in fam]))
            D = np.array([[per[x][g] - j1[g] for x in fam] for g in G])
            e["M1"] = compare_family(D, fam, delta, [(f, f / DELTA_SHARE * delta) for f in DELTA_REPORTED]) if len(G) else None
            vj1 = gmeans(K, "J1", lambda v: v["heldout"]["value_sem"])
            pv = {x: gmeans(K, x, lambda v: v["heldout"]["value_sem"]) for x in fam}
            Gv = sorted(set(vj1).intersection(*[set(pv[x]) for x in fam]))
            e["M1_value_reported"] = compare_family(np.array([[pv[x][g] - vj1[g] for x in fam] for g in Gv]), fam,
                                                    delta) if len(Gv) else None
            e["social_gain"] = {arm: {r: _gm(list(gmeans(K, arm, lambda v, r=r: v["heldout"]["social_gain"].get(r)).values()))
                                      for r in _social_rels(units, approach, K)} for arm in arms_present}
        # ---- D1-M2, D1-M4 (Holm families over the arms vs J1)
        e["M2"] = {x: m2_test(units, approach, K, "J1", x) for x in fam}
        e["M4"] = {x: m4_test(units, approach, K, "J1", x) for x in fam}
        for M in ("M2", "M4"):
            adj = holm([e[M][x]["p"] if e[M][x] else None for x in fam])
            for x, pa in zip(fam, adj):
                if e[M][x]:
                    e[M][x]["p_holm"] = pa
        for x in fam:
            m2, m4 = e["M2"][x], e["M4"][x]
            if m2:
                for mn in (M2_MIN, M2_REPORTED):
                    m2[f"changed_at_{mn:g}"] = bool(m2["p_holm"] < ALPHA and m2["drop"] >= mn)
                    m2[f"changed_uncorrected_at_{mn:g}"] = bool(m2["p"] < ALPHA and m2["drop"] >= mn)
                m2["changed"] = m2[f"changed_at_{M2_MIN:g}"]
            if m4:
                for mn in (M4_MIN, M4_REPORTED):
                    m4[f"more_at_{mn:g}"] = bool(m4["p_holm"] < ALPHA and m4["delta_bar"] >= mn)
                    m4[f"more_uncorrected_at_{mn:g}"] = bool(m4["p"] < ALPHA and m4["delta_bar"] >= mn)
                m4["more"] = m4[f"more_at_{M4_MIN:g}"]
        # ---- D1-M3
        e["M3"] = {arm: m3_summary(units, approach, K, arm) for arm in arms_present}
        # ---- journal problem per arm (RL-402), with the reported readings
        e["problem"] = {}
        for x in fam:
            m2, m4 = e["M2"][x], e["M4"][x]
            if not (m2 and m4):
                e["problem"][x] = None; continue
            gx = set(m2["per_group_drop"]) & set(m4["per_group_delta"])
            e["problem"][x] = {"problem": bool(m2["changed"] and m4["more"]),
                               "share_fits_above_J1_group_mean": m4["share_fits_above_ref_group_mean"],
                               "groups_showing": int(sum(m4["per_group_delta"][g] >= M4_MIN and m2["per_group_drop"][g] >= M2_MIN for g in gx)),
                               "M3_flag": e["M3"][x]["flag_share"] if e["M3"].get(x) else None}
        # ---- robustness readings
        e["robustness"] = {}
        for R, A in ROBUST + (("J2_s065", "J1_s065"),):
            if not (gmeans(K, R, ex) and gmeans(K, A, ex)):
                continue
            e["robustness"][f"{R} vs {A}"] = robust_pair(units, approach, K, R, A, gmeans)
        # Holm across the D1-M2 tests of the three robustness pairs (owner, RL-411); the contrast pair is reported only
        rnames = [f"{R} vs {A}" for R, A in ROBUST if (e["robustness"].get(f"{R} vs {A}") or {}).get("M2")]
        adj = holm([e["robustness"][n]["M2"]["p"] for n in rnames])
        for n, pa in zip(rnames, adj):
            m2 = e["robustness"][n]["M2"]; m2["p_holm"] = pa
            for mn in (M2_MIN, M2_REPORTED):
                m2[f"changed_at_{mn:g}"] = bool(pa < ALPHA and m2["drop"] >= mn)
        cn = "J2_s065 vs J1_s065"
        if (e["robustness"].get(cn) or {}).get("M2"):
            m2 = e["robustness"][cn]["M2"]
            for mn in (M2_MIN, M2_REPORTED):
                m2[f"changed_at_{mn:g}"] = m2[f"changed_uncorrected_at_{mn:g}"]
        rep["per_K"][str(K)] = e
    rep["arms"], rep["decision"] = decide(rep, cand)
    rep["robustness_summary"] = robustness_summary(rep, cand, approach)
    return rep


def _social_rels(units, approach, K):
    rs = set()
    for (a, k, _), u in units.items():
        if a == approach and k == K:
            for v in u["fits"].values():
                rs |= set(v["heldout"]["social_gain"])
    return sorted(rs)


def _group_fits(units, approach, K, A, B):
    """Per group: (ids of A's fits, ids of B's fits)."""
    out = {}
    for (a, k, g), u in units.items():
        if a == approach and k == K:
            ia = [i for i in u["order"] if u["fits"][i]["arm"] == A]
            ib = [i for i in u["order"] if u["fits"][i]["arm"] == B]
            if len(ia) >= 2 and len(ib) >= 2:
                out[g] = (ia, ib)
    return out


def m2_test(units, approach, K, A, B, measures=("A_article", "B_article", "A_model", "B_model", "R2_article")):
    """D1-M2 (RL-394/395): within-group label permutation of (mean cross − mean within agreement), one-sided (lower)."""
    gf = _group_fits(units, approach, K, A, B)
    if len(gf) < 2:
        return None
    res = {}
    for meas in measures:
        stats_g, obs_g, within_g, cross_g = [], {}, {}, {}
        for g, (ia, ib) in sorted(gf.items()):
            ids = ia + ib; n = len(ids); P = units[(approach, K, g)]["fam"][_fam(A, B)]["pairs"]
            S = np.full((n, n), np.nan)
            for i, j in itertools.combinations(range(n), 2):
                v = P.get(f"{ids[i]}|{ids[j]}", P.get(f"{ids[j]}|{ids[i]}"))
                S[i, j] = S[j, i] = v[meas] if v else np.nan
            if np.isnan(S[np.triu_indices(n, 1)]).any():
                continue
            iu = np.triu_indices(n, 1); L = np.arange(n) < len(ia); same = (L[:, None] == L[None, :])[iu]
            within_g[g] = float(S[iu][same].mean()); cross_g[g] = float(S[iu][~same].mean())
            obs_g[g] = cross_g[g] - within_g[g]
            stats_g.append(label_stats_agreement(S, len(ia)))
        if len(obs_g) < 2:
            continue
        obs = float(np.mean(list(obs_g.values())))
        p, null = perm_test(stats_g, obs, "less", seed=_seed(approach, K, A, B, meas))
        size = np.array([within_g[g] - cross_g[g] for g in obs_g])
        w = float(np.mean(list(within_g.values())))
        res[meas] = {"statistic": obs, "p": p, "null": null, "within": w, "cross": float(np.mean(list(cross_g.values()))),
                     "size": float(size.mean()), "size_se": float(size.std(ddof=1) / math.sqrt(len(size))),
                     "drop": float(size.mean() / w) if w else None, "groups": len(obs_g),
                     "per_group_drop": {g: (within_g[g] - cross_g[g]) / within_g[g] if within_g[g] else None for g in obs_g}}
    if "A_article" not in res:
        return None
    out = dict(res["A_article"]); out["reported"] = {m: {k: v for k, v in r.items() if k != "per_group_drop"}
                                                    for m, r in res.items() if m != "A_article"}
    return out


def m4_arm_summary(units, approach, K, arm, fam):
    """D1-M4 readings of one arm's fits, on the article set of comparison fam."""
    v = [u["fam"][fam]["m4"][i] for (a, k, _), u in units.items() if a == approach and k == K and fam in u["fam"]
         for i, x in u["fits"].items() if x["arm"] == arm and u["fam"][fam]["m4"].get(i)]
    if not v:
        return None
    sh = [d for x in v for d in x["shaped_detail"]]
    return {"fits": len(v), "journal_shaped_mean": float(np.mean([x["journal_shaped"] for x in v])),
            "counts": np.bincount([x["journal_shaped"] for x in v]).tolist(),
            "associated_mean": float(np.mean([x["associated"] for x in v])),
            "singleton_top_mean": float(np.mean([x["singleton_top"] for x in v])),
            "mass_in_shaped_mean": float(np.mean([x["mass_in_shaped"] for x in v])),
            "size_quantiles_mean": np.mean([x["size_quantiles"] for x in v], 0).tolist(),
            "shaped_communities": len(sh),
            "shaped_memberships": {k: float(np.median([d[k] for d in sh])) for k in
                                   ("share_ge_0.3", "share_ge_0.5", "share_lt_0.1", "contribution_from_members", "size", "j_k")} if sh else None,
            "shaped_journals": dict(sorted(((str(j), sum(d["journal"] == j for d in sh)) for j in {d["journal"] for d in sh}),
                                           key=lambda x: -x[1]))}


def m4_test(units, approach, K, A, B):
    """D1-M4 arm comparison (RL-397/399): Δ̄ = mean over groups of (mean count B − mean count A), one-sided (greater)."""
    gf = _group_fits(units, approach, K, A, B)
    stats_g, d_g, var_g, above, n_fits = [], {}, [], 0, 0
    mass_d = []
    for g, (ia, ib) in sorted(gf.items()):
        F = units[(approach, K, g)]["fam"].get(_fam(A, B), {}).get("m4", {})
        if any(F.get(i) is None for i in ia + ib):
            continue
        ca = [F[i]["journal_shaped"] for i in ia]; cb = [F[i]["journal_shaped"] for i in ib]
        d_g[g] = float(np.mean(cb) - np.mean(ca))
        var_g.append(np.var(cb, ddof=1) / len(cb) + np.var(ca, ddof=1) / len(ca))
        stats_g.append(label_stats_diff(ca, cb))
        above += sum(c > np.mean(ca) for c in cb); n_fits += len(cb)
        mass_d.append(np.mean([F[i]["mass_in_shaped"] for i in ib]) - np.mean([F[i]["mass_in_shaped"] for i in ia]))
    if len(d_g) < 2:
        return None
    obs = float(np.mean(list(d_g.values())))
    p, null = perm_test(stats_g, obs, "greater", seed=_seed(approach, K, A, B, "m4"))
    return {"delta_bar": obs, "p": p, "null": null, "groups": len(d_g), "per_group_delta": d_g,
            "delta_bar_se": float(np.std(list(d_g.values()), ddof=1) / math.sqrt(len(d_g))),
            "detection_limit": detection_limit(float(np.sum(var_g)) / len(var_g) ** 2),
            "share_fits_above_ref_group_mean": above / n_fits if n_fits else None,
            "mass_reading": float(np.mean(mass_d)), "mass_reading_se": float(np.std(mass_d, ddof=1) / math.sqrt(len(mass_d))),
            "covered_articles_per_group": {g: units[(approach, K, g)]["fam"][_fam(A, B)]["covered_articles"] for g in d_g},
            A: m4_arm_summary(units, approach, K, A, _fam(A, B)), B: m4_arm_summary(units, approach, K, B, _fam(A, B))}


def m3_summary(units, approach, K, arm):
    byg = defaultdict(list)
    for (a, k, g), u in units.items():
        if a == approach and k == K:
            byg[g] += [v["m3"] for v in u["fits"].values() if v["arm"] == arm]
    byg = {g: v for g, v in byg.items() if v}
    if not byg:
        return None
    gm = lambda fn: [float(np.mean([fn(x) for x in v])) for v in byg.values()]
    rels = sorted(next(iter(byg.values()))[0]["s"])
    se = lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None
    out = {"groups": len(byg), "s_semantic": float(np.mean(gm(lambda x: x["s_semantic"]))),
           "c": {r: float(np.mean(gm(lambda x, r=r: x["c"][r]))) for r in rels},
           "s": {r: float(np.mean(gm(lambda x, r=r: x["s"][r]))) for r in rels},
           "p": {r: float(np.mean(gm(lambda x, r=r: x["p"][r]))) for r in rels}}
    if "article_journal" in rels:
        sj = gm(lambda x: x["s_journal"])
        out.update({"s_journal": float(np.mean(sj)), "s_journal_se": se(sj), "p_journal": float(np.mean(gm(lambda x: x["p_journal"]))),
                    "flag_share": float(np.mean([x["flag"] for v in byg.values() for x in v])),
                    "flag": bool(np.mean(sj) >= out["s_semantic"])})
    return out


def robust_pair(units, approach, K, R, A, gmeans):
    ex = lambda v: v["heldout"]["existence_sem"]
    a, r = gmeans(K, A, ex), gmeans(K, R, ex)
    aref = gmeans(K, A, lambda v: v["heldout"]["reference_sem"])
    G = sorted(set(a) & set(r))
    delta = DELTA_SHARE * float(np.mean([a[g] - aref[g] for g in a if g in aref]))
    out = {"M1": compare_family(np.array([[r[g] - a[g]] for g in G]), [R], delta)[R] if len(G) else None, "delta": delta}
    m2 = m2_test(units, approach, K, A, R)
    if m2:
        for mn in (M2_MIN, M2_REPORTED):
            m2[f"changed_uncorrected_at_{mn:g}"] = bool(m2["p"] < ALPHA and m2["drop"] >= mn)
    out["M2"] = m2
    m4 = m4_test(units, approach, K, A, R)
    if m4:
        for mn in (M4_MIN, M4_REPORTED):
            m4[f"more_at_{mn:g}"] = bool(m4["p"] < ALPHA and m4["delta_bar"] >= mn)
    out["M4"] = m4
    soc = {}
    rel = AFFIL[approach]
    for name, fn in ((f"{rel} held-out gain", lambda v: v["heldout"]["social_gain"].get(rel)),
                     (f"people coherence ({PEOPLE[approach]})", lambda v: (v["coherence"] or {}).get(PEOPLE[approach])),
                     ("journal coherence", lambda v: (v["coherence"] or {}).get("journal"))):
        ga, gr = gmeans(K, A, fn), gmeans(K, R, fn)
        Gs = sorted(set(ga) & set(gr))
        if len(Gs) > 1:
            d = np.array([gr[g] - ga[g] for g in Gs])
            soc[name] = {"R_minus_A": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "A": float(np.mean([ga[g] for g in Gs])),
                         "better_beyond_1se": bool(d.mean() > d.std(ddof=1) / math.sqrt(len(d))),
                         "worse_beyond_1se": bool(-d.mean() > d.std(ddof=1) / math.sqrt(len(d)))}
    out["social"] = soc
    return out


def decide(rep, cand):
    """Decision rule (RL-402), per approach. Returns per-arm summaries over the candidate K and the default J*."""
    arms = {}
    for x in COMPARED:
        cls = {K: (rep["per_K"][str(K)].get("M1") or {}).get(x, {}).get("class") for K in cand}
        if all(c is None for c in cls.values()):
            continue
        prob = {K: (rep["per_K"][str(K)]["problem"].get(x) or {}).get("problem") for K in cand}
        n_prob = sum(bool(v) for v in prob.values())
        sj = [(rep["per_K"][str(K)]["M3"].get(x) or {}).get("s_journal") for K in cand]
        dm = [(rep["per_K"][str(K)].get("M1") or {}).get(x, {}).get("mean") for K in cand]
        arms[x] = {"class_per_K": {str(K): c for K, c in cls.items()}, "class": majority(list(cls.values())),
                   "class_uncorrected_per_K": {str(K): (rep["per_K"][str(K)].get("M1") or {}).get(x, {}).get("class_uncorrected") for K in cand},
                   "problem_per_K": {str(K): v for K, v in prob.items()}, "problem_K": [K for K, v in prob.items() if v],
                   "problem_share": n_prob / len(cand) if cand else None,
                   "excluded_journal_problem": bool(cand) and n_prob >= len(cand) / 2,
                   "M3_flag_per_K": {str(K): (rep["per_K"][str(K)]["M3"].get(x) or {}).get("flag") for K in cand},
                   "s_journal_mean_over_candidate_K": _gm([v for v in sj if v is not None]),
                   "delta_mean_over_candidate_K": _gm([v for v in dm if v is not None])}
    if not cand or not arms:
        return arms, {"J_star": None, "reason": "no candidate K or no compared arm"}
    reasons = []
    remaining = []
    for x, a in arms.items():
        if a["class"] == "misleads":
            reasons.append(f"{x} excluded: misleads (majority of candidate K)")
        elif a["excluded_journal_problem"]:
            reasons.append(f"{x} excluded: journal problem at {a['problem_K']} ({a['problem_share']:.0%} of candidate K)")
        else:
            remaining.append(x)
    sj = lambda x: arms[x]["s_journal_mean_over_candidate_K"] or 0.0
    adds = [x for x in remaining if arms[x]["class"] == "adds information"]
    d_avg = _gm([rep["per_K"][str(K)]["delta"] for K in cand if rep["per_K"][str(K)]["delta"] is not None])
    if adds:
        sc = {x: arms[x]["delta_mean_over_candidate_K"] for x in adds}
        best = max(sc, key=sc.get)
        tied = [x for x in adds if sc[best] - sc[x] <= d_avg]
        if len(tied) > 1:
            pick = "J2" if "J2" in tied else max(tied, key=sj)
            reasons.append(f"adds information: {adds}; within δ̄ {d_avg:.4g} of the best ({best}): {tied} → {pick}"
                           + (" (J2 among them)" if pick == "J2" else " (highest s_journal)"))
        else:
            pick = best
            reasons.append(f"adds information: {adds}; best mean Δ over candidate K: {best}")
    elif remaining:
        pick = "J2" if "J2" in remaining else max(remaining, key=sj)
        reasons.append(f"no arm adds information; remaining {remaining} → {pick}"
                       + (" (J2, the decided default P4)" if pick == "J2" else " (highest s_journal)"))
    else:
        pick = "J1"
        reasons.append("only J1 remains → J* = J1; journals profiled after the fit (mean of their articles' memberships)")
    return arms, {"J_star": pick, "remaining": remaining, "reasons": reasons, "candidate_K": cand,
                  "note": "default of the rule; the owner decides after the text check (step 5)"}


def robustness_summary(rep, cand, approach):
    out = {}
    for name in ("J2_logdamp vs J2", "J1_s065 vs J1", "J2_s065 vs J2", "J2_s065 vs J1_s065"):
        per = {K: rep["per_K"][str(K)]["robustness"].get(name) for K in cand}
        per = {K: v for K, v in per.items() if v}
        if not per:
            continue
        R, A = name.split(" vs ")
        s = {"class_per_K": {str(K): (v["M1"] or {}).get("class") for K, v in per.items()},
             "memberships_changed_per_K": {str(K): (v["M2"] or {}).get(f"changed_at_{M2_MIN:g}") for K, v in per.items()},
             "memberships_changed_uncorrected_per_K": {str(K): (v["M2"] or {}).get(f"changed_uncorrected_at_{M2_MIN:g}") for K, v in per.items()},
             "band": rep["bands"].get(R, {}).get("band"), "reference_band": rep["bands"].get(A, {}).get("band")}
        s["class"] = majority(list(s["class_per_K"].values()))
        s["band_moved"] = s["band"] != s["reference_band"]
        s["to_owner"] = bool(s["band_moved"] or any(s["memberships_changed_per_K"].values()))
        out[name] = s
    if "J2_s065 vs J2" in out:
        per = {K: rep["per_K"][str(K)]["robustness"].get("J2_s065 vs J2") for K in cand}
        soc = [v["social"] for v in per.values() if v]
        soc_better = any(any(x["better_beyond_1se"] for n, x in s.items() if n != "journal coherence") for s in soc)
        soc_worse = any(any(x["worse_beyond_1se"] for n, x in s.items() if n != "journal coherence") for s in soc)
        c = out["J2_s065 vs J2"]["class"]
        out["dominance_065"] = {"strictly_inferior": bool(c == "misleads" and not soc_better),
                                "strictly_superior": bool(c == "adds information" and not soc_worse),
                                "class": c, "a_social_reading_better_beyond_1se": soc_better,
                                "a_social_reading_worse_beyond_1se": soc_worse}
        out["dominance_065"]["stage3_065_arm_runs"] = not (out["dominance_065"]["strictly_inferior"] or out["dominance_065"]["strictly_superior"])
    if "J2_s065 vs J1_s065" in out:
        w1 = majority([((rep["per_K"][str(K)].get("M1") or {}).get("J2") or {}).get("class") for K in cand])
        out["D1_contrast_J2_vs_J1"] = {"at_W1": w1, "at_065": out["J2_s065 vs J1_s065"]["class"],
                                       "changes": w1 != out["J2_s065 vs J1_s065"]["class"]}
    return out


# ================================================================================================ driver
def collect(root, stage, label=_arm_of, approaches=None, Ks=None, groups=None):
    import glob
    from .aggregate import load_fit, latest_fits
    fits = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(root, stage, "*", "measures.json"))])
    units = defaultdict(lambda: defaultdict(list)); skipped = defaultdict(int)
    for f in fits:
        j = f["job"]; arm = label(f)
        if arm is None or (approaches and j["approach"] not in approaches) or (Ks and j["K"] not in Ks) \
                or (groups is not None and j["group"] not in groups):
            continue
        if not f["result"]["converged"]:
            skipped[f"{j['approach']}|{j['K']}|{arm}"] += 1; continue
        units[(j["approach"], j["K"], j["group"])][arm].append(f)
    for u in units.values():
        for a in u:
            u[a].sort(key=lambda f: f["job"]["seed"])
    return {k: dict(v) for k, v in units.items()}, dict(skipped)


def run(root, stage, build=None, workers=1, coherence=True, label=_arm_of, approaches=None, Ks=None, groups=None):
    from . import data as D
    units, skipped = collect(root, stage, label, approaches, Ks, groups)
    b = D.Build(build or D.BUILD); r, c, _ = b.ties("article_journal")
    jour = -np.ones(b.sizes["article"], dtype=np.int64); jour[r] = c
    _G.update(units=units, journal_of_article=jour, coherence=coherence)
    keys = sorted(units)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = dict(pool.imap_unordered(unit, keys))
    else:
        res = dict(map(unit, keys))
    rep = {"_status": {"stage": stage, "build": b.path, "units": len(keys), "non_converged_left_out": skipped,
                       "fits_per_unit_arm": {f"{k[0]}|{k[1]}|{k[2]}": {a: len(v) for a, v in units[k].items()} for k in keys}}}
    for a in sorted({k[0] for k in keys}):
        rep[f"approach_{a}"] = summarise(res, a)
    rep["_units"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}
    return rep


def _f(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) and x is not None else "—"


def print_report(rep):
    for name, s in rep.items():
        if not name.startswith("approach_"):
            continue
        print(f"\n=== {name}: candidate K {s['candidate_K']}  (bands: " +
              ", ".join(f"{a} {b['band']}" for a, b in s["bands"].items()) + ")")
        for K, e in s["per_K"].items():
            m1 = e.get("M1") or {}
            print(f"  K {K}: δ {_f(e['delta'], '.5f')}  half-split margin {_f(e['half_split_margin'], '.5f')}")
            for x in COMPARED:
                if x not in m1:
                    continue
                m2, m4, m3 = e["M2"].get(x) or {}, e["M4"].get(x) or {}, e["M3"].get(x) or {}
                print(f"    {x:4s} M1 Δ {_f(m1[x].get('mean'), '+.5f')}±{_f(m1[x].get('se'), '.5f')} {m1[x]['class']:16s}"
                      f" (uncorr. {m1[x].get('class_uncorrected')})"
                      f" | M2 drop {_f(m2.get('drop'), '+.3f')} p_holm {_f(m2.get('p_holm'), '.4f')} changed {m2.get('changed')}"
                      f" | M4 Δ̄ {_f(m4.get('delta_bar'), '+.2f')} p_holm {_f(m4.get('p_holm'), '.4f')} more {m4.get('more')}"
                      f" | M3 s_j {_f(m3.get('s_journal'), '.4f')} s_sem {_f(m3.get('s_semantic'), '.3f')}"
                      f" | problem {(e['problem'].get(x) or {}).get('problem')}")
        print("  arms:", {x: (a["class"], a["problem_K"]) for x, a in s["arms"].items()})
        print("  decision:", s["decision"].get("J_star"), "—", "; ".join(s["decision"].get("reasons", [])))
        if s.get("robustness_summary"):
            print("  robustness:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("class", "band_moved", "to_owner", "strictly_inferior", "strictly_superior", "stage3_065_arm_runs", "changes")}
                                               for k, v in s["robustness_summary"].items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT); ap.add_argument("--stage", default="stage1")
    ap.add_argument("--build", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--no-coherence", action="store_true")
    a = ap.parse_args()
    rep = run(a.root, a.stage, a.build, a.workers, not a.no_coherence)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else
              (int(x) if isinstance(x, np.integer) else (float(x) if isinstance(x, np.floating) else str(x))))
    print_report(rep)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
