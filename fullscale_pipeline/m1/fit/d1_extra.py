# D1 — the four readings not in fit/d1_analysis.py (owner 2026-11-28: "yes, write the four missing D1 readings"; RL-429).
# Written while the D1 analysis job runs, before any D1 result is read. Readings only; nothing here changes a decision.
#
# 1. idea   — idea coherence of article groupings (community readings of the robustness arms, RL-403; definition as
#             FINDINGS §33 / fit/weight_curve_community.py), per stage-1 fit, over all article pairs:
#               idea_corr_w      Pearson(co-membership Σ_k p_ik p_jk, W-aware cosine of the child-hyperedge sets) — the
#                                FINDINGS §33 definition, kept as the reference reading;
#               idea_excess      mean co-membership of pairs sharing ≥ 1 child hyperedge − mean over all pairs;
#               idea_corr_cos, idea_excess_cos   the same with the cosine of the two membership vectors instead of
#                                co-membership; idea_spearman_cos_size   Spearman(cosine, W-aware cosine) with both
#                                articles' sizes (log child-hyperedge counts, summed) partialled out — beside the reference
#                                because co-membership mixes how concentrated each membership is with how similar two
#                                articles are (RL-421: Spearman(size, entropy) −0.46 to −0.60).
#             People coherence (co-membership excess of pairs sharing a social entity) is already in d1_analysis (per fit,
#             "coherence"). Summary per approach and K: arm group means (mean over seeds), then each J arm − J1, each
#             robustness pair R − A (J2_logdamp − J2, J1_s065 − J1, J2_s065 − J2) and the contrast J2_s065 − J1_s065, as
#             mean ± SE over hold-out groups, relative to the reference and in the reference's seed SD. Reported only.
#             Heavy (all 4,500 fits): run on CSF, `bash fit/slurm/d1_extra_idea.sh` (≈ 5 CPU-h, 16 workers).
# 2. entity-totals — the entity-total diagnostic of the damping arm (RL-274, amendment c; design doc 2026-11-19), now
#             over the 10 hold-out groups of stage 1: for each damped relation, entities on the damped side by their number
#             of visible ties d (bands 1, 2–3, 4–7, 8–15, 16–31, 32+): entities, mean entity total and the band's share of
#             the relation's visible mass, undamped vs damped (× log(1+d)/d, mass restored); author×article also by team
#             size. Mean, min and max over groups. Data only, no fits (minutes).
# 3. recheck — the required stage-2 re-check (pre-registration §5, decided 2026-11-05): J1, J* and the runner-up at the
#             final K₁, K₂ on D1-M1, M2, M4, with d1_analysis's own code (unit, summarise, decide) restricted to those arms
#             and K (candidate K = the final K). Runner-up = the arm the decision rule picks when J* is left out (none if
#             J* = J1). Proposed operational "contradiction" (for the owner): at a final K the rule, applied to
#             {J1, J*, runner-up}, does not return J*, or J* is classed misleads or has a journal problem there; both
#             results are then put to the owner. Fits: stage-1 fits when the final K is coarse (3, 5, 8, 13, 20); otherwise
#             `recheck-jobs` writes the job list (10 groups × 5 seeds per arm and K, settings of stage 1; λ_child given
#             explicitly, since the pass-1 basis exists only at coarse K) — not submitted.
# 4. text   — text-check material (pre-registration §5 step 5: "top articles and child hyperedges of journal-shaped
#             communities"): per approach, arm (default J* and J1 from the D1 result) and candidate K, the fits of hold-out
#             group 1 (seeds 1–5); per fit the journal-shaped communities by D1-M4's own test (articles covered in that
#             fit, coverage ≥ 0.5, with a journal; 999 journal-label shuffles; p < 0.05 and j_k ≥ 0.25); per community the
#             journal, j_k, size, how many of the journal's articles are members (share ≥ 0.3), the 10 articles with the
#             largest share in k (PMCID, title, journal, share) and the 15 child hyperedges carrying most of k's
#             weighted child mass (typed atoms), plus the 10 articles of that journal with the largest share in k. Writes Markdown for reading + JSON. Titles from the OpenAlex works cache
#             (by PMID), journal names from the NLM catalogue cache — public metadata only, no author data.
#
# Usage (from fullscale_pipeline/m1, PYTHONPATH=.):
#   python -m fit.d1_extra idea --stage stage1 --out fit/results/stage1/d1_idea_coherence.json --workers 16
#   python -m fit.d1_extra entity-totals --out fit/results/stage1/d1_entity_totals.json
#   python -m fit.d1_extra recheck --d1 fit/results/stage1/d1_analysis.json --final-K 1:13,32 2:13,20 \
#       --stage-for 13:stage1 20:stage1 32:stage2_recheck --out fit/results/stage2/d1_recheck.json
#   python -m fit.d1_extra recheck-jobs --approach 1 --K 32 --arms J1,J2,J4 --lambda-child 12.3 --out <jobs.jsonl>
#   python -m fit.d1_extra text --d1 fit/results/stage1/d1_analysis.json --out fit/results/stage1/d1_text_check
import os
os.environ.setdefault("M1_ALIGNED", "0")
import argparse, contextlib, csv, json, math
from collections import defaultdict
import numpy as np, scipy.sparse as sp
from scipy.stats import rankdata

from . import d1_analysis as A

ROOT = A.ROOT
FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
TITLES_PMID = FP + "corpus_statistics/article_types_valid.csv"           # pmcid → pmid
OPENALEX = FP + "diagnostics/author_coverage/openalex_works.json"          # pmid → title (untracked cache)
NLM = FP + "diagnostics/author_coverage/nlm_journals.json"                 # NLM ID → journal title (untracked cache)
PAIRS = tuple(("J1", x) for x in A.COMPARED) + tuple((ref, r) for r, ref in A.ROBUST) + (("J1_s065", "J2_s065"),)


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return str(x)


# ----------------------------------------------------------------------------------------------- 1. idea coherence
_CONTENT = {}


def _content(rds):
    from .swap_link_measure import _WN
    ca = rds["child_article"]; n = ca.shape[0]
    X = sp.csr_matrix((np.ones(len(ca.tie_row)), (ca.tie_row, ca.tie_col)), shape=ca.shape)
    size = np.asarray(X.sum(1)).ravel()
    Xw = (X + X @ _WN).tocsr(); nw = np.sqrt(np.asarray(Xw.multiply(Xw).sum(1)).ravel())
    iu = np.triu_indices(n, 1)
    jw = ((Xw @ Xw.T).toarray() / np.maximum(np.outer(nw, nw), 1e-300))[iu]
    sh = (X @ X.T).toarray()[iu] > 0
    ls = np.log(np.maximum(size, 1)); Zm = np.column_stack([np.ones(len(jw)), rankdata(ls[iu[0]] + ls[iu[1]])])
    rj = rankdata(jw); res_jw = rj - Zm @ np.linalg.lstsq(Zm, rj, rcond=None)[0]
    return {"iu": iu, "jw": jw, "sh": sh, "Zm": Zm, "res_jw": res_jw}


def idea_one(f, rds, c):
    from . import measures as M
    from .aggregate import load_UZ
    U, Z = load_UZ(f["dir"])
    P = M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()})["article"]["P"]
    r, s = c["iu"]
    cm = (P @ P.T)[c["iu"]]
    nP = np.sqrt((P * P).sum(1)); cos = cm / np.maximum(nP[r] * nP[s], 1e-300)
    rc = rankdata(cos); res = rc - c["Zm"] @ np.linalg.lstsq(c["Zm"], rc, rcond=None)[0]
    return {"idea_corr_w": float(np.corrcoef(cm, c["jw"])[0, 1]), "idea_excess": float(cm[c["sh"]].mean() - cm.mean()),
            "idea_corr_cos": float(np.corrcoef(cos, c["jw"])[0, 1]), "idea_excess_cos": float(cos[c["sh"]].mean() - cos.mean()),
            "idea_spearman_cos_size": float(np.corrcoef(res, c["res_jw"])[0, 1]),
            "co_membership_mean": float(cm.mean()), "cosine_mean": float(cos.mean())}


def _idea_unit(task):
    """One (approach, group): the content readings once, then every fit of that hold-out group."""
    from .aggregate import _rds_of
    key, fits = task
    c = None; out = []
    for f in fits:
        rds = {rd.name: rd for rd in _rds_of(f["job"])}
        if c is None:
            c = _content(rds)
        j = f["job"]
        try:
            out.append({"approach": j["approach"], "K": j["K"], "group": j["group"], "seed": j["seed"],
                        "arm": A._arm_of(f), "dir": os.path.basename(f["dir"]), **idea_one(f, rds, c)})
        except Exception as e:                    # one failing fit must not lose the others; reported
            out.append({"approach": j["approach"], "K": j["K"], "group": j["group"], "seed": j["seed"],
                        "arm": A._arm_of(f), "error": repr(e)})
    return out


IDEA_KEYS = ("idea_corr_w", "idea_excess", "idea_corr_cos", "idea_excess_cos", "idea_spearman_cos_size")


def summarise_pairs(rows, keys, pairs=PAIRS):
    """rows: per fit dicts with approach, K, group, seed, arm and reading keys. Per approach and K: arm group means and
    pairwise differences (B − A) as mean ± SE over the shared hold-out groups, relative to A and in A's seed SD."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" not in r:
            by[(r["approach"], r["K"], r["arm"], r["group"])]["_"].append(r)
    out = {}
    for (a, K) in sorted({(r["approach"], r["K"]) for r in rows}):
        e = {"arms": {}, "pairs": {}}
        arms = sorted({r["arm"] for r in rows if r["approach"] == a and r["K"] == K})
        gm = {}; sd = {}
        for arm in arms:
            groups = sorted({g for (aa, KK, x, g) in by if (aa, KK, x) == (a, K, arm)})
            gm[arm] = {k: {g: float(np.mean([r[k] for r in by[(a, K, arm, g)]["_"]])) for g in groups} for k in keys}
            sd[arm] = {k: float(np.nanmean([np.std([r[k] for r in by[(a, K, arm, g)]["_"]], ddof=1)
                                            if len(by[(a, K, arm, g)]["_"]) > 1 else np.nan for g in groups])) for k in keys}
            e["arms"][arm] = {k: {"mean": float(np.mean(list(gm[arm][k].values()))), "seed_sd": sd[arm][k],
                                  "groups": len(groups)} for k in keys}
        for ref, x in pairs:
            if ref not in gm or x not in gm:
                continue
            d = {}
            for k in keys:
                gs = sorted(set(gm[ref][k]) & set(gm[x][k]))
                v = np.array([gm[x][k][g] - gm[ref][k][g] for g in gs])
                if len(v) < 2:
                    continue
                m, se = float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))
                base = float(np.mean([gm[ref][k][g] for g in gs]))
                d[k] = {"diff": m, "se": se, "groups": len(v), "relative": m / base if base else None,
                        "in_ref_seed_sd": m / sd[ref][k] if sd[ref][k] else None}
            e["pairs"][f"{x} vs {ref}"] = d
        out[f"approach_{a}|K_{K}"] = e
    return out


def run_idea(root, stage, workers):
    units, skipped = A.collect(root, stage)
    by_group = defaultdict(list)
    for (a, K, g), arms in units.items():
        for arm, fs in arms.items():
            by_group[(a, g)].extend(fs)
    tasks = sorted(by_group.items(), key=lambda t: -len(t[1]))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_idea_unit, tasks, chunksize=1)
    else:
        res = list(map(_idea_unit, tasks))
    rows = [r for rs in res for r in rs]
    return {"_status": {"stage": stage, "fits": len(rows), "errors": sum("error" in r for r in rows),
                        "non_converged_left_out": skipped},
            "summary": summarise_pairs(rows, IDEA_KEYS), "per_fit": rows}


def print_idea(rep):
    print(rep["_status"])
    for cell, e in rep["summary"].items():
        print("==", cell)
        for p, d in e["pairs"].items():
            print(f"  {p:22s} " + "  ".join(f"{k} {v['diff']:+.4f}±{v['se']:.4f}" for k, v in d.items()
                                             if k in ("idea_corr_w", "idea_spearman_cos_size")))


# --------------------------------------------------------------------------------------------- 2. entity totals
def run_entity_totals(build=None, groups=range(10)):
    from . import data as D
    from .damping_entity_totals import by_band, band, LABELS
    b = D.Build(build or D.BUILD); per = defaultdict(list)
    for a in (1, 2):
        for g in groups:
            base = {rd.name: rd for rd in D.fit_data(b, a, group=g)}
            damp = {rd.name: rd for rd in D.fit_data(b, a, group=g, log_damping="data")}
            for name, rd in damp.items():
                if rd.damping is None:
                    continue
                side = rd.damping["side"]; n = rd.shape[0] if side == "row" else rd.shape[1]
                ent = rd.tie_row if side == "row" else rd.tie_col; d = np.bincount(ent, minlength=n)
                r = {"q": rd.damping["q"], "by_degree": {"undamped": by_band(ent, base[name].tie_val, n, d),
                                                          "damped": by_band(ent, rd.tie_val, n, d)}}
                if name == "author_article":
                    m = rd.shape[1]; team = np.bincount(rd.tie_col, minlength=m)
                    r["by_team_size"] = {"undamped": by_band(rd.tie_col, base[name].tie_val, m, team),
                                         "damped": by_band(rd.tie_col, rd.tie_val, m, team)}
                per[(name, a)].append(r)
            print(f"approach {a} group {g + 1} done", flush=True)
    out = {}
    for (name, a), rs in per.items():
        e = {"approach": a, "groups": len(rs), "q": [min(r["q"] for r in rs), max(r["q"] for r in rs)], "bands": {}}
        for part in ("by_degree", "by_team_size"):
            if part not in rs[0]:
                continue
            for L in LABELS:
                for state in ("undamped", "damped"):
                    v = [r[part][state][L]["mass_share"] for r in rs]
                    t = [r[part][state][L]["mean_total"] for r in rs if r[part][state][L]["mean_total"] is not None]
                    e["bands"].setdefault(part, {}).setdefault(L, {})[state] = {
                        "entities_mean": float(np.mean([r[part][state][L]["entities"] for r in rs])),
                        "mass_share_mean": float(np.mean(v)), "mass_share_min": float(np.min(v)), "mass_share_max": float(np.max(v)),
                        "mean_total_mean": float(np.mean(t)) if t else None}
        out[f"{name}|approach_{a}"] = e
    return out


def print_entity_totals(rep):
    for name, e in rep.items():
        print(f"== {name} (approach {e['approach']}, {e['groups']} groups, q {e['q'][0]:.3f}–{e['q'][1]:.3f})")
        for part, bands in e["bands"].items():
            print(f"  {part}:")
            for L, s in bands.items():
                u, d = s["undamped"], s["damped"]
                if u["entities_mean"]:
                    print(f"    {L:6s} n {u['entities_mean']:8.1f}  mass share {u['mass_share_mean']:.3f} "
                          f"[{u['mass_share_min']:.3f}, {u['mass_share_max']:.3f}] → {d['mass_share_mean']:.3f} "
                          f"[{d['mass_share_min']:.3f}, {d['mass_share_max']:.3f}]")


# ----------------------------------------------------------------------------------------------- 3. stage-2 re-check
@contextlib.contextmanager
def _compared(arms):
    old = A.COMPARED
    A.COMPARED = tuple(arms)
    try:
        yield
    finally:
        A.COMPARED = old


def runner_up(rep_approach):
    """The arm the decision rule picks when J* is left out (None if J* = J1 or nothing else remains)."""
    js = rep_approach["decision"]["J_star"]
    if js in (None, "J1"):
        return None
    others = [x for x in A.COMPARED if x != js]
    with _compared(others):
        _, dec = A.decide(rep_approach, rep_approach["candidate_K"])
    return dec["J_star"] if dec["J_star"] != "J1" else None


def run_recheck(d1_path, final_K, stage_for, root, workers):
    d1 = json.load(open(d1_path)); out = {"_source": d1_path, "approaches": {}}
    for a, Ks in final_K.items():
        r1 = d1[f"approach_{a}"]; js = r1["decision"]["J_star"]; ru = runner_up(r1)
        arms = [x for x in ("J1", js, ru) if x]
        e = {"J_star_stage1": js, "runner_up": ru, "arms": arms, "per_K": {}}
        for K in Ks:
            stage = stage_for.get(K, "stage1")
            label = lambda f, arms=arms: (A._arm_of(f) if A._arm_of(f) in arms else None)
            with _compared([x for x in arms if x != "J1"]):
                rep = A.run(root, stage, workers=workers, label=label, approaches=[a], Ks=[K])
                ra = rep[f"approach_{a}"]
                # candidate K of the re-check = the final K itself
                arms_K, dec = A.decide(ra, [K])
            m1 = (ra["per_K"].get(str(K)) or {}).get("M1") or {}
            prob = (ra["per_K"].get(str(K)) or {}).get("problem") or {}
            contradiction = (dec["J_star"] != js) or (js != "J1" and ((m1.get(js) or {}).get("class") == "misleads"
                                                                       or bool((prob.get(js) or {}).get("problem"))))
            e["per_K"][str(K)] = {"stage": stage, "decision_at_K": dec, "arms": arms_K, "M1": m1, "problem": prob,
                                  "contradiction_proposed_definition": bool(contradiction)}
        out["approaches"][str(a)] = e
    return out


def recheck_jobs(approach, K, arms, lambda_child, study="A", groups=range(10), seeds=range(1, 6)):
    """Fits for the re-check at a non-coarse K: stage-1 settings, the given λ_child (absolute), arms from STAGE1_ARMS."""
    from . import plan as PL
    stage = "stage2_recheck" if study == "A" else "B_stage2_recheck"
    terms = {"lambda_child": float(lambda_child), "w_form": "contrast"}
    out = []
    for name in arms:
        arm = dict(PL.STAGE1_ARMS[name], j=name)
        for g in groups:
            for s in seeds:
                j = dict(PL.job(stage, approach, g, K, 1, s, PL.FRACTION[1], dtype="float32", terms=terms, arm=arm, max_epochs=8000),
                         social_link_start="diagonal", offdiag_level=0.1, offdiag_semantic=True)
                if study == "B":
                    j["build"] = PL.BUILD_B
                out.append(j)
    return out


# ------------------------------------------------------------------------------------------- 4. text-check material
def _metadata():
    pmid = {r["pmcid"]: r["pmid"] for r in csv.DictReader(open(TITLES_PMID)) if r.get("pmid")}
    title_by_pmid = {}
    if os.path.exists(OPENALEX):
        W = json.load(open(OPENALEX))
        for w in W["by_id"].values():
            if w and w.get("pmid"):
                title_by_pmid[str(w["pmid"])] = w.get("title") or ""
    nlm = json.load(open(NLM)) if os.path.exists(NLM) else {}
    return pmid, title_by_pmid, {k: (v or {}).get("title", "") for k, v in nlm.items()}


_BUILDS = {}


def _build_journal(path):
    from . import data as D
    if path not in _BUILDS:
        b = D.Build(path); r, c, _ = b.ties("article_journal")
        jour = -np.ones(b.sizes["article"], dtype=np.int64); jour[r] = c
        _BUILDS[path] = (b, jour)
    return _BUILDS[path]


def text_fit(f, rng, n_sh=A.N_COMM_SHUFFLES, n_art=10, n_child=15):
    from . import measures as M, data as D
    from .aggregate import load_UZ, _rds_of
    U, Z = load_UZ(f["dir"]); rds = _rds_of(f["job"])
    wm = M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()})
    cov = M.entity_coverage(U, Z, rds)
    b, jour = _build_journal(f["job"].get("build") or D.BUILD)
    keep = cov["article"] >= A.COVERAGE_C; kj = keep & (jour >= 0)
    uj, lab = np.unique(jour[kj], return_inverse=True)
    P = wm["article"]["P"]; res = A.journal_shaped(P[kj], lab, len(uj), rng, n_sh)
    arts = np.nonzero(kj)[0]; Yc = wm["child"]["Y"]
    comms = []
    for d in res["shaped_detail"]:
        k = d["k"]; jid = int(uj[d["journal"]])
        top = arts[np.argsort(-P[arts, k])[:n_art]]
        mine = arts[jour[arts] == jid]; topj = mine[np.argsort(-P[mine, k])[:n_art]]
        ch = np.argsort(-Yc[:, k])[:n_child]; tot = max(Yc[:, k].sum(), 1e-300)
        comms.append({**d, "journal_nlm": b.index["journals"][jid],
                      "top_articles": [{"pmcid": b.index["articles"][i], "share": float(P[i, k]),
                                        "journal_nlm": b.index["journals"][jour[i]] if jour[i] >= 0 else None,
                                        "same_journal": bool(jour[i] == jid)} for i in top],
                      "top_journal_articles": [{"pmcid": b.index["articles"][i], "share": float(P[i, k])} for i in topj],
                      "top_children": [{"id": b.index["children"][i], "atoms": b.index["child_atoms"][b.index["children"][i]],
                                        "share_of_k_child_mass": float(Yc[i, k] / tot)} for i in ch]})
    return {"journal_shaped": res["journal_shaped"], "communities": comms, "covered_with_journal": int(kj.sum())}


def run_text(d1_path, root, stage, arms_extra=(), group=0, out_prefix=None):
    d1 = json.load(open(d1_path)); pmid, tit, jn = _metadata()
    units, _ = A.collect(root, stage)
    rep = {}
    for a in (1, 2):
        ra = d1.get(f"approach_{a}")
        if not ra:
            continue
        arms = list(dict.fromkeys([ra["decision"]["J_star"], "J1", *arms_extra]))
        for K in ra["candidate_K"]:
            for arm in arms:
                fits = (units.get((a, K, group)) or {}).get(arm) or []
                for f in fits:
                    rng = np.random.default_rng([a, K, group, 107, f["job"]["seed"]])
                    rep[f"approach {a} | K {K} | {arm} | group {group + 1} | seed {f['job']['seed']}"] = text_fit(f, rng)
    md = ["# D1 text-check material (pre-registration §5 step 5)", "",
          f"Source: `{d1_path}`; fits of hold-out group {group + 1}; journal-shaped = D1-M4 test (p < 0.05, j_k ≥ 0.25) on the "
          "articles covered in that fit. Shares: an article's membership share in community k; a child hyperedge's share "
          "of k's weighted child mass. Titles: OpenAlex; journal names: NLM catalogue.", ""]
    for name, r in rep.items():
        md.append(f"## {name} — {r['journal_shaped']} journal-shaped communities")
        for cm in r["communities"]:
            md += ["", f"### Community {cm['k']}: {jn.get(cm['journal_nlm'], cm['journal_nlm'])} (j_k {cm['j_k']:.2f}; "
                       f"size {cm['size']:.3f}; {cm['journal_articles']} journal articles, {cm['share_ge_0.3']:.0%} with share ≥ 0.3)",
                   "", "| Article | Title | Journal | Share | Same journal |", "|---|---|---|---|---|"]
            for t in cm["top_articles"]:
                md.append(f"| {t['pmcid']} | {tit.get(pmid.get(t['pmcid'], ''), '').replace('|', '/')} | "
                          f"{jn.get(t['journal_nlm'], t['journal_nlm'] or '')} | {t['share']:.2f} | {'yes' if t['same_journal'] else ''} |")
            md += ["", f"Articles of {jn.get(cm['journal_nlm'], cm['journal_nlm'])} with the largest share in k:", "",
                   "| Article | Title | Share |", "|---|---|---|"]
            for t in cm["top_journal_articles"]:
                md.append(f"| {t['pmcid']} | {tit.get(pmid.get(t['pmcid'], ''), '').replace('|', '/')} | {t['share']:.2f} |")
            md += ["", "| Child hyperedge (typed atoms) | Share of k's child mass |", "|---|---|"]
            for ch in cm["top_children"]:
                md.append(f"| {' '.join(ch['atoms'])} | {ch['share_of_k_child_mass']:.4f} |")
        md.append("")
    if out_prefix:
        json.dump(rep, open(out_prefix + ".json", "w"), indent=1, default=_json_default)
        open(out_prefix + ".md", "w").write("\n".join(md))
    return rep


# ------------------------------------------------------------------------------------------------------------ main
def _parse_final_K(items):
    out = {}
    for it in items:
        a, ks = it.split(":"); out[int(a)] = [int(k) for k in ks.split(",")]
    return out


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("idea"); p.add_argument("--root", default=ROOT); p.add_argument("--stage", default="stage1")
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("entity-totals"); p.add_argument("--build", default=None); p.add_argument("--out", required=True)
    p = sub.add_parser("recheck"); p.add_argument("--d1", required=True); p.add_argument("--final-K", nargs="+", required=True)
    p.add_argument("--stage-for", nargs="*", default=[]); p.add_argument("--root", default=ROOT)
    p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("recheck-jobs"); p.add_argument("--approach", type=int, required=True); p.add_argument("--K", type=int, required=True)
    p.add_argument("--arms", required=True); p.add_argument("--lambda-child", type=float, required=True)
    p.add_argument("--study", default="A"); p.add_argument("--out", required=True)
    p = sub.add_parser("text"); p.add_argument("--d1", required=True); p.add_argument("--root", default=ROOT)
    p.add_argument("--stage", default="stage1"); p.add_argument("--arms-extra", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd != "recheck-jobs":
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.cmd == "idea":
        rep = run_idea(a.root, a.stage, a.workers); json.dump(rep, open(a.out, "w"), indent=1, default=_json_default); print_idea(rep)
    elif a.cmd == "entity-totals":
        rep = run_entity_totals(a.build); json.dump(rep, open(a.out, "w"), indent=1, default=_json_default); print_entity_totals(rep)
    elif a.cmd == "recheck":
        sf = {int(k): v for k, v in (x.split(":") for x in a.stage_for)}
        rep = run_recheck(a.d1, _parse_final_K(a.final_K), sf, a.root, a.workers)
        json.dump(rep, open(a.out, "w"), indent=1, default=_json_default)
        for ap_, e in rep["approaches"].items():
            print(f"approach {ap_}: J* {e['J_star_stage1']}, runner-up {e['runner_up']}")
            for K, x in e["per_K"].items():
                print(f"  K {K}: rule → {x['decision_at_K']['J_star']}; contradiction {x['contradiction_proposed_definition']}")
    elif a.cmd == "recheck-jobs":
        jobs = recheck_jobs(a.approach, a.K, a.arms.split(","), a.lambda_child, a.study)
        with open(a.out, "w") as fh:
            for j in jobs:
                fh.write(json.dumps(j) + "\n")
        print(f"{len(jobs)} jobs → {a.out} (not submitted)")
    elif a.cmd == "text":
        rep = run_text(a.d1, a.root, a.stage, [x for x in a.arms_extra.split(",") if x], out_prefix=a.out)
        print(f"{len(rep)} fits; wrote {a.out}.md / .json")


if __name__ == "__main__":
    main()
