# Diagonal-start test — reading (ticket 107; owner 2026-11-17/18). Expected outcomes and the reading rule are stated in
# plans/ticket107_start_test.md BEFORE the fits are read; this script applies them unchanged.
# Revision 2026-11-18 (owner: "I approve your revised reading substituting the midpoint line"; anchor: "for child*article
# we should still have >0.5 for within community main diagonal"; both directions, failures studied; article×article
# included under both losses). All readings use the ALIGNED numbering (GLOSSARY "Alignment"; RL-260) — run
# `python -m fit.align --stages start_test` first.
# Arms (same data, loss, K*, hold-out group 1, seeds 1–10, terms off, same membership start per seed):
#   reference = stage-0 pass-1 fits (hybrid start: semantic cores diagonal, social cores random);
#   diagonal  = start_test fits (hybrid start with every social core started diagonal, init.diagonal_social_links).
# Per fit:
#   - convergence; data loss (final epoch) and per-relation loss; held-out existence and value (decided two-part score),
#     existence per relation with hidden cells;
#   - DIAGONAL RULE, every relation except child×article (owner's relaxed rule): community k meets it in a relation if its
#     own-numbered cell of the parts table, part(k,k), exceeds every other cell of its row AND of its column. Measure:
#     share of the relation's reconstructed tie weight (half credit) in communities that meet it; row-only and
#     column-only shares and counts beside it;
#   - ANCHOR GUARD, child×article: within share Σ_k part(k,k) / T, and the communities whose own-numbered cell is more
#     than half of their child×article tie weight on both sides (count, share of tie weight). The ">0.5" requirement
#     applies per community (ANCHOR_UNIT = "community", owner 2026-11-18): the guard holds if every community passes in
#     every converged fit of the diagonal arm; paired differences in the count and tie-weight share are reported;
#   - FAILURES (owner: "it is interesting to study cases when row or column is not beaten"): every community that fails
#     the rule in a direction, with its own share, its main partner (number and share) and its tie weight; and per
#     community number the main partner in every relation (to read chains such as authors of k → organisations of l →
#     journals of l2);
#   - effective number of partner communities per relation (reported, no longer classifying);
#   - presence summaries and detachment (as before).
# Per arm: stability (aggregate.group_checks), and RECURRENCE of failures across seeds: each facet is matched across two
# seeds on its own (measures.match on that facet's weighted memberships, independent of numbering); a failure of
# community k toward main partner l recurs in another seed if the matched community fails in the same direction toward
# the matched partner. A failure "recurs" if it does so in more than half of the arm's other converged seeds.
# Paired comparison by seed (both converged): difference diagonal − reference, mean, SE = SD/√n; "better"/"worse" beyond
# 2 SE (decided 2026-11-17), otherwise "within noise".
# Outcome per relation, squared error (pre-stated): rule share higher beyond 2 SE in the diagonal arm and fit not worse
# and anchor guard met → 1 "start artefact"; higher but fit worse or anchor guard not met → 3 "trade-off"; otherwise
# → 2 "start not the cause" (recurrence of failures then reported per arm). Setting: 4 "mixed" if relations differ.
# KL is the control: "changed" if data loss, existence, value or any relation's rule share differs beyond 2 SE.
# Usage: python -m fit.start_test_analysis --out fit/results/stage0/start_test.json
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
os.environ["M1_ALIGNED"] = "1"
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits, group_checks

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
LOSS = {2: "squared_error", 1: "KL"}
ANCHOR = "child_article"
ANCHOR_UNIT = "community"   # DECIDED 2026-11-18 (owner: "I think we should seek to have 0.5 per community"): every community
                            # in every converged fit of the arm; "fit" (within share > 0.5) kept for dry runs only
SINGLE = {"author_affiliation": "organisation", "article_affiliation": "organisation", "article_journal": "journal"}


def load(stage, diagonal):
    fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, D.stage(stage), "*", "measures.json"))]
    out = defaultdict(list)
    for f in latest_fits(fs):
        j = f["job"]
        if j["group"] != 0 or j["seed"] not in range(1, 11) or j["K"] != KSTAR[j["beta"]] or (j.get("terms") or {}) \
                or (j.get("arm") or {}) or (j.get("social_link_start") == "diagonal") != diagonal:
            continue
        out[(j["approach"], j["beta"])].append(f)
    return out


def diagonal_rule(P):
    """Owner's relaxed rule on one parts table (aligned numbering)."""
    K = P.shape[0]; T = P.sum(); d = np.diag(P)
    off = P.copy(); np.fill_diagonal(off, -np.inf)
    rmain, cmain = off.argmax(1), off.argmax(0)
    row_ok = d > off.max(1); col_ok = d > off.max(0); both = row_ok & col_ok
    rw, cw = P.sum(1), P.sum(0); h = 0.5 * (rw + cw) / T
    fails = []
    for k in range(K):
        if not row_ok[k]:
            fails.append({"k": k, "side": "row", "own": float(d[k] / max(rw[k], 1e-300)), "main": int(rmain[k]),
                          "main_share": float(P[k, rmain[k]] / max(rw[k], 1e-300)), "weight": float(rw[k] / T)})
        if not col_ok[k]:
            fails.append({"k": k, "side": "column", "own": float(d[k] / max(cw[k], 1e-300)), "main": int(cmain[k]),
                          "main_share": float(P[cmain[k], k] / max(cw[k], 1e-300)), "weight": float(cw[k] / T)})
    return {"both_share": float(h[both].sum()), "row_share": float((rw / T)[row_ok].sum()),
            "col_share": float((cw / T)[col_ok].sum()), "n_both": int(both.sum()), "n_row": int(row_ok.sum()),
            "n_col": int(col_ok.sum()), "K": K}, fails


def anchor_guard(P):
    T = P.sum(); d = np.diag(P); rw, cw = P.sum(1), P.sum(0); h = 0.5 * (rw + cw) / T
    ok = (d / np.maximum(rw, 1e-300) > 0.5) & (d / np.maximum(cw, 1e-300) > 0.5)
    return {"within_share": float(d.sum() / T), "K": int(len(d)), "n_communities_over_half": int(ok.sum()),
            "share_communities_over_half": float(h[ok].sum())}


def effective_partners(P):
    P = P / P.sum(); rw = P.sum(1); Pr = P / np.maximum(rw[:, None], 1e-300)
    ent = -(Pr * np.log(np.clip(Pr, 1e-300, None))).sum(1)
    return float((rw * np.exp(ent)).sum() / rw.sum())


def per_fit(f, rds):
    m = f["measures"]; r = f["result"]
    out = {"seed": f["job"]["seed"], "converged": bool(r["converged"]) or INCLUDE_UNCONVERGED, "converged_actual": bool(r["converged"]),
           "epochs": r["epochs"],
           "data_loss": float(r["data_loss_history"][-1]), "relation_loss": r["final_relation_loss"],
           "existence": m["heldout_two_part_combined"]["existence"], "value": m["heldout_two_part_combined"]["value"],
           "existence_by_relation": {k: v["ap_within_class"] for k, v in m["heldout_two_part"].items() if v}}
    dp = m["domain_presence"]; S = np.array(dp["semantic"]["mean"]); O = np.array(dp["social"]["mean"])
    share = np.mean([np.array(v["half_credit"]) for v in m["presence"].values()], 0)
    out["mixed_0.1"] = float(share[(S >= 0.1) & (O >= 0.1)].sum())
    out["weaker_median"] = float(np.median(np.minimum(S, O)))
    U, Z = load_UZ(f["dir"])
    fp = {x: np.array(v) for x, v in m["facet_presence"].items()}
    out["rule"], out["failures"], out["partners"], out["main_partner"] = {}, {}, {}, {}
    det = 0
    for name, rd in rds.items():
        P, _ = M.parts(U, Z, rd)
        out["partners"][name] = effective_partners(P)
        off = P.copy(); np.fill_diagonal(off, -np.inf)
        out["main_partner"][name] = {"row": P.argmax(1).tolist(), "column": P.argmax(0).tolist()}
        if name == ANCHOR:
            out["anchor"] = anchor_guard(P)
        else:
            out["rule"][name], out["failures"][name] = diagonal_rule(P)
        if name in SINGLE:
            side = P.shape[0] * P.sum(0) / P.sum()
            det += int(((fp[rd.row_facet] >= 0.5) & (side < 0.1)).sum())
    out["detached"] = det
    rel = {k: tuple(v) for k, v in r["relations"].items()}
    return out, M.weighted_memberships(U, Z, rel)


def recurrence(F, W, rds):
    """F: per-fit dicts of one arm (converged only), W: their weighted memberships. Per relation: each failure's share of
    the other seeds in which it recurs (facets matched across seeds one at a time)."""
    n = len(F); out = {}
    if n < 2:
        return {"_note": f"recurrence needs at least 2 converged seeds; {n} available"}
    facets = sorted(W[0])
    perm = {(s, t, fa): M.match(W[s], W[t], [fa]) for s in range(n) for t in range(n) if s != t for fa in facets}
    for name, rd in rds.items():
        if name == ANCHOR:
            continue
        a, b = rd.row_facet, rd.col_facet
        have = [{(x["k"], x["side"], x["main"]) for x in F[t]["failures"][name]} for t in range(n)]
        rows = []
        for s in range(n):
            for x in F[s]["failures"][name]:
                fk, fm = (a, b) if x["side"] == "row" else (b, a)
                hits = sum((int(perm[(s, t, fk)][x["k"]]), x["side"], int(perm[(s, t, fm)][x["main"]])) in have[t]
                           for t in range(n) if t != s)
                rows.append(dict(x, seed=F[s]["seed"], recurs_in=hits / max(n - 1, 1)))
        tot = sum(x["weight"] for x in rows)
        out[name] = {"failures_per_fit": len(rows) / n,
                     "weight_share_recurring": (sum(x["weight"] for x in rows if x["recurs_in"] > 0.5) / tot) if tot else None,
                     "recurring": [x for x in rows if x["recurs_in"] > 0.5]}
    return out


def paired(ref, arm, get):
    a = {x["seed"]: get(x) for x in arm if x["converged"]}; b = {x["seed"]: get(x) for x in ref if x["converged"]}
    d = np.array([a[s] - b[s] for s in sorted(set(a) & set(b)) if a[s] is not None and b[s] is not None])
    if len(d) < 2:
        return {"n": int(len(d))}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"n": int(len(d)), "mean_difference": float(d.mean()), "se": float(se), "lower_by_2se": bool(d.mean() < -2 * se),
            "higher_by_2se": bool(d.mean() > 2 * se)}


def anchor_ok(A):
    conv = [x for x in A if x["converged"]]
    if not conv:
        return None                                              # no converged fit: the guard cannot be read
    if ANCHOR_UNIT == "fit":
        return all(x["anchor"]["within_share"] > 0.5 for x in conv)
    if ANCHOR_UNIT == "community":
        return all(x["anchor"]["n_communities_over_half"] == x["anchor"]["K"] for x in conv)
    raise SystemExit("ANCHOR_UNIT not set: the owner's choice of unit for the child×article > 0.5 requirement is pending")


INCLUDE_UNCONVERGED = False   # supplementary reading (2026-11-18): every fit at its last epoch counts as if converged;
                              # NOT the decided rule (which pairs converged seeds only); output labelled as such


def run(dry_run=False):
    ref = load("stage0_pass1", False)
    arm = ref if dry_run else load("start_test", True)          # dry run: reference as both arms (code check only)
    build = D.Build(D.BUILD)
    rep = {}
    for (a, beta) in sorted(arm):
        rds = {rd.name: rd for rd in D.fit_data(build, a, group=0)}
        name = f"approach_{a}_{LOSS[beta]}"
        arms = {}
        for lab, fs in (("reference", ref[(a, beta)]), ("diagonal", arm[(a, beta)])):
            fs = sorted(fs, key=lambda f: f["job"]["seed"])
            res = [per_fit(f, rds) for f in fs]
            g, _ = group_checks(fs)
            conv = [i for i, (x, _) in enumerate(res) if x["converged"]]
            arms[lab] = {"fits": [x for x, _ in res],
                         "stability": {"gate_passes": int(sum(x is False for x in g["gate_fails"])),
                                       "converged": len(conv),
                                       **({k: g["stability"][k] for k in ("A_adjusted", "B_adjusted", "per_facet_B")} if "stability" in g else {})},
                         "recurrence": recurrence([res[i][0] for i in conv], [res[i][1] for i in conv], rds)}
        R, A = arms["reference"]["fits"], arms["diagonal"]["fits"]
        cmp = {k: paired(R, A, lambda x, k=k: x[k]) for k in ("data_loss", "existence", "value", "mixed_0.1", "weaker_median", "detached")}
        for r in R[0]["existence_by_relation"]:
            cmp[f"existence_{r}"] = paired(R, A, lambda x, r=r: x["existence_by_relation"].get(r))
        for r in R[0]["relation_loss"]:
            cmp[f"loss_{r}"] = paired(R, A, lambda x, r=r: x["relation_loss"][r])
        cmp["anchor_within_share"] = paired(R, A, lambda x: x["anchor"]["within_share"])
        cmp["anchor_communities_over_half"] = paired(R, A, lambda x: x["anchor"]["n_communities_over_half"])
        cmp["anchor_share_communities_over_half"] = paired(R, A, lambda x: x["anchor"]["share_communities_over_half"])
        med = lambda L, g: (float(np.median([g(x) for x in L if x["converged"]])) if any(x["converged"] for x in L) else None)
        rel = {}
        for r in R[0]["rule"]:
            rel[r] = {"rule_share": {"reference": med(R, lambda x: x["rule"][r]["both_share"]), "diagonal": med(A, lambda x: x["rule"][r]["both_share"])},
                      "paired_rule_share": paired(R, A, lambda x, r=r: x["rule"][r]["both_share"]),
                      "row_col": {lab: {k: med(L, lambda x, k=k: x["rule"][r][k]) for k in ("row_share", "col_share", "n_both", "n_row", "n_col")}
                                  for lab, L in (("reference", R), ("diagonal", A))},
                      "partners": {"reference": med(R, lambda x: x["partners"][r]), "diagonal": med(A, lambda x: x["partners"][r])},
                      "paired_partners": paired(R, A, lambda x, r=r: x["partners"][r])}
        anchor = {lab: {k: med(L, lambda x, k=k: x["anchor"][k]) for k in ("within_share", "n_communities_over_half", "share_communities_over_half")}
                  for lab, L in (("reference", R), ("diagonal", A))}
        worse = any(cmp[k].get(f) for k, f in (("data_loss", "higher_by_2se"), ("existence", "lower_by_2se"), ("value", "lower_by_2se")))
        better = any(cmp[k].get(f) for k, f in (("data_loss", "lower_by_2se"), ("existence", "higher_by_2se"), ("value", "higher_by_2se")))
        guard = anchor_ok(A)
        if beta == 2:
            n_pairs = cmp["data_loss"]["n"]
            if n_pairs < 2:
                reading = {"outcome": None, "note": f"not readable by the decided rule: {n_pairs} seed(s) converged in both arms",
                           "anchor_guard_met": guard}
            else:
                per_rel = {}
                for r, v in rel.items():
                    hi = v["paired_rule_share"].get("higher_by_2se")
                    per_rel[r] = 1 if (hi and not worse and guard) else (3 if hi else 2)
                kinds = set(per_rel.values())
                reading = {"outcome": 4 if len(kinds) > 1 else kinds.pop(), "per_relation": per_rel, "fit_worse": worse,
                           "fit_better": better, "anchor_guard_met": guard, "seeds_paired": n_pairs}
        else:
            changed = worse or better or any(v["paired_rule_share"].get("lower_by_2se") or v["paired_rule_share"].get("higher_by_2se")
                                             for v in rel.values())
            reading = {"control_changed": bool(changed), "fit_worse": worse, "fit_better": better, "anchor_guard_met": guard}
        rep[name] = {"n_fits": {k: len(v["fits"]) for k, v in arms.items()}, "paired": cmp, "relations": rel, "anchor": anchor,
                     "reading": reading, "arms": arms}
        print(name, json.dumps(reading), flush=True)
    return rep


def main():
    global ANCHOR_UNIT
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="reference fits as both arms; checks the code without reading start-test fits")
    ap.add_argument("--anchor-unit", choices=("fit", "community"), default=None, help="dry run only")
    ap.add_argument("--include-unconverged", action="store_true", help="supplementary reading, not the decided rule")
    a = ap.parse_args()
    global INCLUDE_UNCONVERGED
    INCLUDE_UNCONVERGED = a.include_unconverged
    if a.dry_run:
        ANCHOR_UNIT = a.anchor_unit or "fit"
    elif a.anchor_unit:
        raise SystemExit("--anchor-unit is for dry runs; the real reading uses ANCHOR_UNIT as decided by the owner")
    rep = run(a.dry_run)
    rep["_reading"] = "SUPPLEMENTARY: every fit at its last epoch, converged or not" if a.include_unconverged else "decided rule: converged seeds only"
    json.dump(rep, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
