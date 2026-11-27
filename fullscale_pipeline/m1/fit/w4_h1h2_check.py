# W4 follow-up (owner 2026-11-27: "yes, build and run parts A–C"; RL-417). Saved fits only (the W4 cells, 4 conditions ×
# seeds), no new fits. Question: the word-similarity regulariser (child×child W, weight λ_child = r × pass-1 basis) adds
# co-membership to article pairs sharing a social tie even after the swap (RL-415/416). Is that H1 (memberships become
# more concentrated, arrangement unchanged) or H2 (content separates articles less, so social ties decide more)?
# Notation: p_i = article i's weighted membership (ticket 100); C_ij = Σ_k p_ik p_jk; excess E = mean C over the pairs
# sharing an entity − mean C over all article pairs; "off" = λ_child 0, "on" = the cell's r; same seed → same swap.
# Part A (H1 share): sharpen each article's off membership, p^γ_i renormalised, γ_i chosen so its entropy equals its on
#   entropy (arrangement of the off fit, concentration of the on fit) → E_conc1; reverse: flatten the on memberships to
#   the off entropies → E_conc2. H1 share = (E_conc1 − E_off)/(E_on − E_off) (order 1), (E_on − E_conc2)/(E_on − E_off)
#   (order 2); main = their mean. Robustness: one common γ per fit matching the all-pairs mean C of the on fit.
#   Added after the first test (λ_child raised per-article entropy while the all-pairs mean rose): a community-size
#   counterfactual (columns rescaled so each community's mean membership equals the other fit's, matched by size rank;
#   "size") and both together ("both"); same two orders. "conc" = the entropy counterfactual above.
# Part B (direction of the rest): side profile of article i for relation r touching articles = Σ over i's visible ties
#   of tie value × (partner's weighted membership passed through Q_r), normalised; Q_r = relation r's rebuilt tie weight
#   split by community pair (GLOSSARY E1 parts), conditioned on the partner's community (column-normalised). Leaves out
#   the article's own membership; gauge-free (parts and weighted memberships are). Diagonal Q → mean partner membership.
#   B1 distinctiveness: mean over articles of ½Σ|d_i − mean d| (total variation from the mean side profile).
#   B2 overlap: mean over articles of Σ_k min(p_ik, d_ik) for off, on and the Part-A counterfactual (sharpened off p
#   with the off side profiles). H2 predicts: content side less distinctive with λ_child; overlap with the social sides
#   up beyond the counterfactual, with the content side down.
# Part C (where the addition sits): deciding-entity pairs split by also sharing a journal (J) and an organisation (O);
#   addition A = Σ_g s_g A_g exactly, s_g = share of the pairs in subgroup g, A_g = mean_g ΔC − ΔC̄_all; with the
#   Part-A split inside each subgroup.
# Uncertainty: mean ± SE over the paired seeds; shares as ratio of seed means with a 95 % bootstrap interval over seeds.
# Usage (from fullscale_pipeline/m1): PYTHONPATH=. python -m fit.w4_h1h2_check fit/results/stage0/w4_h1h2_check.json
import json, math, sys
from multiprocessing import Pool
import numpy as np, scipy.sparse as sp
sys.path.insert(0, ".")
from fit import w4_swap_analysis as W, measures as M
from fit.aggregate import load_UZ
from fit.align import fit_rds
from fit.swap_link_measure import incidence, pair_mask
from fit.plan import W4_CELLS

EPS = 1e-300
_cache = {}


def entropy(P):
    return -(P * np.log(np.clip(P, EPS, None))).sum(1)


def power(P, g):
    g = np.asarray(g, float).reshape(-1, 1) if np.ndim(g) else g
    Q = np.where(P > 0, np.exp(g * np.log(np.clip(P, EPS, None))), 0.0)
    s = Q.sum(1, keepdims=True)
    return np.where(s > 0, Q / np.maximum(s, EPS), 0.0)


def match_entropy(P, target, lo=-5.0, hi=5.0, it=60):
    """Per row: γ = e^x with entropy(power(P, γ)) = target (bisection on x; entropy falls with γ). Returns Q, clipped share."""
    a = np.full(len(P), lo); b = np.full(len(P), hi)
    for _ in range(it):
        m = (a + b) / 2
        too_flat = entropy(power(P, np.exp(m))) > target      # entropy still above target → need larger γ
        a = np.where(too_flat, m, a); b = np.where(too_flat, b, m)
    x = (a + b) / 2
    return power(P, np.exp(x)), float(((x < lo + 1e-3) | (x > hi - 1e-3)).mean())


def all_mean(P):
    n = len(P); s = P.sum(0)
    return float((s @ s - (P * P).sum()) / (n * (n - 1)))


def match_all_mean(P, target, lo=-5.0, hi=5.0, it=60):
    fa = all_mean(power(P, math.exp(lo))) - target; fb = all_mean(power(P, math.exp(hi))) - target
    if fa * fb > 0:
        return None
    a, b = lo, hi
    for _ in range(it):
        m = (a + b) / 2; fm = all_mean(power(P, math.exp(m))) - target
        if (fm > 0) == (fb > 0):
            b, fb = m, fm
        else:
            a, fa = m, fm
    return power(P, math.exp((a + b) / 2))


def match_sizes(P, target, it=300):
    """Added 2026-11-27 after the first test showed λ_child flattens per-article memberships while the all-pairs mean
    rises: community-size counterfactual. Columns are rescaled (then rows renormalised, iterated) so that the mean
    membership of each community equals the target fit's, matched by rank of size (labels differ between fits)."""
    t = np.empty(P.shape[1]); t[np.argsort(-P.mean(0))] = np.sort(target)[::-1]
    Q = P.copy()
    for _ in range(it):
        Q = Q * (t / np.maximum(Q.mean(0), EPS)); s = Q.sum(1, keepdims=True); Q = np.where(s > 0, Q / np.maximum(s, EPS), 0.0)
    return Q


def match_both(P, sizes, ent, rounds=8):
    Q = P
    for _ in range(rounds):
        Q = match_sizes(Q, sizes, it=100); Q, _ = match_entropy(Q, ent)
    return Q


def pair_mean(P, r, c):
    return float((P[r] * P[c]).sum(1).mean()) if len(r) else float("nan")


def side_profiles(U, Z, rds, Pm):
    """Per relation touching articles: (n_art × K) side profiles (rows sum to 1 where the article has ties) and mask."""
    n = Pm["article"].shape[0]; out = {}
    for rd in rds:
        if "article" not in (rd.row_facet, rd.col_facet):
            continue
        Pt, _ = M.parts(U, Z, rd)
        acc = np.zeros((n, Pt.shape[0]))
        if rd.row_facet == "article":                     # partner on the column side: Q(k_article | l_partner)
            Qc = Pt / np.maximum(Pt.sum(0, keepdims=True), EPS)
            np.add.at(acc, rd.tie_row, rd.tie_val[:, None] * (Pm[rd.col_facet][rd.tie_col] @ Qc.T))
        if rd.col_facet == "article":                     # partner on the row side: Q(l_article | k_partner)
            Qr = Pt / np.maximum(Pt.sum(1, keepdims=True), EPS)
            np.add.at(acc, rd.tie_col, rd.tie_val[:, None] * (Pm[rd.row_facet][rd.tie_row] @ Qr))
        s = acc.sum(1); has = s > 0
        acc[has] /= s[has, None]
        out[rd.name] = (acc, has)
    return out


def entities(ap_, rds, n):
    ent = {}
    if ap_ == 1:
        Ea = incidence(rds["author_article"], n); ent["author"] = Ea
        A2 = rds["author_affiliation"]
        ent["organisation"] = sp.csr_matrix((Ea @ sp.csr_matrix((np.ones(len(A2.tie_row)), (A2.tie_row, A2.tie_col)), shape=A2.shape)) > 0, dtype=float)
    else:
        ent["co-author tie"] = incidence(rds["article_article"], n)
        ent["organisation"] = incidence(rds["article_affiliation"], n)
    ent["journal"] = incidence(rds["article_journal"], n)
    return ent


def load(f):
    U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
    Pm = {k: v["P"] for k, v in M.weighted_memberships(U, Z, rel).items()}
    rds = {rd.name: rd for rd in fit_rds(f["job"], _cache)}
    return U, Z, Pm, rds


def one(task):
    (ap_, K, r), seed, fs = task
    res = {}
    for p in (0.0, 1.0):
        Uo, Zo, Po, rds = load(fs[("off", p)]); Un, Zn, Pn, _ = load(fs[("on", p)])
        n = Po["article"].shape[0]
        Xoff, Xon = Po["article"], Pn["article"]
        Xc1, clip1 = match_entropy(Xoff, entropy(Xon))
        Xc2, clip2 = match_entropy(Xon, entropy(Xoff))
        Xg = match_all_mean(Xoff, all_mean(Xon))
        V = {"off": Xoff, "on": Xon, "conc1": Xc1, "conc2": Xc2,
             "size1": match_sizes(Xoff, Xon.mean(0)), "size2": match_sizes(Xon, Xoff.mean(0)),
             "both1": match_both(Xoff, Xon.mean(0), entropy(Xon)), "both2": match_both(Xon, Xoff.mean(0), entropy(Xoff))}
        if Xg is not None:
            V["concg"] = Xg
        allm = {k: all_mean(x) for k, x in V.items()}
        ent = entities(ap_, rds, n)
        pairs = {e: pair_mask(E) for e, E in ent.items()}
        A = {e: {k: pair_mean(x, *pairs[e]) - allm[k] for k, x in V.items()} for e in ent}
        # Part C: deciding entity pairs split by also sharing a journal / an organisation
        dec = "author" if ap_ == 1 else "co-author tie"
        rr, cc = pairs[dec]
        Sj = (ent["journal"] @ ent["journal"].T).tocsr(); So = (ent["organisation"] @ ent["organisation"].T).tocsr()
        fj = np.asarray(Sj[rr, cc]).ravel() > 0; fo = np.asarray(So[rr, cc]).ravel() > 0
        C = {}
        for gname, mask in (("journal+org", fj & fo), ("journal only", fj & ~fo), ("org only", ~fj & fo), ("neither", ~fj & ~fo),
                            ("journal", fj), ("no journal", ~fj)):
            C[gname] = {"n": int(mask.sum()),
                        **{k: (pair_mean(x, rr[mask], cc[mask]) - allm[k]) if mask.any() else float("nan") for k, x in V.items()}}
        # Part B: side profiles, distinctiveness, overlap
        so_, sn_ = side_profiles(Uo, Zo, list(rds.values()), Po), side_profiles(Un, Zn, list(rds.values()), Pn)
        B = {}
        for name in so_:
            do, ho = so_[name]; dn, hn = sn_[name]; h = ho & hn
            tv = lambda d: float(0.5 * np.abs(d[h] - d[h].mean(0)).sum(1).mean())
            ov = lambda x, d: float(np.minimum(x[h], d[h]).sum(1).mean())
            B[name] = {"articles": int(h.sum()), "tv_off": tv(do), "tv_on": tv(dn),
                       "ov_off": ov(Xoff, do), "ov_on": ov(Xon, dn), "ov_conc1": ov(Xc1, do)}
        B["_membership"] = {"tv_off": float(0.5 * np.abs(Xoff - Xoff.mean(0)).sum(1).mean()),
                            "tv_on": float(0.5 * np.abs(Xon - Xon.mean(0)).sum(1).mean()),
                            "entropy_off": float(entropy(Xoff).mean()), "entropy_on": float(entropy(Xon).mean()),
                            "size_sumsq_off": float((Xoff.mean(0) ** 2).sum()), "size_sumsq_on": float((Xon.mean(0) ** 2).sum()),
                            "size_match_residual": float(np.abs(np.sort(V["size1"].mean(0)) - np.sort(Xon.mean(0))).sum()),
                            "both_entropy_residual": float(np.abs(entropy(V["both1"]) - entropy(Xon)).mean())}
        res[f"p{int(p)}"] = {"A": A, "C": C, "B": B, "all_mean": allm, "clip": [clip1, clip2], "global_gamma_found": Xg is not None}
    return (ap_, K, r), seed, res


def ms(v):
    v = np.asarray([x for x in v if x == x], float)
    return [float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")]


def share(num, den, reps=2000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed); n = len(num)
    bs = [num[i].mean() / den[i].mean() for i in (rng.integers(0, n, n) for _ in range(reps))]
    return [float(num.mean() / den.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarise(rows):
    out = {}
    for cell, per in rows.items():
        seeds = sorted(per); c = {"seeds": len(seeds)}
        for pk in ("p0", "p1"):
            g = lambda fn: [fn(per[s][pk]) for s in seeds]
            d = {}
            for e in per[seeds[0]][pk]["A"]:
                a = lambda k: np.array(g(lambda x: x["A"][e].get(k, float("nan"))))
                tot = a("on") - a("off")
                d[e] = {"addition": ms(tot)}
                for v in ("conc", "size", "both"):
                    o1, o2 = a(v + "1") - a("off"), a("on") - a(v + "2")
                    d[e][v] = {"part_order1": ms(o1), "part_order2": ms(o2), "share_order1": share(o1, tot),
                               "share_order2": share(o2, tot), "share_main": share((o1 + o2) / 2, tot)}
                if all(per[s][pk]["global_gamma_found"] for s in seeds):
                    d[e]["H1_share_global_gamma"] = share(a("concg") - a("off"), tot)
            CC = {}
            for gname in per[seeds[0]][pk]["C"]:
                nn = np.array(g(lambda x: x["C"][gname]["n"])); N = np.array(g(lambda x: sum(x["C"][h]["n"] for h in ("journal+org", "journal only", "org only", "neither"))))
                ag = lambda k: np.array(g(lambda x: x["C"][gname][k]))
                sg = nn / N; Ag = ag("on") - ag("off")
                dec = "author" if cell[0] == 1 else "co-author tie"
                tot = np.array(g(lambda x: x["A"][dec]["on"] - x["A"][dec]["off"]))
                ok = nn > 0
                CC[gname] = {"pairs": ms(nn), "subgroup_share": ms(sg), "subgroup_addition": ms(Ag[ok]) if ok.any() else None,
                             "contribution_pct": share(np.where(ok, sg * np.nan_to_num(Ag), 0), tot) if ok.all() else None,
                             **{f"{v}_share_within": (share(((ag(v + "1") - ag("off")) + (ag("on") - ag(v + "2")))[ok] / 2, Ag[ok]) if ok.sum() > 2 else None)
                                for v in ("conc", "size", "both")}}
            BB = {}
            for name in per[seeds[0]][pk]["B"]:
                b = lambda k: np.array(g(lambda x: x["B"][name][k]))
                if name == "_membership":
                    BB[name] = {k: ms(b(k)) for k in ("tv_off", "tv_on", "entropy_off", "entropy_on", "size_sumsq_off", "size_sumsq_on",
                                                      "size_match_residual", "both_entropy_residual")}
                    BB[name]["size_sumsq_on_minus_off"] = ms(b("size_sumsq_on") - b("size_sumsq_off"))
                    BB[name]["tv_on_minus_off"] = ms(b("tv_on") - b("tv_off")); BB[name]["entropy_on_minus_off"] = ms(b("entropy_on") - b("entropy_off"))
                    continue
                BB[name] = {"articles": ms(b("articles")), "tv_off": ms(b("tv_off")), "tv_on_minus_off": ms(b("tv_on") - b("tv_off")),
                            "ov_off": ms(b("ov_off")), "ov_conc_minus_off": ms(b("ov_conc1") - b("ov_off")),
                            "ov_on_minus_conc": ms(b("ov_on") - b("ov_conc1")), "ov_on_minus_off": ms(b("ov_on") - b("ov_off"))}
            c[pk] = {"A": d, "C": CC, "B": BB, "clip_share_max": float(max(max(per[s][pk]["clip"]) for s in seeds))}
        out[f"A{cell[0]} K{cell[1]} r{cell[2]:g}"] = c
    return out


def main():
    fits = W.collect(W.ROOT); tasks = []
    for ap_, K, r in W4_CELLS:
        keys = {(t, p): (ap_, K, r if t == "on" else 0.0, p) for t in ("on", "off") for p in (0.0, 1.0)}
        seeds = sorted(set.intersection(*[set(fits[k]) for k in keys.values()]))
        for s in seeds:
            tasks.append(((ap_, K, r), s, {tp: fits[k][s] for tp, k in keys.items()}))
    with Pool(12) as P:
        res = P.map(one, tasks)
    rows = {}
    for cell, s, x in res:
        rows.setdefault(cell, {})[s] = x
    out = summarise(rows)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    f = lambda v: f"{v[0]:+.4f}±{v[1]:.4f}"; sh = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for cell, c in out.items():
        for pk in ("p1", "p0"):
            print(f"== {cell} {pk} (seeds {c['seeds']}, max clipped share {c[pk]['clip_share_max']:.3f})")
            for e, d in c[pk]["A"].items():
                print(f"  A {e:14s} addition {f(d['addition'])}  " + "  ".join(f"{v}: main {sh(d[v]['share_main'])} o1 {sh(d[v]['share_order1'])} o2 {sh(d[v]['share_order2'])}" for v in ("conc", "size", "both")))
            for gname, d in c[pk]["C"].items():
                print(f"  C {gname:13s} pairs {d['pairs'][0]:.0f} share {d['subgroup_share'][0]:.3f} addition " + (f(d['subgroup_addition']) if d['subgroup_addition'] else "—")
                      + (f" contribution {sh(d['contribution_pct'])}" if d['contribution_pct'] else "") + "".join(f" {v} within {sh(d[v + '_share_within'])}" for v in ("conc", "size", "both") if d[v + '_share_within']))
            for name, d in c[pk]["B"].items():
                if name == "_membership":
                    print(f"  B membership tv off {d['tv_off'][0]:.4f} on−off {f(d['tv_on_minus_off'])}  entropy off {d['entropy_off'][0]:.3f} on−off {f(d['entropy_on_minus_off'])}"
                          f"  Σm² off {d['size_sumsq_off'][0]:.4f} on−off {f(d['size_sumsq_on_minus_off'])}  residuals size {d['size_match_residual'][0]:.4f} entropy {d['both_entropy_residual'][0]:.4f}")
                else:
                    print(f"  B {name:20s} n {d['articles'][0]:.0f} tv off {d['tv_off'][0]:.4f} on−off {f(d['tv_on_minus_off'])} | overlap off {d['ov_off'][0]:.4f}"
                          f" conc−off {f(d['ov_conc_minus_off'])} on−conc {f(d['ov_on_minus_conc'])} on−off {f(d['ov_on_minus_off'])}")


if __name__ == "__main__":
    main()
