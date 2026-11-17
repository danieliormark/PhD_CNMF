# Mono-domain communities in stage-0 pass 1 — the remaining steps of the decided rule (ticket 107; rule 2026-11-07,
# per-fit counts RL-242). Saved fits only (λ = 0, K*: squared error 20, KL 13; both approaches; 5 hold-out groups × 10
# seeds). Per community flagged mono-domain in a fit (P_k(D1) < 0.1 and P_k(D2) ≥ 0.5):
#   recurrence   within its hold-out group, every other seed's communities are matched to this seed's (Hungarian on the
#                weighted membership columns stacked over all facets, cosine; measures.match, the Track A/B matching).
#                Seeds whose matched community is mono-domain in the same direction are counted; the decided rule asks
#                for "the majority of seeds" (> half of the converged seeds, this seed included). Reported beside it: the
#                cosine of the match (how similar the matched community is) and the matched community's low-domain
#                presence (does it stay low even when it misses the 0.1 line).
#   tie weight   presence in the present domain and in the absent one; share of each relation's reconstructed tie weight
#                (half credit); compared with the other communities of the same fits.
#   members      per domain, entities whose largest weighted membership is k (no cut-off) and entities with weighted
#                membership in k ≥ 0.5 (sensitivity reading only; q and m are decided under item (l)) — the membership
#                half of the combined reading (2026-11-07): members in the absent domain → "narrow case".
#   chance check the decided rule exempts a low value that comes from relation matrices fitted only at chance: per fit,
#                within-class AP of the held-out relations of the absent domain against its chance level.
#   which relation  presence of the community in each relation of the absent domain.
# Also: counts per fit with the low line at 0.05 / 0.1 / 0.2 (sensitivity, decided to be reported) and, for every cell,
# the distribution of the lower of a community's two presences.
# Usage: python -m fit.mono_domain_pass1 --out results/stage0/mono_domain_pass1.json
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from . import measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
SOCIAL_FACETS = ("article", "author", "organisation", "journal")
SEMANTIC_FACETS = ("child", "parent")


def fits_for(approach, beta):
    fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass1", "*", "measures.json"))]
    out = defaultdict(list)
    for f in latest_fits(fs):
        j = f["job"]
        if (j["approach"], j["beta"], j["K"]) != (approach, beta, KSTAR[beta]) or (j.get("terms") or {}):
            continue
        if j["seed"] in range(1, 11) and f["result"]["converged"]:
            out[j["group"]].append(f)
    return out


def flags(S, O, low=0.1, high=0.5):
    """Per community: 'social only', 'semantic only' or None."""
    return ["social only" if (s < low and o >= high) else "semantic only" if (o < low and s >= high) else None
            for s, o in zip(S, O)]


def per_fit(f):
    m = f["measures"]
    S = np.array(m["domain_presence"]["semantic"]["mean"]); O = np.array(m["domain_presence"]["social"]["mean"])
    U, Z = load_UZ(f["dir"])
    wm = M.weighted_memberships(U, Z, f["result"]["relations"])
    largest = {fa: np.bincount(d["P"].argmax(1), minlength=d["P"].shape[1]) for fa, d in wm.items()}
    ge05 = {fa: (d["P"] >= 0.5).sum(0) for fa, d in wm.items()}
    dom = {r: v["domain"] for r, v in m["presence"].items()}
    two = m["heldout_two_part"]
    chance = {r: {"ap": v["ap_within_class"], "chance": v["ap_within_class_chance"]} for r, v in two.items() if v}
    return {"seed": f["job"]["seed"], "S": S, "O": O, "flags": flags(S, O), "wm": wm, "largest": largest, "ge05": ge05,
            "presence": {r: np.array(v["presence"]) for r, v in m["presence"].items()},
            "half_credit": {r: np.array(v["half_credit"]) for r, v in m["presence"].items()},
            "domain_of": dom, "chance": chance}


def members(fit, k, facets):
    return (int(sum(fit["largest"][fa][k] for fa in facets if fa in fit["largest"])),
            int(sum(fit["ge05"][fa][k] for fa in facets if fa in fit["ge05"])))


def run():
    rep = {}
    for a in (1, 2):
        for beta in (2, 1):
            name = f"approach_{a}_{'KL' if beta == 1 else 'squared_error'}"
            groups = fits_for(a, beta)
            inst, others, sens, lower, chance_rows = [], [], defaultdict(list), [], []
            for g, fs in sorted(groups.items()):
                F = [per_fit(f) for f in sorted(fs, key=lambda f: f["job"]["seed"])]
                facets = sorted(F[0]["wm"])
                for fit in F:
                    for lo in (0.05, 0.1, 0.2):
                        sens[lo].append(sum(x is not None for x in flags(fit["S"], fit["O"], low=lo)))
                    lower.extend(np.minimum(fit["S"], fit["O"]).tolist())
                    for r, v in fit["chance"].items():
                        chance_rows.append({"group": g, "seed": fit["seed"], "relation": r, "domain": fit["domain_of"][r], **v})
                    K = len(fit["S"])
                    for k in range(K):
                        if fit["flags"][k] is None:
                            soc_m = members(fit, k, SOCIAL_FACETS); sem_m = members(fit, k, SEMANTIC_FACETS)
                            others.append({"S": float(fit["S"][k]), "O": float(fit["O"][k]), "social_largest": soc_m[0],
                                           "semantic_largest": sem_m[0]})
                # recurrence across seeds
                for i, fi in enumerate(F):
                    for k, fl in enumerate(fi["flags"]):
                        if fl is None:
                            continue
                        absent = "semantic" if fl == "social only" else "social"
                        same, cos, low_vals = 1, [], []
                        for j, fj in enumerate(F):
                            if j == i:
                                continue
                            perm = M.match(fi["wm"], fj["wm"], facets)
                            kk = perm[k]
                            nz = lambda Y: Y / max(np.linalg.norm(Y), 1e-300)
                            A = nz(np.concatenate([fi["wm"][fa]["Y"][:, k] for fa in facets]))
                            B = nz(np.concatenate([fj["wm"][fa]["Y"][:, kk] for fa in facets]))
                            cos.append(float(A @ B))
                            same += fj["flags"][kk] == fl
                            low_vals.append(float((fj["S"] if absent == "semantic" else fj["O"])[kk]))
                        absent_facets = SEMANTIC_FACETS if absent == "semantic" else SOCIAL_FACETS
                        present_facets = SOCIAL_FACETS if absent == "semantic" else SEMANTIC_FACETS
                        inst.append({
                            "group": g, "seed": fi["seed"], "k": k, "type": fl, "n_seeds": len(F),
                            "seeds_mono_same_direction": same, "recurs": same > len(F) / 2,
                            "match_cosine_median": float(np.median(cos)) if cos else None,
                            "matched_low_presence_median": float(np.median(low_vals)) if low_vals else None,
                            "matched_low_presence_below_0.2": int(sum(v < 0.2 for v in low_vals)),
                            "P_present": float((fi["O"] if absent == "semantic" else fi["S"])[k]),
                            "P_absent": float((fi["S"] if absent == "semantic" else fi["O"])[k]),
                            "absent_relation_presence": {r: float(p[k]) for r, p in fi["presence"].items() if fi["domain_of"][r] == absent},
                            "present_relation_half_credit": {r: float(h[k]) for r, h in fi["half_credit"].items() if fi["domain_of"][r] != absent},
                            "members_present_largest": members(fi, k, present_facets)[0],
                            "members_absent_largest": members(fi, k, absent_facets)[0],
                            "members_absent_ge_0.5": members(fi, k, absent_facets)[1],
                        })
            # typical match cosine of ordinary communities, for reading the recurrence cosine (group 1, seed 1 vs others)
            F0 = [per_fit(f) for f in sorted(groups[0], key=lambda f: f["job"]["seed"])]
            facets = sorted(F0[0]["wm"])
            base_cos = []
            for fj in F0[1:]:
                perm = M.match(F0[0]["wm"], fj["wm"], facets)
                for k in range(len(perm)):
                    nz = lambda Y: Y / max(np.linalg.norm(Y), 1e-300)
                    A = nz(np.concatenate([F0[0]["wm"][fa]["Y"][:, k] for fa in facets]))
                    B = nz(np.concatenate([fj["wm"][fa]["Y"][:, perm[k]] for fa in facets]))
                    base_cos.append(float(A @ B))
            n_fits = sum(len(v) for v in groups.values())
            rep[name] = {"fits": n_fits, "instances": inst,
                         "per_fit_counts_by_low_line": {str(lo): float(np.mean(v)) for lo, v in sens.items()},
                         "share_fits_with_one_by_low_line": {str(lo): float(np.mean(np.array(v) > 0)) for lo, v in sens.items()},
                         "lower_presence_quantiles": dict(zip(("1%", "5%", "25%", "50%"), np.quantile(lower, [0.01, 0.05, 0.25, 0.5]).tolist())),
                         "chance_check": chance_rows, "others": others,
                         "all_community_match_cosine_group1_seed1": dict(zip(("10%", "25%", "50%", "75%"), np.quantile(base_cos, [0.1, 0.25, 0.5, 0.75]).tolist()))}
            print(name, "fits", n_fits, "mono instances", len(inst), "recurring", sum(x["recurs"] for x in inst), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    json.dump(run(), open(a.out, "w"), indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (int(x) if isinstance(x, np.integer) else str(x)))


if __name__ == "__main__":
    main()
