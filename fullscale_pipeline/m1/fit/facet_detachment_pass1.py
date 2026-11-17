# Facet detachment in stage-0 pass 1 (ticket 107; owner request 2026-11-17 after RL-252). Saved fits only, λ = 0, terms off.
# "Detached" pattern: community k is present on one facet f (facet presence P_k(f) ≥ 0.5) while a facet g that f is tied
# to directly (they share a relation matrix) is nearly absent (P_k(g) < 0.1). Same lines as the decided mono-domain
# rule; 0.2 reported as sensitivity. Example: parent present, child absent — parent hyperedges whose child hyperedges
# (the only facet parents are tied to) sit in other communities.
# Facet presence P_k(f) = K × mean over relations touching f of k's side reading on f's side (measures.facet_presence).
# Part A (every K): rate per fit by approach × loss × K, per pattern (f → g); spread over hold-out groups and seeds.
# Part B (K* only): recurrence across seeds (Track A/B matching, as mono_domain_pass1) and the share of the community's
#   tie weight in the f–g relation that sits in blocks shared with other communities (1 − within ÷ half credit).
# Usage: python -m fit.facet_detachment_pass1 --out results/stage0/facet_detachment_pass1.json
import argparse, glob, json, os
from collections import defaultdict
import numpy as np
from . import measures as M
from .aggregate import load_fit, load_UZ, latest_fits

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
KSTAR = {2: 20, 1: 13}
LOSS = {2: "squared_error", 1: "KL"}


def all_fits():
    fs = [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "stage0_pass1", "*", "measures.json"))]
    out = []
    for f in latest_fits(fs):
        j = f["job"]
        if (j.get("terms") or {}) or (j.get("arm") or {}) or not f["result"]["converged"]:
            continue
        out.append(f)
    return out


def pairs_of(m):
    """Directly tied facet pairs (f, g, relation), both directions; self-relations skipped."""
    out = []
    for r, v in m["presence"].items():
        a, b = v["row_facet"], v["col_facet"]
        if a != b:
            out += [(a, b, r), (b, a, r)]
    return out


def detached(m, low=0.1, high=0.5):
    fp = {f: np.array(v) for f, v in m["facet_presence"].items()}
    res = []
    for f, g, r in pairs_of(m):
        for k in np.where((fp[f] >= high) & (fp[g] < low))[0]:
            res.append((int(k), f, g, r))
    return res


def part_a(fits):
    cells = defaultdict(lambda: {"fits": 0, "counts": defaultdict(list), "by_group": defaultdict(lambda: defaultdict(list))})
    for f in fits:
        j = f["job"]; m = f["measures"]
        key = f"approach_{j['approach']}_{LOSS[j['beta']]}_K{j['K']}"
        c = cells[key]; c["fits"] += 1
        pats = [f"{x[1]}->{x[2]}" for x in detached(m)]
        pats02 = [f"{x[1]}->{x[2]}" for x in detached(m, low=0.2)]
        names = {f"{a}->{b}" for a, b, _ in pairs_of(m)}
        for n in names:
            c["counts"][n].append(pats.count(n))
            c["counts"][n + " (0.2)"].append(pats02.count(n))
            c["by_group"][n][j["group"]].append(pats.count(n))
    out = {}
    for key, c in cells.items():
        row = {"fits": c["fits"]}
        for n, v in c["counts"].items():
            v = np.array(v)
            row[n] = {"per_fit": float(v.mean()), "share_fits_with_one": float((v > 0).mean())}
        row["per_group_share_fits_with_one"] = {n: {str(g): float((np.array(v) > 0).mean()) for g, v in sorted(gd.items())}
                                                for n, gd in c["by_group"].items()}
        out[key] = row
    return out


def part_b(fits):
    rep = {}
    for a in (1, 2):
        for beta in (2, 1):
            name = f"approach_{a}_{LOSS[beta]}"
            groups = defaultdict(list)
            for f in fits:
                j = f["job"]
                if (j["approach"], j["beta"], j["K"]) == (a, beta, KSTAR[beta]) and j["seed"] in range(1, 11):
                    groups[j["group"]].append(f)
            inst, other_off = [], defaultdict(list)
            for g, fs in sorted(groups.items()):
                fs = sorted(fs, key=lambda f: f["job"]["seed"])
                wm = []
                for f in fs:
                    U, Z = load_UZ(f["dir"]); wm.append(M.weighted_memberships(U, Z, f["result"]["relations"]))
                facets = sorted(wm[0])
                perm = {(i, jj): M.match(wm[i], wm[jj], facets) for i in range(len(fs)) for jj in range(len(fs)) if i != jj}
                for i, f in enumerate(fs):
                    m = f["measures"]; fp = {x: np.array(v) for x, v in m["facet_presence"].items()}
                    det = detached(m)
                    flagged = {(k, r) for k, _, _, r in det}
                    for r, v in m["presence"].items():
                        off = 1 - np.array(v["within_over_half_credit"])
                        for k in range(len(off)):
                            if (k, r) not in flagged:
                                other_off[r].append(float(off[k]))
                    for k, ff, gg, r in det:
                        same, low_vals = 1, []
                        for jj, h in enumerate(fs):
                            if jj == i:
                                continue
                            kk = perm[(i, jj)][k]
                            fq = {x: np.array(v) for x, v in h["measures"]["facet_presence"].items()}
                            same += bool(fq[ff][kk] >= 0.5 and fq[gg][kk] < 0.1)
                            low_vals.append(float(fq[gg][kk]))
                        dom = m["domain_presence"]
                        inst.append({"group": g, "seed": f["job"]["seed"], "k": k, "pattern": f"{ff}->{gg}", "relation": r,
                                     "n_seeds": len(fs), "seeds_same_pattern": same, "recurs": same > len(fs) / 2,
                                     "matched_absent_facet_presence_median": float(np.median(low_vals)),
                                     "P_present_facet": float(fp[ff][k]), "P_absent_facet": float(fp[gg][k]),
                                     "facet_presence": {x: float(v[k]) for x, v in fp.items()},
                                     "P_semantic": float(dom["semantic"]["mean"][k]), "P_social": float(dom["social"]["mean"][k]),
                                     "off_diagonal_share_in_relation": float(1 - m["presence"][r]["within_over_half_credit"][k])})
                print(name, "group", g, "instances so far", len(inst), flush=True)
            rep[name] = {"instances": inst,
                         "other_communities_off_diagonal_share_median": {r: float(np.median(v)) for r, v in other_off.items()},
                         "other_communities_off_diagonal_share_q10_q90": {r: [float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))] for r, v in other_off.items()}}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fits = all_fits()
    print("fits", len(fits), flush=True)
    rep = {"part_a": part_a(fits)}
    print("part A done", flush=True)
    rep["part_b"] = part_b(fits)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
