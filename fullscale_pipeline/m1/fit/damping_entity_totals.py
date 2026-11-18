# Entity-total diagnostic of the social log-damping arm (amendment c, GPT review RL-272/273; cheap reporting, accepted
# 2026-11-18). No fits. Hold-out group 1, visible ties. For each damped relation: entities on the damped side grouped
# by their number of visible ties d (bands 1, 2–3, 4–7, 8–15, 16–31, 32+); per band: entities, mean entity total (sum
# of its tie values) and the band's share of the relation's visible mass — undamped (Frobenius-scaled, as now) vs
# damped (the arm: × log(1+d)/d, mass restored). author×article also by the article's team size (its number of
# authors, same bands) and by productivity band × team-size band (share of the relation's mass), because the article
# side already carries the team-size weighting, so the two corrections multiply.
# Usage: python -m fit.damping_entity_totals --out fit/results/stage0/damping_entity_totals.json
import argparse, json
import numpy as np
from . import data as D

EDGES = [1, 2, 4, 8, 16, 32]
LABELS = ["1", "2-3", "4-7", "8-15", "16-31", "32+"]


def band(d):
    return np.searchsorted(EDGES, d, side="right") - 1


def by_band(ent, v, n, d):
    tot = np.bincount(ent, weights=v, minlength=n); b = band(np.maximum(d, 1)); has = d > 0
    return {LABELS[i]: {"entities": int((has & (b == i)).sum()),
                        "mean_total": float(tot[has & (b == i)].mean()) if (has & (b == i)).any() else None,
                        "mass_share": float(tot[has & (b == i)].sum() / v.sum())} for i in range(len(LABELS))}


def run():
    build = D.Build(D.BUILD); rep = {}
    for a in (1, 2):
        base = {rd.name: rd for rd in D.fit_data(build, a, group=0)}
        damp = {rd.name: rd for rd in D.fit_data(build, a, group=0, log_damping="data")}
        for name, rd in damp.items():
            if rd.damping is None or name in rep:
                continue
            side = rd.damping["side"]; n = rd.shape[0] if side == "row" else rd.shape[1]
            ent = rd.tie_row if side == "row" else rd.tie_col
            d = np.bincount(ent, minlength=n)
            r = {"q": rd.damping["q"], "damped_side": side,
                 "by_degree": {"undamped": by_band(ent, base[name].tie_val, n, d), "damped": by_band(ent, rd.tie_val, n, d)}}
            if name == "author_article":
                m = rd.shape[1]; team = np.bincount(rd.tie_col, minlength=m)
                r["by_team_size"] = {"undamped": by_band(rd.tie_col, base[name].tie_val, m, team),
                                     "damped": by_band(rd.tie_col, rd.tie_val, m, team)}
                pb, tb = band(np.maximum(d, 1))[rd.tie_row], band(np.maximum(team, 1))[rd.tie_col]
                for lab, v in (("undamped", base[name].tie_val), ("damped", rd.tie_val)):
                    M = np.zeros((len(LABELS), len(LABELS))); np.add.at(M, (pb, tb), v)
                    r.setdefault("productivity_x_team_share", {})[lab] = (M / v.sum()).round(4).tolist()
            rep[name] = r
            print(name, f"q {r['q']:.3f}", flush=True)
            for i, L in enumerate(LABELS):
                u, x = r["by_degree"]["undamped"][L], r["by_degree"]["damped"][L]
                if u["entities"]:
                    print(f"  d {L:6s} n {u['entities']:6d}  mean total {u['mean_total']:.4f} → {x['mean_total']:.4f}"
                          f"  mass share {u['mass_share']:.3f} → {x['mass_share']:.3f}", flush=True)
            if name == "author_article":
                print("  by team size:")
                for L in LABELS:
                    u, x = r["by_team_size"]["undamped"][L], r["by_team_size"]["damped"][L]
                    if u["entities"]:
                        print(f"  team {L:6s} n {u['entities']:6d}  mean total {u['mean_total']:.4f} → {x['mean_total']:.4f}"
                              f"  mass share {u['mass_share']:.3f} → {x['mass_share']:.3f}", flush=True)
                print("  productivity band (rows) × team-size band (columns), share of mass, undamped / damped:")
                for i, L in enumerate(LABELS):
                    u = r["productivity_x_team_share"]["undamped"][i]; x = r["productivity_x_team_share"]["damped"][i]
                    if sum(u):
                        print(f"  {L:6s} " + "  ".join(f"{p:.3f}/{q:.3f}" for p, q in zip(u, x)), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
