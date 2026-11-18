# Are the communities that fail the owner's child×article requirement (RL-266) ghost communities? (owner, 2026-11-18)
# Pass-1 fits at K*, squared error (both approaches), all groups, converged seeds, aligned numbering. Per community:
#   fails = own-numbered cell ≤ half of its child×article tie weight (children side or articles side);
#   M1 ghost (decided 2026-11-07, measures.ghosts): no placed entity, in any facet, has it as its largest weighted
#     membership; reported: number of placed entities whose largest weighted membership is the community, per facet;
#   size: the community's half credit averaged over relations (its share of the fit's reconstructed tie weight),
#     compared with 1/K and with the toy-era low-mass line 0.5/K (CLAUDE.md §4.22).
# Usage: python -m fit.anchor_failures_vs_ghosts --out results/stage0/aligned/anchor_failures_vs_ghosts.json
import argparse, json, os
from collections import defaultdict
import numpy as np
os.environ["M1_ALIGNED"] = "1"
from . import data as D, measures as M
from .aggregate import load_UZ, load_entity
from .facet_detachment_pass1 import all_fits, KSTAR, LOSS


def run():
    fits = [f for f in all_fits() if f["job"]["beta"] == 2 and f["job"]["K"] == KSTAR[2] and f["result"]["converged"]]
    by = defaultdict(list)
    for f in fits:
        by[(f["job"]["approach"], f["job"]["group"])].append(f)
    build = D.Build(D.BUILD); rows = defaultdict(list)
    for (a, g), F in sorted(by.items()):
        rd = [r for r in D.fit_data(build, a, group=g) if r.name == "child_article"][0]
        ents = [load_entity(f["dir"]) for f in F]
        placed = {x: M.placed_entities([e[x]["reproduced_share"] for e in ents], ents[0][x]["reproduced_share_degree_only"]) for x in ents[0]}
        for f in F:
            U, Z = load_UZ(f["dir"]); K = U["article"].shape[1]
            P, T = M.parts(U, Z, rd); d = np.diag(P)
            fail = ~((d / np.maximum(P.sum(1), 1e-300) > 0.5) & (d / np.maximum(P.sum(0), 1e-300) > 0.5))
            wm = M.weighted_memberships(U, Z, {r: tuple(v) for r, v in f["result"]["relations"].items()})
            mains = {x: np.bincount(wm[x]["P"][placed[x]].argmax(1), minlength=K) for x in wm}
            pres = f["measures"]["presence"]
            size = np.mean([np.array(v["half_credit"]) for v in pres.values()], 0)
            for k in range(K):
                rows[(f"approach_{a}_{LOSS[2]}", bool(fail[k]))].append(
                    {"ghost": all(mains[x][k] == 0 for x in mains), "size_x_K": float(size[k] * K),
                     **{f"main_{x}": int(mains[x][k]) for x in mains}})
    rep = {}
    for (cell, fl), L in sorted(rows.items()):
        keys = [k for k in L[0] if k.startswith("main_")]
        rep.setdefault(cell, {})["failing" if fl else "passing"] = {
            "communities": len(L), "ghosts": int(sum(x["ghost"] for x in L)),
            "size_x_K_median": float(np.median([x["size_x_K"] for x in L])),
            "below_half_fair_share": int(sum(x["size_x_K"] < 0.5 for x in L)),
            **{f"{k}_median": float(np.median([x[k] for x in L])) for k in keys}}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    rep = run(); json.dump(rep, open(ap.parse_args().out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
