"""Ticket 105, test T2: matching validity on real T2 data.

For every config and every T2 K in 2-5, with the config's selected T1 knee model as the
prior, fit T2 at the 10 stability seeds and record the frozen T1->T2 match:
pairs accepted and rejected (z against the matched-z chance threshold), unmatched T1/T2
communities, persisting entities. Seed consistency: after aligning the seeds' own
communities to seed 1000 (Hungarian on the stacked T2 weighted memberships, the §S5 way),
does a given T1 community map to the same T2 community?

The weight only acts after the freeze, so the match itself does not depend on it; it is
run at weight 0.1 (a placeholder until T5) and the stopping rule needs some weight > 0.

    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE python t2_matching.py [C1 C2 ...]

T2_EXCLUDE=core_atom,fringe_atom drops those facets from the persisting set (the prior and the
match use only the remaining facets; the model itself is unchanged). Added 2026-10-06 for the
owner's planned full-scale configuration, which has no atom facets; output files get a suffix.
"""
import importlib.util, os, sys, json, collections, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c13v10", f"{E}/chunk13v10.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
from scipy.optimize import linear_sum_assignment

configs = sys.argv[1:] or m.CONFIG_IDS
EXCLUDE = [f for f in os.environ.get("T2_EXCLUDE", "").split(",") if f]
SUFFIX = ("_no_" + "_".join(EXCLUDE)) if EXCLUDE else ""
sel = json.load(open(f"{E}/results/v9.4.t1_v2/model_selection.json"))
raw1 = m.load_and_validate_data(OUT + "Star_extended_matrices_t1_v2.pkl")
raw2 = m.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
LZ = 0.12  # a mid-range lambda_z_offdiag (log-uniform 1e-4..1); the match is read at the freeze
out = {}
for c in configs:
    soc, sem, anc = m.get_active_facets(c)
    s = sel[c]
    mdir = f"{E}/results/v9.4.t1_v2/{c}/K_{s['K']}/pareto_models/{s['chosen']['trial']}"
    prior = m.build_temporal_prior(mdir, raw1, raw2, soc, sem)
    for f in EXCLUDE:
        prior["facets"].pop(f, None)
    prior["source"]["n_persisting"] = {f: int(len(v["idx"])) for f, v in prior["facets"].items()}
    prior["source"]["excluded_facets"] = EXCLUDE
    for K in (2, 3, 4, 5):
        recs = []
        for seed in m.STABILITY_SEEDS:
            m.set_seeds(seed)
            U, Z, d = m.run_inner_solver(raw_data=raw2, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=K,
                                         dimensions=raw2["dimensions"],
                                         params={"lambda_z_offdiag": LZ, "lambda_temporal": 0.1},
                                         device=m.DEVICE, seed_function=lambda s=seed: m.set_seeds(s),
                                         inner_epochs=m.EXTENDED_EPOCHS, temporal_prior=prior)
            Un = {f: u.cpu().numpy() for f, u in U.items()}
            Zn = {r: z.cpu().numpy() for r, z in Z.items()}
            P2, _ = m.compute_weighted_membership(Un, Zn, m.build_presence_masks(raw2, soc, sem))
            recs.append({"seed": seed, "converged": d["converged"], "epochs": len(d["loss_history"]),
                         "math_loss": d["math_loss"], "match": d["temporal_match"], "P2": P2})
        # seed consistency: align each seed's columns to the first seed's, then compare maps
        ref = np.vstack([recs[0]["P2"][f] for f in sorted(recs[0]["P2"])])
        maps = []
        for r in recs:
            cur = np.vstack([r["P2"][f] for f in sorted(r["P2"])])
            cn = lambda A: A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
            _, col = linear_sum_assignment(-(cn(ref).T @ cn(cur)))   # ref col j <-> cur col col[j]
            to_ref = {int(col[j]): j for j in range(K)}
            maps.append({p["t1"]: to_ref[p["t2"]] for p in r["match"]["pairs"]})
        t1_ids = sorted({a for mp in maps for a in mp})
        consistency = {}
        for a in t1_ids:
            targets = collections.Counter(mp.get(a, "unmatched") for mp in maps)
            consistency[a] = {"modal_target": str(targets.most_common(1)[0][0]),
                              "share_modal": targets.most_common(1)[0][1] / len(maps),
                              "targets": {str(k): v for k, v in targets.items()}}
        n_pairs = [len(r["match"]["pairs"]) for r in recs]
        out[f"{c}_K{K}"] = {
            "K1": prior["K1"], "persisting": prior["source"]["n_persisting"],
            "converged": sum(r["converged"] for r in recs),
            "epochs_median": float(np.median([r["epochs"] for r in recs])),
            "accepted_pairs_per_seed": n_pairs,
            "z_threshold_median": float(np.median([r["match"]["z_threshold"] for r in recs])),
            "accepted_z": [p["z"] for r in recs for p in r["match"]["pairs"]],
            "rejected_z": [p["z"] for r in recs for p in r["match"]["rejected"]],
            "seed_consistency": consistency,
        }
        print(f"{c} K1={prior['K1']} K2={K}: converged {out[f'{c}_K{K}']['converged']}/10, accepted pairs per "
              f"seed {n_pairs}, z threshold {out[f'{c}_K{K}']['z_threshold_median']:.2f}, consistency "
              + ", ".join(f"T1 {a}->{v['modal_target']} {v['share_modal']:.0%}" for a, v in consistency.items()),
              flush=True)
json.dump(out, open(f"t2_matching_{'_'.join(configs)}{SUFFIX}.json", "w"), indent=1, default=float)
