"""Within-config model selection (ticket 91). Run after merge_chunk13_reports.py.

Rule, applied separately to each config (never across configs, ticket 93):
1. A model qualifies if all stability seeds converged and both Track A and
   Track B exceed the largest chance-level score of that model's own shuffled
   pairs (stability_qualifies, computed in Section 5).
   --margin adds a required excess over that chance maximum (default 0).
2. Among K values with at least one qualifying model, take the K with the
   highest scout hypervolume.
3. Within that K, take the knee of the qualifying models: both objectives are
   rescaled to [0, 1] over the qualifying models, and the model closest to
   (0, 0) is chosen.

Writes model_selection.json and model_selection.md next to the reports.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/v9.3.t1_v2")
ap.add_argument("--margin", type=float, default=0.0,
                help="required excess of each track over its chance maximum")
args = ap.parse_args()

scout = json.load(open(os.path.join(args.results, "scout_methodology_report.json")))
configs = sorted(k for k in scout if not k.startswith("_"))
selection, md = {}, ["# Model selection (ticket 91 rule)\n", f"Margin over chance maximum: {args.margin}\n"]

for c in configs:
    candidates = {}
    for K_str, rep in scout[c].items():
        if K_str.startswith("_"):
            continue
        kd = os.path.join(args.results, c, f"K_{K_str}", "pareto_models")
        mp = os.path.join(kd, "experiment_manifest.json")
        if not os.path.exists(mp):
            continue
        manifest = json.load(open(mp))
        for info in manifest["archived_models"]:
            meta = json.load(open(os.path.join(kd, info["folder_name"], "model_metadata.json")))
            a = meta.get("consensus_track_A_unweighted", {})
            b = meta.get("consensus_track_B_magnitude_weighted", {})
            ok = (meta.get("stability_seeds_converged") == len(meta.get("stability_test_seeds_used", []))
                  and a and b
                  and a["global_js_similarity_mean"] > a["chance_max"] + args.margin
                  and b["global_cosine_similarity_mean"] > b["chance_max"] + args.margin)
            if ok:
                candidates.setdefault(K_str, []).append(dict(
                    trial=info["folder_name"], math_loss=meta["optuna_math_loss"],
                    soc_penalty=meta["optuna_soc_penalty"],
                    track_A=a["global_js_similarity_mean"], track_A_chance_max=a["chance_max"],
                    track_B=b["global_cosine_similarity_mean"], track_B_chance_max=b["chance_max"],
                    thin_front=manifest.get("thin_front"), deep_dived=manifest.get("deep_dived")))
    md.append(f"\n## {c}\n")
    if not candidates:
        selection[c] = None
        md.append("No qualifying model at any K.\n")
        continue
    best_K = max(candidates, key=lambda k: scout[c][k].get("hypervolume", 0.0))
    models = candidates[best_K]
    pts = np.array([[x["math_loss"], x["soc_penalty"]] for x in models])
    span = pts.max(0) - pts.min(0)
    norm = (pts - pts.min(0)) / np.where(span > 0, span, 1.0)
    chosen = models[int(np.argmin(np.linalg.norm(norm, axis=1)))]
    selection[c] = dict(K=int(best_K), hypervolume=scout[c][best_K].get("hypervolume"),
                        qualifying_by_K={k: len(v) for k, v in candidates.items()}, chosen=chosen)
    md.append(f"Qualifying models by K: {selection[c]['qualifying_by_K']}  \n")
    md.append(f"K chosen by hypervolume among qualifying K: **{best_K}** "
              f"(hypervolume {scout[c][best_K].get('hypervolume'):.4f})  \n")
    md.append(f"Knee model: **{chosen['trial']}** -- math_loss {chosen['math_loss']:.4f}, "
              f"sociological_penalty {chosen['soc_penalty']:.4f}, Track A {chosen['track_A']:.3f} "
              f"(chance max {chosen['track_A_chance_max']:.3f}), Track B {chosen['track_B']:.3f} "
              f"(chance max {chosen['track_B_chance_max']:.3f})\n")

json.dump(selection, open(os.path.join(args.results, "model_selection.json"), "w"), indent=2)
open(os.path.join(args.results, "model_selection.md"), "w").write("\n".join(md))
print("\n".join(md))
