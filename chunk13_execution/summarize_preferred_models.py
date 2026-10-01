"""Rank archived Pareto models per config and surface diagnostic info for picking
a 'preferred' model. No single winner exists in the data — this just makes the
tradeoffs visible instead of requiring a manual read of the raw JSON files.

For each config: picks the K with the best scout-phase hypervolume (skipping any
K whose Pareto front is empty), then lists every archived model at that K sorted
by math_loss, with its sociological-penalty breakdown and stability consensus
(if the stability test completed for that model).
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/v9.2.t1_v2")
ap.add_argument("--configs", default="C1,C2,C3,C4,C5,C6")
args = ap.parse_args()
configs = args.configs.split(",")

scout = json.load(open(os.path.join(args.results, "scout_methodology_report.json")))
stability = json.load(open(os.path.join(args.results, "master_dual_track_stability_report.json")))

for c in configs:
    ks = {k: v for k, v in scout[c].items() if not k.startswith("_")}
    usable_ks = {k: v for k, v in ks.items() if v.get("pareto_size", 0) > 0}
    if not usable_ks:
        print(f"\n=== {c}: NO USABLE K (every K's Pareto front is empty) ===")
        continue
    best_k = max(usable_ks, key=lambda k: usable_ks[k]["hypervolume"])
    print(f"\n=== {c}: best K = {best_k} (hv={ks[best_k]['hypervolume']:.4f}, "
          f"pareto_size={ks[best_k]['pareto_size']}) ===")
    other_ks = [k for k in ks if k != best_k]
    empty = [k for k in other_ks if ks[k].get("pareto_size", 0) == 0]
    if empty:
        print(f"    (K={','.join(empty)} produced zero archived models — "
              f"excluded from consideration, not just losing on hypervolume)")

    k_dir = os.path.join(args.results, c, f"K_{best_k}", "pareto_models")
    manifest_path = os.path.join(k_dir, "experiment_manifest.json")
    if not os.path.exists(manifest_path):
        print("    no manifest found")
        continue
    manifest = json.load(open(manifest_path))
    rows = []
    for info in manifest["archived_models"]:
        name = info["folder_name"]
        meta = json.load(open(os.path.join(k_dir, name, "model_metadata.json")))
        ua = meta["user_attrs"]
        cons = stability.get(c, {}).get(best_k.strip() if isinstance(best_k, str) else str(best_k), {}).get(name)
        stab = (f"A={cons['Track_A_Unweighted_Mean']:.2f}±{cons['Track_A_Unweighted_SD']:.2f}"
                if cons else "FAILED/MISSING")
        rows.append((meta["optuna_math_loss"], meta["optuna_soc_penalty"], name, ua, stab))
    rows.sort(key=lambda r: r[0])
    print(f"    {'trial':<12}{'math_loss':<11}{'soc_pen':<10}{'collapse':<10}{'dev_k':<8}{'stability (Track A)'}")
    for math_loss, soc_pen, name, ua, stab in rows:
        print(f"    {name:<12}{math_loss:<11.4f}{soc_pen:<10.4f}"
              f"{ua.get('collapse_pen', 0):<10.4f}{ua.get('mean_dev_k', 0):<8.3f}{stab}")
