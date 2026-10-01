"""Combine the per-config report files written by `chunk13v9.py --config Cn`."""
import argparse, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/v9.4.t1_v2")
ap.add_argument("--configs", default="C1,C2,C3,C4,C5,C6")
args = ap.parse_args()
configs = args.configs.split(",")

problems = []
for stem in ("scout_methodology_report", "master_dual_track_stability_report", "run_manifest"):
    merged = {}
    for c in configs:
        p = os.path.join(args.results, f"{stem}_{c}.json")
        if not os.path.exists(p):
            problems.append(f"missing {p}")
            continue
        d = json.load(open(p))
        if stem == "run_manifest":
            merged[c] = d
            continue
        for k, v in d.items():
            if k.startswith("_"):
                merged.setdefault(k, v)
            elif k in merged:
                problems.append(f"{stem}: config {k} appears in more than one file")
            else:
                merged[k] = v
    out = os.path.join(args.results, f"{stem}.json" if stem != "run_manifest" else "run_manifests.json")
    json.dump(merged, open(out, "w"), indent=4)
    print(f"wrote {out}: {sorted(k for k in merged if not k.startswith('_'))}")

manifests = json.load(open(os.path.join(args.results, "run_manifests.json")))
for key in ("script_sha256", "data_sha256", "pipeline_version", "coherence_pen_weight", "domain_balance_pen_weight"):
    vals = {m.get(key) for m in manifests.values()}
    if len(vals) > 1:
        problems.append(f"tasks disagree on {key}: {vals}")

for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
