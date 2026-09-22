#!/usr/bin/env python3
"""
Stage A6 (mechanical, no agent) — build results_index.json from diagnostic_results/*.json.

For each result file, keep every top-level scalar, plus any key that looks like an aggregate
("summary", "cell_summaries", "per_slice_so_far", "per_slice", etc.) truncated to a sane size.
Large per-trial/per-cell raw arrays ("results", "records") are dropped except for a length count
and, if small (<=30 items) and each item is a flat dict of scalars, kept in full — this is meant to
give Stage C adjudication agents the aggregate numbers FINDINGS.md would actually cite, without
ballooning the ledger with 2000-record Optuna traces.
"""
import json
import os

RESULTS_DIR = "diagnostic_results"
OUT_PATH = "/mnt/hum01-home01/p91688di/PhD_CNMF/audit_workdir/ledger/results_index.json"

AGGREGATE_KEY_HINTS = ("summary", "cell_summaries", "per_slice", "verdict", "config")
RAW_KEY_HINTS = ("results", "records", "trials", "raw")


def is_flat_scalar_dict(d):
    return isinstance(d, dict) and all(
        not isinstance(v, (dict, list)) or (isinstance(v, list) and len(v) <= 10)
        for v in d.values()
    )


def summarize(obj, key_hint=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lower = k.lower()
            if isinstance(v, list):
                if len(v) <= 30 and all(is_flat_scalar_dict(x) or not isinstance(x, dict) for x in v):
                    out[k] = v
                elif any(h in lower for h in AGGREGATE_KEY_HINTS):
                    out[k] = v[:30]
                    out[f"_{k}_truncated_from"] = len(v)
                else:
                    out[f"_{k}_len"] = len(v)
                    out[f"_{k}_dropped_raw"] = True
            elif isinstance(v, dict):
                out[k] = summarize(v, k)
            else:
                out[k] = v
        return out
    return obj


def main():
    index = {}
    errors = {}
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            index[stem] = {
                "source_file": path,
                "mtime": os.path.getmtime(path),
                "content": summarize(data),
            }
        except Exception as e:
            errors[stem] = str(e)

    with open(OUT_PATH, "w") as f:
        json.dump({"results": index, "errors": errors, "n_files": len(index)}, f, indent=1, default=str)

    print(f"Wrote {len(index)} entries ({len(errors)} errors) to {OUT_PATH}")
    if errors:
        for k, v in errors.items():
            print(f"  ERROR {k}: {v}")


if __name__ == "__main__":
    main()
