# Reconstruction loss (math_loss, Optuna objective 0) per K, from the per-config/K Optuna databases of a run
# (ticket 107 §6, 2026-10-14). Counts only COMPLETE trials flagged converged and not duplicates (ticket 92).
# Reads SQLite directly, no Optuna import. Slow on network storage (a few minutes per run).
#   python3 k_loss_by_K.py results/v9.2.t1_v2 results/v9.4.t1_v2
import collections, glob, json, os, sqlite3, statistics, sys

for run in sys.argv[1:] or ["results/v9.2.t1_v2", "results/v9.4.t1_v2"]:
    res = collections.defaultdict(dict)
    for db in sorted(glob.glob(os.path.join(run, "C*", "K_*", "optuna_study.db"))):
        parts = db.split(os.sep)
        cfg, k = parts[-3], int(parts[-2][2:])
        c = sqlite3.connect(db)
        vals = []
        for (tid,) in c.execute("select trial_id from trials where state='COMPLETE'").fetchall():
            v = c.execute("select value from trial_values where trial_id=? and objective=0", (tid,)).fetchone()
            ua = dict(c.execute("select key, value_json from trial_user_attributes where trial_id=?", (tid,)).fetchall())
            if v is None or json.loads(ua.get("converged", "false")) is not True:
                continue
            if "duplicate_of" in ua and json.loads(ua["duplicate_of"]) is not None:
                continue
            vals.append(v[0])
        res[cfg][k] = (min(vals), statistics.median(vals), len(vals)) if vals else None
    print(f"\n=== {run}: math_loss per K, best / median (converged non-duplicate trials)")
    for cfg in sorted(res):
        print(cfg, " | ".join(f"K{k}: " + (f"{r[0]:.4f} / {r[1]:.4f} ({r[2]})" if r else "none converged")
                              for k, r in sorted(res[cfg].items())))
