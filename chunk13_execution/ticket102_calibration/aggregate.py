# Seed-paired comparison of each (cell, setting) against the lambda_conc=0 baseline.
import json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(f"{HERE}/rows_main.json")) + (json.load(open(f"{HERE}/rows_warm.json")) if os.path.exists(f"{HERE}/rows_warm.json") else [])
stab = {(r["config"], r["K"], r["lambda_conc"], r["gamma"], r["warmup"]): r
        for f in ["stab_main.json", "stab_warm.json"] if os.path.exists(f"{HERE}/{f}") for r in json.load(open(f"{HERE}/{f}"))}
base = {(r["config"], r["K"], r["seed"]): r for r in rows if r["lambda_conc"] == 0}


def art_changed(a, b, K):
    a, b = np.array(a), np.array(b); ok = (a >= 0) & (b >= 0)
    C = np.zeros((K, K))
    for i, j in zip(a[ok], b[ok]): C[i, j] += 1
    r, c = linear_sum_assignment(-C)
    return 1.0 - C[r, c].sum() / max(ok.sum(), 1)


def mass_ratio(r, b):
    mb = dict(zip(b["_ids"], b["_mass"])); x = []
    for i, v in zip(r["_ids"], r["_mass"]):
        if mb.get(i, 0) > 0: x.append(v / mb[i])
    x = np.array(x)
    return float(np.median(x)), float(np.percentile(x, 5)), float(np.mean(x < 0.1))


out = []
settings = sorted({(r["lambda_conc"], r["gamma"], r["warmup"]) for r in rows if r["lambda_conc"] > 0})
cells = sorted({(r["config"], r["K"]) for r in rows})
for c, K in cells:
    for st in [(0.0, 0.5, 300)] + settings:
        g = [r for r in rows if (r["config"], r["K"]) == (c, K) and (r["lambda_conc"], r["gamma"], r["warmup"]) == st]
        if not g: continue
        d = {k: [] for k in ["excl", "loss", "penB", "penA", "soc", "unsup", "H", "art", "mmed", "m5", "mlow", "uneng"]}
        for r in g:
            b = base.get((c, K, r["seed"]))
            if b is None: continue
            d["excl"].append(r["shared_excl_frac"] - b["shared_excl_frac"])
            d["loss"].append(r["math_loss"] - b["math_loss"])
            d["penB"].append(r["penB_w"] - b["penB_w"]); d["penA"].append(r["penA_w"] - b["penA_w"])
            d["soc"].append(r["soc_penalty"] - b["soc_penalty"]); d["unsup"].append(r["unsup_excl_frac"] - b["unsup_excl_frac"])
            d["H"].append((r["shared_H_mean"] or 0) - (b["shared_H_mean"] or 0))
            d["art"].append(art_changed(r["_art_top"], b["_art_top"], K))
            mm = mass_ratio(r, b); d["mmed"].append(mm[0]); d["m5"].append(mm[1]); d["mlow"].append(mm[2])
            d["uneng"].append(r["sem_unengaged"] - b["sem_unengaged"])
        s = stab.get((c, K) + st, {})
        se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        out.append(dict(config=c, K=K, lambda_conc=st[0], gamma=st[1], warmup=st[2], n=len(g),
                        converged=sum(r["converged"] for r in g), epochs=float(np.mean([r["epochs"] for r in g])),
                        excl=float(np.mean([r["shared_excl_frac"] for r in g])),
                        d_excl=float(np.mean(d["excl"])), se_excl=se(d["excl"]),
                        d_loss=float(np.mean(d["loss"])), rel_loss=float(np.mean(d["loss"]) / np.mean([base[(c, K, r["seed"])]["math_loss"] for r in g])),
                        d_penB=float(np.mean(d["penB"])), d_penA=float(np.mean(d["penA"])), d_soc=float(np.mean(d["soc"])),
                        d_unsup=float(np.mean(d["unsup"])), d_H=float(np.mean(d["H"])),
                        art_changed=float(np.mean(d["art"])), mass_median_ratio=float(np.mean(d["mmed"])),
                        mass_p5_ratio=float(np.mean(d["m5"])), mass_frac_below_0p1=float(np.mean(d["mlow"])),
                        d_unengaged=float(np.mean(d["uneng"])), min_share=float(np.mean([r["min_share"] for r in g])),
                        stab_A=s.get("A"), stab_B=s.get("B"), A_chance_max=s.get("A_chance_max"), B_chance_max=s.get("B_chance_max"),
                        stab_n=s.get("n")))
json.dump(out, open(f"{HERE}/summary.json", "w"), indent=1)
print(len(out))
