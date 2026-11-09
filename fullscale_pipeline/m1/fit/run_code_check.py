# The code-check world fits (decided 2026-11-08): a code/optimiser check, never evidence about the method.
# Two starts compared on the same worlds (owner 2026-11-09): "nndsvd" (the toy's start) and "random" (per seed); the
# default start is decided later. Owner's context: the NNDSVD start was introduced because random starts left very low or
# absent core-matrix values between semantic units and articles (child×article), which defeated the detection of
# heterogeneous communities — so the child×article measures below are reported for both starts.
# 26 fits (the 2 fits at K* + 1 for the empty-column rule follow at checkpoint 3, with the measures module):
#   part noiseless2 / noiseless1: noiseless world, squared error / generalised KL, 3 seeds × 2 starts
#       pass: relative reconstruction error < 1e-3 in every relation and every planted community matched with cosine
#       ≥ 0.99, in most seeds (else a bug);
#   part poisson (KL) / gaussian (squared error): 1 fit started at the planted truth + 3 seeds × 2 starts
#       pass: random-seed fits reach the truth-started fit's loss and smallest matched cosine within the seeds' own
#       spread, in most seeds (else optimiser settings).
# Usage (from fullscale_pipeline/m1): python -m fit.run_code_check --part P --out DIR ; then --verdict --out DIR
import argparse, glob, json, os, time
import numpy as np
import torch

from . import data as D, worlds as Wd
from .solver import fit
from .loss import RelationTensors, total_loss
from .model import Model

STARTS = ("nndsvd", "random")


def sq_error_per_relation(res, rds):
    rts = [RelationTensors(rd) for rd in rds]
    m = Model({f: u.shape[0] for f, u in res["U"].items()}, res["relations"], res["K"], init_U=res["U"], init_Z=res["Z"])
    with torch.no_grad():
        _, parts = total_loss(dict(m.U), dict(m.Z), rts, 2)
    return {k: float(np.sqrt(max(v.item(), 0.0))) for k, v in parts.items()}   # visible ‖X‖ = 1 per relation


def tie_weight_reading(U, Z, rel, relations, perm=None):
    """Within-community share of relation rel's reconstructed tie weight and each community's relative presence
    (half credit × K), from part(k,l) = Z[k,l] a_k b_l. perm maps planted → fitted community (for per-community order)."""
    a_f, b_f = relations[rel]
    P = Z[rel] * np.outer(U[a_f].sum(0), U[b_f].sum(0))
    T = P.sum()
    h = (np.diag(P) + 0.5 * (P.sum(1) - np.diag(P) + P.sum(0) - np.diag(P))) / T
    pres = len(h) * h
    if perm is not None:
        pres = np.array([pres[perm[k]] for k in sorted(perm)])
    return float(np.trace(P) / T), pres


def summarise(res, rds, Ut, Zt):
    cos, perm = Wd.matched_cosines(res["U"], Ut, res["Z"], Zt, res["relations"])
    err = sq_error_per_relation(res, rds)
    out = {"seed": res["seed"], "start": res["settings"]["start"], "beta": res["beta"], "converged": res["converged"],
           "epochs": res["epochs"], "seconds": res["seconds"], "final_data_loss": res["data_loss_history"][-1],
           "relative_error_per_relation": err, "max_relative_error": max(err.values()),
           "matched_cosines": [round(float(c), 6) for c in cos], "min_cosine": float(cos.min())}
    for rel in ("child_article", "parent_child"):
        w_fit, p_fit = tie_weight_reading(res["U"], res["Z"], rel, res["relations"], perm)
        w_true, p_true = tie_weight_reading(Ut, Zt, rel, res["relations"])
        out[f"{rel}_within_share"] = {"fit": round(w_fit, 4), "planted": round(w_true, 4)}
        out[f"{rel}_presence"] = {"fit": [round(float(x), 3) for x in p_fit], "planted": [round(float(x), 3) for x in p_true]}
    return out


def run_part(part, out_dir, max_epochs):
    build = D.Build()
    res = {}
    if part.startswith("noiseless"):
        beta = int(part[-1])
        rds, Ut, Zt = Wd.code_check_world("noiseless", seed=1, build=build)
        jobs = [(f"noiseless_beta{beta}_{st}_seed{s}", beta, s, st) for st in STARTS for s in (1, 2, 3)]
    else:
        beta = 1 if part == "poisson" else 2
        rds, Ut, Zt = Wd.code_check_world(part, seed=1, build=build)
        r = fit(rds, 6, beta, seed=0, max_epochs=max_epochs, init_U=Ut, init_Z=Zt)
        res[f"{part}_truth_start"] = summarise(r, rds, Ut, Zt)
        print(f"{part}_truth_start: {res[f'{part}_truth_start']}", flush=True)
        jobs = [(f"{part}_{st}_seed{s}", beta, s, st) for st in STARTS for s in (1, 2, 3)]
    for name, beta, s, st in jobs:
        r = fit(rds, 6, beta, seed=s, max_epochs=max_epochs, start=st)
        res[name] = summarise(r, rds, Ut, Zt)
        print(f"{name}: {res[name]}", flush=True)
    json.dump(res, open(os.path.join(out_dir, f"{part}.json"), "w"), indent=1)


def verdicts(out_dir):
    fits = {}
    for f in glob.glob(os.path.join(out_dir, "*.json")):
        if not f.endswith("verdicts.json"):
            fits.update(json.load(open(f)))
    v = {}
    for st in STARTS:
        for beta in (2, 1):
            fs = [fits[f"noiseless_beta{beta}_{st}_seed{s}"] for s in (1, 2, 3)]
            passed = [f["max_relative_error"] < 1e-3 and f["min_cosine"] >= 0.99 for f in fs]
            v[f"noiseless_beta{beta}_{st}"] = {"passed_seeds": int(sum(passed)), "pass": bool(sum(passed) >= 2)}
        for version in ("poisson", "gaussian"):
            tr = fits[f"{version}_truth_start"]
            fs = [fits[f"{version}_{st}_seed{s}"] for s in (1, 2, 3)]
            losses = np.array([f["final_data_loss"] for f in fs]); cmin = np.array([f["min_cosine"] for f in fs])
            loss_margin = max(losses.max() - losses.min(), 1e-4 * abs(tr["final_data_loss"]))
            cos_margin = max(cmin.max() - cmin.min(), 1e-3)
            passed = [(f["final_data_loss"] - tr["final_data_loss"] <= loss_margin) and
                      (tr["min_cosine"] - f["min_cosine"] <= cos_margin) for f in fs]
            v[f"{version}_{st}"] = {"passed_seeds": int(sum(passed)), "pass": bool(sum(passed) >= 2),
                                    "loss_margin": float(loss_margin), "cosine_margin": float(cos_margin)}
    json.dump(v, open(os.path.join(out_dir, "verdicts.json"), "w"), indent=1)
    print(json.dumps(v, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("noiseless2", "noiseless1", "poisson", "gaussian"))
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.verdict:
        verdicts(a.out)
    else:
        run_part(a.part, a.out, a.max_epochs)


if __name__ == "__main__":
    main()
