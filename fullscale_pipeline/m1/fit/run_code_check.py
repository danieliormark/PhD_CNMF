# The code-check world fits (decided 2026-11-08): a code/optimiser check, never evidence about the method.
# 14 fits here (the 2 fits at K* + 1 for the empty-column rule follow at checkpoint 3, when the measures module exists):
#   noiseless world: squared error and generalised KL × 3 random seeds          (pass: relative reconstruction error
#                    < 1e-3 and every planted community matched with cosine ≥ 0.99, in most seeds → else a bug);
#   Poisson world, KL; Gaussian world, squared error: 3 random seeds + 1 fit started at the planted truth each
#                    (pass: random-start fits reach the started-at-truth fit's loss and matched cosine within the seeds'
#                    own spread, in most seeds → else optimiser settings).
# Usage (from fullscale_pipeline/m1): python -m fit.run_code_check [--max-epochs N] [--out file.json]
import argparse, json, time
import numpy as np
import torch

from . import data as D, worlds as Wd
from .solver import fit
from .loss import RelationTensors, total_loss
from .model import Model


def sq_error_per_relation(res, rds):
    rts = [RelationTensors(rd) for rd in rds]
    m = Model({f: u.shape[0] for f, u in res["U"].items()}, res["relations"], res["K"], init_U=res["U"], init_Z=res["Z"])
    with torch.no_grad():
        _, parts = total_loss(dict(m.U), dict(m.Z), rts, 2)
    return {k: float(np.sqrt(max(v.item(), 0.0))) for k, v in parts.items()}   # visible ‖X‖ = 1 per relation


def summarise(res, rds, Ut):
    cos, _ = Wd.matched_cosines(res["U"], Ut)
    err = sq_error_per_relation(res, rds)
    return {"seed": res["seed"], "beta": res["beta"], "converged": res["converged"], "epochs": res["epochs"],
            "seconds": res["seconds"], "final_data_loss": res["data_loss_history"][-1],
            "relative_error_per_relation": err, "max_relative_error": max(err.values()),
            "matched_cosines": [round(float(c), 6) for c in cos], "min_cosine": float(cos.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--out", default="code_check_results.json")
    a = ap.parse_args()
    build = D.Build()
    out = {"max_epochs": a.max_epochs, "fits": {}}
    t0 = time.time()
    rds, Ut, Zt = Wd.code_check_world("noiseless", seed=1, build=build)
    for beta in (2, 1):
        for s in (1, 2, 3):
            r = fit(rds, 6, beta, seed=s, max_epochs=a.max_epochs)
            out["fits"][f"noiseless_beta{beta}_seed{s}"] = summarise(r, rds, Ut)
            print(f"noiseless β={beta} seed {s}: {out['fits'][f'noiseless_beta{beta}_seed{s}']}", flush=True)
    for version, beta in (("poisson", 1), ("gaussian", 2)):
        rds, Ut, Zt = Wd.code_check_world(version, seed=1, build=build)
        r = fit(rds, 6, beta, seed=0, max_epochs=a.max_epochs, init_U=Ut, init_Z=Zt)
        out["fits"][f"{version}_truth_start"] = summarise(r, rds, Ut)
        print(f"{version} truth start: {out['fits'][f'{version}_truth_start']}", flush=True)
        for s in (1, 2, 3):
            r = fit(rds, 6, beta, seed=s, max_epochs=a.max_epochs)
            out["fits"][f"{version}_seed{s}"] = summarise(r, rds, Ut)
            print(f"{version} seed {s}: {out['fits'][f'{version}_seed{s}']}", flush=True)
    # verdicts
    v = {}
    for beta in (2, 1):
        fs = [out["fits"][f"noiseless_beta{beta}_seed{s}"] for s in (1, 2, 3)]
        passed = [f["max_relative_error"] < 1e-3 and f["min_cosine"] >= 0.99 for f in fs]
        v[f"noiseless_beta{beta}"] = {"passed_seeds": int(sum(passed)), "pass": sum(passed) >= 2}
    for version in ("poisson", "gaussian"):
        tr = out["fits"][f"{version}_truth_start"]
        fs = [out["fits"][f"{version}_seed{s}"] for s in (1, 2, 3)]
        losses = np.array([f["final_data_loss"] for f in fs]); cmin = np.array([f["min_cosine"] for f in fs])
        loss_margin = max(losses.max() - losses.min(), 1e-4 * abs(tr["final_data_loss"]))
        cos_margin = max(cmin.max() - cmin.min(), 1e-3)
        passed = [(f["final_data_loss"] - tr["final_data_loss"] <= loss_margin) and
                  (tr["min_cosine"] - f["min_cosine"] <= cos_margin) for f in fs]
        v[version] = {"passed_seeds": int(sum(passed)), "pass": sum(passed) >= 2,
                      "loss_margin": float(loss_margin), "cosine_margin": float(cos_margin)}
    out["verdicts"] = v
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
