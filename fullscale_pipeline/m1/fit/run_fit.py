# F8 (1/3) — run M1 fits from a job list (ticket 107). One job = one fit; a SLURM array task runs a slice of the list
# in a pool of single-threaded processes (determinism: one thread per fit, tickets 76/101).
# Per fit, written under <out>/<stage>/<job_id>/ (RDS: author data stays on the cluster):
#   spec.json        the job (every setting), manifest (code commit and file hashes, build and hold-out manifests, host,
#                    SLURM ids, start/end time)
#   result.json      convergence, epochs, histories, logs (membership change, block sizes), term info, seconds
#   UZ.npz           raw U per facet and raw Z per relation, float32 (archive; measures are computed in float64 first)
#   measures.json    per-fit measures: presence/domain/facet presence and mono flags, empty columns (5 references),
#                    held-out per relation and combined, training AUC by band (child×article both sides), loading
#                    distributions, floor diagnostics
#   entity.npz       per facet: reproduced share, degree-only reproduced share, tie counts (for placed entities)
# Usage: python -m fit.run_fit --jobs JOBS.jsonl --out ROOT [--task T --per-task P] [--only i,j,...]
import argparse, hashlib, json, os, socket, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def job_id(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def manifest(build_path):
    files = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {"code_commit": commit, "code_dirty": dirty, "code_sha": {f: _sha(os.path.join(HERE, f)) for f in files},
            "build_manifest_sha": _sha(build_path + "manifest.json"),
            "holdout_manifest_sha": _sha(build_path + "holdout/manifest.json"),
            "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
            "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID")}


def build_terms(job, rds, build):
    """Terms from the job's 'terms' dict: lambda_child (W regulariser), floor {lambda, gamma, omega, beta_w},
    ceiling {lambda, rho, beta_w, facets}, o3 {lambda}. λ = 0 or missing → not built."""
    import scipy.sparse as sp
    from . import terms as T
    t = job.get("terms") or {}
    out = []
    W = None
    def getW():
        nonlocal W
        if W is None:
            W = sp.load_npz(build.path + "W_child_child.npz").tocsr()
        return W
    if t.get("lambda_child", 0) > 0:
        out.append((T.WordSimilarity("child", getW()), t["lambda_child"]))
    fl = t.get("floor") or {}
    if fl.get("lambda", 0) > 0:
        ca = next(rd for rd in rds if rd.name == "child_article")
        out.append((T.ConcentrationFloor(ca, fl["gamma"], fl["omega"], fl.get("beta_w", 0.0), getW() if fl.get("beta_w", 0) else None), fl["lambda"]))
    ce = t.get("ceiling") or {}
    if ce.get("lambda", 0) > 0:
        for f in ce.get("facets", ["child", "article"]):
            out.append((T.SpreadCeiling(f, rds, ce["rho"], ce.get("beta_w", 0.0), getW() if ce.get("beta_w", 0) else None), ce["lambda"]))
    o3 = t.get("o3") or {}
    if o3.get("lambda", 0) > 0:
        out.append((T.OffDiagonalShare([rd.name for rd in rds]), o3["lambda"]))
    return out


def run_one(job, out_root):
    import torch
    torch.set_num_threads(1)
    from . import data as D, measures as M
    from .solver import fit
    jid = job_id(job)
    d = os.path.join(out_root, job["stage"], jid)
    if os.path.exists(os.path.join(d, "measures.json")):
        return jid, "exists"
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    build = D.Build(job.get("build", D.BUILD))
    arm = job.get("arm") or {}
    rds = D.fit_data(build, job["approach"], group=job["group"], journal_multiplier=arm.get("journal_multiplier", 1.0),
                     semantic_split=arm.get("semantic_split"), drop=tuple(arm.get("drop", ())), use_journal=arm.get("use_journal", True))
    spec = {"job": job, "job_id": jid, "manifest": manifest(build.path), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(spec, open(os.path.join(d, "spec.json"), "w"), indent=1)
    dtype = torch.float32 if job.get("dtype") == "float32" else torch.float64
    res = fit(rds, job["K"], job["beta"], seed=job["seed"], terms=build_terms(job, rds, build),
              max_epochs=job.get("max_epochs", 5000), lr=job["frac"], lr_mode=job.get("lr_mode", "recompute"),
              warmup=job.get("warmup", 300), dtype=dtype, start=job.get("start", "nndsvd_sub0.8"), tol=job.get("tol", 1e-4))
    U = {f: u.astype(np.float64) for f, u in res["U"].items()}; Z = {r: z.astype(np.float64) for r, z in res["Z"].items()}
    np.savez_compressed(os.path.join(d, "UZ.npz"), **{f"U__{f}": u.astype(np.float32) for f, u in U.items()},
                        **{f"Z__{r}": z.astype(np.float32) for r, z in Z.items()})
    rel = res["relations"]
    wm = M.weighted_memberships(U, Z, rel)
    pres = M.presence(U, Z, rds)
    dom = M.domain_presence(pres); fac = M.facet_presence(pres)
    hos = {rd.name: M.heldout(U, Z, rd) for rd in rds}
    hos = {k: v for k, v in hos.items() if v is not None}
    w = {rd.name: rd.weight for rd in rds}
    em = M.entity_measures(U, Z, rds)
    np.savez_compressed(os.path.join(d, "entity.npz"), **{f"{f}__{k}": v for f, e in em.items()
                        for k, v in e.items() if k in ("reproduced_share", "reproduced_share_degree_only", "ties")})
    ca = next((rd for rd in rds if rd.name == "child_article"), None)
    auc = {}
    if ca is not None:
        for side in ("row", "col"):
            a, ties = M.entity_training_auc(U, Z, ca, side=side, seed=job["seed"])
            auc[f"child_article_{side}"] = M.banded_mean(a, ties)
    tolist = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    meas = {"presence": {r: {k: tolist(v) for k, v in p.items()} for r, p in pres.items()},
            "domain_presence": {k: ({kk: tolist(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in dom.items()},
            "facet_presence": {f: v.tolist() for f, v in fac.items()},
            "mono_flags": M.mono_flags(dom, fac),
            "empty_columns": M.empty_columns(pres, rds, wm),
            "heldout": hos,
            "heldout_combined": {"squared_error": M.combined_heldout(hos, w, "squared_error"),
                                 "kl_error": M.combined_heldout(hos, w, "kl_error"),
                                 "semantic_only_squared_error": M.combined_heldout({k: v for k, v in hos.items() if k in ("child_article", "parent_child")}, w, "squared_error"),
                                 "semantic_only_kl_error": M.combined_heldout({k: v for k, v in hos.items() if k in ("child_article", "parent_child")}, w, "kl_error")},
            "training_auc_by_band": auc,
            "loading_distributions": M.loading_distributions(wm)}
    if ca is not None:
        fd = M.floor_diagnostics(ca, wm)
        meas["floor_diagnostics"] = {"n": fd["n"], "share_largest_above": fd["share_largest_above"],
                                     "mean_between_bits": float(fd["between"].mean()) if fd["n"] else None,
                                     "mean_within_bits": float(fd["within"].mean()) if fd["n"] else None,
                                     "mean_own_bits": float(fd["own_spread"].mean()) if fd["n"] else None}
    json.dump(meas, open(os.path.join(d, "measures.json"), "w"), default=tolist)
    keep = {k: v for k, v in res.items() if k not in ("U", "Z")}
    keep["seconds_total"] = round(time.time() - t0, 1)
    keep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(keep, open(os.path.join(d, "result.json"), "w"), default=tolist)
    return jid, "done"


def run_safe(job, out_root):
    """run_one with the error caught and written to error.json, so one failing fit does not stop the others of its task."""
    import traceback
    try:
        if job.get("fail_on_purpose"):
            raise RuntimeError("deliberate failure (rehearsal)")
        return run_one(job, out_root)
    except Exception as e:
        d = os.path.join(out_root, job["stage"], job_id(job)); os.makedirs(d, exist_ok=True)
        json.dump({"job": job, "error": repr(e), "traceback": traceback.format_exc(), "host": socket.gethostname(),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(d, "error.json"), "w"), indent=1)
        return job_id(job), "error: " + repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=None); ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.only:
        sel = [int(i) for i in a.only.split(",")]
    elif a.task is not None:
        sel = list(range(a.task * a.per_task, min(len(jobs), (a.task + 1) * a.per_task)))
    else:
        sel = list(range(len(jobs)))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max(1, min(a.per_task, len(sel)))) as ex:
        for jid, st in ex.map(run_safe, [jobs[i] for i in sel], [a.out] * len(sel)):
            print(jid, st, flush=True)


if __name__ == "__main__":
    main()
