# D1-M4 noise check (RL-397/398). Usage (from fullscale_pipeline/m1, PYTHONPATH=.): python fit/d1m4_check.py OUT.json [--k20]
# D1-M4 illustration on saved stage-0 pass-3 fits (J2 setting): journal-shaped communities per fit, as pre-registered
# (share_kj over covered articles with a journal; journals with >= 2 trial articles; 999 label shuffles; p < 0.05 and j_k >= 0.25).
import os, sys, glob, json
os.environ["M1_ALIGNED"] = "1"
from collections import defaultdict
import numpy as np, scipy.sparse as sp
sys.path.insert(0, ".")
from fit import data as D, measures as M
from fit.aggregate import load_fit, load_UZ, latest_fits, _rds_of, COVERAGE_C

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits/stage0_pass3"
CELLS = {(1, 5, 0.3), (1, 13, 0.3), (2, 5, 0.3), (2, 13, 1.0)}          # run 1 (seeds 1–5)
CELLS_K20 = {(1, 20, 1.0), (2, 20, 1.0)}                                  # run 2 (seeds 1–10): --k20
NSH = 999

def jshaped(P, lab, J, rng):
    n, K = P.shape
    size = np.bincount(lab, minlength=J); ok = size >= 2
    def jk(l):
        S = sp.csr_matrix((np.ones(n), (l, np.arange(n))), shape=(J, n)) @ P
        S = S / np.maximum(P.sum(0), 1e-300)
        return S[ok].max(0), S.argmax(0)
    real, arg = jk(lab)
    ge = np.zeros(K)
    for _ in range(NSH):
        ge += jk(rng.permutation(lab))[0] >= real
    p = (1 + ge) / (1 + NSH)
    single = ~ok[arg]
    return {"journal_shaped": int(((p < 0.05) & (real >= 0.25)).sum()), "associated": int(((p < 0.05) & (real < 0.25)).sum()),
            "singleton_top": int(single.sum()), "jk": real.tolist()}

def main():
    global CELLS, MAXSEED
    MAXSEED = 5
    if "--k20" in sys.argv:
        CELLS, MAXSEED = CELLS_K20, 10
    b = D.Build(D.BUILD); r, c, _ = b.ties("article_journal")
    jour = -np.ones(b.sizes["article"], dtype=np.int64); jour[r] = c
    fits = latest_fits([load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, "*", "measures.json"))])
    by = defaultdict(list)
    for f in fits:
        j = f["job"]
        if (j["approach"], j["K"], float(j["arm"]["r"])) in CELLS and f["result"]["converged"] and j["seed"] <= MAXSEED:
            by[(j["approach"], j["K"], float(j["arm"]["r"]), j["group"])].append(f)
    out = []
    rng = np.random.default_rng(0)
    for key, fs in sorted(by.items()):
        fs = sorted(fs, key=lambda f: f["job"]["seed"])
        rel = {k: tuple(v) for k, v in fs[0]["result"]["relations"].items()}
        UZ = [load_UZ(f["dir"]) for f in fs]
        covs = [M.entity_coverage(U, Z, _rds_of(f["job"])) for (U, Z), f in zip(UZ, fs)]
        keep = M.covered_entities(covs, COVERAGE_C)["article"] & (jour >= 0)
        _, lab = np.unique(jour[keep], return_inverse=True); J = lab.max() + 1
        for (U, Z), f in zip(UZ, fs):
            P = M.weighted_memberships(U, Z, rel)["article"]["P"][keep]
            res = jshaped(P, lab, J, rng)
            out.append({"approach": key[0], "K": key[1], "r": key[2], "group": key[3], "seed": f["job"]["seed"],
                        "n_articles": int(keep.sum()), **res})
            print(key, f["job"]["seed"], res["journal_shaped"], res["associated"], res["singleton_top"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)

if __name__ == "__main__":
    main()
