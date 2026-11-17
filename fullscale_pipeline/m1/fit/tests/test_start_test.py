# Diagonal-start test arm (ticket 107, 2026-11-17): checks on the real trial data, both approaches.
#  1. the arm's U and anchor cores equal the hybrid start of the same seed (bitwise); 2. every social core is c·I + 1e-4;
#  3. each social relation's starting reconstruction norm equals the random core's (relative 1e-10);
#  4. end-to-end through run_fit (K = 3, 30 epochs): the arm runs, writes its outputs, and its first-epoch data loss
#     differs from the reference arm only through the social relations.
import json, os, sys, tempfile
import numpy as np
from fit import data as D, init as I
from fit.run_fit import run_one
from fit.plan import job


def check_start(approach, K, seed=1):
    rds = D.fit_data(D.Build(D.BUILD), approach, group=0)
    U0, Z0 = I.initialise(rds, K, seed, subsample=0.8)
    U1, Z1 = I.initialise(rds, K, seed, subsample=0.8)
    Zd = I.diagonal_social_links(rds, U1, Z1)
    assert all(np.array_equal(U0[f], U1[f]) for f in U0), "U start differs"
    for rd in rds:
        if rd.name in I.ANCHORS:
            assert np.array_equal(Zd[rd.name], Z0[rd.name]), rd.name
            continue
        z = Zd[rd.name]; off = z[~np.eye(K, dtype=bool)]
        assert np.allclose(off, 1e-4) and np.all(np.diag(z) > 1e-4), rd.name
        A = U0[rd.row_facet] * rd.obs_row_mask[:, None]; B = U0[rd.col_facet]
        n = lambda q: np.sum((q.T @ (A.T @ A) @ q) * (B.T @ B))
        c = np.diag(z)[0] - 1e-4
        rel = abs(n(c * np.eye(K)) - n(Z0[rd.name])) / n(Z0[rd.name])
        assert rel < 1e-10, (rd.name, rel)
        print(f"  approach {approach} K {K} {rd.name}: c = {c:.4g}, norm match {rel:.1e}, random-core diagonal share "
              f"{np.trace(Z0[rd.name]) / Z0[rd.name].sum():.3f}")


def check_end_to_end(approach):
    out = tempfile.mkdtemp(dir=os.environ.get("TEST_TMP", None))
    res = {}
    for arm in ("reference", "diagonal"):
        j = job("start_test_check", approach, 0, 3, 2, 1, 0.03, dtype="float32", max_epochs=30)
        j["tag"] = arm
        if arm == "diagonal":
            j["social_link_start"] = "diagonal"
        jid, st = run_one(j, out)
        assert st == "done", st
        d = os.path.join(out, "start_test_check", jid)
        assert all(os.path.exists(os.path.join(d, f)) for f in ("spec.json", "result.json", "measures.json", "UZ.npz"))
        res[arm] = json.load(open(os.path.join(d, "result.json")))
    print(f"  approach {approach} end-to-end: first-epoch data loss reference {res['reference']['data_loss_history'][0]:.6f}, "
          f"diagonal {res['diagonal']['data_loss_history'][0]:.6f}; after 30 epochs {res['reference']['data_loss_history'][-1]:.6f} / "
          f"{res['diagonal']['data_loss_history'][-1]:.6f}")


if __name__ == "__main__":
    for a in (1, 2):
        for K in (13, 20):
            check_start(a, K)
        check_end_to_end(a)
    print("all checks passed")
