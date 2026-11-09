# Checkpoint 2 tests (F3 init, F4 terms, F5 solver, F7 worlds), ticket 107. Small synthetic problems only (seconds to a
# few minutes); the code-check world fits themselves run separately (run_code_check.py). Run from fullscale_pipeline/m1:
#   python -m fit.tests.test_checkpoint2
import sys
import numpy as np
import scipy.sparse as sp
import torch

from fit import data as D, terms as T, worlds as Wd
from fit.init import initialise
from fit.solver import fit
from fit.model import Model
from fit.loss import RelationTensors, total_loss

torch.set_num_threads(1)
FAILS = []


def check(name, got, want=True, tol=None):
    ok = (got == want) if tol is None else abs(got - want) <= tol * max(1.0, abs(want))
    print(("PASS " if ok else "FAIL ") + f"{name}: {got}" + ("" if tol is None and want is True else f" (want {want})"))
    if not ok:
        FAILS.append(name)


SMALL = {"article": 40, "child": 120, "parent": 60, "author": 80, "organisation": 20, "journal": 8}
rds, Ut, Zt = Wd.code_check_world("noiseless", seed=1, sizes=SMALL, empty_cap=300)
relations = {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}
rts = [RelationTensors(rd) for rd in rds]
K = 6

# ------------------------------------------------------------------------------------------------ 1. world sanity
mdl = Model({f: u.shape[0] for f, u in Ut.items()}, relations, K, init_U=Ut, init_Z=Zt)
l_truth = float(total_loss(dict(mdl.U), dict(mdl.Z), rts, 2)[0])
check("noiseless world: loss at the planted truth ≈ 0", l_truth, 0.0, 1e-8)
check("semantic-only community absent from every social core matrix",
      all(np.allclose(Zt[r][Wd.SEMANTIC_ONLY], 1e-7) and np.allclose(Zt[r][:, Wd.SEMANTIC_ONLY], 1e-7)
          for r in relations if D.RELATIONS[r][3] == "social"))
check("child-only community has members only among child hyperedges",
      all(np.all(Ut[f][:, Wd.CHILD_ONLY] <= 1e-7) for f in Ut if f != "child") and bool(np.any(Ut["child"][:, Wd.CHILD_ONLY] > 0.1)))
check("every community has pure members in the child facet",
      all(np.any((Ut["child"][:, k] > 0.1) & ((Ut["child"] > 1e-6).sum(1) == 1)) for k in range(K)))
check("one hold-out group hidden in child×article", len(next(rd for rd in rds if rd.name == "child_article").hid_tie_row) > 0)

# ------------------------------------------------------------------------------------------------ 2. initialisation
U1, Z1 = initialise(rds, K, seed=5); U2, Z2 = initialise(rds, K, seed=5); U3, Z3 = initialise(rds, K, seed=6)
check("init: identical for the same seed", all(np.array_equal(U1[f], U2[f]) for f in U1) and all(np.array_equal(Z1[r], Z2[r]) for r in Z1))
check("init: every facet set and positive", set(U1) == set(SMALL) and all(np.all(U1[f] > 0) for f in U1))
diff = np.mean([np.linalg.norm(U1[f] - U3[f]) / np.linalg.norm(U1[f]) for f in U1])
print(f"     init: mean relative difference between seeds 5 and 6 = {diff:.3f} (the NNDSVD part is shared by all seeds)")
rds_h = [D.RelationData(rd.name, rd.row_facet, rd.col_facet, rd.shape, rd.weight, rd.domain, rd.tie_row, rd.tie_col,
                        rd.tie_val, rd.obs_row_mask, rd.diag_masked, rd.hid_tie_row, rd.hid_tie_col,
                        rd.hid_tie_val * 7.0 + 1.0, rd.hid_empty_row, rd.hid_empty_col, rd.fro) for rd in rds]
U4, _ = initialise(rds_h, K, seed=5)
check("init: hidden cells' values cannot change the start", all(np.array_equal(U1[f], U4[f]) for f in U1))

# ------------------------------------------------- 3. term gradients against finite differences (column weights fixed)
W_child = sp.random(SMALL["child"], SMALL["child"], density=0.04, random_state=3, format="csr")
W_child = sp.triu(W_child, 1); W_child = (W_child + W_child.T).tocsr(); W_child.data = 0.4 + 0.6 * W_child.data
ca = next(rd for rd in rds if rd.name == "child_article")
term_list = [T.WordSimilarity("child", W_child),
             T.ConcentrationFloor(ca, gamma=0.8, omega=1.0, beta_w=1.0, W=W_child),
             T.SpreadCeiling("article", rds, rho=0.3), T.SpreadCeiling("child", rds, rho=0.3, beta_w=1.0, W=W_child),
             T.OffDiagonalShare(list(relations))]
torch.manual_seed(0)
model = Model({f: u.shape[0] for f, u in Ut.items()}, relations, K, generator=torch.Generator().manual_seed(2))
_, Un, Zs = model.normalised()
Und = {f: u.detach() for f, u in Un.items()}; Zsd = {r: z.detach() for r, z in Zs.items()}
for t in term_list:
    if t.needs_freeze:
        t.freeze(Und, Zsd, relations)
fixed_w = {f: T.column_weights(Und, Zsd, relations, f) for f in Und}
orig_cw = T.column_weights
T.column_weights = lambda Un_, Zs_, rel_, f: fixed_w[f]          # hold the (detached) column weights fixed
rng = np.random.default_rng(0)
for t in term_list:
    model.zero_grad()
    _, Un, Zs = model.normalised()
    v = t.value(Un, Zs, relations); v.backward()
    worst = 0.0
    for name, p in list(model.U.items()) + list(model.Z.items()):
        for _ in range(4):
            idx = tuple(int(rng.integers(s)) for s in p.shape)
            h = 1e-6 * max(1.0, abs(float(p.data[idx])))
            with torch.no_grad():
                p.data[idx] += h; _, a, b = model.normalised(); fp = float(t.value(a, b, relations))
                p.data[idx] -= 2 * h; _, a, b = model.normalised(); fm = float(t.value(a, b, relations))
                p.data[idx] += h
            fd = (fp - fm) / (2 * h)
            g = 0.0 if p.grad is None else float(p.grad[idx])
            worst = max(worst, abs(fd - g) / max(1e-8, abs(fd), abs(g)))
    check(f"{t.name} ({getattr(t, 'facet', '')}): gradient = finite differences (relative)", worst, 0.0, 1e-5)
T.column_weights = orig_cw

# ------------------------------------------------------------------------------- 4. terms under the scale freedom
vals0 = {t.name + str(i): float(t.value(*model.normalised()[1:], relations)) for i, t in enumerate(term_list)}
c = torch.tensor([2.0, 0.5, 3.0, 1.0, 0.25, 4.0], dtype=torch.float64)
with torch.no_grad():
    model.U["child"].mul_(c)
    for r, (a, b) in relations.items():
        if a == "child": model.Z[r].div_(c[:, None])
        if b == "child": model.Z[r].div_(c[None, :])
vals1 = {t.name + str(i): float(t.value(*model.normalised()[1:], relations)) for i, t in enumerate(term_list)}
check("every term unchanged under the scale freedom", max(abs(vals0[k] - vals1[k]) / max(1e-12, abs(vals0[k])) for k in vals0), 0.0, 1e-9)

# --------------------------------------------------------------------------- 5. floor: hand example and decomposition
X = sp.csr_matrix(([1.0, 1.0, 1.0], ([0, 1, 0], [0, 0, 1])), shape=(2, 2))     # articles × children
rd_hand = D.RelationData("child_article", "article", "child", (2, 2), 1.0, "semantic", X.tocoo().row, X.tocoo().col,
                         X.tocoo().data, np.ones(2, bool), False, np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                         np.zeros(0, int), np.zeros(0, int), 1.0)
fl = T.ConcentrationFloor(rd_hand, gamma=0.8, omega=1.0)
m = np.array([[1.0, 0.0], [0.0, 1.0]])                         # article 0 pure in 0, article 1 pure in 1
q = fl.q.toarray()
mix = q @ m
within = q @ T._entropy_bits_np(np.clip(m, 1e-300, None))
between = T._entropy_bits_np(mix) - within
check("floor hand example: child 0 tied equally to two pure articles → between-article spread 1 bit", float(between[0]), 1.0, 1e-12)
check("floor hand example: child 1 with one article is not targeted", int(fl.n_art[1]), 1)
Pm = rng.dirichlet(np.ones(4), 30); qq = rng.dirichlet(np.ones(30), 5)
mx = qq @ Pm
w_ = qq @ T._entropy_bits_np(Pm)
b_ = T._entropy_bits_np(mx) - w_
js = (qq * (Pm[None] * (np.log2(Pm[None]) - np.log2(mx[:, None]))).sum(-1)).sum(1)   # Σ_a q_a KL(m_a ‖ mix)
check("spread decomposition: between-article spread = generalised Jensen–Shannon divergence", float(np.abs(b_ - js).max()), 0.0, 1e-12)

# ------------------------------------------------------------------------------------------------- 6. solver
r1 = fit(rds, K, 2, seed=11, max_epochs=60, warmup=20)
r2 = fit(rds, K, 2, seed=11, max_epochs=60, warmup=20)
check("solver: two runs with the same seed are bit-identical",
      r1["loss_history"] == r2["loss_history"] and all(np.array_equal(r1["U"][f], r2["U"][f]) for f in r1["U"]))
r3 = fit(rds_h, K, 2, seed=11, max_epochs=60, warmup=20)
check("solver: hidden cells' values cannot change the fit", r1["loss_history"] == r3["loss_history"])
r4 = fit(rds, K, 2, seed=11, max_epochs=60, warmup=20, terms=[(T.WordSimilarity("child", W_child), 0.0)])
check("solver: a term at λ = 0 changes nothing", r1["loss_history"] == r4["loss_history"])
r5 = fit(rds, K, 2, seed=11, max_epochs=80, warmup=20,
         terms=[(T.WordSimilarity("child", W_child), 0.1), (T.ConcentrationFloor(ca, 0.8, 1.0), 0.1)])
check("solver: terms switch on at the warm-up epoch (identical before it)", r5["loss_history"][:20] == r1["loss_history"][:20])
check("solver: terms change the path after the warm-up", r5["loss_history"][25] != r1["loss_history"][25])
check("solver: no stop before warm-up + window when terms are on", r5["epochs"] > 41 or not r5["converged"])
check("solver: membership-change log written every 50 epochs", [l["epoch"] for l in r1["logs"]] == [0, 50])
rk = fit(rds, K, 1, seed=11, max_epochs=40, warmup=20)
check("solver: KL fit runs and stays finite", bool(np.isfinite(rk["loss_history"][-1])))
r32 = fit(rds, K, 2, seed=11, max_epochs=60, warmup=20, dtype=torch.float32)
gap = max(abs(a - b) / abs(b) for a, b in zip(r32["loss_history"][:20], r1["loss_history"][:20]))
check("precision: float32 vs float64 loss over the first 20 epochs, relative gap ≪ stopping threshold 1e-4", gap < 1e-5)
print(f"     float32 vs float64: largest relative loss gap over 20 epochs {gap:.1e}")

# ------------------------------------------------------------------------------------------------ 7. graded swap
sw0, perm0 = Wd.graded_swap(rds, 0.0, seed=3)
check("swap p = 0: identity", np.array_equal(perm0, np.arange(len(perm0))))
sw1, perm1 = Wd.graded_swap(rds, 1.0, seed=3)
check("swap p = 1: semantic relations untouched",
      all(a is b for a, b in zip(rds, sw1) if a.domain == "semantic"))
ok_deg = True
for a, b in zip(rds, sw1):
    if a.domain == "social":
        for side in ("row", "col"):
            da = np.sort(np.bincount(getattr(a, f"tie_{side}"))); db = np.sort(np.bincount(getattr(b, f"tie_{side}")))
            ok_deg &= np.array_equal(da, db)
check("swap p = 1: social relations keep their degree sequences (internal structure kept)", ok_deg)
check("swap p = 1: some articles moved", int((perm1 != np.arange(len(perm1))).sum()) > 0)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
