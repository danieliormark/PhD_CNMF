"""Ticket 105, test T1: unit checks of the temporal prior (chunk13v10.py).

(a) gradcheck of temporal_loss_value (float64; column norms and weights held fixed, as
    they are detached in the solver)
(b) ticket-79 gauge: rescaling a column of U_pos (U_scales follows) leaves the term unchanged
(c) permutation: permuting the T2 columns permutes the match and leaves the loss unchanged
(d) empty persisting set: a real fit with such a prior is identical to a fit without one
(e) a T2 column with zero prior mass (unmatched) gives a finite loss and gradient
(f) the gradient on every non-persisting row is exactly 0, on persisting rows non-zero
(g) for comparison: with column norms NOT detached, non-persisting rows do get a gradient
    (the coupling the stop-gradient removes), and the v9.4 concentration term, which uses
    U_norm directly, has the same coupling (CLAUDE.md §4.25 side note)
(h) the match's false-acceptance rate on unrelated memberships, and recovery of a planted
    permutation

    ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE python t1_unit_checks.py
"""
import importlib.util, os, sys, json, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c13v10", f"{E}/chunk13v10.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(7)
results = {}


def toy_state(K=3):
    N = {"core_atom": 8, "auth": 5}
    U = {f: torch.tensor(rng.random((n, K)) + 0.05, requires_grad=True) for f, n in N.items()}
    w = {f: torch.tensor(rng.random(K) + 0.1) for f in N}
    idx = {"core_atom": np.array([0, 2, 5]), "auth": np.array([1, 3])}
    state = {}
    for f, ix in idx.items():
        Q = rng.dirichlet(np.ones(K), size=len(ix))
        state[f] = (torch.from_numpy(ix), torch.tensor(Q), torch.tensor(rng.random(len(ix)) * 0.8 + 0.2))
    return U, w, state, idx


# (a) gradcheck
U, w, state, idx = toy_state()
scales = {f: torch.norm(u, dim=0, keepdim=True).detach() for f, u in U.items()}
fa = lambda ua, ub: m.temporal_loss_value({"core_atom": ua, "auth": ub}, scales, w, state)
results["a_gradcheck"] = bool(torch.autograd.gradcheck(fa, (U["core_atom"], U["auth"]), eps=1e-6, atol=1e-7))

# (b) gauge
base = float(m.temporal_loss_value(U, {f: torch.norm(u, dim=0, keepdim=True) for f, u in U.items()}, w, state))
c = torch.ones(3); c[1] = 7.3
U2 = {f: (u * c).detach() for f, u in U.items()}
moved = float(m.temporal_loss_value(U2, {f: torch.norm(u, dim=0, keepdim=True) for f, u in U2.items()}, w, state))
results["b_gauge_abs_change"] = abs(moved - base)

# (c) permutation of T2 columns
K1, K2 = 3, 3
P1 = rng.dirichlet(np.ones(K1) * 0.3, size=60)
true_perm = np.array([2, 0, 1])                      # T1 column a lives in T2 column true_perm[a]
P2 = np.zeros((60, K2)); P2[:, true_perm] = P1
P2 = 0.85 * P2 + 0.15 * rng.dirichlet(np.ones(K2), size=60)
acc, rec = m.temporal_match_columns(P1, [P2[:40], P2[40:]])
pi = np.array([1, 2, 0])                              # relabel T2 columns: new col j = old col pi[j]
accp, recp = m.temporal_match_columns(P1, [P2[:40][:, pi], P2[40:][:, pi]])
inv = np.argsort(pi)
results["c_match_recovers_planted"] = sorted(acc) == sorted((a, int(true_perm[a])) for a in range(K1))
results["c_match_permutes_with_columns"] = sorted(accp) == sorted((a, int(inv[b])) for a, b in acc)
Ut = {"core_atom": torch.tensor(rng.random((60, K2)) + 0.05)}
wt = {"core_atom": torch.tensor(rng.random(K2) + 0.1)}
Q, mm, keep = m.temporal_mapped_prior(P1, acc, K2)
st = {"core_atom": (torch.arange(60)[torch.from_numpy(keep)], torch.tensor(Q[keep]), torch.tensor(mm[keep]))}
Qp, mmp, keepp = m.temporal_mapped_prior(P1, accp, K2)
stp = {"core_atom": (torch.arange(60)[torch.from_numpy(keepp)], torch.tensor(Qp[keepp]), torch.tensor(mmp[keepp]))}
l0 = float(m.temporal_loss_value(Ut, {"core_atom": torch.norm(Ut["core_atom"], dim=0, keepdim=True)}, wt, st))
Utp = {"core_atom": Ut["core_atom"][:, pi]}
l1 = float(m.temporal_loss_value(Utp, {"core_atom": torch.norm(Utp["core_atom"], dim=0, keepdim=True)},
                                 {"core_atom": wt["core_atom"][pi]}, stp))
results["c_loss_abs_change_under_permutation"] = abs(l1 - l0)

# (e) unmatched T2 column: zero prior mass there
U, w, state, idx = toy_state()
for f in state:
    ix, Qf, mf = state[f]
    Qf = Qf.clone(); Qf[:, 2] = 0; Qf = Qf / Qf.sum(1, keepdim=True)
    state[f] = (ix, Qf, mf)
sc = {f: torch.norm(u, dim=0, keepdim=True) for f, u in U.items()}
L = m.temporal_loss_value(U, sc, w, state); L.backward()
results["e_finite_loss_and_grad"] = bool(torch.isfinite(L) and all(torch.isfinite(u.grad).all() for u in U.values()))

# (f) gradient only on persisting rows
U, w, state, idx = toy_state()
sc = {f: torch.norm(u, dim=0, keepdim=True) for f, u in U.items()}
m.temporal_loss_value(U, sc, w, state).backward()
zero_outside, nonzero_inside = True, True
for f, u in U.items():
    out = np.setdiff1d(np.arange(u.shape[0]), idx[f])
    zero_outside &= bool((u.grad[out] == 0).all())
    nonzero_inside &= bool((u.grad[idx[f]].abs().sum(1) > 0).all())
results["f_grad_exactly_zero_on_new_rows"] = zero_outside
results["f_grad_nonzero_on_persisting_rows"] = nonzero_inside

# (g) the coupling without the stop-gradient, and in the v9.4 concentration term
U, w, state, idx = toy_state()
X = {}
tot = 0
for f, (ix, Qf, mf) in state.items():
    Un = U[f] / torch.norm(U[f], dim=0, keepdim=True)          # NOT detached
    Xf = Un[ix] * w[f]; p = Xf / Xf.sum(1, keepdim=True)
    tot = tot + (mf * (Qf * (torch.log(Qf + 1e-12) - torch.log(p + 1e-12))).sum(1)).sum()
tot.backward()
results["g_new_row_grad_without_stop_gradient"] = float(max(
    U[f].grad[np.setdiff1d(np.arange(U[f].shape[0]), idx[f])].abs().max() for f in U))
U, w, state, idx = toy_state()
conc = 0
for f, (ix, _, _) in state.items():
    Un = U[f] / torch.norm(U[f], dim=0, keepdim=True)
    Xf = Un[ix] * w[f]; p = Xf / Xf.sum(1, keepdim=True)
    H = -(p * torch.log(p + 1e-12)).sum(1) / np.log(3)
    conc = conc + (torch.clamp(1.0 - H, min=0) ** 2).sum()   # floor 1.0: active for every row
conc.backward()
results["g_v94_conc_term_grad_on_rows_outside_its_set"] = float(max(
    U[f].grad[np.setdiff1d(np.arange(U[f].shape[0]), idx[f])].abs().max() for f in U))

# (h) false acceptance on unrelated memberships; recovery of a planted permutation
fa_pairs, n_pairs, rec_ok = 0, 0, 0
for t in range(200):
    A = rng.dirichlet(np.ones(4) * 0.4, size=120)
    B = rng.dirichlet(np.ones(4) * 0.4, size=120)
    acc_t, _ = m.temporal_match_columns(A, [B[:80], B[80:]], seed=t)
    fa_pairs += len(acc_t); n_pairs += 4
    perm = rng.permutation(4)
    Bp = np.zeros_like(A); Bp[:, perm] = A
    Bp = 0.7 * Bp + 0.3 * rng.dirichlet(np.ones(4), size=120)
    acc_r, _ = m.temporal_match_columns(A, [Bp[:80], Bp[80:]], seed=t)
    rec_ok += sorted(acc_r) == sorted((a, int(perm[a])) for a in range(4))
results["h_false_acceptance_rate_unrelated"] = fa_pairs / n_pairs
results["h_planted_permutation_recovered_share"] = rec_ok / 200

# (d) empty persisting set on a real fit (T2, C2, K=4, 400 epochs so the warm-up is crossed)
torch.set_default_dtype(torch.float32)
raw2 = m.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
soc, sem, anc = m.get_active_facets("C2")
empty = {"facets": {"core_atom": {"idx": np.array([], dtype=np.int64), "P": np.zeros((0, 4))}},
         "K1": 4, "source": {"test": "empty"}}
runs = []
for prior in (None, empty):
    m.set_seeds(42)
    _, _, d = m.run_inner_solver(raw_data=raw2, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=4,
                                 dimensions=raw2["dimensions"],
                                 params={"lambda_z_offdiag": 0.12, "lambda_temporal": 1.0},
                                 device=m.DEVICE, seed_function=m.set_seeds, inner_epochs=400,
                                 temporal_prior=prior)
    runs.append(d["loss_history"])
results["d_empty_set_identical_to_no_prior"] = runs[0] == runs[1]

ok = (results["a_gradcheck"] and results["b_gauge_abs_change"] < 1e-12
      and results["c_match_recovers_planted"] and results["c_match_permutes_with_columns"]
      and results["c_loss_abs_change_under_permutation"] < 1e-12
      and results["d_empty_set_identical_to_no_prior"] and results["e_finite_loss_and_grad"]
      and results["f_grad_exactly_zero_on_new_rows"] and results["f_grad_nonzero_on_persisting_rows"])
results["T1_pass_a_to_f"] = bool(ok)
print(json.dumps(results, indent=2))
json.dump(results, open("t1_unit_checks.json", "w"), indent=2)
sys.exit(0 if ok else 1)
