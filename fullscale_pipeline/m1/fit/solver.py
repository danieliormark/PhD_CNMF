# F5 — the fit loop of the M1 model (ticket 107). Skeleton from chunk13v10.run_inner_solver (v9.4): Adam (lr 0.01),
# clamping after every step (ticket 74), stopping test "relative change of the total loss over 20 epochs < 1e-4",
# two-part when a term is on (no stop until 20 epochs after the warm-up; the data loss must also have stopped changing;
# ticket 102), epoch ceiling. Determinism: one thread, deterministic algorithms, fixed facet order, the fit's own seed
# (CLAUDE.md §4.9, §4.16, tickets 76, 85). Logging every LOG_EVERY epochs: per-relation data loss, term values, and the
# change of every facet's weighted memberships since the last log (mean Jensen–Shannon distance, share of entities whose
# largest membership moved) — the membership-stability record used to re-examine the 300-epoch warm-up (2026-11-06).
import time
import numpy as np
import torch

from .loss import RelationTensors, total_loss
from .model import Model
from .init import initialise, random_start
from . import terms as T

LR, MAX_EPOCHS, WINDOW, TOL, LOG_EVERY, WARMUP = 0.01, 2000, 20, 1e-4, 50, 300


def set_determinism(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def _membership_snapshot(Un, Zs, relations):
    with torch.no_grad():
        return {f: T.weighted_membership(Un, Zs, relations, f).double() for f in sorted(Un)}


def _snapshot_change(prev, cur):
    out = {}
    for f in cur:
        P, Q = prev[f], cur[f]
        M = 0.5 * (P + Q)
        kl = lambda A, B: (A * (torch.log(A.clamp(min=1e-300)) - torch.log(B.clamp(min=1e-300)))).sum(1)
        js = torch.sqrt(torch.clamp(0.5 * kl(P, M) + 0.5 * kl(Q, M), min=0.0) / T.LOG2)
        out[f] = {"mean_js": float(js.mean()), "argmax_moved": float((P.argmax(1) != Q.argmax(1)).double().mean())}
    return out


def fit(rds, K, beta, seed, terms=None, max_epochs=MAX_EPOCHS, lr=LR, warmup=WARMUP, dtype=torch.float64,
        init_U=None, init_Z=None, log_every=LOG_EVERY, tol=TOL, window=WINDOW, start="nndsvd", lr_mode="absolute"):
    """One fit. rds: data.RelationData list; terms: list of (term object, λ) — λ = 0 terms are not built in;
    init_U/init_Z: start values (default: F3 initialisation from `seed`). Returns a result dict."""
    set_determinism(seed)
    t0 = time.time()
    rts = [RelationTensors(rd, dtype=dtype) for rd in rds]
    relations = {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}
    sizes = {}
    for rd in rds:
        sizes[rd.row_facet] = rd.shape[0]; sizes[rd.col_facet] = rd.shape[1]
    start_label = start if init_U is None else "given"
    if init_U is None:
        if start == "nndsvd":
            init_U, init_Z = initialise(rds, K, seed)
        elif start.startswith("nndsvd_sub"):            # e.g. "nndsvd_sub0.8"
            init_U, init_Z = initialise(rds, K, seed, subsample=float(start[len("nndsvd_sub"):]))
        else:
            init_U, init_Z = random_start(rds, K, seed)
    model = Model(sizes, relations, K, init_U=init_U, init_Z=init_Z, dtype=dtype)
    if lr_mode == "absolute":                       # toy: one learning rate for every parameter
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:                                           # "relative": each block's rate × the RMS of its starting entries
        groups = [{"params": [p], "lr": lr * float(p.detach().pow(2).mean().sqrt())} for p in model.parameters()]
        opt = torch.optim.Adam(groups, lr=lr)
    active = [(t, lam) for t, lam in (terms or []) if lam > 0]
    terms_on = bool(active)
    hist_total, hist_data, logs = [], [], []
    converged, frozen = False, False
    prev_snap = None
    epoch = -1
    for epoch in range(max_epochs):
        opt.zero_grad()
        data, parts = total_loss(dict(model.U), dict(model.Z), rts, beta)
        total = data
        term_vals = {}
        if terms_on and epoch >= warmup:
            _, Un, Zs = model.normalised()
            if not frozen:
                with torch.no_grad():
                    Und = {f: u.detach() for f, u in Un.items()}; Zsd = {r: z.detach() for r, z in Zs.items()}
                    for t, _ in active:
                        if t.needs_freeze:
                            t.freeze(Und, Zsd, relations)
                frozen = True
            for t, lam in active:
                v = t.value(Un, Zs, relations)
                term_vals[t.name] = v
                total = total + lam * v
        total.backward()
        opt.step()
        model.clamp_()
        hist_total.append(float(total.item())); hist_data.append(float(data.item()))
        if log_every and (epoch % log_every == 0):
            with torch.no_grad():
                _, Un, Zs = model.normalised()
                snap = _membership_snapshot(Un, Zs, relations)
            logs.append({"epoch": epoch, "data_loss": hist_data[-1], "total_loss": hist_total[-1],
                         "relations": {k: float(v.item()) for k, v in parts.items()},
                         "terms": {k: float(v.item()) for k, v in term_vals.items()},
                         "membership_change": _snapshot_change(prev_snap, snap) if prev_snap is not None else None,
                         "seconds": round(time.time() - t0, 2)})
            prev_snap = snap
        if epoch >= window and not (terms_on and epoch < warmup + window + 1):
            p = hist_total[-window - 1]
            rel = abs(hist_total[-1] - p) / abs(p) if p != 0 else np.inf
            if terms_on:
                pd = hist_data[-window - 1]
                rel = max(rel, abs(hist_data[-1] - pd) / abs(pd) if pd != 0 else np.inf)
            if rel < tol:
                converged = True
                break
    with torch.no_grad():
        _, Un, Zs = model.normalised()
        final_parts = {k: float(v.item()) for k, v in total_loss(dict(model.U), dict(model.Z), rts, beta)[1].items()}
    return {"U": {f: model.U[f].detach().numpy().copy() for f in model.facets},
            "Z": {r: model.Z[r].detach().numpy().copy() for r in sorted(relations)},
            "relations": relations, "K": K, "beta": beta, "seed": seed, "converged": converged, "epochs": epoch + 1,
            "loss_history": hist_total, "data_loss_history": hist_data, "final_relation_loss": final_parts,
            "logs": logs, "terms": [{"name": t.name, "lambda": lam, "info": getattr(t, "info", None)} for t, lam in active],
            "settings": {"start": start_label, "lr": lr, "lr_mode": lr_mode, "max_epochs": max_epochs, "warmup": warmup, "tol": tol, "window": window,
                         "dtype": str(dtype)}, "seconds": round(time.time() - t0, 2)}
