# WHAT: Shared building blocks for diagnostic scripts -- module import, data loading, the
#       three init modes, two fit paths (real solver / instrumented), and the fit cache.
# USE:  from diagnostic_blocks import fit_or_load, evaluate  -- start any new diagnostic here.

"""
diagnostic_blocks.py -- shared building blocks for diagnostic/scratch scripts.

WHY THIS EXISTS
---------------
Every diagnostic script so far (permutation_test.py, uscales_determinacy.py,
threeway_mass_check.py, step1_domain_balance.py, ...) re-typed the same ~150
lines: sparse-matrix loading, the three init modes, and a hand-copied
reimplementation of run_inner_solver's training loop. Copies drift. Once two
copies disagree, results computed under them are no longer comparable -- and
cross-session comparability outranks everything else here.

One definition, imported. Not a template to copy.

    import sys
    sys.path.insert(0, "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution")
    from diagnostic_blocks import load_module, fit_or_load, BLOCKS_VERSION

TWO FIT PATHS -- pick deliberately
----------------------------------
fit_production()   delegates to the REAL chunk13v9.run_inner_solver. No drift
                   possible, because there is no second copy of the loop. Use
                   this unless you specifically need to reach inside training.

fit_instrumented() a reimplementation of the same loop that accepts per-epoch
                   hooks (needed by e.g. the gauge-rescaling test, which must
                   mutate tensors mid-training). SESSION_PROTOCOL §E requires
                   any such reimplementation be checked against the real
                   solver; that check is discharged once here rather than
                   re-owed by every script -- see AGREEMENT CHECK below.

AGREEMENT CHECK (SESSION_PROTOCOL §E)
-------------------------------------
Run `python diagnostic_blocks.py --verify` to re-run it. Result recorded in
diagnostic_results/blocks_agreement_check.json. Re-run after ANY edit to
fit_instrumented or to run_inner_solver.

RETURN CONVENTION
-----------------
Both fit paths return (U_final, Z_final, diagnostics) with all arrays as numpy
(chunk13v9's run_inner_solver returns torch tensors for U_final/Z_final;
diagnostic_blocks converts). This is deliberate: numpy round-trips through .npz for
the fit cache, torch does not.

VERSIONING
----------
BLOCKS_VERSION is stamped into every results file via
diagnostic_run_metadata.build_run_metadata(). Bump it on any change that could
alter numbers. The fit cache additionally keys on a content hash of
chunk13v9.py, so editing the solver invalidates cached fits automatically.

PERMUTATION / MASS HELPERS (added for ticket 79/80/82 work, FINDINGS §16)
---------------------------------------------------------------------
hungarian_diagonal_match(Z, K)   the Hungarian-matching + structure_score
                                 logic, previously copy-pasted across
                                 permutation_test.py and
                                 permutation_consistency_sweep.py. One
                                 definition now.

relation_community_mass(Z, K, threshold=1.0)
                                 the FINDINGS §16 confidence-gated mass
                                 reading: corrected diagonal if this
                                 relation's own structure_score clears
                                 `threshold` (real mislabelling, safe to
                                 correct), raw diagonal otherwise (genuine
                                 mixed coupling, per user instruction: do
                                 not force a diagonal reading onto it).

Does NOT modify chunk13v9.py.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

BLOCKS_VERSION = "1.15.0"

BASE_DIR = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
MODULE_PATH = os.path.join(BASE_DIR, "chunk13v9.py")
OUT_DIR = os.path.join(BASE_DIR, "diagnostic_results")
CACHE_DIR = os.path.join(OUT_DIR, "tensors")

SLICE_PATHS = {
    "T1": "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1.pkl",
    "T2": "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t2.pkl",
    # Ticket 84 Phase 4: chunk12v2.py's output (M_Atom_Child/M_Fringe_Cousin/M_Cousin_Child
    # log-damped, everything else identical -- verified byte-for-byte against the T1/T2
    # entries above via null-rebuild, see plans/ticket84-grammar-hub-downweighting.md).
    # Additive only -- NOT the default; nothing switches to this without slice_name="T1_v2"/
    # "T2_v2" explicitly.
    "T1_v2": "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1_v2.pkl",
    "T2_v2": "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t2_v2.pkl",
}

_MODULE = None
_DATA_CACHE = {}


# =============================================================================
# MODULE / DATA LOADING
# =============================================================================

def load_module(force_reload=False):
    """
    Import chunk13v9.py by path, once per process. Returns the module object.
    Every script needs this; nobody should re-type the importlib dance.
    """
    global _MODULE
    if _MODULE is not None and not force_reload:
        return _MODULE
    spec = importlib.util.spec_from_file_location("chunk13v9_blocks", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chunk13v9_blocks"] = mod
    spec.loader.exec_module(mod)
    _MODULE = mod
    return mod


def load_slice(slice_name="T1"):
    """
    Returns (raw_data, dimensions) for a named slice, memoized per process.
    Loading is not free and every script did it identically.
    """
    if slice_name in _DATA_CACHE:
        return _DATA_CACHE[slice_name]
    mod = load_module()
    raw_data = mod.load_and_validate_data(SLICE_PATHS[slice_name])
    dimensions = raw_data["dimensions"]
    _DATA_CACHE[slice_name] = (raw_data, dimensions)
    return raw_data, dimensions


def chunk13v9_hash():
    """Content hash of the solver source -- the fit cache keys on this."""
    with open(MODULE_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# =============================================================================
# SHARED PIECES (used by both fit paths)
# =============================================================================

def build_active_matrices(raw_data, soc_keys, sem_keys, device=None):
    """
    Frobenius-normalized sparse matrices for the active relations, plus the set
    of facets they touch and the set of zero-norm ("empty") relations.

    Mirrors run_inner_solver's own preamble exactly (ticket 61's empty-relation
    handling included). Kept silent -- the real solver prints a warning per
    absent/empty relation; a diagnostic sweeping 12 cells does not want that.
    """
    mod = load_module()
    device = device if device is not None else mod.DEVICE

    active_matrices, active_facets, empty_relations = {}, set(), set()
    for key in (soc_keys + sem_keys):
        if key not in raw_data:
            continue
        mat = mod.scipy_to_torch_sparse(raw_data[key]).to(device).coalesce()
        fro_norm = torch.norm(mat.values(), p=2)
        normalized_values = mat.values() / fro_norm if fro_norm > 0 else mat.values()
        if fro_norm <= 0:
            empty_relations.add(key)
        active_matrices[key] = torch.sparse_coo_tensor(
            mat.indices(), normalized_values, mat.size(), device=device)
        f1, f2 = mod.RELATION_MAP[key]
        active_facets.update([f1, f2])
    return active_matrices, active_facets, empty_relations


def get_init(mode, active_matrices, active_facets, anchor_keys, dimensions, K, device=None):
    """
    The three initialization modes used across the diagnostic history:

      "production"      NNDSVD + multi-hop propagation, i.e. what the pipeline
                        actually does.
      "random_anchor_z" production U, but anchor Z cores replaced with uniform
                        random -- isolates how much structure the anchor SVD
                        contributes vs. what the fit recovers on its own.
      "fully_random"    both U and Z uniform random -- the null init.

    Returns (U_raw, Z_raw) as positive-valued leaf tensors (ticket 74: direct
    positivity, no softplus -- see CLAUDE.md §4.8).
    """
    mod = load_module()
    device = device if device is not None else mod.DEVICE

    if mode == "production":
        return mod.initialize_tucker_adapted_nndsvd_and_propagate(
            active_matrices, anchor_keys, dimensions, active_facets, K, device)

    if mode == "random_anchor_z":
        U_raw, Z_raw = mod.initialize_tucker_adapted_nndsvd_and_propagate(
            active_matrices, anchor_keys, dimensions, active_facets, K, device)
        for key in [k for k in anchor_keys if k in active_matrices]:
            rand_z = (np.random.rand(K, K).astype(np.float32) + 1e-4)
            Z_raw[key] = torch.tensor(rand_z, device=device, dtype=torch.float32, requires_grad=True)
        return U_raw, Z_raw

    if mode == "fully_random":
        U_raw = {f: torch.tensor((np.random.rand(dimensions[f], K).astype(np.float32) + 1e-4),
                                 device=device, dtype=torch.float32, requires_grad=True)
                 for f in active_facets}
        Z_raw = {rel: torch.tensor((np.random.rand(K, K).astype(np.float32) + 1e-4),
                                   device=device, dtype=torch.float32, requires_grad=True)
                 for rel in active_matrices.keys()}
        return U_raw, Z_raw

    raise ValueError(f"unknown init mode: {mode!r}")


def _to_numpy(d):
    return {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in d.items()}


# =============================================================================
# FIT PATH 1 -- the real solver (default; no drift risk)
# =============================================================================

def fit_production(config_id, K, lambda_l1, lambda_z_offdiag, slice_name="T1",
                   seed=None, inner_epochs=None, learning_rate=None, device=None):
    """
    Delegates to chunk13v9.run_inner_solver. Use this for anything that just
    needs a converged fit.

    lambda_l1 and lambda_z_offdiag are REQUIRED positional-ish arguments with no
    defaults, per SESSION_PROTOCOL §B: both are under active research and a
    default here would quietly become the de facto setting for decisions nobody
    made deliberately.

    Returns (U_final, Z_final, diagnostics), arrays as numpy. diagnostics keeps
    chunk13v9's keys (math_loss, internal_soc_loss, raw_sparsity_loss,
    loss_history, U_scales, converged) and gains "epochs_run".
    """
    mod = load_module()
    device = device if device is not None else mod.DEVICE
    seed = mod.MASTER_SEED if seed is None else seed
    raw_data, dimensions = load_slice(slice_name)
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)

    kwargs = {}
    if inner_epochs is not None:
        kwargs["inner_epochs"] = inner_epochs
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate

    mod.set_seeds(seed)
    U_final, Z_final, diagnostics = mod.run_inner_solver(
        raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys, anchor_keys=anchor_keys,
        K=K, dimensions=dimensions,
        params={"lambda_l1": lambda_l1, "lambda_z_offdiag": lambda_z_offdiag},
        device=device, seed_function=lambda: mod.set_seeds(seed), **kwargs)

    diagnostics["epochs_run"] = len(diagnostics["loss_history"])
    return _to_numpy(U_final), _to_numpy(Z_final), diagnostics


# =============================================================================
# FIT PATH 2 -- instrumented reimplementation (only when you must reach inside)
# =============================================================================

def fit_instrumented(config_id, K, lambda_l1, lambda_z_offdiag, slice_name="T1",
                     seed=None, init_mode="production", inner_epochs=None,
                     learning_rate=None, device=None,
                     post_init_hook=None, epoch_hook=None, extra_loss_fn=None,
                     soc_keys_override=None, relation_weight_multiplier=None):
    """
    Line-for-line reimplementation of run_inner_solver's loop, exposing:

      init_mode       -- the real solver is always "production"; the other two
                         modes exist only here.
      post_init_hook  -- fn(U_raw, Z_raw) called after init + first clamp.
      epoch_hook      -- fn(epoch, state_dict) called after each optimizer.step();
                         state_dict carries U_raw/Z_raw/U_norm/U_scales/Z_scaled/
                         losses. Return True to stop early.
      extra_loss_fn   -- fn(Z_scaled, U_scales, U_norm, epoch) -> torch scalar
                         (already weighted by the caller), ADDED to total_loss
                         before .backward(). Default None -- total_loss is then
                         bit-for-bit what it was before this parameter existed,
                         preserving the verified equivalence to fit_production.
                         For testing whether a CANDIDATE differentiable penalty
                         is exploitable (e.g. via the same U_scales-shrinking
                         route documented in FINDINGS §2 for lambda_z_offdiag,
                         and CONFIRMED for a naive Z_scaled-based domain-balance
                         candidate -- see make_domain_balance_penalty_torch's
                         docstring and domain_penalty_exploitability_grid.json)
                         before deciding whether it belongs in chunk13v9.py at
                         all -- experimental only, not a proposal to add this to
                         the real solver as-is. BLOCKS_VERSION 1.14.0 added the
                         U_norm argument (was Z_scaled, U_scales, epoch only) so
                         a candidate penalty can be built on U_prob/membership
                         space instead of Z_scaled/mass space -- ticket 82's E2
                         settled design needs exactly this (see
                         make_inloop_domain_balance_penalty_torch below).

      soc_keys_override -- Stage 0c (tickets 86/87): if given, REPLACES the
                         soc_keys get_active_facets(config_id) would otherwise
                         return, before build_active_matrices is called -- a
                         relation dropped this way is genuinely absent from
                         the fit (never enters active_matrices/active_facets/
                         the loss), not evaluated post-hoc and excluded. Line-
                         for-line equivalent to ghost_no_affil_ablation.py's
                         hand-copied loop (its only change was this same
                         filter); added here so that ablation and any future
                         one like it go through the one verified training
                         loop instead of a new hand copy. None (default) ->
                         behavior identical to before this parameter existed.
      relation_weight_multiplier -- Stage 0c: optional {relation_key: float},
                         multiplies alpha_weight (w_soc or w_sem, whichever
                         applies) for that relation specifically. Exists for
                         Stage 0c's arm (iii) control -- isolating "does
                         up-weighting S_Art_Auth alone reproduce arm (ii)'s
                         effect" from "does removing affil help" -- a
                         multiplier is not achievable via soc_keys_override
                         alone since it doesn't remove a relation, only
                         reweights it. {} (default) -> behavior identical to
                         before this parameter existed.

    Verified equivalent to fit_production at init_mode="production", extra_loss_fn=None,
    soc_keys_override=None, relation_weight_multiplier=None
    -- see verify_agreement() and diagnostic_results/blocks_agreement_check.json.
    If you edit this function, re-run the check and bump BLOCKS_VERSION.
    """
    mod = load_module()
    device = device if device is not None else mod.DEVICE
    seed = mod.MASTER_SEED if seed is None else seed
    inner_epochs = mod.INNER_EPOCHS if inner_epochs is None else inner_epochs
    learning_rate = mod.LEARNING_RATE if learning_rate is None else learning_rate
    relation_weight_multiplier = relation_weight_multiplier or {}

    raw_data, dimensions = load_slice(slice_name)
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    if soc_keys_override is not None:
        soc_keys = soc_keys_override

    mod.set_seeds(seed)
    active_matrices, active_facets, empty_relations = build_active_matrices(
        raw_data, soc_keys, sem_keys, device)

    w_soc = 0.5 / max(len([k for k in soc_keys if k in active_matrices]), 1)
    w_sem = 0.5 / max(len([k for k in sem_keys if k in active_matrices]), 1)

    U_raw, Z_raw = get_init(init_mode, active_matrices, active_facets,
                            anchor_keys, dimensions, K, device)

    with torch.no_grad():
        for t in list(U_raw.values()) + list(Z_raw.values()):
            t.clamp_(min=1e-7)

    if post_init_hook is not None:
        post_init_hook(U_raw, Z_raw)

    optimizer = torch.optim.Adam(list(U_raw.values()) + list(Z_raw.values()), lr=learning_rate)
    eye_K = torch.eye(K, device=device)
    loss_history = []
    converged = False
    pure_recon_loss_val = 0.0
    sparsity_loss = 0.0
    U_norm, Z_scaled, U_scales = {}, {}, {}

    for epoch in range(inner_epochs):
        optimizer.zero_grad()
        U_pos, Z_pos = U_raw, Z_raw  # ticket 74: direct positivity, no softplus

        U_norm, U_scales = {}, {}
        for f, u_mat in U_pos.items():
            col_norms = torch.norm(u_mat, p=2, dim=0, keepdim=True)
            U_scales[f] = torch.clamp(col_norms, min=1e-9)
            U_norm[f] = u_mat / U_scales[f]

        Z_scaled = {}
        for rel_key, z_core in Z_pos.items():
            f1, f2 = mod.RELATION_MAP[rel_key]
            Z_scaled[rel_key] = z_core * (U_scales[f1].t() @ U_scales[f2])

        recon_loss, sparsity_loss, z_offdiag_loss = 0.0, 0.0, 0.0

        for rel_key, M_sparse in active_matrices.items():
            f1, f2 = mod.RELATION_MAP[rel_key]
            u1, u2, z_core = U_norm[f1], U_norm[f2], Z_scaled[rel_key]
            alpha_weight = (w_soc if rel_key in soc_keys else w_sem) * relation_weight_multiplier.get(rel_key, 1.0)
            M_v = torch.sparse.mm(M_sparse, u2)
            trace_cross = torch.sum(M_v * torch.matmul(u1, z_core))
            c_u = torch.matmul(u1.t(), u1)
            c_v = torch.matmul(u2.t(), u2)
            trace_pred = torch.sum((c_u @ z_core) * (z_core @ c_v))
            target_norm_sq = 0.0 if rel_key in empty_relations else 1.0  # ticket 61
            recon_loss += alpha_weight * torch.clamp(
                target_norm_sq - 2.0 * trace_cross + trace_pred, min=0.0)
            z_offdiag_loss += torch.sum((z_core * (1.0 - eye_K)) ** 2)

        num_sparsity_entries = 0  # ticket 73: MEAN, not raw sum
        for u_mat in U_norm.values():
            sparsity_loss += torch.sum(u_mat)
            num_sparsity_entries += u_mat.numel()
        sparsity_loss = sparsity_loss / num_sparsity_entries

        pure_recon_loss_val = recon_loss.item() if hasattr(recon_loss, "item") else recon_loss
        total_loss = recon_loss + (lambda_l1 * sparsity_loss) + (lambda_z_offdiag * z_offdiag_loss)
        extra_loss_val = 0.0
        if extra_loss_fn is not None:
            extra_term = extra_loss_fn(Z_scaled, U_scales, U_norm, epoch)
            total_loss = total_loss + extra_term
            extra_loss_val = extra_term.item() if hasattr(extra_term, "item") else extra_term
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for t in list(U_raw.values()) + list(Z_raw.values()):
                t.clamp_(min=1e-7)

        loss_history.append(total_loss.item())

        if epoch_hook is not None:
            stop = epoch_hook(epoch, {
                "U_raw": U_raw, "Z_raw": Z_raw, "U_norm": U_norm,
                "U_scales": U_scales, "Z_scaled": Z_scaled,
                "recon_loss": recon_loss, "sparsity_loss": sparsity_loss,
                "z_offdiag_loss": z_offdiag_loss, "total_loss": total_loss,
                "extra_loss_val": extra_loss_val,
            })
            if stop:
                break

        if epoch >= 20:  # relative early stop, identical to the real solver
            prev_loss = loss_history[-21]
            if prev_loss > 0 and abs(loss_history[-1] - prev_loss) / prev_loss < 1e-4:
                converged = True
                break

    U_final = {f: U_norm[f].detach().cpu().numpy() for f in active_facets}
    Z_final = {rel: Z_scaled[rel].detach().cpu().numpy() for rel in Z_scaled}
    U_scales_out = {f: s.squeeze().detach().cpu().numpy() for f, s in U_scales.items()}

    diagnostics = {
        "math_loss": pure_recon_loss_val,
        "internal_soc_loss": float((lambda_l1 * sparsity_loss) + (lambda_z_offdiag * z_offdiag_loss)),
        "raw_sparsity_loss": float(sparsity_loss.item() if hasattr(sparsity_loss, "item") else sparsity_loss),
        "loss_history": loss_history,
        "U_scales": U_scales_out,
        "converged": converged,
        "epochs_run": len(loss_history),
        "init_mode": init_mode,
    }
    return U_final, Z_final, diagnostics


# =============================================================================
# FIT CACHE -- SESSION_PROTOCOL §D, made automatic
# =============================================================================

_RAW_DATA_HASH_CACHE = {}


def _raw_data_hash(slice_name):
    """
    Content hash of the actual matrices `load_slice(slice_name)` returns --
    NOT the pickle file's bytes (pickle is not guaranteed byte-stable across
    versions/machines) and NOT the file path (a path doesn't change when the
    file it points to is regenerated with different content, e.g. after a
    ticket-84-style chunk12 edit). Hashes each relation's canonical CSR
    (indices, indptr, data), sorted by relation key for determinism.

    Ticket 84 (D... Phase 2): added so a cached fit is tied to the input data
    it was actually fit on, not just the solver code (chunk13v9_hash()) and
    the hyperparameters. Before this, regenerating chunk12's output with
    different relation weights would silently reuse 379 pre-existing cached
    fits (52.6MB) against the new data, printing "[cache hit]" as if nothing
    had changed. Memoized per slice_name (small, cheap; load_slice() itself
    is already memoized, so this adds one hash pass over ~10 sparse matrices).
    """
    if slice_name in _RAW_DATA_HASH_CACHE:
        return _RAW_DATA_HASH_CACHE[slice_name]
    raw_data, _ = load_slice(slice_name)
    parts = []
    for key in sorted(raw_data.keys()):
        val = raw_data[key]
        if not sp.issparse(val):
            continue  # e.g. a stray 'dimensions' entry -- never present per §11, but don't assume
        m = val.tocsr()
        parts.append(key.encode())
        parts.append(m.indices.astype(np.int64).tobytes())
        parts.append(m.indptr.astype(np.int64).tobytes())
        parts.append(m.data.astype(np.float64).tobytes())
    digest = hashlib.sha256(b"".join(parts)).hexdigest()[:10]
    _RAW_DATA_HASH_CACHE[slice_name] = digest
    return digest


def _cache_key(config_id, K, lambda_l1, lambda_z_offdiag, slice_name, seed,
               init_mode, path, inner_epochs, learning_rate):
    """
    Identity of a fit = everything that can change its numbers, INCLUDING the
    solver source hash AND the input data's own content hash. Edit
    chunk13v9.py, or regenerate the input pickles with different content
    (same file path, different bytes), and every cached fit affected is
    invalidated automatically rather than silently reused.

    NAMESPACED, NOT INVALIDATING (ticket 84, Phase 2): old cache files under
    the pre-existing naming scheme are left on disk untouched -- they are not
    deleted or migrated. A fit against unchanged data still round-trips to
    the SAME digest as before this change (raw_data_hash is now part of the
    payload, but for unchanged data it's just an additional constant term),
    so this is additive, not a mass cache invalidation.
    """
    data_hash = _raw_data_hash(slice_name)
    payload = json.dumps({
        "config_id": config_id, "K": K, "lambda_l1": lambda_l1,
        "lambda_z_offdiag": lambda_z_offdiag, "slice": slice_name, "seed": seed,
        "init_mode": init_mode, "path": path, "inner_epochs": inner_epochs,
        "learning_rate": learning_rate, "blocks": BLOCKS_VERSION,
        "chunk13v9": chunk13v9_hash(), "raw_data_hash": data_hash,
    }, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{config_id}_K{K}_{slice_name}_{path}_data{data_hash}_{digest}", payload


def fit_or_load(config_id, K, lambda_l1, lambda_z_offdiag, slice_name="T1",
                seed=None, path="production", init_mode="production",
                inner_epochs=None, learning_rate=None, device=None,
                force_refit=False, verbose=True):
    """
    Fit once, reuse forever. Cache hit -> load .npz; miss -> fit, save, return.

    path="production"    -> fit_production (the real solver)
    path="instrumented"  -> fit_instrumented (needed for init_mode != production)

    Saves U_final, Z_final, U_scales, and diagnostics UNCONDITIONALLY
    (SESSION_PROTOCOL §D) -- Run3 and Run11 were once discarded as unlikely to
    matter and became necessary two prompts later; uscales_determinacy.py had to
    re-fit three runs because an earlier save omitted U_scales. Not a judgement
    call any more.

    Returns (U_final, Z_final, diagnostics) with diagnostics["_cache"] set to
    "hit" or "miss".
    """
    if path == "instrumented" and init_mode != "production":
        pass  # non-production inits are only reachable through the instrumented path
    elif path == "production" and init_mode != "production":
        raise ValueError("init_mode != 'production' requires path='instrumented'")

    mod = load_module()
    seed = mod.MASTER_SEED if seed is None else seed
    name, payload = _cache_key(config_id, K, lambda_l1, lambda_z_offdiag, slice_name,
                               seed, init_mode, path, inner_epochs, learning_rate)
    npz_path = os.path.join(CACHE_DIR, f"{name}.npz")

    if os.path.exists(npz_path) and not force_refit:
        npz = np.load(npz_path, allow_pickle=True)
        U_final = {k[3:]: npz[k] for k in npz.files if k.startswith("U__")}
        Z_final = {k[3:]: npz[k] for k in npz.files if k.startswith("Z__")}
        diagnostics = json.loads(str(npz["__diagnostics__"]))
        diagnostics["U_scales"] = {k[8:]: npz[k] for k in npz.files if k.startswith("Uscale__")}
        diagnostics["_cache"] = "hit"
        if verbose:
            print(f"  [cache hit ] {config_id} K={K} {path} -> {os.path.basename(npz_path)}")
        return U_final, Z_final, diagnostics

    if verbose:
        print(f"  [cache miss] {config_id} K={K} {path} -- fitting...")

    fit = fit_production if path == "production" else fit_instrumented
    kwargs = dict(slice_name=slice_name, seed=seed, inner_epochs=inner_epochs,
                  learning_rate=learning_rate, device=device)
    if path == "instrumented":
        kwargs["init_mode"] = init_mode
    U_final, Z_final, diagnostics = fit(config_id, K, lambda_l1, lambda_z_offdiag, **kwargs)

    os.makedirs(CACHE_DIR, exist_ok=True)
    serializable = {k: v for k, v in diagnostics.items() if k != "U_scales"}
    # Write to a per-process temp path, THEN atomically rename onto npz_path
    # (2026-09-01, found live during the Stage 0e SLURM array run: 16
    # concurrent array tasks share this one cache directory, and
    # np.savez(npz_path, ...) previously wrote directly to the final path --
    # not atomic. Two tasks racing on the same cache key (same config/K/
    # lambdas/seed) both passed the cache-miss check, both fit, and both
    # wrote to the SAME file concurrently, interleaving their bytes into a
    # corrupted zip archive ("Bad magic number for central directory") that
    # then broke every later reader, including tasks that had nothing to do
    # with the race. 4 cache entries corrupted, 6/16 array tasks crashed.
    # os.replace() is atomic on the same filesystem (POSIX rename semantics)
    # -- if two processes still race, the slower one's rename simply
    # overwrites the faster one's completed, valid file: redundant compute,
    # never corruption. The temp filename must itself end in ".npz", not
    # just the final one -- np.savez silently appends ".npz" to any string
    # path that doesn't already end with it, which would otherwise turn a
    # ".tmp12345" suffix into ".tmp12345.npz" and defeat the rename.
    tmp_path = os.path.join(CACHE_DIR, f"{name}.tmp{os.getpid()}.npz")
    np.savez(tmp_path,
             **{f"U__{f}": U_final[f] for f in U_final},
             **{f"Z__{r}": Z_final[r] for r in Z_final},
             **{f"Uscale__{f}": diagnostics["U_scales"][f] for f in diagnostics["U_scales"]},
             __diagnostics__=json.dumps(serializable),
             __cache_key__=payload)
    os.replace(tmp_path, npz_path)
    if verbose:
        print(f"    epochs={diagnostics['epochs_run']} converged={diagnostics['converged']} "
              f"math_loss={diagnostics['math_loss']:.6f} -> saved {os.path.basename(npz_path)}")
        if not diagnostics["converged"]:
            print(f"    [!] HIT CEILING -- not converged (SESSION_PROTOCOL §C.4)")

    diagnostics["_cache"] = "miss"
    return U_final, Z_final, diagnostics


# =============================================================================
# EVALUATION passthrough
# =============================================================================

def evaluate(config_id, U_final, Z_final, U_scales, slice_name="T1"):
    """
    Thin wrapper on evaluate_complete_solution -- the single source of truth for
    the sociological penalty (CLAUDE.md §4.13, ticket 69). Never reimplement its
    sequence in a scratch script.
    """
    mod = load_module()
    raw_data, _ = load_slice(slice_name)
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    return mod.evaluate_complete_solution(
        U_final=U_final, Z_final=Z_final, U_scales_out=U_scales,
        raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys, anchor_keys=anchor_keys,
        max_monopoly=mod.MAX_MONOPOLY, entropy_threshold=mod.ENTROPY_THRESHOLD,
        target_coherence=mod.TARGET_COHERENCE)


def presence_masks(config_id, slice_name="T1"):
    """Live-entity masks per facet (ticket 60). SESSION_PROTOCOL §C.5."""
    mod = load_module()
    raw_data, _ = load_slice(slice_name)
    soc_keys, sem_keys, _ = mod.get_active_facets(config_id)
    return mod.build_presence_masks(raw_data, soc_keys, sem_keys)


# =============================================================================
# DOMAIN CLASSIFICATION (ticket 82, this session's terminology correction)
# =============================================================================
#
# NOT read from chunk13v9.py -- no per-facet domain constant exists there, only the
# relation-level soc_keys/sem_keys convention (used for recon_loss's alpha weighting,
# not a per-facet domain claim). This mapping is this session's explicit statement of
# the ORIGINAL design intent, confirmed directly by the user: `art` is a facet
# belonging to the SOCIAL domain (articles are the social/institutional artifact;
# semantic facets describe their CONTENT). No third "hub/neutral" category -- Occam's
# razor, don't invent an entity the design doesn't have.
#
# A relational matrix X_{f1,f2} (RELATION_MAP's f1, f2) is:
#   same-domain   both f1, f2 in the same domain (e.g. auth-affil, both social;
#                 core_atom-core_child_he, both semantic)
#   cross-domain  f1, f2 in different domains (only ever art + a semantic facet,
#                 given art is the only facet touched from both sides)
#
# Sanity check (asserted in domain_community_volumes below, every call): under this
# facet-level rule, "cross-domain" and chunk13v9's own anchor_keys are the SAME SET in
# every config -- confirmed for C1-C6. Not a coincidence worth a separate name; the
# code's "anchor" (an SVD-initialization-mechanism property) and "cross-domain matrix"
# (a domain property) happen to coincide here because the only matrices touching two
# different domains are exactly the three that also get anchor treatment.

FACET_DOMAINS = {
    "art": "soc", "auth": "soc", "affil": "soc", "journ": "soc",
    "core_atom": "sem", "core_child_he": "sem", "parent_he": "sem",
    "fringe_atom": "sem", "cousin_he": "sem",
}


def relation_domain_kind(rel_key):
    """'same_soc' / 'same_sem' / 'cross' for one relational matrix, via RELATION_MAP."""
    mod = load_module()
    f1, f2 = mod.RELATION_MAP[rel_key]
    d1, d2 = FACET_DOMAINS[f1], FACET_DOMAINS[f2]
    if d1 == d2 == "soc":
        return "same_soc"
    if d1 == d2 == "sem":
        return "same_sem"
    return "cross"


def domain_community_volumes(config_id, Z_final, K, cross_domain_handling="split",
                              exclude_relations=frozenset()):
    """
    Per-community soc_vol[k] / sem_vol[k], relation-level (|Z_scaled[k,k]|, RAW --
    no permutation correction applied here, that's a separate, orthogonal question).

    exclude_relations: relation keys dropped entirely before computing volumes (e.g.
    {"S_Auth_Affil"}) -- for testing whether removing specific relations from the
    domain-balance mass/penalty closes the exploitability gap found this session
    (domain_penalty_exploitability_grid.py). Default: nothing excluded, identical
    to prior behaviour.

    cross_domain_handling:
      "naive"    cross-domain (anchor) matrices' full mass credited wholesale to
                 sem_vol only, because they happen to sit in sem_keys. This is what
                 the first domain_balance_no_correction_check.py test did -- kept
                 only as the "what was reported before" comparison point, not
                 recommended.
      "excluded" cross-domain matrices contribute to neither volume. Cleanest
                 possible baseline -- no double-counting question can arise, at the
                 cost of dropping real signal (the art<->semantic coupling itself).
      "split"    each cross-domain matrix's mass split (0.5/n_anchors) into
                 soc_vol and (0.5/n_anchors) into sem_vol -- the candidate from
                 CLAUDE.md §4.18 / FINDINGS §13, confirmed as the preferred choice
                 this session.

    Returns dict with soc_vol, sem_vol (np arrays, length K), num_soc, num_sem
    (matrix counts each volume was built from, for the same per-community
    normalization v8's own formula used), and cross_domain_relations (for the
    anchor_keys sanity check).
    """
    mod = load_module()
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    active = set(Z_final.keys()) - set(exclude_relations)

    same_soc = [r for r in soc_keys if r in active and relation_domain_kind(r) == "same_soc"]
    same_sem = [r for r in sem_keys if r in active and relation_domain_kind(r) == "same_sem"]
    cross = [r for r in (soc_keys + sem_keys) if r in active and relation_domain_kind(r) == "cross"]

    # sanity check: cross-domain (facet-level) must equal anchor_keys (mechanism-level)
    assert set(cross) == set(r for r in anchor_keys if r in active), (
        f"{config_id}: cross-domain matrices {cross} != anchor_keys {anchor_keys} -- "
        f"the coincidence this module's docstring documents has broken, investigate.")

    soc_vol = np.zeros(K)
    for r in same_soc:
        soc_vol += np.abs(np.diag(Z_final[r]))
    sem_vol = np.zeros(K)
    for r in same_sem:
        sem_vol += np.abs(np.diag(Z_final[r]))

    n_anchors = max(len(cross), 1)
    num_soc, num_sem = len(same_soc), len(same_sem)

    if cross_domain_handling == "naive":
        # historical: full mass into sem_vol (matches sem_keys membership literally)
        for r in cross:
            sem_vol += np.abs(np.diag(Z_final[r]))
        num_sem += len(cross)
    elif cross_domain_handling == "split":
        for r in cross:
            m = np.abs(np.diag(Z_final[r]))
            soc_vol += (0.5 / n_anchors) * m
            sem_vol += (0.5 / n_anchors) * m
        num_soc += len(cross)
        num_sem += len(cross)
    elif cross_domain_handling == "excluded":
        pass
    else:
        raise ValueError(f"unknown cross_domain_handling: {cross_domain_handling}")

    return {
        "soc_vol": soc_vol, "sem_vol": sem_vol,
        "num_soc": max(num_soc, 1), "num_sem": max(num_sem, 1),
        "same_soc_relations": same_soc, "same_sem_relations": same_sem,
        "cross_domain_relations": cross,
    }


SOC_EXCLUSIVE_FACETS = {"auth", "affil", "journ"}
SEM_EXCLUSIVE_FACETS = {"core_atom", "core_child_he", "parent_he", "fringe_atom", "cousin_he"}
# `art` deliberately excluded from both -- it's domain-classified social (this
# session's confirmed design intent), but every community gets some article
# representation near-structurally (art is the star topology's own center, touched
# by every active relation via propagation), so its own U_prob column is more a
# RESULT of everything else than an independent domain-exclusive membership signal.
# Matches the same exclusion logic already applied to relational matrices touching
# art in domain_community_volumes -- keep the two consistent, don't re-litigate art's
# role a third time in a new formula.


def domain_membership_distributions(config_id, U_final, presence_masks_dict, K):
    """
    U_prob-based per-community MEMBERSHIP distribution for each domain, pooling
    live entities across all domain-exclusive facets. Since U_prob rows are already
    L1-normalized (compute_probability_distributions), the live-entity mean of
    U_prob[:,k] pooled across a domain's facets is itself a valid probability
    distribution over K (sums to 1) -- "what fraction of live domain-X entities
    belong to community k", directly comparable to Z_scaled-based mass SHARES
    (also sum to 1 over K) without further normalization.

    Returns (soc_membership, sem_membership), each a length-K np array summing to 1
    (or None if a domain has zero live entities across its exclusive facets -- not
    expected in this grid, but not asserted away either).
    """
    mod = load_module()
    U_prob = mod.compute_probability_distributions(U_final)

    def pooled(facet_set):
        rows = []
        for f in facet_set:
            if f not in U_prob:
                continue
            mask = presence_masks_dict.get(f)
            mat = U_prob[f]
            live = mat[mask] if mask is not None else mat
            if live.shape[0] > 0:
                rows.append(live)
        if not rows:
            return None
        pooled_mat = np.concatenate(rows, axis=0)
        return pooled_mat.mean(axis=0)  # length-K, sums to 1 (mean of L1-normalized rows)

    return pooled(SOC_EXCLUSIVE_FACETS), pooled(SEM_EXCLUSIVE_FACETS)


def domain_share_vectors(dv, K):
    """
    Convert domain_community_volumes()'s raw soc_vol/sem_vol into WITHIN-domain
    share vectors (each sums to 1 over K) -- the same units as
    domain_membership_distributions' output, so mass and membership become directly
    comparable without a separate normalization step per caller.
    """
    soc_total, sem_total = dv["soc_vol"].sum(), dv["sem_vol"].sum()
    soc_share = dv["soc_vol"] / soc_total if soc_total > 1e-12 else np.full(K, np.nan)
    sem_share = dv["sem_vol"] / sem_total if sem_total > 1e-12 else np.full(K, np.nan)
    return soc_share, sem_share


def total_variation_distance(p, q):
    """0.5 * sum|p-q| for two same-length probability vectors. 0=identical, 1=disjoint."""
    return float(0.5 * np.sum(np.abs(p - q)))


def facet_hop_distances(config_id):
    """
    BFS hop-distance from the anchor-pinned reference frame (FINDINGS §14), per
    facet, over the config's active relation graph. 0-hop = 'art' plus every
    facet directly touched by an anchor relation (these get SVD init, per §14
    the presumed source of the gauge-fixing pull). Everything else's distance is
    the fewest active-relation hops back to that set.

    Not a claim this is THE right distance metric -- a hypothesis under test
    (this session's M_Child_Parent/hop-distance discussion), computed once here
    so it isn't hand-counted differently by different callers.
    """
    mod = load_module()
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    active_relations = [r for r in (soc_keys + sem_keys) if r in mod.RELATION_MAP]

    adjacency = {}
    for r in active_relations:
        f1, f2 = mod.RELATION_MAP[r]
        adjacency.setdefault(f1, set()).add(f2)
        adjacency.setdefault(f2, set()).add(f1)

    zero_hop = {"art"}
    for a in anchor_keys:
        f1, f2 = mod.RELATION_MAP[a]
        zero_hop.add(f1)
        zero_hop.add(f2)

    dist = {f: 0 for f in zero_hop}
    frontier = list(zero_hop)
    while frontier:
        nxt = []
        for f in frontier:
            for g in adjacency.get(f, ()):
                if g not in dist:
                    dist[g] = dist[f] + 1
                    nxt.append(g)
        frontier = nxt
    return dist


def align_seed_permutation(U_ref_facet, U_seed_facet):
    """
    Community index is only consistent WITHIN one fit -- an independent re-fit
    (different seed) has no reason to label the same latent community 'k' the
    same way. Before comparing two seeds' domain-share readings community-by-
    community, they need aligning, the same idea CLAUDE.md §10 already names for
    T1<->T2 alignment (Hungarian over a shared facet's profile similarity), just
    applied across seeds of the same fit instead of across time slices.

    U_ref_facet, U_seed_facet: U_norm[f] (N x K) for the SAME facet from two
    different seeds of the SAME (config_id, K). 'art' is the natural choice --
    present in every config, touched by every relation type.

    Returns col_ind: a length-K permutation such that seed community i is best
    matched to reference community col_ind[i] (cosine similarity, Hungarian-
    maximised). Apply via apply_seed_permutation.
    """
    ref_norm = U_ref_facet / (np.linalg.norm(U_ref_facet, axis=0, keepdims=True) + 1e-12)
    seed_norm = U_seed_facet / (np.linalg.norm(U_seed_facet, axis=0, keepdims=True) + 1e-12)
    cos_sim = ref_norm.T @ seed_norm  # K x K, [i,j] = cos(ref community i, seed community j)
    _, col_ind = linear_sum_assignment(-cos_sim)
    return col_ind


def apply_seed_permutation(U_final, Z_final, col_ind):
    """Relabel every facet's columns and every relation's Z_scaled (both axes,
    since a relation's row/col community indices are the SAME shared K-space)
    by the permutation align_seed_permutation found. Returns new dicts, inputs
    untouched."""
    U_aligned = {f: u[:, col_ind] for f, u in U_final.items()}
    Z_aligned = {r: z[np.ix_(col_ind, col_ind)] for r, z in Z_final.items()}
    return U_aligned, Z_aligned


def s5_dual_track_alignment(U_ref, U_seed, all_facets):
    """
    Faithful reuse of chunk13v9.py's own §S5 stability-analysis alignment
    (run_dual_track_stability_analysis, Phase 2) -- NOT a reimplementation.
    Calls the real row_normalize/col_normalize/row_wise_cosine_similarity
    functions off the loaded module, exactly the computation production uses
    to align seeds before comparing them, in place of the earlier, weaker
    art-only single-facet Hungarian match (align_seed_permutation above).

    U_ref, U_seed: dict[facet -> N_f x K array] (U_norm), same set of facets
    for both seeds. all_facets: list, must be the SAME deterministic ordering
    on both sides (get_required_facets -- ticket 82/§4.9's stacking guardrail;
    passing an unordered set here would silently reproduce the exact bug §4.9
    warns about).

    Returns (col_ind_A, col_ind_B) -- Track A (JSD/probability-space) and
    Track B (magnitude-weighted-cosine) column alignments. Apply either via
    apply_seed_permutation, same as align_seed_permutation's col_ind.
    """
    mod = load_module()
    S1_raw = np.vstack([U_ref[f] for f in all_facets])
    S2_raw = np.vstack([U_seed[f] for f in all_facets])

    S1_col_prob = mod.col_normalize(S1_raw)
    S2_col_prob = mod.col_normalize(S2_raw)
    K = S1_raw.shape[1]
    cost_A = np.zeros((K, K))
    cost_B = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            cost_A[k1, k2] = jensenshannon(S1_col_prob[:, k1], S2_col_prob[:, k2])
            cost_B[k1, k2] = 1.0 - (
                np.dot(S1_raw[:, k1], S2_raw[:, k2])
                / (np.linalg.norm(S1_raw[:, k1]) * np.linalg.norm(S2_raw[:, k2]) + 1e-9)
            )
    _, col_ind_A = linear_sum_assignment(cost_A)
    _, col_ind_B = linear_sum_assignment(cost_B)
    return col_ind_A, col_ind_B


def s5_dual_track_consensus(U_ref, U_seed, all_facets):
    """
    Extends s5_dual_track_alignment: also computes the CONTINUOUS per-facet
    similarity scores production's own Phase 2 uses (mean_js_sim,
    weighted_cos_sim), not just the alignment permutation. Motivated directly
    by a gap flagged in chat: a binary sign-flip on a derived quantity (e.g.
    domain-skew) conflates "communities broadly agree, one small quantity
    crossed zero" with "communities genuinely disagree" -- these continuous
    scores distinguish the two. Faithful reuse of the real per-facet
    evaluation loop (chunk13v9.py run_dual_track_stability_analysis, Phase 2,
    the block after linear_sum_assignment), not a reimplementation.

    Returns dict:
        col_ind_A, col_ind_B  -- same as s5_dual_track_alignment
        mean_js_sim   dict[facet -> float]   Track A, 1 - mean JS distance per row
        weighted_cos_sim dict[facet -> float] Track B, magnitude-weighted cosine
        pair_global_A, pair_global_B -- mean over facets (matches production's
            "Ontological Parity" pair-level aggregate)
    """
    mod = load_module()
    S1_raw = np.vstack([U_ref[f] for f in all_facets])
    S2_raw = np.vstack([U_seed[f] for f in all_facets])
    S1_col_prob = mod.col_normalize(S1_raw)
    S2_col_prob = mod.col_normalize(S2_raw)
    K = S1_raw.shape[1]
    cost_A = np.zeros((K, K))
    cost_B = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            cost_A[k1, k2] = jensenshannon(S1_col_prob[:, k1], S2_col_prob[:, k2])
            cost_B[k1, k2] = 1.0 - (
                np.dot(S1_raw[:, k1], S2_raw[:, k2])
                / (np.linalg.norm(S1_raw[:, k1]) * np.linalg.norm(S2_raw[:, k2]) + 1e-9)
            )
    _, col_ind_A = linear_sum_assignment(cost_A)
    _, col_ind_B = linear_sum_assignment(cost_B)

    mean_js_sim = {}
    weighted_cos_sim = {}
    for f in all_facets:
        U1_raw, U2_raw = U_ref[f], U_seed[f]
        U1_prob = mod.row_normalize(U1_raw)
        U2_prob_aligned = mod.row_normalize(U2_raw)[:, col_ind_A]
        js_distances = jensenshannon(U1_prob, U2_prob_aligned, axis=1)
        mean_js_sim[f] = float(1.0 - np.nanmean(js_distances))

        U2_raw_aligned = U2_raw[:, col_ind_B]
        cos_similarities = mod.row_wise_cosine_similarity(U1_raw, U2_raw_aligned)
        magnitude_weights = U1_raw.sum(axis=1)
        if magnitude_weights.sum() == 0:
            weighted_cos_sim[f] = float(np.mean(cos_similarities))
        else:
            weighted_cos_sim[f] = float(np.average(cos_similarities, weights=magnitude_weights))

    return {
        "col_ind_A": col_ind_A, "col_ind_B": col_ind_B,
        "mean_js_sim": mean_js_sim, "weighted_cos_sim": weighted_cos_sim,
        "pair_global_A": float(np.mean(list(mean_js_sim.values()))),
        "pair_global_B": float(np.mean(list(weighted_cos_sim.values()))),
    }


def relation_hop_distance(config_id, rel_key):
    """(hop_min, hop_max) of a relation's two facets -- see facet_hop_distances."""
    mod = load_module()
    f1, f2 = mod.RELATION_MAP[rel_key]
    dist = facet_hop_distances(config_id)
    h1, h2 = dist[f1], dist[f2]
    return min(h1, h2), max(h1, h2)


def make_domain_balance_penalty_torch(config_id, K, weight, exclude_relations=frozenset()):
    """
    EXPERIMENTAL, torch-differentiable mirror of domain_community_volumes(...,
    "split") + domain_share_vectors' TVD, for use as fit_instrumented's
    extra_loss_fn -- testing whether a Z_scaled-based domain-balance term is
    exploitable via the SAME U_scales-shrinking route FINDINGS §2 already
    documented for lambda_z_offdiag, before deciding whether this belongs in
    chunk13v9.py. Relation classification (same_soc/same_sem/cross) is computed
    once, outside the returned closure -- it doesn't change during training.

    exclude_relations: relation keys dropped from the penalty's own relation set --
    see domain_community_volumes' identical parameter. Used to test whether
    dropping specific relations (e.g. S_Auth_Affil) closes the exploitability gap.

    Returns fn(Z_scaled, U_scales, epoch) -> torch scalar (already weighted).
    """
    mod = load_module()
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    same_soc = [r for r in soc_keys if r not in exclude_relations and relation_domain_kind(r) == "same_soc"]
    same_sem = [r for r in sem_keys if r not in exclude_relations and relation_domain_kind(r) == "same_sem"]
    cross = [r for r in (soc_keys + sem_keys) if r not in exclude_relations and relation_domain_kind(r) == "cross"]
    n_anchors = max(len(cross), 1)

    def fn(Z_scaled, U_scales, U_norm, epoch):
        # U_norm accepted for signature compatibility with fit_instrumented's
        # BLOCKS_VERSION 1.14.0 extra_loss_fn contract -- unused here, this is
        # the Z_scaled-based candidate specifically, kept as-is (not fixed) so
        # its already-measured exploitability record stays reproducible.
        active = set(Z_scaled.keys())
        soc_vol = torch.zeros(K, dtype=torch.float32)
        for r in same_soc:
            if r in active:
                soc_vol = soc_vol + torch.abs(torch.diagonal(Z_scaled[r]))
        sem_vol = torch.zeros(K, dtype=torch.float32)
        for r in same_sem:
            if r in active:
                sem_vol = sem_vol + torch.abs(torch.diagonal(Z_scaled[r]))
        for r in cross:
            if r in active:
                m = torch.abs(torch.diagonal(Z_scaled[r]))
                soc_vol = soc_vol + (0.5 / n_anchors) * m
                sem_vol = sem_vol + (0.5 / n_anchors) * m

        soc_total = torch.clamp(soc_vol.sum(), min=1e-12)
        sem_total = torch.clamp(sem_vol.sum(), min=1e-12)
        soc_share = soc_vol / soc_total
        sem_share = sem_vol / sem_total
        tvd = 0.5 * torch.sum(torch.abs(soc_share - sem_share))
        return weight * (tvd ** 2)

    return fn


def make_inloop_domain_balance_penalty_torch(config_id, K, weight, tol=0.15, slice_name="T1", power=2):
    """
    Torch-differentiable version of CLAUDE.md §4.18's SETTLED in-loop
    domain-balance design (ticket 82, E2) -- built on U_prob (L1 row-normalized
    per-entity community distribution, derived from U_norm), NOT Z_scaled/mass.

    WHY U_prob, not Z_scaled (make_domain_balance_penalty_torch above): that
    Z_scaled-based candidate was built specifically to test for, and DID find,
    a real exploit -- domain_penalty_exploitability_grid.json (12-cell grid,
    weight 0 vs 10): mass_tvd (its own optimized target) improved in every
    cell, but mem_tvd (U_prob-based real membership imbalance) got WORSE in
    3/12 cells (C3/K4: +0.298, C6/K2: +0.268, C2/K2: +0.188) while the proxy
    "improved" 30-40x. Root cause: Z_scaled = Z_pos * U_scales_f1 * U_scales_f2,
    and ticket 79 already proved U_scales is an undetermined free gauge
    direction -- pooling raw Z_scaled diagonals ACROSS DIFFERENT facets (as any
    Z_scaled-based domain-balance sum must) lets the optimizer manipulate each
    relation's independently-gameable U_scales product to shrink the penalty
    without any real change to which entities belong to which community.

    U_norm (and therefore U_prob, a pure function of it) is PROVEN algebraically
    invariant to that exact transformation (ticket 79: scale one U_pos column by
    c, compensate in Z_pos -- U_norm unchanged to float precision, only U_scales
    moves) -- so this penalty cannot be gamed the same way, BY CONSTRUCTION, not
    just by not having found a counterexample yet. This is why the settled
    design (§4.18) chose U_prob over the Z_scaled route this function's sibling
    was probing.

    Matches E1's own methodology exactly (facet_membership_profile +
    domain_balance_r_k(weighting='entity')) so the in-loop penalty and the
    diagnostic reading measure literally the same quantity, computed
    differentiably here instead of post-hoc on numpy: per domain, an
    entity-count-weighted mean (D2) over that domain's active, live-nonempty
    facets (FACET_DOMAIN) of each facet's own live-entity-masked (ticket 60)
    U_prob row mean. r_k = soc_share_k / (soc_share_k + sem_share_k + eps);
    penalty = mean_k(max(0, |r_k-0.5| - tol)^2) -- the dead-band/hinge shape
    already used throughout this codebase (§4.5), tol=0.15 = D3's decided TOL.

    weight, tol: NOT Optuna-tunable (this session's decision, point 9) --
    fixed constants passed in by the caller, matching the precedent already
    set for TOL/log2/lambda_l1 (this corpus's noise floor can't adjudicate a
    continuous search, FINDINGS §21).

    power: exponent on the excess-past-tol term (default 2, the originally
    settled design). Added this session to test whether a higher exponent
    (excess and its GRADIENT both shrink toward 0 as power rises, for any
    excess < 1 -- checked algebraically: within this problem's actual range
    (excess in [0, 0.5-tol], so <=0.35 here), a higher power's gradient is
    SMALLER than power=2's at every point, not larger, unless weight is also
    rescaled up to compensate) gives a penalty that's gentler on borderline
    communities and much sharper on severely imbalanced ones, once weight is
    chosen accordingly -- rather than assuming "higher power = stronger" at
    a fixed weight, which the algebra above shows is false in this range.

    Returns fn(Z_scaled, U_scales, U_norm, epoch) -> torch scalar (weighted) --
    same 4-argument contract as fit_instrumented's extra_loss_fn (BLOCKS_VERSION
    1.14.0); Z_scaled/U_scales accepted but unused.
    """
    pm = presence_masks(config_id, slice_name=slice_name)
    live_counts = {f: int(m.sum()) for f, m in pm.items()}
    mask_t = {f: torch.from_numpy(m) for f, m in pm.items()}

    soc_facets = sorted(f for f in live_counts if FACET_DOMAIN.get(f) == "social" and live_counts[f] > 0)
    sem_facets = sorted(f for f in live_counts if FACET_DOMAIN.get(f) == "semantic" and live_counts[f] > 0)
    soc_w_total = sum(live_counts[f] for f in soc_facets)
    sem_w_total = sum(live_counts[f] for f in sem_facets)
    eps = 1e-12

    def _domain_share(U_norm, facets, w_total):
        if not facets or w_total == 0:
            return None
        acc = None
        for f in facets:
            if f not in U_norm:
                continue
            u = U_norm[f]
            live = u[mask_t[f]]
            if live.shape[0] == 0:
                continue
            row_sums = torch.sum(torch.abs(live), dim=1, keepdim=True) + eps
            u_prob_live = live / row_sums
            term = live_counts[f] * u_prob_live.mean(dim=0)
            acc = term if acc is None else acc + term
        return None if acc is None else acc / w_total

    def fn(Z_scaled, U_scales, U_norm, epoch):
        soc_share_k = _domain_share(U_norm, soc_facets, soc_w_total)
        sem_share_k = _domain_share(U_norm, sem_facets, sem_w_total)
        if soc_share_k is None or sem_share_k is None:
            return torch.tensor(0.0)
        r_k = soc_share_k / (soc_share_k + sem_share_k + eps)
        excess = torch.clamp(torch.abs(r_k - 0.5) - tol, min=0.0)
        return weight * torch.mean(excess ** power)

    return fn


# =============================================================================
# PERMUTATION / MASS (FINDINGS §16, tickets 79/80/82)
# =============================================================================

def hungarian_diagonal_match(Z, K):
    """
    Given one relation's K x K Z_scaled matrix, find the column permutation
    that maximises matched (within-community) mass. Returns:
      permutation       list[int], length K -- best-matching column order
      is_identity       bool -- best permutation is "do nothing"
      structure_score   float -- min(matched) / max(unmatched). >1 means every
                         within-community entry beats every cross-community
                         entry (clean, correctable mislabelling); <1 means at
                         least one cross-community entry is bigger than some
                         within-community entry (genuine mixed coupling --
                         nothing to correct, see FINDINGS §16).
      diag_share_raw / diag_share_permuted   diagonal's share of total |Z|
                         mass, before/after applying the permutation.
    """
    absZ = np.abs(Z)
    total_mass = absZ.sum()
    diag_share_raw = float(np.sum(np.diag(absZ)) / total_mass) if total_mass > 0 else float("nan")

    row_ind, col_ind = linear_sum_assignment(-absZ)
    perm = col_ind.tolist()
    is_identity = perm == list(range(K))

    matched_vals = np.array([absZ[i, perm[i]] for i in range(K)])
    diag_share_permuted = float(matched_vals.sum() / total_mass) if total_mass > 0 else float("nan")

    mask = np.ones_like(absZ, dtype=bool)
    for i in range(K):
        mask[i, perm[i]] = False
    unmatched_vals = absZ[mask]
    min_matched = float(matched_vals.min())
    max_unmatched = float(unmatched_vals.max()) if unmatched_vals.size else 0.0
    structure_score = min_matched / max_unmatched if max_unmatched > 1e-15 else float("inf")

    return {
        "permutation": perm, "is_identity": is_identity, "structure_score": structure_score,
        "diag_share_raw": diag_share_raw, "diag_share_permuted": diag_share_permuted,
    }


def relation_community_mass(Z, K, threshold=1.0):
    """
    FINDINGS §16's confidence-gated mass reading for one relation, replacing
    the undetermined U_scales (ticket 79). Returns a length-K array: mass[k]
    is community k's within-community contribution from this relation.

    If this relation's own structure_score > threshold: read the CORRECTED
    diagonal, |Z[k, perm[k]]| -- the relation is confidently a relabelling of
    the true within-community structure, so translate through it.
    Otherwise: read the RAW diagonal, |Z[k, k]| -- structure_score <= threshold
    means no relabelling produces a clean separation (genuine cross-community
    coupling), so nothing is corrected; the raw diagonal is the best available
    reading, not a guess dressed up as a fix.

    threshold=1.0 is FINDINGS §16's derived default (natural gap in the score
    distribution, 0.648->1.098, stable across [0.65, 1.5]) -- TOY-CORPUS
    CALIBRATED, re-derive before trusting at 22k-article scale.
    """
    match = hungarian_diagonal_match(Z, K)
    absZ = np.abs(Z)
    if match["structure_score"] > threshold:
        perm = match["permutation"]
        return np.array([absZ[k, perm[k]] for k in range(K)])
    return np.diag(absZ).copy()


def collapse_mass_share(config_id, U_final, Z_final, K):
    """
    Z_scaled-based per-community share, computed the IDENTICAL way
    evaluate_dimensional_collapse does -- calls production's own
    `_relation_community_share` per active relation (reconstruction-space
    share, FINDINGS §8, not the simpler diagonal-sum share) and averages with
    equal weight across relations, then renormalizes to sum to 1. Not a
    reimplementation: literally the same function object as the one
    `collapse_pen` itself is built on, so `collapse_mass_share(...).max()`
    must equal `evaluate(...)["collapse_score"]` -- used as a live
    self-check in collapse_pen_mass_vs_membership_check.py, not assumed.

    Returns a length-K np array summing to 1 (or uniform 1/K if total
    reconstructed mass is ~0, matching the production fallback).
    """
    mod = load_module()
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    structure_threshold = mod.STRUCTURE_SCORE_THRESHOLD
    relation_shares = []
    for rel_key, Z in Z_final.items():
        f1, f2 = mod.RELATION_MAP[rel_key]
        share = mod._relation_community_share(Z, U_final[f1], U_final[f2], structure_threshold)
        relation_shares.append(share)
    community_share = np.mean(relation_shares, axis=0)
    total = community_share.sum()
    return community_share / total if total > 1e-15 else np.full(K, 1.0 / K)


def community_share_vector(mod, U_norm, Z_scaled, structure_threshold):
    """
    Tickets 86/87 Stage 0a: centralized here from 7 near-verbatim copies
    (5 as a locally-defined function of this exact name, 2 inlined directly
    in a loop -- ghost_test3_share_distribution.py / _v2.py) across the
    ghost_* diagnostic scripts, all confirmed identical before this move
    (diffed byte-for-byte). Import this instead of redefining it.

    Same computation as `collapse_mass_share` above -- both call production's
    own `_relation_community_share` per active relation and average with
    equal weight across relations -- kept as a SEPARATE function rather than
    merged into it because the two differ in what they take as input
    (`config_id`+K vs. `mod`+`structure_threshold` passed directly, avoiding
    a redundant `get_active_facets` call already done by the caller) and in
    their near-zero-total fallback (uniform 1/K here vs. the raw near-zero
    vector there) -- `collapse_pen_mass_vs_membership_check.py` depends on
    `collapse_mass_share`'s specific fallback, so merging them was judged a
    larger, separately-scoped change than Stage 0a asked for, not made here.

    Returns a length-K np array (community_share, reconstruction-space,
    permutation-corrected) normalized to sum to 1, EXCEPT when total
    reconstructed mass is ~0, in which case it returns the raw (near-zero)
    unnormalized vector unchanged -- matches every one of the 7 sites this
    was centralized from, verbatim.
    """
    shares = []
    for rel, Z in Z_scaled.items():
        f1, f2 = mod.RELATION_MAP[rel]
        shares.append(mod._relation_community_share(Z, U_norm[f1], U_norm[f2], structure_threshold))
    community_share = np.mean(shares, axis=0)
    total = community_share.sum()
    return community_share / total if total > 1e-15 else community_share


def relation_share_matrix(mod, Z, U_f1, U_f2, structure_threshold, K):
    """
    Full K x K permutation-corrected reconstruction-space share for ONE
    relation -- generalizes production's own `_relation_community_share`
    (which only ever returns the diagonal, length-K) to the full matrix, so
    off-diagonal (between-community) cells can be read on the same corrected
    basis as the diagonal ones. Not a new methodology: calls production's own
    `_hungarian_relabel_relation` for the permutation decision (identical
    structure_score > structure_threshold gate) and the identical trace-based
    `total` reconstructed-mass normalization -- only the "which cells get
    read out" step is extended from diagonal-only to the full grid.

    Self-checked (see z_scaled_offdiag_calibration_test.py): this matrix's
    diagonal matches `_relation_community_share(...)` exactly, since
    Z_permuted[k,k] = Z[k, p[k]] by construction -- the same permuted read
    the production function uses for its own diagonal.

    Caveat, stated rather than assumed: `total` (trace(Z^T c_u Z c_v)) is NOT
    in general equal to sum(Z_permuted**2) unless c_u/c_v are exactly
    identity (perfectly orthogonal columns, not just unit-L2-norm) -- so this
    matrix's cells do not necessarily sum to exactly 1.0 the way the
    diagonal-only vector is documented to. Reported, not hidden, in the
    calibration script's self-check output.
    """
    perm, trusted = mod._hungarian_relabel_relation(Z, structure_threshold)
    p = perm if trusted else list(range(K))
    Z_permuted = Z[:, p]

    c_u = U_f1.T @ U_f1
    c_v = U_f2.T @ U_f2
    total = float(np.trace(Z.T @ c_u @ Z @ c_v))

    if total <= 1e-15:
        return np.full((K, K), 1.0 / (K * K))
    return (Z_permuted ** 2) / total


def membership_share_all_facets(config_id, U_final, presence_masks_dict, K):
    """
    Independent, U_prob-based analog of collapse_mass_share -- NOT derived
    from Z_scaled at all, so it cannot inherit any exploit that lives in the
    Z_scaled pathway (§4.2's scale-invariance loophole, or a novel one).

    For each of the 9 facets, takes the mean U_prob row (L1-normalized, so
    each row already sums to 1 over K) across that facet's LIVE entities only
    (presence_masks, ticket 60) -- "what fraction of this facet's real
    entities belong to community k". Averages across all 9 facets with EQUAL
    WEIGHT, mirroring collapse_mass_share's own equal-weight-per-relation
    choice (facet-level here since U_prob is facet-indexed, not
    relation-indexed) -- not a new convention invented for this check.

    Returns a length-K np array summing to 1 (nan-filled if no facet has any
    live entities, not expected in this grid).
    """
    mod = load_module()
    U_prob = mod.compute_probability_distributions(U_final)
    facet_means = []
    for f, mat in U_prob.items():
        mask = presence_masks_dict.get(f)
        live = mat[mask] if mask is not None else mat
        if live.shape[0] > 0:
            facet_means.append(live.mean(axis=0))
    if not facet_means:
        return np.full(K, np.nan)
    return np.mean(facet_means, axis=0)


# =============================================================================
# DOMAIN BALANCE (ticket 82 / CLAUDE.md §4.18 -- E1 measurement, plan in
# .claude/plans/read-claude-md-in-full-swirling-haven.md)
# =============================================================================

# Declared, not derived from which relations touch a facet -- deriving it from
# relation participation is exactly what makes 'art' look ambiguous (it's
# touched by both S_ and M_ relations) when its actual domain is unambiguous:
# 'art' is a bibliographic/administrative entity, same as auth/journ/affil.
# No facet is mixed. See the domain-balance plan's "Design decisions settled".
FACET_DOMAIN = {
    'art': 'social', 'auth': 'social', 'journ': 'social', 'affil': 'social',
    'core_atom': 'semantic', 'core_child_he': 'semantic', 'parent_he': 'semantic',
    'cousin_he': 'semantic', 'fringe_atom': 'semantic',
}


def facet_membership_profile(config_id, U_final, presence_masks_dict):
    """
    Per-facet U_prob profile, unaggregated: for each active facet, the mean
    U_prob row (L1-normalized, sums to 1 over K) across that facet's LIVE
    entities only. This is the identical per-facet computation
    membership_share_all_facets makes internally, exposed here BEFORE its own
    equal-weight pooling across all 9 facets, so domain_balance_r_k can pool
    by domain (social/semantic) instead of pooling everything together.

    Returns dict[facet -> length-K np array summing to 1] (facets with zero
    live entities are omitted, not nan-filled).
    """
    mod = load_module()
    U_prob = mod.compute_probability_distributions(U_final)
    profiles = {}
    for f, mat in U_prob.items():
        mask = presence_masks_dict.get(f)
        live = mat[mask] if mask is not None else mat
        if live.shape[0] > 0:
            profiles[f] = live.mean(axis=0)
    return profiles


def facet_relation_profile(config_id, Z_final, U_final, K, structure_threshold,
                            exclude_anchors=False):
    """
    Per-facet Z_scaled-based profile -- the outer-loop / reconstruction-space
    analog of facet_membership_profile. For each active facet, averages the
    per-relation share_r (production's own _relation_community_share,
    permutation-corrected reconstruction-space share -- identical call
    collapse_mass_share makes) over every relation touching that facet, then
    renormalizes to sum to 1.

    Each relation credits BOTH facets it touches with its own share_r vector
    -- a true statement ("this relation contributes to both endpoint facets'
    domain profile"), not double-counting. See the domain-balance plan's
    "Outer-loop measure" section for why this differs from a relation-level
    (not facet-level) domain sum, and why that matters for anchor relations
    specifically.

    exclude_anchors=True gives the V2 sensitivity-floor variant used to
    measure how much of a facet's profile is anchor-carried: anchor relations
    (config-specific, from get_active_facets) are skipped when building the
    per-facet average. V1 (exclude_anchors=False, the primary/default
    reading) minus V2 is the "anchor sensitivity" figure the E1 script reports
    per cell -- see the plan for why V1 (anchors included) is primary, not V2.

    Returns dict[facet -> length-K np array summing to 1].
    """
    mod = load_module()
    soc_keys, sem_keys, anchor_keys = mod.get_active_facets(config_id)
    anchor_set = set(anchor_keys) if exclude_anchors else set()
    structure_th = structure_threshold

    facet_rel_shares = {}
    for rel_key, Z in Z_final.items():
        if rel_key in anchor_set:
            continue
        f1, f2 = mod.RELATION_MAP[rel_key]
        share = mod._relation_community_share(Z, U_final[f1], U_final[f2], structure_th)
        facet_rel_shares.setdefault(f1, []).append(np.asarray(share))
        facet_rel_shares.setdefault(f2, []).append(np.asarray(share))

    profiles = {}
    for f, shares in facet_rel_shares.items():
        m = np.mean(shares, axis=0)
        total = m.sum()
        profiles[f] = m / total if total > 1e-15 else np.full(K, 1.0 / K)
    return profiles


def domain_balance_r_k(facet_profiles, K, live_counts=None, weighting='equal'):
    """
    Groups a dict[facet -> length-K profile] (from facet_membership_profile
    or facet_relation_profile) into social vs semantic domain sums via
    FACET_DOMAIN, then computes the per-community balance ratio.

    weighting='equal': every active facet in a domain contributes equally to
        that domain's sum -- matches collapse_mass_share's and
        membership_share_all_facets's own equal-weight-per-unit convention,
        the codebase's established default (see plan's D2).
    weighting='entity': each facet is weighted by its live entity count
        (requires live_counts = dict[facet -> int]). Note this is a
        BETWEEN-facet weight only -- p[f] itself is already a mean over that
        facet's own live entities, so within-facet entity count is already
        accounted for regardless of which between-facet weighting is used.

    r_k = soc_k / (soc_k + sem_k + eps), in [0,1]; 0.5 = a perfectly balanced
    community. dev_k = |r_k - 0.5| is the deviation E1 reports the
    distribution of, to read off a TOL band directly from data (plan's D3).

    Returns dict with soc/sem/r/dev (each length-K, as plain lists for JSON)
    and which facets fed each domain -- so an empty domain is visible, not
    silently nan.
    """
    eps = 1e-12
    soc_facets = [f for f in facet_profiles if FACET_DOMAIN.get(f) == 'social']
    sem_facets = [f for f in facet_profiles if FACET_DOMAIN.get(f) == 'semantic']

    def _weighted_mean(facets):
        if not facets:
            return None
        if weighting == 'equal':
            w = np.ones(len(facets))
        elif weighting == 'entity':
            w = np.array([live_counts[f] for f in facets], dtype=float)
        else:
            raise ValueError(f"unknown weighting: {weighting}")
        w = w / w.sum()
        stacked = np.array([facet_profiles[f] for f in facets])
        return (w[:, None] * stacked).sum(axis=0)

    soc = _weighted_mean(soc_facets)
    sem = _weighted_mean(sem_facets)

    if soc is None or sem is None:
        return {"soc": None, "sem": None, "r": None, "dev": None,
                "soc_facets": soc_facets, "sem_facets": sem_facets}

    r = soc / (soc + sem + eps)
    dev = np.abs(r - 0.5)
    return {"soc": soc.tolist(), "sem": sem.tolist(), "r": r.tolist(), "dev": dev.tolist(),
            "soc_facets": soc_facets, "sem_facets": sem_facets}


# =============================================================================
# AGREEMENT CHECK (SESSION_PROTOCOL §E)
# =============================================================================

def verify_agreement(config_id="C6", K=4, lambda_l1=0.0, lambda_z_offdiag=0.05,
                     slice_name="T1", write=True):
    """
    fit_instrumented(init_mode='production') vs the real run_inner_solver, on an
    identical config. Reports agreement in epochs and math_loss, plus max
    absolute elementwise difference over U_final / Z_final / U_scales.
    """
    print(f"{'='*78}\nSESSION_PROTOCOL §E AGREEMENT CHECK -- diagnostic_blocks v{BLOCKS_VERSION}")
    print(f"{config_id} K={K} {slice_name} lambda_l1={lambda_l1} "
          f"lambda_z_offdiag={lambda_z_offdiag}\n{'='*78}")

    Up, Zp, Dp = fit_production(config_id, K, lambda_l1, lambda_z_offdiag, slice_name=slice_name)
    Ui, Zi, Di = fit_instrumented(config_id, K, lambda_l1, lambda_z_offdiag, slice_name=slice_name)

    def maxdiff(a, b):
        keys = sorted(set(a) & set(b))
        if sorted(a) != sorted(b):
            return float("nan")
        return max((float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) for k in keys),
                   default=0.0)

    res = {
        "blocks_version": BLOCKS_VERSION,
        "chunk13v9_hash": chunk13v9_hash(),
        "config_id": config_id, "K": K, "slice": slice_name,
        "lambda_l1": lambda_l1, "lambda_z_offdiag": lambda_z_offdiag,
        "epochs_production": Dp["epochs_run"], "epochs_instrumented": Di["epochs_run"],
        "epochs_match": Dp["epochs_run"] == Di["epochs_run"],
        "math_loss_production": Dp["math_loss"], "math_loss_instrumented": Di["math_loss"],
        "math_loss_absdiff": abs(Dp["math_loss"] - Di["math_loss"]),
        "converged_production": bool(Dp["converged"]), "converged_instrumented": bool(Di["converged"]),
        "max_absdiff_U_final": maxdiff(Up, Ui),
        "max_absdiff_Z_final": maxdiff(Zp, Zi),
        "max_absdiff_U_scales": maxdiff(Dp["U_scales"], Di["U_scales"]),
    }

    print(f"  epochs      : production={res['epochs_production']}  "
          f"instrumented={res['epochs_instrumented']}  match={res['epochs_match']}")
    print(f"  math_loss   : production={res['math_loss_production']:.10f}  "
          f"instrumented={res['math_loss_instrumented']:.10f}")
    print(f"                absdiff={res['math_loss_absdiff']:.3e}")
    print(f"  max |diff|  : U_final={res['max_absdiff_U_final']:.3e}  "
          f"Z_final={res['max_absdiff_Z_final']:.3e}  U_scales={res['max_absdiff_U_scales']:.3e}")

    ok = (res["epochs_match"] and res["math_loss_absdiff"] < 1e-9
          and res["max_absdiff_U_final"] < 1e-9 and res["max_absdiff_Z_final"] < 1e-9)
    res["verdict"] = "AGREE" if ok else "DISAGREE"
    print(f"\n  VERDICT: {res['verdict']}"
          + ("" if ok else "  <-- fit_instrumented has drifted from run_inner_solver. Fix before use."))

    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "blocks_agreement_check.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  -> saved {OUT_DIR}/blocks_agreement_check.json")
    return res


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_agreement()
    else:
        print(__doc__)
        print(f"BLOCKS_VERSION = {BLOCKS_VERSION}")
        print(f"chunk13v9 hash  = {chunk13v9_hash()}")
        print("\nRun `python diagnostic_blocks.py --verify` for the §E agreement check.")
