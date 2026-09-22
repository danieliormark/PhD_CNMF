# =============================================================================
# CHUNK 13v9 - Module 1: STATIC ARCHITECTURE & TOPOLOGY
# SCOPE: Global constants, HPC paths, and Metagraph configurations (C1-C6)
# =============================================================================

import os
import torch
import pickle
import numpy as np

# -----------------------------------------------------------------------------
# 1.1 HARDWARE CONFIGURATION
# -----------------------------------------------------------------------------
# Automatically binds PyTorch to the GPU if available on your SLURM node
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1.2 GLOBAL CONSTANTS & HEURISTICS
# -----------------------------------------------------------------------------
# Inner Loop (Adam Solver) Constants
# Ticket 72: now actually wired into run_inner_solver's defaults (was dead).
# Ticket 74: 400 was cutting fits off mid-descent, not after convergence (C6/K=4
# measured recon_loss=0.797 at epoch 400 vs a true plateau of 0.773 reached by
# ~epoch 950-1000). INNER_EPOCHS is a CEILING, not a target — the relative
# early-stopping check (§2.2, rel_change < 1e-4 over a 20-epoch window) decides
# actual per-run length, and different configs/anchors may converge at
# different epoch counts. Runs that converge early cost nothing.
INNER_EPOCHS = 2000
LEARNING_RATE = 0.01  # Standard starting point for Adam in NMF

# =============================================================================
# OUTER LOOP: LOCKED DOMAIN CONSTANTS & SOCIOLOGICAL RULES
# =============================================================================

# 1. Dimensional Collapse (Entropy) Rule
# VESTIGIAL as of tickets 79/80/82 (FINDINGS §12/13/16): U_scales, the mass
# input the entropy formula was built on, is proven an undetermined free
# gauge direction (ticket 79), not a mass measure — see evaluate_dimensional_
# collapse's docstring for the rewrite. Kept defined (unused by collapse) only
# because evaluate_complete_solution's external signature still accepts an
# entropy_threshold parameter from all 3 call sites — changing that signature
# was avoided deliberately (CLAUDE.md §3: "verify signatures before changing
# any of them" — 3 call sites, no reason to touch them for this fix).
ENTROPY_THRESHOLD = 0.60       # Minimum allowed normalized Shannon entropy mass floor

# 1b. Dimensional Collapse (Max-Share) Rule — REPLACES the entropy rule above
# for actual collapse detection (ticket 80). Same NUMERIC VALUE as
# ENTROPY_THRESHOLD, reused deliberately: CLAUDE.md §4.4 already established
# 0.60 as the intended target ("no community >0.6 of mass") — this is that
# target applied directly to max-share instead of through entropy, which was
# not a monotone function of max-share and whose correspondence to a fixed
# threshold was K-dependent (see §4.4's worked example). TOY-CORPUS
# CALIBRATED alongside STRUCTURE_SCORE_THRESHOLD below — the 0.60 VALUE
# itself is not toy-corpus-derived (it's the pre-existing design target), but
# whether it's the right cutoff for 22k-scale mass distributions is untested.
MAX_SHARE_THRESHOLD = 0.60

# 1c. Permutation-correction confidence gate (ticket 82, FINDINGS §16).
# structure_score = min(matched)/max(unmatched) per relation; >1.0 means a
# confidently clean Hungarian relabelling exists and is applied; <=1.0 means
# genuine mixed coupling (not a labelling artifact) and the raw diagonal is
# used as-is. TOY-CORPUS CALIBRATED: 1.0 sits in a real gap in this corpus's
# score distribution (0.648 -> 1.098, nothing between), stable under a
# 16-point sensitivity sweep across [0.65, 1.5]. Re-derive before trusting at
# 22k-article scale — do not assume this value ports. See
# evaluate_dimensional_collapse / _hungarian_relabel_relation.
STRUCTURE_SCORE_THRESHOLD = 1.0

# 2. Topological Coherence Rule
TARGET_COHERENCE = 0.50        # Minimum required mean anchor diagonal dominance ratio

# 2b. Per-Community Domain-Balance Rule (ticket 82, E2). D3: TOL=0.15, set
# from a measured multi-seed noise floor (FINDINGS §21), not tuned -- like
# every other threshold in this file (ENTROPY_THRESHOLD, TARGET_COHERENCE),
# NOT Optuna-tunable (§4.7). See CLAUDE.md §4.18 update / FINDINGS §22 for
# the full evidence behind this value and behind the in-loop weight
# (lambda_domain_balance, read in run_inner_solver) currently defaulting to
# 0.0.
DOMAIN_BALANCE_TOL = 0.15

# 3. Doxa / Elite Capture Rule
#MAX_MONOPOLY = 0.60            # Maximum allowable mass concentration for elite entities in a single community deprecated in favour of more flexible 0.85

# 4. Meta-Loss Aggregation Weights (Calibrated via initial trial baseline)
LAMBDA_COLLAPSE = 1.0          # Penalty weight for dimensional collapse
LAMBDA_COH = 1.0               # Penalty weight for topological coherence violations
LAMBDA_SEM = 1.0               # Penalty weight for socio-semantic semantic reality deviations

# 5. Seed and version control for reproducibility
MASTER_SEED = 42
PIPELINE_VERSION = "v9.1.time_slice_1"

# (Note: CORE_THRESHOLD = 0.75 has been intentionally retired and removed 
#  as Section 4A now uses continuous probabilistic message passing instead 
#  of hard article cutoffs).

# -----------------------------------------------------------------------------
# 1.3 HPC INFRASTRUCTURE PATHS
# -----------------------------------------------------------------------------
USER_ID = "p91688di"

# High-Speed parallel scratch space (for Optuna tracking during optimization)
SCRATCH_DIR = f"/scratch/{USER_ID}"
OPTUNA_JOURNAL_PATH = os.path.join(SCRATCH_DIR, "optuna_journal_storage", "optuna.log")

# Safe, backed-up Research Data Storage (for loading raw matrices and saving final artifacts)
# UPDATE THIS PATH to match exactly where your raw data sits
RDS_DIR = f"/mnt/hum01-home01/{USER_ID}/" 

# -----------------------------------------------------------------------------
# 1.4 TOPOLOGICAL SWITCHBOARD (PATCHED FOR SVD ANCHORS)
# -----------------------------------------------------------------------------
def get_active_facets(config_id):
    """
    Defines the active metagraph topology for the given configuration.
    Social facets are static across all configurations.
    Semantic facets dynamically shift how the semantic network anchors to Articles.
    anchor_keys explicitly dictate which matrices receive SVD initialization.
    """
    # Static Social Capital anchors
    soc_keys = ['S_Art_Auth', 'S_Auth_Affil', 'S_Art_Journ']
    
    # Dynamic Semantic Capital & SVD Anchor keys
    if config_id == 'C1':
        sem_keys = ['M_Atom_Child', 'M_Child_Parent', 'M_Parent_Art', 'M_Fringe_Cousin', 'M_Cousin_Parent']
        anchor_keys = ['M_Parent_Art']
    
    elif config_id == 'C2':
        sem_keys = ['M_Atom_Child', 'M_Child_Art', 'M_Fringe_Cousin', 'M_Cousin_Art', 'M_Child_Parent']
        anchor_keys = ['M_Child_Art', 'M_Cousin_Art']
    
    elif config_id == 'C3':
        sem_keys = ['M_Atom_Child', 'M_Child_Parent', 'M_Child_Art', 'M_Fringe_Cousin', 'M_Cousin_Child']
        anchor_keys = [ 'M_Child_Art']
    
    elif config_id == 'C4':
        sem_keys = ['M_Atom_Child', 'M_Child_Art', 'M_Fringe_Cousin', 'M_Cousin_Child']
        anchor_keys = ['M_Child_Art']

    elif config_id == 'C5':
        sem_keys = ['M_Atom_Child', 'M_Child_Art', 'M_Cousin_Art', 'M_Fringe_Cousin']
        anchor_keys = [ 'M_Child_Art', 'M_Cousin_Art']

    elif config_id == 'C6':
            sem_keys = ['M_Atom_Child', 'M_Child_Art', 'M_Cousin_Art', 'M_Fringe_Cousin', 'M_Cousin_Parent', 'M_Child_Parent']
            anchor_keys = [ 'M_Child_Art', 'M_Cousin_Art']
    
    else:
        raise ValueError(f"CRITICAL: Unknown configuration ID: {config_id}")
        
    return soc_keys, sem_keys, anchor_keys

# -----------------------------------------------------------------------------
# 1.5 ONTOLOGICAL DICTIONARY (RELATION MAP) 1.5 and 1.6 added as update on 29.07.2026 21:18
# -----------------------------------------------------------------------------
# This maps every relation matrix to its structural row/col domains (Facets).
# Essential for dynamic dimension discovery and Hungarian matrix alignment.
RELATION_MAP = {
    # Social Ties
    'S_Art_Auth':      ('art', 'auth'),
    'S_Auth_Affil':    ('auth', 'affil'),
    'S_Art_Journ':     ('art', 'journ'),
    
    # Semantic Ties (Grammar & Anchors)
    'M_Atom_Child':    ('core_atom', 'core_child_he'),
    'M_Child_Parent':  ('core_child_he', 'parent_he'),
    'M_Parent_Art':    ('parent_he', 'art'),
    'M_Child_Art':     ('core_child_he', 'art'),
    'M_Fringe_Cousin': ('fringe_atom', 'cousin_he'),
    'M_Cousin_Parent': ('cousin_he', 'parent_he'),
    'M_Cousin_Child':  ('cousin_he', 'core_child_he'),
    'M_Cousin_Art':    ('cousin_he', 'art')
}

# -----------------------------------------------------------------------------
# 1.5b DOMAIN CLASSIFICATION (ticket 82, per-community domain balance) added
# 28.08.2026. Declared, not derived from which relations touch a facet --
# deriving it from relation participation is exactly what makes 'art' look
# ambiguous (touched by both S_ and M_ relations) when its actual domain is
# unambiguous: 'art' is a bibliographic/administrative entity, same kind as
# auth/journ/affil. No facet is mixed -- see CLAUDE.md §4.18's "Design
# settled" discussion. Matches diagnostic_blocks.py's FACET_DOMAIN exactly
# (kept as two independent copies deliberately -- diagnostic_blocks.py is a
# separate, standalone diagnostic module by this project's own convention,
# not imported into production; this is chunk13v9.py's own copy for E2).
# -----------------------------------------------------------------------------
FACET_DOMAIN = {
    'art': 'social', 'auth': 'social', 'journ': 'social', 'affil': 'social',
    'core_atom': 'semantic', 'core_child_he': 'semantic', 'parent_he': 'semantic',
    'cousin_he': 'semantic', 'fringe_atom': 'semantic',
}

# -----------------------------------------------------------------------------
# 1.6 FACET EXTRACTION UTILITY updated 01.08.2026 22:43
# -----------------------------------------------------------------------------
def get_required_facets(soc_keys, sem_keys, anchor_keys=None):
    """
    Derives the deterministic, alphabetically sorted list of facet names 
    required by a given configuration. Uses RELATION_MAP as the single source of truth.
    
    Returns a sorted list to guarantee deterministic tensor stacking.
    """
    if anchor_keys is None:
        anchor_keys = []
        
    required_facets = {
        facet
        for rel in set(soc_keys + sem_keys + anchor_keys)
        if rel in RELATION_MAP
        for facet in RELATION_MAP[rel]
    }
    
    # CRITICAL: Must be sorted to prevent random vertical stacking misalignment
    return sorted(list(required_facets))

# -----------------------------------------------------------------------------
# 1.7 DATA LOADER & DIMENSION INFERENCE ENGINE updated 01.08.2026 22:43 
# -----------------------------------------------------------------------------
def load_and_validate_data(filepath):
    """
    Loads raw SciPy sparse matrices from Chunk 12 and dynamically infers 
    network dimensions to ensure 100% topological accuracy.
    """
    print(f"[*] Loading tensor topology from: {filepath}")
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)

    # 1. Separate the actual mathematical matrices from any metadata
    matrices = {k: v for k, v in raw_data.items() if k != 'dimensions' and k in RELATION_MAP}

    missing = [k for k in RELATION_MAP.keys() if k not in raw_data]
    if missing:
        print(f"[!] NOTE: {len(missing)} relations absent from payload: {missing}")
    
    # 2. Dynamic Dimension Inference (Self-Healing Architecture)
    dimensions = raw_data.get('dimensions', None)
    
    if dimensions is None:
        print("[!] 'dimensions' dictionary not found. Initiating dynamic auto-discovery...")
        dimensions = {}
        
        for rel_key, matrix in matrices.items():
            f1, f2 = RELATION_MAP[rel_key]
            rows, cols = matrix.shape
            
            # Lock or verify Row Dimension
            if f1 not in dimensions:
                dimensions[f1] = rows
            else:
                assert dimensions[f1] == rows, f"Fatal Geometry Mismatch: {f1} expected {dimensions[f1]}, got {rows} in {rel_key}"
                
            # Lock or verify Column Dimension
            if f2 not in dimensions:
                dimensions[f2] = cols
            else:
                assert dimensions[f2] == cols, f"Fatal Geometry Mismatch: {f2} expected {dimensions[f2]}, got {cols} in {rel_key}"
                
        print("[+] Auto-discovery successful. Dimensions mathematically locked:")
        for k, v in dimensions.items():
            print(f"    -> {k.ljust(15)}: {v}")
    
    # Repackage the data cleanly for the downstream PyTorch solver
    clean_data = matrices.copy()
    clean_data['dimensions'] = dimensions

    return clean_data

# -----------------------------------------------------------------------------
# 1.8 PRESENCE MASKS (LIVE-ENTITY DISCOVERY)
# -----------------------------------------------------------------------------
def build_presence_masks(matrices, soc_keys, sem_keys):
    """
    Ticket 60. Entity index maps for auth/affil/journ and all semantic facets
    are GLOBAL across chunk12's time slices, so a slice's matrices are padded
    with structurally-zero rows/cols for entities that only exist in another
    slice (this is deliberate — identical dimensions across slices are what
    make T1<->T2 community comparison possible, and must NOT be "fixed").

    Returns dict[facet -> bool ndarray] marking which entities in that facet
    have at least one non-zero entry in ANY currently-active relation. Union
    across relations, not any single one: e.g. a T1 article can have zero
    journal links but still be live via its author link, so no single
    relation is authoritative for "is this entity live".

    Must be computed fresh per call from the matrices actually loaded for the
    active config/slice — never hardcoded — so it stays correct as further
    time slices are added.
    """
    masks = {}

    for key in (soc_keys + sem_keys):
        if key not in matrices:
            continue

        f1, f2 = RELATION_MAP[key]
        mat = matrices[key]

        row_live = np.asarray(mat.getnnz(axis=1)).flatten() > 0
        col_live = np.asarray(mat.getnnz(axis=0)).flatten() > 0

        masks[f1] = row_live if f1 not in masks else (masks[f1] | row_live)
        masks[f2] = col_live if f2 not in masks else (masks[f2] | col_live)

    return masks
# =============================================================================
# CHUNK 13v9 - Module 2: THE PYTORCH INNER SOLVER (FINAL INTEGRATION)
# SCOPE: Tucker-adapted NNDSVD, Degree-Corrected Multi-Hop Phase 2 Propagation,
#        Direct Positivity via Post-Step Clamping (ticket 74 — softplus removed,
#        see CLAUDE.md §4.8), Forward-Pass Scale Invariance,
#        Relative Convergence, Squared Off-Diagonal Cohesion.
# =============================================================================

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

# -----------------------------------------------------------------------------
# 2.0 HELPERS & EXPLICIT TOPOLOGY MAP
# -----------------------------------------------------------------------------
def scipy_to_torch_sparse(scipy_mat):
    """Converts a scipy sparse matrix to a PyTorch sparse tensor."""
    coo = scipy_mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices), 
        torch.FloatTensor(values), 
        torch.Size(coo.shape)
    )

def inverse_softplus_np(x):
    """Numerically stable inverse softplus in pure numpy to preserve PyTorch leaf tensors."""
    x = np.clip(x, 1e-7, None)
    return x + np.log(1 - np.exp(-x))

def initialize_tucker_adapted_nndsvd_and_propagate(active_matrices, anchor_keys, dimensions, active_facets, K, device):
    """
    Executes Tucker-adapted Canonical NNDSVD, followed by multi-hop
    Degree-Corrected Phase 2 Matrix-Multiplication Propagation.
    Returns PyTorch leaf tensors, positive-valued, for Adam to optimize under
    v8-style direct positivity (ticket 74) — the caller clamps them to
    min=1e-7 after every optimizer.step().
    """
    U_np = {}
    Z_np = {}

    # 1. Base Latent Space (Pure random)
    for facet in active_facets:
        U_np[facet] = np.random.rand(dimensions[facet], K).astype(np.float32) + 1e-4
    for rel in active_matrices.keys():
        Z_np[rel] = np.random.rand(K, K).astype(np.float32) + 1e-4

    initialized_facets = set()

    # 2. Tucker-Adapted Canonical NNDSVD on Anchors
    # Ticket 64 (MULTI-ANCHOR OVERWRITE): every anchor in this pipeline shares
    # the same column facet ('art'). Running a separate SVD per anchor and
    # assigning U_np[f2] independently meant the second anchor silently
    # overwrote the first's article embedding, leaving the first anchor's row
    # facet initialized against an article space it no longer matched.
    # Restored the v7 approach: vstack all active anchors (already
    # Frobenius-normalized, so the joint stack is valid) into one matrix,
    # run a single joint SVD, then slice the left singular vectors back out
    # per anchor by row offset and assign the shared right singular vectors
    # to the common column facet exactly once.
    active_anchor_keys = [key for key in anchor_keys if key in active_matrices]
    if active_anchor_keys:
        anchor_scipy_mats = []
        anchor_f1s = []
        row_offsets = [0]
        f2_shared = None

        for key in active_anchor_keys:
            f1, f2 = RELATION_MAP[key]
            if f2_shared is None:
                f2_shared = f2
            assert f2 == f2_shared, (
                f"Anchor '{key}' column facet '{f2}' does not match shared "
                f"anchor facet '{f2_shared}' — joint anchor SVD requires all "
                f"active anchors to share one column space."
            )
            mat = active_matrices[key].cpu().coalesce()

            scipy_mat = sp.coo_matrix(
                (mat.values().numpy(), (mat.indices()[0].numpy(), mat.indices()[1].numpy())),
                shape=mat.size()
            ).tocsc()

            anchor_scipy_mats.append(scipy_mat)
            anchor_f1s.append(f1)
            row_offsets.append(row_offsets[-1] + scipy_mat.shape[0])

        joint_mat = sp.vstack(anchor_scipy_mats).tocsc()

        k_svd = min(K, min(joint_mat.shape) - 1)
        if k_svd > 0:
            # Correct absolute namespace for SVDS
            u, s, vt = scipy.sparse.linalg.svds(joint_mat, k=k_svd)

            # REVERSE to Descending Order
            idx = np.argsort(s)[::-1]
            s, u, vt = s[idx], u[:, idx], vt[idx, :]

            # PADDING if matrix rank < K
            if k_svd < K:
                pad_k = K - k_svd
                u = np.hstack([u, np.random.rand(u.shape[0], pad_k) * 0.01])
                vt = np.vstack([vt, np.random.rand(pad_k, vt.shape[1]) * 0.01])
                s = np.concatenate([s, np.random.rand(pad_k) * 0.01])

            # Canonical NNDSVD Algorithm (Boutsidis et al., 2008)
            U_new, V_new, S_new = np.zeros_like(u), np.zeros_like(vt.T), np.zeros_like(s)

            for j in range(K):
                uj, vj = u[:, j], vt[j, :]

                up, un = np.maximum(uj, 0), np.maximum(-uj, 0)
                vp, vn = np.maximum(vj, 0), np.maximum(-vj, 0)

                np_norm = np.linalg.norm(up) * np.linalg.norm(vp)
                nn_norm = np.linalg.norm(un) * np.linalg.norm(vn)

                if np_norm >= nn_norm:
                    u_res = up / np.linalg.norm(up) if np.linalg.norm(up) > 0 else up
                    v_res = vp / np.linalg.norm(vp) if np.linalg.norm(vp) > 0 else vp
                    s_res = s[j] * np_norm
                else:
                    u_res = un / np.linalg.norm(un) if np.linalg.norm(un) > 0 else un
                    v_res = vn / np.linalg.norm(vn) if np.linalg.norm(vn) > 0 else vn
                    s_res = s[j] * nn_norm

                U_new[:, j], V_new[:, j], S_new[j] = u_res, v_res, s_res

            # Slice the joint left singular vectors back out per anchor, and
            # assign the shared right singular vectors to the common column
            # facet exactly once.
            for i, key in enumerate(active_anchor_keys):
                f1 = anchor_f1s[i]
                row_start, row_end = row_offsets[i], row_offsets[i + 1]
                U_np[f1] = U_new[row_start:row_end, :].astype(np.float32) + 1e-4
                Z_np[key] = np.diag(S_new).astype(np.float32) + 1e-4
                initialized_facets.add(f1)

            U_np[f2_shared] = V_new.astype(np.float32) + 1e-4
            initialized_facets.add(f2_shared)

    # 3. Degree-Corrected Multi-Hop Phase 2 Propagation
    uninitialized = active_facets - initialized_facets
    while uninitialized:
        progress = False
        for key, mat in active_matrices.items():
            if key not in anchor_keys:
                f1, f2 = RELATION_MAP[key]
                
                if f1 in uninitialized and f2 in initialized_facets:
                    mat_coo = mat.cpu().coalesce()
                    M_scipy = sp.coo_matrix((mat_coo.values().numpy(), (mat_coo.indices()[0].numpy(), mat_coo.indices()[1].numpy())), shape=mat_coo.size())
                    
                    # Row-Normalize M to prevent Hub Dominance
                    row_sums = np.array(M_scipy.sum(axis=1)).flatten()
                    D_inv = np.zeros_like(row_sums)
                    D_inv[row_sums > 0] = 1.0 / row_sums[row_sums > 0]
                    M_row_norm = sp.diags(D_inv).dot(M_scipy)
                    
                    projected = M_row_norm.dot(U_np[f2])
                    norms = np.linalg.norm(projected, axis=0, keepdims=True)
                    norms[norms == 0] = 1.0
                    U_np[f1] = (projected / norms) + 1e-4
                    
                    uninitialized.remove(f1)
                    initialized_facets.add(f1) # Register for the next hop
                    progress = True
                    
                elif f2 in uninitialized and f1 in initialized_facets:
                    mat_coo = mat.cpu().coalesce()
                    M_scipy = sp.coo_matrix((mat_coo.values().numpy(), (mat_coo.indices()[0].numpy(), mat_coo.indices()[1].numpy())), shape=mat_coo.size())
                    
                    # Column-Normalize M (Row-Normalize M.T)
                    col_sums = np.array(M_scipy.sum(axis=0)).flatten()
                    D_inv = np.zeros_like(col_sums)
                    D_inv[col_sums > 0] = 1.0 / col_sums[col_sums > 0]
                    M_T_row_norm = sp.diags(D_inv).dot(M_scipy.T)
                    
                    projected = M_T_row_norm.dot(U_np[f1])
                    norms = np.linalg.norm(projected, axis=0, keepdims=True)
                    norms[norms == 0] = 1.0
                    U_np[f2] = (projected / norms) + 1e-4
                    
                    uninitialized.remove(f2)
                    initialized_facets.add(f2) # Register for the next hop
                    progress = True
        
        # Break if disconnected components exist to prevent infinite loop
        if not progress:
            break

    # 3.5 NNDSVDar-style stabilization noise (ticket 74)
    # Restored from v7's initialize_sequential_svd (the "ar" in NNDSVDar):
    # NNDSVD's non-negative projection zeros out roughly half of every column
    # by construction, and Phase 2 leaves disconnected periphery entities at
    # the raw +1e-4 floor. Lifts every entity off that floor by an amount
    # scaled to its own facet's mean, avoiding literal ties/degenerate zeros
    # in the SVD-derived structure and the multi-hop propagation. Verified via
    # isolation test to be independent of the softplus-vs-clamp question
    # below — kept regardless of which positivity scheme is in use.
    for f, U in U_np.items():
        avg = np.mean(U)
        noise = np.random.rand(*U.shape).astype(np.float32) * (avg / 10.0 + 1e-4)
        U_np[f] = U + noise

    # 4. Conversion to Trainable Leaf Tensors (v8-style direct positivity)
    # Ticket 74: softplus reparameterization removed (see CLAUDE.md §4.8).
    # softplus's gradient, sigmoid(raw), is < 1 everywhere and vanishingly
    # small for the small values this pipeline's NNDSVD/propagation init
    # produces — that attenuation dominated whatever momentum-coherence
    # benefit motivated softplus, and measurably prevented the modelled
    # community structure from ever differentiating beyond initialization
    # noise. Parameters now live directly in the positive space they
    # represent; run_inner_solver clamps them to min=1e-7 after every
    # optimizer.step(), matching v7/v8.
    U_raw = {f: torch.tensor(U_np[f], device=device, dtype=torch.float32, requires_grad=True) for f in active_facets}
    Z_raw = {rel: torch.tensor(Z_np[rel], device=device, dtype=torch.float32, requires_grad=True) for rel in active_matrices.keys()}

    return U_raw, Z_raw

# -----------------------------------------------------------------------------
# 2.2 THE SOLVER FUNCTION (API V2) updated 29.07.2026 20:52
# -----------------------------------------------------------------------------
def run_inner_solver(
    raw_data, 
    soc_keys, 
    sem_keys, 
    anchor_keys, 
    K,                   # [API UPGRADE] Explicit K passed from caller
    dimensions,  
    params,              # [API UPGRADE] Renamed from hyperparams
    device=torch.device("cpu"),
    seed_function=None,  # [API UPGRADE] Accepts the determinism lock
    inner_epochs=INNER_EPOCHS,  # ticket 72: was a hardcoded 400 that coincidentally
    learning_rate=LEARNING_RATE  # matched the (previously dead) module constants; now driven by them
):
    # Enforce absolute determinism inside the mathematical engine if requested
    if seed_function is not None:
        seed_function()

    # Ticket 78 removed lambda_l1 from trial.suggest_float (fixed at 0.0, recorded
    # only via trial.set_user_attr, which does NOT enter trial.params). The §S4
    # archiver and §S5 stability loop both pass trial.params/its serialized form
    # straight through as `params`, so a hard params['lambda_l1'] lookup raised
    # KeyError on those two of three call sites -- the Optuna objective itself
    # never hit this because it builds its own dict with 'lambda_l1' explicit
    # (see create_optuna_objective). Found while mapping this file for the
    # domain-balance design session; not a domain-balance change. Default 0.0
    # matches ticket 78's fixed value.
    lambda_l1 = params.get('lambda_l1', 0.0)
    lambda_z_offdiag = params.get('lambda_z_offdiag', 1.0) # Tuned by Optuna

    # Ticket 82 E2 (in-loop domain-balance term, D1-D3 design; Path B chosen
    # over Path A -- see CLAUDE.md §4.18 update / FINDINGS §22). NOT
    # Optuna-tunable, matching lambda_l1's own precedent (ticket 78): tested
    # at weight in {0.5,1,2,5,20,50,200,400} x power in {2,4}, TOL=0.15,
    # across 2 configs x 2 K x 2 data conditions (chunk12/chunk12v2) x
    # multiple seeds -- only 1 of 20 (weight,power,cell) combinations tested
    # produced a shift exceeding its own seed-noise floor, and weight=5
    # caused real non-convergence in several cells (down to 1/5 seeds
    # converging in one). Default 0.0 -- mechanism fully wired below (not
    # stubbed) so it is ready to reactivate if 22k-scale evidence supports a
    # nonzero value, exactly the lambda_l1/ticket-78 pattern.
    lambda_domain_balance = params.get('lambda_domain_balance', 0.0)

    active_matrices = {}
    active_facets = set()
    empty_relations = set()  # ticket 61: zero-norm relations, see loss loop below

    for key in (soc_keys + sem_keys):
        if key in raw_data:
            mat = scipy_to_torch_sparse(raw_data[key]).to(device).coalesce()

            # FROBENIUS NORMALIZATION WITH NaN GUARD
            fro_norm = torch.norm(mat.values(), p=2)
            normalized_values = mat.values() / fro_norm if fro_norm > 0 else mat.values()
            if fro_norm <= 0:
                empty_relations.add(key)
                print(f"[!] WARNING: relation {key} has zero norm (empty matrix) — "
                      f"target reconstruction norm forced to 0.0 instead of the "
                      f"usual unit-norm assumption.")

            active_matrices[key] = torch.sparse_coo_tensor(mat.indices(), normalized_values, mat.size(), device=device)
            f1, f2 = RELATION_MAP[key]
            active_facets.update([f1, f2])
        else:
            print(f"[!] WARNING: relation {key} absent from payload — excluded from factorization")

    w_soc = 0.5 / max(len([k for k in soc_keys if k in active_matrices]), 1)
    w_sem = 0.5 / max(len([k for k in sem_keys if k in active_matrices]), 1)

    # Ticket 82 E2 -- static topology for the in-loop domain-balance term,
    # computed ONCE here (read-only, not a sociological computation -- §4.1's
    # Epistemic Boundary is about penalties, not about reading which entities
    # are live, which build_presence_masks already does for other purposes
    # in this file). Independent of the identically-named computation
    # evaluate_domain_balance (Module 3) does post-hoc on the final U_prob --
    # deliberately not shared state between the two, matching how §4.1 keeps
    # Module 2 and Module 3 separate.
    db_presence_masks = build_presence_masks(raw_data, soc_keys, sem_keys)
    db_live_counts = {f: int(m.sum()) for f, m in db_presence_masks.items()}
    db_mask_t = {f: torch.from_numpy(m).to(device) for f, m in db_presence_masks.items()}
    db_soc_facets = sorted(f for f in db_live_counts if FACET_DOMAIN.get(f) == 'social'
                            and db_live_counts[f] > 0 and f in active_facets)
    db_sem_facets = sorted(f for f in db_live_counts if FACET_DOMAIN.get(f) == 'semantic'
                            and db_live_counts[f] > 0 and f in active_facets)
    db_soc_total = sum(db_live_counts[f] for f in db_soc_facets)
    db_sem_total = sum(db_live_counts[f] for f in db_sem_facets)

    def _db_domain_share(U_norm_dict, facets, w_total):
        """Entity-count-weighted mean, over `facets`, of each facet's own
        live-entity-masked mean U_prob row -- computed here from U_norm
        directly (L1 row-normalize inline) rather than calling
        compute_probability_distributions, which only accepts numpy and
        would break autograd. Matches E1's own methodology
        (facet_membership_profile + domain_balance_r_k(weighting='entity'))
        exactly, just differentiable."""
        if not facets or w_total == 0:
            return None
        acc = None
        for f in facets:
            if f not in U_norm_dict:
                continue
            live = U_norm_dict[f][db_mask_t[f]]
            if live.shape[0] == 0:
                continue
            row_sums = torch.sum(torch.abs(live), dim=1, keepdim=True) + 1e-12
            u_prob_live = live / row_sums
            term = db_live_counts[f] * u_prob_live.mean(dim=0)
            acc = term if acc is None else acc + term
        return None if acc is None else acc / w_total

    U_raw, Z_raw = initialize_tucker_adapted_nndsvd_and_propagate(active_matrices, anchor_keys, dimensions, active_facets, K, device)

    # v8-style direct positivity (ticket 74): enforce the floor at the start
    # too, not just after each step — matches the clamp regime continuously.
    with torch.no_grad():
        for u in U_raw.values():
            u.clamp_(min=1e-7)
        for z in Z_raw.values():
            z.clamp_(min=1e-7)

    optimizer = torch.optim.Adam(list(U_raw.values()) + list(Z_raw.values()), lr=learning_rate)
    loss_history = []
    
    eye_K = torch.eye(K, device=device)
    
    # Initialize variables to prevent UnboundLocalError
    pure_recon_loss_val = 0.0  
    U_norm = {}
    Z_scaled = {}
    U_scales = {}
    
    # [API UPGRADE] Convergence tracker
    converged = False 
    
    for epoch in range(inner_epochs):
        optimizer.zero_grad()
        
        # 1. DIRECT POSITIVITY (ticket 74 — softplus removed, see CLAUDE.md §4.8)
        # U_raw/Z_raw ARE the positive quantities; positivity is enforced by
        # clamping them to min=1e-7 after each optimizer.step() below (v8
        # scheme), not by a differentiable reparameterization.
        U_pos = U_raw
        Z_pos = Z_raw

        # 2. FORWARD-PASS SCALE-INVARIANCE NORMALIZATION
        U_norm = {}
        U_scales = {}
        for f, u_mat in U_pos.items():
            col_norms = torch.norm(u_mat, p=2, dim=0, keepdim=True)
            U_scales[f] = torch.clamp(col_norms, min=1e-9)
            U_norm[f] = u_mat / U_scales[f]

        Z_scaled = {}
        for rel_key, z_core in Z_pos.items():
            f1, f2 = RELATION_MAP[rel_key]
            scale_matrix = U_scales[f1].t() @ U_scales[f2]
            Z_scaled[rel_key] = z_core * scale_matrix

        # Ticket 82 E2 -- in-loop domain-balance term, computed from U_norm
        # (differentiable) every epoch. weight (lambda_domain_balance)
        # defaults to 0.0 -- see the comment at its read-in above.
        db_soc_share = _db_domain_share(U_norm, db_soc_facets, db_soc_total)
        db_sem_share = _db_domain_share(U_norm, db_sem_facets, db_sem_total)
        if db_soc_share is not None and db_sem_share is not None:
            db_r_k = db_soc_share / (db_soc_share + db_sem_share + 1e-12)
            db_excess = torch.clamp(torch.abs(db_r_k - 0.5) - DOMAIN_BALANCE_TOL, min=0.0)
            domain_balance_loss = torch.mean(db_excess ** 2)
        else:
            domain_balance_loss = torch.tensor(0.0, device=device)

        recon_loss = 0.0
        sparsity_loss = 0.0
        z_offdiag_loss = 0.0

        # A. Mathematical Fidelity
        for rel_key, M_sparse in active_matrices.items():
            f1, f2 = RELATION_MAP[rel_key]
            u1, u2, z_core = U_norm[f1], U_norm[f2], Z_scaled[rel_key]
            
            alpha_weight = w_soc if rel_key in soc_keys else w_sem
            
            M_v = torch.sparse.mm(M_sparse, u2)
            u_z = torch.matmul(u1, z_core)
            trace_cross = torch.sum(M_v * u_z)
            
            c_u = torch.matmul(u1.t(), u1)
            c_v = torch.matmul(u2.t(), u2)
            trace_pred = torch.sum((c_u @ z_core) * (z_core @ c_v))

            # Ticket 61: ||X||^2 is only 1.0 because X was Frobenius-normalized
            # to unit norm; an empty relation (fro_norm == 0) was never
            # normalized and truly has ||X||^2 == 0. Hardcoding 1.0 here gave
            # empty relations a permanent, unremovable loss floor.
            target_norm_sq = 0.0 if rel_key in empty_relations else 1.0
            relation_recon = torch.clamp(target_norm_sq - 2.0 * trace_cross + trace_pred, min=0.0)
            recon_loss += alpha_weight * relation_recon
            
            # SQUARED Z Off-Diagonal Cohesion Penalty
            z_offdiag_loss += torch.sum((z_core * (1.0 - eye_K)) ** 2)

        # B. Sociological Reality (Standard L1 on Scale-Identifiable Space)
        # Ticket 73: MEAN over all U_norm entries, not a raw sum. U_norm's
        # columns are unit-L2, so sum(u_mat) is structurally ~sqrt(N) per
        # column regardless of what the model learned; summed across ~9
        # facets this landed at O(100), swamping recon_loss (O(1)) and making
        # most of Optuna's lambda_l1 search range either inert or wildly
        # disproportionate. Matches v8's l1_penalty / (global_dims[f] * K)
        # normalization in spirit.
        num_sparsity_entries = 0
        for facet, u_mat in U_norm.items():
            sparsity_loss += torch.sum(u_mat)
            num_sparsity_entries += u_mat.numel()
        sparsity_loss = sparsity_loss / num_sparsity_entries

        # Capture un-penalized metric for Optuna
        pure_recon_loss_val = recon_loss.item() if hasattr(recon_loss, 'item') else recon_loss
        
        # Current Loss Aggregation
        total_loss = (recon_loss + (lambda_l1 * sparsity_loss) + (lambda_z_offdiag * z_offdiag_loss)
                      + (lambda_domain_balance * domain_balance_loss))
        total_loss.backward()
        
        optimizer.step()

        # v8-style direct positivity (ticket 74): hard clamp after each step.
        with torch.no_grad():
            for u in U_raw.values():
                u.clamp_(min=1e-7)
            for z in Z_raw.values():
                z.clamp_(min=1e-7)

        loss_value = total_loss.item()
        loss_history.append(loss_value)
        
        # C. RELATIVE EARLY STOPPING CONVERGENCE CHECK
        if epoch >= 20:
            prev_loss = loss_history[-21]
            if prev_loss > 0:
                rel_change = abs(loss_history[-1] - prev_loss) / prev_loss
                if rel_change < 1e-4:
                    converged = True
                    break

    # Safely detach and move final matrices to CPU
    # Wait to cast to numpy until returned, or keep as torch tensors based on your pipeline 
    # (Module 4 accepts torch tensors and saves them via torch.save)
    U_final = {f: U_norm[f].detach() for f in active_facets}
    Z_final = {rel: Z_scaled[rel].detach() for rel in Z_scaled.keys()}
    
    # Safely format and export the scale masses for the Collapse Check
    U_scales_out = {f: scale.squeeze().detach().cpu().numpy() for f, scale in U_scales.items()}

    # [API UPGRADE] The Clean Diagnostics Dictionary
    diagnostics = {
        "math_loss": pure_recon_loss_val,
        "internal_soc_loss": float((lambda_l1 * sparsity_loss) + (lambda_z_offdiag * z_offdiag_loss)
                                    + (lambda_domain_balance * domain_balance_loss)),
        # Ticket 78: raw, UNWEIGHTED sparsity_loss, kept observable even when
        # lambda_l1=0 zeroes its contribution to internal_soc_loss above.
        "raw_sparsity_loss": float(sparsity_loss.item() if hasattr(sparsity_loss, "item") else sparsity_loss),
        # Ticket 82 E2: raw, UNWEIGHTED in-loop domain-balance penalty, kept
        # observable even when lambda_domain_balance=0 zeroes its
        # contribution to internal_soc_loss above -- same pattern as
        # raw_sparsity_loss.
        "raw_domain_balance_penalty": float(domain_balance_loss.item()
                                             if hasattr(domain_balance_loss, "item") else domain_balance_loss),
        "loss_history": loss_history,
        "U_scales": U_scales_out,
        "converged": converged
    }

    return U_final, Z_final, diagnostics
# =============================================================================
# MODULE 3: META-EVALUATOR (OUTER LOOP DIAGNOSTICS)
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: Probability Mapping
# -----------------------------------------------------------------------------
import numpy as np

def compute_probability_distributions(U_final):
    """
    Converts L2-normalized factor matrices into L1 row-normalized probability distributions.
    Allows for the Bourdieusian interpretation: "X% of this actor's capital is in Community K".
    
    Args:
        U_final (dict): Dictionary of facet embeddings {facet_name: ndarray of shape (N, K)}
        
    Returns:
        U_prob (dict): Dictionary of L1 row-normalized probability matrices.
    """
    U_prob = {}
    
    for facet, u_mat in U_final.items():
        # Calculate the L1 norm of each row (sum of absolute values)
        row_sums = np.sum(np.abs(u_mat), axis=1, keepdims=True)
        
        # Apply epsilon guard to prevent NaN division for completely isolated actors
        row_sums_guarded = row_sums + 1e-12
        
        # Row normalization
        U_prob[facet] = u_mat / row_sums_guarded
        
    return U_prob

# -----------------------------------------------------------------------------
# SECTION 2: Config-Invariant Dimensional Collapse Check
# -----------------------------------------------------------------------------
# REWRITTEN (tickets 79/80/82; FINDINGS §12/13/16). The two helpers below
# restore, with a revised criterion, a mechanism (identify_leaf_nodes /
# diagnose_leaf_Z / correct_all_leaf_nodes) that existed in chunk13v3.py and
# chunk13v4.py and was lost in a later rewrite — see evaluate_dimensional_
# collapse's docstring below for the criterion this version uses and why.
from scipy.optimize import linear_sum_assignment

def _hungarian_relabel_relation(Z, structure_threshold):
    """
    Per-relation permutation-correction test (ticket 82, FINDINGS §16).
    Extracted as its own function — not inlined — so it can be unit-tested
    and audited independently of the full evaluation pipeline.

    Hungarian-matches Z's rows to columns (scipy.optimize.linear_sum_assignment)
    to find the diagonal-maximizing permutation. structure_score =
    min(matched)/max(unmatched): >1 means EVERY within-community entry beats
    EVERY cross-community entry under this permutation — a confidently clean
    relabelling, safe to apply. <=1 means at least one cross-community entry
    is as large as some within-community entry: genuine mixed coupling (e.g.
    an article-author tie that really does bridge communities), not a
    labelling artifact — do NOT correct it. Forcing a clean reading onto a
    genuinely mixed relation would erase real signal, not fix an error.

    Verified head-to-head against chunk13v3.py/v4.py's original criterion
    (diagonal_mass/hungarian_mass < 0.7): 12 of 94 relation-instances
    disagreed on this corpus, 8 of them cases where the older criterion would
    have "corrected" a relation independently identified elsewhere (FINDINGS
    §2, §14, §16) as genuinely mixed. This criterion was kept for that reason.

    structure_threshold is TOY-CORPUS CALIBRATED — see
    STRUCTURE_SCORE_THRESHOLD's definition. Re-derive before 22k-article use.

    Returns (permutation: list[int], trusted: bool).
    """
    K = Z.shape[0]
    absZ = np.abs(Z)
    total_mass = absZ.sum()
    if total_mass <= 1e-15:
        return list(range(K)), False

    _, col_ind = linear_sum_assignment(-absZ)
    perm = col_ind.tolist()
    matched = np.array([absZ[i, perm[i]] for i in range(K)])

    mask = np.ones_like(absZ, dtype=bool)
    for i in range(K):
        mask[i, perm[i]] = False
    unmatched = absZ[mask]
    max_unmatched = unmatched.max() if unmatched.size else 0.0
    structure_score = matched.min() / max_unmatched if max_unmatched > 1e-15 else float("inf")

    return perm, structure_score > structure_threshold


def _relation_community_share(Z, U_final_f1, U_final_f2, structure_threshold):
    """
    Reconstruction-space per-community share for one relation (FINDINGS §8,
    originating with a Gemini-based review; tickets 79/82).

    share_k = within_k / total, where within_k is community k's within-
    community reconstructed mass and total = ||U1.Z.U2^T||^2_F is the
    relation's ENTIRE reconstructed mass (diagonal and off-diagonal). Because
    U_norm's columns are unit-L2-norm by construction (CLAUDE.md §4.3),
    within_k reduces algebraically to Z[k, pi(k)]^2 exactly (verified, not
    assumed) — pi is the identity unless the correction above is trusted.

    Chosen over the simpler diagonal-sum share (share_k = mass_k/sum(mass))
    because that version structurally cannot see how much of a relation's
    real signal is off-diagonal — it always forces the K diagonal entries to
    sum to 1, whether they represent 5% or 95% of what the relation actually
    reconstructs. Reconstruction-space share does not have this blind spot: a
    relation with high interference (most of its mass genuinely
    cross-community) contributes correspondingly little to any single
    community's share, without a separate weighting scheme layered on top.
    Verified this changes a real verdict on this corpus (C6/K=2: fires under
    diagonal-sum share, does not under reconstruction-space, traced to a
    relation with interference=0.83 — 83% of its reconstructed mass is
    genuinely cross-community).
    """
    K = Z.shape[0]
    perm, trusted = _hungarian_relabel_relation(Z, structure_threshold)
    p = perm if trusted else list(range(K))

    c_u = U_final_f1.T @ U_final_f1
    c_v = U_final_f2.T @ U_final_f2
    total = float(np.trace(Z.T @ c_u @ Z @ c_v))

    within = np.array([Z[k, p[k]] ** 2 for k in range(K)])
    return within / total if total > 1e-15 else np.full(K, 1.0 / K)


def evaluate_dimensional_collapse(U_final, Z_final, max_share_threshold=MAX_SHARE_THRESHOLD,
                                   structure_threshold=STRUCTURE_SCORE_THRESHOLD):
    """
    REVISED (tickets 79/80/82; FINDINGS §12/13/16). Calculates community mass
    from Z_scaled (relation-level, permutation-corrected), NOT U_scales —
    U_scales is proven an undetermined free gauge direction in the loss, not
    a mass measure (ticket 79, FINDINGS §12): scaling a U_pos column by any
    constant c and compensating in the touching Z_pos entries leaves
    recon_loss/sparsity_loss/z_offdiag_loss/U_norm/Z_scaled all unchanged to
    float precision while U_scales changes by exactly c.

    Penalizes the model if reconstruction-space mass collapses onto a single
    community, using a direct max-share penalty (ticket 80) rather than
    normalized Shannon entropy — entropy was not a monotone function of
    max-share and its correspondence to a fixed threshold was K-dependent (a
    60%-dominant 2-community split reads entropy~=0.97, nowhere near a 0.60
    floor; see CLAUDE.md §4.4 for the full derivation).

    No presence_masks / live-entity normalisation needed here (contrast the
    old U_scales-based version) — every relation's input is already
    Frobenius-normalised to ||X||^2=1 before fitting, which is what makes
    relation-level mass comparable across relations without a separate
    facet-size correction.

    Aggregation across relations is EQUAL WEIGHT (every active relation
    counts once). Alternatives tested and rejected: nnz-weighting revives the
    scaling problem Frobenius normalisation was built to remove;
    structure_score-weighting contradicts its own use as the correction gate
    above (it would penalise genuinely mixed relations twice — once by not
    correcting them, again by down-weighting their vote); reconstruction-
    quality-weighting (a relation's own recon_loss) tied with equal-weight
    everywhere tested, at the cost of duplicating Module 2's loss computation
    inside the evaluator. Reconstruction-space share's own interference term
    already provides a form of down-weighting for unreliable relations, which
    is why a second explicit weighting layer was not added.

    Non-leaf cross-relation conflicts: this function corrects each relation
    INDEPENDENTLY. It does NOT resolve disagreement between multiple
    relations sharing a non-leaf facet. Empirically, zero such conflicts were
    found anywhere in the C1-C6 x K in {2,4} grid once leaf-exclusion and
    confidence-filtering were combined correctly (ticket 82's register entry
    corrects an earlier, buggy claim of one such conflict). If a genuine
    non-leaf conflict is ever found (plausible at 22k-article scale, where a
    denser topology could produce one), this function will NOT detect or
    resolve it — a known limitation, not silently assumed away.

    Args:
        U_final (dict): facet -> ndarray (N, K). Used only for the
            reconstruction-space `total` term (c_u = U1^T U1 etc.), NOT for
            N_f/live-entity normalisation (not needed — see above).
        Z_final (dict): relation_key -> ndarray (K, K), Z_scaled.
        max_share_threshold (float): ceiling on the largest community's
            share of reconstruction-space mass. Default is the SAME NUMERIC
            VALUE as the old ENTROPY_THRESHOLD (0.60) — CLAUDE.md §4.4
            already established that value as the intended target ("no
            community >0.6 of mass"); this reuses the target, not a new pick.
        structure_threshold (float): TOY-CORPUS CALIBRATED — see
            STRUCTURE_SCORE_THRESHOLD. Re-derive before 22k-article use.

    Returns:
        collapse_pen (float): penalty in [0.0, 1.0], same squared ceiling
            shape as evaluate_socio_semantic_reality's MAX_MONOPOLY penalty
            (this file, Part B) — ported, not reinvented.
        max_share (float): the actual max community share (for logging).
            This is what the "collapse_score" key in
            evaluate_complete_solution's returned dict now means — the KEY
            NAME is unchanged for contract compatibility (CLAUDE.md §3), its
            MEANING is not (was normalized_entropy).
    """
    if not Z_final:
        return 0.0, 1.0

    K = next(iter(Z_final.values())).shape[0]
    if K < 2:
        return 0.0, 1.0

    relation_shares = []
    for rel_key, Z in Z_final.items():
        f1, f2 = RELATION_MAP[rel_key]
        share = _relation_community_share(Z, U_final[f1], U_final[f2], structure_threshold)
        relation_shares.append(share)

    community_share = np.mean(relation_shares, axis=0)
    total_share = community_share.sum()
    community_share = community_share / total_share if total_share > 1e-15 else np.full(K, 1.0 / K)

    max_share = float(community_share.max())
    collapse_pen = (max(0.0, max_share - max_share_threshold) / (1.0 - max_share_threshold)) ** 2

    # Cast to native Python float — numpy scalars aren't JSON-serializable
    # via stdlib json, which is what Optuna's trial.set_user_attr(...) uses
    # under the hood (ticket 68).
    return float(collapse_pen), float(max_share)
# -----------------------------------------------------------------------------
# SECTION 3: Topological Coherence Diagnostic (The Mean Anchor Rule)
# -----------------------------------------------------------------------------
def evaluate_topological_coherence(Z_final, target_coherence, semantic_anchors):
    """
    Evaluates the structural cohesion of communities by examining the interaction 
    density (main diagonal) of the core tensor Z for semantic-article anchors.
    
    Modeling Choice: Uses the "Mean Anchor Rule" to allow for polycentric 
    communities that might be crisp at one semantic resolution (e.g., M_Child_Art) 
    but noisier at another (e.g., M_Cousin_Art).
    
    Args:
        Z_final (dict): The scale-absorbed core matrices from Module 2.
        target_coherence (float): Domain constant (e.g., 0.50).
        
    Returns:
        coherence_pen (float): Normalized penalty [0.0 - 1.0].
        weakest_mean (float): The lowest community mean cohesion (for logging).
    """
    # Filter to evaluate ONLY the anchors active in this specific config
    active_anchors = [rel for rel in Z_final.keys() if rel in semantic_anchors]
    
    # Honest Telemetry Guard (Returns np.nan if no semantic anchors are active)
    if not active_anchors:
        return 0.0, np.nan 

    # Extract K dynamically from the first available anchor matrix
    first_anchor = active_anchors[0]
    K = Z_final[first_anchor].shape[0]
    
    # =========================================================================
    # -> MICRO-PATCH 1: Core Shape Validation
    # Ensure all active core tensor slices match the expected (K, K) dimensions
    # =========================================================================
    for anchor in active_anchors:
        if Z_final[anchor].shape != (K, K):
            raise ValueError(
                f"Core matrix '{anchor}' shape {Z_final[anchor].shape} "
                f"does not match expected dimensions ({K}, {K})."
            )

    community_mean_scores = []
    
    # Calculate cohesion ratios per community
    for k in range(K):
        k_anchor_ratios = []
        
        for anchor in active_anchors:
            Z_mat = Z_final[anchor]
            
            diagonal_val = Z_mat[k, k]
            row_sum = np.sum(Z_mat[k, :])
            col_sum = np.sum(Z_mat[:, k])
            
            # Symmetric Cohesion Ratio with Epsilon Guard
            denominator = 0.5 * (row_sum + col_sum) + 1e-12
            raw_ratio = diagonal_val / denominator
            
            # =================================================================
            # -> MICRO-PATCH 2: Numerical Ratio Clipping
            # Protect against floating-point drift (e.g., 1.00000003)
            # =================================================================
            ratio = np.clip(raw_ratio, 0.0, 1.0)
            
            k_anchor_ratios.append(ratio)
            
        # Mean Cohesion Score across available anchors for Community k
        mean_cohesion_k = np.mean(k_anchor_ratios)
        community_mean_scores.append(mean_cohesion_k)
        
    # Weakest Community Bottleneck
    weakest_mean = np.min(community_mean_scores)
    
    # Continuous penalty [0.0 - 1.0]
    coherence_pen = (max(0.0, target_coherence - weakest_mean) / target_coherence)**2

    # Cast to native Python float — Z_final's numpy arrays keep the source
    # torch tensors' dtype (often float32), and numpy scalars aren't
    # JSON-serializable via stdlib json (Optuna's trial.set_user_attr(...)).
    # float(np.nan) stays nan and round-trips fine through json.dumps.
    return float(coherence_pen), float(weakest_mean)

import numpy as np
import scipy.sparse as sp

# -----------------------------------------------------------------------------
# SECTION 4: Unified Socio-Semantic Reality Check
# -----------------------------------------------------------------------------
def evaluate_socio_semantic_reality(U_prob, raw_data, active_matrices, max_monopoly, presence_masks=None):
    """
    Evaluates the continuous socio-semantic reality (Part A) and uniquity smoothing (Part B).

    Args:
        U_prob (dict): L1 row-normalized probability matrices.
        raw_data (dict): The original SciPy sparse matrices (assumes shape: Target x Source).
        active_matrices (list/set): The string keys of matrices active in the current config.
        max_monopoly (float): Domain constant (e.g., 0.60).
        presence_masks (dict, optional): Ticket 60 — per-facet bool mask of
            live entities (see build_presence_masks). Dead rows (entities that
            belong to another chunk12 time slice) sit near the init/clamp
            floor, and after L1 row-normalization that floor noise can look
            like an extreme single-community loading — a phantom hoarding
            elite that isn't real. When given, Part A's baseline and Part B's
            monopoly scan are restricted to live entities only.

    Returns:
        socio_semantic_pen (float): Normalized penalty [0.0 - 1.0].
        Penalty_A (float): Core semantic reality penalty.
        Penalty_B (float): Ubiquity smoothing penalty.
    """
    target_facets = ['core_child_he', 'cousin_he', 'core_atom', 'fringe_atom']
    
    # -> MICRO-PATCH A1: Guard against missing primary facet
    if 'art' not in U_prob:
        raise KeyError("Target facet 'art' is missing from U_prob. Check upstream data.")
        
    K = U_prob['art'].shape[1]
    # Guard against division by zero in entropy calculations if only 1 community exists
    if K < 2:
        return 1.0
    part_a_scores = []
    
    # =========================================================================
    # PART A: Continuous Semantic Relevance via Multi-Path Summation
    # =========================================================================
    # -> Accumulator for Part B (Dynamic Entropy)
    empirical_mass_capture = {facet: np.zeros((U_prob[facet].shape[0], K)) for facet in target_facets if facet in U_prob}

    for k in range(K):
        W_art = U_prob['art'][:, k]
        
        # Dead Community Guard
        # Ticket 25: only penalize facets actually present in this config —
        # previously appended 1.0 for all four target_facets unconditionally,
        # inflating Penalty_A for configs with fewer active facets.
        if np.sum(W_art) < 1e-12:
            for facet in target_facets:
                if facet in U_prob:
                    part_a_scores.append(1.0)
            continue
            
        propagated_weights = {}
        
       # 1. Level 1: Parents
        if 'M_Parent_Art' in active_matrices:
            # Runtime Shape Guard to prevent silent transposition bugs
            if raw_data['M_Parent_Art'].shape[1] != len(W_art):
                raise ValueError("M_Parent_Art orientation mismatch")
            propagated_weights['parent_he'] = raw_data['M_Parent_Art'].dot(W_art)
            
        # 2. Level 2: Children (Direct vs. Parent-Mediated Paths)
        child_paths = []
        if 'M_Child_Art' in active_matrices:
            if raw_data['M_Child_Art'].shape[1] != len(W_art):
                raise ValueError("M_Child_Art orientation mismatch")
            child_paths.append(raw_data['M_Child_Art'].dot(W_art))
            
        if 'M_Child_Parent' in active_matrices and 'parent_he' in propagated_weights:
            if raw_data['M_Child_Parent'].shape[1] != len(propagated_weights['parent_he']):
                raise ValueError("M_Child_Parent orientation mismatch")
            child_paths.append(raw_data['M_Child_Parent'].dot(propagated_weights['parent_he']))
            
        if child_paths:
            # Summation across valid topological paths (rewarding multi-path grounding)
            propagated_weights['core_child_he'] = sum(child_paths)
            
        # 3. Level 2: Cousins (Direct, Parent-Mediated, OR Child-Mediated via M_Cousin_Child)
        cousin_paths = []
        if 'M_Cousin_Art' in active_matrices:
            if raw_data['M_Cousin_Art'].shape[1] != len(W_art):
                raise ValueError("M_Cousin_Art orientation mismatch")
            cousin_paths.append(raw_data['M_Cousin_Art'].dot(W_art))
            
        if 'M_Cousin_Parent' in active_matrices and 'parent_he' in propagated_weights:
            if raw_data['M_Cousin_Parent'].shape[1] != len(propagated_weights['parent_he']):
                raise ValueError("M_Cousin_Parent orientation mismatch")
            cousin_paths.append(raw_data['M_Cousin_Parent'].dot(propagated_weights['parent_he']))
            
        if 'M_Cousin_Child' in active_matrices and 'core_child_he' in propagated_weights:
            # -> CRITICAL FIX: Explicitly supports C3 and C4 configurations!
            if raw_data['M_Cousin_Child'].shape[1] != len(propagated_weights['core_child_he']):
                raise ValueError("M_Cousin_Child orientation mismatch")
            cousin_paths.append(raw_data['M_Cousin_Child'].dot(propagated_weights['core_child_he']))
            
        if cousin_paths:
            propagated_weights['cousin_he'] = sum(cousin_paths)
            
        # 4. Level 3: Atoms
        if 'M_Atom_Child' in active_matrices and 'core_child_he' in propagated_weights:
            if raw_data['M_Atom_Child'].shape[1] != len(propagated_weights['core_child_he']):
                raise ValueError("M_Atom_Child orientation mismatch")
            propagated_weights['core_atom'] = raw_data['M_Atom_Child'].dot(propagated_weights['core_child_he'])
            
        if 'M_Fringe_Cousin' in active_matrices and 'cousin_he' in propagated_weights:
            if raw_data['M_Fringe_Cousin'].shape[1] != len(propagated_weights['cousin_he']):
                raise ValueError("M_Fringe_Cousin orientation mismatch")
            propagated_weights['fringe_atom'] = raw_data['M_Fringe_Cousin'].dot(propagated_weights['cousin_he'])
            
        # 5. Evaluate Alignment against Empirical Baseline
        for facet in target_facets:
            if facet in U_prob:
                if facet not in propagated_weights:
                    # If active configuration lacks weights for this facet, 
                    # it is a total structural failure for community k -> Assign maximum penalty (1.0)
                    part_a_scores.append(1.0)
                    continue
                    
                # -----------------------------------------------------------
                # THIS IS EXACTLY WHERE THE FIX GOES:
                # -----------------------------------------------------------
                W_facet = propagated_weights[facet]
                
                # -> MICRO-PATCH A2: Guard against dimensional misalignment (MOVED FIRST)
                if len(W_facet) != U_prob[facet].shape[0]:
                    raise ValueError(f"Dimension mismatch for facet '{facet}': "
                                     f"W_facet length {len(W_facet)} != U_prob length {U_prob[facet].shape[0]}")
                
                # -> NEW: Capture this community's empirical mass for Part B
                empirical_mass_capture[facet][:, k] = W_facet 
                
                total_w = np.sum(W_facet)
                
                if total_w < 1e-12:
                    # Zero-weight propagated vector means complete semantic disconnect -> Penalty = 1.0
                    part_a_scores.append(1.0)
                    continue
                # -----------------------------------------------------------
                    
                model_probs = U_prob[facet][:, k]
                # Ticket 60: baseline over live entities only — dead (padded)
                # rows would otherwise pull the empirical baseline down.
                if presence_masks is not None and facet in presence_masks and presence_masks[facet].any():
                    baseline = np.mean(model_probs[presence_masks[facet]])
                else:
                    baseline = np.mean(model_probs)
                weighted_avg = np.sum(W_facet * model_probs) / total_w
                
                if baseline < 1e-12:
                    score = 1.0
                else:
                    score = (max(0.0, baseline - weighted_avg) / baseline)**2
                    
                part_a_scores.append(score)

    # =========================================================================
    # PART B: Dynamic Doxa Hoarding (Empirical Entropy Check)
    # =========================================================================
    hoarding_penalties = []
    
    for facet in target_facets:
        if facet not in U_prob or facet not in empirical_mass_capture:
            continue
            
        # 1. Normalize empirical mass into a probability distribution across K communities
        E_mat = empirical_mass_capture[facet]
        row_sums = np.sum(E_mat, axis=1, keepdims=True)
        
        # Avoid division by zero for nodes with no empirical mass
        valid_mask = (row_sums[:, 0] > 1e-12)

        # Ticket 60: exclude dead (padded) rows — their init/clamp-floor values
        # survive L1 row-normalization as noise that can spike max_learned_loading
        # and register as a phantom hoarding elite.
        if presence_masks is not None and facet in presence_masks:
            valid_mask = valid_mask & presence_masks[facet]

        if not np.any(valid_mask):
            continue
            
        P_emp = np.zeros_like(E_mat)
        P_emp[valid_mask] = E_mat[valid_mask] / row_sums[valid_mask]
        
        # 2. Calculate Normalized Shannon Entropy (0.0 = Niche, 1.0 = Universal Doxa)
        # Add epsilon to prevent log(0)
        epsilon = 1e-12
        entropy = -np.sum(P_emp * np.log(P_emp + epsilon), axis=1)
        max_possible_entropy = np.log(K)
        normalized_entropy = entropy / max_possible_entropy
        
        # 3. Find the Model's maximum learned community loading for each node
        U_sem = U_prob[facet]
        max_learned_loading = np.max(U_sem, axis=1)
        
        # 4. Calculate Hoarding Penalty
        # Penalty triggers if max_learned_loading > max_monopoly
        # Smooth scaling: (loading - threshold) / (1 - threshold)
        base_penalty = (np.maximum(0.0, max_learned_loading - max_monopoly) / (1.0 - max_monopoly))**2
        
        # Multiply by empirical entropy: We ONLY care if high-entropy Doxa is hoarded.
        # Hoarding a niche term (entropy ~ 0) yields ~0 penalty.
        doxa_hoarding_penalty = base_penalty * normalized_entropy
        
        # Average the penalty across all VALID semantic nodes for this facet
        mean_facet_penalty = np.mean(doxa_hoarding_penalty[valid_mask])
        hoarding_penalties.append(mean_facet_penalty)
        
    Penalty_B = np.mean(hoarding_penalties) if hoarding_penalties else 0.0

    # =========================================================================
    # FINAL METRIC
    # =========================================================================
    Penalty_A = np.mean(part_a_scores) if part_a_scores else 1.0
    socio_semantic_pen = (Penalty_A + Penalty_B) / 2.0

    # Cast to native Python float — Penalty_A/Penalty_B are np.mean(...)
    # results (numpy float64) whenever their source lists are non-empty, and
    # numpy scalars aren't JSON-serializable via stdlib json (Optuna's
    # trial.set_user_attr(...)).
    return float(socio_semantic_pen)

import optuna
import torch
import numpy as np

# -----------------------------------------------------------------------------
# SECTION 4B: Per-Community Domain-Balance Diagnostic (ticket 82, E2 -- Path B)
# -----------------------------------------------------------------------------
def evaluate_domain_balance(U_prob, presence_masks, tol=DOMAIN_BALANCE_TOL):
    """
    Outer-loop, U_prob-based domain-balance check (ticket 82 E2, D1-D3
    design; CLAUDE.md §4.18 update / FINDINGS §22 has the full derivation
    and evidence).

    WHY U_prob, not Z_scaled: a Z_scaled-based version of this same idea was
    tested separately this session and found exploitable -- an unconstrained
    per-relation scaling freedom (ticket 79: scaling one U_pos column by any
    constant c and compensating in the touching Z_pos entries leaves
    recon_loss/U_norm/Z_scaled unchanged while U_scales moves by exactly c)
    let a Z_scaled-based reading "improve" 30-40x in its own terms while the
    real, U_prob-based membership imbalance got WORSE in 3 of 12 tested
    cells. U_prob is a pure function of U_norm, and U_norm is PROVEN
    algebraically invariant to that exact transformation -- so this reading
    cannot be gamed the same way, by construction.

    Per community k: soc_share_k / sem_share_k = entity-count-weighted mean
    (D2), over that domain's live-nonempty facets (FACET_DOMAIN), of each
    facet's own live-entity-masked (ticket 60) mean U_prob row.
    r_k = soc_share_k / (soc_share_k + sem_share_k + eps). Penalty:
    mean_k(max(0, |r_k-0.5| - tol)^2) -- same squared-ceiling-past-tolerance
    shape as every other penalty in this file (§4.5); tol defaults to
    DOMAIN_BALANCE_TOL = 0.15 (D3, FINDINGS §21), NOT Optuna-tunable (§4.7).

    Matches E1's own diagnostic methodology exactly
    (diagnostic_blocks.facet_membership_profile + domain_balance_r_k
    weighting='entity') and the in-loop term in run_inner_solver -- same
    formula, computed here post-hoc/non-differentiably on the FINAL U_prob
    rather than differentiably per-epoch on U_norm.

    Unlike collapse_pen/coherence_pen/semantic_pen, this check currently has
    NO active in-loop counterpart (lambda_domain_balance defaults to 0.0 in
    run_inner_solver -- see the comment at its read-in there for why) -- it
    still functions exactly like those three: a pure outer-loop reality
    check that differentiates trials on this axis regardless of whether
    anything in the training loss is pushing toward it. This is the SAME
    architecture collapse_pen/coherence_pen already use (neither has an
    in-loop counterpart either) -- not a new pattern.

    Args:
        U_prob (dict): facet -> ndarray (N, K), L1 row-normalized
            (compute_probability_distributions' output).
        presence_masks (dict): facet -> bool ndarray (ticket 60,
            build_presence_masks' output) -- caller-supplied so this
            function doesn't recompute it a second time when
            evaluate_complete_solution already has it.
        tol (float): dead-band half-width around r_k=0.5. Default
            DOMAIN_BALANCE_TOL.

    Returns:
        domain_balance_pen (float): penalty in [0.0, ~0.1225] (max possible:
            (0.5-tol)^2 when a community is 100% one domain).
        mean_dev_k (float): raw mean |r_k-0.5| across communities, for
            logging/observability -- mirrors collapse_score/
            weakest_coherence's role for the other two checks.
    """
    soc_facets = sorted(f for f in presence_masks
                         if FACET_DOMAIN.get(f) == 'social' and f in U_prob
                         and presence_masks[f].sum() > 0)
    sem_facets = sorted(f for f in presence_masks
                         if FACET_DOMAIN.get(f) == 'semantic' and f in U_prob
                         and presence_masks[f].sum() > 0)
    if not soc_facets or not sem_facets:
        return 0.0, 0.0

    def _weighted_share(facets):
        total_w = sum(int(presence_masks[f].sum()) for f in facets)
        if total_w == 0:
            return None
        acc = None
        for f in facets:
            mask = presence_masks[f]
            live = U_prob[f][mask]
            if live.shape[0] == 0:
                continue
            w = int(mask.sum())
            term = w * live.mean(axis=0)
            acc = term if acc is None else acc + term
        return None if acc is None else acc / total_w

    soc_share = _weighted_share(soc_facets)
    sem_share = _weighted_share(sem_facets)
    if soc_share is None or sem_share is None:
        return 0.0, 0.0

    eps = 1e-12
    r_k = soc_share / (soc_share + sem_share + eps)
    dev_k = np.abs(r_k - 0.5)
    excess = np.clip(dev_k - tol, a_min=0.0, a_max=None)
    domain_balance_pen = float(np.mean(excess ** 2))
    mean_dev_k = float(np.mean(dev_k))

    # Cast to native Python float — see ticket 68's JSON-serializability note.
    return domain_balance_pen, mean_dev_k


# -----------------------------------------------------------------------------
# SECTION 5: The Optuna Objective Aggregator (The Wrapper)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5.1 UNIFIED SOCIOLOGICAL EVALUATION PIPELINE
# -----------------------------------------------------------------------------
def evaluate_complete_solution(
    U_final,
    Z_final,
    U_scales_out,
    raw_data,
    soc_keys,
    sem_keys,
    anchor_keys,
    max_monopoly,
    entropy_threshold,
    target_coherence
):
    """
    Computes every outer-loop sociological metric from a solved model.
    This function is the single source of truth for sociological evaluation.
    """

    # Convert tensors -> numpy if necessary
    U_numpy = {
        k: v.detach().cpu().numpy() if hasattr(v, "detach") else v
        for k, v in U_final.items()
    }

    Z_numpy = {
        k: v.detach().cpu().numpy() if hasattr(v, "detach") else v
        for k, v in Z_final.items()
    }

    active_matrices = {
        k for k in (soc_keys + sem_keys)
        if k in raw_data
    }

    # Ticket 60: recomputed fresh from the matrices actually loaded for this
    # config/slice — see build_presence_masks (§1.8).
    presence_masks = build_presence_masks(raw_data, soc_keys, sem_keys)

    U_prob = compute_probability_distributions(U_numpy)

    # Section 2 — REVISED (tickets 79/80/82): Z_scaled-based, not U_scales-
    # based (U_scales is an undetermined free gauge direction, ticket 79).
    # U_scales_out and entropy_threshold (this function's own parameters,
    # unchanged — see CLAUDE.md §3 on not touching call-site signatures) are
    # no longer forwarded here; presence_masks likewise not needed (Frobenius
    # normalisation already makes relation-level mass comparable). Both
    # params are otherwise VESTIGIAL for this call — kept only so this
    # function's own external signature (3 call sites) doesn't need to change.
    collapse_pen, collapse_score = evaluate_dimensional_collapse(
        U_final=U_numpy,
        Z_final=Z_numpy,
    )

    # Section 3
    coherence_pen, weakest_mean = evaluate_topological_coherence(
        Z_final=Z_numpy,
        target_coherence=target_coherence,
        semantic_anchors=frozenset(anchor_keys)
    )

    # Section 4
    active_matrices_set = {k for k in (soc_keys + sem_keys) if k in raw_data}
    socio_semantic_pen = evaluate_socio_semantic_reality(
        U_prob=U_prob,
        raw_data=raw_data,
        active_matrices=active_matrices_set,
        max_monopoly=max_monopoly,
        presence_masks=presence_masks
    )

    # Section 4B (ticket 82, E2 -- Path B). U_prob and presence_masks are
    # already computed above for Section 4 -- reused, not recomputed.
    domain_balance_pen, mean_dev_k = evaluate_domain_balance(
        U_prob=U_prob,
        presence_masks=presence_masks,
    )

    sociological_penalty = collapse_pen + coherence_pen + socio_semantic_pen + domain_balance_pen

    return {
        "collapse_pen":         collapse_pen,
        "collapse_score":       collapse_score,
        "coherence_pen":        coherence_pen,
        "weakest_coherence":    weakest_mean,
        "semantic_pen":         socio_semantic_pen,
        "domain_balance_pen":   domain_balance_pen,
        "mean_dev_k":           mean_dev_k,
        "sociological_penalty": sociological_penalty
    }

#=============================================================================
# 5.2 optuna objective 
#==============================================================================

def create_optuna_objective(raw_data, soc_keys, sem_keys, anchor_keys, dimensions, K_fixed, max_monopoly=0.85, device="cpu", seed=42, seed_function=None):
    """
    Factory function that builds the Optuna objective.
    Now operates in 2D Pareto Mode: (Mathematical Fit, Sociological Reality).
    """
    def objective(trial):
        try:
            # 0. Strict Determinism for the Trial (Moved from Module 4)
            # Assuming set_seeds is imported/available in this namespace
            set_seeds(seed)
            
            # Ticket 78: lambda_l1 fixed at 0.0, removed from Optuna's search
            # (was ticket 77's [0.75, 74.6] range). FINDINGS §6: at
            # lambda_l1=0, live entities sit at L1/L2 ratio 1.14-1.35
            # (comfortably committed; max=2.0 would be fully smeared), with
            # only 0.0-4.0% of live entities near the smeared end (>1.8) per
            # facet -- the hairball this term exists to prevent does not occur
            # on this toy corpus. Across its former range the term evacuated
            # row mass 5-7x while flattening shape toward the smeared end (the
            # opposite of concentration), at ~20% recon_loss cost, because it
            # penalizes column concentration (few entities per community) not
            # row concentration (few communities per entity) -- see FINDINGS
            # §6 for the L1-vs-L2-under-fixed-norm argument for why.
            # This is a TOY-CORPUS decision, not a permanent one: re-test at
            # 22k articles, where far more entities per community make
            # smearing more available and the calculus may differ. The L1/L2
            # ratio per live entity (range [1, sqrt(K)]) is the detector to
            # use for that retest, not U_prob row-max (FINDINGS §5/§6).
            # sparsity_loss itself is still computed below (diagnostics
            # exposes it raw, unweighted, so it stays observable) -- only its
            # weight in the loss is zeroed.
            lambda_l1 = 0.0
            trial.set_user_attr("lambda_l1_fixed", True)
            lambda_z_offdiag = trial.suggest_float('lambda_z_offdiag', 1e-4, 1.0, log=True)
            
            trial.set_user_attr("lambda_l1", lambda_l1)
            trial.set_user_attr("lambda_z_offdiag", lambda_z_offdiag)
            
            hyperparams = {
                'K': K_fixed,
                'lambda_l1': lambda_l1,
                'lambda_z_offdiag': lambda_z_offdiag
            }
            
            # 1. Inner Solver (Mathematical Fit)
            U_final, Z_final, diagnostics = run_inner_solver(
                raw_data=raw_data,
                soc_keys=soc_keys,
                sem_keys=sem_keys,
                anchor_keys=anchor_keys,
                K=K_fixed,
                params=hyperparams,
                dimensions=dimensions,
                device=device
            )

            # Ticket 75: flag (do not prune) non-converged trials. A trial that
            # hits the epoch ceiling still returns a plausible-looking
            # recon_loss and would otherwise enter the Pareto front
            # indistinguishable from a converged one (demonstrated Runs 8/11/12
            # in the diagnostic investigation). Flagging, not pruning, because
            # NSGAIISampler learns from returned values — a pruned trial
            # teaches it nothing and it keeps resampling the same bad region.
            # Downstream consumers (§S2/S3 hypervolume, §S4 archiver) filter on
            # this attr; §S5 already raises on non-convergence per seed.
            converged = bool(diagnostics.get("converged", False))
            epochs_run = len(diagnostics.get("loss_history", []))
            trial.set_user_attr("converged", converged)
            trial.set_user_attr("epochs_run", epochs_run)
            if not converged:
                print(f"[!] Trial {trial.number} did not converge "
                      f"({epochs_run} epochs, K={K_fixed}) — flagged.")

            # 2. Outer Evaluators (Sociological Reality)
            # Per CLAUDE.md §4.13: the sociological penalty must be recomputed
            # identically everywhere it's needed (Optuna loop, §S4 archiver,
            # §S5 stability analysis) via evaluate_complete_solution — the
            # single source of truth — rather than reimplemented per call
            # site. §5.1 and §5.2 used to run the same evaluation sequence
            # written out twice, which is how they drifted (this function was
            # still passing raw torch tensors into numpy-only evaluators).
            evaluation = evaluate_complete_solution(
                U_final=U_final,
                Z_final=Z_final,
                U_scales_out=diagnostics["U_scales"],
                raw_data=raw_data,
                soc_keys=soc_keys,
                sem_keys=sem_keys,
                anchor_keys=anchor_keys,
                max_monopoly=max_monopoly,
                entropy_threshold=ENTROPY_THRESHOLD,  # Assuming imported globally
                target_coherence=TARGET_COHERENCE     # Assuming imported globally
            )

            pure_recon_loss_val = diagnostics["math_loss"]
            sociological_penalty = evaluation["sociological_penalty"]

            # Log all penalties for post-hoc analysis and plotting
            trial.set_user_attr("pure_recon", pure_recon_loss_val)
            trial.set_user_attr("collapse_pen", evaluation["collapse_pen"])
            trial.set_user_attr("coherence_pen", evaluation["coherence_pen"])
            trial.set_user_attr("semantic_pen", evaluation["semantic_pen"])
            trial.set_user_attr("collapse_score_raw", evaluation["collapse_score"])
            trial.set_user_attr("weakest_coherence_raw", evaluation["weakest_coherence"])
            trial.set_user_attr("sociological_penalty", sociological_penalty)

            # 3. Multi-Objective Return
            # Dimension 1: Adam's Math Loss | Dimension 2: Bourdieusian Reality
            return pure_recon_loss_val, sociological_penalty

        except (RuntimeError, FloatingPointError, ValueError) as e:
            # Log before pruning: a blanket ValueError catch previously masked a
            # coding bug (5-value unpack of a 3-value return) as silent trial
            # failure across the entire grid, with empty Pareto fronts as the
            # only symptom. Surface the real exception so that regression can't
            # hide behind "numerically failed trial" again.
            print(f"[!] TRIAL {trial.number} FAILED — pruning. Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise optuna.TrialPruned()

    return objective

# =============================================================================
# MODULE 4: MASTER EXECUTION LOOP (2D Pareto & Stability Architecture)
# SCOPE: Hyperparameter Optimization, Deterministic Archiving, Configuration Grid
# VERSION: v4.2_pareto
# =============================================================================

#=============================================================================
# Module 4: Section 1 M4S1
#=============================================================================

import os
# -> CRITICAL MLOPS FIX: Must be set BEFORE importing torch for CUDA determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import json
import time
import random
import pickle
import numpy as np
import torch
import optuna

# =============================================================================
# PHASE 1A: HARDWARE & MLOPS CONSTANTS
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing Module 4 on DEVICE: {DEVICE}")

# Pipeline Meta-Data
CONFIG_IDS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
MASTER_SEED = 42

# Domain Constants (Outer-Loop Sociology)
MAX_MONOPOLY = 0.85
ENTROPY_THRESHOLD = 0.60  
TARGET_COHERENCE = 0.50

# Search Grid Definition
# Ticket 59: T1 has ~2,300 non-zero observations total; K=15 gave ~43,900 free
# parameters (19/observation), and C1's sole anchor M_Parent_Art (160x25, 63
# non-zeros) asked svds for k_svd=min(15,24)=15 components — more than the
# matrix's rank supports, and equal to K so the padding branch never triggers.
K_LIST = [2, 3, 4, 5, 6]
N_TRIALS = 200

# Global Base Output Directory 
BASE_RESULTS_DIR = os.path.join("results", PIPELINE_VERSION)
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# =============================================================================
# PHASE 1B: THE MASTER SEED & ENVIRONMENT LOGGING
# =============================================================================
def set_seeds(seed=MASTER_SEED):
    """Locks the stochastic environment across all underlying libraries."""
    # Ticket 76: multi-threaded BLAS/OMP reduction ordering is non-associative
    # and was the confirmed root cause of run-to-run non-determinism at the
    # same MASTER_SEED (diagnostic investigation: identical-seed repeats were
    # bit-identical, zero max diff across the entire loss history, once
    # single-threaded). Set here — not only at module import — so it's
    # reasserted on every call: every real entry point (objective(), the §S4
    # archiver, the §S5 stability loop) calls set_seeds() as its first action,
    # which guarantees this is in effect before any tensor op in each of them,
    # not just before the very first one at import time.
    torch.set_num_threads(1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ticket 43: verified this CAN run without warn_only (see CLAUDE.md §4.16)
    # — a 50-epoch C6/K=4 CPU fit completed with no exception. CUDA is
    # unavailable in this environment (login/incline node), so only the CPU
    # path is confirmed; GPU determinism on a compute node is unverified.
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(MASTER_SEED)

def get_environment_info():
    """Captures library versions and global constants for long-term reproducibility."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "master_seed": MASTER_SEED,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "optuna": optuna.__version__,
        "device": str(DEVICE)
    }

# -> FIXED: Save environment metadata to disk immediately upon initialization
env_path = os.path.join(BASE_RESULTS_DIR, "environment_metadata.json")
with open(env_path, "w") as f:
    json.dump(get_environment_info(), f, indent=4)
print(f"[*] Environment metadata saved to: {env_path}")

# =============================================================================
# MODULE 4  M4S2&3
# Sections  2 & 3: ADAPTIVE GRID DISPATCHER & PARETO MAPPER
# SCOPE: K-Capacity Adaptive Budgeting (Scout -> compute_hypervolume Filter -> Deep Dive)
# =============================================================================

import os
import gc
import json
import time
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna._hypervolume import compute_hypervolume  

# =============================================================================
# -> MICROPATCH: RUNTIME API GUARD
# compute_hypervolume is an internal API. We lock the script to warn you if the cluster 
# updates Optuna, which could silently break the compute_hypervolume import.
# =============================================================================
EXPECTED_OPTUNA_VERSION = "4.1.0"  # IMPORTANT: Change this to your exact installed version!
if optuna.__version__ != EXPECTED_OPTUNA_VERSION:
    print(f"\n[!] HPC WARNING: Script tested on Optuna {EXPECTED_OPTUNA_VERSION}, but found {optuna.__version__}.")
    print("[!] The internal API `optuna._hypervolume.compute_hypervolume` may be unstable.\n")

# Adaptive Budgeting & Evaluation Constants
SCOUT_TRIALS = 100       
DEEP_DIVE_TRIALS = 100   
HV_MARGIN = 0.10         

# -----------------------------------------------------------------------------
# HELPER: Objective Factory Wrapper
# -----------------------------------------------------------------------------
def build_objective(raw_data, dimensions, soc_keys, sem_keys, anchor_keys, K):
    return create_optuna_objective(
        raw_data=raw_data,
        soc_keys=soc_keys,
        sem_keys=sem_keys,
        anchor_keys=anchor_keys,
        dimensions=dimensions,
        K_fixed=K,
        max_monopoly=MAX_MONOPOLY,
        device=DEVICE,
        seed=MASTER_SEED,
        seed_function=set_seeds  
    )

# -----------------------------------------------------------------------------
# MASTER ORCHESTRATOR
# -----------------------------------------------------------------------------
def run_adaptive_grid(filepath):
    raw_data = load_and_validate_data(filepath)
    dimensions = raw_data.get('dimensions', None)
    if dimensions is None:
        raise KeyError("CRITICAL ERROR: 'dimensions' metadata missing from raw_data payload.")
    
    total_configs = len(CONFIG_IDS)
    total_Ks = len(K_LIST)
    
    top_k_to_keep = max(2, int(0.4 * total_Ks))
    print(f"[*] ADAPTIVE BUDGET: Keeping top {top_k_to_keep} capacities per configuration.")
    
    methodology_report = {
        "_environment": get_environment_info()
    }

    for c_idx, config_id in enumerate(CONFIG_IDS, 1):
        print(f"\n{'='*75}")
        print(f"[***] PROCESSING CONFIGURATION {c_idx}/{total_configs}: {config_id} [***]")
        print(f"{'='*75}")
        
        soc_keys, sem_keys, anchor_keys = get_active_facets(config_id)
        required_facets = get_required_facets(soc_keys, sem_keys, anchor_keys)
        missing_dims = [f for f in required_facets if f not in dimensions]
        if missing_dims:
            raise KeyError(f"Missing dimensions for facets in {config_id}: {missing_dims}")

        pareto_fronts = {}
        methodology_report[config_id] = {}

        # =====================================================================
        # PHASE 2A: THE SCOUT PHASE (Broad K-Search)
        # =====================================================================
        for k_idx, K in enumerate(K_LIST, 1):
            if K < 2:
                raise ValueError(f"Invalid K={K}. Community detection requires K>=2.")
                
            print(f"\n[*] SCOUTING K-CAPACITY {k_idx}/{total_Ks} | K={K}")
            
            exp_dir = os.path.join(BASE_RESULTS_DIR, config_id, f"K_{K}")
            os.makedirs(exp_dir, exist_ok=True)
            
            db_path = os.path.join(exp_dir, "optuna_study.db")
            storage_name = f"sqlite:///{db_path}"
            study_name = f"{config_id}_K{K}_{PIPELINE_VERSION}"
            
            sampler = optuna.samplers.NSGAIISampler(seed=MASTER_SEED)
            study = optuna.create_study(
                study_name=study_name, storage=storage_name,
                directions=["minimize", "minimize"], sampler=sampler, load_if_exists=True  
            )
            
            objective = build_objective(raw_data, dimensions, soc_keys, sem_keys, anchor_keys, K)

            # Ticket 75/46: budget on COMPLETED-AND-CONVERGED trials, not raw
            # trial count. len(study.trials) counts pruned/failed trials
            # (ticket 46) and, now, would also count converged-blind completed
            # trials that flagging alone would silently subtract from the
            # usable budget. Loop until SCOUT_TRIALS usable trials exist, with
            # a safety valve so a systematically non-converging config/K prints
            # a visible warning instead of burning an unbounded number of fits.
            max_attempts = SCOUT_TRIALS * 3
            start_time = time.perf_counter()
            while True:
                n_usable = len([t for t in study.trials
                                if t.state == TrialState.COMPLETE
                                and t.user_attrs.get("converged", False)])
                if n_usable >= SCOUT_TRIALS:
                    break
                if len(study.trials) >= max_attempts:
                    print(f"[!] {config_id} K={K}: stopped Scout at {len(study.trials)} "
                          f"attempts with only {n_usable}/{SCOUT_TRIALS} converged usable "
                          f"trials. Non-convergence may be systematic here.")
                    break
                try:
                    study.optimize(objective, n_trials=1, n_jobs=1)
                except Exception as e:
                    err_path = os.path.join(exp_dir, "unexpected_error.log")
                    with open(err_path, "w") as f:
                        f.write(str(e))
                    raise RuntimeError(f"Study failed for {config_id} K={K}. See {err_path}") from e
            runtime_sec = time.perf_counter() - start_time

            # Filter the Pareto front to converged trials only (ticket 75). A
            # non-converged trial can transiently sit on study.best_trials and
            # would otherwise inflate/distort the hypervolume that decides
            # which K values survive to the Deep Dive.
            converged_trials = [t for t in study.best_trials
                                 if t.user_attrs.get("converged", False)]
            if not converged_trials and study.best_trials:
                print(f"[!] {config_id} K={K}: {len(study.best_trials)} Pareto trial(s) "
                      f"found but ZERO are converged — Pareto front for this K is empty "
                      f"after filtering.")
            valid_pts = []
            for t in converged_trials:
                if t.values and len(t.values) == 2 and np.isfinite(t.values).all():
                    valid_pts.append(t.values)

            pts_array = np.array(valid_pts)
            pareto_fronts[K] = {'study_name': study_name, 'storage': storage_name, 'pts': pts_array}

            n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
            n_failed = len([t for t in study.trials if t.state == TrialState.FAIL])
            n_converged = len([t for t in study.trials
                                if t.state == TrialState.COMPLETE
                                and t.user_attrs.get("converged", False)])
            n_not_converged = n_complete - n_converged

            methodology_report[config_id][K] = {
                "pareto_size": len(pts_array),
                "trials_completed": n_complete,
                "trials_pruned": n_pruned,
                "trials_failed": n_failed,
                "trials_converged": n_converged,
                "trials_not_converged": n_not_converged,
                "runtime_seconds": runtime_sec,
                "best_math_loss": float(np.min(pts_array[:, 0])) if len(pts_array) > 0 else None,
                "best_soc_penalty": float(np.min(pts_array[:, 1])) if len(pts_array) > 0 else None
            }
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # =====================================================================
        # PHASE 2B: THE DYNAMIC compute_hypervolume HYPERVOLUME FILTER
        # =====================================================================
        # Dynamically build the reference point from all Scout Pareto fronts
        all_points_list = [d["pts"] for d in pareto_fronts.values() if len(d["pts"]) > 0]
        if not all_points_list:
            raise RuntimeError(f"No valid Pareto points produced for configuration {config_id}")
            
        all_points = np.vstack(all_points_list)
        max_math = float(np.max(all_points[:, 0]))
        max_soc = float(np.max(all_points[:, 1]))
        
        ref_point = np.array([max_math * (1.0 + HV_MARGIN), max_soc * (1.0 + HV_MARGIN)])
        
        # Save exact ref coordinates for reproducibility
        methodology_report[config_id]["_reference_point"] = {
            "math": float(ref_point[0]),
            "soc": float(ref_point[1])
        }
        
        k_hypervolumes = {}
        for K, data in pareto_fronts.items():
            pts = data['pts']
            hv = compute_hypervolume(pts, ref_point) if len(pts) > 0 else 0.0
            k_hypervolumes[K] = hv
            methodology_report[config_id][K]["hypervolume"] = float(hv)
            
        surviving_Ks = sorted(k_hypervolumes, key=k_hypervolumes.get, reverse=True)[:top_k_to_keep]
        print(f"\n[***] SCOUT COMPLETE FOR {config_id} | SURVIVING K: {surviving_Ks} [***]")

        # =====================================================================
        # PHASE 3: THE DEEP DIVE (Narrow & Deep)
        # =====================================================================
        for dive_idx, K in enumerate(surviving_Ks, 1):
            print(f"\n[*] DEEP DIVE {dive_idx}/{top_k_to_keep} | Topology {config_id} | K={K}")
            
            target_data = pareto_fronts[K]
            sampler = optuna.samplers.NSGAIISampler(seed=MASTER_SEED)
            study = optuna.create_study(
                study_name=target_data['study_name'], storage=target_data['storage'],
                directions=["minimize", "minimize"], sampler=sampler, load_if_exists=True  
            )
            
            objective = build_objective(raw_data, dimensions, soc_keys, sem_keys, anchor_keys, K)

            # Ticket 75/46: same converged-usable-count budgeting as Scout.
            total_target = SCOUT_TRIALS + DEEP_DIVE_TRIALS
            max_attempts = total_target * 3
            while True:
                n_usable = len([t for t in study.trials
                                if t.state == TrialState.COMPLETE
                                and t.user_attrs.get("converged", False)])
                if n_usable >= total_target:
                    break
                if len(study.trials) >= max_attempts:
                    print(f"[!] {config_id} K={K}: stopped Deep Dive at {len(study.trials)} "
                          f"attempts with only {n_usable}/{total_target} converged usable "
                          f"trials. Non-convergence may be systematic here.")
                    break
                try:
                    study.optimize(objective, n_trials=1, n_jobs=1)
                except Exception as e:
                    err_path = os.path.join(BASE_RESULTS_DIR, config_id, f"K_{K}", "unexpected_error_deepdive.log")
                    with open(err_path, "w") as f:
                        f.write(str(e))
                    raise RuntimeError(f"Deep Dive failed for {config_id} K={K}") from e

            converged_best = [t for t in study.best_trials
                               if t.user_attrs.get("converged", False)]
            print(f"[*] Deep Dive Complete: {config_id} | K={K} | "
                  f"Valid Pareto Models: {len(study.best_trials)} "
                  f"(converged: {len(converged_best)})")
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save the methodology defense report to disk
    report_path = os.path.join(BASE_RESULTS_DIR, "scout_methodology_report.json")
    with open(report_path, "w") as f:
        json.dump(methodology_report, f, indent=4)
    print(f"\n[***] PIPELINE ROUTING COMPLETE. Methodology Report saved to: {report_path}")

# =============================================================================
# MODULE 4 M4S4
# SECTION 4: DETERMINISTIC RE-RUN & ARCHIVING (The Extraction Engine)
# SCOPE: Extract geometrically spaced Pareto models, verify absolute determinism,
#        sanitize tensors, generate Rich Manifests, and archive safely.
# =============================================================================

import os
import json
import torch
import gc
import shutil
import optuna
import numpy as np
import traceback

# -----------------------------------------------------------------------------
# HELPER: JSON Sanitizer
# -----------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.Tensor): 
            return obj.detach().cpu().tolist() if obj.numel() > 1 else obj.item()
        return super(NumpyEncoder, self).default(obj)

def extract_and_archive_pareto_front(filepath, report_path=None, max_models_per_front=10):
    print(f"\n{'='*75}")
    print("[***] INITIATING PHASE 4: PARETO EXTRACTION & MANIFEST GENERATION [***]")
    print(f"{'='*75}")

    raw_data = load_and_validate_data(filepath)
    dimensions = raw_data.get('dimensions', None)
    if dimensions is None:
        raise KeyError("CRITICAL ERROR: 'dimensions' metadata missing from raw_data payload.")

    if report_path is None:
        report_path = os.path.join(BASE_RESULTS_DIR, "scout_methodology_report.json")
    
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Methodology report not found at {report_path}.")
        
    with open(report_path, "r") as f:
        methodology_report = json.load(f)

    env_info = get_environment_info()
    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["cuda_version"] = torch.version.cuda

    for config_id in CONFIG_IDS:
        if config_id not in methodology_report:
            continue
            
        soc_keys, sem_keys, anchor_keys = get_active_facets(config_id)
        valid_Ks = [k for k in methodology_report[config_id].keys() if not k.startswith("_")]
        
        for K_str in valid_Ks:
            K = int(K_str)
            exp_dir = os.path.join(BASE_RESULTS_DIR, config_id, f"K_{K}")
            db_path = os.path.join(exp_dir, "optuna_study.db")
            
            if not os.path.exists(db_path):
                continue

            study = optuna.load_study(
                study_name=f"{config_id}_K{K}_{PIPELINE_VERSION}", 
                storage=f"sqlite:///{db_path}"
            )
            
            # Ticket 75: filter to converged trials before archiving. This is
            # the most important of the three filter points — a non-converged
            # model archived here gets stability-tested (§S5) and reported as
            # a result, not just left as a transient Optuna Pareto point.
            converged_front = [t for t in study.best_trials
                                if t.user_attrs.get("converged", False)]
            if not converged_front and study.best_trials:
                print(f"[!] {config_id} K={K}: {len(study.best_trials)} Pareto trial(s) "
                      f"on record but ZERO are converged — nothing to archive for this K.")

            # Sort initial front purely by math loss (X-axis) to organize the curve
            pareto_trials = sorted(converged_front, key=lambda t: t.values[0])
            
            # -----------------------------------------------------------------
            # MLOps FIX: Cumulative Euclidean Distance Sampling
            # -----------------------------------------------------------------
            if len(pareto_trials) > max_models_per_front:
                pts = np.array([[t.values[0], t.values[1]] for t in pareto_trials])
                pt_min = pts.min(axis=0)
                pt_max = pts.max(axis=0)
                norm_pts = (pts - pt_min) / (pt_max - pt_min + 1e-9)
                
                dists = np.zeros(len(pareto_trials))
                for i in range(1, len(pareto_trials)):
                    dists[i] = dists[i-1] + np.linalg.norm(norm_pts[i] - norm_pts[i-1])
                
                target_dists = np.linspace(0, dists[-1], max_models_per_front)
                
                selected_trials = []
                selected_indices = set()
                
                for td in target_dists:
                    idx = int(np.argmin(np.abs(dists - td)))
                    if idx not in selected_indices:
                        selected_indices.add(idx)
                        selected_trials.append(pareto_trials[idx])
            else:
                selected_trials = pareto_trials
                
            print(f"\n[*] Archiving {len(selected_trials)} geometrically spaced models for {config_id} K={K}...")
            
            archive_dir = os.path.join(exp_dir, "pareto_models")
            os.makedirs(archive_dir, exist_ok=True)
            
            experiment_manifest = {
                "pipeline_version": PIPELINE_VERSION,
                "config_id": config_id,
                "K": K,
                "total_pareto_models": len(pareto_trials),
                "archived_models": [],
                "environment_info": env_info
            }

            for trial in selected_trials:
                set_seeds(MASTER_SEED)
                
                trial_name = f"trial_{str(trial.number).zfill(4)}"
                trial_dir = os.path.join(archive_dir, trial_name)
                
                if os.path.exists(trial_dir):
                    shutil.rmtree(trial_dir)
                os.makedirs(trial_dir)
                
                try:
                    # Execute deterministic re-run
                    U_final, Z_final, diagnostics = run_inner_solver(
                        
                        raw_data=raw_data,
                        soc_keys=soc_keys,
                        sem_keys=sem_keys,
                        anchor_keys=anchor_keys,
                        K=K,
                        dimensions=dimensions,
                        params=trial.params,
                        device=DEVICE,
                        seed_function=set_seeds  
                    )

                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    evaluation = evaluate_complete_solution(
                        U_final=U_final,
                        Z_final=Z_final,
                        U_scales_out=diagnostics["U_scales"],
                        raw_data=raw_data,
                        soc_keys=soc_keys,
                        sem_keys=sem_keys,
                        anchor_keys=anchor_keys,
                        max_monopoly=MAX_MONOPOLY,
                        entropy_threshold=ENTROPY_THRESHOLD,
                        target_coherence=TARGET_COHERENCE
                    )

                    diagnostics.update(evaluation)

                    # -------------------------------------------------------------
                    # MLOps FIX: Reproducibility Integrity Check
                    # -------------------------------------------------------------
                    new_math = diagnostics["math_loss"]
                    new_soc = diagnostics["sociological_penalty"]

                    math_diff = abs(new_math - trial.values[0])
                    soc_diff = abs(new_soc - trial.values[1])

                    if math_diff > 1e-4 or soc_diff > 1e-4:
                        print(f"[!] REPRODUCIBILITY WARNING (Trial {trial.number}): "
                              f"Math Δ {math_diff:.2e} | Soc Δ {soc_diff:.2e}. "
                              f"Determinism may be compromised.")

                    # Safely offload to CPU before serialization
                    U_cpu = {k: v.detach().cpu() for k, v in U_final.items()}
                    Z_cpu = {k: v.detach().cpu() for k, v in Z_final.items()}

                    u_path = os.path.join(trial_dir, "U_matrices.pt")
                    z_path = os.path.join(trial_dir, "Z_core.pt")

                    torch.save(U_cpu, u_path)
                    torch.save(Z_cpu, z_path)

                    if not os.path.exists(u_path) or not os.path.exists(z_path):
                        raise IOError("Silent cluster filesystem failure: Saved tensor not found on disk.")

                    # Log comprehensive metadata
                    model_metadata = {
                        "pipeline_version": PIPELINE_VERSION,
                        "trial_number": trial.number,
                        "config_id": config_id,
                        "K": K,
                        "optuna_math_loss": trial.values[0],
                        "optuna_soc_penalty": trial.values[1],
                        "fresh_diagnostics": diagnostics,
                        "hyperparameters": trial.params,
                        "user_attrs": trial.user_attrs,
                        "system_attrs": trial.system_attrs,
                        "tensor_metadata": {
                            "device_saved": "cpu",
                            "dtype_U": {k: str(v.dtype) for k, v in U_cpu.items()},
                            "dtype_Z": {k: str(v.dtype) for k, v in Z_cpu.items()},
                            "torch_version": torch.__version__
                        }
                    }

                    with open(os.path.join(trial_dir, "model_metadata.json"), "w") as f:
                        json.dump(model_metadata, f, indent=4, cls=NumpyEncoder)

                    experiment_manifest["archived_models"].append({
                        "trial_number": trial.number,
                        "folder_name": trial_name,
                        "math_loss": trial.values[0],
                        "soc_penalty": trial.values[1]
                    })

                except KeyboardInterrupt:
                    print("[!] User aborted pipeline during extraction. Exiting safely.")
                    raise

                except Exception as e:
                    err_path = os.path.join(trial_dir, "extraction_error.log")
                    with open(err_path, "w") as f:
                        f.write(traceback.format_exc())
                    print(f"[!] Extraction failed for Trial {trial.number}. Check traceback log.")

                finally:
                    if 'U_final' in locals(): del U_final
                    if 'Z_final' in locals(): del Z_final
                    if 'U_cpu' in locals(): del U_cpu
                    if 'Z_cpu' in locals(): del Z_cpu
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

            manifest_path = os.path.join(archive_dir, "experiment_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(experiment_manifest, f, indent=4, cls=NumpyEncoder)
                
            print(f"[*] Successfully archived {len(experiment_manifest['archived_models'])} models to {archive_dir}")

    print(f"\n{'='*75}")
    print("[***] PHASE 4 COMPLETE. ALL PARETO ARTIFACTS SECURED FOR PHASE 5. [***]")
    print(f"{'='*75}")

# =============================================================================
# MODULE 4, SECTION 5: POST-HOC DUAL-TRACK STABILITY ANALYSIS
# SCOPE: Disk-streamed pairwise consensus, Objective Variance Tracking, 
#        Reproducibility Delta, Failed Seed Logging, I/O & Shape Integrity,
#        Ontological Parity, and Maximum Entropy Zero-Row Handling.
# =============================================================================

import os
import json
import torch
import shutil
import numpy as np
import gc
import traceback
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

# -----------------------------------------------------------------------------
# CONSTANTS & SETTINGS
# -----------------------------------------------------------------------------
N_STABILITY_SEEDS = 10
STABILITY_SEEDS = [1000 + i for i in range(N_STABILITY_SEEDS)]

# Ticket 53: NumpyEncoder was redefined here without the torch.Tensor case
# Module 4 §4 needs; since both classes share this file's single namespace,
# whichever definition runs last wins everywhere, silently breaking tensor
# serialization in §4. Kept the one definition (§4, includes torch.Tensor).

# -----------------------------------------------------------------------------
# HELPER MATH FUNCTIONS
# -----------------------------------------------------------------------------
def row_normalize(matrix):
    """
    Converts raw interactions into node-level probability distributions.
    MLOps FIX: Uses Maximum Entropy (1/K) for completely pruned nodes (zero-rows).
    """
    K = matrix.shape[1]
    row_sums = matrix.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.flatten() == 0)
    
    # Temporarily set zero-sums to 1.0 to avoid division by zero
    row_sums[zero_rows] = 1.0  
    prob_matrix = matrix / row_sums
    
    # Apply uniform distribution to rows that had no capital
    prob_matrix[zero_rows, :] = 1.0 / K
    return prob_matrix

def col_normalize(matrix):
    """Converts raw interactions into community-level emission profiles."""
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1e-9
    return matrix / col_sums

def row_wise_cosine_similarity(A, B):
    """Fast, vectorized row-wise Cosine Similarity calculation."""
    dot_product = np.sum(A * B, axis=1)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    norms = norm_A * norm_B
    norms[norms == 0] = 1e-9
    return dot_product / norms

# -----------------------------------------------------------------------------
# THE CONSENSUS ENGINE
# -----------------------------------------------------------------------------
def run_dual_track_stability_analysis(filepath):
    print(f"\n{'='*75}")
    print(f"[***] INITIATING SECTION 5: DUAL-TRACK CONSENSUS ({N_STABILITY_SEEDS} SEEDS) [***]")
    print(f"{'='*75}")

    raw_data = load_and_validate_data(filepath)
    dimensions = raw_data.get('dimensions', None)
    
    master_stability_report = {}

    for config_id in CONFIG_IDS:
        soc_keys, sem_keys, anchor_keys = get_active_facets(config_id)
        
        # MLOps FIX: True Facet Extraction. Derive deterministic nodes from Relations 29.07.2026 20:59
        # This guarantees we stack "art", "auth", etc., NOT "S_Art_Auth"
        ALL_FACETS = get_required_facets(soc_keys, sem_keys, anchor_keys)
        
        config_dir = os.path.join(BASE_RESULTS_DIR, config_id)
        if not os.path.exists(config_dir): continue
            
        master_stability_report[config_id] = {}

        for k_folder in os.listdir(config_dir):
            if not k_folder.startswith("K_"): continue
            
            K = int(k_folder.split("_")[1])
            manifest_path = os.path.join(config_dir, k_folder, "pareto_models", "experiment_manifest.json")
            if not os.path.exists(manifest_path): continue
                
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                
            print(f"\n[*] Evaluating Stability for {config_id} | K={K} ({len(manifest['archived_models'])} models)")
            master_stability_report[config_id][K] = {}

            for model_info in manifest["archived_models"]:
                trial_name = model_info["folder_name"]
                trial_dir = os.path.join(config_dir, k_folder, "pareto_models", trial_name)
                metadata_path = os.path.join(trial_dir, "model_metadata.json")
                
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    
                params = metadata["hyperparameters"]
                optuna_math_loss = metadata.get("optuna_math_loss", 0.0)
                optuna_soc_penalty = metadata.get("optuna_soc_penalty", 0.0)
                
                print(f"    -> Stress-testing {trial_name}...")
                
                temp_tensor_dir = os.path.join(trial_dir, "temp_stability_tensors")
                os.makedirs(temp_tensor_dir, exist_ok=True)
                
                seed_math_losses = []
                seed_soc_penalties = []
                failed_seeds_log = []
                
                try:
                    # ---------------------------------------------------------
                    # PHASE 1: GENERATE & DISK-STREAM THE RANDOM SEEDS
                    # ---------------------------------------------------------
                    for seed_idx, seed_val in enumerate(STABILITY_SEEDS):
                        try:
                            set_seeds(seed_val)
                            
                            U_final, Z_final, diagnostics = run_inner_solver(
                                raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys,
                                anchor_keys=anchor_keys, K=K, dimensions=dimensions,
                                params=params, device=DEVICE, seed_function=set_seeds
                            )

                            if torch.cuda.is_available(): torch.cuda.synchronize()

                            if not diagnostics.get("converged", True):
                                raise RuntimeError("Silent optimizer failure: Convergence not reached.")

                            # Per §4.13: Module 2 computes no sociological metrics, so the
                            # sociological penalty must be recomputed per seed, never read
                            # back from the archived Optuna trial.
                            evaluation = evaluate_complete_solution(
                                U_final=U_final,
                                Z_final=Z_final,
                                U_scales_out=diagnostics["U_scales"],
                                raw_data=raw_data,
                                soc_keys=soc_keys,
                                sem_keys=sem_keys,
                                anchor_keys=anchor_keys,
                                max_monopoly=MAX_MONOPOLY,
                                entropy_threshold=ENTROPY_THRESHOLD,
                                target_coherence=TARGET_COHERENCE
                            )

                            seed_math_losses.append(diagnostics.get("math_loss", 0.0))
                            seed_soc_penalties.append(evaluation["sociological_penalty"])
                                
                            U_np = {f: u.detach().cpu().numpy() for f, u in U_final.items()}
                            np.savez(os.path.join(temp_tensor_dir, f"seed_{seed_idx}.npz"), **U_np)
                                
                            del U_final, U_np
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            
                        except Exception as e:
                            failed_seeds_log.append({
                                "seed_index": seed_idx,
                                "seed_value": seed_val,
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            })
                            break # Abort generating further seeds for this model

                    if failed_seeds_log:
                        metadata["failed_stability_seeds"] = failed_seeds_log
                        with open(metadata_path, "w") as f:
                            json.dump(metadata, f, indent=4, cls=NumpyEncoder)
                        print(f"       [!] Stability test aborted for {trial_name}. See metadata for traceback.")
                        continue

                    # Calculate Reproducibility Delta & Objective Variance (ddof=1)
                    math_mean = float(np.mean(seed_math_losses))
                    soc_mean = float(np.mean(seed_soc_penalties))
                    
                    obj_variance = {
                        "math_loss_mean": math_mean,
                        "math_loss_sd": float(np.std(seed_math_losses, ddof=1)) if len(seed_math_losses) > 1 else 0.0,
                        "soc_penalty_mean": soc_mean,
                        "soc_penalty_std": float(np.std(seed_soc_penalties, ddof=1)) if len(seed_soc_penalties) > 1 else 0.0
                    }
                    
                    reproducibility_delta = {
                        "delta_math_loss": abs(optuna_math_loss - math_mean),
                        "delta_soc_penalty": abs(optuna_soc_penalty - soc_mean)
                    }

                    # ---------------------------------------------------------
                    # PHASE 2: PAIRWISE DUAL-TRACK CONSENSUS
                    # ---------------------------------------------------------
                    track_A_unweighted_jsd = {f: [] for f in ALL_FACETS}
                    track_B_weighted_cosine = {f: [] for f in ALL_FACETS}
                    
                    # Track global scores per pair to calculate accurate Standard Deviation 
                    # while preserving Ontological Parity (Mean of Means)
                    pair_global_scores_A = []
                    pair_global_scores_B = []
                    
                    seed_pairs = list(itertools.combinations(range(N_STABILITY_SEEDS), 2))
                    
                    for (s1, s2) in seed_pairs:
                        # MLOps FIX: Context Managers to prevent File Descriptor exhaustion
                        with np.load(os.path.join(temp_tensor_dir, f"seed_{s1}.npz")) as data1, \
                             np.load(os.path.join(temp_tensor_dir, f"seed_{s2}.npz")) as data2:
                            
                            # I/O Integrity check
                            missing_facets = (set(ALL_FACETS) - set(data1.files)) | (set(ALL_FACETS) - set(data2.files))
                            if missing_facets:
                                raise IOError(f"Corrupted Disk Save: Missing facets {missing_facets}")
                                
                            # Matrix Shape Verification (Catching Dimensionality Collapses)
                            for f in ALL_FACETS:
                                assert data1[f].shape[1] == K, f"Collapse Seed {s1}: Facet {f} cols != {K}"
                                assert data2[f].shape[1] == K, f"Collapse Seed {s2}: Facet {f} cols != {K}"
                            
                            S1_raw = np.vstack([data1[f] for f in ALL_FACETS]) 
                            S2_raw = np.vstack([data2[f] for f in ALL_FACETS])
                            
                            # --- TRACK A ALIGNMENT (JSD / Probability Space) ---
                            S1_col_prob = col_normalize(S1_raw)
                            S2_col_prob = col_normalize(S2_raw)
                            cost_A = np.zeros((K, K))
                            
                            # --- TRACK B ALIGNMENT (Cosine / Magnitude Space) ---
                            cost_B = np.zeros((K, K))
                            
                            for k1 in range(K):
                                for k2 in range(K):
                                    cost_A[k1, k2] = jensenshannon(S1_col_prob[:, k1], S2_col_prob[:, k2])
                                    cost_B[k1, k2] = 1.0 - (np.dot(S1_raw[:, k1], S2_raw[:, k2]) / 
                                                     (np.linalg.norm(S1_raw[:, k1]) * np.linalg.norm(S2_raw[:, k2]) + 1e-9))
                                    
                            _, col_ind_A = linear_sum_assignment(cost_A)
                            _, col_ind_B = linear_sum_assignment(cost_B)

                            # --- FACET-LEVEL EVALUATION ---
                            current_pair_A_facets = []
                            current_pair_B_facets = []
                            
                            for f in ALL_FACETS:
                                U1_raw = data1[f]
                                U2_raw = data2[f]
                                
                                # Track A Evaluation: Unweighted JSD
                                U1_prob = row_normalize(U1_raw)
                                U2_prob_aligned = row_normalize(U2_raw)[:, col_ind_A]
                                js_distances = jensenshannon(U1_prob, U2_prob_aligned, axis=1)
                                
                                mean_js_sim = 1.0 - np.nanmean(js_distances)
                                track_A_unweighted_jsd[f].append(mean_js_sim)
                                current_pair_A_facets.append(mean_js_sim)
                                
                                # Track B Evaluation: Magnitude-Weighted Cosine
                                U2_raw_aligned = U2_raw[:, col_ind_B]
                                cos_similarities = row_wise_cosine_similarity(U1_raw, U2_raw_aligned)
                                
                                magnitude_weights = U1_raw.sum(axis=1) 
                                
                                if magnitude_weights.sum() == 0:
                                    weighted_cos_sim = float(np.mean(cos_similarities))
                                else:
                                    weighted_cos_sim = float(np.average(cos_similarities, weights=magnitude_weights))
                                    
                                track_B_weighted_cosine[f].append(weighted_cos_sim)
                                current_pair_B_facets.append(weighted_cos_sim)
                                
                            # Preserve Ontological Parity for the Global SD Calculation
                            pair_global_scores_A.append(np.mean(current_pair_A_facets))
                            pair_global_scores_B.append(np.mean(current_pair_B_facets))
                            
                    # ---------------------------------------------------------
                    # PHASE 3: AGGREGATION & REPORTING
                    # ---------------------------------------------------------
                    # Global Means and SDs (Ontological Parity Preserved via pair_global_scores)
                    global_A_mean = float(np.mean(pair_global_scores_A))
                    global_A_sd = float(np.std(pair_global_scores_A, ddof=1)) if len(pair_global_scores_A) > 1 else 0.0
                    
                    global_B_mean = float(np.mean(pair_global_scores_B))
                    global_B_sd = float(np.std(pair_global_scores_B, ddof=1)) if len(pair_global_scores_B) > 1 else 0.0
                    
                    # Individual Facet Stats
                    facet_A_stats = {
                        f: {
                            "mean": float(np.mean(scores)),
                            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                        } for f, scores in track_A_unweighted_jsd.items()
                    }
                    
                    facet_B_stats = {
                        f: {
                            "mean": float(np.mean(scores)),
                            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                        } for f, scores in track_B_weighted_cosine.items()
                    }
                    
                    # Update local metadata with all explicit reproducibility trackers
                    metadata["stability_test_seeds_used"] = STABILITY_SEEDS
                    metadata["reproducibility_delta"] = reproducibility_delta
                    metadata["objective_variance"] = obj_variance
                    metadata["consensus_track_A_unweighted"] = {
                        "global_js_similarity_mean": global_A_mean,
                        "global_js_similarity_sd": global_A_sd,
                        "facet_stats": facet_A_stats
                    }
                    metadata["consensus_track_B_magnitude_weighted"] = {
                        "global_cosine_similarity_mean": global_B_mean,
                        "global_cosine_similarity_sd": global_B_sd,
                        "facet_stats": facet_B_stats
                    }
                    
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
                        
                    master_stability_report[config_id][K][trial_name] = {
                        "Track_A_Unweighted_Mean": global_A_mean,
                        "Track_A_Unweighted_SD": global_A_sd,
                        "Track_B_Weighted_Mean": global_B_mean,
                        "Track_B_Weighted_SD": global_B_sd
                    }
                    print(f"       [+] Track A: {global_A_mean:.4f} (±{global_A_sd:.4f}) | Track B: {global_B_mean:.4f} (±{global_B_sd:.4f})")
                    
                except Exception as e:
                    err_path = os.path.join(trial_dir, "stability_error.log")
                    with open(err_path, "w") as f:
                        f.write(traceback.format_exc())
                    print(f"       [!] Stability test failed for {trial_name}. See logs.")
                    
                finally:
                    if os.path.exists(temp_tensor_dir):
                        shutil.rmtree(temp_tensor_dir)

    report_path = os.path.join(BASE_RESULTS_DIR, "master_dual_track_stability_report.json")
    with open(report_path, "w") as f:
        json.dump(master_stability_report, f, indent=4, cls=NumpyEncoder)
        
    print(f"\n{'='*75}")
    print(f"[***] SECTION 5 COMPLETE. Stability Report saved to: {report_path} [***]")
    print(f"{'='*75}")

if __name__ == "__main__":
    DATA_FILEPATH = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1.pkl"
    run_adaptive_grid(DATA_FILEPATH)
    extract_and_archive_pareto_front(DATA_FILEPATH)
    run_dual_track_stability_analysis(DATA_FILEPATH)