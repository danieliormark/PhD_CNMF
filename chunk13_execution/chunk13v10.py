# =============================================================================
# CHUNK 13v10 - Module 1: STATIC ARCHITECTURE & TOPOLOGY
# v10 = v9.4 plus the temporal prior (ticket 105, CLAUDE.md §4.25). With no prior
# supplied, or LAMBDA_TEMPORAL = 0, every fit is v9.4's.
# SCOPE: Global constants, HPC paths, and Metagraph configurations (C1-C6)
# =============================================================================

import os
import math
# torch must be imported before numpy: importing numpy first changes fits
# (2026-10-02, C1/K=3 seed 42: math_loss 0.8446383 / 1476 epochs instead of
# 0.8447862 / 1452), presumably because the two ship different BLAS/OpenMP
# runtimes and the first one loaded is used by both. Scripts that load this
# module must not import numpy before it to reproduce pipeline fits exactly.
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
# Ticket 96: ceiling for (a) a trial that hits INNER_EPOCHS unconverged while
# non-dominated by every converged trial in its study, and (b) stability
# refits. The cap only bounds the loop (no schedule depends on it), so a
# refit with this cap reproduces the first INNER_EPOCHS epochs exactly and
# continues. Tested 2026-10-01: such trials converged by 2,025-2,472 epochs;
# stability seeds of models that had failed at 2,000 converged 10/10 by 2,876.
EXTENDED_EPOCHS = 8000
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

# 2c. In-loop shared-entity concentration floor (ticket 102, v9.4). For each
# live entity of the four lower semantic facets, s_i = normalised entropy of
# the community distribution it inherits from the articles that use it
# (article weighted membership pushed down the raw relations, the same
# propagation as Penalty_B), computed once at epoch CONC_WARMUP and then held
# fixed. Loss: mean_i max(0, CONC_GAMMA * s_i - H(p_i))^2, p_i = the entity's
# reconstruction-weighted membership (ticket 100). One-sided: an entity is
# only pushed when its spread across communities is below the floor its usage
# implies. Weight NOT Optuna-tunable (the lambda_domain_balance precedent).
# Calibrated 2026-10-02 (FINDINGS §31): 20 config/K cells x 5 seeds x 15
# settings, plus an independent check against the abstracts' text. At
# 0.1/0.8/300: shared-but-exclusive entities 25% -> 3% (graph measure) and
# 49% -> 19% (text measure); niche-in-text words wrongly spread 10% -> 16%
# (ticket 103, open); recon cost +0.5% mean. Set LAMBDA_CONC = 0.0 to run
# without it (exactly v9.3's trajectory).
LAMBDA_CONC = 0.1
CONC_GAMMA = 0.8
CONC_WARMUP = 300
CONC_FACETS = ['core_child_he', 'cousin_he', 'core_atom', 'fringe_atom']

# Ticket 105 (v10): temporal prior, CLAUDE.md §4.25. When a T1 prior is given,
# each persisting entity's T2 weighted membership is pulled towards its T1
# membership, KL(prior || current), from epoch TEMPORAL_WARMUP on. At that
# epoch T1 and T2 communities are matched (rectangular Hungarian on membership
# columns over the persisting entities); a pair is kept only if its similarity
# beats the TEMPORAL_MATCH_QUANTILE of TEMPORAL_MATCH_SHUFFLES row-shuffles.
# Weight NOT Optuna-tunable (it would always be driven to 0). LAMBDA_TEMPORAL
# is set by calibration (test T5); until then it is 0 and a --prior run must
# give --lambda-temporal explicitly.
LAMBDA_TEMPORAL = 0.0
TEMPORAL_WARMUP = CONC_WARMUP
TEMPORAL_MATCH_SHUFFLES = 200
TEMPORAL_MATCH_QUANTILE = 0.95
TEMPORAL_FACETS = ['auth', 'affil', 'journ', 'parent_he', 'core_child_he', 'core_atom',
                   'cousin_he', 'fringe_atom']   # every global facet; 'art' is slice-specific

# 3. Doxa / Elite Capture Rule
#MAX_MONOPOLY = 0.60            # Maximum allowable mass concentration for elite entities in a single community deprecated in favour of more flexible 0.85

# 4. Meta-Loss Aggregation Weights (Calibrated via initial trial baseline)
LAMBDA_COLLAPSE = 1.0          # Penalty weight for dimensional collapse
LAMBDA_COH = 1.0               # Penalty weight for topological coherence violations
LAMBDA_SEM = 1.0               # Penalty weight for socio-semantic semantic reality deviations

# 5. Seed and version control for reproducibility
MASTER_SEED = 42
PIPELINE_VERSION = "v10.0"   # set to v10.0.<slice>.<prior|free> in __main__

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
    # Ticket 85: sorted, never the raw set. The draw order of np.random.rand
    # decides which facet gets which random numbers, and U_np's key order
    # (set here) also drives the noise draw in step 3.5. Set order depends
    # on PYTHONHASHSEED, so iterating the raw set gave a different fit per
    # process launch at the same MASTER_SEED.
    for facet in sorted(active_facets):
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
    U_raw = {f: torch.tensor(U_np[f], device=device, dtype=torch.float32, requires_grad=True) for f in sorted(active_facets)}
    Z_raw = {rel: torch.tensor(Z_np[rel], device=device, dtype=torch.float32, requires_grad=True) for rel in active_matrices.keys()}

    return U_raw, Z_raw

# -----------------------------------------------------------------------------
# 2.1b TEMPORAL PRIOR HELPERS (ticket 105, CLAUDE.md §4.25)
# -----------------------------------------------------------------------------
def temporal_match_columns(P1, P2_blocks, shuffles=None, quantile=None, seed=0):
    """Match T1 community columns to T2 community columns on persisting entities.
    P1: (n x K1) T1 memberships; P2_blocks: list of per-facet (n_f x K2) T2
    memberships whose stacked rows align with P1's.

    Similarity S[a,b] = cosine between T1 column a and T2 column b. Its chance
    distribution comes from shuffling the T2 rows within each facet `shuffles`
    times (ticket 95's logic): this keeps each column's size and shape and
    destroys which entity is which. Each pair is standardised against its own
    chance distribution, z = (S - mean) / sd, so large communities do not match
    merely because they are large. The rectangular Hungarian assignment is
    solved on z. A matched pair is accepted only if its z exceeds the `quantile`
    of the matched z values obtained by running the same assignment on each
    shuffle, so the selection made by the assignment is part of the null
    (comparing the best pairs with a same-pair null accepted 20% of pairs between
    unrelated memberships; test T1h). Local generator; global random state
    untouched. Returns (accepted [(a, b)], record)."""
    from scipy.optimize import linear_sum_assignment as _lsa
    shuffles = TEMPORAL_MATCH_SHUFFLES if shuffles is None else shuffles
    quantile = TEMPORAL_MATCH_QUANTILE if quantile is None else quantile
    P2 = np.vstack(P2_blocks)
    cn = lambda A: A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    C1 = cn(P1)
    S = C1.T @ cn(P2)
    rng = np.random.default_rng(seed)
    null = np.empty((shuffles,) + S.shape)
    for t in range(shuffles):
        null[t] = C1.T @ cn(np.vstack([B[rng.permutation(len(B))] for B in P2_blocks]))
    mu, sd = null.mean(0), null.std(0) + 1e-12
    Z = (S - mu) / sd
    matched_null = []
    for t in range(shuffles):
        Zt = (null[t] - mu) / sd
        r, c = _lsa(-Zt)
        matched_null.extend(Zt[r, c].tolist())
    thr = float(np.quantile(matched_null, quantile))
    ra, cb = _lsa(-Z)
    record = {"pairs": [], "rejected": [], "z_threshold": thr}
    accepted = []
    for a, b in zip(ra, cb):
        rec = {"t1": int(a), "t2": int(b), "similarity": float(S[a, b]), "z": float(Z[a, b])}
        if Z[a, b] > thr:
            accepted.append((int(a), int(b))); record["pairs"].append(rec)
        else:
            record["rejected"].append(rec)
    record["unmatched_t1"] = [a for a in range(S.shape[0]) if a not in {x for x, _ in accepted}]
    record["unmatched_t2"] = [b for b in range(S.shape[1]) if b not in {y for _, y in accepted}]
    return accepted, record


def temporal_mapped_prior(P1f, accepted, K2):
    """T1 memberships of one facet mapped onto the T2 columns: matched T1
    column a -> its T2 column b, zeros on unmatched T2 columns, unmatched T1
    columns dropped. Returns (Q with rows summing to 1, m = share of the T1
    membership on matched communities, keep mask of rows with m > 0)."""
    Q = np.zeros((P1f.shape[0], K2))
    for a, b in accepted:
        Q[:, b] = P1f[:, a]
    m = Q.sum(1)
    keep = m > 1e-12
    return np.where(keep[:, None], Q / np.where(keep, m, 1.0)[:, None], 0.0), m, keep


def temporal_loss_value(U_pos, U_scales, weights, state):
    """Sum_i m_i KL(Q_i || p_i) / Sum_i m_i over the frozen persisting entities,
    with p_i = (u_i / sg(column norms)) * sg(column weights), row-normalised. The
    column norms (U_scales) and weights are detached, so the gradient reaches
    only the persisting entities' own rows of U_pos. Returns a torch scalar, or
    None if the state is empty."""
    t_num, t_den = None, 0.0
    for f, (idx, Q, m) in state.items():
        X = (U_pos[f][idx] / U_scales[f].detach()) * weights[f].detach()
        p = X / (X.sum(1, keepdim=True) + 1e-12)
        kl = (Q * (torch.log(Q + 1e-12) - torch.log(p + 1e-12))).sum(1)
        term = (m * kl).sum()
        t_num = term if t_num is None else t_num + term
        t_den += float(m.sum())
    return t_num / t_den if (t_num is not None and t_den > 0) else None


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
    learning_rate=LEARNING_RATE,  # matched the (previously dead) module constants; now driven by them
    temporal_prior=None  # ticket 105: output of build_temporal_prior, or None (= v9.4)
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
    # Ticket 102: shared-entity concentration floor, see LAMBDA_CONC above.
    lambda_conc = params.get('lambda_conc', LAMBDA_CONC)
    conc_gamma = params.get('conc_gamma', CONC_GAMMA)
    conc_warmup = int(params.get('conc_warmup', CONC_WARMUP))
    # Ticket 105: temporal prior, see LAMBDA_TEMPORAL above and CLAUDE.md §4.25.
    lambda_temporal = float(params.get('lambda_temporal', LAMBDA_TEMPORAL))
    temporal_warmup = int(params.get('temporal_warmup', TEMPORAL_WARMUP))
    temporal_on = (temporal_prior is not None and lambda_temporal > 0
                   and any(len(v['idx']) for v in temporal_prior['facets'].values()))

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

    def _column_weights(Un, Zs, f):
        """Torch version of compute_weighted_membership's column weights:
        w_k^2 = sum over relations touching f of the squared Frobenius norm of
        the slice of U_f Z U_g^T that passes through column k (equal to
        diag(Z (U_g^T U_g) Z^T)_k, computed via K x K Gram matrices). Columns
        below 1e-3 of the largest weight get 0. Detached: the weights only
        decide which columns count, no gradient flows through them."""
        with torch.no_grad():
            w = torch.zeros(K, device=device, dtype=Un[f].dtype)
            for r, z in Zs.items():
                a, b = RELATION_MAP[r]
                if a == f:
                    w = w + torch.diagonal(z @ (Un[b].t() @ Un[b]) @ z.t())
                if b == f:
                    w = w + torch.diagonal(z.t() @ (Un[a].t() @ Un[a]) @ z)
            w = torch.sqrt(torch.clamp(w, min=0.0))
            return torch.where(w < 1e-3 * w.max(), torch.zeros_like(w), w)

    def _db_domain_share(U_norm_dict, Zs, facets, w_total):
        """Entity-count-weighted mean, over `facets`, of each facet's own
        live-entity-masked mean membership row, differentiable. Ticket 100
        (v9.4): membership is reconstruction-weighted (columns weighted by
        _column_weights, unengaged entities -- weighted mass below 1e-3 x the
        live median -- dropped), matching evaluate_domain_balance, which now
        reads compute_weighted_membership's output. Weight is 0.0 by default
        (ticket 82), so this only feeds raw_domain_balance_penalty."""
        if not facets or w_total == 0:
            return None
        acc = None
        for f in facets:
            if f not in U_norm_dict:
                continue
            live = U_norm_dict[f][db_mask_t[f]] * _column_weights(U_norm_dict, Zs, f)
            if live.shape[0] == 0:
                continue
            row_sums = torch.sum(torch.abs(live), dim=1, keepdim=True)
            with torch.no_grad():
                pos = row_sums[row_sums > 0]
                eng = (row_sums >= 1e-3 * torch.median(pos)) if pos.numel() else torch.zeros_like(row_sums, dtype=torch.bool)
            u_prob_live = torch.where(eng, live / (row_sums + 1e-12), torch.zeros_like(live))
            term = db_live_counts[f] * u_prob_live.mean(dim=0)
            acc = term if acc is None else acc + term
        return None if acc is None else acc / w_total

    # Ticket 102: shared-entity concentration floor. conc_state is filled once,
    # at epoch conc_warmup: facet -> (live-entity index tensor, frozen s_i).
    conc_state = {}

    def _conc_freeze(Un, Zs):
        """s_i = normalised entropy of the community distribution entity i
        inherits from the articles that use it: the articles' weighted
        membership (unengaged articles zeroed) pushed down the raw relations
        along the same paths as socio_semantic_parts' Penalty_B. Computed once
        from detached values and then held fixed, so the term has a fixed
        target and passes no gradient into the article memberships."""
        with torch.no_grad():
            Xa = (Un['art'] * _column_weights(Un, Zs, 'art')).cpu().numpy().astype(np.float64)
        ma = Xa.sum(1)
        live_a = db_presence_masks.get('art', np.ones(len(ma), bool))
        pos = ma[live_a & (ma > 0)]
        eng = ma >= 1e-3 * np.median(pos) if pos.size else np.zeros(len(ma), bool)
        P_art = np.where(eng[:, None], Xa / np.where(ma > 0, ma, 1.0)[:, None], 0.0)
        act = set(active_matrices)
        E = {f: np.zeros((Un[f].shape[0], K)) for f in CONC_FACETS if f in Un}
        for k in range(K):
            W = P_art[:, k]
            pw = {}
            if 'M_Parent_Art' in act:
                pw['parent_he'] = raw_data['M_Parent_Art'].dot(W)
            cp = []
            if 'M_Child_Art' in act:
                cp.append(raw_data['M_Child_Art'].dot(W))
            if 'M_Child_Parent' in act and 'parent_he' in pw:
                cp.append(raw_data['M_Child_Parent'].dot(pw['parent_he']))
            if cp:
                pw['core_child_he'] = sum(cp)
            cu = []
            if 'M_Cousin_Art' in act:
                cu.append(raw_data['M_Cousin_Art'].dot(W))
            if 'M_Cousin_Parent' in act and 'parent_he' in pw:
                cu.append(raw_data['M_Cousin_Parent'].dot(pw['parent_he']))
            if 'M_Cousin_Child' in act and 'core_child_he' in pw:
                cu.append(raw_data['M_Cousin_Child'].dot(pw['core_child_he']))
            if cu:
                pw['cousin_he'] = sum(cu)
            if 'M_Atom_Child' in act and 'core_child_he' in pw:
                pw['core_atom'] = raw_data['M_Atom_Child'].dot(pw['core_child_he'])
            if 'M_Fringe_Cousin' in act and 'cousin_he' in pw:
                pw['fringe_atom'] = raw_data['M_Fringe_Cousin'].dot(pw['cousin_he'])
            for f in E:
                if f in pw:
                    E[f][:, k] = np.asarray(pw[f]).ravel()
        for f, Ef in E.items():
            tot = Ef.sum(1)
            keep = (tot > 1e-12) & db_presence_masks.get(f, np.ones(len(tot), bool))
            if not keep.any():
                continue
            Pe = Ef[keep] / tot[keep, None]
            s = -(Pe * np.log(Pe + 1e-12)).sum(1) / np.log(K)
            conc_state[f] = (torch.from_numpy(np.where(keep)[0]).to(device),
                             torch.from_numpy(s).to(device=device, dtype=Un[f].dtype))

    # Ticket 105: temporal prior. temporal_state is filled once, at epoch
    # temporal_warmup: facet -> (index tensor of persisting entities kept,
    # prior mapped into this fit's column order Q (rows sum to 1), weight m =
    # share of the entity's T1 membership on matched communities).
    temporal_state = {}
    temporal_match = {}

    def _temporal_freeze(Un, Zs, epoch):
        """Match the T1 communities to this fit's communities on the persisting
        entities (temporal_match_columns), then freeze the prior in this fit's
        column order (temporal_mapped_prior). Reads detached values only."""
        facets = [f for f in sorted(temporal_prior['facets'])
                  if f in Un and len(temporal_prior['facets'][f]['idx'])]
        blocks = []
        for f in facets:
            idx = np.asarray(temporal_prior['facets'][f]['idx'], dtype=np.int64)
            with torch.no_grad():
                X = (Un[f][torch.from_numpy(idx).to(device)]
                     * _column_weights(Un, Zs, f)).cpu().numpy().astype(np.float64)
            ms = X.sum(1)
            ok = ms > 0
            P1 = np.asarray(temporal_prior['facets'][f]['P'], dtype=np.float64)
            conf = np.asarray(temporal_prior['facets'][f].get('conf', np.ones(len(idx))), dtype=np.float64)
            if ok.any():
                blocks.append((f, idx[ok], P1[ok], X[ok] / ms[ok, None], conf[ok]))
        temporal_match.update({"epoch": int(epoch), "K1": int(temporal_prior['K1']), "K2": int(K),
                               "pairs": [], "rejected": [], "unmatched_t1": [], "unmatched_t2": [],
                               "n_entities": {b[0]: int(len(b[1])) for b in blocks}})
        if not blocks:
            return
        accepted, record = temporal_match_columns(np.vstack([b[2] for b in blocks]), [b[3] for b in blocks])
        temporal_match.update(record)
        for f, idx, P1f, _, conf in blocks:
            Q, m, keep = temporal_mapped_prior(P1f, accepted, K)
            m = m * conf          # confidence weighting (1 when the prior has none)
            keep = keep & (m > 1e-12)
            if keep.any():
                temporal_state[f] = (torch.from_numpy(idx[keep]).to(device),
                                     torch.from_numpy(Q[keep]).to(device=device, dtype=Un[f].dtype),
                                     torch.from_numpy(m[keep]).to(device=device, dtype=Un[f].dtype))

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
    recon_history = []  # ticket 102: second stopping criterion

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
        db_soc_share = _db_domain_share(U_norm, Z_scaled, db_soc_facets, db_soc_total)
        db_sem_share = _db_domain_share(U_norm, Z_scaled, db_sem_facets, db_sem_total)
        if db_soc_share is not None and db_sem_share is not None:
            db_r_k = db_soc_share / (db_soc_share + db_sem_share + 1e-12)
            db_excess = torch.clamp(torch.abs(db_r_k - 0.5) - DOMAIN_BALANCE_TOL, min=0.0)
            domain_balance_loss = torch.mean(db_excess ** 2)
        else:
            domain_balance_loss = torch.tensor(0.0, device=device)

        # Ticket 102: concentration floor, active from epoch conc_warmup on.
        conc_loss = None
        if lambda_conc > 0 and epoch >= conc_warmup:
            if not conc_state:
                _conc_freeze(U_norm, Z_scaled)
            terms = []
            for f, (idx, s) in conc_state.items():
                X = U_norm[f][idx] * _column_weights(U_norm, Z_scaled, f)
                p = X / (X.sum(1, keepdim=True) + 1e-12)
                H = -(p * torch.log(p + 1e-12)).sum(1) / math.log(K)
                terms.append(torch.clamp(conc_gamma * s - H, min=0.0) ** 2)
            if terms:
                conc_loss = torch.cat(terms).mean()

        # Ticket 105: temporal prior, active from epoch temporal_warmup on.
        # p_i = (u_i / sg(column norms)) * sg(column weights), row-normalised:
        # the gradient reaches only the persisting entity's own raw row, so new
        # entities joining an old community are not pushed away (§4.25).
        temporal_loss = None
        if temporal_on and epoch >= temporal_warmup:
            if not temporal_match:
                _temporal_freeze(U_norm, Z_scaled, epoch)
            temporal_loss = temporal_loss_value(
                U_pos, U_scales, {f: _column_weights(U_norm, Z_scaled, f) for f in temporal_state},
                temporal_state)

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
        if conc_loss is not None:
            total_loss = total_loss + lambda_conc * conc_loss
        if temporal_loss is not None:
            total_loss = total_loss + lambda_temporal * temporal_loss
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
        recon_history.append(pure_recon_loss_val)

        # C. RELATIVE EARLY STOPPING CONVERGENCE CHECK
        # Ticket 102: with the concentration term on, (a) no stop until 20
        # epochs after it switches on, and (b) reconstruction must also have
        # stopped changing. Without (b), reconstruction rising while the term
        # falls can look flat in the total: in the 2026-10-02 grid 48% of fits
        # stopped before their baseline at +1.4% mean recon cost; with (b),
        # +0.5% (FINDINGS §31). Without the term the check is unchanged.
        # Ticket 105: the temporal prior uses the same two rules.
        conc_on = lambda_conc > 0
        if (epoch >= 20 and not (conc_on and epoch < conc_warmup + 21)
                and not (temporal_on and epoch < temporal_warmup + 21)):
            prev_loss = loss_history[-21]
            if prev_loss > 0:
                rel_change = abs(loss_history[-1] - prev_loss) / prev_loss
                if conc_on or temporal_on:
                    r_prev = recon_history[-21]
                    if r_prev > 0:
                        rel_change = max(rel_change, abs(recon_history[-1] - r_prev) / r_prev)
                if rel_change < 1e-4:
                    converged = True
                    break

    # Safely detach and move final matrices to CPU
    # Wait to cast to numpy until returned, or keep as torch tensors based on your pipeline 
    # (Module 4 accepts torch tensors and saves them via torch.save)
    U_final = {f: U_norm[f].detach() for f in sorted(active_facets)}  # ticket 85
    Z_final = {rel: Z_scaled[rel].detach() for rel in Z_scaled.keys()}
    
    # Safely format and export the scale masses for the Collapse Check
    U_scales_out = {f: scale.squeeze().detach().cpu().numpy() for f, scale in U_scales.items()}

    # [API UPGRADE] The Clean Diagnostics Dictionary
    diagnostics = {
        "math_loss": pure_recon_loss_val,
        "internal_soc_loss": float((lambda_l1 * sparsity_loss) + (lambda_z_offdiag * z_offdiag_loss)
                                    + (lambda_domain_balance * domain_balance_loss)
                                    + (lambda_conc * conc_loss if conc_loss is not None else 0.0)
                                    + (lambda_temporal * temporal_loss if temporal_loss is not None else 0.0)),
        # Ticket 102: raw, unweighted concentration-floor loss at the last
        # epoch (None if the term never switched on), and its settings.
        "raw_conc_loss": float(conc_loss.item()) if conc_loss is not None else None,
        "conc_settings": {"lambda_conc": lambda_conc, "conc_gamma": conc_gamma, "conc_warmup": conc_warmup,
                          "n_entities": int(sum(len(v[0]) for v in conc_state.values()))},
        # Ticket 105: raw temporal loss at the last epoch (None if the prior was
        # off or never switched on), the frozen T1->T2 matching, and settings.
        "raw_temporal_loss": float(temporal_loss.item()) if temporal_loss is not None else None,
        "temporal_match": dict(temporal_match) if temporal_match else None,
        "temporal_settings": {"lambda_temporal": lambda_temporal, "temporal_warmup": temporal_warmup,
                              "match_shuffles": TEMPORAL_MATCH_SHUFFLES,
                              "match_quantile": TEMPORAL_MATCH_QUANTILE,
                              "prior": temporal_prior.get('source') if temporal_prior is not None else None,
                              "n_entities": int(sum(len(v[0]) for v in temporal_state.values()))},
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


def community_share_vector(U_final, Z_final, structure_threshold=STRUCTURE_SCORE_THRESHOLD):
    """Length-K relation-level mass share per community (ticket 86's ghost
    measure). Equal-weight mean over relations of _relation_community_share,
    renormalised. Used by evaluate_dimensional_collapse and describe_solution."""
    K = next(iter(Z_final.values())).shape[0]
    relation_shares = []
    for rel_key, Z in Z_final.items():
        f1, f2 = RELATION_MAP[rel_key]
        relation_shares.append(_relation_community_share(Z, U_final[f1], U_final[f2], structure_threshold))
    community_share = np.mean(relation_shares, axis=0)
    total_share = community_share.sum()
    return community_share / total_share if total_share > 1e-15 else np.full(K, 1.0 / K)


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

    community_share = community_share_vector(U_final, Z_final, structure_threshold)

    max_share = float(community_share.max())
    collapse_pen = (max(0.0, max_share - max_share_threshold) / (1.0 - max_share_threshold)) ** 2

    # Cast to native Python float — numpy scalars aren't JSON-serializable
    # via stdlib json, which is what Optuna's trial.set_user_attr(...) uses
    # under the hood (ticket 68).
    return float(collapse_pen), float(max_share)
# -----------------------------------------------------------------------------
# SECTION 3: Topological Coherence Diagnostic (The Mean Anchor Rule)
# -----------------------------------------------------------------------------
def coherence_vector(Z_final, semantic_anchors):
    """Per-community anchor cohesion (mean over active anchors of
    Z[k,k] / (0.5*(row_sum+col_sum))). None if no anchor is active."""
    # Filter to evaluate ONLY the anchors active in this specific config
    active_anchors = [rel for rel in Z_final.keys() if rel in semantic_anchors]
    
    # Honest Telemetry Guard (Returns np.nan if no semantic anchors are active)
    if not active_anchors:
        return None 

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
    return np.array(community_mean_scores)


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
    community_mean_scores = coherence_vector(Z_final, semantic_anchors)
    if community_mean_scores is None:
        return 0.0, np.nan

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
def socio_semantic_parts(U_prob, raw_data, active_matrices, max_monopoly, presence_masks=None):
    """Returns (Penalty_A, Penalty_B); evaluate_socio_semantic_reality averages them.
    Penalty_A = item-to-community attribution given the higher-order item.
    Penalty_B = shared-term concentration (formerly called 'hoarding').

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
        return 1.0, 1.0
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

    # Cast to native Python float — Penalty_A/Penalty_B are np.mean(...)
    # results (numpy float64) whenever their source lists are non-empty, and
    # numpy scalars aren't JSON-serializable via stdlib json (Optuna's
    # trial.set_user_attr(...)).
    return float(Penalty_A), float(Penalty_B)


def evaluate_socio_semantic_reality(U_prob, raw_data, active_matrices, max_monopoly, presence_masks=None):
    """Mean of socio_semantic_parts' two penalties, in [0.0, 1.0]."""
    penalty_a, penalty_b = socio_semantic_parts(U_prob, raw_data, active_matrices, max_monopoly, presence_masks)
    return float((penalty_a + penalty_b) / 2.0)

import optuna
import torch
import numpy as np

# -----------------------------------------------------------------------------
# SECTION 4B: Per-Community Domain-Balance Diagnostic (ticket 82, E2 -- Path B)
# -----------------------------------------------------------------------------
def domain_balance_rk(U_prob, presence_masks):
    """Per-community social share r_k = soc/(soc+sem), entity-count-weighted
    over live facets (ticket 82 D2). None if a domain has no live facet."""
    soc_facets = sorted(f for f in presence_masks
                         if FACET_DOMAIN.get(f) == 'social' and f in U_prob
                         and presence_masks[f].sum() > 0)
    sem_facets = sorted(f for f in presence_masks
                         if FACET_DOMAIN.get(f) == 'semantic' and f in U_prob
                         and presence_masks[f].sum() > 0)
    if not soc_facets or not sem_facets:
        return None

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
        return None

    eps = 1e-12
    r_k = soc_share / (soc_share + sem_share + eps)
    return r_k


def evaluate_domain_balance(U_prob, presence_masks, tol=DOMAIN_BALANCE_TOL):
    """
    Outer-loop, U_prob-based domain-balance check (ticket 82 E2, D1-D3
    design; CLAUDE.md §4.18 update / FINDINGS §22 has the full derivation
    and evidence). Since v9.4 (ticket 100) evaluate_complete_solution passes
    compute_weighted_membership's output in place of U_prob, with unengaged
    entities masked out; it is built from U_norm and Z_scaled, both unchanged
    by the ticket-79 transformation described below.

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
    r_k = domain_balance_rk(U_prob, presence_masks)
    if r_k is None:
        return 0.0, 0.0
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
# Weights on two outer-loop terms, set to 0.0 by owner decision (2026-09-30).
# Both terms are still computed and logged unweighted; only their
# contribution to sociological_penalty is switched off, so either can be
# re-enabled by changing its weight. coherence_pen: never fired on the toy
# grid and reads a gauge-dependent input (CLAUDE.md §4.17). domain_balance_pen:
# its input dev_k carries no signal about true domain balance on this corpus
# (FINDINGS §25). Re-evaluate both at 22k scale.
COHERENCE_PEN_WEIGHT = 0.0
DOMAIN_BALANCE_PEN_WEIGHT = 0.0

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

    # Ticket 100 (v9.4): the semantic and domain-balance penalties read
    # reconstruction-weighted membership, not U_prob. U_prob row-normalises
    # U_norm, whose columns have unit norm whether or not the community does
    # any reconstruction for that facet, so an entity with nothing on the used
    # columns read as ~1.0 on an unused one (FINDINGS §30). Unengaged entities
    # (no weight on any used column) are left out of both penalties: they are
    # excluded, not measured (CLAUDE.md ticket 100's residual limitation).
    P_w, engaged = compute_weighted_membership(U_numpy, Z_numpy, presence_masks)
    pm_w = {f: presence_masks[f] & engaged[f] for f in presence_masks if f in engaged}

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
        U_prob=P_w,
        raw_data=raw_data,
        active_matrices=active_matrices_set,
        max_monopoly=max_monopoly,
        presence_masks=pm_w
    )

    # Section 4B (ticket 82, E2 -- Path B). Weighted membership and its masks
    # are already computed above for Section 4 -- reused, not recomputed.
    domain_balance_pen, mean_dev_k = evaluate_domain_balance(
        U_prob=P_w,
        presence_masks=pm_w,
    )

    sociological_penalty = (collapse_pen
                            + COHERENCE_PEN_WEIGHT * coherence_pen
                            + socio_semantic_pen
                            + DOMAIN_BALANCE_PEN_WEIGHT * domain_balance_pen)

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

def compute_weighted_membership(U_final, Z_final, presence_masks=None, engaged_rel=1e-3):
    """Reconstruction-weighted membership (ticket 100). Since v9.4 this is
    what evaluate_complete_solution's semantic and domain-balance penalties
    read (owner decision 2026-10-02), not only a diagnostic.

    U_prob row-normalises U_norm, whose columns all have unit L2 norm whether or
    not the community is used. If a facet's community column carries no
    reconstruction mass (e.g. C1/K=3/trial_0052: atom community 0 has an
    all-zero row in Z['M_Atom_Child'], C1's only atom relation), an entity
    with nothing on the used columns gets U_prob ~ 1.0 on the unused one.
    Here column k of facet f is weighted by the reconstruction mass it carries
    across every relation touching f (gauge-invariant: built from U_norm and
    Z_scaled); columns below 1e-3 of the facet's largest weight get 0. An
    entity is 'engaged' if its weighted mass is >= engaged_rel x the facet's
    median weighted mass over live entities (presence_masks; padded rows of
    the other time slice would otherwise drag the median down); unengaged
    entities get an all-zero row.
    Returns (P, engaged)."""
    P, engaged = {}, {}
    for f, Uf in U_final.items():
        w = np.zeros(Uf.shape[1])
        for rel, Z in Z_final.items():
            f1, f2 = RELATION_MAP[rel]
            if f1 == f:
                w += np.linalg.norm(Z @ U_final[f2].T, axis=1) ** 2
            if f2 == f:
                w += np.linalg.norm(U_final[f1] @ Z, axis=0) ** 2
        w = np.sqrt(w)
        w = np.where(w < 1e-3 * w.max(), 0.0, w) if w.max() > 0 else w
        X = Uf * w
        mass = X.sum(axis=1)
        ref = mass[presence_masks[f]] if presence_masks is not None and f in presence_masks else mass
        pos = ref[ref > 0]
        engaged[f] = mass >= engaged_rel * np.median(pos) if pos.size else np.zeros(len(mass), bool)
        P[f] = np.where(engaged[f][:, None], X / np.where(mass > 0, mass, 1.0)[:, None], 0.0)
    return P, engaged


def temporal_prior_from_model(U1, Z1, raw_prior, raw_data, soc_keys, sem_keys, source=None):
    """Ticket 105: the prior for a later slice from an earlier slice's fitted model
    given as numpy arrays (U_norm per facet, Z_scaled per relation). Reads the
    earlier model's weighted membership (ticket 100) with the earlier slice's
    presence masks and keeps, per global facet, the persisting entities: live in
    the earlier slice, engaged in its model, and live in this slice (raw_data).
    Returns {"facets": {f: {"idx", "P"}}, "K1", "source"}; P rows sum to 1."""
    pm1 = build_presence_masks(raw_prior, soc_keys, sem_keys)
    pm2 = build_presence_masks(raw_data, soc_keys, sem_keys)
    P1, eng1 = compute_weighted_membership(U1, Z1, pm1)
    K1 = int(next(iter(U1.values())).shape[1])
    facets = {}
    for f in TEMPORAL_FACETS:
        if f not in P1 or f not in pm1 or f not in pm2:
            continue
        if len(pm1[f]) != len(pm2[f]):
            raise ValueError(f"facet {f}: {len(pm1[f])} entities in the prior slice, {len(pm2[f])} here")
        idx = np.where(pm1[f] & eng1[f] & pm2[f])[0]
        facets[f] = {"idx": idx, "P": P1[f][idx]}
    source = dict(source or {}, K1=K1, n_persisting={f: int(len(v["idx"])) for f, v in facets.items()})
    return {"facets": facets, "K1": K1, "source": source}


def _js_distance_rows(P, Q):
    """Row-wise Jensen-Shannon distance (natural log), in [0, sqrt(ln 2)]."""
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.where(A > 0, A * np.log(np.where(A > 0, A, 1.0) / np.where(B > 0, B, 1.0)), 0.0).sum(1)
    return np.sqrt(np.clip(0.5 * kl(P, M) + 0.5 * kl(Q, M), 0.0, None))


def temporal_prior_from_seeds(models, raw_prior, raw_data, soc_keys, sem_keys,
                              consensus=True, confidence=True, source=None):
    """Ticket 105 prior variants (CLAUDE.md §4.25, owner decision 2026-10-07): use the
    earlier slice's knee model AND its stability-seed refits. models[0] is the knee
    (reference) model; models[1:] are seed refits; each is (U, Z) as numpy dicts.

    Each seed's communities are aligned to the reference by a Hungarian assignment
    on the cosine between community columns of the weighted membership, stacked
    over every live, engaged entity of every facet (the §S5 way).
      consensus=True  : the prior is each entity's mean aligned profile over the
                        models in which it is engaged (rows renormalised), instead
                        of the knee model's profile.
      confidence=True : each persisting entity gets conf = 1 - (mean JS distance of
                        its aligned per-model profiles from their mean) / sqrt(ln 2),
                        in [0, 1]; the solver multiplies its weight m_i by conf, so
                        entities whose T1 profile is not reproducible are barely pulled.
    Biases (documented in §4.25): the consensus pulls unstable entities towards a
    mixed profile; confidence measures optimisation stability only (not data
    uncertainty, not correctness) and tracks how well connected an entity is."""
    from scipy.optimize import linear_sum_assignment as _lsa
    pm1 = build_presence_masks(raw_prior, soc_keys, sem_keys)
    pm2 = build_presence_masks(raw_data, soc_keys, sem_keys)
    Ps, Es = zip(*[compute_weighted_membership(U, Z, pm1) for U, Z in models])
    K1 = int(next(iter(models[0][0].values())).shape[1])
    facets_all = sorted(f for f in Ps[0] if f in pm1)
    live = {f: pm1[f] & Es[0][f] for f in facets_all}
    cn = lambda A: A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    ref = np.vstack([Ps[0][f][live[f]] for f in facets_all])
    aligned, align_sim = [], []
    for P in Ps:
        cur = np.vstack([P[f][live[f]] for f in facets_all])
        S = cn(ref).T @ cn(cur)
        _, col = _lsa(-S)                       # reference column j <-> this model's column col[j]
        aligned.append({f: P[f][:, col] for f in facets_all})
        align_sim.append([float(S[j, col[j]]) for j in range(K1)])
    facets = {}
    for f in TEMPORAL_FACETS:
        if f not in Ps[0] or f not in pm2:
            continue
        if len(pm1[f]) != len(pm2[f]):
            raise ValueError(f"facet {f}: {len(pm1[f])} entities in the prior slice, {len(pm2[f])} here")
        idx = np.where(pm1[f] & Es[0][f] & pm2[f])[0]
        stack = np.stack([A[f][idx] for A in aligned])            # models x n x K1
        eng = np.stack([E[f][idx] for E in Es])                     # models x n
        w = eng[:, :, None].astype(float)
        mean = (stack * w).sum(0) / np.maximum(w.sum(0), 1.0)
        mean = mean / np.where(mean.sum(1, keepdims=True) > 0, mean.sum(1, keepdims=True), 1.0)
        P = mean if consensus else Ps[0][f][idx]
        entry = {"idx": idx, "P": P}
        if confidence:
            d = np.stack([_js_distance_rows(stack[s], mean) for s in range(len(models))])
            d = np.where(d < 1e-6, 0.0, d)     # rounding residue of the square root, not disagreement
            d = np.where(eng, d, np.nan)
            entry["conf"] = np.clip(1.0 - np.nanmean(d, 0) / np.sqrt(np.log(2.0)), 0.0, 1.0)
        facets[f] = entry
    src = dict(source or {}, K1=K1, n_models=len(models), consensus=bool(consensus),
               confidence=bool(confidence), seed_alignment_similarity=align_sim,
               n_persisting={f: int(len(v["idx"])) for f, v in facets.items()})
    if confidence:
        src["confidence_median"] = {f: float(np.median(v["conf"])) if len(v["conf"]) else None
                                    for f, v in facets.items()}
    return {"facets": facets, "K1": K1, "source": src}


def build_temporal_prior(prior_model_dir, raw_prior, raw_data, soc_keys, sem_keys):
    """Ticket 105: temporal_prior_from_model for an archived model folder
    (U_matrices.pt, Z_core.pt); the source records the folder and file hashes."""
    import hashlib
    paths = {n: os.path.join(prior_model_dir, n) for n in ("U_matrices.pt", "Z_core.pt")}
    U1 = {f: u.detach().cpu().numpy().astype(np.float64) for f, u in torch.load(paths["U_matrices.pt"]).items()}
    Z1 = {r: z.detach().cpu().numpy().astype(np.float64) for r, z in torch.load(paths["Z_core.pt"]).items()}
    source = {"model_dir": os.path.abspath(prior_model_dir),
              "sha256": {n: hashlib.sha256(open(q, "rb").read()).hexdigest() for n, q in paths.items()}}
    return temporal_prior_from_model(U1, Z1, raw_prior, raw_data, soc_keys, sem_keys, source)


def load_selected_priors(selection_path, prior_data_path, data_path, configs):
    """Ticket 105: one T1 prior per config, from select_models.py's
    model_selection.json (the knee model of the chosen K). Model folders are
    found next to it: <results>/<config>/K_<K>/pareto_models/<trial>."""
    selection = json.load(open(selection_path))
    root = os.path.dirname(os.path.abspath(selection_path))
    raw_prior = load_and_validate_data(prior_data_path)
    raw_data = load_and_validate_data(data_path)
    priors = {}
    for c in configs:
        sel = selection.get(c)
        if not sel:
            print(f"[!] {c}: no selected T1 model, so no temporal prior for this config")
            continue
        model_dir = os.path.join(root, c, f"K_{sel['K']}", "pareto_models", sel["chosen"]["trial"])
        soc_keys, sem_keys, _ = get_active_facets(c)
        priors[c] = build_temporal_prior(model_dir, raw_prior, raw_data, soc_keys, sem_keys)
        print(f"[*] {c}: temporal prior from {model_dir} (K1={priors[c]['K1']}, "
              f"persisting {priors[c]['source']['n_persisting']})")
    return priors


def describe_solution(U_final, Z_final, raw_data, soc_keys, sem_keys, anchor_keys, max_monopoly):
    """Per-community detail behind sociological_penalty, logged per trial and
    per archived model (ticket 94). Logging only: nothing here enters the
    objective. *_weighted keys use compute_weighted_membership; since v9.4
    (ticket 100) they are the values the objective uses. The unsuffixed keys
    are the old U_prob readings, kept for comparison with v9.2/v9.3 runs."""
    to_np = lambda d: {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)) for k, v in d.items()}
    U, Z = to_np(U_final), to_np(Z_final)
    pm = build_presence_masks(raw_data, soc_keys, sem_keys)
    act = {k for k in (soc_keys + sem_keys) if k in raw_data}
    U_prob = compute_probability_distributions(U)
    pa, pb = socio_semantic_parts(U_prob, raw_data, act, max_monopoly, pm)
    P_w, engaged = compute_weighted_membership(U, Z, pm)
    pm_w = {f: pm[f] & engaged[f] for f in pm if f in engaged}
    pa_w, pb_w = socio_semantic_parts(P_w, raw_data, act, max_monopoly, pm_w)
    coh = coherence_vector(Z, frozenset(anchor_keys))
    rk = domain_balance_rk(U_prob, pm)
    rk_w = domain_balance_rk(P_w, pm_w)
    as_list = lambda x: None if x is None else [float(v) for v in x]
    return {
        "item_attribution_pen": float(pa),
        "shared_term_concentration_pen": float(pb),
        "item_attribution_pen_weighted": float(pa_w),
        "shared_term_concentration_pen_weighted": float(pb_w),
        "community_share": as_list(community_share_vector(U, Z)),
        "coherence_by_community": as_list(coh),
        "domain_rk": as_list(rk),
        "domain_rk_weighted": as_list(rk_w),
        "unengaged_live_entities": {f: int((pm[f] & ~engaged[f]).sum()) for f in pm if f in engaged},
    }


def _non_dominated_by_converged(study, values):
    """True if no converged COMPLETE trial in the study dominates `values`."""
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        if not t.user_attrs.get("converged", False) or t.values is None:
            continue
        a, b = t.values
        if a <= values[0] and b <= values[1] and (a < values[0] or b < values[1]):
            return False
    return True


def create_optuna_objective(raw_data, soc_keys, sem_keys, anchor_keys, dimensions, K_fixed, max_monopoly=0.85, device="cpu", seed=42, seed_function=None, temporal_prior=None):
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
            
            # Ticket 92: one tuned parameter and a fixed seed, so a repeated
            # lambda_z_offdiag reproduces an earlier fit exactly. Reuse that
            # trial's result without refitting; tagged duplicate_of so the
            # usable-trial counters and the archiver skip it.
            for prev in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
                if (prev.params.get("lambda_z_offdiag") == lambda_z_offdiag
                        and "duplicate_of" not in prev.user_attrs and prev.values is not None):
                    for key, val in prev.user_attrs.items():
                        trial.set_user_attr(key, val)
                    trial.set_user_attr("duplicate_of", prev.number)
                    return tuple(prev.values)

            def _fit_and_evaluate(cap):
                # Re-seed immediately before the fit, so the objective's fit and
                # the archiver's re-run (which re-seeds inside run_inner_solver)
                # start from the same random state.
                set_seeds(seed)
                U_f, Z_f, diag = run_inner_solver(
                    raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys,
                    anchor_keys=anchor_keys, K=K_fixed, params=hyperparams,
                    dimensions=dimensions, device=device, inner_epochs=cap,
                    temporal_prior=temporal_prior)
                ev = evaluate_complete_solution(
                    U_final=U_f, Z_final=Z_f, U_scales_out=diag["U_scales"],
                    raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys,
                    anchor_keys=anchor_keys, max_monopoly=max_monopoly,
                    entropy_threshold=ENTROPY_THRESHOLD, target_coherence=TARGET_COHERENCE)
                return U_f, Z_f, diag, ev

            # 1-2. Fit, then evaluate (Per CLAUDE.md §4.13, evaluate_complete_solution
            # is the single source of truth for the sociological penalty).
            U_final, Z_final, diagnostics, evaluation = _fit_and_evaluate(INNER_EPOCHS)
            epoch_cap = INNER_EPOCHS

            # Ticket 96: a trial that hits INNER_EPOCHS unconverged but is not
            # dominated by any converged trial so far is refitted with
            # EXTENDED_EPOCHS (an exact continuation). Others keep the
            # INNER_EPOCHS cap, so extra cost goes only to trials that could
            # reach the front.
            if (not diagnostics.get("converged", False) and _non_dominated_by_converged(
                    trial.study, (diagnostics["math_loss"], evaluation["sociological_penalty"]))):
                U_final, Z_final, diagnostics, evaluation = _fit_and_evaluate(EXTENDED_EPOCHS)
                epoch_cap = EXTENDED_EPOCHS

            # Ticket 75: flag (do not prune) non-converged trials. Downstream
            # consumers (§S2/S3 hypervolume, §S4 archiver) filter on this attr.
            converged = bool(diagnostics.get("converged", False))
            epochs_run = len(diagnostics.get("loss_history", []))
            trial.set_user_attr("converged", converged)
            trial.set_user_attr("epochs_run", epochs_run)
            trial.set_user_attr("epoch_cap", epoch_cap)
            trial.set_user_attr("conc_settings", diagnostics.get("conc_settings"))  # ticket 102
            trial.set_user_attr("raw_conc_loss", diagnostics.get("raw_conc_loss"))
            trial.set_user_attr("raw_temporal_loss", diagnostics.get("raw_temporal_loss"))  # ticket 105
            trial.set_user_attr("temporal_match", diagnostics.get("temporal_match"))
            trial.set_user_attr("temporal_settings", diagnostics.get("temporal_settings"))
            if not converged:
                print(f"[!] Trial {trial.number} did not converge "
                      f"({epochs_run} epochs, cap {epoch_cap}, K={K_fixed}) — flagged.")

            pure_recon_loss_val = diagnostics["math_loss"]
            sociological_penalty = evaluation["sociological_penalty"]

            # Log all penalties for post-hoc analysis and plotting
            trial.set_user_attr("pure_recon", pure_recon_loss_val)
            trial.set_user_attr("collapse_pen", evaluation["collapse_pen"])
            trial.set_user_attr("coherence_pen", evaluation["coherence_pen"])
            trial.set_user_attr("semantic_pen", evaluation["semantic_pen"])
            trial.set_user_attr("domain_balance_pen", evaluation["domain_balance_pen"])
            trial.set_user_attr("mean_dev_k", evaluation["mean_dev_k"])
            trial.set_user_attr("coherence_pen_weight", COHERENCE_PEN_WEIGHT)
            trial.set_user_attr("domain_balance_pen_weight", DOMAIN_BALANCE_PEN_WEIGHT)
            trial.set_user_attr("collapse_score_raw", evaluation["collapse_score"])
            trial.set_user_attr("weakest_coherence_raw", evaluation["weakest_coherence"])
            trial.set_user_attr("sociological_penalty", sociological_penalty)
            # Ticket 94: per-community detail (logging only, not in the objective).
            for key, val in describe_solution(U_final, Z_final, raw_data, soc_keys, sem_keys,
                                              anchor_keys, max_monopoly).items():
                trial.set_user_attr(key, val)

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
# Ticket 88: DEVICE, MASTER_SEED, ENTROPY_THRESHOLD and TARGET_COHERENCE were
# each re-declared here with values identical to Module 1's (§1.1/§1.2). Deleted
# per CLAUDE.md §7 ("Module 1 owns all constants... delete the later copy") —
# Module 1's declarations are the single source of truth and remain in scope for
# everything textually below them.
print(f"[*] Initializing Module 4 on DEVICE: {DEVICE}")

# Pipeline Meta-Data
CONFIG_IDS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# Domain Constants (Outer-Loop Sociology)
MAX_MONOPOLY = 0.85

# Search Grid Definition
# Ticket 59: T1 has ~2,300 non-zero observations total; K=15 gave ~43,900 free
# parameters (19/observation), and C1's sole anchor M_Parent_Art (160x25, 63
# non-zeros) asked svds for k_svd=min(15,24)=15 components — more than the
# matrix's rank supports, and equal to K so the padding branch never triggers.
K_LIST = [2, 3, 4, 5]  # ticket 99: K=6 dropped (converged in only 1 of 6 configs, 2026-09-30 run)
N_TRIALS = 200

# Global Base Output Directory 
BASE_RESULTS_DIR = os.path.join("results", PIPELINE_VERSION)
# Set to "_<config>" by --config so parallel per-config tasks write separate
# report files; merge_chunk13_reports.py combines them.
REPORT_SUFFIX = ""
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
# Ticket 105: config -> temporal prior (build_temporal_prior), filled by --prior-selection.
TEMPORAL_PRIORS = {}

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
        "device": str(DEVICE),
        "coherence_pen_weight": COHERENCE_PEN_WEIGHT,
        "domain_balance_pen_weight": DOMAIN_BALANCE_PEN_WEIGHT,
        "penalty_membership": "reconstruction_weighted",  # ticket 100, v9.4
        "lambda_conc": LAMBDA_CONC,                       # ticket 102, v9.4
        "conc_gamma": CONC_GAMMA,
        "conc_warmup": CONC_WARMUP,
        "lambda_temporal": LAMBDA_TEMPORAL,               # ticket 105, v10
        "temporal_warmup": TEMPORAL_WARMUP,
        "temporal_match_shuffles": TEMPORAL_MATCH_SHUFFLES,
        "temporal_match_quantile": TEMPORAL_MATCH_QUANTILE,
        "temporal_priors": {c: p["source"] for c, p in TEMPORAL_PRIORS.items()},
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
# Ticket 98: an archived front with fewer distinct models than this is flagged thin.
THIN_FRONT_SIZE = 3

# -----------------------------------------------------------------------------
# HELPER: Objective Factory Wrapper
# -----------------------------------------------------------------------------
def build_objective(raw_data, dimensions, soc_keys, sem_keys, anchor_keys, K, temporal_prior=None):
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
        seed_function=set_seeds,
        temporal_prior=temporal_prior
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
            
            objective = build_objective(raw_data, dimensions, soc_keys, sem_keys, anchor_keys, K,
                                        temporal_prior=TEMPORAL_PRIORS.get(config_id))

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
                                and t.user_attrs.get("converged", False)
                                and "duplicate_of" not in t.user_attrs])
                n_fitted = len([t for t in study.trials if "duplicate_of" not in t.user_attrs])
                if n_usable >= SCOUT_TRIALS:
                    break
                if n_fitted >= max_attempts or len(study.trials) >= 10 * max_attempts:
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
            n_duplicates = len([t for t in study.trials if "duplicate_of" in t.user_attrs])
            n_extended = len([t for t in study.trials if t.user_attrs.get("epoch_cap", INNER_EPOCHS) > INNER_EPOCHS])

            methodology_report[config_id][K] = {
                "pareto_size": len(pts_array),
                "trials_completed": n_complete,
                "trials_pruned": n_pruned,
                "trials_failed": n_failed,
                "trials_converged": n_converged,
                "trials_not_converged": n_not_converged,
                "trials_duplicate": n_duplicates,
                "trials_epoch_extended": n_extended,
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
        
        # Ticket 93: this reference point is built from THIS config's fronts
        # only, so hypervolumes are comparable across K within one config and
        # never across configs (configs reconstruct different relation sets).
        ref_point = np.array([max_math * (1.0 + HV_MARGIN), max_soc * (1.0 + HV_MARGIN)])
        
        # Save exact ref coordinates for reproducibility
        methodology_report[config_id]["_reference_point"] = {
            "math": float(ref_point[0]),
            "soc": float(ref_point[1]),
            "scope": "within_config_only"
        }
        
        k_hypervolumes = {}
        for K, data in pareto_fronts.items():
            pts = data['pts']
            hv = compute_hypervolume(pts, ref_point) if len(pts) > 0 else 0.0
            k_hypervolumes[K] = hv
            methodology_report[config_id][K]["hypervolume"] = float(hv)
            
        surviving_Ks = sorted(k_hypervolumes, key=k_hypervolumes.get, reverse=True)[:top_k_to_keep]
        print(f"\n[***] SCOUT COMPLETE FOR {config_id} | SURVIVING K: {surviving_Ks} [***]")
        for K_rep in K_LIST:
            if K_rep in methodology_report[config_id]:
                methodology_report[config_id][K_rep]["deep_dived"] = K_rep in surviving_Ks

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
            
            objective = build_objective(raw_data, dimensions, soc_keys, sem_keys, anchor_keys, K,
                                        temporal_prior=TEMPORAL_PRIORS.get(config_id))

            # Ticket 75/46: same converged-usable-count budgeting as Scout.
            total_target = SCOUT_TRIALS + DEEP_DIVE_TRIALS
            max_attempts = total_target * 3
            while True:
                n_usable = len([t for t in study.trials
                                if t.state == TrialState.COMPLETE
                                and t.user_attrs.get("converged", False)
                                and "duplicate_of" not in t.user_attrs])
                n_fitted = len([t for t in study.trials if "duplicate_of" not in t.user_attrs])
                if n_usable >= total_target:
                    break
                if n_fitted >= max_attempts or len(study.trials) >= 10 * max_attempts:
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
    report_path = os.path.join(BASE_RESULTS_DIR, f"scout_methodology_report{REPORT_SUFFIX}.json")
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
        report_path = os.path.join(BASE_RESULTS_DIR, f"scout_methodology_report{REPORT_SUFFIX}.json")
    
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
            converged_front = []
            seen_lambdas = set()
            for t in sorted(study.best_trials, key=lambda t: t.number):
                lam = t.params.get("lambda_z_offdiag")
                if (t.user_attrs.get("converged", False) and "duplicate_of" not in t.user_attrs
                        and lam not in seen_lambdas):
                    seen_lambdas.add(lam)
                    converged_front.append(t)
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
                "deep_dived": methodology_report[config_id][K_str].get("deep_dived"),
                "thin_front": len(pareto_trials) < THIN_FRONT_SIZE,
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
                        seed_function=set_seeds,
                        inner_epochs=trial.user_attrs.get("epoch_cap", INNER_EPOCHS),
                        temporal_prior=TEMPORAL_PRIORS.get(config_id)  # ticket 105
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
                        "solution_detail": describe_solution(U_final, Z_final, raw_data, soc_keys,
                                                             sem_keys, anchor_keys, MAX_MONOPOLY),
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
def _pair_tracks(U1, U2, facets, K):
    """Track A/B agreement between two fits (dicts facet -> (N, K) U_norm).
    Communities are matched once on the vertically stacked facets (Hungarian),
    then per facet:
      Track A = 1 - mean over entities of the Jensen-Shannon distance between
                the two row-normalised membership vectors (every live or dead
                entity counts equally, so barely engaged entities add noise);
      Track B = magnitude-weighted mean over entities of the cosine between the
                two raw membership rows (weights = the entity's row sum in U1,
                so strongly engaged entities dominate).
    Both are entity-level; returns (A, B, per-facet A dict, per-facet B dict)."""
    S1_raw = np.vstack([U1[f] for f in facets])
    S2_raw = np.vstack([U2[f] for f in facets])
    S1_col_prob, S2_col_prob = col_normalize(S1_raw), col_normalize(S2_raw)
    cost_A = np.zeros((K, K)); cost_B = np.zeros((K, K))
    for k1 in range(K):
        for k2 in range(K):
            cost_A[k1, k2] = jensenshannon(S1_col_prob[:, k1], S2_col_prob[:, k2])
            cost_B[k1, k2] = 1.0 - (np.dot(S1_raw[:, k1], S2_raw[:, k2]) /
                                    (np.linalg.norm(S1_raw[:, k1]) * np.linalg.norm(S2_raw[:, k2]) + 1e-9))
    _, col_ind_A = linear_sum_assignment(cost_A)
    _, col_ind_B = linear_sum_assignment(cost_B)
    fa, fb = {}, {}
    for f in facets:
        js = jensenshannon(row_normalize(U1[f]), row_normalize(U2[f])[:, col_ind_A], axis=1)
        fa[f] = 1.0 - np.nanmean(js)
        cos = row_wise_cosine_similarity(U1[f], U2[f][:, col_ind_B])
        w = U1[f].sum(axis=1)
        fb[f] = float(np.mean(cos)) if w.sum() == 0 else float(np.average(cos, weights=w))
    return float(np.mean(list(fa.values()))), float(np.mean(list(fb.values()))), fa, fb


def run_dual_track_stability_analysis(filepath):
    """Section 5. Refits every archived model at N_STABILITY_SEEDS seeds with the
    EXTENDED_EPOCHS cap (ticket 96), records how many seeds converged
    (ticket 95: a failed seed no longer aborts the model, it is counted), and
    compares every pair of converged seeds on Track A/B. For the same pairs it
    also computes a chance level: the second fit's entity rows are shuffled
    within each facet, which keeps every facet's membership distribution but
    breaks entity correspondence (2026-10-01 measurement: chance level is
    0.47-0.67 for Track A and 0.40-0.66 for Track B depending on K, far above
    the 0.17 theoretical minimum). A model 'qualifies' (ticket 91, applied by
    select_models.py) only if every seed converged and both tracks exceed the
    largest chance-level pair score."""
    print(f"\n{'='*75}")
    print(f"[***] INITIATING SECTION 5: DUAL-TRACK CONSENSUS ({N_STABILITY_SEEDS} SEEDS) [***]")
    print(f"{'='*75}")

    raw_data = load_and_validate_data(filepath)
    dimensions = raw_data.get('dimensions', None)
    master_stability_report = {}

    for config_id in CONFIG_IDS:
        soc_keys, sem_keys, anchor_keys = get_active_facets(config_id)
        ALL_FACETS = get_required_facets(soc_keys, sem_keys, anchor_keys)
        config_dir = os.path.join(BASE_RESULTS_DIR, config_id)
        if not os.path.exists(config_dir):
            continue
        master_stability_report[config_id] = {}

        for k_folder in sorted(os.listdir(config_dir)):
            if not k_folder.startswith("K_"):
                continue
            K = int(k_folder.split("_")[1])
            manifest_path = os.path.join(config_dir, k_folder, "pareto_models", "experiment_manifest.json")
            if not os.path.exists(manifest_path):
                continue
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

                try:
                    fits, seed_log = [], []
                    for seed_val in STABILITY_SEEDS:
                        set_seeds(seed_val)
                        U_final, Z_final, diagnostics = run_inner_solver(
                            raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys,
                            anchor_keys=anchor_keys, K=K, dimensions=dimensions,
                            params=params, device=DEVICE,
                            # Bound to seed_val (ticket 89): run_inner_solver calls
                            # seed_function() with no argument.
                            seed_function=lambda s=seed_val: set_seeds(s),
                            inner_epochs=EXTENDED_EPOCHS,
                            temporal_prior=TEMPORAL_PRIORS.get(config_id))  # ticket 105
                        conv = bool(diagnostics.get("converged", False))
                        # Ticket 105: each seed matches T1 to its own communities;
                        # recorded so seed-to-seed consistency of the match can be read.
                        seed_log.append({"seed": seed_val, "converged": conv,
                                         "epochs_run": len(diagnostics["loss_history"]),
                                         "temporal_match": diagnostics.get("temporal_match")})
                        if not conv:
                            continue
                        # Per §4.13: recompute the penalty per seed, never reuse Optuna's.
                        evaluation = evaluate_complete_solution(
                            U_final=U_final, Z_final=Z_final, U_scales_out=diagnostics["U_scales"],
                            raw_data=raw_data, soc_keys=soc_keys, sem_keys=sem_keys,
                            anchor_keys=anchor_keys, max_monopoly=MAX_MONOPOLY,
                            entropy_threshold=ENTROPY_THRESHOLD, target_coherence=TARGET_COHERENCE)
                        fits.append({"U": {f: u.detach().cpu().numpy() for f, u in U_final.items()},
                                     "math": diagnostics["math_loss"],
                                     "soc": evaluation["sociological_penalty"]})
                        del U_final, Z_final
                        gc.collect()

                    n_conv = len(fits)
                    metadata["stability_test_seeds_used"] = STABILITY_SEEDS
                    metadata["stability_seed_log"] = seed_log
                    metadata["stability_seeds_converged"] = n_conv
                    result = {"n_seeds": len(STABILITY_SEEDS), "n_converged": n_conv}

                    if n_conv >= 2:
                        math_l = [x["math"] for x in fits]; soc_l = [x["soc"] for x in fits]
                        metadata["objective_variance"] = {
                            "math_loss_mean": float(np.mean(math_l)), "math_loss_sd": float(np.std(math_l, ddof=1)),
                            "soc_penalty_mean": float(np.mean(soc_l)), "soc_penalty_std": float(np.std(soc_l, ddof=1))}
                        metadata["reproducibility_delta"] = {
                            "delta_math_loss": abs(optuna_math_loss - float(np.mean(math_l))),
                            "delta_soc_penalty": abs(optuna_soc_penalty - float(np.mean(soc_l)))}
                        rng = np.random.default_rng(STABILITY_SEEDS[0])
                        A, B, nA, nB = [], [], [], []
                        fa_all = {f: [] for f in ALL_FACETS}; fb_all = {f: [] for f in ALL_FACETS}
                        for i, j in itertools.combinations(range(n_conv), 2):
                            a, b, fa, fb = _pair_tracks(fits[i]["U"], fits[j]["U"], ALL_FACETS, K)
                            A.append(a); B.append(b)
                            for f in ALL_FACETS:
                                fa_all[f].append(fa[f]); fb_all[f].append(fb[f])
                            shuffled = {f: fits[j]["U"][f][rng.permutation(fits[j]["U"][f].shape[0])] for f in ALL_FACETS}
                            a0, b0, _, _ = _pair_tracks(fits[i]["U"], shuffled, ALL_FACETS, K)
                            nA.append(a0); nB.append(b0)
                        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                        metadata["consensus_track_A_unweighted"] = {
                            "global_js_similarity_mean": float(np.mean(A)), "global_js_similarity_sd": sd(A),
                            "chance_mean": float(np.mean(nA)), "chance_max": float(np.max(nA)),
                            "facet_stats": {f: {"mean": float(np.mean(v)), "sd": sd(v)} for f, v in fa_all.items()}}
                        metadata["consensus_track_B_magnitude_weighted"] = {
                            "global_cosine_similarity_mean": float(np.mean(B)), "global_cosine_similarity_sd": sd(B),
                            "chance_mean": float(np.mean(nB)), "chance_max": float(np.max(nB)),
                            "facet_stats": {f: {"mean": float(np.mean(v)), "sd": sd(v)} for f, v in fb_all.items()}}
                        qualifies = (n_conv == len(STABILITY_SEEDS)
                                     and float(np.mean(A)) > float(np.max(nA))
                                     and float(np.mean(B)) > float(np.max(nB)))
                        result.update({
                            "Track_A_Unweighted_Mean": float(np.mean(A)), "Track_A_Unweighted_SD": sd(A),
                            "Track_A_Chance_Mean": float(np.mean(nA)), "Track_A_Chance_Max": float(np.max(nA)),
                            "Track_B_Weighted_Mean": float(np.mean(B)), "Track_B_Weighted_SD": sd(B),
                            "Track_B_Chance_Mean": float(np.mean(nB)), "Track_B_Chance_Max": float(np.max(nB)),
                            "qualifies": bool(qualifies)})
                        print(f"       [+] {n_conv}/{len(STABILITY_SEEDS)} seeds converged | "
                              f"Track A {np.mean(A):.4f} (chance max {np.max(nA):.4f}) | "
                              f"Track B {np.mean(B):.4f} (chance max {np.max(nB):.4f}) | qualifies={qualifies}")
                    else:
                        result["qualifies"] = False
                        print(f"       [!] only {n_conv}/{len(STABILITY_SEEDS)} seeds converged — no consensus computed")
                    metadata["stability_qualifies"] = result["qualifies"]
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
                    master_stability_report[config_id][K][trial_name] = result

                except Exception:
                    with open(os.path.join(trial_dir, "stability_error.log"), "w") as f:
                        f.write(traceback.format_exc())
                    print(f"       [!] Stability test failed for {trial_name}. See logs.")

    report_path = os.path.join(BASE_RESULTS_DIR, f"master_dual_track_stability_report{REPORT_SUFFIX}.json")
    with open(report_path, "w") as f:
        json.dump(master_stability_report, f, indent=4, cls=NumpyEncoder)
    print(f"\n{'='*75}")
    print(f"[***] SECTION 5 COMPLETE. Stability Report saved to: {report_path} [***]")
    print(f"{'='*75}")

if __name__ == "__main__":
    import argparse, hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIG_IDS, default=None,
                    help="run one config only (for a SLURM array); default: all")
    # v10 (ticket 105): T2 by default. chunk12v2 data (ticket 84 grammar damping).
    OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
    ap.add_argument("--data", default=OUT + "Star_extended_matrices_t2_v2.pkl")
    ap.add_argument("--prior-selection", default=None,
                    help="model_selection.json of the earlier slice's run (e.g. results/v9.4.t1_v2/"
                         "model_selection.json); one prior per config from its knee model")
    ap.add_argument("--prior-data", default=OUT + "Star_extended_matrices_t1_v2.pkl",
                    help="the earlier slice's data (presence masks for the prior)")
    ap.add_argument("--lambda-temporal", type=float, default=None,
                    help="weight of the temporal prior (required with --prior-selection "
                         "while LAMBDA_TEMPORAL is uncalibrated)")
    args = ap.parse_args()
    DATA_FILEPATH = args.data
    if args.config is not None:
        CONFIG_IDS = [args.config]
        REPORT_SUFFIX = f"_{args.config}"
    if args.lambda_temporal is not None:
        LAMBDA_TEMPORAL = args.lambda_temporal
    if args.prior_selection is not None:
        if LAMBDA_TEMPORAL <= 0:
            raise SystemExit("--prior-selection needs a positive --lambda-temporal "
                             "(LAMBDA_TEMPORAL is not calibrated yet, ticket 105 test T5)")
        TEMPORAL_PRIORS = load_selected_priors(args.prior_selection, args.prior_data,
                                               DATA_FILEPATH, CONFIG_IDS)
    # Results go to results/v10.0.<slice>.<prior|free>, with fresh study names.
    _slice = os.path.basename(DATA_FILEPATH).replace("Star_extended_matrices_", "").replace(".pkl", "")
    PIPELINE_VERSION = f"v10.0.{_slice}.{'prior' if TEMPORAL_PRIORS else 'free'}"
    BASE_RESULTS_DIR = os.path.join("results", PIPELINE_VERSION)
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(BASE_RESULTS_DIR, "environment_metadata.json"), "w") as f:
        json.dump(get_environment_info(), f, indent=4)
    with open(DATA_FILEPATH, "rb") as f:
        data_sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(__file__, "rb") as f:
        script_sha256 = hashlib.sha256(f.read()).hexdigest()
    run_manifest = dict(get_environment_info(), configs=CONFIG_IDS, data_path=DATA_FILEPATH,
                        prior_selection=args.prior_selection,
                        prior_data=args.prior_data if TEMPORAL_PRIORS else None,
                        data_sha256=data_sha256, script_sha256=script_sha256,
                        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                        slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                        # Ticket 101: fits are deterministic on one machine but differ across
                        # CPU types; record the arithmetic path so runs can be compared.
                        hostname=os.uname().nodename,
                        cpu_model=next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                        if l.startswith("model name")), None),
                        torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
                        ATEN_CPU_CAPABILITY=os.environ.get("ATEN_CPU_CAPABILITY"),
                        MKL_CBWR=os.environ.get("MKL_CBWR"))
    with open(os.path.join(BASE_RESULTS_DIR, f"run_manifest{REPORT_SUFFIX}.json"), "w") as f:
        json.dump(run_manifest, f, indent=4)
    print(f"[*] Configs: {CONFIG_IDS} | data: {DATA_FILEPATH} (sha256 {data_sha256[:12]})")
    run_adaptive_grid(DATA_FILEPATH)
    extract_and_archive_pareto_front(DATA_FILEPATH)
    run_dual_track_stability_analysis(DATA_FILEPATH)