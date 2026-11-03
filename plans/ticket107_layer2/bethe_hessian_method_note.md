# Method note for an external check: Bethe Hessian community counts via Sylvester's inertia

Written 2026-11-03 for review by another model. Self-contained; the script is `bethe_hessian_counts.py` (same folder),
its output `bethe_hessian_counts.txt`.

## Context (minimum needed)

A research project fits a collective non-negative matrix tri-factorisation: several relation matrices share one set
of K soft communities (each entity has a non-negative membership vector over K). The trial data are 2,136 scientific
articles (psychology). Seven sparse, non-negative relation matrices, each between two entity types:

| Matrix | Rows × columns | Non-zeros |
|---|---|---|
| child×parent (phrase units × sentence-level statements) | 97,850 × 35,983 (stored transposed) | 231,223 |
| child×article (phrase units × articles) | 97,850 × 2,136 (stored transposed) | 175,086 |
| author×article | 10,052 × 2,136 | 11,131 |
| author×affiliation | 10,052 × 2,511 | 12,512 |
| article×affiliation | 2,136 × 2,511 | 6,051 |
| article×journal (each article has exactly one journal) | 2,136 × 384 | 2,132 |
| article×article (shared persons; symmetric, diagonal empty) | 2,136 × 2,136 | 2,214 stored cells |

Degrees are very unequal: e.g. a few phrase units occur in more than 1,000 of the 2,136 articles, while most occur in one.

**Goal of the computation.** A rough, data-based upper end for the grid of K values to search (the model's K is chosen
later by held-out prediction with a stability gate). Only a range is needed, not a point estimate.

**Why not plain spectra.** Horn's parallel analysis (singular values of each matrix vs degree-preserving shuffles)
failed: real singular values were below the shuffled ones (e.g. 0.38 of the shuffle at the first layer), the known
localisation effect of high-degree nodes in sparse graphs (Krzakala et al. 2013, PNAS).

## Method

1. **Graph.** Each matrix is binarised (non-zero → 1; weights ignored). A rectangular matrix X (m × n) becomes the
   adjacency of the bipartite graph A = [[0, X], [Xᵀ, 0]] on m + n nodes; article×article is used as is. Isolated nodes
   (degree 0) are removed.
2. **Bethe Hessian** (Saade, Krzakala & Zdeborová 2014, NeurIPS; Le & Levina 2015, arXiv 1507.00827):
   H(r) = (r² − 1) I − r A + D, D = diag(degrees). Estimated number of communities = number of negative eigenvalues
   of H(r). Two choices of r were used:
   - moment-based: r = √(Σ d² / Σ d − 1) (Saade et al.; the square root of the non-backtracking spectral radius
     estimate);
   - mean-degree: r = √(mean degree) (Le & Levina's "BHm").
3. **The shortcut: counting negative eigenvalues without computing them.** By Sylvester's law of inertia, if
   P H Pᵀ = L D' Lᵀ with P a permutation, L unit lower-triangular and D' diagonal, then H and D' have the same number
   of negative eigenvalues. The factorisation was obtained with SciPy's SuperLU (`scipy.sparse.linalg.splu`) using
   `permc_spec="MMD_AT_PLUS_A"` (a symmetric fill-reducing ordering), `diag_pivot_thresh=0.0` (always take the diagonal
   pivot, i.e. no row pivoting) and `options={"SymmetricMode": True}`. Then H = Pᵀ L U P with U = D' Lᵀ, and the count is
   the number of negative entries on the diagonal of U. Per run the script checks that the row and column permutations
   are identical (`perm_r == perm_c`: true in every run) and reports the smallest |pivot| (smallest 3e-3; no zero
   pivots).
4. **Validation.** On three matrices the count was also computed directly with Lanczos (`scipy.sparse.linalg.eigsh`,
   smallest algebraic eigenvalues, moment-based r): article×article 15, article×affiliation 20, article×journal 0 —
   identical to the inertia counts. Lanczos took up to 7 minutes on the 4,614-node graph and would take hours on the
   100k–134k-node graphs; the inertia count took at most 20 s.

## Results (number of negative eigenvalues; moment-based r / mean-degree r)

child×parent 4 / 23; child×article 5 / 1; author×article 41 / 82; author×affiliation 36 / 100; article×affiliation
20 / 23; article×article 15 / 21; article×journal 0 / 0. The project reads article×journal's 0 as an artefact (the
graph is a disjoint union of stars, one per journal) and the semantic counts as unreliable (strong dependence on r).
Proposed use: K_max ≈ 20 (order of the stable article-level counts), extended if the best K found is at 20.

## Questions for the reviewer

1. Is the inertia shortcut valid here? In particular: H is symmetric and indefinite; is LU without row pivoting under
   a symmetric ordering numerically safe enough for an inertia count (risk of pivot growth or a wrong sign), and are the
   checks above (identical permutations, no tiny pivots, agreement with Lanczos on three graphs) sufficient? Would an
   LDLᵀ with Bunch–Kaufman pivoting (e.g. MUMPS, CHOLMOD-like tools) or a few shifted checks be advisable?
2. Is the Bethe Hessian count applicable to bipartite graphs built this way (A's spectrum is symmetric, ±λ), or does
   it double-count or miss communities?
3. Which choice of r is appropriate for graphs with very unequal degrees (degree-corrected structure)? Is there a
   corrected variant that should be preferred?
4. Is binarisation acceptable for a rough range, given that the model is fitted on weighted matrices?
5. Is it reasonable to use per-matrix counts as a rough upper end for a K shared by all matrices?
