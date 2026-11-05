# Ticket 107, layer 2 — how the numbers were obtained

Calculations behind the layer-2 decisions (design doc `plans/ticket107-m1-chunk12v3-design.md`, addenda 2026-11-02 (1)–(4),
2026-11-03). All scripts are read-only, read the build `psych_trial_v2` on RDS
(`/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v2/`, matrices after layer-1 weighting) and print counts or
shares only, no names of persons. Run from `fullscale_pipeline/m1/` with the `tensor_env` Python
(`/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env/bin/python ../../plans/ticket107_layer2/<script>.py`).
Each script's output is saved beside it as `<script>.txt`.

## Terms

- **Frobenius norm** ‖X‖_F = √(sum of squared cells). Option (a), the decided scale basis, divides each matrix by it.
- **Singular values** s₁ ≥ s₂ ≥ … ≥ 0: the strengths of the rank-1 layers into which a matrix decomposes exactly;
  Σ s_m² = ‖X‖_F². Computed with `scipy.sparse.linalg.svds` (Lanczos), checked once against eigenvalues of XXᵀ / XᵀX
  (agreement to ~1e-15).
- **Best share capturable by K communities alone** = (s₁² + … + s_K²)/‖X‖_F². By the Eckart–Young theorem no rank-K
  reconstruction of that matrix alone does better, so it is an upper bound for the model (whose reconstruction of one
  matrix with K communities has rank ≤ K, and whose communities are shared across matrices).
- **Pull** = relation weight × best share at K, as a percentage of the sum over the matrices of one fit: an upper-bound
  indicator of where fitting gains most. **Realised share** (in a fitted model) and **influence** (change of the
  communities when a matrix is removed or reweighted) are measured only in fits.
- **Relation weights**: decided rule w = 0.5 / n_d (n_d = number of active matrices of the domain; child×article counted
  semantic), i.e. 0.25 for child×parent and child×article and 1/6 for each of the three social matrices.

## Scripts

| Script | What it computes | Used for |
|---|---|---|
| `capturable_shares.py` | best share at K = 2…30 for every matrix (31 singular values, tol 1e-10); pull by domain under the decided weights | L2.1 table at K = 4/10; L2.5 behaviour across K; A1's ratio (child×article / child×parent share 2.5 at K = 4, ~2.1 for K ≥ 8) |
| `layer1_character.py` | what the first layer is: effective number of rows/columns carrying it (1/Σu⁴), cosine with row/column norms, top child hyperedges; σ₁ before and after centring (column- or row-means subtracted, via an implicit operator) | why σ₁-based scaling (option b) measures different things in the two domains |
| `entropy_measures.py` | (E1) Shannon entropy of the cells (shares x²/‖X‖²) as an effective number of cells; (E2) effective number of layers ‖X‖⁴/Σs⁴, with Σs⁴ = ‖XXᵀ‖_F² summed over row blocks (no full SVD) | entropy-based weights considered and not adopted; weight per effective tie (L2.4) |
| `journal_variants.py` | article×journal variants computed **exactly from journal sizes** n_j, because each variant is block-structured: binary partition s² = n_j; same-journal article×article with ties 1: (n−1)² once and 1 (n−1 times); ties 1/(n−1) (Newman 2001): 1 once and 1/(n−1)² (n−1 times); columns × 1/√n: 1 per journal; idf column weight: n·idf² | journal ablation arms J4/J5; why a plain same-journal article×article matrix is worse. Check: the binary case reproduces the SVD value 0.2298 |
| `journal_checks.py` | (1) share of within-journal article pairs with the same OpenAlex primary subfield, by journal size, against random pairs: Σ_c c(c−1)/(n(n−1)); (2) owner's popularity check: per journal, 2015–2025, psychology-primary articles in the working corpus and their share of the journal's corpus articles (journals with ≥ 10 corpus articles) | the qualitative differences between journals; the case for testing journal weightings |
| `weighting_options.py` | pull at K = 4 for the domain-split options A–E, both approaches, four journal variants (arithmetic on the K = 4 shares) | L2.2 decision (option A) |
| `sparse_matrices.py` | author×article single-article persons and their mass; share at K = 4 without them; how much of author×affiliation and of article–organisation links runs through them; article×article degree and concentration; weight per effective tie | L2.4 decisions (keep all persons; arm without article×article) |
| `parallel_analysis.py` | Horn's parallel analysis: singular values against degree-preserving shuffles (each tie keeps its row and value, columns permuted across ties) | shows that plain spectra fail here (real below shuffled) |
| `bethe_hessian_counts.py` | Bethe Hessian community counts (two choices of r) by the inertia shortcut | starting K range (K_max ≈ 20) |
| `bethe_hessian_checks.py` | checks of the shortcut asked for by the external review: residuals, Bunch–Kaufman LDLᵀ, shifted counts, counts over a grid of r | numerical soundness; sensitivity to r |
| `d1_examples.py` | D1 worked examples: the journal shuffle test with trial journal sizes and invented memberships (200 vs 999 shuffles); Monte Carlo precision of a shuffle percentile; critical values and power of single, Dunnett and Bonferroni tests (5 arms vs J1; 5 or 10 groups; own or pooled error) | `D1_journal_preregistration.md` |
| `test_architecture.py` | the staged plan of the first fits: arms, their weights by the rule, fit counts | arm plan |

**Pre-registration documents:** `D1_journal_preregistration.md` (journal weighting: arms, terms, measures D1-M1–M4, decision rule, multiplicity and power).

## The Bethe Hessian shortcut (details in `bethe_hessian_method_note.md`)

1. Each matrix is binarised and turned into a graph: a rectangular X becomes the bipartite adjacency A = [[0, X], [Xᵀ, 0]];
   article×article is used as is; entities without ties are removed.
2. H(r) = (r² − 1)I − rA + D (D = diagonal of tie counts). The estimated number of communities is the number of negative
   eigenvalues of H(r). r is a scale parameter: r = √(Σd²/Σd − 1) (moment-based, Saade, Krzakala & Zdeborová 2014) or
   r = √(mean tie count) (Le & Levina 2015); every r used is > 1.
3. **Shortcut.** The eigenvalues are not computed. By Sylvester's law of inertia, if PHPᵀ = LDLᵀ (L unit lower
   triangular, D diagonal), H and D have the same number of negative eigenvalues. The factorisation comes from SciPy's
   SuperLU (`splu`) with a symmetric fill-reducing ordering (`MMD_AT_PLUS_A`), diagonal pivots only
   (`diag_pivot_thresh=0`) and `SymmetricMode=True`; the count is the number of negative entries on the diagonal of U.
   Seconds instead of the hours Lanczos would take on the 100k–134k-node graphs.
4. **Checks** (`bethe_hessian_checks.txt`): row and column permutations identical; ‖PHP − LU‖/‖H‖ and
   ‖U − diag(U)Lᵀ‖/‖U‖ ≤ 4e-14; identical counts from Lanczos (three graphs) and from a dense LDLᵀ with Bunch–Kaufman
   pivoting (five graphs up to 12.2k nodes); counts unchanged under H ± δI (δ = 1e-6 × max |diagonal|).
5. **Reading** (after the owner's triangulation with Gemini and ChatGPT): the counts are exploratory diagnostics, not
   estimates or upper bounds of the shared K. They depend strongly on r (child×parent 1–51; author×affiliation 27–108;
   article×affiliation 5–23; child×article 1–7). article×journal's 0 is structural: a disjoint union of stars has no
   negative Bethe Hessian eigenvalues for r > 1. K_max = 20 is a motivated initial search range; K = 32 is added to
   stage 2 unconditionally, and the grid is extended if the best K or the band reaches the top.

## Not reproduced by a script

The single-matrix NMF check at K = 4 (captured share ≈ the SVD bound: child×parent 0.0122 vs 0.0123, child×article
0.0303 vs 0.0310, article×journal 0.2298) and the Lanczos timing on article×affiliation (7 min) were one-off checks in
the session of 2026-11-02/03; their numbers are recorded in the design doc.
