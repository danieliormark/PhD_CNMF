# Glossary — terms and notation for the factorisation and its inputs

Created 2026-10-29 at the owner's request. One entry per agreed term: definition, notation, where it lives in the code,
and an example where one helps. **Rules:**
1. Explanations of the model use these terms and symbols only. A quantity without an entry is either named in full or
   added here first.
2. **The glossary is extended whenever a new concept arises** (a new term, a new symbol, a new code location). The new
   entry is added in the same commit as the work that introduced it, and the owner is told.
3. One symbol, one meaning. Before a symbol is used for something new, check the notation table below; if it is taken,
   choose another and record it.
4. Status of each entry: **code** (exists in `chunk13_execution/chunk13v9.py`, line numbers as of 2026-10-29), **decided**
   (agreed for the M1 builder or the next chunk13 version, not yet coded), **toy** (true of the toy-corpus pipeline only).

Words not to use (owner, 2026-09-30 to 2026-10-29): "hub" (say "an entity with N ties", "a frequent atom");
"neighbour" (say "the hyperedges it has a non-zero similarity with in W"); "length" of a column (say Euclidean norm);
"moving a row" (say "changing the values of the cells in a row"); "a community's factor"; "gradient flows" (say
"the gradient of the loss with respect to …", or "computed without gradient").

## Notation table

| Symbol | Meaning | Entry |
|---|---|---|
| f, g | facets (child, parent, article, author, affiliation, journal) | Facet |
| i, j, m | entities (rows); i, j a pair, m any other entity | Entity |
| k, l | communities (columns of a factor matrix; rows/columns of a core matrix) | Community |
| K | number of communities | Community |
| N_f | number of entities of facet f (rows of U^(f)) | Factor matrix |
| r = (f, g) | a relation between facets f and g | Relation |
| X^(f,g) | data matrix of relation (f, g), N_f × N_g | Data matrix |
| X̂^(f,g) | its reconstruction | Reconstruction |
| U^(f) | factor matrix of facet f, N_f × K | Factor matrix |
| U^(f)_ik | cell of U^(f): entity i, community k | Factor matrix |
| ‖U^(f)_·k‖ | Euclidean norm of column k of U^(f) | Euclidean norm of a column |
| Ũ^(f) | normalised factor matrix | Normalised factor matrix |
| Z^(f,g) | core matrix of relation (f, g), K × K | Core matrix |
| Z̃^(f,g) | scaled core matrix | Core matrix |
| w^(f)_k | community weight of column k in facet f | Community weight |
| p_i | membership shares of entity i (length K) | Membership shares |
| W | similarity matrix of a regulariser (one facet, N × N) | Similarity matrix |
| W_ij | similarity of entities i and j | Similarity matrix |
| D_i | total similarity of entity i, Σ_j W_ij | Row normalisation of W |
| N_W | number of entities with at least one non-zero W_ij | Regulariser |
| λ_child, λ_article, λ_conc | weights of the child×child regulariser, the article×article regulariser, the concentration floor | Regulariser; Concentration floor |
| τ | similarity threshold for keeping a tie in W | Tie selection |
| idf(v) | inverse document frequency of word v | idf |
| n_p | number of articles of person p (in the subset) | article×article similarity |

**Resolved clash (2026-10-29):** "d" had been used both for the distance between membership shares and for an entity's
total similarity. Now: the distance is written out, ‖p_i − p_j‖², and the total similarity is D_i. Core matrices are
indexed by relation, Z^(f,g), never Z_ij, because i and j denote entities.

## Data and model structure

**Facet** — a type of entity: child hyperedge, parent hyperedge, article, author, affiliation, journal. Each facet has
one factor matrix. *(decided: ticket 107 design doc §1–2)*

**Entity** — one row of a facet: one child hyperedge, one article, one person. Indexed i, j, m.

**Child hyperedge** — the child-level semantic unit: children and former cousins merged; identity = the set of its
typed atoms (`content_key` in G3 output), so the same word set is one entity wherever it occurs. *(decided, design doc §1)*
Example: `(dummy_sibling help/P/en)` and `(dummy_cousin help/P/en)` are one child hyperedge.

**Relation** — a pair of facets whose ties form a data matrix, r = (f, g), e.g. (child, article). `RELATION_MAP`
(line 191) maps each relation to its two facets.

**Data matrix X^(f,g)** — rows = entities of f, columns = entities of g, cell = the weighted tie. Each data matrix is
scaled to Frobenius norm 1 before fitting (so every relation enters the loss with the same total size; line ~860,
`target_norm_sq = 1.0`).

**Community** — one of the K latent groups; column k of every factor matrix. Indexed k, l.

**Factor matrix U^(f)** — N_f × K, non-negative; cell U^(f)_ik = how strongly entity i loads on community k. These cells
are parameters, changed by Adam (`U_raw` / `U_pos`, lines 785–797). Every column has N_f cells.

**Euclidean norm of a column, ‖U^(f)_·k‖** — √(Σ_m (U^(f)_mk)²), over all N_f cells of column k. For a single column this
equals its Frobenius norm. Grows in proportion to the values: if every cell doubles, the norm doubles. Code: `col_norms`,
`U_scales` (lines 801–802). The norms themselves are not determined by the data (ticket 79): they can change without
changing the fit.

Example (child facet, 3 hyperedges, K = 2):

| | community 1 | community 2 |
|---|---|---|
| A | 0.8 | 0.2 |
| B | 0.3 | 0.6 |
| C | 0.5 | 0.5 |
| **Euclidean norm** | **√(0.64+0.09+0.25) = 0.990** | **√(0.04+0.36+0.25) = 0.806** |

**Normalised factor matrix Ũ^(f)** — every cell divided by its column's Euclidean norm, Ũ^(f)_ik = U^(f)_ik / ‖U^(f)_·k‖,
so every column has norm 1. Computed at every training step (`U_norm`, line 803); the reconstruction uses it. Any cell's
change changes its column's norm and so every normalised value in that column, by the common ratio old norm / new norm.

Example (from the table above): A's normalised values are (0.8/0.990, 0.2/0.806) = (*0.808*, *0.248*); C's are
(0.505, 0.620). If A's community-1 cell drops from 0.8 to 0.7, the column-1 norm becomes 0.911 and C's normalised
community-1 value becomes 0.5/0.911 = **0.549**, although C's cell did not change.

**Core matrix Z^(f,g)** — K × K, non-negative; cell Z^(f,g)_kl = how strongly community k of facet f connects to
community l of facet g in relation (f, g). Parameters, changed by Adam (`Z_raw` / `Z_pos`). **Scaled core matrix
Z̃^(f,g)** — Z^(f,g)_kl × ‖U^(f)_·k‖ × ‖U^(g)_·l‖ (`Z_scaled`, lines 806–809); with it, Ũ^(f) Z̃^(f,g) Ũ^(g)ᵀ =
U^(f) Z^(f,g) U^(g)ᵀ, so normalising the columns changes nothing in the reconstruction (scale invariance, CLAUDE.md §4.2).

**Reconstruction X̂^(f,g)** — Ũ^(f) Z̃^(f,g) Ũ^(g)ᵀ. Each cell is a sum over all community pairs (k, l) of
Ũ^(f)_ik × Z̃^(f,g)_kl × Ũ^(g)_jl.

**Community weight w^(f)_k** — the size of the part of the reconstruction that uses column k of facet f: in every
relation the facet takes part in, take all products that involve column k, and compute their Euclidean norm; sum the
squares over relations and take the root. Columns below 1e-3 of the largest weight get 0. Computed without gradient
(`torch.no_grad()`): its current value is used, but Adam's update cannot act through it. Code: `_column_weights`
(line 660, training), `compute_weighted_membership` (line 1750, evaluation). *(ticket 100)*
Example: a column with sizeable cells but a near-zero row in every core matrix it meets has weight ≈ 0.

**Membership shares p_i** — entity i's normalised values multiplied by the community weights, then divided by their
sum so that the K shares sum to 1: p_ik = Ũ_ik w_k / Σ_l Ũ_il w_l. Also called "weighted membership" (ticket 100).

Example (A above): with weights (1, 1), p_A = (0.808, 0.248) / 1.056 = (**0.765**, 0.235); with weights (1, 0.5),
p_A = (0.808, 0.124) / 0.932 = (**0.867**, 0.133).

## Training

**Loss (total loss)** — the number training minimises: reconstruction loss + every penalty term, each multiplied by its
weight. Computed once per training step (the forward pass).

**Reconstruction loss** — Σ over relations of (domain weight × ‖X^(f,g) − X̂^(f,g)‖²), computed with the trace form
(lines 840–862).

**Term (of the loss)** — one summand of the loss: the reconstruction of one relation, the concentration floor, or one
pair of entities in a regulariser.

**Gradient** — for one parameter (one cell of a factor matrix or core matrix): the rate at which the total loss changes
when that parameter changes slightly. A term has no gradient of its own; one speaks of the gradient of a term *with
respect to* a parameter, and the total gradient of a parameter is the sum of these over all terms.
Example: a gradient of 0.5 means raising the cell by 0.001 raises the loss by about 0.0005.

**Backpropagation** — the automatic computation of all gradients from the recorded way the loss was computed from the
parameters (PyTorch autograd).

**Adam** — the optimiser: changes every parameter against its gradient, using running averages of past gradients and a
per-parameter step size; then every cell is clamped to ≥ 1e-7 (lines 891–898).

**Training step / epoch** — one forward pass, one backpropagation, one Adam update. The code's loop counter calls it
an epoch (line 785).

**Computed without gradient (`torch.no_grad()`, `.detach()`, stop-gradient)** — a quantity is used by its current value,
but backpropagation treats it as a constant, so Adam cannot change parameters through it. Standard PyTorch tool.
*Not* used on the column norms in the regularisers (2026-10-29): it would not keep the effect of an update away from
other entities' normalised values (design doc addendum 2026-10-27 (2), correction).

**Warm-up** — the number of training steps before a penalty term is switched on: 300 for the concentration floor
(code) and the child×child regulariser (decided).

## Regularisers *(decided, not yet coded)*

**Regulariser** — a penalty term that pulls entities with a non-zero similarity in W towards similar membership shares:
R = λ · (1/N_W) · Σ_i Σ_j (W_ij / D_i) · ‖p_i − p_j‖². Not a data matrix: it is not reconstructed. Two in the trial:
child×child (word similarity) and article×article (shared authors).

**Similarity matrix W** — N × N over the entities of one facet; W_ij = similarity of i and j (0 = no tie); symmetric;
fixed before fitting.

**Row normalisation of W** — each entity's similarities divided by its total similarity D_i = Σ_j W_ij, so they sum to
1 per entity.

Example: A has W_AB = 0.6, W_AC = 0.3 (D_A = 0.9) → 0.667 and 0.333; B and C have only their tie to A → 1 each.

**Distance between membership shares** — squared Euclidean distance ‖p_i − p_j‖² = Σ_k (p_ik − p_jk)²; between 0
and 2. Hellinger distance as a check.

**N_W** — the number of entities of the facet with at least one non-zero W_ij; the sum of the entity terms is divided by
it, so λ means the pull on an average entity.

**idf(v)** — ln(number of child hyperedges / number of child hyperedges containing word v). Computed over the trial's
child hyperedges; numbers, citation tokens and URL tokens are masked (no idf, no tie). A word occurs at most once in a
child hyperedge, so tf-idf = idf.

**idf-weighted Jaccard (child×child similarity)** — Σ idf over the words two child hyperedges share ÷ Σ idf over all
words of the two. Model names included.
Example (Psychology trial): "measure self_report versus" / "measure self_report": idf measure 5.7, self_report 6.9,
versus 5.1 → (5.7 + 6.9) / (5.7 + 6.9 + 5.1) = **0.71**.

**Tie selection** — for each child hyperedge: the 5 most similar child hyperedges reaching τ = 0.4, plus every further
one with exactly the 5th value; if fewer reach τ, only those. A tie counts when either hyperedge selected the other
(the OR rule; a symmetric k-nearest-neighbour graph).

**article×article tie weight** — w(A,B) = Σ over persons p shared by articles A and B of 1/(n_p − 1), divided by the
number of distinct persons on A or B; n_p = number of trial articles of person p. The numerator is Newman's (2001)
collaboration weight applied to the article side of the article×person table (a person on many articles adds less to
each pair of them); the division by the union of the two author sets is a Jaccard-type normalisation (large teams add
less). Equals plain Jaccard when every shared person has exactly 2 articles. Gives the cells of the article×article
matrix (approach 2, main solution); the same weights would give W of the fallback regulariser.
Example: A by {x, y, z}, B by {x, w}; x on 2 trial articles → 1/(2−1) / 4 = **0.25**; x on 6 → 1/5 / 4 = **0.05**.
(Corrected 2026-10-29: was called "damped Jaccard", and was wrongly marked as a regulariser in the trial.)

## Other penalty terms *(code, toy calibration)*

**Concentration floor** — λ_conc · mean_i max(0, γ s_i − H(p_i))², over child-level entities; s_i = normalised entropy
of the community distribution entity i receives from the articles that use it, frozen at warm-up; H = normalised
entropy of p_i (lines 824–835; ticket 102). λ_conc = 0.1, γ = 0.8 were calibrated on the toy corpus; to be recalibrated
on the Psychology trial.
