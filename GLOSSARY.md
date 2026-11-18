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
| part(k,l) | reconstructed tie weight of relation r attributed to the community pair (k, l) | Reconstructed tie weight |
| T_r | total reconstructed tie weight of relation r (Σ_kl part(k,l)) | Reconstructed tie weight |
| h_k(r), p_k(r), P_k(D), P_k(f) | half-credit share, relative presence, presence in domain D, facet presence | Presence |
| π_f | renumbering (permutation) of facet f's communities | Numbering freedom; Alignment |

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
Example (invented authors): article A by Smith, Lee, Kim; article B by Smith, Patel. Shared person: Smith; distinct
persons on A or B: 4. Smith on 2 trial articles (A and B only) → (1/(2−1)) / 4 = **0.25**; Smith on 6 trial articles →
(1/5) / 4 = **0.05**. Two 20-author articles sharing Smith (on 2 trial articles) → 1 / 39 = **0.026**.
Property: each person adds a total of exactly 1 to each of their articles, however many articles they have.
(Corrected 2026-10-29: was called "damped Jaccard", and was wrongly marked as a regulariser in the trial.)

## Other penalty terms *(code, toy calibration)*

**Concentration floor** — λ_conc · mean_i max(0, γ s_i − H(p_i))², over child-level entities; s_i = normalised entropy
of the community distribution entity i receives from the articles that use it, frozen at warm-up; H = normalised
entropy of p_i (lines 824–835; ticket 102). λ_conc = 0.1, γ = 0.8 were calibrated on the toy corpus; to be recalibrated
on the Psychology trial.
**[Ticket 107 re-specification, decided 2026-11-07]** Spread in bits (not normalised by log₂K); minimum = γ ×
(between-article spread + ω × within-article spread) of the article mix; ω and β arms; see the design doc, step 4.

**Spread** — entropy of a membership in bits (log base 2; the unit is Shannon's 1948 "bit", word credited by him to
J. W. Tukey). 0 = all in one community; 1 bit = as spread as an even split between 2 communities; log₂K = even over all
K. **Effective number of communities** = 2^spread (Hill number of order 1). Example: (0.9, 0.1, 0, 0) → 0.47 bits →
1.38 communities.

**Tie weight** — the value of a non-zero cell of a relation matrix. **Entities tied to entity i in relation r** — the
entities with a non-zero cell in i's row (or column) of relation matrix r.

**Relation mix (relation-r mix of entity i)** — the tie-weighted average of the weighted memberships (ticket 100) of the
entities tied to i in relation r. Examples: a child hyperedge's article mix (child×article; the floor's mix); an
article's hyperedge mix, author mix, journal membership. Not to be called "neighbours".

**Within-article / between-article spread** — the article mix's spread splits exactly into the mean of the articles'
own spreads (within) plus the rest (between; a generalised Jensen–Shannon divergence, never negative).

**Excess spread** — max(0, own spread − ρ × the spread of the relation mix); for an article, against its most spread
relation mix (an article is over-spread only if none of its relations supports its spread). Diagnostic of smearing;
target of the possible ceiling term S3 (ticket 107 step 5, decided 2026-11-07).

## Column weights and reconstructed tie weight *(documented 2026-11-07, ticket 107 step 6)*

**Column weight w_k(f)** (ticket 100; `_column_weights` / `compute_weighted_membership`, chunk13v10.py) — for facet f
and community k: w_k² = Σ over relations r touching f of ‖slice_k‖², where slice_k = U_f[:,k] · (Z[k,:] · U_gᵀ) is the
part of r's reconstruction U_f Z U_gᵀ that comes from column k of U_f (U column-normalised, L2 norm 1; Z = Z_scaled;
g = r's other facet). Because ‖U_f[:,k]‖ = 1, ‖slice_k‖² = Z[k,:] (U_gᵀU_g) Z[k,:]ᵀ (U_gᵀU_g = Gram matrix of g's
columns, cell (l,m) = cosine between columns l and m). Counts between-community core cells too (Z[k,l], l ≠ k).
Example: K = 2, Z = [[0.8, 0.1], [0.2, 0.5]], Gram [[1, 0.3], [0.3, 1]] → w = (0.84, 0.59). **Cut-off:** w_k below
1e-3 × the facet's largest weight is set to 0 (no derivation recorded; part of item (l), weak attribution).
**Detached:** in in-loop terms the weights are recomputed each epoch but treated as constants in the gradient, so a term
cannot be lowered by changing a community's weight. **Not a share:** community parts overlap, so squared weights do not
add up to the reconstruction's squared norm. **Weighted membership** of entity i = its column-normalised row × w,
rescaled to sum to 1 ≈ the share of i's reconstructed ties passing through each community (row i of slice_k has size
u_ik · w_k per relation); counterpart of Lin et al.'s p(k|i) ∝ p(i|k) p(k). Why needed: every U column has norm 1 even
when the community carries nothing in that facet; toy example (FINDINGS §30): atom "computational", raw row (0.0048,
3e-7, 3e-6), read 0.999 in a community with zero reconstructed data. The cut-off does not remove the community (it stays
in the fit, in K, and possibly alive in other facets); it removes it from that facet's membership reading. Entities whose
weighted mass falls below 1e-3 × the live median are **unengaged** and are left out of membership-based measures (not
reassigned). Squared size (these weights) leans towards the squared-error loss; summed size towards generalised KL.

**Reconstructed tie weight** — the sum of the cells of a relation's reconstruction. It splits exactly into non-negative
parts per community pair (k, l): part(k,l) = Z[k,l] × Σ_i U_f[i,k] × Σ_j U_g[j,l] (same value from raw or normalised
matrices; unchanged by the scale freedom of ticket 79). **Wording:** "community k carries x% of the reconstructed tie
weight of relation r" — a share of the model's reconstruction, not of the data, and not "explained variance" (no
orthogonal split, no reference model). Report it with the relation's fit (held-out score). Under generalised KL,
stationary points of two-factor NMF preserve the data's row and column sums (Ho & Van Dooren 2008, statement checked;
for our collective tri-factorisation at most approximately).

**Domain of a facet / of a relation matrix** — facets: social (articles, authors, affiliations, journals) or semantic
(hyperedges). Relation matrices (decided 2026-11-07, ticket 107 step 6): **semantic** — child×article, child×parent;
**social** — approach 1: author×article, author×affiliation, article×journal; approach 2: article×article,
article×affiliation, article×journal. A community's presence in a domain = mean relative presence over that domain's
relation matrices (equal weight). The toy classified only facets.

**Presence (relative presence p_k(r); presence in a domain P_k(D))** — decided 2026-11-07. h_k(r) = community k's
half-credit share of relation r's reconstructed tie weight; p_k(r) = K × h_k(r) (1 = average community, 0 = absent);
P_k(D) = mean of p_k(r) over the domain's relation matrices. Reads as "how much of the modelled ties of that domain is
built with community k, compared with an average community". **Mono-domain community:** P_k in one domain < 0.1 (less
than a tenth of an average community's share: nearly absent) and ≥ 0.5 in the other (at least half an average share:
clearly present), in most seeds, not caused by a relation fitted at chance. Full definition and example: design doc,
step 6.

## Community numbering and link structure *(2026-11-17, ticket 107, RL-253–RL-259)*

**Numbering freedom** — with free core matrices, renumbering the communities of ANY one facet f (reordering the columns of
U^(f) by a permutation π_f) and reordering the matching side of every core matrix touching f leaves every reconstructed
value unchanged: (U^(f) P)(Pᵀ Z) = U^(f) Z, because P Pᵀ = I. So the loss never fixes that community k of one facet is
community k of another (the owner's decided reading R-A, 2026-11-06); only the start (NNDSVD aligns children, parents and
articles; propagation aligns authors, organisations and journals to articles) and the optimisation path do. Verified on
a real fit (RL-259, max change 1e-16). What would fix it: a term that treats matching numbers specially (the toy's λ_z,
O3, a diagonal-constrained core). Earlier wording "single-relation facets only" (RL-253/255) was wrong.

**Alignment (renumbering)** — DECIDED 2026-11-17 (owner: "I approve the renumbering"; "I agree with proposal a"): after
each fit, at read time (saved fits untouched), every facet's numbering is aligned to the articles' through the chain of
relations (children → articles via child×article; parents → children via child×parent; authors → articles via
author×article; organisations → authors via author×affiliation (approach 1) or → articles via article×affiliation
(approach 2); journals → articles via article×journal). Each step: one-to-one assignment (Hungarian) maximising the tie
weight on matching numbers. Always applied (option a), with the **match strength** reported beside every reading.
Readings that combine facets by community number are reported aligned, with the fitted numbering beside them. History:
chunk13v3/v4 relabelled leaf facets physically; chunk13v9 relabelled per relation at read time with a confidence rule
(CLAUDE.md §4.20, FINDINGS §14/§16). Explanation E2.

**Match strength** — after alignment, the share of a relation's reconstructed tie weight on matching numbers,
Σ_k part(k,k) ÷ T_r (1 = one-to-one; low = genuinely spread, so the alignment is partial).

**Off-diagonal share** — decided stage-0 diagnostic (2026-11-06/07): 1 − Σ_k part(k,k) ÷ T_r (tie-weight form, main),
Σ_{k≠l} Z̃_kl² ÷ Σ Z̃² (core-cell form, beside it). Depends on the numbering: read it as fitted AND after alignment.
Per community: 1 − part(k,k) ÷ (h_k(r) T_r).

**Effective number of partner communities** — per relation: for each row community k, exp of the entropy of how its
row of parts divides over the column communities (all in one → 1; 50/50 over two → 2; 90/10 → 1.4; even over four → 4),
averaged over k weighted by the row's tie weight. Does not depend on numbering. Measures genuine spread: one community
of one facet tied to several communities of another (pass 1: KL ≈ 1–1.8 at every K; squared error 3–5 at K* and rising
with K; RL-259).

**Detached community (facet detachment)** — facet presence P_k(f) ≥ 0.5 while a facet g tied to f directly has P_k(g) <
0.1 (lines of the mono-domain rule). Depends on the numbering (RL-253, RL-255).

**Paired 2-SE rule** — DECIDED 2026-11-17 for the diagonal-start test (owner: "I approve the 2se rule"): difference
between arms per seed (same start per seed), mean over seeds, SE = SD ÷ √n; better / worse only beyond 2 SE.
Explanation E3.

## Explanations — answers to frequent questions *(started 2026-11-17 at the owner's request)*

Fuller answers to questions that come up repeatedly. Before explaining one of these again, the assistant points to the
entry to re-read and adds only what is new. Each entry: the question, the answer, a worked example, and its limits.
**These entries are explanations, not the record** (owner, 2026-11-18). Where an entry differs from the code, RUN_LOG,
FINDINGS or the design doc, those are right and the entry is corrected.

### E1. Which community does a tie belong to? (attribution of reconstructed tie weight to community pairs)

*Asked 2026-11-17 as "how do we know the tie × community membership?". Naming:* a tie does not belong to one community.
What the model gives is a split of each **reconstructed** tie value over **pairs** of communities, one community from
each facet of the relation. Name used here: **attribution of reconstructed tie weight to community pairs** (the parts
part(k,l)).

**How.** The reconstructed value of a cell (author i, organisation j) is a sum over community pairs:
x̂_ij = Σ_k Σ_l U_ik × Z_kl × U'_jl (U = authors' factor matrix, U' = organisations', Z = the relation's core matrix). Each
term is the part of that cell built through the pair (author community k, organisation community l). Summed over all
cells of the relation (observed rows × all columns, with or without an observed tie, hidden cells included) this gives
part(k,l) = Z_kl × Σ_i U_ik × Σ_j U'_jl, and Σ_kl part(k,l) = T_r. From the K × K table of parts: presence (half credit:
own block + half of every shared block), off-diagonal share, effective number of partner communities.

**Worked example** (made up, K = 3). Author i: loadings 0.8 on community 1, 0.2 on 2. Organisation j: 1.0 on community 3.
Core: Z_13 = 0.5, Z_23 = 0.1, all other cells 0. x̂_ij = 0.8 × 0.5 × 1.0 + 0.2 × 0.1 × 1.0 = 0.40 + 0.02 = 0.42 → 0.40
to the pair (1, 3), 0.02 to (2, 3). If, summed over all cells, author community 1's whole row of parts lies in
organisation column 3, author community 1 has one partner community.

**Weighting (asked 2026-11-17: "do we account for domain and/or facet weights, as well as other weighting?").**
- *Indirectly, yes:* everything that shapes the fit shapes U and Z and so the parts — tie values from the build (tf,
  damping and other weighting schemes), each relation's Frobenius normalisation, and the relation weights w_r in the
  loss (0.5 ÷ number of relations per domain; journal multiplier). Example: a larger weight on article×journal makes the
  fit spend more on journals, which changes U and Z.
- *Directly, no:* the parts of a fit are computed from its U and Z for each relation on its own and read as shares of
  that relation's T_r; w_r is not applied. Presence in a domain is the plain mean over the domain's relations (equal
  weight, decided 2026-11-07); facet presence the plain mean over the relations touching the facet; weighted
  memberships (column weights) sum over relations without w_r; no entity is weighted.
- Other conventions: core cells at the positivity floor (≤ 1e-7) count as 0; rows with no data (missing affiliation or
  journal) are left out of the row sums; the effective number of partners averages communities by their tie weight.
- Trade-off: reading with w_r would show what the loss emphasised; reading without shows each relation on its own terms
  (the decided choice for presence).

**The table of parts is not the core matrix** (asked 2026-11-17). Z_kl is the strength of the tie between community k of
one facet and community l of the other **per unit of loading**; part(k,l) = Z_kl × (sum of the first facet's loadings on
k) × (sum of the second facet's loadings on l) is how much reconstructed tie weight actually goes through the pair.
Example: Z_13 = 0.5 but organisation community 3 has loadings summing to 0.01 → part(1,3) is small; Z_23 = 0.05 with
loading sums 40 and 30 → part(2,3) = 60 is large. Z also depends on the scale freedom (a column of U multiplied by c,
the matching row of Z divided by c, nothing reconstructed changes; ticket 79), the parts do not.

**Limits.** The parts belong to the model's reconstruction, not to the observed ties; they are not "explained variance".
They depend on the numbering of communities across facets (E2), except the effective number of partners.

### E2. Why are communities renumbered after the fit, and how? (numbering freedom, alignment)

*Asked 2026-11-17.* **Cause:** (1) the model leaves each facet's numbering free (entry "Numbering freedom"); (2) our
readings and R-A treat the same number as one community in every facet (presence by domain, mono-domain rule,
detachment, Track A/B matching of stacked facets); (3) the fit's numbering comes only from the start, and on the social
side it is not kept: the social core matrices start random, so from the first step author community 4 is tied to a mix
of article communities, and an author column can end closest to another article column (KL author×article: 44% of the
tie weight off matching numbers as fitted, 8% after alignment). Result without alignment: "community 4" can combine one
group's articles with another group's authors, differently in each seed.

**Worked example** (made up, K = 3). Author communities 1 = physics, 2 = biology, 3 = economics. The fit puts physics
organisations in organisation column 2, biology in 3, economics in 1; the core ties author 1 → organisation 2, 2 → 3,
3 → 1, nothing on the diagonal. Reconstructed ties are right, but read by number "community 1" = physics authors +
economics organisations, the off-diagonal share is 1 and every community looks detached from its organisations.
Renaming organisation columns 2 → 1, 3 → 2, 1 → 3 (and the core's columns the same way) makes the core diagonal,
changes no reconstructed value, and community 1 = physics authors + physics organisations. Real case: KL approach 1,
author×affiliation, share on matching numbers 0.23 as fitted → 0.96 aligned, ≈ 1.04 partner communities.

**Is the aligned numbering "the correct" one?** Not in the model's own terms: every numbering gives the same loss, so the
fit has no correct or wrong numbering, and nothing in the fit is corrected. The error was in our *reading* of the fitted
numbering as if one number were one group across facets (R-A). Alignment chooses, for each facet, the numbering under
which community k is the community most strongly tied (by reconstructed tie weight) to community k of the facet it is
matched to, ending at the articles. Where the structure is one-to-one (KL) this yields a numbering under which one number
is one group in every facet; where it is spread (squared error) no numbering can do that, and the match strength shows
how far it succeeds.

**Main partner, shared main partner** (asked 2026-11-17). In the table of parts of author×affiliation (rows: author
communities, columns: organisation communities), an author community's **main partner** is the organisation community
with the largest part in its row. Two author communities have a **shared main partner** when both rows peak in the same
column: the authors of both communities are tied mainly to organisations that belong to the same organisation community
(e.g. two topic groups at the same set of universities). One-to-one alignment can give that column the number of only
one of them; the other keeps its own number on an organisation column that carries almost nothing and reads as
"detached". Measured (RL-262): KL approach 1, 68 of 80 detached author communities are of this kind.
*Why "main partner" at all under a soft partition* (asked 2026-11-17): the substantive, soft quantity is a community's
whole row of parts (how its tie weight divides over the other facet's communities; summarised by the number of partner
communities). The main partner matters only because a numbering is hard — each community gets one number — so
alignment must pick one partner per community. Under KL the main partner carries most of the row (≈ 0.96 of the weight
for author×affiliation) and little is lost; under squared error it carries about 0.4–0.6 and the hard choice is lossy
(low match strength). A shared main partner is a diagnostic of the alignment (and of a facet using fewer distinct
communities than K), not a property of entities.

**What alignment does to seed agreement** (measured 2026-11-17, RL-263). Track A/B match communities across two seeds
with **one** matching for all facets together, so they read the numbering across facets. Under the aligned numbering
this asks: do two seeds agree on which article community each organisation (author, journal) community is aligned to?
Under KL, yes. Under squared error, only at chance level for the social facets (chance-adjusted Track B ≈ 0), although
each social facet matched on its own agrees about as well as the semantic facets (≈ 0.45–0.6). Example: in seed 1,
organisation community A is tied 40 % to article community 4 and 35 % to article community 9, so it is aligned to 4; in
seed 2 the same organisations are tied 35 % / 40 %, so they are aligned to 9. The organisations' grouping is the same;
their number is not. The higher agreement under the fitted numbering came from the start (social memberships propagated
from articles keep the start's numbering), not from the data.

**How (decided):** entry "Alignment". **What it cannot do:** remove genuine spread (squared error: one author community
tied to several organisation communities; the best match then carries only part of the weight — hence match strength
is reported) or match two communities to one partner (one-to-one). **Relation to Hungarian matching elsewhere:** Track
A/B uses it to match communities across seeds; alignment uses it to match facets within one fit.

### E3. Why 2 SE and not 1 SE when comparing two arms?

*Asked 2026-11-17 (diagonal-start test).* Paired design: each seed starts both arms from the same memberships; the
difference per seed (arm − reference) isolates the arm's effect. SE = SD of the differences ÷ √n says how much their
mean would move with other seeds; with no real effect the mean is noise around 0, mostly within ±1 SE. **Worked example**
(made up): 10 differences in data loss, mean +0.0010, SD 0.0020 → SE 0.00063 → mean = 1.6 SE: "worse" under a 1-SE line,
"within noise" under 2 SE. **Why 2 SE:** with no real effect and 10 seeds, the chance of crossing the line on the "worse"
side by noise is ≈ 17% (1 SE) vs ≈ 4% (2 SE) per measure, ≈ 43% vs ≈ 11% for at least one of three measures
(independence assumed). A false "worse" would turn a decision (start artefact → O3 needed), so the line is strict. The
λ rule's one-SE line (Breiman) answers a different question — which settings are *not distinguishable* from the best —
where erring towards "equivalent" is intended. **Trade-off:** a real effect below ≈ 2 SE is read as noise; paired
differences are always reported in full.

### E4. What does presence mean at the level of an entity, a fit, a set of seeds, and the model?

*Asked 2026-11-17.* **Entity:** has no presence of its own; it contributes through its loadings (times how strongly its
community is tied, through the core, to the other facet). Its own reading is the weighted membership (shares sum to 1),
which can disagree with presence: membership gives every entity a full split however little weight it carries.
**Community in one fit** (one hold-out group, one seed): p_k(r) = how much of relation r the model builds through k
relative to an average community (1 = 1/K of T_r; 0.1 = a tenth). Example: K = 20, p = 0.056 → 0.28% of child×article's
reconstructed tie weight vs 5% average. It counts reconstructed weight, including predictions in empty cells.
**Across seeds / hold-out groups:** numbers are arbitrary across fits, so communities are matched; one fit's value is
noisy (a matched community's weak-domain presence went from 0.076 in one seed to a median 0.26 in the others), hence
"in the majority of seeds" in the mono-domain rule. **Model:** Σ_k p_k(r) = K for each relation — presence shows how the
model divides each relation among communities (one gains, others lose), not how well it fits. Depends on the numbering
across facets (E2).

### E5. When does an arm "stay near one partner"? (the midpoint line of the diagonal-start test)

*Asked 2026-11-17 (three times; restructured to start from the problem).*
**The problem.** The owner's reading R-A says one community is one group across facets: community 4's articles, authors,
organisations and journals belong together. That can hold only if, in each relation, a community's reconstructed tie
weight goes mostly to **one** community of the other facet (one partner). Pass 1 shows it does under KL (≈ 1–1.8 partner
communities) but not under squared error (3–5 at K*, rising with K).
**Two candidate mechanisms.** (a) The start: every social core matrix starts random, so from the first step each
community is tied to many communities of the other facet, and squared error may simply not move away from that.
(b) The loss and the data: squared error prefers to tie a community of one facet to several communities of the other,
wherever it starts. (a) can be fixed by changing the start (O1 suffices); (b) needs a term (O3) or a revised R-A.
**The test.** The diagonal-start arm changes only the start: each social core matrix starts diagonal, so every community
starts with exactly one partner. If the arm ends near one partner, the start was the cause (a); if it spreads again, the
loss and data are (b). The rule below is the line that classifies the arm's end state.
**Number of partner communities** (entry "Effective number of partner communities"): for one community, how many
communities of the other facet its reconstructed ties go to — 1 if all to one, 2 if 50/50, 4 if even over four;
averaged over communities weighted by tie weight; does not depend on numbering.
**Reference points:** 1 = the R-A pattern; the **reference value** = the median number of partners in the pass-1 fits
(random social start) for the same relation, loss and approach, hold-out group 1, seeds 1–10 (e.g. squared error
approach 1 author×affiliation 3.46).
**Rule (proposed):** the arm "stays near one partner" if its median is closer to 1 than to the reference value, i.e.
below the midpoint (1 + 3.46) ÷ 2 = 2.23; otherwise it "returns to spread".
**Worked example:** arm median 1.6 → stays near one partner → mechanism (a) (outcome 1 if fit is not worse, 3 if
worse); median 3.1 → returns to spread → mechanism (b) (outcome 2).
**Why a midpoint:** the question is whether the arm ends nearer the R-A pattern or nearer the random-start pattern; the
midpoint is the neutral line between the two. **Trade-off:** a median of 2.2 counts as "near one" although spread over
about two communities (the number is reported beside the verdict). **Alternatives:** an absolute line (e.g. 1.5, about
where KL sits) — stricter, but chosen by eye; a paired 2-SE test on the number of partners — says whether the arms
differ, not whether the arm is near one.
