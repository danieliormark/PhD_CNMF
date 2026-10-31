# Ticket 107 — the new matrix builder (M1, "chunk12v3"): representation decisions

> **STATUS (2026-10-13): DESIGN IN PROGRESS. Sections 1 and 2 are owner decisions; section 3
> records observations about the temporal extension (not decisions); section 5 lists what is
> still open. Nothing is built.** Decisions were reached in discussion with the owner on
> 2026-10-12/13 (the owner also consulted colleagues on the semantic side). Later decisions are
> added with dates; this text stays. *[2026-10-15: comparison criteria refined (§5 item 2); K-selection
> literature check, hold-out procedure and the decisions so far in §6.7.]*

## Context

The toy matrix builder is `chunk12v2.py` (tensor_data_staging/toy_large; ticket 84). Its
successor, chunk12v3, is the design for the full-scale M1 (fullscale_pipeline/PIPELINE.md, M1
row) and is to be tried together with the temporal extension (ticket 105, CLAUDE.md §4.25).
Owner's plan: (1) agree the representation, (2) implement for T1, (3) calibrate and implement
for T2.

The main idea (owner): represent everything as hyperedges, and stop modelling individual
words as entities. Motive: fewer relations, and single words were not consistently assigned to
communities (CLAUDE.md §4.24: in the C2/C6 K=4 knee models, articles and child/cousin
hyperedges kept their main community across seeds 72-84% of the time, words barely above
chance). Note recorded in the discussion: this instability shows that words are poor
*reported community members*; it does not by itself show that removing them leaves the
article and hyperedge communities unchanged, because a word's own label can be unstable while
it still carries "these articles share vocabulary" into the reconstruction. Section 1's word
regulariser keeps that information without making words entities.

## 1. Semantic representation — DECIDED (owner, 2026-10-13)

**One child-level hyperedge layer.** Child and cousin hyperedges are merged into a single
facet ("child" from here on means the merged layer). The owner and colleagues agreed that it is
not a crucial distinction whether an association sits inside the parent clause (child) or
elsewhere in the sentence (cousin).

- **Identity = the word set, without the role.** One ID per set of typed atoms (G3 v2's content
  key, `sha1(sorted atoms)`), so the same association has one entity wherever it occurs:
  `(dummy_sibling help/P/en)` and `(dummy_cousin help/P/en)` are one entity, as are
  `(focal_he X)` and `(sibling_he X)`. Atom types stay part of the identity (`help/P` ≠ `help/C`).
- **Why this was needed (checked 2026-10-12):** in `chunk12.py` and `chunk12v2.py` (identically) a
  child's ID is its full edge string including the role, and child and cousin have separate index
  maps, so the same word set used in both positions was two unrelated entities. The only
  child–cousin relation, `M_Cousin_Child`, records co-occurrence in the same sentence, not
  identity. On the toy, 75 word sets occur both as child and as cousin (51 verb groups, 24
  argument bags), and 6 word sets occur under two roles inside the child facet.
- Supersedes the options discussed on 2026-10-12: (A) same label, two independent rows; (B) two
  facets plus an identity link; (C) one facet. The merge is (C); B's identity link is no longer
  needed.

**Three inputs drive the child layer's communities:**

| Input | Form | Why this form |
|---|---|---|
| Used in the same sentence: child×parent | **Relation** | Observed association structure, the core of "idea = shared associations". A relation can express that one community's phrases are used together with another community's phrases (its community-to-community block); a regulariser could only push co-used hyperedges into the same community. |
| Used in the same article: child×article | **Relation** | The bridge to the social side (journals, affiliations and authors attach through article memberships); absence of a tie is informative about the article's content. |
| Similar content (shared words): child×child | **Regulariser** | Derived from words, which are deliberately not modelled; sharing no word does not mean dissimilar (synonyms); it should only pull similar hyperedges together, not push others apart, and should not consume reconstruction capacity. |

- **Parent stays a separate facet** (owner, 2026-10-13): a parent is a sentence's statement, the
  set of child hyperedges in it. Former cousins tie to the parents of their unit, so a sentence
  with two focal terms ties its cousins to both parents. Chosen over a child×child co-occurrence
  relation, which would lose which sentence and groups of three or more and would need damping of
  focal-only hyperedges such as `(focal_he llm/C/focal)`.
- **Rule used to choose relation versus regulariser** (agreed in discussion): a relation when the
  tie is observed data whose pattern the communities should explain, absence is informative, and
  the tie may link *different* communities; a regulariser when the information is a prior derived
  from features not modelled, absence is uninformative, and the only message is "similar →
  similar memberships".

**The word-similarity regulariser — form discussed, details open.** Graph-regularised NMF
(Cai et al. 2011): λ · Σ similarity(i,j) · distance(membership_i, membership_j). It is ridge
applied to differences between neighbours rather than to values. Points raised:
- Plain Jaccard is too coarse and dominated by frequent words. Toy check (840 child and cousin
  hyperedges): median 2 words per hyperedge; 8,641 pairs share at least one word (2.5% of pairs)
  against 81 pairs with identical word sets; 7,204 of the linked pairs have Jaccard 0.1-0.2; the
  most frequent words are `model`, `language_model`, `language`, `bert`, `llms`, `task`; 149
  hyperedges share no word with any other. At full scale every `focal_he` contains a canonical
  focal atom. Proposed: idf-weighted similarity over words (here idf on *words* is appropriate),
  focal atoms left out, each hyperedge's top-k neighbours above a threshold, atom types kept.
- A second-order ("similar if their neighbours are similar") measure was assessed and **not
  recommended**: it is SimRank / co-similarity; it duplicates the second-order smoothing the
  factorisation already performs; iterating it converges towards a popularity ranking (the
  dominant eigenvector), so frequent hyperedges become similar to everything; and it is
  infeasible at full scale. If first-order overlap proves too sparse, word embeddings from a
  pretrained model are the principled alternative (a theory decision).
- Model-specific cautions: giving every hyperedge the same membership satisfies the penalty
  perfectly, so λ must be calibrated (as ticket 102's weight), not tuned by Optuna; the penalty
  must compare membership shares, not raw factor rows (ticket 79 gauge); KL is a candidate
  distance, as in the temporal prior.

## 2. Social representation — DECIDED: two approaches, compared (owner, 2026-10-13)

| | Relations | Status |
|---|---|---|
| **Approach 1: author facet** | author×article, author×affiliation, article×journal | Main comparison |
| **Approach 2: no author facet** | article×article (shared authors), article×affiliation, article×journal | Main comparison |

**Approach 2.** Articles are author sets, affiliations are author sets, journals are sets of
articles. Authors are not entities; which authors belong to which article and affiliation is
recovered after the fit (an author's community profile from their articles). The owner's
choice to drop authors as entities is deliberate; communities are attributed primarily to the
collectives.
- The factorisation sees only relation matrices, so without article×article it would not see that
  two articles share authors at all. Article×article makes that overlap visible without naming the
  authors.
- article×article needs: each shared author contributing 1/(that author's paper count − 1), so
  prolific authors do not create dense blocks of pairs; the diagonal excluded from the loss; new
  chunk13 code for a relation with the same facet on both sides. Shared authors require knowing
  that two bylines are the same person (OpenAlex author ID or ORCID); bylines with neither
  contribute nothing to it.
- article×affiliation is built per byline (the affiliation stated on that article), proposed
  weight log2(1+n)/n × 1/m for an author listing m affiliations on an article with n authors,
  which inherits chunk12v2's article-author damping (ticket 63) and per-author affiliation split
  without mixing an author's institutions across papers. It needs no author identity.

**Approach 1.** chunk12v2's social relations with authors kept as a facet (ticket 63 damping on
author×article; author×affiliation normalised per author over the slice). Known limitation:
author×affiliation is aggregated over the slice, so an author who changes institution within a
slice ties all their papers to both (toy: 2 of 16 multi-paper authors in T1, 1 of 13 in T2; rare
with one-year slices).

**Comparison table drawn up in the discussion** (author facet vs article×article): the author
facet keeps who wrote what but adds one membership row per author (toy ~460, full scale ~142k),
86% of them single-paper authors with one tie each (many single-tie entities went with poor
community recovery on the toy, FINDINGS §25); article×article adds only a K×K block, drops
single-paper authors automatically, couples articles directly, but loses who and the authors'
paper counts, and gives no continuity across slices (see section 3).

**Options considered and not adopted (2026-10-13):**
- **article×article as a regulariser instead of a relation — deferred**, to be tried only if the
  relation dominates or fragments the article communities. Against it as the main form: a
  regulariser can only say "shared authors → same community", hiding co-authorship across
  communities (the bridging that heterogeneous communities are about), and it would leave the
  social side of the loss with two relations.
- **Regulariser pulling together affiliations that share authors, within a slice — rejected**:
  mostly redundant with article×affiliation (co-authoring institutions already share articles; a
  dual affiliation already ties both to the same article); what it adds (someone listed A on one
  paper and B on another) says people moved, not that A and B work on similar ideas; it needs author
  identity; large institutions would be pulled towards one averaged profile.
- **The same across slices — rejected in this form** (continuity belongs to the temporal prior;
  see section 3).
- Superseded on 2026-10-12, recorded for completeness: an article×author relation restricted to
  authors on 2+ papers (keeps who without single-tie columns, but makes authors entities again).

**Comparability note.** Approach 1 uses author×affiliation, approach 2 article×affiliation. The
relations both share — on which held-out ties can be compared fairly — are article×journal and
the semantic relations (child×article, child×parent). Using article×affiliation in both would add
affiliation fit to the comparison (the trade-off: approach 1 would then have no author×affiliation).

## 3. Temporal extension — observations (2026-10-12/13; not decisions)

- **Persisting entities change with the facets.** Articles never persist (slice-specific). On the
  toy, under chunk12v2's facets: authors 4, affiliations 5, journals 3, parents 0, child 24, cousin
  34 (atoms, now dropped: 123 and 118). With role-free IDs the merged child layer's persistence is
  not yet counted (merging can only add matches). The earlier T2 test with atoms excluded (70
  persisting entities, CLAUDE.md §4.25) accepted 11.9% of community pairs; C2 (28.5%) and C5
  (20.0%) improved, C3 and C6 fell to chance. The toy can test the mechanics; calibration stays on
  planted data, as decided for ticket 105.
- **Parents carry no continuity:** 99% of toy parents occur in a single article and none occurs in
  both slices. Exact hyperedge identity makes persistence sensitive to G3's rules (e.g. the
  deliberately specific verb groups, G3_POSTPROCESSING.md §6 item 8).
- **Approach 2 loses continuity through people.** article×article ties exist only within a slice,
  so the persisting social entities are journals and affiliations only. Approach 1 keeps
  persisting authors as prior entities. At full scale 14.0% of authors whose OpenAlex author ID or
  ORCID we know have more than one corpus paper (CORPUS_STATISTICS.md §9.9).
- **Idea for the T2 stage, not adopted:** an author-based prior in approach 2 — a T2 article whose
  authors wrote T1 articles weakly pulled towards those T1 articles' memberships. It is a
  regulariser by nature (shrinkage towards a fixed target), unlike a cross-slice affiliation
  regulariser, which would mix "the institution persists" with "people moved". Caution: the
  current design deliberately gives new entities no prior so that emergence stays free; this would
  bias returning authors' new articles towards old communities (hold back topic switching). Only
  the planted split and emergence tests (ticket 105 T3) could show whether a weak version is safe.
- **Repeat authorship falls by year because of right-censoring:** among authors whose OpenAlex ID
  or ORCID we know, the share of a year's authors with more than one corpus paper falls from 38-44%
  (2019-2022) to 17.3% (2026), because later cohorts have had less time to publish again
  (CORPUS_STATISTICS.md §9.9). Later slices will show fewer returning authors for that reason
  alone.
- **idf must be computed per slice or cumulatively up to the slice** (ticket 106): global df lets T2
  articles shape T1's weights. Proposed as a change that is better in every respect for the
  temporal setting.
- **Synthetic venues would create false continuity.** Discussed for patching the toy (2026-10-12):
  a pseudo-journal holding articles from both slices would give the prior artificial continuity,
  so pseudo-journals would have to stay within one slice. Moot if the test uses the full-scale
  sample (section 4).
- **Affiliation at publication.** The owner's rule (affiliation as stated on each paper, not a
  profile) is what article×affiliation per byline implements; it also removes within-slice mixing
  for authors who change institution.

## 4. Where the test runs (owner, 2026-10-12)

Test the new M1 on a sample of the full-scale corpus rather than on the toy, which has more
missing data (49% of toy authorships have no institution; 59 of 121 toy articles have only a
repository or no venue) and was curated by the old G3 rules from abstracts. The full-scale sample
has G3 v2 hyperedges, journals from PubMed for every article, affiliation at publication, and
full-text focal sentences. **Slices: open.** *[Decided 2026-10-13 (owner): T1 = 2020, T2 = 2021 — small for fast building, tests and calibration (2020: 91 articles, 1,654 G3 parents, 460 bylines; 2021: 385 articles, 6,725 parents, 2,064 bylines; toy T1 for comparison: 25 articles, 63 parents). Both years are in the BERT era, so little emergence is expected; emergence behaviour is calibrated on planted data. If the comparison criteria do not separate the approaches clearly at this size, repeat the comparison on 2021/2022 (385/744).]* Proposed 2019 (T1) / 2020 (T2); the working corpus
has only 17 / 91 articles in those years (2021: 385, 2022: 744, 2023: 1,914, 2024: 4,956), so
alternatives were put to the owner: (i) T1 2019-2022 (1,237) vs T2 2023 (1,914), (ii) T1 2022 (744)
vs T2 2023 (1,914) — both a pre/post-ChatGPT break, a genuine emergence case.

## 5. Open — to be decided

1. ~~Slices~~ *[decided 2026-10-13: 2020/2021, section 4]*; still open: which date defines a paper's year (PubMed print or electronic date, or
   OpenAlex year).
2. How the configurations are compared, fixed before results (ticket 91 precedent). They
   reconstruct different matrices, so reconstruction loss and hypervolume cannot rank them (ticket
   93). Candidates: chance-adjusted seed stability per shared facet; held-out ties on shared
   relations; a text check; temporal matching quality.
   *[2026-10-13: the owner likes the four candidates. From Tang, Wang, Zhao & Wang 2025 (CLAUDE.md §10):
   (A) the ℓ2,∞ membership error is dropped (owner) — it needs the true memberships, so planted tests only;
   its maximum is driven by the worst, usually single-tie, entity; and its informative part, the mean distance
   of fitted to true profiles, is already in §4.25's revised calibration measures. (B) Matched cosine between
   membership matrices — to consider as a fifth criterion: agreement between the two approaches on the facets
   both have (articles, child hyperedges, parents, journals, affiliations), with a shuffle-based chance level as
   for Track A/B (ticket 95); costs no new fits (uses the stability runs' memberships, a K×K matching, a shuffle
   null — seconds); if the approaches agree, the simpler one wins. (C) Dirichlet(α) membership draws — to
   consider for the planted calibration only (temporal prior, regulariser λ), combined with the degree- and
   weight-preserving generator of FINDINGS §25; the cost is planted fits, which multiply with each α level (two
   levels proposed: clear-cut and mixed); not for comparing the approaches unless the real-data criteria fail,
   since that would need authors, affiliations, sentences and word sets generated jointly.]*
   *[2026-10-14/15, owner agreed:]*
   - **Matched cosine** is used for the comparison *between* the approaches, read between two reference points:
     the shuffle null (floor) and the agreement between two seeds of the *same* approach, measured the same way
     (ceiling). Agreement near the ceiling means the approaches find the same communities, and the simpler one
     wins. Within one approach it is a benchmark and a per-community diagnostic (which communities reproduce),
     not a second stability gate; Track A/B remain the gate.
   - **Dirichlet(α) draws are not a comparison criterion.** They are only the way the soft true profiles of the
     planted calibration worlds are drawn (CLAUDE.md §4.25 update: planted truth must be soft). Taken off the
     comparison list. Reason against using planted worlds to compare the approaches: the generator would have to
     create bylines, and any rule for how co-authorship arises is the model behind one of the two approaches, which
     would then win by construction.
   - **Held-out fit is used at least in the trial** (owner, 2026-10-15). Comparison list as proposed 2026-10-14
     (not separately re-confirmed as a table): (1) held-out fit on the shared relations — main ranking; (2) Track
     A/B stability against chance — gate; (3) matched cosine between approaches; (4) text check; (5) T1↔T2 matching
     quality, T2 stage only. The same hidden cells of the shared relations are used in both approaches, so the
     K-selection fits also score the comparison.
3. Weighting inside each matrix. Status quo (chunk12v2): anchors `log1p(tf)·idf` with global idf;
   clause relations binary; ticket 63 damping on article×author; per-author normalisation on
   author×affiliation; then every relation scaled to unit norm and the loss split 0.5 social /
   0.5 semantic equally among relations (chunk13). Issues raised for chunk12v3: idf now down-weights
   the hyperedges shared across articles, which are the semantic glue; article length (full scale:
   G3 parents per article p10 1, median 11, p90 83, max 669); generic hyperedges in child×parent;
   weights of the new relations; the regulariser's λ. To be discussed after facets and relations are
   settled (owner).
4. Code scope: chunk12v3 reading G3 v2's JSONL output; chunk13 changes (facet lists, a same-facet
   relation, the regulariser, penalties and stability for the new facets, persisting entities for
   the temporal prior).
5. **Data preparation (owner, 2026-10-13):** authors can list more than one affiliation within a slice;
   institutions must be deduplicated first (OpenAlex institution IDs, then text matching for the rest), at the
   data-preparation step, not yet.
6. **K selection — raised 2026-10-13, under discussion** *[expanded 2026-10-14 into §6 below, with evidence]*. Status quo (`select_models.py`, ticket 91): K fixed per
   study over 2-5; a model qualifies on stability (all seeds converge, both tracks above their own shuffled chance
   maximum); among qualifying K the highest scout hypervolume wins, then the knee. Hypervolume rewards lower
   reconstruction loss, which falls with K, so in practice the rule picks the largest K that still passes
   stability (v9.4: K = 4 in five configs, 5 in C4). Discussion points (assistant's proposal, not decided):
   in-sample loss is not an Occam signal; established routes are complexity penalties (AIC/BIC/MDL; poor fit for
   this loss), out-of-sample fit (bi-cross-validation, Owen & Perry 2009; network cross-validation, Chen & Lei
   2018), stability against a random baseline (Brunet et al. 2004; Lange et al. 2004; Wu et al. 2016), Bayesian
   relevance determination (Tan & Févotte 2013; Peixoto 2014), and a null-calibrated elbow (Frigyesi & Höglund
   2008). Proposal: neither depenalise high K nor add a complexity penalty; check each criterion for K-dependence
   that is mechanical (thresholds or baselines not scaled with K) and make it K-aware; choose K by held-out fit
   with chance-adjusted stability as the gate, not by hypervolume; search by a coarse-to-fine K grid at full scale,
   optionally narrowed by a spectral pre-estimate (Le & Levina 2015); warm starts and relevance pruning as
   experiments only.

## 6. Choosing K — discussion record (2026-10-13/14); not decided

Recorded at the owner's request so the reasoning survives a compacted conversation. Nothing here is an owner
decision yet; the proposal (6.5) is the assistant's.

### 6.1 Status quo

`select_models.py` (ticket 91), per config: K is fixed per Optuna study over `K_LIST` (2-5 since v9.3); a model
qualifies if all 10 stability seeds converge and both stability tracks beat the maximum of their own shuffled chance
level (ticket 95); among K values with a qualifying model the highest scout hypervolume wins; then the knee of that
K's front. Hypervolume is built on reconstruction loss, which falls with K (6.2), so it favours the largest K; in
practice the rule picks **the largest K that still passes stability** (v9.4: K = 4 in five configs, K = 5 in C4).
Stability is doing the braking. Ticket 91 left open whether hypervolume should choose K at all.

### 6.2 Evidence on the toy corpus

**In-sample fit improves with every K** (checked 2026-10-14, `chunk13_execution/k_loss_by_K.py`, all COMPLETE,
converged, non-duplicate trials in each config/K Optuna database). Best reconstruction loss (`math_loss`; 1.0 = the
model that reconstructs nothing):

| Run | Config | K=2 | K=3 | K=4 | K=5 | K=6 |
|---|---|---|---|---|---|---|
| v9.2 | C1 | 0.889 | 0.844 | 0.800 | 0.758 | 0.738 |
| | C2 | 0.875 | 0.823 | 0.774 | 0.740 | 0.733 (16 trials) |
| | C3 | 0.879 | 0.832 | 0.789 | 0.774 | 0.738 (74) |
| | C4 | 0.880 | 0.837 | 0.786 | 0.753 | 0.724 (78) |
| | C5 | 0.871 | 0.820 | 0.768 | 0.726 | 0.705 (53) |
| | C6 | 0.877 | 0.827 | 0.778 | 0.746 | none converged |
| v9.4 | C1 | 0.891 | 0.843 | 0.800 | 0.760 | — |
| | C2 | 0.875 | 0.824 | 0.775 | 0.740 | — |
| | C3 | 0.881 | 0.833 | 0.789 | 0.774 | — |
| | C4 | 0.881 | 0.840 | 0.788 | 0.754 | — |
| | C5 | 0.871 | 0.819 | 0.768 | 0.726 | — |
| | C6 | 0.878 | 0.825 | 0.779 | 0.754 | — |

Other cells have 100-200 trials. Best and median both fall with K in all 12 config/run series, with diminishing gains
(v9.2: about 0.05 from K=2 to 3, 0.007-0.036 from K=5 to 6). The gap between median and best trial grows with K: at
K=2 the median is within 0.001-0.003 of the best; in v9.4 at K=4-5, C3 and C4's medians sit 0.029-0.049 above it —
the optimiser reaches good solutions less reliably as K grows.

**Almost every quality measure worsens with K** (FINDINGS §27-§29): convergence (at K=6 only C1 reached its 100
converged trials; C6 none); stability failures (every archived C3 and C4 model at K=5/6 failed); low-mass ("ghost")
communities (0% of models at K=2/3, 11% at K=4, 20% at K=5, 56% at K=6); item-to-community attribution (Penalty_A,
0.023 at K=2 to 0.142 at K=6 among stability-verified models). FINDINGS: "K=5/6 is where this corpus runs out of real
structure to support that many communities."

**Some trends with K are mechanical, not evidence:** Track A's chance level itself depends on K (0.47-0.67, FINDINGS
§27 correction), which is why stability is judged against each model's own shuffled baseline; the largest community's
mass share (`max_share`) and the concentration of shared terms (Penalty_B) improve automatically with K, because
mass spreads over more communities.

### 6.3 Why in-sample loss falls with K, and where "more parameters are harder" shows

The loss measures how well the model reproduces the data, not how well the parameters are pinned down. A K+1 model
contains the K model as a special case (an extra community with zero weight), so its best reachable fit cannot be
worse; in the extreme it would memorise the matrices. The intuition that more parameters are harder to get right is
also true, but it concerns their reliability: each parameter rests on less evidence, so solutions fit noise and differ
between runs. That shows up as seed instability, low-mass communities, poor attribution, and (if measured) worse fit
on held-out ties — not in in-sample loss. The one place it reaches the observed loss is the search: more local
minima and slower convergence (6.2's widening median-best gap). Polynomial analogy: a higher degree passes closer to
every point, but its coefficients swing between samples and it predicts new points worse.

### 6.4 Established routes to choosing K

| Route | How complexity is penalised | References | Fit for this model |
|---|---|---|---|
| Complexity penalty | AIC/BIC/MDL: fit minus a cost per parameter | Squires et al. 2017 (MDL for NMF) | Poor: squared-error loss plus penalties; the parameter count is ambiguous (scale freedom, clamping) |
| Out-of-sample fit | Hold out entries; overfitting shows as worse held-out fit, so nothing needs penalising | Owen & Perry 2009 (bi-cross-validation for NMF); Chen & Lei 2018 (network cross-validation) | Good; already planned (CLAUDE.md §10 item 1: held-out ties, masking adjusted per relation) |
| Stability | Largest K that stays reproducible, compared with a random baseline | Brunet et al. 2004 (cophenetic correlation); Lange et al. 2004; Wu et al. 2016 (staNMF) | Good; what the current rule effectively does |
| Bayesian | Marginal likelihood penalises complexity automatically; relevance priors switch off unused components | Tan & Févotte 2013 (ARD-NMF); Peixoto 2014 (MDL for block models) | Possible, but a different modelling framework |
| Null-calibrated elbow | Add communities only while the real data's loss falls faster than on randomised data with the same degrees | Frigyesi & Höglund 2008 | Feasible: the degree- and weight-preserving generator exists (FINDINGS §25) |

### 6.5 Proposal (assistant, 2026-10-13/14)

1. **Neither depenalise high K nor add a complexity penalty.** The deterioration in 6.2 is the Occam signal
   (communities without support), and should act as such.
2. **Audit each criterion for mechanical K-dependence** (a threshold or baseline that does not scale with K, a chance
   level that changes with K) and make it K-aware, as the 0.5/K low-mass threshold already is (ticket 86) — do not
   boost or penalise K wholesale.
3. **Choose K by held-out fit, with chance-adjusted stability as the gate; keep the knee within K.** Do not choose K by
   hypervolume (it inherits the capacity effect). The null-calibrated elbow is a cheap second opinion.
4. **Search.** At trial scale (K 2-6, fits in seconds) trying every K is not the bottleneck; the cost is the many
   Optuna trials per K, and with one tuned parameter Optuna is effectively random search (CLAUDE.md §4.12), so a small
   fixed grid of that parameter per K would save more. At full scale, where K may be 10-50: a coarse-to-fine K grid
   (2, 4, 8, 16, 32, then refine), optionally narrowed first by a spectral pre-estimate on article-level matrices
   (eigengap or Bethe Hessian; Le & Levina 2015). Warm starts (initialise K+1 from K by splitting the community with
   most residual) and relevance pruning (one fit at a large K with a group penalty on community mass) only as
   experiments: the first adds path dependence, to which this pipeline is sensitive (CLAUDE.md §4.25 update); the
   second moves the choice into a hyperparameter and overlaps with low-mass communities (ticket 86).
5. **Temporal.** T2's free K uses the same criteria; the planned with/without-prior comparison of the chosen K (ticket
   105, test T6) checks whether the prior biases K.

### 6.6 Open for the owner

Whether to replace hypervolume as the K selector; which held-out design (fraction and masking per relation, CLAUDE.md
§10 item 1); whether to add the null-calibrated elbow; the K range and search scheme for the 2020/2021 trial and for
full scale.

### 6.7 Update 2026-10-14/15: literature check, the hold-out procedure, what is agreed

**Corrections to 6.4/6.5 and to the discussion, found by the literature check:**
- Edge cross-validation (Li, Levina & Zhu 2020, *Biometrika* 107(2)) guarantees, asymptotically, that it does **not
  choose too few** communities; the authors state cross-validation gives no protection against choosing **too many**.
  The assistant had said it tends to underestimate K; that was wrong. The parsimony comes from the one-standard-error
  rule, not from the method.
- DFMF (Žitnik & Zupan 2015), the closest published relative of our model, chooses each rank in a preset range by
  repeated halving (test midpoint and ends, keep the better half), judging explained variance, residual sum of squares
  and the cophenetic coefficient, taking the rank where the cophenetic coefficient starts to fall (as summarised by the
  search; full text not read).
- Squires, Prügel-Bennett & Niranjan 2017 (MDL rank selection for NMF) is in *Neural Computation* 29(8); per its
  institution's summary it argues cross-validation is often impractical and "rarely works" for NMF.
- Klami, Bouchard & Tripathi 2014: group sparsity infers whether each factor is shared or private to one matrix; not
  confirmed that it also chooses the total number of factors.

**Methods found (sources verified 2026-10-15; details from abstracts or summaries unless stated):**

| Family | Method | Fit for this model |
|---|---|---|
| Hold-out | Edge cross-validation (Li et al. 2020): ~10% of all node pairs held out, empty ones included; 3 random splits, or 20 repeats keeping the most frequent choice; squared error, deviance or AUC; weighted networks covered | General enough; full text read for these details |
| | Node cross-validation (Chen & Lei 2018) | Much worse than edge CV in harder settings (one simulation: 49% vs 0% correct) |
| | Bi-cross-validation (Owen & Perry 2009) | Reported unstable in practice (Kanagal & Sindhwani 2010) |
| | Cell-wise ("imputation") CV for NMF | Most accurate metric in a comparative study, "no solution worked perfectly" |
| | **solrCMF** (Held, Lindbäck & Jörnsten 2024, arXiv): collective factorisation, cell-wise 10-fold CV per data matrix, MSE, one-standard-error rule | **Closest precedent for 6.5's proposal**; full text read for these details |
| | Linked Matrix Factorization (2017): cell-wise CV, ranks raised one at a time until the error stops falling | Forward-search variant |
| Stability | Consensus/cophenetic (Brunet et al. 2004); proportion of ambiguous clustering (Șenbabaoğlu et al. 2014) | Șenbabaoğlu et al. show consensus clustering finds "stable" clusters in data without structure — supports judging stability against chance; stability tends to favour coarse K (nmfkc documentation) |
| Model-based | ARD-NMF (Tan & Févotte 2013, *TPAMI*); MDL for NMF (Squires et al. 2017); layered and weighted SBM with MDL (Peixoto 2015, *PRE* 92; 2018, *PRE* 97); group-sparse CMF (Klami et al. 2014) | Different frameworks; usable only as independent checks |
| Tests | StGoF (Jin, Ke, Luo & Wang 2023, *JASA*); multilayer tests and modularity (2024-26 preprints) | Built for hard memberships (StGoF overestimates K with many mixed nodes) or for several layers over one node set; single-layer methods "can not be immediately applied" to multilayer data |

No method was found for exactly this setting (soft memberships, many entity types, many relations). Real-data results
in the literature are imperfect and criteria often disagree.

**The hold-out procedure (as explained to the owner 2026-10-15):**
- What is hidden: a random share of the *cells* of each relation matrix, every cell (empty or not) with the same
  chance; never entities. Empty cells must be included, or a model predicting ties everywhere would pass.
- How: hidden cells are left out of the loss through a mask (no value, not 0). Only for the NNDSVD start, which needs a
  complete matrix, are hidden cells set to 0 and visible values scaled by 1/(share visible), as in edge CV. In code: the
  trace-trick loss minus the hidden cells' terms.
- Scoring: mean squared error between the model's value and the true value on the hidden cells; weighted values are
  not a problem. Reported per relation, combined with the training weights, and separately for hidden ties and hidden
  empty cells.
- An entity's last tie is never hidden (single-tie entities then contribute no hidden cells; stated as a limit).
- Weights (idf, damping, normalisation) are computed from the full data and treated as fixed (a small leak, stated).
- Repetition: hidden share and number of folds are linked (k folds hide 1/k each); alternative: repeated random 10%
  splits. **Deferred (owner).**
- Choice of K: smallest K whose mean held-out loss is within one standard error of the best (as in solrCMF).
- During K selection a small λ_z_offdiag grid replaces the full Optuna search; Optuna and the knee run at the chosen K.
  Trial estimate: 9 K × 5 splits × 3 λ × 2 seeds ≈ 270 fits per approach and slice.

**Agreed (owner, 2026-10-15):** the hold-out is used at least in the trial; the trial also tests a coarse grid (owner's
proposal: K = 3, 5, 8). Proposed (assistant): run the full grid 2-10 anyway and replay the coarse search on its results
(look only at 3, 5, 8, refine around the best), so the coarse search is checked against the full grid at no extra cost.

**Still open:** folds; the T1→T2 order (T2's K chosen from fits without the prior, then the prior run at that K) and
whether K should be chosen automatically at all — the owner's concerns (2026-10-15): later slices may grow by thousands
of articles and need several new communities, so a ±1 check is too narrow; and the criteria may disagree about K.
Planted worlds (validating the K procedure; calibrating the regulariser λ and the temporal prior's weight) are proposed
and under discussion; owner's concern recorded: if they were built from fitted T1 profiles they would inherit the
model's error — the proposal draws the truth synthetically and borrows only sizes, degrees and weights from the data.

### 6.8 K selection — decisions (owner, 2026-10-17)

- **Decision protocol (decided).** Each measure has one role, fixed in advance: (1) stability against chance (Track A/B,
  ticket 95) is the **gate** — a K whose solutions do not reproduce is excluded; (2) held-out fit with the
  one-standard-error rule defines the **band** of K values the data cannot tell apart; (3) inspection and the text
  check **choose within the band**, with reasons recorded; (4) hypervolume plays **no role in choosing K**, only in
  choosing the knee within a K. Disagreement rules: if no K in the band passes the gate, take the largest K below the
  band that passes, and flag it; if the band is wide, report it as a range and give the main results at two
  resolutions; always check the substantive conclusions at neighbouring K.
- **Trial K range (decided):** full grid K = 2-10 per slice; the coarse search (3, 5, 8, then refinement around the best)
  is replayed on those results to check it against the full grid at no extra cost.
- **Temporal variants (decided):** option 2 and option 2 with smoothing (§7.4) are fitted separately, so each chooses
  its own K per slice.
- **Hold-out in the trial (owner, 2026-10-17):** compare three schemes — 5-fold (20% hidden per fold, every cell hidden
  once), 10-fold (10%), and relation-by-relation (hide 20% of one relation's cells while the other relations stay complete,
  as in De Bacco, Power, Larremore & Moore 2017, *PRE* 95, 042317) — and see how each behaves. Empty cells are hidden at
  the same share as ties. **Per-entity cap (approved where pertinent):** no entity loses more than one-third of its ties in
  a fold; an entity's last tie is never hidden. Literature range of held-out shares in networks: 5-20% (Vallès-Català et
  al. 2018: 5%; Li, Levina & Zhu 2020: 10%; De Bacco et al. 2017: 20%); a larger share favours selecting the right model
  (Shao 1993), a smaller one keeps sparse data fittable (Vallès-Català et al.; Ghasemian, Hosseinmardi & Clauset 2020).
- **Hold-out scheme comparison (owner, 2026-10-17):** run on one combination only, social Approach 1 (author facet: the most relations); the chosen scheme is then used everywhere.
- **Grid and coarse search both run in the trial (owner, 2026-10-17, confirming the trial range above).**
- **Still open:** whether a small λ_z_offdiag grid replaces the full Optuna search while K is chosen (the "K-selection
  stage" / "model-selection stage" proposal, to be re-explained); how the smoothing weight is set.

## 7. Temporal prior under ticket 107 — owner-requested checks (2026-10-15)

### 7.1 The distance in the prior: compare KL with bounded alternatives (owner, 2026-10-15)

The temporal prior (CLAUDE.md §4.25) penalises KL(T1 membership ‖ T2 membership) for persisting entities. KL is
defined for soft profiles (any probability vectors; non-integer values are no issue), and MetaFac and FacetNet use it,
but it weighs kinds of change unevenly. Example, T1 profile (0.7, 0.3):

| T2 profile | change | KL(T1 ‖ T2) | Jensen-Shannon (max ln 2) |
|---|---|---|---|
| (0.5, 0.5) | more mixed | 0.08 | 0.02 |
| (0.3, 0.7) | main community swapped | 0.34 | 0.08 |
| (0.99, 0.01) | specialised in the old main community | 0.78 | 0.10 |
| (0.01, 0.99) | old main community abandoned | 2.62 | 0.32 |

- KL judges ratios, so losing a minority share costs more than a swap: it would resist **specialisation** more than
  **migration**. It is unbounded: leaving a community completely costs without limit (floored at 1e-12).
- Its asymmetry is deliberate: T2 mass on a community where T1 had none costs nothing directly, which keeps emergence
  free. A symmetric measure (JS) charges for that too, but a bounded amount.
- Under the squared-error data term KL has no Dirichlet pseudo-count meaning (§4.25 point 2), so the choice is not
  forced by the model.

**Decided:** in the planted two-slice calibration worlds, compare KL with at least one bounded alternative —
Jensen-Shannon divergence and/or Hellinger distance — at matched strength (each at its own calibrated weight). Add a
planted **specialisation** scenario (persisting entities concentrating on their old main community) to the existing
growth / migration / split / merge / emergence scenarios. Compare: recovery of the planted changes (migration,
specialisation, split, emergence) and the stability gain. Because JS charges for mass on new communities, the emergence
scenario must be checked under the JS arm in particular. Cost: one or two extra arms on worlds built anyway.

### 7.2 T1↔T2 matching: one-to-many, graded (a transition matrix), plus descriptive labels — DECIDED (owner, 2026-10-16)

**Replaces** the one-to-one Hungarian matching of CLAUDE.md §4.25 (each T1 community paired with at most one T2
community, a pair kept or dropped).

**Why.** T2's K is searched over its full range, not around K₁ (2026-10-15), and the corpus grows 2-4× per year, so
splits are likely. One-to-one matching can pair a T1 community A with only one of its children: A's persisting members
are pulled towards that one child, the other children look new, and the split is hidden — the "artificial emergence"
risk already recorded in §4.25. Merges have the mirror problem.

**The rule.** Every T1 community A and T2 community B get a link weight w(A→B) in [0, 1]: the share of A's persisting
members' membership mass that lands in B, counted only above its own chance level (row-shuffle null, as in ticket 95),
so a chance-level overlap gives exactly 0; then normalised over B for each A. A persisting entity's prior in T2 terms
is its T1 profile multiplied by this table: a transition matrix, the form probabilistic dynamic community models use
for community change (e.g. Yang et al. 2009, cited by MetaFac). Graded rather than binary: no hard threshold where a
link jumps from 0 to full strength; splits and merges come out naturally; weak evidence gives a weak pull.

**Known costs (accepted as the least wrong option for a growing corpus):**
- K₁ × K₂ candidate links instead of at most K₁ pairs: more chance links, so the chance correction must account for
  the number of links tested (the selection-aware null of the 2026-10-07 update, extended to all links).
- The pull is divided among an old community's descendants ("stay somewhere among A's descendants"): weaker, so less
  continuity is bought at a given weight.
- Splits and merges together form a web, not pairs; reporting is harder (handled by the labels below).
- Circularity, shared with the old matching: the weights come from the T2 fit itself at the freeze epoch, so the prior
  partly reinforces what the fit was already doing.

**Descriptive labels (approved, owner 2026-10-16).** After each T2 fit, each T2 community is labelled from where its
persisting members' mass came from: *continues* A, *descends from* A (one of several children of a split), *merges* A
and B, or *new*; each T1 community without a descendant is *dissolved*. Reported by facet type (semantic, people,
venues/institutions), because each kind of continuity means something different.

**Order of work:** the matching rule is fixed before the prior's weight is calibrated (a different matching changes what
the prior does). The temporal architecture itself is reopened (owner, 2026-10-16: expansive growth); see the persistence
measurement, to follow.

### 7.3 Temporal design: options compared (discussion 2026-10-15/16); owner prefers option 2

**Why reopened.** The corpus grows 2-4× a year (2020: 91 articles … 2025: 10,058), and entity persistence is low
(CORPUS_STATISTICS.md §9.11): about nine in ten entities are new in every slice (child hyperedges 2-11% persisting,
parents ~1%, authors 3-10%, journals 16-58%, affiliations 13-59%). About half of the ties touch a recurring entity, but
mostly through generic phrasing; distinctive continuity (multi-word, idf-weighted) is 12-28%. An entity-wise prior
(option 1) would act on ~5% of entities, dominated by the least distinctive ones.

**Options and the literature's taxonomy** (Rossetti & Cazabet 2018, *ACM Computing Surveys* 51(2), full text read):

| Survey class | Definition (survey) | Here |
|---|---|---|
| Instant-optimal | communities at t depend only on the data at t; then matched across steps | option 2; option 3b |
| Temporal trade-off | communities at t balance the data at t against the past | option 1: the prior (MetaFac, FacetNet, CLAUDE.md §4.25) |
| Cross-time | one partition searched for all steps at once | option 3a: one joint factorisation |

The survey: instant-optimal methods' main weakness is that an algorithm's own instability cannot be told from real
change (remedies: track stable community cores, or match all steps at once); most cross-time methods have a fixed number
of communities and no merge or split. 3a is therefore set aside (one K for the whole period; a persisting entity has one
membership for the whole period); it remains the natural home for cross-slice citation ties if those are ever added.

Terms: *step community* = a community found by fitting one slice alone; its *profile* = each active entity's loading on
it (weighted membership); *dynamic community* = a community followed over slices, made of step communities;
*recurring entity* = one that can appear in more than one slice (child hyperedges, parents, journals, affiliations,
authors; never articles). Social **Approach 1/2** (with/without author facet) is a separate axis from these temporal
**options**. Community change types are called **scenarios** (owner; Palla et al. and Greene et al. call them events).

**Shared layer 1 (options 2 and 3b).** Each slice is fitted alone with its own K_t, chosen by the agreed procedure
(hold-out, stability gate, one-standard-error rule). Both options link step communities across slices only through
recurring entities; the article part of a community plays no role in the linking.

**Option 2 — step-by-step tracking** (Greene, Doyle & Cunningham 2010, full text read). Each first-slice step community
starts a dynamic community; its *front* is its latest step community (not a Pareto front). Each new step community is
compared with every front; a pair is linked if the similarity exceeds the link threshold. Greene et al.: Jaccard overlap
of hard member sets (overlap allowed, no graded loadings), threshold θ set by hand, θ ∈ [0.1, 0.5] tested, low θ (≤ 0.3,
best 0.1) best on planted benchmarks; death after d = 3 unobserved steps. **Our adaptation:** soft profiles; the similarity
is the §7.2 link weight (or cosine); the threshold is replaced by a chance floor (entity-shuffle null, ticket 95), per
facet type. Scenarios: one link = continuation; one front linked by several = split; several fronts linked to one =
merge; no link = birth; front unlinked for d slices = death; >10% size change = growth/contraction. Chaining compares
each step with its latest predecessor, so gradual **drift** is followed (A → A′ → A″ even if A″ shares little with A).
Assumes: continuity is local and gradual; time order matters; layer-1 communities are stable enough to compare.

**Option 3b — a second factorisation over all slices** (Greene & Cross 2017, full text read; they track topics, not
heterogeneous communities). Table **B**: rows = every step community of every slice (Σ K_t rows); columns = recurring
entities; entries = the step community's loading (Greene & Cross keep each row's top t = 10-20 terms, rest 0). NMF
B ≈ U · V with k′ components:
- **U** (step communities × k′): one row per step community, **one column per dynamic community**; entry = how much
  that step community belongs to that dynamic community.
- **V** (k′ × recurring entities): one row per dynamic community (a single profile for the whole period),
  **one column per recurring entity**; entry = that entity's weight in that dynamic community.
Greene & Cross choose each window's k and k′ by a coherence score (k′ range 25-90) and assign each window topic to its
maximum-weight dynamic topic (hard). Time order is not used. Scenarios read from U: birth/death = first/last slice with
a step community; split = two step communities of one slice in one dynamic community; merge = one step community
loading on two dynamic communities (soft weights only; invisible under hard assignment). Each step community is compared
with a shared time-free centre, so strong drift shows as a break. Assumes: a dynamic community has a recognisable
profile over the whole period; time order does not matter; each step community counts as one row regardless of size
(later, larger slices with more step communities shape V more).

**Choosing the number of communities:**

| | Option 2 | Option 3b |
|---|---|---|
| per slice (K_t) | agreed procedure, free per slice | same (identical layer 1) |
| number of dynamic communities | not chosen: results from the linking | chosen: k′ (Greene & Cross by coherence; here by our procedure on B: hold-out cells of B + seed stability; cheap, B is tiny) |
| extra decisions | link threshold or chance floor; d (slices before death) | k′; top-t truncation; reading U (hard or soft cut-off) |
| wrong choice | too low: spurious continuations; too high: spurious births/deaths | k′ too small: new themes forced into old (false continuity); too large: continuations break |

**Intermittence** (a community absent for a slice, then back) is unrealistic in a field growing 2-4× a year (owner).
Option 2 can rule it out (d = 1) or check it (d = 2, count cases); 3b ignores order, so it can only be checked afterwards.

**Against the article's assumptions:**

| Assumption / feature | Option 2 | Option 3b |
|---|---|---|
| K changes over time | yes, per slice | yes, per slice |
| expansive growth | new step communities simply start chains | later slices give more rows; V leans to recent themes |
| modest distinctive continuity (12-28%) | a weak real link per step suffices | every step community must resemble one shared profile |
| fast-moving field (ChatGPT vocabulary break) | drift and breaks shown step by step | breaks and drift both show as new dynamic communities |
| soft, heterogeneous communities | soft similarity + per-facet chance floors (our adaptation) | per-facet block weighting in B (our adaptation) |
| intermittence unrealistic | can be ruled out | only checked afterwards |
| layer-1 instability | affects every link; mitigated by profiles from reproducible members only ("cores") | same, slightly dampened |

**Trade-offs.** Option 2: explicit drift and all scenarios incl. merges; simple, auditable rules; reuses §7.2 link
weights and labels; no extra factorisation; established (the survey's typical instant-optimal example). Against: only
consecutive slices compared, so one wrong link propagates down a chain; long-range relations only via the chain; the
threshold/chance floor is a decision. Option 3b: one global view (a weak step does not decide alone); long-range
recurrence visible directly; k′ chosen with our usual procedure. Against: merges invisible without soft weights; drift
penalised; time order ignored; one more choice and one more seed-dependent factorisation; less established for
heterogeneous communities. Main trade-off: following change step by step (2) vs identifying lasting themes over the
whole period (3b). Both can be computed from the same layer-1 fits.

**Owner (2026-10-16): option 2 fits the article's assumptions and goals better.**

**Mechanisms that would no longer fit, or would need changing (under 2 or 3b):**
1. The temporal prior (ticket 105, `chunk13v10.py`) is unused: fits are independent per slice. Its calibration and the
   KL vs bounded-alternative comparison of §7.1 become moot for the prior; the same question returns as the choice of
   similarity measure between profiles.
2. The §7.2 graded link weights become option 2's similarity, computed after the fits instead of steering T2; the
   descriptive labels map onto option 2's scenarios (for 3b they would be derived from U).
3. The planted two-slice worlds change purpose: from calibrating a prior to checking that the tracking recovers planted
   scenarios. 3b needs at least three slices.
4. The 2020/2021 trial cannot tell 2 from 3b (with two slices both reduce to linking two sets of communities); three
   slices would be needed (e.g. 2020, 2021, 2022: 91, 385, 744 articles).
5. Per-slice idf (ticket 106) stays necessary but weights the same entity differently in each slice: profiles must be
   compared after normalising each, or links reflect weighting rather than membership.
6. Global entity indexing across slices (chunk12 design) must be kept: both options need an entity to carry the same ID
   in every slice.
7. Social Approach 2 (no author facet) cannot use authors to link slices; authors carry 5-13% of author ties' continuity
   (§9.11) — a small but real loss for "same people" continuity.
8. Seed instability: both options compare single fits; using only reproducible members (per-entity Track A/B) or a
   consensus over seeds becomes a new step before the linking.

Sources: Rossetti & Cazabet 2018 (arXiv 1707.03186); Greene, Doyle & Cunningham 2010 (ASONAM; UCD report); Greene &
Cross 2017 (arXiv 1607.03055); Palla, Barabási & Vicsek 2007 (*Nature* 446).

### 7.4 Temporal options to test, year definition, and the production/visibility reasoning (2026-10-17)

**Decided (owner, 2026-10-17): the trial tests two temporal options.**
1. **Option 2 (clean):** each slice is fitted on its own articles and their ties only; step communities are linked
   across slices afterwards (graded links with chance floors per facet type, §7.2; scenario labels).
2. **Option 2 with temporal smoothing of the previous slice:** the fit for slice t also includes slice t−1's articles
   and all their ties, multiplied by a factor γ < 1. Linking as in 1.
Option 2 corresponds to MetaFac's α = 0 baseline ("MF"). The smoothed variant belongs to the survey's "temporal
trade-off" class (a sliding window carrying *data*), unlike MetaFac's prior, which carries the previous *model*, acts
only on persisting entities and needs matching. The MetaFac-style prior (ticket 105) is not part of either: options 2
are instant-optimal by definition; the prior would act on ~5% of entities, mostly generic ones (CORPUS_STATISTICS.md
§9.11); MetaFac assumes the same entities and a fixed K; and it imposes the continuity we want to describe (ticket 105 T3
showed it hiding splits). γ, and whether γ is fixed or estimated, is not yet decided. The owner will also discuss both
options with colleagues.

**Year definition — proposed (assistant): OpenAlex `publication_year`.** It is available for all but 22 articles,
equals the electronic-publication year for 98.5% of articles, and was used in every slice count recorded so far (only
§9.3's by-year byline table used PubMed electronic-else-print, which agrees for 97.9%). CORPUS_STATISTICS.md §9.12.

**Recorded, not used now (owner, 2026-10-17): production date vs visibility date.**
*Reasoning.* An article has a production time (written, submitted) and a visibility time (published, readable by
others); slices by publication year use the observer's clock. An article plays two roles: as a *product* it expresses
the landscape its authors could see while writing, plus their own contribution; as a *resource* it joins the landscape
from publication on. The vanguard of the research front (work written at the same time) is largely invisible to
authors, so co-membership of a community does not require mutual awareness (communities here rest on shared
associations, venues and institutions, not interaction; consistent with an idea enacted similarly in several places),
but influence can only run from the visible to the not-yet-written.
*Measurement idea.* Production slices by received date (87% coverage, §9.12); landscape L(τ) = articles visible before
τ (by e-publication date). Each recurrence of an entity can then be classified as *inherited* (its earlier use was
visible before the later article was produced) or *parallel* (not yet visible: independent emergence, Merton's
multiples, or a common outside source), splitting observed continuity into possible uptake and convergence.
*Why not now (owner):* the received → publication lag (median 140 days) varies by discipline and journal; writing can
take years, with intermediate results shown at conferences and seminars; preprints make work visible earlier than
e-publication; received is the latest, not the earliest, point of the writing. To revisit in later articles or if
reviewers ask.

**Slice boundaries (owner, 2026-10-16/17).** Calendar boundaries are arbitrary except where a rupture anchors them
(ChatGPT, 30 Nov 2022, visible as the dip in distinctive continuity, §9.11). Shifting boundaries by months or changing
slice length can change the communities found (the modifiable temporal unit problem; Sulo, Berger-Wolf & Grossman 2010).
Candidate response, not decided: report only scenarios that survive shifted boundaries (e.g. January vs July starts,
12 vs 18 months).

### 7.5 Planted worlds: how they would be used (explained 2026-10-17; role and limits per owner)

**Architecture deferred (owner, 2026-10-17):** a robustness check run after the analysis; its exact design is discussed later. Open question for then: how well the generator reproduces higher-order patterns of the data, and whether specific synthetic-data methods exist for heterogeneous, soft-partition networks.

**Role (owner):** a validation / robustness check of the procedures, not a step that produces results. It is still the
main evidence that the K procedure works for this model type, for which no published validation exists, so it supports a
methodological claim (appendix material).

**Procedure.** (1) Skeleton from a real slice: entity sets of one social approach's relations (each approach validated in
its own representation, never against the other), each entity's number of ties per relation, the weighting rules.
(2) Planted truth: K_true; each entity's true profile drawn from a Dirichlet distribution (two mixing levels); per relation
a community-to-community propensity table (e.g. 0.8 within / 0.2 across, and a harder 0.6 / 0.4). (3) Ties: each entity
keeps its real number of ties per relation; partners drawn with probability ∝ the partner's real degree × the profiles'
overlap through the propensity table. (4) Our procedure runs unchanged. (5) Scoring against the truth.
Uses: **U1** does the K procedure (hold-out, stability gate, one-standard-error rule) recover K_true (or put it in its band),
and how close are fitted to true profiles per facet; **U2** multi-slice worlds (persisting entities in the measured
proportions, §9.11) to test the linking, fitted at the true K. Cost: U1 ≈ 100-150 fits in a reduced design (1 mixing level,
2 planted K, 2 worlds, coarse K grid); U2 tens of fits.

**What the generator reproduces and what not.** Reproduced: each entity's exact number of ties in every relation (so an
entity's degrees across relations keep their real correlation), approximately the partners' degrees, the weight
distribution. In the FINDINGS §25 version real weights are *shuffled* onto the new ties, which breaks the link between a
tie's weight and its endpoints (damping by author count, idf by document frequency). **Improvement adopted for planning:**
generate tie counts, then apply chunk12v3's own weighting rules, so weights relate to degrees exactly as in the real
matrices. Not reproduced: co-author cliques, nesting of child hyperedges in parents (sentence structure), topical
clustering of journals, the fact that one article's ties in different relations come from one text.

**Scenarios are interpretations (owner, 2026-10-17).** Under soft membership there is no established quantitative
definition of split, merge etc. (Palla et al. and Greene et al. define them on crisp sets), and soft partitions allow
more nuanced ideal types of redistribution of loadings. So U2 does not score scenario labels. It plants a **known
redistribution of membership mass between communities (a true transition table)** and checks how well the estimated
graded link weights (§7.2) recover that table; scenarios remain an interpretive reading of the recovered table.

## 8. Agenda: revisiting the optimisation mechanisms (recorded 2026-10-19, owner)

Purpose: before the trial, decide for every in-loop weight and outer-loop penalty whether to keep, recalibrate,
redesign, move between loops, or drop it; then decide whether Optuna stays. Nothing is run until the prerequisites
are done, so every test runs on the real trial data.

### 8.0 Inventory (from CLAUDE.md §4, §8; FINDINGS)

**Inner loop** (changes what a fit converges to):

| Mechanism | What it does | Current | Toy evidence | Problem type |
|---|---|---|---|---|
| Off-diagonal coupling λ_z_offdiag | penalises ties between different communities in Z | only Optuna-tuned weight (1e-4–1) | inflates auth/affil collinearity (0.26 → 0.70 → 0.87 at 0/0.05/0.5); pushes against shared semantics (§9); no consistent lever on shared-term concentration | side effects; role unexamined |
| L1 sparsity λ_l1 (tickets 73/77/78) | mean of all U_norm entries | fixed 0 | penalises column concentration, not row concentration; spread memberships at every weight, cost fit; the "hairball" it targets did not occur | mechanism design |
| In-loop domain balance (ticket 82) | per-community social vs semantic membership share, dead band ±0.15 | weight 0 | 1 of 20 settings beat seed noise; weight 5 broke convergence; input carries no signal on the toy (FINDINGS §25) | data/input; toy noise floor |
| Concentration floor (ticket 102) | pushes shared semantic entities to spread as far as their articles do | 0.1 / 0.8 / warm-up 300 | shared-but-exclusive 25% → 3%; spreads niche words (ticket 103); gradient leak onto other rows (ticket 105 T1g) | side effects; defined on words |
| Word-similarity regulariser (ticket 107) | pulls hyperedges with shared words together | new | — | calibration |

Also: early stopping (two-part test), extended epochs (ticket 96), initialisation.

**Outer loop** (scores finished fits; Optuna's second objective; decides front and knee; changes no fit):

| Penalty | What it does | Weight | Toy evidence | Problem type |
|---|---|---|---|---|
| Collapse | largest community's share of reconstructed mass > 0.60 | 1 | fired 2/12 cells, only K = 2; fixed threshold means different things at each K | threshold, K-dependence |
| Coherence | anchor-relation Z diagonal cohesion, weakest community < 0.50 | 0 | never fired; gauge-dependent input | input validity |
| Item attribution (Penalty_A) | do the semantic items a community's articles use belong to it | ½ of semantic_pen | with Penalty_B, most of Optuna's 2nd axis | re-specify propagation paths (atoms gone, child/cousin merged) |
| Shared-term concentration (Penalty_B) | shared entities > 0.85 in one community, × sharedness | ½ of semantic_pen | 21% of shared entities > 0.9 without the floor; λ_z no lever | re-specify facets |
| Domain balance (outer) | as in-loop | 0 | planted: ~0.10 regardless of truth | data/input |

Aggregation: unweighted sum (ticket 35). **Diagnostics only:** low-mass ("ghost") communities (ticket 86; no noise floor),
unengaged entities (ticket 100), auth/affil collinearity, Track A/B (now the K gate).
**Linked pairs:** concentration floor ↔ Penalty_B; in-loop ↔ outer domain balance. **Opposing pushes:** the concentration
floor spreads shared entities; a row-sparsity redesign of L1 would concentrate them.

### 8.1 Prerequisites (owner, 2026-10-19)

- **P1 — data quality check and improvement:** author matching where applicable; checking that the postprocessing
  (G3) worked exactly as intended; related input checks.
- **P2 — weighting inside each relation matrix and building the chunk12v3 relation matrices** for the trial,
  following the decided representation and options. Weighting changes the inputs most mechanisms read (idf decides
  which entities count as shared; per-relation scaling sets domain shares and λ_l1's useful range; weights set the
  scale of every noise floor).
- Then: adapting the model code (masked loss, new facets, same-facet relation, regulariser, re-specified penalties).

### 8.2 Agenda

1. **Principles** (one discussion, no data needed): (a) what each layer is for — in-loop terms shape fits, outer
   penalties rank finished fits, diagnostics describe; (b) when an outer penalty gets an in-loop counterpart — proposed
   rule: only if an independent check can show the counterpart does the right thing (ticket 102's text check), else it
   only lowers its own score (ticket 82's exploit); (c) every threshold set against the measured seed noise of its own
   quantity on trial data, and checked for K-dependence (§6.5 point 2); (d) what counts as "working": fires on real
   data; moves its target beyond seed noise; an independent check agrees; acceptable cost in fit and convergence.
2. **Mechanism by mechanism** (purpose under ticket 107; re-specification for the new facets; toy evidence; options:
   keep / recalibrate / redesign / move between loops / drop; what the trial must measure):
   (a) off-diagonal coupling; (b) L1 sparsity — drop or redesign as row sparsity, with its interplay with the floor;
   (c) concentration floor — new facets, gradient leak, how "shared" is computed (ticket 103), interplay with idf;
   (d) domain balance in-loop and outer — signal at trial scale, recalibrated band, sensitivity, or drop;
   (e) collapse — K-aware threshold, possibly merged with low-mass detection into one mass-distribution check;
   (f) coherence — gauge-free redesign or drop; (g) Penalty_A — propagation paths for child × parent and child ×
   article, parents as targets (ticket 66); (h) Penalty_B — facets, 0.85 threshold, relation to the floor;
   (i) low-mass communities — outer check or diagnostic; (j) word-similarity regulariser calibration;
   (k) aggregation — unweighted sum or weights (ticket 35).
3. **Placement map:** one table — each mechanism in-loop / outer / diagnostic / dropped, with a value or calibration rule.
4. **Optuna decision:** with the map known — how many weights remain tuned and how many penalty components rank fits —
   decide whether Optuna's multi-objective search stays or a grid suffices.
5. **Trial test design** for the points that need data, judged by 1(d); folded into the trial procedure.

### 8.3 P1 progress and decisions (2026-10-19)

Scripts: `fullscale_pipeline/corpus_statistics/trial_p1_checks.py`, `trial_institution_lineage.py` (read-only except
an OpenAlex institutions fetch, cached in diagnostics).

- **E, trial article set (owner):** research articles and all reviews (narrative, systematic, scoping,
  meta-analyses); OpenAlex `publication_year`. Excluded: 13 editorials/letters/comments in 2021. The 5 articles of 2021
  without a PubMed ID (type not available from PubMed) were checked by OpenAlex type (all "article") and by reading
  their texts: four research papers, one survey / research-agenda paper (PMC8523728, header classification "survey
  articles") — all included (owner). **Trial: 2020 = 91 articles, 2021 = 372.**
- **F, C:** coverage is high (2020/2021: 99% of articles with an OpenAlex institution ID and with an author whose
  OpenAlex ID or ORCID we know; 95-98% of bylines with a known author ID; 96-97% with an institution ID); every trial
  article has an NLM journal ID; 148 journals, no title or ID duplicates.
- **D, G3 invariants (automated):** 0 violations of every rule checked (6,821 units, 8,137 parents). 9 parents hold the
  same word set twice (e.g. "DistilBERT: DistilBERT is…", "from dataset to dataset"). **Decision (owner):** these are
  genuine; with hyperedges as the units, a repeated hyperedge in a sentence counts twice, as the text was written.
- **D, observation (owner asked to record):** "BERT_base has less parameter information compared with BERT_large"
  becomes `(focal_he bert/C/focal)` and `(sibling_he bert/C/focal)`: G3's canonical focal terms map both variants to
  `bert`, so the base/large contrast is lost. More generally, lemmatisation and canonicalisation merge forms the authors
  kept apart. **For interpretation (owner):** once the model has run on the processed, lemmatised words, community
  descriptions should be traced back to the original wording (G3 keeps, for every atom, its source words and token
  positions), so that readings are made on what authors actually wrote.
- **B, institutions (owner: use the higher-level entity, for a denser network):** 817 institutions on trial bylines,
  all fetched. Finding: OpenAlex `lineage` lists all ancestors (sorted by ID, not by level); institutions often have
  several parents (joint research units, e.g. Institut Necker: CNRS, Inserm, Université Paris Cité); and the registry
  data has errors (CONICET listed under a geodetic observatory). "Top of the lineage" is therefore not well defined;
  the rule is being decided. 86 institutions have a parent present; 64 "related" (not parent-child) pairs among
  top-level entities (often university–hospital) stay separate under any lineage rule.

**P1 results, continued (2026-10-19).**
- **A, author linking checks** (`name_linking_checks.py`, `pmc_vs_pubmed_authors.py`): (A1) linking bylines without an
  ID *to each other* (hide-and-recover, 3 random draws of 8,374 hidden bylines): with the current name key, precision
  0.85-0.94, recall 0.88-0.92; 0.30-0.65% of the hidden bylines end up in a wrong pair. (A2) name-key variants: surname
  + first given name (current) is the best balance; first initial only doubles to triples false merges (precision
  0.63-0.77; 6.2% of single-article people falsely linked to a known person vs 3.5%); all given names loses recall
  (0.81-0.85) for no gain in precision. Linking to a person whose ID we know, current key: 83.7% correct, 0.4% wrong
  among people with 2+ articles; 3.5% of single-article people linked (upper bound of false merges, part are OpenAlex
  profile splits). (A3) PMC full-text author lists vs PubMed's: 97.3% of PubMed authors found in the PMC header
  (trial 99.8%); the header lists more lines in 48% of articles because it also carries reviewers, titles and author
  biographies, and is cut at 120 lines (very long author lists). PubMed's list is the cleaner source; no sign of
  authors missing from it.
- **B, institutions (option 4, owner):** rule (1), climb to the nearest university parent and never above it, plus a
  hand review of every trial institution with a university parent: `corpus_statistics/institution_review_trial.csv`
  (108 cases: 35 merged, 73 kept, each with its reason; script `institution_review_trial.py`). Merged: sub-units,
  university medical centres and hospitals recorded as part of the university, joint research units with one
  university parent, one duplicate record (Université Jean Monnet). Kept: universities under state systems, federations
  or alliances (57), units with 2+ university parents (8), agencies, national laboratories, health clusters, trusts.
  Two caveats: OpenAlex/ROR record today's structure, not 2020-2021 (e.g. UT Health San Antonio merged with UTSA only in
  2025, kept apart); rule (1) does not merge sub-units of companies or agencies (Google DeepMind / Google, Data61 /
  CSIRO, CNR institutes).
- **D, hand-read sample:** `fullscale_pipeline/g3_trial_sample/sample_30.md` (30 units, seed 107) and
  `reading_notes.md`. Structure coherent; seven points noted, chiefly: atom provenance is per sentence, not per
  hyperedge, when a word repeats (21.1% of trial hyperedges), which limits tracing communities back to the exact
  original words; the same focal term sometimes typed `/en` instead of `/focal` (separate entities); participles as
  the parent's verb group (`use` 9.7% of parents, `base` 1.8%).

**P1 follow-ups (2026-10-19, owner questions on D and A).**
- **A applied (owner):** current method (corroborated rule, surname + first given name) on the 74 trial bylines
  without an ID: 23 linked to a person whose ID we know (all with a shared institution or co-author on other
  articles), 2 ambiguous (not linked), 49 stay their own person; no unlinked-to-unlinked link. Table:
  `corpus_statistics/author_links_trial.csv` (script `author_links_trial.py`).
- **D2, focal terms typed as ordinary words (owner: they should be part of focal hyperedges; investigate):** 177 of
  10,160 focal-term occurrences in the trial (1.7%), in 68 articles, carry `/en` instead of `/focal`. Causes, from
  every case: (a) hyphenated model names containing a focal term (FARM-BERT, CT-BERT, Med-BERT, Bioclinical-BERT,
  Sentence-BERT, bert-base-uncased, roberta-large): G2 keeps such words apart and the focal-mention detector does not
  list the focal part; (b) other spellings: "Bert", "bert", "Roberta" (possibly deliberate protection against personal
  names, to be checked); (c) lists where only some occurrences were listed. Unlisted occurrences also get no parent of
  their own. Fix: to be decided (owner).
- **D3 (owner):** by rule the main verb of the sentence becomes a cousin when the focal term's clause is a participle
  phrase; acceptable for the analysis if sibling and cousin are assigned correctly; at interpretation the main arguments
  may be shown separately.
- **D4.1, numbers and special characters (owner: keep legitimate content, drop formatting remnants):** in the trial,
  numbers in 2.6% of hyperedges (mostly plain-text sizes, counts, scores — kept by the 2026-10-10 decision), decimals
  1.3% (result values), letters+digits 2.4% (names: f1, word2vec, t5, i2b2; and `REFnnnnnn`), years 0.2%, other encoded
  characters 0.3%. Leaks found: PDF ligatures (`ﬁne`, `classiﬁcation`: to normalise, not drop); words glued to
  brackets (`lucene(a`, `rnn)-based`); formula fragments (`o(n2`, `p(st+1|s0`, `t′`, `xα`); stray quote atoms;
  footnote/citation markers left as numbers ("PubMed 1 abstracts", "KeyBERT 12, 28"; the open RL-118 leak); one
  figure-panel caption kept as prose (PMC8490152, 40-atom sibling). Accented names (Fréchet) are legitimate.
  **`REFnnnnnn` tokens are global cited-work IDs** (`citation_works_v2.jsonl`: one ID per work, e.g. REF000001 cited
  in 22 articles), kept by decision (G3 §6 item 6): a narrative citation inside a focal sentence is a partial citation
  link between articles (293 occurrences in the trial).
- **D5, one focal term attached to the wrong clause (PMC9166894.r1.L288.S6.U1):** a parser error, not a G3 rule: in
  "it is imperative that the item generating method—in our case language models—reliably produce items …" graphbrain
  did not build "… method … reliably produce items" as a clause; it left the noun phrase with the focal term and
  "reliably produce items …" as two separate pieces of the argument of "it is imperative". G3 therefore found "it is
  imperative" as the clause governing the focal term (verb group `be`) and put "reliably produce items …" in a sibling.
  With a correct parse the parent would be `produce`, and "it is imperative" a cousin.
- **D6, lone modifier as a cousin:** by rule (G3 §3: focal atoms are left out of cousins). In a sentence with two focal
  occurrences, each parent's cousins include the other parent's clause with its focal atoms removed; "fine_tuning the
  pretrained language model BERT" thus leaves the cousin `(cousin_he pretrained/M/en)` for the second parent.
- **D7, long hyperedges:** 68 of 49,867 trial hyperedges have 20+ atoms; the largest are plain-text lists (models,
  methods, workflow steps) and long argument chains — legitimate — except the figure-panel caption above.

**P1 owner decisions and follow-ups (2026-10-19, RL-149).**
- **A:** approved (current linking method, applied).
- **D4.1 and the caption (owner: approve the handling, implement; the caption must be dropped):** implemented in the G3
  test script and checked on shards 0 and 20 (G3_POSTPROCESSING.md §6 item 9). Not yet run on the corpus: one G3 rerun
  after the D2 decision.
- **D2 (owner: treat hyphenated model names as focal words, other spellings too when LLMs are meant; rerun earlier
  stages if needed; new ids/hashes must not conflict; reprocess misclassified sentences):** the corpus-wide scan shows
  a larger gap than the trial's G3 output: (1) spellings of listed terms — hyphen prefix (XLM-RoBERTa, Med-PaLM,
  CT-BERT) 11,916 occurrences, glued versions (ChatGPT4, ChatGPT3.5) 2,758, plurals (GPTs) 1,194, checkpoint names
  (bert-base-uncased) 3,073, "language-model" 480; (2) a bug: "LLaMA" alone never matched (2,403); (3) "Bert" / "Roberta"
  in articles with LLM evidence are the models (samples read); (4) model names not on the list (DistilBERT, ClinicalBERT,
  scGPT, MedGemma; tools such as BERTopic, BERTScore, GPTZero, Ollama) 46,991 occurrences in 5,180 articles. Few new
  articles; about 9,000 new focal lines from (1)–(3) and 21,000 more from (4). Proposed: incremental rerun (P2 with the
  new matcher; P6 only on new blocks; existing REF ids frozen, new works numbered from REF010556; G2 only on new
  sentences, plus re-parse of sentences where an unlisted name was joined by G2's hyphen rule if (4) is adopted; G3 in
  full; checks that every existing sentence id and hash is unchanged and new ids are disjoint). **Waiting for the owner:**
  fix (1)–(3)?; hyphenated name as one focal atom (`ct_bert`, like `smiles_bert`) or the listed part only?; adopt (4)
  (a definition question: what counts as an LLM)?; the incremental route?
- **D5:** the error is in spaCy's dependency parse used by graphbrain (G2), not in G3: "models" read as the verb of
  "method", "produce" a second clause without a subject. The focal hyperedge in that unit is "the item generating method
  — in our case language models" (`case generate item language_model method our`). Writing the dash-enclosed aside in
  brackets gives the correct parse; a G2 rule for it is possible (re-parse), not measured, not applied.
- **D6:** the rule "focal atoms are left out of the cousins" (G3_POSTPROCESSING §3) comes unchanged from the June toy
  script; no reason is recorded. Inferred purpose: the focal term is the centre of every parent and is in every
  sentence, so in cousins it would make unrelated contexts alike and repeat a second parent's focal term. Relevant to
  chunk12v3, where child and cousin hyperedges share one layer by word set: keeping focal atoms in cousins could make a
  cousin identical to a focal hyperedge. Owner to confirm or revise.
- **Owner decisions, 2026-10-19 (second round; RL-150, RL-151; the full list is in ROLLING_NOTES.md):** D4.1 approved
  (production run with the other G3 changes); D2 groups 1–3 approved, hyphenated names as one atom (`ct_bert`); group 4
  approved with false-positive checks, counting only research articles, historical articles and reviews (draft rule:
  ~97% precision on 150 occurrences; 50 candidate new articles to review); new papers only through missed models, not
  tools; incremental run approved in general, to be launched after the remaining details are agreed; D6 sentence-centric
  parents decided (G3_POSTPROCESSING.md §6 item 10); D5 count approved, after the corpus is fixed. Found on the way:
  B06 (inverted auxiliary) fixed in G3; A12 a test-harness artefact; curly apostrophes after plurals derail spaCy's
  parse in 880 units — a G2 fix with re-parsing proposed for the incremental step.

## Addendum 2026-10-21 — child/cousin ties to parents (owner decisions; details in G3_POSTPROCESSING.md §6 item 11)
- Two relations between the merged child-level facet and the parent facet: **nested in** (child × parent) and
  **co-occurs in the sentence** (cousin × parent), each with its own core matrix and weight, as `M_Child_Parent` /
  `M_Cousin_Parent` in the toy; a regulariser is the fallback if the co-occurrence relation is too sparse.
- Cousin ties in sentences with several parents are not split 1/n; generic, frequent material is to be down-weighted by
  a frequency-based rule tested before use (candidate: tf-idf; conflicts to weigh: §4.21 idf versus shared meaning).
- Provenance carries occurrence rows with typed links (contains / co-occurs / embedded); the builder must use each
  cousin's full parent list (chunk12 kept only the last parent per sentence).

## Addendum 2026-10-22 — model-name granularity and sparsity (owner, to do before M1 weighting is fixed)

G3 now writes one focal atom per model at product-tier level (owner decisions RL-177/RL-179: `gpt_4o` and `gpt_4o_mini`,
`gemini_2_5_pro` and `gemini_2_5_flash`, `claude_sonnet_4_5` are distinct atoms; sizes, base/large, instruct/chat,
dates and snapshot codes are folded away). This splits the counts of a model family over several atoms (e.g.
`gpt_4o` ≈ 32,000 mentions vs `gpt_4o_mini` ≈ 2,500). The family relation is not stored by G3; it can be recovered from
the canonical names (family + version + tier). **To do in M1 (owner, 2026-10-22):** count the sparsity of the
semantic facets (focal atoms, the child/cousin hyperedges they enter, their degree distributions and the share of
degree-1 entities) under different model-merging levels — (a) as written by G3 (tier level), (b) version level (tiers
merged: `gpt_4o_mini` → `gpt_4o`), (c) family level (`gpt_4o` → `gpt`), (d) generic terms only (every named model →
its class, e.g. `llm`) — before the weighting and the choice of level are decided.

## Addendum 2026-10-23 — numbers: members of hyperedges, not of the model (owner)

Owner: numbers are valuable for interpretation but usually meaningless for the modelling; keep them as members of
hyperedges, but leave them out of the modelling (weight 0 or a mask). G3 keeps plain-text numbers as atoms (2026-10-09
decision): 144,599 integer atoms of 1–3 digits plus decimals in the RL-182 output; a sample of 40 read in context: about
5–10% are leaked labels or citation numbers (RL-118), the rest counts and values ("29 of 31", "45 answers", "12 GB").
**To do in M1:** build the entity identity of every hyperedge (the merged child/cousin layer's word-set key, parents'
keys) and every word-level relation or regulariser from the non-number atoms only; the numbers stay in the G3 records and
the occurrence rows, so interpretation shows them. Consequence to state and measure: hyperedges that differ only in their
numbers become one entity ("29 of 31 questions" and "28 of 31 questions" -> one `question` hyperedge), which lowers
sparsity; count it together with the model-merging levels (addendum 2026-10-22). This also settles the RL-118 leak for
the model without a G3 change.

## Addendum 2026-10-24 — first trial: no temporal extension, a subset of the corpus (owner)

Owner, after discussion with colleagues: (1) **the first trial does not test the temporal extension** (option 2 and its
γ variant of §7.4 are set aside for now, possibly returned to later); the trial is cross-sectional. The 2020/2021 slices
(§4) were chosen for the temporal test and are no longer binding. (2) **The trial runs on a subset of the data: one
discipline, or articles around one model or model family**, chosen after looking at the data — articles per model family,
whether OpenAlex or PMC/MEDLINE/PubMed/Entrez metadata reliably separate disciplines — and preferring a part with cleaner
data (fewer missing affiliations, better author identification).
Data look (assistant, 2026-10-24, RL-186 output): 41,753 valid articles, 27,271 of them from the current corpus with
OpenAlex records (author IDs, one OpenAlex field per work, MEDLINE publication types), 14,481 new without any author or
field data yet (B3); institutions fetched only for the 2020/2021 trial. Fields (current part): Medicine 10,391,
Computer Science 5,525, Biochemistry/Genetics/Molecular Biology 3,112, Psychology 1,853, Social Sciences 1,412, Health
Professions 957, Engineering 817, Neuroscience 657; all authors with an OpenAlex ID 72–83% per field; 2024–2026 hold
almost all articles. Families (articles mentioning / the only named family / main family): GPT 20,603 / 10,028 / 16,983;
BERT 9,895 / 3,184 / 6,016; Gemini 4,161; Llama 3,284; Claude 2,988; DeepSeek 2,327; RoBERTa 1,941; ESM 1,162 / 516 /
849; 9,796 articles name only generic terms. Proposed (not decided): a field subset of 1,500–3,000 articles, the field
cross-checked against NLM's journal Broad Subject Terms; no filtering on author-ID completeness (it removes large teams
and 2025 papers disproportionately), author linking instead; before choosing, an OpenAlex fetch for all valid articles
(topic hierarchy, authorships with institutions) and the NLM catalog subjects, to tabulate candidate subsets with their
coverage. Open for the owner: field or family; the discipline source; the fetch; the size.

## Addendum 2026-10-24 — author and institution merging rules (owner, after RL-188)

Decided (owner): (1a) a person split over several OpenAlex author IDs is merged through a shared ORCID; (1d) where the
ORCID PubMed gives (publisher-supplied) and the OpenAlex ORCID of the aligned authorship differ (892 bylines), PubMed's
is used; (2) institutions are climbed to the nearest university parent (never above it, rule (1) of §8.3 B); the hand
review and the treatment of bylines without an institution ID are decided after the trial subset is chosen. Open:
(1b) OpenAlex IDs carrying 2+ PubMed ORCIDs (102 IDs, 522 bylines, 485 articles, 5 articles with only such authors):
owner's proposal to drop these authors and the articles where they are the only authors, against splitting the ID by
ORCID (sample read 2026-10-24: mostly OpenAlex merging different people, several on the same article; a few one person
with two ORCIDs); (1c) linking of the 12,106 bylines without ID or ORCID, option B (step 1 of author_links_trial.py:
to an identified person, 1,424 bylines) or C (steps 1 and 2: also among unidentified bylines, +548), measured on the
working corpus v2 (scratch, read-only).
*[2026-10-26: the open points above (1b, 1c) are decided, and two further cases (an ORCID on different people; an
OpenAlex ID on different names) were found and decided. The full rule set R1–R6, the name test, examples, checks, bias
and the simulated result are in `fullscale_pipeline/AUTHOR_IDENTITY.md` (RUN_LOG RL-189 to RL-192); that file supersedes
this addendum on author identity.]*

## Addendum 2026-10-26 — the first-trial subset: Psychology (owner)

Owner decisions (2026-10-26, RL-193 to RL-198): the first M1 trial runs on one discipline, **Psychology**, defined by the
OpenAlex primary-topic field (not by any of an article's topics; NLM Broad Subject Terms rejected as the source: 54%
coverage, megajournal "Science"). Reason: few texts, several model families, general and possibly subfield-specific
models; Biochemistry (model-rich, narrow models tied to subfields — an expected finding, not a defect) is larger than
needed for a trial. Article list **`fullscale_pipeline/corpus_statistics/psychology_trial_pmcids.txt`, 2,140 articles** (RL-199)
(`psychology_trial_subset.py`; each exclusion with its stage and reason in `psychology_trial_exclusions.csv`):
- 2,255 working-corpus-v2 articles with primary field Psychology;
- −93: no focal unit left after the demotion of non-LLM "language model" mentions (G3_POSTPROCESSING.md §8 item 12;
  their G3 records for M1 come from `PG/g3_v2_lm_demotion/demoted/` for the 120 rerun articles, `PG/g3_v2/` otherwise);
- −22: dropped by the owner in the field review (assistant flagged 67 of 2,162 on title, keywords, journal, abstract
  when unsure; owner kept 45, dropped 22; RL-199).
Owner rulings for "Psychology": HRI and social robotics, emotion recognition and mental-health detection belong to it.
Not yet done for the subset: the institution hand review and bylines without an institution ID (§ author addendum),
the M1 author table (AUTHOR_IDENTITY.md R1–R6 applied to data), A3 part 2 (lost sentences).

## Addendum 2026-10-26 (2) — data preparation for the Psychology trial (RL-200)

Done: institutions merged at the organisation level (AUTHOR_IDENTITY.md §9; 2,716 → 2,570 for Psychology); the author
table built by the decided rules (§10; 10,051 persons in Psychology); trial list 2,139 (one article lost its only author).
Open for the owner: (1) research organisations as umbrellas or as one institution; (2) health systems; (3) the French
experimental universities; (4) bylines without an institution ID (proposed: no affiliation tie); (5) the 4 no-PMID
articles; A3 part 2 proposed deferred (14 Psychology articles affected). Next after these: weighting inside each matrix
(§5 item 3), then the chunk12v3 builder.
*[2026-10-26, owner: question (4) decided — bylines without an institution ID get no affiliation tie, provided the model
does not read the absence as similarity. Requirement for chunk12v3/chunk13: unknown affiliation cells are missing
(excluded from the loss, the same mechanism as the hold-out's hidden cells, §6.8), not observed zeros; chunk13's
squared-error loss over all cells would otherwise treat them as "no tie" and pull such entities towards
low-affiliation communities. Psychology: 578 persons (approach 1), 32 articles (approach 2) wholly unknown.]*
*[2026-10-26, owner, to discuss with the missingness technique: an alternative to missing cells is **placeholder
organisations** — each unknown affiliation becomes its own placeholder ("organisation placeholder 1, 2, … n"), which
behaves like an ordinary affiliation tie to a one-member organisation; two bylines share a placeholder only where there is
evidence of the same unknown organisation (e.g. the same organisation named in the affiliation text, or co-authors of one
article with an identical affiliation). Psychology: 627 no-ID bylines with affiliation text, 413 distinct strings. The
exact technique (missing cells vs placeholders) is decided later.]*
*[2026-10-26, owner (RL-201): research organisations not merged; health systems merged (hospitals into their system,
teaching hospitals into their system, never into the university); French experimental universities stay groupings;
Psychology trial 2,716 institutions → 2,519 organisations. No-PMID articles: authors from OpenAlex, pending a check
against the PMC header (2 of 4 differ in author count).]*
*[2026-10-27, owner confirmed A3 part 2 and its inclusion in the trial (RL-203): 379 lost sentences recovered in 180
articles (G2 v3.3, G3 rerun, no article status change); the Psychology trial gains 33 sentences in 11 articles; the
trial list (2,139) and the language-model demotion are unchanged.]*

## Addendum 2026-10-27 — relations versus regularisers for the trial (owner)

Matrices in both approaches (owner): child×parent, child×article (child = the merged child/cousin layer), article×journal;
approach 1 author×article, author×affiliation; approach 2 article×affiliation (and article×article, below).
- **child×child (shared words): regulariser** (confirms §1). The weighting of the similarity is to be discussed.
- **affiliation×affiliation (shared people, moves): neither a matrix nor a regulariser** for now. Measured in the
  Psychology trial: of organisation pairs linked through one person, only 65 are not already on a common article.
- **article×article (approach 2): open — sparsity.** Psychology trial: 2,139 articles; 1,108 article pairs share at
  least one person (2,216 non-zero cells of 4.57 million, 0.05%); mean degree 1.04 (2.99 among the 741 articles with any
  tie); 1,398 articles (65%) have none; 208 separate groups, the largest 159 articles, 144 of them a single pair. Full
  corpus for comparison: mean degree 4.14, 62% of articles tied (the discipline subset cuts collaborators' other papers).
  Reference points: a block-model community structure is detectable from a graph alone only above the Kesten–Stigum
  bound d·λ² > 1 (d mean degree, λ ≤ 1 the community signal), so d ≈ 1 is at the limit even for perfectly separated
  communities; low-rank recovery needs of the order of n·K·log n observed entries (≈ 65,600 for K = 4). Discussion with
  the owner pending.
*[2026-10-27, owner-requested check of option (b), `corpus_statistics/coauthor_graph_check.py`, outputs in
`corpus_statistics/coauthor_graph/`: the in-subset shared-author graph is not one giant component (largest 159 of 741
tied articles, 208 components); Leiden and Louvain agree (215 communities, modularity 0.97, 99% of tie weight inside
communities); inside the largest component Leiden finds 8 groups joined by single bridges. Tied articles share an
OpenAlex topic 45% of the time against 8% at random. Two-hop ties through off-subset corpus articles (owner's option 1):
874 tied articles (+133), mean degree 2.01, but one component of 430 articles, topic agreement of tied pairs 34%, and 62%
of the two-hop pairs come from a few outside articles with 10+ trial neighbours. Hy-MMSBM (K = 4, persons as
hyperedges): memberships nearly hard (93.7% of articles ≥ 0.9 on one community); with only 4 communities it must lump
unrelated groups (topic agreement within a community 14%, near random 8%).]*

## Addendum 2026-10-27 (2) — weighting of the two regularisers (discussion; decided points marked)

- **child×child similarity: model names included (owner):** model choice signals tasks and community or disciplinary
  preferences. Measured (2,000 sampled child hyperedges, idf cosine, 5 nearest neighbours, similarity ≥ 0.5): ties 7,127
  without model names, 7,054 with them; with them 638 ties (9%) involve a model name and 126 (1.8%) rest on a model name
  alone. Word weights: a child hyperedge is a word set, so each word occurs once in it and tf-idf equals idf.
- **article×article weight — DECIDED (owner, 2026-10-27):** w(A,B) = Σ over shared persons p of 1/(n_p − 1), divided by
  the size of the union of the two author sets (prolific authors and team size both damped).
- **Per-entity normalisation — DECIDED (owner, 2026-10-27):** row normalisation of W: each entity's tie weights divided by
  its total, R = (λ/N) Σ_i Σ_j (W_ij/d_i)‖p_i − p_j‖² (equal memberships cost nothing; the symmetric normalised
  Laplacian tr(PᵀL_sym P) was dropped because it penalises equal memberships of entities with different tie totals).
- Open: the distance, the scaling of the penalty by the number of
  regularised entities, k and τ for child×child, λ calibration (held-out ties).
- **child×child tie selection — DECIDED (owner, 2026-10-27):** for each child hyperedge keep the 5 most similar
  hyperedges with similarity ≥ 0.5, all candidates tied at the 5th value kept (5 is a minimum); a tie is kept if either
  side selects the other. Be ready to switch to k ≈ 11 (order of log n for 98,949 hyperedges, von Luxburg 2007).
- **λ calibration — DECIDED (owner, 2026-10-27):** outside Optuna; target = prediction of held-out ties (child×article,
  child×parent). The check on hidden similarity ties is deferred to the robustness checks, with the planted worlds.
  Open: one λ for all K or a λ per K (see the reply of 2026-10-27); which similarity measure.
- **child×child similarity measure — DECIDED (owner, 2026-10-27): idf-weighted Jaccard** (Σ idf over shared words ÷ Σ idf
  over all words of the two sets; model names included). idf-weighted cosine kept as the comparison arm. Its strictness
  with a short set contained in a longer one is either accepted or offset by a lower τ, to be judged in the blind reading
  check (pairs between 0.4 and 0.5 shown).
- **Reading check of the similarity measures (2026-10-27, assistant read first, owner's request;
  `corpus_statistics/child_similarity_check.py`, outputs and judgements in `corpus_statistics/child_similarity_check/`).**
  50 random child hyperedges (2+ words), each one's own selection (5 most similar, ≥ 0.5, ties at the 5th kept), the two
  measures in unlabelled random order; judgements written before unblinding (blinding partial: cosine values run higher).
  Per tie: idf-weighted Jaccard 42 ties, same idea 57%, partial 33%, no 10%, 27/50 hyperedges without a tie;
  idf-weighted cosine 197 ties, 30% / 51% / 19%, 4/50 without. 60 pairs with idf-weighted Jaccard 0.4–0.5: same idea
  32%, partial 62%, no 7%. Share of hyperedges with at least one tie of their own (Jaccard, 1,500 sampled): τ 0.5 53%
  (1.2 ties each), 0.4 69% (2.3), 0.3 85% (3.5). Numbers were still in the vocabulary here (the builder removes them).
  One reader; the owner may re-read the files.
- **child×child tie selection — DECIDED (owner, 2026-10-27, after the reading check):** idf-weighted Jaccard, threshold
  τ = 0.4 (replaces 0.5). For each child hyperedge: the 5 most similar hyperedges reaching τ, plus every further one with
  exactly the 5th value (5 is a minimum whenever 5 candidates pass τ; if fewer pass, only those); a tie also counts when
  only the other hyperedge selected it. Numbers excluded from the vocabulary, model names included.
- **Regulariser form — DECIDED (owner, 2026-10-27):** R = λ · (1/N) · Σ_i Σ_j (W_ij / d_i) · ‖p_i − p_j‖², with p the
  weighted membership shares, squared Euclidean distance (Hellinger as a check), d_i = Σ_j W_ij, N = number of entities of
  the facet with at least one tie; W symmetric: a tie counts when either hyperedge selected the other (the "OR" rule, a symmetric k-nearest-neighbour graph; confirmed by the owner, rather than the mutual "AND" rule). Calibration arm for
  child×child: the same measure at τ = 0.5.
- **child×child regulariser, further points (owner, 2026-10-28):** starts after a warm-up of 300 epochs (as the
  concentration floor and the temporal prior), to be re-examined once membership stability over the iterations is
  measured; the concentration floor (λ 0.1, γ 0.8, calibrated on the toy corpus) is recalibrated on the Psychology trial
  first, then λ_child is calibrated with the floor on, with a check that the floor's own measure does not shift; idf across
  time slices decided later (cf. ticket 106). Numbers are not in the vocabulary, so they get no idf. Vocabulary
  composition in the trial (distinct words / occurrences): concept 12,511 / 225,913; modifier 5,804 / 97,550; predicate
  2,167 / 41,616; model names 401 / 26,663; modal, auxiliary, negation and contrast words 25 / 13,524; numbers 2,594 /
  9,171; citation tokens 727 / 865; URL tokens 1 / 1; 898 hyperedges consist only of numbers, citation or URL tokens.
- **Citation and URL tokens (owner, 2026-10-28):** masked for the statistics of the child×child similarity (no idf, not
  counted in a hyperedge's weighted size, cannot create a tie), but kept in the factorisation (hyperedge identity and
  matrices unchanged), where they help interpretation; placeholders (one shared token per kind) were the alternative.
- **Correction (assistant, 2026-10-29): stop-gradient on the column norms does not keep the regulariser away from
  hyperedges with no non-zero similarity in W.** Checked on a 3-hyperedge example (one plain gradient step): with the full
  gradient, a hyperedge C outside W gets a gradient on its own cells, and the gradient of every column is orthogonal to the
  column (Σ value × gradient = 0, because the normalised factor matrix does not change when a column is rescaled); with
  stop-gradient, C's cells are untouched but the tied hyperedges' update changes the column norms, and C's normalised
  values change by the same amount (0.50508 → 0.50477 and 0.62017 → 0.62048 in both cases). Every hyperedge's
  normalised values depend on its column's norm, so a change to any hyperedge's cells moves all others in that column,
  by a factor common to the column, whatever the gradient path. Stop-gradient on the norms is therefore dropped from the
  proposal: the full gradient is used (the community weights stay as computed in the code, without gradient). The same
  reasoning applies to the "leak" measured for the concentration floor (T1g) and to the temporal prior's design claim
  that its gradient "reaches only the persisting entity's own raw row" (CLAUDE.md §4.25): true for the raw cells, not for
  the normalised values the model uses.
- **Building W — DECIDED (owner, 2026-10-29):** for the trial, W is built by brute force (every pair of child hyperedges
  sharing a word is compared). The prefix-filtering algorithm (Chaudhuri et al. 2006; Bayardo et al. 2007; one global
  word order by idf) is run alongside; if both give the identical W (every pair, every value), the faster algorithm is
  used from then on (full corpus).
- **Status of the regularisers (assistant, 2026-10-29, on the owner's question).** article×article is a regulariser in
  the trial, approach 2 (owner, 2026-10-27; density too low for a matrix; damped Jaccard weight), recorded here because
  only ROLLING_NOTES had it. W (child×child) and the article×article tie table are part of the chunk12v3 builder; the
  builder's run is a computation and is started only with the owner's permission. Still open: (1) λ per K — proposed
  2026-10-27 (calibrate λ at every K; one common λ unless the best λ of two K values differ by more than about 2 standard
  errors of the paired seed-by-seed difference; otherwise λ and K chosen on separate held-out sets, Varma & Simon 2006),
  not yet confirmed by the owner; (2) the article×article regulariser's own settings: same form, distance and warm-up as
  child×child?, λ_article's held-out target, and the order of calibration (floor → λ_child → λ_article, or λ_child and
  λ_article on a joint grid); (3) the corpus over which idf is computed for the trial (trial child hyperedges or the
  whole corpus); (4) the trigger for switching from 5 to about 11 ties. Deferred by the owner: idf across time slices,
  re-examining the warm-up of 300, the check of hidden similarity ties (robustness checks). Pending check (needs
  permission, after the builder): the effect of column normalisation on hyperedges with no tie in W.
- **CORRECTION (assistant, 2026-10-29): article×article was not decided as a regulariser.** The assistant recommended
  option (b) on 2026-10-27, the owner did not choose it (the next message spoke of "the article×article matrix"), and the
  assistant then wrote "you chose option (b)" and recorded it in ROLLING_NOTES and in the status point above. Owner
  (2026-10-29): **article×article is a relation matrix in approach 2 (main solution); the regulariser is the fallback**,
  as in the decision of 2026-10-13 (tried if the relation dominates or fragments the article communities). The damped
  Jaccard weight (decided 2026-10-27) gives the matrix cells. Its sparsity in the Psychology trial (2,216 non-zero cells,
  mean degree 1.04, 65% of articles untied) is therefore a point for the weighting of the data matrices (its share in the
  loss) and for the missingness technique.
- **Decisions (owner, 2026-10-29) on the open points above:** (1) **λ per K** approved: λ calibrated at every K; one
  common λ unless the best values differ by more than about 2 standard errors of the paired seed-by-seed difference,
  otherwise λ and K chosen on separate held-out sets. (2) **article×article fallback regulariser**, if used: same form,
  distance and warm-up as child×child; λ_article targeted at the held-out ties of child×article, article×journal and
  article×affiliation; calibration order concentration floor → λ_child → λ_article (λ_child fixed), then one re-check of
  λ_child. (3) **idf is computed over the subcorpus the trial is run on** (the trial's child hyperedges). (4) **k ≈ 11**
  only as an extra calibration arm, run only if λ_child gives no held-out gain at k = 5.

## Addendum 2026-10-29 — hold-out: decisions and open points (owner)

- **Hidden share: 10%** (owner, 2026-10-29), drawn at random. The record before this said: compare 5-fold (20%),
  10-fold (10%) and relation-by-relation (2026-10-17); repeated random 10% splits were proposed as an alternative.
  Drawing is stratified by matrix and, within each matrix, separately for ties and empty cells, so every relation (and
  through them every facet) is represented in proportion. Open: 10 folds (every cell hidden once, 10 fits per setting)
  or about 5 repeated random 10% splits (5 fits; some cells hidden twice or never).
- **Relation-by-relation scheme dropped** (owner, 2026-10-29): it multiplies the fits by the number of relations
  (about 5 × 5 = 25 per setting against 5–10), and it answers a different question (predicting one layer from the
  others, De Bacco et al. 2017).
- **Weights rebuilt per fold from visible cells** (owner, 2026-10-29, accepted because the cost is small): matrix
  scaling, document-frequency weights of the data matrices (to be decided in the weighting step), team-size and
  person-article counts where those cells can be hidden. The hidden cells' true values are taken at the fold's weights.
  W (child×child) is not affected: idf(v) counts child hyperedges containing a word, and hiding cells removes no child
  hyperedge and no word.
- **Scores:** mean squared error on hidden cells chooses K and λ, reported against the trivial prediction 0 and
  separately for hidden ties and hidden empty cells; AUC reported alongside (proposed, awaiting owner).
- **To measure once the builder exists:** the cost of the hidden empty cells (about 21 million in child×article at
  10%); the effect of the per-entity cap (no entity loses more than one third of its ties, so an entity needs 3 ties
  before one can be hidden) on which child hyperedges get hidden ties. Candidate remedy for the first: hide all sampled
  ties plus a sample of empty cells (negative sampling, standard in link prediction).
- **Splits (owner's proposal, 2026-10-29, assistant concurs):** each matrix's cells (ties and empty cells drawn
  separately) are divided at random into 10 groups; 5 of the 10 are used as the hidden sets of 5 fits, so no cell is
  hidden twice. Each fit is a fold of 10-fold cross-validation; the mean over the 5 has the same expectation as the
  10-fold mean, with more variance; the other 5 groups can be added later if the standard errors are too wide.
- **AUC reported (owner, 2026-10-29).** Proposed (assistant): pre-declared as the first criterion inside the band of K
  (or λ) values that MSE cannot tell apart, before inspection; awaiting the owner.
- **Heterogeneous relations (owner's concern, 2026-10-29).** Held-out scores are reported per relation, never pooled
  without weights: hiding article×journal asks "where was it published" (one tie per article), child×article "does
  this article state this idea" (many-to-many), article×article "do these articles share a person". Consequence of the
  cap and the last-tie rule: relations with one tie per entity on one side (article×journal; author×article for the 86%
  of single-article authors) get almost no hidden ties, only hidden empty cells, so their score says little about ties.
  Proposal: K and λ chosen on the combined score weighted as in training, over the relations where ties can be hidden;
  the others reported descriptively. Leak check from multi-relational link prediction (inverse or duplicated relations,
  Toutanova & Chen 2015; Dettmers et al. 2018): no matrix in either approach encodes another matrix's tie
  deterministically; weights are rebuilt per fold.
- **Decisions (owner, 2026-10-29, continued):** article×journal takes no part in the hold-out (no hidden cells of
  either kind); matrices where almost no tie can be hidden get no hidden cells; K and λ are chosen on the combined
  held-out score of the matrices with hidden cells, each weighted as in training (child×article, probably child×parent,
  article×affiliation; author×affiliation in approach 1). All matrices stay fully in the training loss of every fit;
  the restriction concerns only which hidden cells are scored. **AUC is reported as a consultative measure, not a
  decision criterion.**
- **Per-entity cap (owner's reformulation, 2026-10-29):** hidden ties are drawn only from ties whose two entities both
  have at least 3 ties (the eligible population); the 10 groups are drawn so that no entity loses more than one third
  of its ties in any group (an entity with 3 ties loses at most 1). The 10% is a share of eligible ties; empty cells
  are drawn at 10% of the empty cells of the same matrices. Open: one third (eligible from 3 ties) or 30% (eligible
  from 4 ties, since 1 of 3 is 33%).
- **Leak check:** no separate check run; the structural argument above is documented (no duplicated relation, weights
  per fold, W built from word sets only). Noted: a child×article tie can be inferred legitimately from sibling child
  hyperedges of the same parent in the same article; this is prediction from visible data, not a leak.
- **Role of the held-out score (as recorded, 2026-10-17 and 2026-10-27):** for K it defines the band (stability is the
  gate, inspection and the text check choose within the band); for λ_child and λ_article it is the calibration target.
- **Eligibility threshold — DECIDED (owner, 2026-10-29, assistant's proposal):** an entity is eligible from 2 ties;
  per group it loses at most max(1, ⌊n/3⌋) of its n ties and never its last one. Reason: most child hyperedges have 1–2
  articles and are the ones the regulariser is for; from 3 ties they would never be tested. Held-out scores are
  reported by the tie count of the child hyperedge (2, 3–5, 6+), which needs no extra fits and shows whether the 2-tie
  entities behave differently.
- **λ calibration criterion — CONFIRMED (owner delegated, 2026-10-29; assistant's conclusion):** held-out prediction is
  the main criterion for λ_child and λ_article. In-sample loss always prefers λ = 0 and stability tends to rise with
  λ (tied hyperedges forced alike reproduce trivially), so neither can choose λ; held-out prediction measures the
  regulariser's purpose, better placement of sparsely tied entities. Guard checks at the chosen λ: stability not worse
  than at λ = 0, and the text check on W-tied hyperedges. Within one standard error of the best, the larger λ is taken
  (the 1-SE rule as in glmnet).

## Addendum 2026-10-30 — layer 1 weighting: child-level facet (discussion; decided points marked)

Scripts: `plans/ticket107_layer1/` (Psychology trial, 2,139 articles; loader as in `child_similarity_check.py`).

- **Hidden share — DECIDED (owner, 2026-10-30):** 10% stays. The 90.7% of child hyperedges in one article can never
  be tested at any share; 10% of the 86,122 ties of hyperedges with ≥2 articles gives ≈8,600 hidden ties per fold.
- **Loss function — owner, 2026-10-30:** switching from squared error (Frobenius, inherited from v7/v8, never chosen
  against KL/Poisson) to generalised KL is to be discussed once the layer-1 weights are agreed, unless it bears
  directly on them. Assistant's assessment: layer-1 *shapes* barely depend on it (share of child×article loss on
  single-article hyperedges under tf-idf: 0.65 under squared error, 0.60 under a KL-like linear weighting), but every
  *ratio* stated as influence does (value 0.5 = half the influence under KL, a quarter under squared error), as do the
  between-matrix weights of layer 2 and the Frobenius normalisation of ticket 61. **Order DECIDED (owner,
  2026-10-30):** layer 1 → loss → layer 2; until then ratios are stated as influence.
- **child×parent, one matrix or two — owner, 2026-10-30:** prefers one matrix (the child/cousin merge was made to
  reduce the number of relations) unless there is a fundamental objection; this revises the 2026-10-21 two-relation
  decision. Assistant's assessment: no fundamental objection. Consequences of one matrix: one core matrix, so nesting
  and sentence co-occurrence are assumed to follow the same between-community pattern; the cousin value must be
  chosen as an influence (0.5 under squared error = ¼ of a nested tie; 1/√2 ≈ 0.71 = ½); a pair that is both nested
  and co-occurring takes the nested value (1,057 such pairs); the hold-out stratifies by tie kind. Counts: 120,666
  nested ties, 105,910 cousin ties; cousin share of the matrix's squared mass 0.18 at value 0.5, 0.30 at 0.71.
  **DECIDED (owner, 2026-10-30):** one child×parent matrix with the details above; a cousin tie gets half the
  influence of a nested tie, i.e. value 1/√2 under the current squared-error loss (0.5 if the loss becomes KL).
- **Frequency weighting of child×article — DECIDED (owner, 2026-10-30):** classic log(1+tf)·idf, idf over the trial
  subcorpus (idf 7.97 for a single-article hyperedge, 1.63 for "use"). Single-article hyperedges reach other
  articles through W.
- **Article length (4B) — measured:** the number of distinct child hyperedges of an article is
  uncorrelated with its total sentence count (Spearman −0.005; naive split of the pure-text files), so it measures
  how much of the article is about the focal terms, not its length. Reviews (MEDLINE tags; 318 of 2,125 typed) are
  longer (OR 1.59 per doubling of sentences) but *less* likely among articles with many hyperedges (OR 0.90 per
  doubling; systematic reviews 0.81, narrative reviews 0.96 n.s.). Articles with more hyperedges are somewhat more
  cited: Spearman 0.16 within the subset, 0.18 within the corpus (reference lists of 39,299 corpus articles,
  `citation_dictionary_v2.jsonl`, matched by PMCID/PMID/DOI), 0.16 for the field- and year-normalised OpenAlex
  percentile (1,414 articles; the 715 newer articles have no OpenAlex metrics yet). Top 5% band of research articles:
  0.99 within-corpus citations on average, 35% cited at all; bottom quarter 0.18 and 11%. Caveat: the corpus is
  recent (2025–26 dominate), so incoming citations are sparse (365 of 2,139 cited within the corpus).
- **Article length — DECIDED (owner, 2026-10-30, conditional on the measure being established and pertinent;
  assistant's assessment below):** partial normalisation of each article's row of child×article, so that an
  article's influence (its row's total squared weight) grows with the square root of its unnormalised influence:
  row ← row / ‖row‖^½ (after tf-idf; the whole-matrix Frobenius normalisation of ticket 61 follows and keeps the
  ratios). Effect: the 90th-percentile article counts ≈3× the median instead of ≈10×, the longest ≈7× instead of
  ≈45×. Established: partial length normalisation is standard in information retrieval (pivoted normalisation,
  Singhal, Buckley & Mitra 1996; the length parameter b of BM25, Robertson et al. 1994), and 1/√length is the
  classic Lucene length norm; full L2 normalisation is the usual choice in NMF on text. The exponent ½ is a
  convention, not derived for our data. Pertinent: the measured facts above (focal-heavy articles are not reviews
  and are somewhat more cited) argue against full normalisation, the 45× range against none.

## Addendum 2026-10-30 (2) — layer 1 weighting: social matrices (assistant's proposals, owner to decide)

Scripts: `plans/ticket107_layer1/social_stats.py`, `org_topic.py`. Influence = a row's or column's total squared
weight (current squared-error loss). Topic agreement = share of article pairs with the same OpenAlex primary topic
(random pairs 0.083).

Facts (Psychology trial): 11,157 kept bylines; team size median 4, p90 9, p99 19, max 69; team size is unrelated to
citations (Spearman −0.01 within the subset, −0.03 within the corpus, 0.02 normalised percentile). 10,072 persons,
92% on one trial article, max 17; pairs of articles sharing a person agree on topic 0.53, the same for persons with
2–5 and 6–15 articles. 2,519 organisations; articles per organisation median 1, max 56; topic agreement 0.17 (2–5
articles), 0.15 (6–15), 0.12 (16+; 34 organisations holding 10,055 of 18,927 organisation-linked pairs). 384
journals; the top 10 hold 41% of articles; topic agreement 0.14 in the top 10, 0.24 in journals with 2–10 articles.

- **P1 author×article (approach 1):** the current log2(1+n)/n per tie was designed for linear mass; under squared
  error it gives an article an influence of 1.0 (solo), 1.34 (n = 5), 0.64 (n = 50). Proposal: 1/√n per tie, so
  every article has influence 1 whatever its team (fractional counting expressed as influence; Newman 2001;
  Perianes-Rodriguez, Waltman & van Eck 2016). Persons not damped (topic agreement does not fall with productivity).
- **P2 author×affiliation (approach 1):** same per-person shares as now (bylines naming the organisation), but
  normalised to influence 1 per person (L2 instead of L1: two equal organisations 0.71 each instead of 0.5). Persons with no organisation: missing row (decided 2026-10-26).
- **P3 article×affiliation (approach 2):** share of the article's affiliated bylines from each organisation (a byline
  with k organisations gives 1/k to each; fractional institution counting, Waltman & van Eck 2015), influence 1 per
  article. Articles with no organisation: missing row.
- **P4 article×journal:** binary. Articles without a journal ID (4): missing row.
- **P5 article×article (approach 2):** weight as decided (2026-10-29); the diagonal is masked (self-ties are
  undefined); articles with no tie keep observed zeros.
*[2026-10-30, owner, on the social proposals:*
- *P1: the current log2(1+n)/n author×article weight is **rejected** (owner: "not acceptable", because articles with
  3–5 authors get the highest influence: 1.33–1.34 against 1.0 solo and 0.64 at n = 50). Replacement pending. Owner
  asked for consistency with the team-size correction used elsewhere: article×article divides by the size of the
  union of the two author sets and uses 1/(n_p − 1) per shared person (Newman 2001), i.e. linear fractional
  counting. Linear 1/n per author×article tie gives an article influence 1/n under squared error (a 20-author
  article 0.05); 1/√n is the same principle (each article one vote) under squared error, and becomes 1/n if the loss
  becomes KL. Flagged: the article×article weight was decided as a value, not as an influence; to be revisited with
  the loss.*
- *P5: **DECIDED** — the article×article diagonal is masked (missing, not zero); the 1,398 articles without ties keep
  observed zeros.*
- *P4.3, **deferred** (owner): entities whose communities lie mostly outside the corpus or the subset (a journal,
  organisation or person with a few corpus articles among a large off-corpus output) get memberships decided by the
  little that is observed (the boundary specification problem, Laumann, Marsden & Prensky 1983). Options to discuss:
  state memberships as relative to the corpus; report each entity's coverage (share of its total OpenAlex output
  inside the corpus) and flag low-coverage entities in interpretation; external side information (journal subject
  categories, total output) as a prior or regulariser; judge such entities on the full corpus rather than the trial
  subset.]*
*[2026-10-30, owner, social weights continued:*
- *P1 **DECIDED** (owner): weights must change monotonically with team size; an article's influence is independent of
  its number of authors. Under squared error: 1/√n per author×article tie (influence 1 per article; each author gets
  1/n of it); 1/n if the loss becomes KL. Owner asked whether persons can still be down-weighted with 1/(n_p − 1):
  in article×article n_p is the person's number of trial articles, not the team size; the within-article
  down-weighting of authors of large teams is already in 1/√n; a productivity weight on persons would be compatible
  only as a redistribution inside each article's row (renormalised to influence 1) — not proposed (topic agreement
  of articles sharing a person does not fall with productivity).*
- *P2 **DECIDED** (owner): each person's author×affiliation row keeps its proportions and is scaled to influence 1
  (L2 row normalisation). No inflation: every value is ≤ 1 after the row scaling, and the whole matrix is normalised
  to Frobenius norm 1 afterwards (ticket 61); its weight against other matrices is set in layer 2.]*
*[2026-10-30, **P4 DECIDED (owner): no frequency down-weighting of journals or organisations** — journal ties binary,
organisation ties as in P2/P3. Owner's reasoning: large, popular journals can be expected to shape communities around
them through selection (authors submit selectively, journals accept selectively), so their greater influence is a
true signal; down-weighting by frequency would also make journals with few subset articles the most influential.
Smoothing (e.g. a logarithm of journal size) may be considered later; for now ties stay as they are.]*
*[2026-10-30, **P2 and P3 DECIDED (owner): a byline naming several organisations splits its vote equally (1/k to
each)**, in author×affiliation (per person) and article×affiliation (per article). Owner's reasoning: we cannot tell
which organisation's resources (equipment, social capital) the author drew on most for a given paper. Then each row
is scaled so that the squares of its values sum to 1 (before the whole-matrix normalisation of ticket 61). Source of
multiple organisations: the article's own affiliation lines as resolved by OpenAlex per authorship (one byline, several
addresses); Psychology trial after the organisation merge: 0 organisations 5.7% of bylines, 1 73.8%, 2 15.2%, 3+ 5.3%.]*

## Addendum 2026-10-31 — loss function discussion (open; nothing decided)

- **Candidates:** squared error (β = 2, current), generalised KL (β = 1), β-divergence in between (Févotte & Idier
  2011). Both are maximum likelihood in disguise: squared error = Gaussian noise of equal spread, KL = Poisson noise.
- **Owner (2026-10-31):** compare β = 1 and β = 2 on the M1 trial, but not by their final losses (different scales).
  Loss-neutral criteria proposed by the assistant: agreement of the two solutions (if memberships agree, the choice
  does not matter); held-out ties scored by a scale-free ranking measure and cross-scored under both losses; seed
  stability; recovery of planted communities generated under both noise types; the text check; optionally held-out
  Tweedie likelihood with β as a parameter (Şimşekli, Cemgil & Yılmaz 2013).
- **Off-corpus / peripheral entities (owner, 2026-10-31; links to the deferred P4.3):** treated as a methodological
  problem, not only a computational one: the model sees only the corpus, so entities whose communities lie mostly
  outside it cannot be reconstructed well, and some misfit is inevitable. Literature and approaches, and the link to
  the loss, in the assistant's reply of 2026-10-31; to be continued.
- **2026-10-31, explained to the owner (no decision):** (i) a *background component*: each cell is reconstructed as
  community part + g_i·q_j, with g_i ≥ 0 a free background loading per entity and q_j a fixed profile (the other
  entity's share of all ties), so entities whose ties scatter across communities are explained as "generic" instead
  of distorting a community; its share of an entity's reconstruction is itself a diagnostic. Precedents: noise
  component in model-based clustering (Banfield & Raftery 1993), degree-corrected null term (Karrer & Newman 2011),
  background topics, affine NMF. A model change, to be tested on planted data. (ii) *per-entity fit and coverage*:
  share of each entity's ties the model reconstructs (per relation and combined), against its coverage (share of
  its output inside the subset / corpus; for articles, the share of sentences with a focal extraction); diagnostic
  only, not used in training; needs OpenAlex counts for journals, organisations and persons.
- **DECIDED (owner, 2026-10-31):** per-entity fit and coverage reporting is implemented as a post-fit diagnostic; it is
  also to serve as an additional criterion when comparing K values and approach 1 vs approach 2 (and β = 1 vs 2). The
  background component is kept as a deferred robustness check (a later test arm, planted test first). Owner's caveat:
  it does not model unobserved or partially observed communities, so it does only part of that job.

## Addendum 2026-10-31 (2) — remaining M1 points

- **Model-name level — measured** (`plans/ticket107_layer1/model_levels.py`, `model_pairs.py`; numbers left out of
  identity). Child hyperedges: tier level (G3) 97,931, version level 97,878, family level 97,349 (−0.6%); share in one
  article 0.906 / 0.906 / 0.905; child×article density 8.36e-4 / 8.37e-4 / 8.39e-4; focal atoms 401 / 366 / 279;
  hyperedges with a named model 11,866 / 11,813 / 11,284, 95% in one article at every level. Merging hardly changes
  sparsity. Pairs of hyperedges identical except for the model: same version, different tier 49; same family,
  different version 1,077; bare family vs versioned 287; different families 23,089. **Owner's proposal (2026-10-31):**
  keep the atoms apart and give related model names partial similarity in W (e.g. 0.9 same family different version,
  0.66 at family level; numbers not insisted on). Assistant: established as soft matching (soft cardinality, Jimenez,
  Gonzalez & Gelbukh 2010; soft cosine, Sidorov et al. 2014); ladder of values to be agreed.
- **child×parent frequency weighting — DECIDED (owner, 2026-10-31):** child side × idf counted over parents, with the
  formula of chunk12/chunk12v2, ln(N/(df+1)) + 1 (N = parents in the trial, df = parents containing the child
  hyperedge); ties otherwise binary. This revises ticket 84 D8 (child×parent left binary) for M1.
- **M1 for squared error first — DECIDED (owner, 2026-10-31):** the builder writes the squared-error formulas; the KL
  version of M1 is built later, for the β comparison, once the Optuna and penalty mechanisms are settled.
- **Unknown affiliations:** the 2026-10-26 decision (missing cells) stands unless the owner chooses placeholders;
  assistant's recommendation of 2026-10-31: missing cells (placeholders would be the missing-indicator method; the
  no-ID bylines are mostly companies and local services on one article, so placeholders would link almost nothing
  while taking row weight from known organisations).
- **Soft model-name similarity in W — owner's ladder (2026-10-31):** same family, different tier 0.95; same family,
  different version 0.8; bare family vs versioned 0.66; `chatgpt` vs other GPT names 0.66 or 0.5 (assistant to choose
  from the data); domain-adapted vs base model about 0.5 (to check against τ). Measured (`soft_tau.py`, pairs identical
  except for one model name; share reaching τ 0.4 / 0.5): tier 0.95 → 1.00 / 0.98 (49 pairs); version 0.8 → 0.98 / 0.87
  (875); bare vs versioned 0.66 → 0.80 / 0.71 (238); chatgpt 0.5 → 0.62 / 0.44, 0.66 → 0.70 / 0.58 (650); domain-adapted
  0.5 → 0.24 / 0.11 (699). A pair of hyperedges made of the model names alone has similarity s/(2 − s) (equal idf):
  0.95 → 0.90, 0.8 → 0.67, 0.66 → 0.49, 0.5 → 0.33, so at τ = 0.4 a value of 0.5 links two names only when other words
  are shared. Assistant: chatgpt 0.66 (the same relation as a bare family name: the product, version unspecified);
  domain-adapted 0.5 kept (a tie needs support from shared words); the domain-adapted list must be hand-curated
  (substring rules catch wrong names). **Data issue found:** 31 model atoms with a glued citation number (`bert28`,
  `roberta34`; 43 mentions, 30 articles); proposed builder fix: family + 2–3 digits that is not a known version →
  the family.
- **Unknown affiliations — DECIDED (owner, 2026-10-31):** missing cells (confirms 2026-10-26); no placeholders.
- **child×parent idf — reopened (owner, 2026-10-31).** Measured (`cp_idf.py`): 36,016 parents, 225,108 ties; a
  (child, parent) pair recurs in 0.6% of ties (so tf-idf and idf are the same up to a constant); most frequent children
  `be` 5,693 parents, `llm` 5,130, `use` 3,344, `chatgpt` 1,619; squared mass of children in 51+ parents 0.34 binary vs
  0.13 with idf; children in one parent 0.39 vs 0.58; top 20 children 0.136 vs 0.026.
- **`chatgpt` vs other GPT names — DECIDED (owner, 2026-10-31): 0.5**, because unlike other model names ChatGPT is also
  used in a general sense. Ladder complete: tier 0.95, version 0.8, bare family vs versioned 0.66, chatgpt 0.5,
  domain-adapted 0.5 (hand-curated list), different families 0.
- **child×parent idf, continued (owner's questions, 2026-10-31; `one_word.py`).** (1) Exempting one-word child
  hyperedges: they are 6.8% of the children in child×parent but hold 45% of its ties, and 84% of the children found in
  51+ parents are one word (`be`, `llm`, `use`, `chatgpt`, `show`…), so an exemption would undo most of idf's effect;
  rare one-word children (`empathy`, `anhedonia`) already keep a high idf. Assistant: no exemption. (2) Frequency
  basis: parent frequency and article frequency of a child agree largely (Spearman 0.78; 459 of the 500 most
  down-weighted children are the same), but differ for children used in many claims of one article (`gpt_5_mini` 37
  claims in 1 article, `cbs llm` 35 in 1): parent frequency treats such within-article repetition as genericness.
  Assistant: count article frequency (articles of the trial containing the child) for both child×article and
  child×parent, one idf per child hyperedge; within-article repetition stays in the number of ties, as centrality.
  Owner to decide. Note: `REFnnnnnn` one-word children are cited works, kept by the decision of 2026-10-28.
- **child×parent idf basis, checked (`idf_basis.py`):** switching from parent frequency to article frequency moves
  4.8% of the matrix's squared mass between children; the 286 children repeated within articles (5+ claims, 3+ claims
  per article) hold 2.4% (parent basis) vs 2.7% (article basis); children in 100+ articles 8.0% vs 5.5%. The
  conceptual case for the article basis (claims of one article are not independent evidence of genericness; one idf
  per child for both matrices) is real but not decisive, and parent frequency is the standard construction (df over
  the columns of the matrix being weighted). Assistant withdraws the article-basis proposal: **parent-frequency idf**
  recommended. **DECIDED (owner, 2026-10-31):** no exemption of one-word children from idf.
- **child×parent idf — DECIDED (owner, 2026-10-31): parent frequency**, ln(N/(df+1)) + 1, N = trial parents, df =
  parents containing the child; value of a tie = idf (nested) or idf/√2 (cousin); a recurring pair counts once.
  Owner asked whether idf must be adjusted to squared error or to matrix normalisation. Assistant: no adjustment.
  The whole-matrix normalisation divides every cell by one constant and leaves the ratios unchanged. Under squared
  error a tie's influence is idf² (the generic/rare ratio 3.8 in value becomes about 14 in influence); this is the
  standard tf-idf practice with squared-error NMF and the same as child×article, so both matrices keep idf on the
  values. For the later KL version, idf on the values gives influence ∝ idf, a milder down-weighting; to be stated
  when the β comparison is set up (part of what it compares).
- **CORRECTION (assistant, 2026-10-31): "a recurring pair counts once" was the assistant's own proposal, not an owner
  decision, and it was miscounted.** Recount (`pair_repeat.py`): of 225,108 child–parent pairs, 3,409 recur: 1,012 are
  a child both nested in and co-occurring with the same parent (already decided: nested value, once); 1,259 come from
  a claim repeated in several sentences of one article; 1,089 from a claim (same parent word set) made in several
  articles (326 parents); 49 from a child written twice inside one parent. Genuine repetitions: about 2,400 pairs
  (1.1%), not 0.6%. The last case is covered by the owner's decision of 2026-10-19 (a repeated hyperedge counts twice,
  as the text was written), which "counts once" would have contradicted. Options to decide: binary, log(1 + tf)·idf as
  in child×article, or raw tf·idf.
