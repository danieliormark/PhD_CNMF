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
