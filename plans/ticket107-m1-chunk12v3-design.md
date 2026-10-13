# Ticket 107 — the new matrix builder (M1, "chunk12v3"): representation decisions

> **STATUS (2026-10-13): DESIGN IN PROGRESS. Sections 1 and 2 are owner decisions; section 3
> records observations about the temporal extension (not decisions); section 5 lists what is
> still open. Nothing is built.** Decisions were reached in discussion with the owner on
> 2026-10-12/13 (the owner also consulted colleagues on the semantic side). Later decisions are
> added with dates; this text stays.

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
full-text focal sentences. **Slices: open.** Proposed 2019 (T1) / 2020 (T2); the working corpus
has only 17 / 91 articles in those years (2021: 385, 2022: 744, 2023: 1,914, 2024: 4,956), so
alternatives were put to the owner: (i) T1 2019-2022 (1,237) vs T2 2023 (1,914), (ii) T1 2022 (744)
vs T2 2023 (1,914) — both a pre/post-ChatGPT break, a genuine emergence case.

## 5. Open — to be decided

1. Slices (section 4) and which date defines a paper's year (PubMed print or electronic date, or
   OpenAlex year).
2. How the configurations are compared, fixed before results (ticket 91 precedent). They
   reconstruct different matrices, so reconstruction loss and hypervolume cannot rank them (ticket
   93). Candidates: chance-adjusted seed stability per shared facet; held-out ties on shared
   relations; a text check; temporal matching quality.
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
