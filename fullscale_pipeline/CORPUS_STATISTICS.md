# Full-scale corpus statistics and author identification (2026-10-02)

Descriptive statistics of the current full-scale corpus, an assessment of how authors can be
identified, and a check of what is lost where they cannot. Read-only analysis: no pipeline output was
changed. Scripts: `fullscale_pipeline/corpus_statistics/` (tracked). Data they read and write:
`fullscale_pipeline/diagnostics/openalex_authors/` (local, not in git, like the other diagnostics
caches). The OpenAlex key is read from the `OPENALEX_API_KEY` environment variable and is not stored
in any file.

*[2026-10-10: §1-§7 below describe the full 34,662-article G2 corpus, before the G3 exclusions
(G3_POSTPROCESSING.md §6 item 6) existed, and use the publisher's self-declared "Subjects:" field
and OpenAlex. §8 is a separate, narrower, differently-sourced analysis: the 28,075 articles G3 marks
`valid` (G3's actual scope for M1), using only MEDLINE's own PublicationType tags, never the
publisher's self-declared field or OpenAlex. Do not mix the two: §1-§7's article-type row (a) and
§8's primary type are not the same measurement over the same corpus.]*
*[2026-10-10: §9 (author coverage) uses the same narrowed corpus as §8, after the retraction filter:
the 28,029-article working corpus.]*

## 1. Inputs

| Input | Stage | What it contributes |
|---|---|---|
| `LCS/focal_sentences_v2.jsonl` | P7 v2 (RL-069) | the 34,662 articles with focal sentences, every focal sentence (702,948) and the focal terms in it |
| `LCS/clean_corpus_v2/` | P1c | text of all 39,299 whitelisted articles, 47,619 region files |
| `LCS/pure_text_corpus/` headers | R2 | PMID, DOI, `Subjects:` field (article type), ORCID-tagged author lines |
| `phase5_graphbrain/g2_v3/g2_parsed_v3.jsonl` | G2 v3 | hyperedges ("units") per focal sentence |
| OpenAlex API (works, authors) | external | author IDs, citation metrics, h-indices |

`clean_corpus_v2` holds one file per *region window* of an article: `<PMCID>.txt` is region 1,
`<PMCID>.r2.txt` etc. are further regions (P2's own docstring). 31,460 articles have one region file,
7,366 two, 468 three, 5 four or five. All counts below sum every region of an article.

Whole whitelist (39,299 articles): **230.8M words, 10.2M sentences**. The 4,637 articles without a
focal sentence account for 22.9M of those words.

## 2. Summary table

Each row counts only articles with **at least one focal sentence meeting the row's criterion**. "Any
focal item" means any term of `focal_terms.py` accepted by P2 (including the 18,582 sentences that are
focal only after coreference resolution; no article qualifies through those alone).

| | Articles | Words | Sentences | Focal sentences | Hyperedges | Author bylines | Distinct authors (OpenAlex ID) | Distinct ORCIDs |
|---|---|---|---|---|---|---|---|---|
| All | 34,662 | 207.9M | 9.20M | 702,948 | 710,869 | 215,961 | 161,807 | 55,490 |
| (a) article/review only | 29,606 | 184.5M | 8.14M | 606,986 | 613,419 | 186,338 | 142,652 | 49,191 |
| (b) LLM + BERT + top-20 | 34,592 | 207.4M | 9.18M | 690,499 | 698,378 | 215,542 | 161,523 | 55,427 |
| (a) and (b) | 29,547 | 184.2M | 8.12M | 596,617 | 603,015 | 185,988 | 142,417 | 49,135 |

Columns:
- **Words, Sentences**: the whole cleaned text of the row's articles (whitespace words; a simple
  regex sentence split, not P2's segmentation).
- **Focal sentences, Hyperedges**: only sentences meeting the row's criterion, and the G2 v3 units
  parsed from them. In (b) a sentence that mentions only a model outside the set is not counted.
- **Author bylines**: OpenAlex author lists (see §4). **Distinct authors**: distinct OpenAlex author
  IDs; a lower bound, because 6.8% of bylines have no ID (§4).
- Every article in the table is G2-parsed (G2 v3's input is the same file), so a "G2-parsed" column
  is always 100% and is omitted.

Row definitions:
- **(a)** uses the publisher's `Subjects:` field in the PMC header, which is free text ("Article",
  "Research Article", "Review", "Letter to the Editor", ...). Classified by substring:
  `excluded_format` (letter, correction, erratum, retraction, comment, editorial, news,
  announcement) first, then `review`, then `article_like` (article, research, original, study,
  report, investigation), else `other` (viewpoints, perspectives, protocols, data descriptors, ...).
  (a) = article_like + review. Focal articles: article_like 25,083, review 4,523, other 4,453,
  excluded 603.
- **(b)** = sentences mentioning the generic LLM terms, bare BERT, or one of the 20 most frequent
  model families (by focal-sentence count, below). Only 70 focal articles mention nothing but models
  outside this set, so (b) barely reduces the corpus.

## 3. Model families and the top 20

Families group versions of one model or vendor line (ChatGPT, GPT-3/3.5/4/4o/4.5/5, InstructGPT,
GPT-4V → GPT; Llama 2/3, LLaMA → Llama; Mistral, Mixtral → Mistral; Claude variants → Claude), but
keep distinct named models apart even when they share an architecture name (BioBERT, SciBERT,
DNABERT, ... are separate families). Bare or generic terms (LLM/LLMs/large language model(s), BERT,
GPT, "language model", "Transformer model") are their own buckets. Grouping: `family_map.py`; some
calls are judgement (Gemini, Bard and PaLM kept as three Google families). 196 canonical terms in
`focal_terms.py` (208 lines in `focal_words.txt`, which also lists spelling variants) give 165
families; 147 distinct families occur in the corpus's focal sentences. P2 records 231 distinct
term strings, including spelling variants its rules accept (`llama-3.1`, `gpt 3.5`, `gpt4v`, ...);
`family_of_term()` in `family_map.py` maps them (GPT and Llama version spellings by pattern, other
names case-insensitively).

Top 20 families by number of focal sentences (P2 terms):

| # | Family | Focal sentences | # | Family | Focal sentences |
|---|---|---|---|---|---|
| 1 | LLM (generic term) | 306,714 | 11 | RoBERTa | 7,472 |
| 2 | GPT (OpenAI) | 237,262 | 12 | Bard (Google) | 5,990 |
| 3 | BERT (bare) | 53,378 | 13 | BioBERT | 5,117 |
| 4 | "language model" (generic) | 35,057 | 14 | Mistral | 3,901 |
| 5 | Gemini (Google) | 29,677 | 15 | Qwen | 3,739 |
| 6 | Transformer (generic) | 29,299 | 16 | Grok (xAI) | 3,513 |
| 7 | GPT (bare) | 24,670 | 17 | ESM-2 | 3,490 |
| 8 | DeepSeek | 20,039 | 18 | DNABERT | 1,900 |
| 9 | Llama (Meta) | 16,250 | 19 | Copilot (Microsoft/GitHub) | 1,776 |
| 10 | Claude (Anthropic) | 15,750 | 20 | Gemma (Google) | 1,731 |

Next: PaLM (Google) 1,450, PubMedBERT 1,444.

## 4. Author identification

**PMC and PubMed/MEDLINE assign no author identifiers.** The only identifier they carry is the
author's ORCID, when the publisher supplied it: 67,067 of 262,692 author lines in the PMC headers
(25.5%), 55,490 distinct ORCIDs. ORCID alone therefore cannot count or link authors.

**OpenAlex** gives every author on a work an algorithmically disambiguated author ID. Matching, by
identifier only (never by title or name):
1. PMID from the article's own PMC header (34,620 of 34,662 have one), 100 per query, exact match.
2. Every match checked against the DOI in the PMC header. No PMID returned more than one record.
3. DOI lookup for articles without a PMID match and to settle conflicts.

| Outcome | Articles |
|---|---|
| PMID match, DOI agrees | 34,374 (99.2%) |
| PMID match, header has no DOI | 15 |
| Found by DOI only (OpenAlex record lacks the PMID, or no PMID) | 163 |
| PMID record and DOI record differ: same paper in another version (arXiv preprint, later F1000 revision); DOI record kept, titles agree 100% | 73 |
| PMID record's DOI differs, no DOI record: titles agree exactly, kept | 12 |
| Not found | 25 |

**Header author lines are not a reliable byline count.** The PMC header parser also counts
letter-labelled affiliation lines, "Corresponding author" notes and "Academic Editor" lines; OpenAlex
has fewer authors than header lines in 57.8% of articles (median −1), and on inspection OpenAlex was
right in every case checked (e.g. 34 authors vs 71 header lines). Byline counts use OpenAlex.

**Coverage per article:**

| | Articles | In OpenAlex | All authors have an ID | ≥1 author has an ID | Authors listed, none with an ID |
|---|---|---|---|---|---|
| All | 34,662 | 34,637 | 25,502 (73.6%) | 34,382 (99.2%) | 255 |
| (a) | 29,606 | 29,586 | 21,708 (73.3%) | 29,375 (99.2%) | 211 |
| (b) | 34,592 | 34,567 | 25,452 (73.6%) | 34,312 (99.2%) | 255 |
| (a) and (b) | 29,547 | 29,527 | 21,667 (73.3%) | 29,316 (99.2%) | 211 |

The 14,785 bylines without an ID (6.8%) are spread over 9,135 articles; in 6,365 of them only one
author lacks an ID. They are ordinary personal names, not consortia (65 look like group names):
OpenAlex lists the person on the paper but has not attached an author profile. Distinct authors are
therefore between 161,807 and at most about 176,600 (if every ID-less byline were a different person).

## 5. Are the unidentified authors important?

Influence measured with OpenAlex's field- and year-normalised citation percentile and FWCI (1.0 =
world average for the field and year), compared within team-size groups because larger teams are
both more likely to have an unidentified author and more cited.

| Team size | Affected papers | top 10% | top 1% | median FWCI | Fully identified | top 10% | top 1% | median FWCI |
|---|---|---|---|---|---|---|---|---|
| All | 9,135 | 41.6% | 8.4% | 1.82 | 25,502 | 42.7% | 9.5% | 1.84 |
| 1 | 86 | 35% | 5.9% | 1.26 | 1,320 | 41% | 9.4% | 1.54 |
| 2–3 | 1,174 | 38% | 7.5% | 1.38 | 6,952 | 43% | 10.0% | 1.81 |
| 4–6 | 3,274 | 39% | 6.7% | 1.69 | 10,014 | 42% | 8.6% | 1.81 |
| 7–10 | 2,715 | 41% | 7.9% | 1.80 | 5,230 | 42% | 9.0% | 1.89 |
| 11+ | 1,886 | 51% | 13.0% | 2.59 | 1,986 | 50% | 13.6% | 2.42 |

- **No systematic loss of influential work.** Overall identical (median percentile 0.861 in both);
  among small teams affected papers are slightly *less* cited, and the highest co-author h-index on
  the team is lower (median 13 vs 20 for 2–3 authors).
- **Position:** unidentified bylines are 16.4% first, 73.3% middle, 10.3% last authors, against
  16.0 / 68.6 / 15.4% of all bylines; senior (last) authors are identified more reliably. 3,615 papers
  (10.4%) lack an ID for their first or last author.
- **Partly recoverable:** 29.6% of unidentified names occur elsewhere in the corpus with an ID; 15.1%
  match exactly one ID (first + last name), so could be linked by name (an assumption, not proof).
  Those matched people are typical researchers (median h-index 9 vs 8 for all identified authors).
- **Specific prominent losses.** The most-cited affected papers include AlphaFold ("Highly accurate
  protein structure prediction", PMC8371605: 2 of 34 authors), Med-PaLM (PMC10396962: 3 of 32),
  Med-PaLM 2 (PMC11922739: first author unidentified), a single-cell multi-omics review
  (PMC10242609: Rahul Satija) and the Molecular Transformer (PMC6764164: all 7 authors unidentified,
  so the paper has no author links at all).
- **Strong year pattern:** share of papers with an unidentified author 2023 17%, 2024 24%, **2025
  41%**, 2026 16%. This tracks OpenAlex's processing of recent records, not the papers, and would
  make a 2025 slice lose far more co-authorship links than its neighbours. Field differences are
  small (19–29%).

Recommendations, not yet decided: keep unidentified authors as paper-specific placeholder nodes
rather than dropping them; link them by name where the name maps to exactly one ID in the corpus;
re-query OpenAlex for 2025 papers later.

## 6. Corrections to figures given earlier in conversation (2026-10-02)

An earlier in-session table (never written to the repository) reported 39,299 articles, 187.7M
words, 8.34M sentences, 300,465 author bylines and 28,861 articles for (b). Three errors:
1. It treated `.rN.txt` region files as revisions and kept only the highest-numbered one, so for
   about 7,800 articles words, sentences and term hits came from a fragment (often a 20-word ethics
   statement). Corrected totals: 230.8M words and 10.2M sentences for the whitelist.
2. Its byline count came from the PMC header parser, about 22% too high (§4).
3. Row (b) and its top-20 list came from that same pass; the P2-based counts above supersede them
   (GPT-4V had also been counted as its own family instead of within GPT).
4. A first version of this document's own family mapping missed 53 P2 spellings (e.g. `llama-3.1`,
   `gpt 3.5`), undercounting Llama by about 2,700 focal sentences and leaving about 2,100 Llama- or
   GPT-variant-only sentences out of row (b). Fixed before commit with `family_of_term()`; article
   counts were unaffected.

## 7. Reproduction

From `fullscale_pipeline/corpus_statistics/`, in `tensor_env`, in this order:
`headers.py` (PMC headers), `wordcount.py` (all-region word and sentence counts),
`OPENALEX_API_KEY=... python3 verify.py` (OpenAlex matching, ~350 requests),
`table.py` (§2–§4), `coverage.py` (coverage table), `OPENALEX_API_KEY=... python3 influence_fetch.py`
(~2,000 requests), `influence_analysis.py` (§5). `fullscale_analysis.py` is the earlier pass, kept for
`parse_header`, `classify_subject` and the `Subjects:` values it recorded; its word and term counts
are superseded (note at its top). OpenAlex results change as OpenAlex updates its records; the cached
responses used here are in `diagnostics/openalex_authors/`.

## 8. Article types of the 28,075-article valid G3/M1 corpus (2026-10-10)

**Scope, deliberately narrower than §1-§7: only the 28,075 articles G3 marks `status: "valid"`**
(`PG/g3_v1/g3_articles_v1.jsonl`; 34,662 G2-parsed articles minus 6,466 excluded as focal-term false
positives, G3_POSTPROCESSING.md §6 item 6, minus 121 further with no usable G3 parent, §6 item 8
step 9). This is the corpus M1 will actually build relation matrices from, not the full G2 corpus
§1-§7 describe. The owner asked for this before moving to author counts, so that the corpus itself
is fixed first: nothing outside this 28,075-article set, and no later step touching authors, is
part of this entry.

**Source, deliberately narrow: MEDLINE's own `PublicationType` tags, from NCBI Entrez, for the
article's own PMID — not OpenAlex, and not the publisher's self-declared JATS "Subjects:" line
that §1-§7's row (a) and the older `diagnostics/paper_type_table.py` use.** The PMID comes from the
PMC header of the article itself (`headers.json`, R2 stage, already fetched for §4 — PMC/Entrez
metadata, not OpenAlex). `PublicationType` is MEDLINE's own curated, controlled-vocabulary tag list
per PMID (`corpus_statistics/pubmed_pubtype_fetch.py`, NCBI `efetch`, cached in
`diagnostics/openalex_authors/pubmed_pubtypes.json`; already fetched for all 34,620 PMIDs of the
full G2 corpus when this was first run 2026-10-06, so the 28,075-article subset needed no new
fetch). An article can carry several tags (most carry one, up to six seen).

**Where PMC/MEDLINE/Entrez/PubMed lack the information — reported, not filled in from elsewhere:**
**38 of the 28,075 valid articles have no PMID in their PMC header at all**, so PMC/Entrez gives no
link to a MEDLINE record and no type can be read for them from this source. Of the remaining 28,037
(all with a PMID), every one was found by Entrez `efetch` and carries at least one `PublicationType`
tag — no further gaps.

**Raw MEDLINE `PublicationType` tags** (multi-label; 39 distinct tags seen; counts are articles, not
tag instances — an article with several tags is counted once per tag it carries):

| PublicationType (MEDLINE) | n articles | PublicationType (MEDLINE) | n articles |
|---|---|---|---|
| Journal Article | 27,671 | Dataset | 29 |
| Review | 3,807 | Historical Article | 25 |
| Research Support, Non-U.S. Gov't | 2,549 | Research Support, N.I.H., Intramural | 18 |
| Comparative Study | 894 | Conference Proceedings | 12 |
| Research Support, N.I.H., Extramural | 633 | Comment | 12 |
| Systematic Review | 583 | English Abstract | 11 |
| Scoping Review | 324 | Network Meta-Analysis | 8 |
| Validation Study | 225 | Introductory Journal Article | 5 |
| Editorial | 221 | Clinical Trial, Phase II | 4 |
| Observational Study | 204 | Clinical Trial | 4 |
| Research Support, U.S. Gov't, Non-P.H.S. | 166 | Pragmatic Clinical Trial | 4 |
| Randomized Controlled Trial | 166 | Consensus Statement | 3 |
| Multicenter Study | 153 | Clinical Trial, Phase I | 3 |
| Meta-Analysis | 109 | Equivalence Trial | 3 |
| Evaluation Study | 82 | Evidence Synthesis | 3 |
| Letter | 69 | Interview | 1 |
| News | 58 | Video-Audio Media | 1 |
| Retracted Publication | 46 | Guideline | 1 |
| Research Support, U.S. Gov't, P.H.S. | 45 | Clinical Study | 1 |
| Clinical Trial Protocol | 45 | | |

**46 of the 28,075 are tagged `Retracted Publication`** by MEDLINE (co-occurring with another tag,
e.g. still "Journal Article" — this is a status flag, not a content type, so it does not form its
own row below). The `Research Support, *` and `English Abstract` tags are administrative (funding
source, language note), not content type, and likewise form no row of their own.
*[2026-10-10, owner: retracted publications are to be removed from the corpus. See the "Working
corpus" note below — acted on, not left as a flag.]*

**Single-label primary type**, assigned by a fixed priority order over the tags above only (most
specific evidence-synthesis/study-design tag first, down to a bare "Journal Article"; full order and
tag groupings in `article_types_valid.py`), so every article gets exactly one row:

| Primary type | n articles |
|---|---|
| Research article (Journal Article tag, no more specific type) | 22,672 |
| Review (narrative / other) | 3,802 |
| Systematic review / meta-analysis / scoping review | 926 |
| Editorial / letter / comment / news | 360 |
| Clinical trial / protocol | 218 |
| No PMID in PMC header | 38 |
| Dataset / data paper | 29 |
| Historical article | 18 |
| Conference proceedings / abstract | 12 |
| **Total** | **28,075** |

No article fell into "Unclassified" or "Other" — every one of the 28,037 with a PMID carries at
least one tag this scheme recognises. The "Guideline" (1) and "Consensus Statement" (3) tags seen in
the raw table above never win the single-label slot: all four of those articles also carry a
higher-priority tag (Review, Scoping Review, or Systematic Review), checked directly — the priority
order, not a missing category, is why "Guideline / consensus statement" shows no row.

**What "Historical Article" means (owner question, 2026-10-10).** It is NLM's own MeSH publication
type (Unique ID `D016456`, tree V02.530), scope note: an article or part of an article "giving an
account of past events or circumstances significant in a field of study" (often checked together
with the Biography heading). It is assigned mechanically to a citation indexed with a historical
MeSH descriptor, the "History" subheading, or a Personal Name as Subject entry — it marks how NLM
*indexed* the record, not a judgement that the paper itself is old or a history paper in the way a
historian would use the term. Checked against our own 18 primary-"Historical article" pmcids
(titles pulled from `pure_text_corpus` headers): most are substantively historical in the expected
sense — restoring/attributing ancient texts with deep nets (PMC8907065), a dataset of 19th-century
fauna built with an LLM (PMC13012729), century-old Irish folk cures (PMC12704799), medieval rabbinic
literature (PMC11262698), political uses of the ancient past (PMC11373803), a 30-year retrospective
of one symposium's collaboration networks (PMC11747933), 50/60-year bibliometric retrospectives of a
journal (PMC12539508, PMC12881886) — matching the "History" MeSH-subheading auto-rule for anniversary
reviews. One is not: PMC11725834 ("Across the firewall: foreign media's role in shaping Chinese
social media narratives on the Russo-Ukrainian war") is about an ongoing conflict, not history in the
ordinary sense; most likely NLM's historical-period MeSH indexing (e.g. a "History, 21st Century"
descriptor) fired on the subject matter, not the paper's own framing. So the tag is a reliable signal
that NLM's indexers attached a historical MeSH heading, not a guarantee every such paper "is" a
history paper by the field's own lights — worth a manual read of the 18 (or all 25 raw-tagged) before
treating this as a clean category, if it matters downstream.

**Working corpus (owner decision, 2026-10-10): retracted publications are removed.** The 46 articles
MEDLINE tags `Retracted Publication` are taken out of the corpus that any further analysis (author
counts included) is confined to. This is a new filter layered on top of G3's own output, not a G3
rerun: retraction is an article-level fact from PubMed, independent of G3's focal-term/parse-level
exclusions (§6 item 6), so `PG/g3_v1/` itself is untouched — nothing is silently dropped, every
excluded pmcid and its reason is recorded. **New working-corpus size: 28,029** (28,075 valid − 46
retracted). `corpus_statistics/working_corpus.py` writes `working_corpus_pmcids.txt` (the 28,029
pmcids — the frozen reference set for every later step) and `corpus_exclusions_post_g3.csv` (the 46
exclusions, with their PMID and MEDLINE tags, for the record). **Known gap, not resolved by this:**
the 38 valid articles with no PMID have no PubMed record to check, so their retraction status is
unknown, not confirmed negative — they are kept in the working corpus on the absence of evidence,
which is weaker than the positive confirmation the other 27,991 have.

**Reproduction:** `corpus_statistics/article_types_valid.py` (reads `PG/g3_v1/g3_articles_v1.jsonl`,
`diagnostics/openalex_authors/headers.json`, `diagnostics/openalex_authors/pubmed_pubtypes.json`;
no network call needed, both caches already complete for this corpus). Writes
`corpus_statistics/article_types_valid.csv` (28,075 rows: pmcid, pmid, the raw `|`-joined
PublicationType tags, the primary type, and a `retracted` flag) and prints both tables above.
Then `corpus_statistics/working_corpus.py` applies the retraction filter and writes
`working_corpus_pmcids.txt` and `corpus_exclusions_post_g3.csv`.

## 9. Author coverage of the working corpus (2026-10-10)

Scope: the working corpus of §8 (`working_corpus_pmcids.txt`, 28,029 articles); the analysis covers
the 27,991 with a PMID, and reports the 38 without one separately. Pipeline agreed with the owner
before running: article-anchored; PubMed's own author list is the byline backbone; OpenAlex only as
the identity layer (author IDs), matched per article, never author-first.

### 9.1 Repository and journal versions of one article (checked first, owner's request)

The concern: one paper present twice, as a repository copy (preprint server, PMC's preprint
collection) and as the journal version; and, for journal data, a repository location taken instead
of the journal. Checked on the 28,029, not assumed from the preprocessing record:

- **No repository copies remain in the corpus.** F1 (PIPELINE.md, rule R1) removed every article whose
  PMC header says "Article version: preprint" (1,475). In the working corpus the header version is "1"
  (28,013) or "2" (16, journal revisions); no journal abbreviation or publisher is a repository
  (bioRxiv, medRxiv, Research Square, arXiv, SSRN, Preprints, Zenodo, OSF, ...); no header DOI is a
  repository DOI.
- **No identifier shared by two articles:** 0 duplicate PMIDs, 0 duplicate DOIs, 0 OpenAlex works
  matched to two articles.
- **Identical normalised titles: 2 pairs, both different papers, kept.** PMC12507154 / PMC12947111
  (Indian J Ophthalmol): an original study (6 authors) and a piece by 2 other authors under the same
  title. PMC7618139 / PMC7618188: two papers by overlapping teams in Cortex and NeuroImage, different
  author lists and PMIDs.
- **OpenAlex side.** OpenAlex keeps separate records for a preprint and its journal version, and the
  PMID is sometimes attached to the preprint record. The matching of §4 already keeps, per article,
  the OpenAlex record whose DOI is the article's own PMC-header DOI, i.e. the version in the corpus.
  In the working corpus this overrode the PMID hit 69 times: the PMID hit was a repository record
  30 times (arXiv, bioRxiv/medRxiv, Research Square, SSRN, ChemRxiv, figshare and institutional
  repositories), another version of the same journal article 36 times (mostly F1000Research-type
  versions, `.1` in PMC and `.2` in OpenAlex), and something else 3 times. **No kept record has a repository DOI.** 12 kept records are another
  publication than the corpus version (a conference abstract, a chapter, a later journal record;
  the corpus version is not in OpenAlex); their author lists agree name by name with PubMed in all
  12, so their author IDs are used. 22 recent articles (2025: 9, 2026: 13) have no OpenAlex record.
- **Journal data:** taken from PubMed (`NlmUniqueID`, ISSN), which for this corpus is always the
  journal. OpenAlex locations are not used for the journal; where they are needed later (e.g. the 38
  articles without a PMID), a location whose source is a repository is ignored and the journal
  location used.

### 9.2 Method

1. `pubmed_authors_fetch.py`: NCBI `efetch` (MEDLINE citation XML) for the 27,991 PMIDs: author list in
   order (surname, forenames, ORCID where the publisher supplied it, affiliation text), group
   (collective) authors, journal, dates. All 27,991 found; no author list is marked incomplete
   (`CompleteYN`); every article has at least one personal author; 217 also list a group author.
2. OpenAlex: the cached record of §4 (`openalex_verified.json`, author ID, ORCID and name per
   authorship; fetched 2026-10-02), restricted to the working corpus.
3. Alignment inside each article: PubMed bylines and OpenAlex authorships are aligned in order
   (longest common subsequence); a pair counts only if the surname agrees (all surname tokens present,
   or the joined surname equal to consecutive tokens: "de Freitas" / "DeFreitas") and, where both sides
   have given names, an initial agrees. No pair is made on position alone. Check: of 163,646 aligned
   bylines with an OpenAlex ID, 0 have a disagreeing given-name initial (a first version, matching the
   joined surname anywhere inside the name, paired 9 wrong bylines, e.g. "He" inside "Shusheng";
   fixed before these figures).
4. A byline is identified if it has an OpenAlex author ID or an ORCID. Distinct people: OpenAlex IDs and
   ORCIDs merged where they co-occur on a byline (union-find); unidentified bylines give the upper bound.

`author_coverage.py` writes `corpus_statistics/author_coverage_by_article.csv` (per article, counts
only, no names) and, untracked, `diagnostics/author_coverage/bylines.jsonl` and
`author_coverage_summary.json`.

### 9.3 Results

| Bylines (PubMed personal authors, 27,991 articles) | n | % |
|---|---|---|
| Total | 175,502 | 100 |
| Aligned to an OpenAlex authorship | 174,732 | 99.6 |
| With an OpenAlex author ID | 163,646 | 93.2 |
| With an ORCID from PubMed (publisher-supplied) | 58,663 | 33.4 |
| Identified (OpenAlex ID or ORCID) | 166,821 | 95.1 |
| Neither | 8,681 | 4.9 |

| Articles (27,991) | n | % |
|---|---|---|
| Every byline has an OpenAlex ID | 20,765 | 74.2 |
| Every byline identified (OpenAlex ID or ORCID) | 22,474 | 80.3 |
| At least one OpenAlex ID | 27,761 | 99.2 |
| No byline identified | 107 | 0.4 |
| First or last author unidentified | 1,846 | 6.6 |

**Distinct people:** 131,756 OpenAlex author IDs and 104,689 ORCIDs; merged, **133,128 identified
people**; with each of the 8,681 unidentified bylines counted as a separate person, at most 141,809.

**Why 770 bylines have no aligned authorship:** OpenAlex lists fewer authors than PubMed (338 bylines,
56 articles; e.g. one author for a nine-author paper), OpenAlex's list is cut at 100 authorships
(258, 7 articles; a single-work query returns the full list, so this is fixable at the next fetch),
no OpenAlex record (106, 22 articles), names that do not match (68, 46 articles). In the other
direction 2,859 OpenAlex authorships have no PubMed byline: mostly members of a group author that
OpenAlex lists individually, and authorships OpenAlex duplicates within one work. They are not
bylines and are not counted.

**OpenAlex disambiguation, checked against the publisher-supplied ORCID (independent of OpenAlex):**
of 45,898 ORCIDs seen with an OpenAlex ID, 501 (1.1%) appear under two or more OpenAlex IDs (one
person split into several profiles; the merge above joins them); of 46,361 OpenAlex IDs seen with a
PubMed ORCID, 55 (0.12%) carry two or more ORCIDs (different people merged into one profile; not
corrected); 566 bylines have a PubMed ORCID different from the ORCID on the OpenAlex profile.

**By year** (bylines; % with OpenAlex ID / % identified): 2019 58 (100 / 100); 2020 460 (92.4 / 94.8);
2021 2,064 (96.5 / 97.6); 2022 3,866 (96.9 / 97.5); 2023 10,701 (96.7 / 97.5); 2024 29,754
(94.8 / 95.9); **2025 65,785 (89.2 / 91.9)**; 2026 62,814 (95.9 / 97.3). The 2025 dip seen in §5 remains.

**By primary type** (§8; % with OpenAlex ID / % identified): research article 143,508 bylines
(93.0 / 94.9); review 22,288 (94.3 / 95.4); systematic review etc. 5,604 (94.2 / 96.1); clinical
trial/protocol 1,971 (92.7 / 95.5); editorial/letter/comment/news 1,530 (95.0 / 96.6); dataset 329
(93.0 / 94.8); conference proceedings 182 (96.2 / 96.7); historical article 90 (92.2 / 93.3). No type
stands out.

**The 38 articles without a PMID** have no PubMed author list; all 38 have an OpenAlex record (found by
DOI), with 214 authorships in total. Not included in the tables above.

**Compared with §4** (34,662 articles, OpenAlex's own author lists as bylines): there, 73.6% of
articles had an ID for every author and 6.8% of bylines lacked one. On the working corpus, with
PubMed's list as the byline, 74.2% of articles have an OpenAlex ID for every byline and 6.8% of bylines
lack one; ORCID lowers the unidentified share to 4.9%.

**Not done here** (later steps): affiliation coverage (owner: later); institutions, which need a new
OpenAlex fetch (the cache holds no institutions or locations; the earlier key is not in this
environment, so `OPENALEX_API_KEY` must be set when it is run); the decision on unidentified bylines
(placeholder nodes or name linking, §5).

**Reproduction:** from `corpus_statistics/`: `NCBI_API_KEY=... python3 pubmed_authors_fetch.py`
(about 9 min; cache `diagnostics/author_coverage/pubmed_authors.json`), then
`python3 author_coverage.py` (about 20 s).
