# Full-scale corpus statistics and author identification (2026-10-02)

Descriptive statistics of the current full-scale corpus, an assessment of how authors can be
identified, and a check of what is lost where they cannot. Read-only analysis: no pipeline output was
changed. Scripts: `fullscale_pipeline/corpus_statistics/` (tracked). Data they read and write:
`fullscale_pipeline/diagnostics/openalex_authors/` (local, not in git, like the other diagnostics
caches). The OpenAlex key is read from the `OPENALEX_API_KEY` environment variable and is not stored
in any file.

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
