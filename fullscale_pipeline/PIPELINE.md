# Full-scale pipeline (PMC / LLM corpus) — stage record

Last reconciled with the files on disk: **2026-09-24**. Companion: [`RUN_LOG.md`](RUN_LOG.md)
(append-only record of every run and code change). Standing rule: `SESSION_PROTOCOL.md` §J.
This document covers the full-scale pipeline only.

**Purpose.** A tractable, reviewer-facing record of how the full-scale corpus is retrieved,
preprocessed, parsed and (later) analysed: which scripts, which files they read and write, and the
order in which they were run. **The pipeline is expected to change** — several historical scripts
are imperfect (see §5) — so this document separates *what stage exists* from *which script
currently implements it*. Stage IDs stay fixed when a script is replaced; the script column changes
and the change is logged in §5 (Deviations) and in `RUN_LOG.md`.

**Abbreviations used for paths**

| Short | Path |
|---|---|
| `R` | `/mnt/hum01-rds/Basov/p91688di` (RDS storage; **not** under version control) |
| `LCS` | `R/llm_corpus_staging` |
| `PP` | `R/pmc_preprocessing` |
| `PG` | `R/phase5_graphbrain` |

**Evidence tags.** `[LOG]` a log file states it · `[CODE]` read from the script · `[MTIME]` file
modification time · `[DERIVED]` arithmetic on verified counts · `[INFERENCE]` not directly evidenced.

**Status vocabulary.** `DONE-current` ran on the current corpus definition (2026-09-23 list of
46,241 IDs) with the current script · `DONE-stale` ran earlier (May–June 2026) on the older, smaller
corpus and/or with an older script version, and has **not** been re-run · `GAP` no generating code
found · `NOT-BUILT` / `NOT-RUN` not yet done.

---

## 1. Stage table

| ID | Stage | Launched as | Reads | Writes | Last run | Status |
|---|---|---|---|---|---|---|
| R1 | PMC ID query (Entrez esearch by month, then esummary) | `python3 -u query_pmc_entrez.py` (no SLURM; run under `nohup`) | NCBI Entrez, `db=pmc`, filter `open access` | `LCS/target_pmcids.txt`, `LCS/target_metadata.json` | 2026-09-23 (46,241 IDs); earlier 2026-05-13 (34,800) | DONE-current |
| R2 | Full-text fetch, first pass | `python3 fetch_s3_pmc.sh` — **a Python file despite the `.sh` name**; running it with `bash` fails | `LCS/target_pmcids.txt`; `s3://pmc-oa-opendata/<PMCID>.<v>/<PMCID>.<v>.txt`, v in 1..4 | `LCS/pure_text_corpus/<PMCID>.txt` | 2026-05-09 (34,398 of 34,453) | DONE-stale (bucket versions outside 1–4 are missed, §5 D4) |
| R3 | Delta fetch: set difference, then fetch only missing | `python3 isolate_delta.py`, then `sbatch submit_delta_fetch.sh` (runs `fetch_delta_pmc.py`) | `LCS/target_pmcids.txt`, `LCS/pure_text_corpus/` | `LCS/delta_pmcids.txt`, `LCS/pure_text_corpus/` | 2026-05-14 (402 IDs); 2026-09-24 job 21295326 (11,555 IDs, 11,491 fetched) | DONE-current (64 targets still missing) |
| P0 | Content regions: which lines of each text are analysed (start, end, extra whitelisted regions) | `python3 pmc_preprocessing/build_content_regions.py` (rebuilt 2026-09-24, D8) | `LCS/target_pmcids.txt`, `LCS/pure_text_corpus/` | `PP/content_regions_v2.csv` (46,177 rows; 41,243 with `include`=1). Earlier May generator lost; its outputs `PP/sequence_metadata_relaxed.csv` (31,511 rows), `sequence_metadata.csv`, `full_sequence_mapping.txt`, `missing_reference_sequences.txt`, `reference_aliases_full.txt` remain on disk | 2026-09-24 21:56 [LOG] | DONE-current (P1 onward still read the May CSV) |
| F1 | Article blacklist, rules R1 to R3 and R5: preprints (header says "Article version: preprint"); paper types (PubMed publication type, else the journal's Subjects line; a specific label beats the generic "Journal Article"); articles with no usable type; comment-like titles with no partner needed | `python3 pmc_preprocessing/article_blacklist.py --workers 8` (one command applies every rule; PubMed answers are cached in `LCS/blacklist_cache/`; network on the first run, key from `NCBI_API_KEY`) | `LCS/target_pmcids.txt`, `LCS/target_metadata.json`, `LCS/pure_text_corpus/` (Subjects, PMID and Article version lines), PubMed esummary and efetch | `LCS/article_blacklist.csv` (2,683 rows: preprint 1,475, case report 447, no PubMed record 256+57, comment or reply 194+83, correction/erratum/retraction 112, guideline/consensus 39, copies 13, similar introduction 4, retracted-republished 2, address 1), `LCS/blacklist_decisions.csv`, `LCS/blacklist_summary.json`, `LCS/blacklist_cache/` | 2026-09-25 18:34 [LOG] | DONE-current (not yet applied by any later stage) |
| F2 | Duplicate screen (rule R4), run inside the same command as F1: title Levenshtein distance of 0.30 or less among the articles not blacklisted by R1 to R3; per pair, in order: comment or reply, erratum or republication link, no PubMed record, authors (same or partly overlapping: definite copy if at least 50% of the shorter text's 8-word phrases are shared; different authors within title distance 0.15: opening-text distance of 0.5 or less). Both members are blacklisted except in the comment, erratum and no-record cases | same command as F1 | as F1, plus PubMed efetch for the 951 articles in pairs | 824 pairs in `LCS/blacklist_decisions.csv`; 49 same or overlapping-author pairs marked `pending_window_check` (§6 item 13) | 2026-09-25 18:34 [LOG] | DONE-current |
| F3 | Author-based blacklist expansion, run after F1/F2 and appended to the same list (to be folded into `article_blacklist.py` later; re-running that script rebuilds the list without these rows, so F3 must then be run again). Our theory concerns scientists as authors, so blacklisted: book chapters whose author block lists the volume editors (header Journal ID is an ISBN), articles for which PubMed lists no authors, articles whose PubMed authors are only group or consortium names, and articles with no PubMed record and no author line in the header. One-author articles are kept | `python3 pmc_preprocessing/author_rules_expansion.py` (dry run) then `... --apply` (backs up and appends; PubMed cache in `LCS/blacklist_cache/pubmed_esummary_authors.json`, key from `NCBI_API_KEY`) | `LCS/article_blacklist.csv`, `LCS/target_metadata.json`, `LCS/pure_text_corpus/` (Journal ID, PMID, author lines), PubMed esummary | 116 rows appended to `LCS/article_blacklist.csv` (no authors in PubMed 50, volume-editor book chapters 33, group-only 21, no author line in the header 12); `LCS/author_rules_decisions.csv`; backup `LCS/article_blacklist.before_author_rules_20260925.csv` | 2026-09-25 18:58 [LOG] | DONE-current |
| F4 | Final exclusion list and whitelist: combines the article blacklist (F1 to F3) with the structural status of P0 (headingless, no Results/Discussion/Conclusion heading, no end marker) into one product; rerun after any change to the blacklist or to P0. The structural policy for articles with headings but no closing section is still open and currently excludes them (`STRUCTURAL` in the script) | `python3 pmc_preprocessing/build_exclusion_list.py` (`--replace` to rebuild; the old files are kept as `.previous`) | `LCS/article_blacklist.csv`, `PP/content_regions_v2.csv`, `LCS/target_metadata.json` | `LCS/article_exclusions.csv` (6,878 rows: blacklist 2,799, structure 4,079), `LCS/article_whitelist.txt` (39,299 PMCIDs) | 2026-09-25 19:07 [LOG] | DONE-current; P1 onward to read the whitelist |
| P1 | Citation masking (regex; citations become `__CITE_<pmcid>_NNN__`) inside the content boundaries only **Rebuilt 2026-09-25 as `citation_masking_v2.py` (D9) and run on the whole whitelist (RL-054, final): 39,299 articles, `LCS/masked_corpus_v2/` (47,619 region files, line counts unchanged), `LCS/citation_dictionary_v2.jsonl` (412,616 cited works, 90% matched to the visible reference list), `citation_marks_v2.jsonl`, `superscript_residuals_v2.jsonl`.** | `nohup python3 citation_standartization_soft_masking.py` (an sbatch wrapper `.sh` exists but no SLURM output from it exists) | `PP/sequence_metadata_relaxed.csv` (to be switched to the boundaries of `PP/content_regions_v2.csv` and the articles of `LCS/article_whitelist.txt`), `LCS/pure_text_corpus/` | `LCS/masked_corpus_v1/` (28,284 files), `PP/citation_vault_light_masking_v1.jsonl` | 2026-05-16 | DONE-current (v2) |
| P1b | Non-prose removal: tables, LaTeX, formulas and statistics are removed from the masked text so that graphbrain reads only prose. One output line per input line (removed lines become empty), so alignment with the raw text is kept. Rules T1 to T9 (tables, captions, LaTeX, formulas, statistics, symbols, hexadecimal codes) are listed in the script's docstring | `python3 pmc_preprocessing/nonprose_removal_v1.py --selftest`, then `python3 pmc_preprocessing/nonprose_removal_v1.py --workers 8` (about 1 minute; refuses to overwrite its output) | `LCS/masked_corpus_v2/`, `PP/content_regions_v2.csv`, `LCS/pure_text_corpus/` (tab-separated cells of the raw lines, because the masked files have lost their tabs) | `LCS/clean_corpus_v1/` (same file names), `LCS/nonprose_removal_v1_summary.json`, `LCS/nonprose_examples_v1.jsonl` | tested 2026-09-25 on scratch samples (300 and several of 600 to 700 articles), not yet run on the corpus | BUILT, NOT RUN |
| P2 | Sentence split (sciSpaCy `en_core_sci_sm`), focal-word regex, keep hit sentence ±1 | `nohup python3 focal_window_extraction.py` | `LCS/masked_corpus_v1/` | `LCS/focal_extractions_v1.jsonl` (22,795 documents) | 2026-05-16; **script edited 2026-09-23, not re-run** | DONE-stale |
| P2d | Diagnostic: why documents produced no focal window | `python3 exclusion_diagnostics.py` | P0 csv, `LCS/pure_text_corpus/`, `LCS/focal_extractions_v1.jsonl` | `PP/exclusion_report.csv` (5,489 rows) | 2026-05-17 | DONE (diagnostic, not a chain input; uses an outdated hardcoded word list) |
| P3 | Inventory of citation tokens found inside the extracted windows **Superseded by the dictionary of cited works written by P1 v2 (D9).** | `python3 citation_resolution_1_inventory.py` | `LCS/focal_extractions_v1.jsonl`, `PP/citation_vault_light_masking_v1.jsonl` | `PP/api_inventory_target.json` (13,271 documents) | 2026-05-18 | DONE-stale |
| P4 | Resolve tokens through Entrez `efetch` (batches of 200, 0.35 s apart, no API key); positional index for numeric citations, year match for author-year; unresolved tokens get `__REF_HASH_<sha256[:10]>__` **Superseded by the dictionary of cited works written by P1 v2 (D9).** | `python3 citation_resolution_2_api.py` | `PP/api_inventory_target.json`, NCBI Entrez | `PP/global_translation_dictionary.json` (105,389 tokens), optional `PP/phase2_api_errors.log` | 2026-05-18 | DONE-stale |
| P5 | Replace per-document tokens with the global reference ids **Superseded by the dictionary of cited works written by P1 v2 (D9).** | `python3 citation_resolution_3_translate.py` | `PP/global_translation_dictionary.json`, `LCS/focal_extractions_v1.jsonl` | `LCS/focal_extractions_v2_resolved.jsonl` (22,795) | 2026-05-18 | DONE-stale |
| P6 | Coreference resolution (fastcoref `LingMessCoref`, CUDA if available) and re-check that each block still contains a focal word | `python3 citation_resolution_4_coref.py` | `LCS/focal_extractions_v2_resolved.jsonl`, `PP/focal_words.txt` | `LCS/focal_extractions_v3_graphbrain_ready.jsonl` (22,572 kept, 223 dropped); `PP/phase4_document_errors.log` is created only if documents fail (none did) | 2026-05-19/20 | DONE-stale |
| P7 | Restore the documents dropped in P6, unmutated, flagged `restored_from_v2` | `python3 citation_resolution_4b_restore.py` | `LCS/focal_extractions_v2_resolved.jsonl`, v3 | appends to v3: **223** records; v3 then has 22,795 unique PMCIDs | 2026-05-20 | DONE-stale |
| G1 | Split v3 into 50 shards of ≈456 records | `python3 01_matrix_partition.py` | `LCS/focal_extractions_v3_graphbrain_ready.jsonl` | `PG/input_shards/shard_01..50.jsonl` (22,795 lines in total) | 2026-05-21 | DONE-stale |
| G2 | Graphbrain parsing of each shard (`en_core_web_trf`); each sentence stored as `('source', pmcid, main_edge)` so that every edge stays tied to its article | `sbatch submit_v2.sh` (array 1-50%15, `multicore`, 4 CPUs, 16 GB, 24 h) runs `parse_stage_v2.py` | `PG/input_shards/` | `PG/output_sqlite_v2/db_provenance_NN.sqlite` (50 files, 83 GB), logs in `PG/scripts/logs_v2/` | 2026-06-01, array job 15772611 | DONE-stale |
| G3 | Curation: writes `source_core`, `source_periphery`, `source_fringe` links | `sbatch submit_4h.sh` (array 1-500 of which only 1–50 do work, `serial`, 1 core, 12 h) runs `chunk_4h_hpc.py <db>` | `PG/output_sqlite_v2/db_provenance_NN.sqlite`, `PP/focal_words.txt`, `PG/nltk_abridged_stopwords_list.txt`, spaCy `en_core_web_sm` | `PG/postprocessed_output/db_curated_NN.sqlite` (50 files, 2.9 GB), logs `PG/logs/cluster_<task>.log` | 2026-05-30 | DONE-stale; **older than its raw db in 50 of 50 shards** (§6) |
| M1 | Matrix builder for the full corpus: turns the curated links into relation matrices. Needs article metadata and authorship edges, which do not yet exist at full scale | — | `PG/postprocessed_output/` | — | — | NOT-BUILT |
| A1 | Solver and evaluation on the full-scale matrices | — | M1 output | — | — | NOT-RUN |

Where several stages read `PP/focal_words.txt`: **only P6 and G3 do.** P2 and P2d carry their own
hardcoded lists (§5 D3).

## 2. Data flow

```
R1 target_pmcids.txt ──► R2/R3 pure_text_corpus/  (46,242 files)
                                │
        P0 sequence_metadata_relaxed.csv (GAP) ──┐
                                ▼                ▼
                       P1 masked_corpus_v1/ + citation vault
                                ▼
                       P2 focal_extractions_v1.jsonl ──► P2d exclusion_report.csv
                                ▼                       (side branch)
        P3 api_inventory_target.json ──► P4 global_translation_dictionary.json
                                ▼
                       P5 focal_extractions_v2_resolved.jsonl
                                ▼
                       P6 (+P7) focal_extractions_v3_graphbrain_ready.jsonl
                                ▼
                       G1 input_shards/  ──► G2 output_sqlite_v2/ ──► G3 postprocessed_output/
                                                                          ▼
                                                             M1 (not built) ──► A1
```

## 3. Corpus funnel (counts re-derived 2026-09-24)

| Point | Count | Note |
|---|---|---|
| Target list, 2026-05-13 | 34,800 | superseded; backup `LCS/target_pmcids.pre_llm_terms_backup_20260923_133436.txt` |
| Target list, 2026-09-23 | 46,241 | current |
| Full texts on disk | 46,242 | 46,177 of the current targets, plus 65 files not in the current list; 34,751 before the 2026-09-24 delta |
| Current targets without a text | 64 | 22 exist in the bucket under version numbers outside 1–4; 42 have no folder in the bucket [checked with `aws s3 ls`] |
| Rows in P0 csv | 31,511 | covers only the May-era texts |
| Blacklisted (F1 to F3, 2026-09-25) | 2,799 | of 46,177 texts, leaving 43,378: F1/F2 2,683 and F3 116 (author rules); reasons in the F1 and F3 rows; the list is meant to grow |
| Whitelist after F4 (2026-09-25) | 39,299 | 46,177 minus 2,799 blacklisted minus 4,079 excluded by P0 structure (no closing heading 2,484, headingless 1,234, no end marker 361); file `LCS/article_whitelist.txt` |
| Masked by P1 v2 (2026-09-25) | 39,299 articles, 47,619 regions | one masked file per region of every whitelisted article; `LCS/masked_corpus_v2/` |
| Rows in `content_regions_v2.csv` (P0, 2026-09-24) | 46,177 | `include`=1: 41,243; no closing heading 2,796; headingless 1,635; no end marker 503 |
| Documents masked (P1) | 28,284 | rows with both content boundaries |
| Lost between texts and P1 | 6,467 | 34,751 − 28,284 `[DERIVED]`: 3,240 without a P0 row plus 3,227 without boundaries |
| Documents with focal windows (P2) | 22,795 | 5,489 without = exactly the rows of `exclusion_report.csv` |
| Documents in v3 (P6+P7) | 22,795 | 22,572 kept + 223 restored |
| Shards / raw DBs / curated DBs | 50 / 50 / 50 | |

The "22k" corpus is therefore a **filtered subset** of what was retrieved. It covers 22,756 of the
46,241 current targets; 23,485 current targets have never been through P1–G3, and 39 of its
documents are not in the current list `[DERIVED]`.

Why P2 excluded documents (`PP/exclusion_report.csv`, 5,489 rows): focal words only after the end of
the main text 4,878 (88.9%); no focal word in the raw file 303 (5.5%); before and after 221 (4.0%);
before the start only 75 (1.4%); regex discrepancy in the main body 12 (0.2%).

## 4. Stage notes

- **R1.** Every term is a quoted exact phrase tagged `[Text Word]` or `[Title/Abstract]` (no
  substring matching). 2026-09-23 added 33 general LLM/chatbot names, version-qualified where a bare
  name collides with other biomedical uses (bare GPT = a liver-enzyme abbreviation, Llama = the
  animal, PaLM = the plant, Perplexity = an NLP metric). The search slices by month up to the run
  date, so **the result depends on when it is run**. Supports an NCBI key from the environment
  variable `NCBI_API_KEY` (interval 0.11 s with a key, 0.35 s without); the 2026-09-23 run used
  the anonymous rate. `EMAIL` in the script is still a placeholder.
- **R2/R3.** Text only (`.txt`); the same bucket folders also hold `.xml`, `.pdf`, `.json`, figures and
  supplements, none of which are fetched. `fetch_s3_pmc.sh`, `fetch_delta_pmc.py` and its two
  identical copies share one fetch loop; only `fetch_delta_pmc.py` is used.
  `submit_delta_fetch.sh` requests one core because the `serial` partition allows only one; the
  eight download threads are I/O-bound.
- **P1.** Masks numeric brackets, numeric parentheses, verbal parentheses and active author–year
  forms, only between `content_start_line` and `content_end_line`; deletes and rewrites the vault.
- **P1 v2 (2026-09-25).** One masked file per region of each whitelisted article: `<PMCID>.txt` is the main window and
  `<PMCID>.r2.txt`, `.r3.txt` ... are the extra windows (supplementary information with prose, ethics statements). 39,299
  articles give 47,619 files: 7,839 articles have an `r2` window, 473 an `r3`, 5 an `r4` and 3 an `r5`. Every file of an article
  belongs to that article: the PMCID in the name is the link, and the citation numbers `a<PMCID digits>r<n>` are shared across
  its windows. **The focal-window stage has to read the extra windows together with the main file and tag every extracted piece
  with its region.** Masked files keep one line per raw line (checked line by line), but tabs are collapsed to spaces, so table
  rows can no longer be recognised in them; a later step must look for tab-separated cells in the raw text at the same line.
- **P2.** Deletes and rewrites `focal_extractions_v1.jsonl` (fixed name). Word list is hardcoded.
- **P4.** Numeric citations are resolved by position in the article's `<ref>` list; author–year
  citations by the first reference with the same year (a crude proxy). No NCBI key is used.
- **P6/P7.** P6 re-filters after coreference; 223 documents lost their focal word and were
  restored by P7 without mutation. The `restored_from_v2` flag identifies them.
- **G2.** The v1 parse (`02_transformer_node.py`, `submit_array.sh`, `PG/output_sqlite/`, job
  15138115, 2026-05-21/22, 50 databases) stored bare edges with no link to the article and is
  superseded. Resumable through `PG/scripts/logs_v2/progress_v2_NN.log`; clear these to force a
  clean re-run.
- **G3.** Deletes and recreates each `db_curated_NN.sqlite` (safe to re-run). Only task IDs 1–50 do
  work. `PG/curated_sqlite_v2/` holds two trial databases only and is **not** the full-scale output
  despite the name.
- **Downstream contract (for M1).** `chunk_4h_hpc.py` writes `source_core`, `source_periphery` and
  `source_fringe` links with ids `<pmcid>::<hash>::<hash>`. The matrix builder will have to loop
  over or merge the 50 curated databases and be supplied with article metadata and authorship
  edges, which have not yet been built for the full corpus.

## 5. Deviations from the historical scripts

| ID | Date | Change / finding | Affected |
|---|---|---|---|
| D1 | 2026-09-23 | Query and word lists expanded by 33 general LLM/chatbot names in `query_pmc_entrez.py`, `focal_window_extraction.py` and `focal_words.txt` (exact-phrase policy; reasons in R1 note). List grew 34,800 → 46,241. P2 has **not** been re-run with the new list. | R1 (re-run), P2 (script only), P6/G3 read `focal_words.txt` |
| D2 | 2026-09-23 | NCBI API-key support (environment variable) and adaptive request interval added to R1. P4 has no key support. | R1, P4 |
| D3 | open | Focal-word list has three unsynchronised sources: hardcoded in `focal_window_extraction.py` and `exclusion_diagnostics.py`, file `focal_words.txt` for P6/G3. Should become one file read by all. | P2, P2d, P6, G3 |
| D4 | open (found 2026-09-24) | Fetch loops try versions 1–4 only. 22 current targets exist under other versions (`.319` ×15, `.358` ×2, `.5` ×4, `.7`, `.8`). | R2, R3 |
| D5 | 2026-09-24 | New SLURM wrapper `submit_delta_fetch.sh` for the delta fetch; the older wrapper `run_extraction.sh` runs a different, crashed prototype and must not be used. | R3 |
| D6 | 2026-05-23 (historical) | Graphbrain parse changed from v1 (no provenance) to v2 (`('source', pmcid, main_edge)`). | G2 |
| D7 | 2026-05-20 (historical) | Coreference filter drops documents whose focal word vanished; patched by restoring them (P7). | P6, P7 |
| D8 | 2026-09-24 | P0 rebuilt as `build_content_regions.py` (new output `content_regions_v2.csv`; May CSV kept). Start = first Introduction heading, else the `U+009F`-framed `=` divider. End = first back-matter heading after the last body-type heading, else References, else end of file. Window 2+ = whitelisted Supplementary Information (prose only) and Ethics statements, same PMCID. Reason: the May end rule cut articles short (Supplementary Information inside the abstract block; Ethical Considerations inside Methods). | P0; P1 onward stale |
| D9 | 2026-09-25 | P1 rebuilt as `citation_masking_v2.py`. Numeric citations and parenthetical author-year citations are deleted from the text; narrative author-year citations become `a<PMCID digits>r<n>`, with the same n for the same work in an article (its position in the article's reference list when it can be matched, otherwise numbered on from the size of the list); a dictionary of cited works with aliases, mention counts, matched reference text and DOI is written; line breaks are kept. P3 to P5 (citation resolution) are superseded. Reasons: the old masks broke graphbrain parses and over-masked (§6 items 15 and 16), and numeric citations cannot be resolved when the reference list is not visible. Sequel: P2 must read `masked_corpus_v2` and its region files. | P1 to P7 |
| D10 | 2026-09-25 | P0: extra windows (supplementary, ethics) end at the next heading of any kind, 60 lines after their heading, or the References heading, whichever comes first. Before, they ended only at the next back-matter heading, so an ethics window could run for hundreds of lines (longest 8,679; 973 windows over 60 lines) and, without a References heading, to the end of the file. Main windows, statuses and the whitelist are unchanged (checked for all 46,177 articles); extra windows fell from 10,485 to 9,120. | P0, P1 |

## 6. Known gaps and staleness (read before trusting any downstream number)

1. **The May P0 generator was lost** (only consumers reference `sequence_metadata_relaxed.csv`). P0 was
   rebuilt on 2026-09-24 (D8). The May CSV and everything built from it (P1 onward) remain in place.
2. **Everything from P1 onward has run only on the May-era subset.** The ~11.5k newly fetched texts
   and the 2026-09 list have never been through P0–G3.
3. **G3 predates its input.** All 50 curated databases (2026-05-30) are older than the raw v2
   databases they read (2026-06-01, job 15772611) `[MTIME]`. Whether the content differs is not
   established; `nltk_abridged_stopwords_list.txt` (2026-06-29) also postdates the curated build
   `[INFERENCE]`. Re-running G3 is safe and would settle it.
4. **Outputs have fixed names and are overwritten in place** (P1 vault, P2, P3, P4, P5, P6 delete and
   rewrite; G3 recreates). A re-run would destroy the artifacts G1–G3 were built on. Back up or
   rename first.
5. **P2 outputs used the older word list** (chatbot names absent) `[INFERENCE from the list history]`.
6. **64 targets without a text and 65 orphan files** (in the old list, not in the new). Left in place.
7. **No environment lockfile.** Versions below were read from the installed packages on 2026-09-24;
   the May–June runs may have used other versions.
8. **Time-dependent inputs:** the PMC query result changes over time (65 earlier articles disappeared
   from the re-run); P4 depends on live NCBI records.
9. **Figure and table lines must be removed in preprocessing (planned, flagged 2026-09-24).**
   Captions and labels such as `Fig. 1`, `Table 2` and table cell fragments occur as heading-like
   lines in the texts. No stage removes them yet; add the step to P1/P2 before the next full run.
10. **The May content-end rule cut many articles short `[DERIVED]`;** fixed in the rebuilt P0 (D8), but
    P1 onward still use the May boundaries and must be re-run from P1 on `content_regions_v2.csv`
    on `content_regions_v2.csv` and the whitelist of F4.
11. **Windows that ran on past their section (found 2026-09-25, fixed in D10).** Extra supplementary and ethics windows ran on for
    hundreds of lines and, without a References heading, to the end of the file. Capped in P0 (D10). Most of the "reference lists
    inside windows" first reported were section titles such as "3.1.2 References" inside the article, not reference lists.
12. **Non-prose text before graphbrain: script built and tested (2026-09-25), not yet run on the corpus (stage P1b).** Removed by
    `nonprose_removal_v1.py`: table rows (found in the raw text), caption lines ("Table 3 ...", "Fig. 1 ..." followed by a capitalised
    word), LaTeX, formulas, statistics, hexadecimal codes. Symbols become words or disappear ("45 °C" to "45 degrees Celsius", "m²" to
    "m squared", "≥ 18" to "at least 18", "10×" to "10 times", a dimension such as "3 × 3" is deleted, "±" is deleted, "=" is deleted
    inside formula fragments and becomes "equals" only in plain statements such as "temperature equals 0.7"). Running sentences that mention a
    table or figure ("Table 5 shows ...") stay. 107 hard cases in the script's self-test pass; on 700 fresh articles no line count changed and
    every cleaned line is the masked line with removals and the inserted words only. **Known limits:** some formula fragments inside mixed
    prose lines survive (about 1.6 per 1,000 lines still contain an "equals" that is formula residue, and prose with several equalities);
    URLs and DOIs are kept (about 500 lines in 600 articles contain one; a candidate for a later rule); a few number lists remain.
    Table rows are found from the raw text, so the script needs the raw files; a row with a single tab-separated cell is treated as prose.
13. **Text-overlap test for the article blacklist moves to the focal windows (planned, decided 2026-09-25).**
    Pairs with same or partly overlapping authors and titles within 0.30, other than clear copies, are to be compared
    on the context windows around the focal terms (after P2). Both articles are then blacklisted if more than 10% of
    the shorter window's 8-word phrases are shared; shared text unrelated to LLMs does not count. Until then these
    pairs (49 today) stay in the corpus.
14. **Group and consortium authors: decision to revisit (flagged 2026-09-25).** F3 blacklists the 21 articles whose PubMed
    authors are only group names, because our theory concerns scientists as authors. Some are relevant to LLM research
    (the three CHART chatbot-reporting papers, the NLLB Team translation paper, the expert-level academic questions
    benchmark). The rows carry the reason `authors:group_only`, so the decision can be reversed by removing them.
15. **P1 masks many strings that are not citations `[DERIVED]`; fixed in the rebuilt P1 (D9), run 2026-09-25.** Checked on 47 artificial cases and on the May vault
    (1,062,412 masks): equation labels and enumerations such as `(1)`, `(2)` (219,600 masks are a single small number in
    parentheses), intervals such as `[0, 1]` (2,353), and parentheses with a year that is a date, a quantity or a sample
    size (32,529 have no author-like name before the year). Superscript numeric citations and `[Author, year]` brackets
    are not masked. A parenthesis holding several citations becomes one token. Numbers in parentheses are a real citation
    style in some articles (for example PMC13121147), so they cannot simply be excluded; the citation style has to be
    detected per article. P1 also collapses all line breaks.
16. **P4 resolves numeric citations by the wrong index and creates false shared references `[DERIVED]`; P3 to P5 are superseded by the P1 v2 dictionary (D9).**
    `citation_resolution_2_api.py` takes the running counter in the token (`..._019__`) as the reference number instead of
    the number inside the brackets; among 460,547 tokens of the form `[n]` in the May vault the two agree in 9.8%. Citations
    it cannot resolve get a hash of the raw string, so `(1)` or `[18]` from different articles receive the same reference
    id: of 63,473 hashed tokens in the May dictionary, 49,716 (78%) share a hash id with another article (`(1)` is shared
    by 1,271 articles). In the factorisation this would create shared cited works that do not exist. P1 and P4 must be
    rebuilt and tested before the whitelisted corpus goes through them.
17. **The environment cannot run graphbrain or spaCy as it stands `[DERIVED]`.** `tensor_env` has numpy 2.2.x, installed on
    2026-07-11, while spaCy 3.4.4 and thinc 8.1.12 (installed 2026-05-16, before the graphbrain runs G2 and G3) need
    numpy 1.x; importing spaCy fails with a binary incompatibility. The May runs therefore used a different numpy. G2 and
    G3 cannot be rerun in this environment without pinning numpy below 2 (a separate environment is safer). A test on
    2026-09-25 used numpy 1.26.4 from a temporary folder, without changing `tensor_env`.
18. **Superscript citations: handled in P1 v2, weak cases left for post-processing (flagged 2026-09-25).** P1 v2 removes a
    superscript number glued to a word ("development1", "models1,2", "et al.43") when the evidence is strong (numbers within the
    reference count, no veto, and an ordinary word in an article that cites by superscript, or "et al.N", or a lower-case word with
    a period or list before the number); 320,707 removed in 11,956 superscript-style articles. **Not removed, to be handled in
    post-processing:** weak cases such as "systems3" (an ordinary word plus one number in an article not recognised as
    superscript-style) and surname plus number ("Hinton25"): 22,625 candidates, the first 20 per article listed in
    `LCS/superscript_residuals_v2.jsonl` (17,570 rows). Some of them are variables or equation fragments ("equation5", "pred2"), so a
    review is needed before deleting. Model names (GPT5, Llama3, gpt2) are protected by a name list built from articles that do not
    cite by superscript.

## 7. Environment

Conda environment `tensor_env` (`/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env`), read
2026-09-24: Python 3.10.20; graphbrain 0.7.0; spaCy 3.4.4 with `en_core_web_trf` 3.4.0 (G2),
`en_core_web_sm` 3.4.1 (G3), `en_core_sci_sm` 0.5.1 and scispacy 0.5.1 (P2); fastcoref 2.1.6 (P6);
torch 2.11.0; transformers 4.25.1; numpy 2.2.5 (a leftover 2.2.6 record also exists; see §6 item 17); scipy 1.15.3; pandas 2.3.3; rapidfuzz 3.14.5 (added 2026-09-25 with `pip install --no-deps`, F2); awscli 1.44.78 (R2/R3;
`boto3` is not installed). Cluster: CSF3, partitions `serial` (one core only) and `multicore`; short
network-only commands (R1, isolate_delta) were run directly on the `incline` login host, which has no
`sbatch`.

## Appendix A — files that are NOT part of the current chain

| File(s) | Why excluded |
|---|---|
| `R/run_extraction.sh`, `R/extract_pmc_llm.py` | Earlier prototype (Europe PMC search, XML into `llm_xml_corpus/`, absent); only run crashed (`No module named 'boto3'`, `slurm-14097087.err`) |
| `R/fetch_pmc_data.sh`, `R/urllist.short.txt` | Failed (404s, `slurm-13388368.out`) |
| `R/map_s3_paths.py`, `LCS/s3_manifest.txt`, `R/fetch_s3_pmc.sh.save`, `LCS/s3_download.log` | Manifest route; no later script reads the manifest |
| `R/fetch_delta_txt.py`, `R/fetch_delta_corpus.py` | Byte-identical to `fetch_delta_pmc.py`, never launched |
| `R/parse_delta_xml.py`, `R/extract_metadata.py`, `R/explore_matrices.py`, `R/pmc_toy_exploration.ipynb`, `R/parsed_matrices/`, `R/PMC12810641.1/`, `slurm-13531724.out`, `slurm-13532457.out` | One-article April test |
| `R/estimate_arxiv_corpus.py`, `R/arxiv_estimation_log.txt` | arXiv count, unrelated |
| `PP/fetch_article_types.py`, `PP/sequence_metadata_typed*.csv`, `PP/api_fetch*.log` | Side branch (article types); no later stage reads it |
| `PP/verify_boundaries.py` | Ad hoc check of P0 boundaries |
| `PP/build_article_blacklist.py`, `append_blacklist_preprints.py`, `fetch_pubmed_types_unclassified.py`, `refine_blacklist_unclassified.py`, `title_levenshtein_duplicates.py`, `resolve_duplicate_pairs.py`, `heading_levenshtein_duplicates.py`; `LCS/article_blacklist.previous_incremental_20260925.csv` with its `.before_*` backups, `article_blacklist_removed_20260925.csv`, `LCS/title_screen/`, `LCS/heading_screen/`, `PP/pubmed_types_unclassified_20260925.csv` | The incremental blacklist build of 2026-09-25 (RL-026 to RL-039) and the heading screen, superseded by the single script `article_blacklist.py` (RL-040, RL-041); kept until their deletion is approved |
| `PG/scripts/02_transformer_node.py`, `submit_array.sh`, `PG/output_sqlite/` | v1 parse (75 GB), superseded by v2 |
| `PG/scripts/trial_run_v2.py`, `verify_provenance.py`, `verify_v2_provenance.py`, `trial_postprocess.py`, `trial_generational.py`, `PG/curated_sqlite_v2/` | Trial and diagnostic scripts and their outputs |

## Appendix B — commands (all run from `R`; no secrets are stored in any script)

| Stage | Command | Evidence |
|---|---|---|
| R1 | `export NCBI_API_KEY=<key>; nohup python3 -u query_pmc_entrez.py > <log> 2>&1 &` | `[LOG]` `pmc_metadata_log.txt` (May); session log 2026-09-23 |
| R2 | `nohup python3 fetch_s3_pmc.sh > LCS/modern_download.log 2>&1 &` | `[LOG]` |
| R3 | `python3 isolate_delta.py` then `sbatch submit_delta_fetch.sh` | `[LOG]` `slurm-21295326.out` |
| F2 | same command as F1 | `[LOG]` RL-041 |
| F3 | `python3 pmc_preprocessing/author_rules_expansion.py`, then `... --apply` | `[LOG]` RL-043 |
| F4 | `python3 pmc_preprocessing/build_exclusion_list.py` | `[LOG]` RL-045 |
| F1 | `python3 pmc_preprocessing/article_blacklist.py --workers 8` (first run also `--out` and `--compare` to test; refuses to overwrite its output) | `[LOG]` RL-041 |
| P0 | `python3 pmc_preprocessing/build_content_regions.py` (writes `PP/content_regions_v2.csv`; refuses to overwrite) | `[LOG]` RL-025 |
| P1 | `nohup python3 pmc_preprocessing/citation_standartization_soft_masking.py > PP/masking_execution.log 2>&1 &` | `[LOG]` |
| P1 (v2) | `python3 pmc_preprocessing/citation_masking_v2.py --build-lexicon` (once), `--selftest`, then `python3 pmc_preprocessing/citation_masking_v2.py --workers 8` (6 minutes on `incline`; refuses to overwrite its outputs) | `[LOG]` RL-051, RL-053, RL-054 |
| P1b | `python3 pmc_preprocessing/nonprose_removal_v1.py --selftest`, then `python3 pmc_preprocessing/nonprose_removal_v1.py --workers 8` | `[LOG]` RL-056, RL-057 (tests only) |
| P2 | `nohup python3 pmc_preprocessing/focal_window_extraction.py > PP/extraction.log 2>&1 &` | `[LOG]` |
| P2d | `python3 pmc_preprocessing/exclusion_diagnostics.py` | `[MTIME]` |
| P3–P5 | `python3 pmc_preprocessing/citation_resolution_{1_inventory,2_api,3_translate}.py` in order | `[LOG]` `phase2.log`, `phase3.log` (P4, P5) |
| P6, P7 | `python3 pmc_preprocessing/citation_resolution_4_coref.py`, then `..._4b_restore.py` | `[LOG]` `phase4.log`; P7 `[MTIME]` |
| G1 | `python3 phase5_graphbrain/scripts/01_matrix_partition.py` | `[MTIME]` |
| G2 | `sbatch phase5_graphbrain/scripts/submit_v2.sh` | `[LOG]` `PG/scripts/logs_v2/node_15772611_*` |
| G3 | `sbatch phase5_graphbrain/scripts/submit_4h.sh` | `[LOG]` `PG/logs/cluster_*.log` |

## Appendix C — script inventory (sha256, first 12 hex characters, as of 2026-09-24)

| File (under `R`) | sha256 | mtime |
|---|---|---|
| `query_pmc_entrez.py` | 1d057b072f50 | 2026-09-23 13:47 |
| `fetch_s3_pmc.sh` | d48bfcb30b94 | 2026-05-09 18:33 |
| `isolate_delta.py` | 8b3877541844 | 2026-05-14 01:39 |
| `fetch_delta_pmc.py` | b620c1b961d8 | 2026-05-14 15:38 |
| `submit_delta_fetch.sh` | 69545e630577 | 2026-09-24 13:15 |
| `pmc_preprocessing/article_blacklist.py` | 27f6f58fec51 | 2026-09-25 18:34 |
| `pmc_preprocessing/author_rules_expansion.py` | 54ac879abe13 | 2026-09-25 18:57 |
| `pmc_preprocessing/build_exclusion_list.py` | fb8362059530 | 2026-09-25 19:06 |
| `pmc_preprocessing/citation_masking_v2.py` | 2d378caabeb6 | 2026-09-25 21:23 |
| `pmc_preprocessing/citation_lexicon_v1.json` (input of P1 v2: 15,487 ordinary words, 2,729 names; built by `--build-lexicon`) | da5024dfdaa3 | 2026-09-25 20:37 |
| `pmc_preprocessing/nonprose_removal_v1.py` | 282bc1d24475 | 2026-09-25 22:26 |
| `pmc_preprocessing/build_content_regions.py` | 64e959d673a0 | 2026-09-25 20:33 |
| `pmc_preprocessing/citation_standartization_soft_masking.py` | 7acf961258e6 | 2026-05-16 01:03 |
| `pmc_preprocessing/citation_standartization_soft_masking.sh` | dfb1035f90b0 | 2026-05-16 01:05 |
| `pmc_preprocessing/focal_window_extraction.py` | a8b50a542e10 | 2026-09-23 13:31 |
| `pmc_preprocessing/exclusion_diagnostics.py` | bed02af9a327 | 2026-05-17 00:43 |
| `pmc_preprocessing/citation_resolution_1_inventory.py` | fa97765de2aa | 2026-05-18 19:21 |
| `pmc_preprocessing/citation_resolution_2_api.py` | 07ae408249a3 | 2026-05-18 19:26 |
| `pmc_preprocessing/citation_resolution_3_translate.py` | 84f9c722ad25 | 2026-05-18 19:35 |
| `pmc_preprocessing/citation_resolution_4_coref.py` | da9d09307d8f | 2026-05-19 20:01 |
| `pmc_preprocessing/citation_resolution_4b_restore.py` | 861b71001a65 | 2026-05-20 21:58 |
| `pmc_preprocessing/focal_words.txt` | 5a163624bf28 | 2026-09-23 13:31 |
| `phase5_graphbrain/scripts/01_matrix_partition.py` | 89be16b155f0 | 2026-05-21 00:35 |
| `phase5_graphbrain/scripts/02_transformer_node.py` | cf8c8b02e28b | 2026-05-21 00:37 |
| `phase5_graphbrain/scripts/submit_array.sh` | 55b80cddd73e | 2026-05-21 00:51 |
| `phase5_graphbrain/scripts/parse_stage_v2.py` | 086a5d4237af | 2026-05-31 18:29 |
| `phase5_graphbrain/scripts/submit_v2.sh` | f8021d0db570 | 2026-05-23 20:27 |
| `phase5_graphbrain/scripts/chunk_4h_hpc.py` | dfc6cc5e9662 | 2026-05-29 23:36 |
| `phase5_graphbrain/scripts/submit_4h.sh` | cc13dba06e28 | 2026-05-30 19:19 |
| `phase5_graphbrain/nltk_abridged_stopwords_list.txt` | 2b6c7d9fdae9 | 2026-06-29 21:42 |
