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
| P1b | Non-prose removal: tables, LaTeX, formulas and statistics are removed from the masked text so that graphbrain reads only prose. One output line per input line (removed lines become empty), so alignment with the raw text is kept. Rules T1 to T9 (tables, captions, LaTeX, formulas, statistics, symbols, hexadecimal codes) are listed in the script's docstring | `python3 pmc_preprocessing/nonprose_removal_v1.py --selftest`, then `python3 pmc_preprocessing/nonprose_removal_v1.py --workers 8` (about 1 minute; refuses to overwrite its output) | `LCS/masked_corpus_v2/`, `PP/content_regions_v2.csv`, `LCS/pure_text_corpus/` (tab-separated cells of the raw lines, because the masked files have lost their tabs) | `LCS/clean_corpus_v1/` (same file names), `LCS/nonprose_removal_v1_summary.json`, `LCS/nonprose_examples_v1.jsonl` | run 2026-09-25 on the whole whitelist (39,299 articles, 47,619 files; 9.4% of the words removed; no line moved), RL-060 | DONE-current |
| P1c | Three more non-prose rules on top of the P1b output (a complement, P1b stays as it was): T10 abbreviation lists ("ABBR: meaning; ...", also "Abbreviations:" lines), T11 "Alt text:" figure descriptions, T12 table footnotes (lines that follow a table block and start like a footnote: "Note:", "Data are presented as ...", "X indicates ...", a footnote sign). One output line per input line | `python3 pmc_preprocessing/nonprose_extra_rules_v1.py --selftest`, then `python3 pmc_preprocessing/nonprose_extra_rules_v1.py --workers 8` (about 5 minutes; refuses to overwrite its output) | `LCS/clean_corpus_v1/`, `LCS/pure_text_corpus/` (raw table rows), `PP/content_regions_v2.csv` | `LCS/clean_corpus_v2/` (same file names), `LCS/nonprose_extra_v1_summary.json`, `LCS/nonprose_extra_examples_v1.jsonl` | 2026-09-26 on the whole whitelist (39,299 articles, 47,619 files; 23,228 lines blanked, 0.21% of the words), RL-062 | DONE-current |
| P2 | **v2 (2026-09-26).** Focal sentences with the sentence before and after, from the cleaned region files. Sentences are split line by line (sciSpaCy `en_core_sci_sm`); heading lines are never sentences and stop a window; touching selected sentences form one block, so no sentence is emitted twice; every sentence has an id `PMCID.rN.L<raw line>.S<k>` and a 12-character hash of id and text (rN = region window, r1 the main window). Word list and guard rules (exact capitalisation, homonym patterns, article evidence for ambiguous names) are in `focal_terms.py`; every article gets a status row and errors are written to their own file. Supersedes `focal_window_extraction.py` (v1) | `python3 pmc_preprocessing/focal_terms.py --selftest`, then `python3 pmc_preprocessing/focal_extraction_v2.py --workers 8` (about 14 minutes; refuses to overwrite its output; needs the scratch numpy 1.x, item 17) | `LCS/clean_corpus_v2/`, `LCS/article_whitelist.txt`, `PP/content_regions_v2.csv`, `PP/focal_terms.py` | `LCS/focal_extractions_v2.jsonl` (34,662 articles, 268,532 blocks, 1,297,393 sentences), `LCS/focal_status_v2.csv`, `LCS/focal_errors_v2.jsonl` (empty), `LCS/focal_extraction_v2_summary.json` | 2026-09-26 on the whole whitelist, RL-064 | DONE-current |
| P2d | Diagnostic: why documents produced no focal window | `python3 exclusion_diagnostics.py` | P0 csv, `LCS/pure_text_corpus/`, `LCS/focal_extractions_v1.jsonl` | `PP/exclusion_report.csv` (5,489 rows) | 2026-05-17 | DONE (diagnostic, not a chain input; uses an outdated hardcoded word list) |
| P3 | Inventory of citation tokens found inside the extracted windows **Superseded by the dictionary of cited works written by P1 v2 (D9).** | `python3 citation_resolution_1_inventory.py` | `LCS/focal_extractions_v1.jsonl`, `PP/citation_vault_light_masking_v1.jsonl` | `PP/api_inventory_target.json` (13,271 documents) | 2026-05-18 | DONE-stale |
| P4 | Resolve tokens through Entrez `efetch` (batches of 200, 0.35 s apart, no API key); positional index for numeric citations, year match for author-year; unresolved tokens get `__REF_HASH_<sha256[:10]>__` **Superseded by the dictionary of cited works written by P1 v2 (D9).** | `python3 citation_resolution_2_api.py` | `PP/api_inventory_target.json`, NCBI Entrez | `PP/global_translation_dictionary.json` (105,389 tokens), optional `PP/phase2_api_errors.log` | 2026-05-18 | DONE-stale |
| P5 | Replace per-document tokens with the global reference ids **Superseded by the dictionary of cited works written by P1 v2 (D9).** | `python3 citation_resolution_3_translate.py` | `PP/global_translation_dictionary.json`, `LCS/focal_extractions_v1.jsonl` | `LCS/focal_extractions_v2_resolved.jsonl` (22,795) | 2026-05-18 | DONE-stale |
| P6 | **v1, superseded by P6 v2 (D13).** Coreference resolution (fastcoref `LingMessCoref`, CUDA if available) and re-check that each block still contains a focal word | `python3 citation_resolution_4_coref.py` | `LCS/focal_extractions_v2_resolved.jsonl`, `PP/focal_words.txt` | `LCS/focal_extractions_v3_graphbrain_ready.jsonl` (22,572 kept, 223 dropped); `PP/phase4_document_errors.log` is created only if documents fail (none did) | 2026-05-19/20 | DONE-stale |
| P7 | **Superseded by P6 v2 (D13): no article can lose its focal word when only pronouns are replaced.** Restore the documents dropped in P6, unmutated, flagged `restored_from_v2` | `python3 citation_resolution_4b_restore.py` | `LCS/focal_extractions_v2_resolved.jsonl`, v3 | appends to v3: **223** records; v3 then has 22,795 unique PMCIDs | 2026-05-20 | DONE-stale |
| P6 (v2) | **Built 2026-09-26.** Pronoun coreference resolution on the P2 v2 blocks. spaCy `en_coreference_web_trf` proposes clusters, LingMess (fastcoref) must agree; the dependency parser `en_core_web_trf` gives the core element of the antecedent. Rules in D13. Only pronouns inside a block are changed; the two sentences before the block are given to the models as context and never written out. Sentence ids, hashes, raw lines and text of P2 are kept; each sentence gains `text_resolved`, `hash_resolved` (same formula as the P2 hash) and a list of replacements. Supersedes P6 v1 and P7 | on CSF: `sbatch pmc_preprocessing/submit_coref_v2.sh` (array of 100 shards, 4 cores, 12 h, each shard resumable), then `python pmc_preprocessing/coref_resolution_v2.py --merge --nshards 100` (refuses unless every article is present once with status ok); needs `PYTHONPATH=$HOME/np1_for_spacy` (item 17) | `LCS/focal_extractions_v2.jsonl`, `LCS/clean_corpus_v2/` (context sentences only), `PP/content_regions_v2.csv`, `PP/focal_terms.py` | `LCS/coref_v2/`: `coref_resolved_v2.jsonl`, `coref_np_links_v2.jsonl` (noun-phrase links, metadata only), `coref_rejected_v2.jsonl` (rejected replacements with reason), `coref_status_v2.jsonl`, `coref_errors_v2.jsonl`, `coref_v2_summary.json` (per-shard files `*.shardKKKofNNN.jsonl` before the merge) | run 2026-09-27/28 on the whole whitelist (34,662 of 39,299 articles produced a resolved record; 4,637 had no focal sentence at P2 and are absent here too), CSF jobs 21401638 + 21489427 (retry of shard 86, OOM at 16 GB, reran at 32 GB), merged and verified on a 1% sample, RL-065 to RL-067 | DONE-current |
| P7 (v2) | **Run 2026-09-28, rerun after a DOI-cleaning fix (RL-069).** Focal sentences for graphbrain with one id per cited work across articles. Keeps a sentence only when it holds a focal term (focal in P2, or made focal by a coreference replacement); every citation token of the article inside a kept sentence is replaced by a shared id `REF<6 digits>`: identified by DOIs and PMCIDs read from the full raw reference line, else by same first author, year within one, and title-word overlap (Jaccard >= 0.6); a DOI that is a truncated form of another is not used, and only a PMID glued to a bioRxiv/medRxiv DOI is cut from a DOI (IEEE and ACM DOIs end in ".NNNNNNN" by themselves); an unresolved or ambiguous token gets an id of its own and is never shared. Rules in D14. Replaces P7 v1 | `python3 pmc_preprocessing/focal_citations_v2.py --selftest`, then `python3 pmc_preprocessing/focal_citations_v2.py` (29 s on incline, no models; refuses to overwrite its output) | `LCS/coref_v2/coref_resolved_v2.jsonl`, `LCS/citation_dictionary_v2.jsonl`, `LCS/pure_text_corpus/` (reference lines), `PP/content_regions_v2.csv`, `PP/focal_terms.py` | `LCS/focal_sentences_v2.jsonl` (34,662 articles, 702,948 sentences, each with P2 id and hash, `text_final`, `hash_final`, citations), `LCS/citation_works_v2.jsonl` (10,555 works, one record each with all its articles, tokens, aliases and reference lines), `LCS/focal_citations_v2_summary.json` | 2026-09-28 on the whole corpus, RL-068 (superseded), RL-069 | DONE-current |
| G2 (v3) | **Run 2026-09-29 (RL-074 to RL-077).** Rerun of G2 with corrected sentence units and the other fixes agreed with the owner (item 30's scope paragraph): unit ids and hashes, the punctuation sentence-end rule (replaces G2 v2's whole-sentence-per-call units), glued-citation removal, non-prose removal (captions, DOI captions, caption bodies, reference lines, short headings), URL/DOI tokenisation, symbol normalisation, and hyphen joining (kept apart where a hyphenated word contains a focal term). Merging is a separate script from the parser (RL-076), for reproducibility. Replaces G2 v2 | CSF test `sbatch PG/scripts/check_csf_g2v3.sh` (RL-074/RL-075, ALL G2 v3 CHECKS OK); then `sbatch PG/scripts/submit_g2_v3.sh` (array 0-49, `multicore`, 4 CPUs, 8 GB, 6 h, job 21557322 — only shard 49 finished under this array job itself, the other 49 each finished under a separately submitted job id, cause not diagnosed, RL-077); then on incline `python3 PG/scripts/merge_g2_v3.py --nshards 50` (38 min, re-derives every check from the raw input and the shard files rather than trusting the shards' own done files beyond a consistency check) | `LCS/focal_sentences_v2.jsonl` | `PG/g2_v3/shards/` (per-shard `g2_parsed_NNN.jsonl`, `g2_errors_NNN.jsonl`, `g2_done_NNN.json`, `db_provenance_NNN.sqlite`, `url_table_NNN.json`); after merge `PG/g2_v3/g2_parsed_v3.jsonl` (34,662 articles, 2.2 GB), `PG/g2_v3/g2_v3_summary.json`, `PG/g2_v3/url_table_v3.json` (1,674 tokens) | 2026-09-29, jobs 21557322 + the 49 individually-submitted job ids in `g2_v3_summary.json`, RL-077 | DONE-current; 702,948 sentences (696,351 ok, 6,591 non-prose, 6 empty), 710,869 units, max depth 101, 1 error line (one pathological long-list sentence, not a defect) |
| G2 (v2) | **Superseded by G2 v3 (item 30, RL-077).** Run 2026-09-28/29 (RL-072).** Graphbrain parse (`en_core_web_trf`, lemmas) of every focal sentence of P7 v2, one sentence per parser call (`text_final`), so every edge keeps its sentence id and hash; graphbrain's own splitter turns a few sentences into two units, all kept with an index. Graphbrain failures (no edge) and exceptions are counted and written out instead of skipped; articles go to shard i by input position (position % 50); resumable; checked merge. Rules in D15. Replaces G1 and G2 v1 | CSF check `sbatch PG/scripts/check_csf_g2.sh` (job 21504977, ALL OK); then `sbatch PG/scripts/submit_g2_v2.sh` (array 0-49, `multicore`, 4 CPUs, 8 GB, 6 h, job 21506341, 50 tasks 22:26-22:54 2026-09-28) runs `graphbrain_parse_v2.py --shard i --nshards 50`; then on incline `PYTHONPATH=$HOME/np1_for_spacy python PG/scripts/graphbrain_parse_v2.py --merge --nshards 50` (32m35s, checks the JSONL against the input and each database against its done file) | `LCS/focal_sentences_v2.jsonl` | `PG/g2_v2/shards/db_provenance_NNN.sqlite` (50 files, 118 GB total; graphbrain database per shard in the May form that G3 reads: `('source', PMCID, main_edge)` and lemma edges; each source edge has the attribute `occurrences` with sid, hash_final, unit, text and atom-to-word positions), `g2_parsed_NNN.jsonl` (same content as text, the task's checkpoint), `g2_errors_NNN.jsonl` (1,734 lines total), `g2_done_NNN.json`; after merge `PG/g2_v2/g2_parsed_v2.jsonl` (34,662 articles, 1.8 GB; one record per article: sentences with sid, hash_final, status, units with main_edge and lemma edges as graphbrain strings), `PG/g2_v2/g2_v2_summary.json` | 2026-09-28/29, jobs 21504977 (check), 21506341 + 49 sibling array-task job ids (full run), RL-072 | DONE-stale; **superseded by G2 v3 above (item 30, RL-077)** |
| G1 | **Superseded — no current stage reads its output (item 30 lineage check, 2026-09-29).** Split v3 into 50 shards of ≈456 records, a pre-partitioning step for the plain "G2" stage below. G2 v2 and G2 v3 shard internally (`position % nshards` on the input JSONL) and never read `PG/input_shards/`, so this step has no consumer in the current chain | `python3 01_matrix_partition.py` | `LCS/focal_extractions_v3_graphbrain_ready.jsonl` | `PG/input_shards/shard_01..50.jsonl` (22,795 lines in total) | 2026-05-21 | DONE-stale; **superseded, fed only the plain G2 stage below** |
| G2 | **Superseded by G2 v2 then G2 v3 (item 30, RL-072/RL-077); its own output is what G3 below still (stalely) reads — G3 has not yet been rerun against G2 v2 or G2 v3.** Graphbrain parsing of each shard (`en_core_web_trf`); each sentence stored as `('source', pmcid, main_edge)` so that every edge stays tied to its article | `sbatch submit_v2.sh` (array 1-50%15, `multicore`, 4 CPUs, 16 GB, 24 h) runs `parse_stage_v2.py` | `PG/input_shards/` | `PG/output_sqlite_v2/db_provenance_NN.sqlite` (50 files, 83 GB), logs in `PG/scripts/logs_v2/` | 2026-06-01, array job 15772611 | DONE-stale; **superseded by G2 v2/v3 above; still the only G2 output G3 has actually consumed** |
| G3 | Curation: writes `source_core`, `source_periphery`, `source_fringe` links | `sbatch submit_4h.sh` (array 1-500 of which only 1–50 do work, `serial`, 1 core, 12 h) runs `chunk_4h_hpc.py <db>` | `PG/output_sqlite_v2/db_provenance_NN.sqlite`, `PP/focal_words.txt`, `PG/nltk_abridged_stopwords_list.txt`, spaCy `en_core_web_sm` | `PG/postprocessed_output/db_curated_NN.sqlite` (50 files, 2.9 GB), logs `PG/logs/cluster_<task>.log` | 2026-05-30 | DONE-stale; **older than its raw db in 50 of 50 shards** (§6); **curated only 359 of the 22,601 parsed articles** (checked 2026-09-28, RL-071): the loop over the raw database crashes with a RecursionError on the first very deep edge (whole-block parses nested up to 383 levels), outside the per-sentence error handling. Must be fixed before G3 is rerun; **revision planned, item 29** |
| M1 | Matrix builder for the full corpus: turns the curated links into relation matrices. Needs article metadata and authorship edges, which do not yet exist at full scale | — | `PG/postprocessed_output/` | — | — | NOT-BUILT |
| A1 | Solver and evaluation on the full-scale matrices | — | M1 output | — | — | NOT-RUN |

Where several stages read `PP/focal_words.txt`: **only P6 and G3 do** (since 2026-09-26 the file is generated from `focal_terms.py`, which P2 v2 uses;
P2d still has its own old list, §5 D3).

## 2. Data flow

Current chain (whitelisted articles only; each arrow is a stage of §1, all run 2026-09-25 to 2026-09-28):

```
R1 target_pmcids.txt ──► R2/R3 pure_text_corpus/ (46,242 files)
        │
        ├─► P0 content_regions_v2.csv ─┐          F1/F2 article_blacklist.csv ◄─ PubMed types, titles, authors (F3 appended)
        │                              ▼                          │
        │                       F4 article_exclusions.csv + article_whitelist.txt (39,299 articles)
        ▼                              │
P1 v2  masked_corpus_v2/  (47,619 region files) + citation_dictionary_v2.jsonl
        ▼
P1b    clean_corpus_v1/ ──► P1c clean_corpus_v2/            (tables, formulas, statistics, abbreviation lists, footnotes removed)
        ▼
P2 v2  focal_extractions_v2.jsonl (34,662 articles, 268,532 blocks; sentence ids and hashes)   ◄─ focal_terms.py
        ▼
P6 v2  coref_v2/coref_resolved_v2.jsonl (+ noun-phrase links, rejected replacements)           ◄─ 2 context sentences from clean_corpus_v2
        ▼
P7 v2  focal_sentences_v2.jsonl (702,948 focal sentences, text_final) + citation_works_v2.jsonl (10,555 works)
        ▼
G2 v2 graphbrain parse g2_v2/shards/db_provenance_NNN.sqlite (superseded) ──► G2 v3 rerun with sentence units (done, item 30) g2_v3/g2_parsed_v3.jsonl (34,662 articles, 710,869 units) ──► G3 curation (revision planned, item 29) ──► M1 (not built) ──► A1
```

Legacy May chain (superseded, kept for the record; its outputs describe the old 22k corpus and are stale, §6 item 2):

```
R1 ──► R2/R3 ──► P0 sequence_metadata_relaxed.csv (GAP) ──► P1 masked_corpus_v1/ ──► P2 focal_extractions_v1.jsonl ──► P2d exclusion_report.csv
   P3 api_inventory_target.json ──► P4 global_translation_dictionary.json ──► P5 focal_extractions_v2_resolved.jsonl
   ──► P6 (+P7) focal_extractions_v3_graphbrain_ready.jsonl ──► G1 input_shards/ ──► G2 output_sqlite_v2/ ──► G3 postprocessed_output/
```

## 3. Corpus funnel (counts re-derived 2026-09-24)

| Point | Count | Note |
|---|---|---|
| Target list, 2026-05-13 | 34,800 | superseded; backup `LCS/target_pmcids.pre_llm_terms_backup_20260923_133436.txt` |
| Target list, 2026-09-23 | 46,241 | current |
| Full texts on disk | 46,242 | 46,177 of the current targets, plus 65 files not in the current list; 34,751 before the 2026-09-24 delta |
| Current targets without a text | 64 | 22 exist in the bucket under version numbers outside 1–4; 42 have no folder in the bucket [checked with `aws s3 ls`] |
| Rows in P0 csv, May version | 31,511 | superseded by the rebuilt P0 (row below); covered only the May-era texts |
| Blacklisted (F1 to F3, 2026-09-25) | 2,799 | of 46,177 texts, leaving 43,378: F1/F2 2,683 and F3 116 (author rules); reasons in the F1 and F3 rows; the list is meant to grow |
| Whitelist after F4 (2026-09-25) | 39,299 | 46,177 minus 2,799 blacklisted minus 4,079 excluded by P0 structure (no closing heading 2,484, headingless 1,234, no end marker 361); file `LCS/article_whitelist.txt` |
| Masked by P1 v2 (2026-09-25) | 39,299 articles, 47,619 regions | one masked file per region of every whitelisted article; `LCS/masked_corpus_v2/` |
| Rows in `content_regions_v2.csv` (P0, 2026-09-24) | 46,177 | `include`=1: 41,243; no closing heading 2,796; headingless 1,635; no end marker 503 |
| Documents masked (P1) | 28,284 | rows with both content boundaries |
| Lost between texts and P1 | 6,467 | 34,751 − 28,284 `[DERIVED]`: 3,240 without a P0 row plus 3,227 without boundaries |
| Cleaned by P1b then P1c (2026-09-25/26) | 39,299 articles, 47,619 regions | `LCS/clean_corpus_v1/`, then `LCS/clean_corpus_v2/`; P1c blanked 23,228 lines |
| Articles with focal sentences, P2 v2 (2026-09-26) | 34,662 | of 39,299; 4,637 without (item 24); 268,532 blocks, 1,297,393 sentences (684,366 focal); file `LCS/focal_extractions_v2.jsonl` |
| Articles with a resolved record, P6 v2 (2026-09-27/28) | 34,662 | same articles as P2 v2; 203,493 of 285,941 personal pronouns replaced (11,445 with an antecedent in the context sentences), 48,906 rejected, 34,025 demonstratives left, 340,271 noun-phrase links; `LCS/coref_v2/` |
| Focal sentences after P7 v2 (2026-09-28) | 702,948 | of 1,297,393; 34,662 articles; 12,644 citation tokens in 14,154 places became 10,555 works (5,696 identified by DOI or PMCID, 2,644 by reference text only, 2,215 unresolved), 964 cited in two or more articles; `LCS/focal_sentences_v2.jsonl`, `LCS/citation_works_v2.jsonl` (after the DOI fix of RL-069) |
| Documents with focal windows (P2 v1, May) | 22,795 | superseded; 5,489 without = exactly the rows of `exclusion_report.csv` |
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
- **P6/P7 (v1, superseded).** P6 re-filtered after coreference; 223 documents lost their focal word and were
  restored by P7 without mutation. The `restored_from_v2` flag identifies them.
- **P6 v2 (CSF).** One array task per shard (`submit_coref_v2.sh`, 100 shards, 4 cores, 12 h), every task with `--resume`; a task that fails is resubmitted alone with
  `sbatch --array=<id> ...`. **Memory:** the largest article (a block of 2,009 words) needs about 17.5 GB, the default 16 GB is too little for it: shard 86 was killed and rerun with
  `--mem=32G` (RL-067); give tasks that die with exit 137 `--mem=32G`. The merge (`--merge --nshards 100`) refuses unless every article is present once with status ok, and the per-shard
  files can be deleted after a byte comparison with the merged files (done, RL-067 follow-up).
- **P7 v2.** Runs on incline in about 30 s, no models. It refuses to overwrite its outputs; to rerun, rename the three outputs first (as done for RL-069). The DOI fix of RL-069 is
  in the script; the outputs of RL-068 are kept as `*.before_doi_fix_20260928` and must not be used.
- **G2 v2.** Run 2026-09-28/29 on the whole corpus (RL-072): 34,662 articles, 702,948 sentences, 730,292 parse units (701,237 ok, 1,711 partial, 0 exceptions), 725,513 database source edges, maximum edge depth 93 across all shards. Each CSF task ran about 25-35 minutes (`multicore`, 4 CPUs, 8 GB; measured 0.14-0.26 s per sentence on a CSF node vs 0.1 s on incline). A failed array task is resubmitted alone (`sbatch --array=<i> ...`) and resumes after its last complete article; a finished task (with `g2_done_NNN.json`) does nothing when resubmitted. Needs the numpy-1 workaround (§ environment). Graphbrain 0.7 needs `parser.atom2token = {}` before each call (the attribute is None by default and would otherwise collect every token of the run).
- **G2 (v1/v2 of May).** The v1 parse (`02_transformer_node.py`, `submit_array.sh`, `PG/output_sqlite/`, job
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
| D3 | 2026-09-26 (was open) | The focal-word list had three unsynchronised sources. Now `pmc_preprocessing/focal_terms.py` is the source for P2 v2, and `focal_words.txt` (read by P6 and G3) is generated from it with `python3 focal_terms.py --write-flat-list`. `exclusion_diagnostics.py` (P2d) still carries its own old list. The flat file is looser than P2 (P6 and G3 match it case-insensitively without the guard rules, item 24). | P2, P6, G3 |
| D4 | open (found 2026-09-24) | Fetch loops try versions 1–4 only. 22 current targets exist under other versions (`.319` ×15, `.358` ×2, `.5` ×4, `.7`, `.8`). | R2, R3 |
| D5 | 2026-09-24 | New SLURM wrapper `submit_delta_fetch.sh` for the delta fetch; the older wrapper `run_extraction.sh` runs a different, crashed prototype and must not be used. | R3 |
| D6 | 2026-05-23 (historical) | Graphbrain parse changed from v1 (no provenance) to v2 (`('source', pmcid, main_edge)`). | G2 |
| D7 | 2026-05-20 (historical) | Coreference filter drops documents whose focal word vanished; patched by restoring them (P7). Not needed from P6 v2 on (D13). | P6, P7 |
| D8 | 2026-09-24 | P0 rebuilt as `build_content_regions.py` (new output `content_regions_v2.csv`; May CSV kept). Start = first Introduction heading, else the `U+009F`-framed `=` divider. End = first back-matter heading after the last body-type heading, else References, else end of file. Window 2+ = whitelisted Supplementary Information (prose only) and Ethics statements, same PMCID. Reason: the May end rule cut articles short (Supplementary Information inside the abstract block; Ethical Considerations inside Methods). | P0; P1 onward stale |
| D9 | 2026-09-25 | P1 rebuilt as `citation_masking_v2.py`. Numeric citations and parenthetical author-year citations are deleted from the text; narrative author-year citations become `a<PMCID digits>r<n>`, with the same n for the same work in an article (its position in the article's reference list when it can be matched, otherwise numbered on from the size of the list); a dictionary of cited works with aliases, mention counts and matched reference text is written (it has a DOI field, but the field is empty in every entry because of a bug in the DOI pattern, item 27); line breaks are kept. P3 to P5 (citation resolution) are superseded. Reasons: the old masks broke graphbrain parses and over-masked (§6 items 15 and 16), and numeric citations cannot be resolved when the reference list is not visible. Sequel: P2 must read `masked_corpus_v2` and its region files. | P1 to P7 |
| D10 | 2026-09-25 | P0: extra windows (supplementary, ethics) end at the next heading of any kind, 60 lines after their heading, or the References heading, whichever comes first. Before, they ended only at the next back-matter heading, so an ethics window could run for hundreds of lines (longest 8,679; 973 windows over 60 lines) and, without a References heading, to the end of the file. Main windows, statuses and the whitelist are unchanged (checked for all 46,177 articles); extra windows fell from 10,485 to 9,120. | P0, P1 |
| D11 | 2026-09-26 | Non-prose removal is two layers: P1b (unchanged, output `clean_corpus_v1/`) and P1c (`nonprose_extra_rules_v1.py`, reads `clean_corpus_v1/`, writes `clean_corpus_v2/`). P1c was added after a check of the cleaned corpus found lines with no rule to remove them: about 2,400 abbreviation lists, about 1,500 table footnotes and 246 "Alt text:" descriptions. Later stages read `clean_corpus_v2/`. P1c changes lines only to empty, so line alignment with the raw text is unchanged. | P1c, P2 |
| D12 | 2026-09-26 | P2 rewritten as `focal_extraction_v2.py` + `focal_terms.py` (the v1 script `focal_window_extraction.py` is kept and not used). Changes: reads the cleaned region files instead of the whole masked text; line-by-line sentence splitting; headings excluded; sentence ids and hashes; per-article status and an error file instead of silent skips; no overwrite of earlier output. Word list (owner decisions 2026-09-26): added bare BERT, RoBERTa, GPT, Gemini, Bard, Llama/LLaMA (any hyphen or spelling), Mistral, PaLM, Gemma, "language model(s)", GPT-2, GPT4, "Chat GPT"; not added: transformer, foundation models, chatbots, Copilot (bare), generative AI. Ambiguous names (Claude, Grok, RITA, GPN, KeAP, GROVER, Galactica, KPGT and other short model acronyms) and the bare names GPT, Gemini, Bard, Llama, Mistral, PaLM, Gemma count only with exact capitalisation, no homonym pattern nearby (Claude Bernard, Grover's quantum search, the liver enzyme GPT, ...) and article evidence (the article has an unambiguous LLM or model term); no nearby-word shortcut (strict version). No cap on block length. `focal_words.txt` regenerated (185 to 208 terms; old file kept as `focal_words.before_v2_20260926.txt`). | P2, P6, G3 |
| D13 | 2026-09-26 | P6 rewritten as `coref_resolution_v2.py` (owner decisions 2026-09-26). (1) Coreference model: spaCy `en_coreference_web_trf` (the pipeline graphbrain integrates) instead of LingMess alone; LingMess is kept as a second opinion and a replacement is made only when both models link the pronoun to the same antecedent. (2) Only third-person personal pronouns (it, its, itself, they, them, their, theirs, themselves, he, him, his, she, her, hers, himself, herself) are replaced; v1 replaced every later mention, including noun phrases, with the first mention's text (in 143 test blocks 63% of its replacements were noun phrases and 25 of 235 used a pronoun as the replacement). (3) The replacement is the core element of the antecedent, not the whole mention: appositions, relative and participial clauses, parentheticals and prepositional tails are dropped ("ChatGPT, one of the most well-known LLMs today" gives "ChatGPT"); a reversed apposition gives the name; a coordination is kept whole up to 10 words; possessives get 's; a demonstrative determiner becomes "the"; a core is accepted only when its head is a noun, and partitives ("two of the models") are rejected. The antecedent is the cluster mention with a focal term, else a proper noun, else the earliest. (4) Number agreement: a plural pronoun needs a plural antecedent and a singular one a singular antecedent. (5) **Demonstratives (this, these, that, those) are never resolved**: they usually refer to a clause or a finding, and the model linked them to verbs in most of 15 test cases; they are counted per article. (6) **Noun phrases ("the model", "the chatbot") are never replaced**: their wording is perception data, and the models merge them too eagerly (three evaluated systems linked to one); their links go to a separate file with a LingMess agreement flag. (7) Context: the two sentences before the block (from the P1/P1b/P1c-cleaned region file, split with P2's splitter, stopped by a heading) are given to the models and never written out; a replacement whose antecedent lies there records its raw line. (8) Every rejected replacement is written with its reason. (9) P7's restore step is no longer needed. (10) Runs as a sharded array job with resume and a checked merge. | P6, P7 |
| D14 | 2026-09-28 | Owner decisions for the input of graphbrain (G2). (1) Only sentences that contain a focal term are kept: 702,948 of 1,297,393 (18,582 of them only through a coreference replacement); the context sentences that P2 added, whose job was resolving pronouns in P6, are dropped. (2) One string per cited work across articles (`REF000131` etc., uppercase: graphbrain reads it as one proper-noun atom, lowercase variants flip between proper and common noun). Unresolved citations keep an id of their own (owner: acceptable). Only works cited in kept sentences get an id (10,555 of 412,616 in the P1 v2 dictionary); the old dictionary is not modified, `citation_works_v2.jsonl` is the new global dictionary. (3) DOIs are extracted from the full raw reference lines with a corrected pattern (cleaning rules in RL-069), because the P1 v2 `doi` field is empty (item 27); this reverses part of D9's claim that the P1 v2 dictionary replaces P3 to P5: for cross-article identity it does not, P7 v2 does. | P1, P7 |
| D15 | 2026-09-28 | G2 rewritten as `graphbrain_parse_v2.py` (the May `parse_stage_v2.py` is kept and not used). Defects of the May script: (1) it parsed whole blocks, so graphbrain re-split them and each edge was tied only to the article (`('source', pmcid, edge)`), not to a sentence; blocks over 512 transformer tokens produced a truncation warning; (2) sentences graphbrain failed on (main_edge None, including the exceptions graphbrain catches internally) were skipped without a count; (3) `parser.atom2token` was set to {} once and never cleared, so it kept every spaCy token of the task in memory; (4) progress was logged per article after the edges were written, without a check of the finished shard. New: input is P7 v2 (`text_final`, focal sentences only, shared citation ids), one sentence per call, sid and hash_final kept, status per sentence (ok / partial / no_edge / exception) and an error file, edges stored as graphbrain strings in JSONL (hedge(string) restores them; checked for every edge), shards by input position with resume and a checked merge. Output (revised the same day at the owner's request): a graphbrain database per shard in the form G3 reads (`('source', PMCID, main_edge)` plus lemma edges), each source edge with the attribute `occurrences` (sid, hash_final, unit, text, atom-to-word positions), plus the same content as JSONL (checkpoint for resume, and the merged flat record). The database is built on the node's local disk in one transaction and copied to the RDS drive, because in May all 826 errors of G2 were SQLite `disk I/O error`s on the RDS drive (shard 48, 195 articles lost blocks). G1 (separate partition step) is no longer needed. Checked against the May outputs (RL-071): the May G2 parsed 22,601 of 22,795 articles (the 194 missing all in shard 48, disk I/O errors), but the May G3 curated only 359 articles because it crashed on edges nested up to 383 levels; per-sentence parsing keeps edges shallow (at most 93 levels in the 40 longest sentences of the corpus, 18 in a 770-sentence trial), and G2 v2 raises the recursion limit, records the maximum depth per shard and notes every edge deeper than 100 levels. | G1, G2, G3 |

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
5. **P2 outputs used the older word list** (chatbot names absent) `[INFERENCE from the list history]`. Superseded by P2 v2 (D12), which uses the new list.
6. **64 targets without a text and 65 orphan files** (in the old list, not in the new). Left in place.
7. **No environment lockfile.** Versions below were read from the installed packages on 2026-09-24;
   the May–June runs may have used other versions.
8. **Time-dependent inputs:** the PMC query result changes over time (65 earlier articles disappeared
   from the re-run); P4 depends on live NCBI records.
9. **Figure and table lines (flagged 2026-09-24; handled in P1b, run on the corpus 2026-09-25, RL-060).** Table rows, bare
   labels such as `Fig. 1` and `Table 2`, and caption lines ("Table 3" or "Fig. 1" followed by a capitalised word) are removed by the
   non-prose script (item 12). Running sentences that mention a table or figure ("Table 5 shows ...") stay.
10. **The May content-end rule cut many articles short `[DERIVED]`;** fixed in the rebuilt P0 (D8), but
    P1 onward still use the May boundaries and must be re-run from P1 on `content_regions_v2.csv`
    on `content_regions_v2.csv` and the whitelist of F4.
11. **Windows that ran on past their section (found 2026-09-25, fixed in D10).** Extra supplementary and ethics windows ran on for
    hundreds of lines and, without a References heading, to the end of the file. Capped in P0 (D10). Most of the "reference lists
    inside windows" first reported were section titles such as "3.1.2 References" inside the article, not reference lists.
12. **Non-prose text before graphbrain: script built, tested and run on the corpus (2026-09-25, stage P1b, RL-060).** Removed by
    `nonprose_removal_v1.py`: table rows (found in the raw text), caption lines ("Table 3 ...", "Fig. 1 ..." followed by a capitalised
    word), LaTeX, formulas, statistics, hexadecimal codes. Symbols become words or disappear ("45 °C" to "45 degrees Celsius", "m²" to
    "m squared", "≥ 18" to "at least 18", "10×" to "10 times", a dimension such as "3 × 3" is deleted, "±" is deleted, "=" is deleted
    inside formula fragments and becomes "equals" only in plain statements such as "temperature equals 0.7"). Running sentences that mention a
    table or figure ("Table 5 shows ...") stay. 107 hard cases in the script's self-test pass; on 700 fresh articles no line count changed and
    every cleaned line is the masked line with removals and the inserted words only. **Known limits:** some formula fragments inside mixed
    prose lines survive (about 1.6 per 1,000 lines still contain an "equals" that is formula residue, and prose with several equalities);
    URLs and DOIs are kept (about 500 lines in 600 articles contain one; a candidate for a later rule); a few number lists remain.
    Table rows are found from the raw text, so the script needs the raw files; a row with a single tab-separated cell is treated as prose.
    **Stage P1c (2026-09-26, RL-061/062)** adds three rules on top of this output (D11): abbreviation lists (5,655 lines), "Alt text:" lines (246) and table
    footnotes (17,327; only lines that follow a table block and start like a footnote, so the same sentence in the Methods stays). After P1c the cleaned
    corpus still holds about 670 lines that look like abbreviation lists (mostly prose with inline glosses) and about 530 footnote-like lines that do not
    follow a table. Data lists such as "Age: 45; Sex: male; BMI: 22" have the same shape as abbreviation lists and are removed with them.
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
    **The P2 v2 run (2026-09-26, RL-064) also used numpy 1.26.4 from a temporary scratch folder (`PYTHONPATH`).** That folder is not a lasting part of the
    environment: the session's scratch space was wiped once during the work and the copy had to be reinstalled (`pip install --no-deps --target <folder> numpy==1.26.4`).
    Until a separate environment with numpy below 2 exists, any rerun of P2, P6 or G2/G3 needs this workaround.
    **Permanent folder since 2026-09-26: `$HOME/np1_for_spacy`** (numpy 1.26.4, `pip install --no-deps --target ~/np1_for_spacy numpy==1.26.4`), used by
    `submit_coref_v2.sh`. Checked on a CSF compute node (job 21401159, script `diagnostics/p6_prototype/check_csf_env.sh`, not in git): without the folder spaCy and
    fastcoref fail to import; with it spaCy 3.4.4 and all four models (en_core_sci_sm, en_core_web_trf, en_coreference_web_trf, LingMess from the local cache) load and run.
18. **Superscript citations: handled in P1 v2, weak cases left for post-processing (flagged 2026-09-25).** P1 v2 removes a
    superscript number glued to a word ("development1", "models1,2", "et al.43") when the evidence is strong (numbers within the
    reference count, no veto, and an ordinary word in an article that cites by superscript, or "et al.N", or a lower-case word with
    a period or list before the number); 320,707 removed in 11,956 superscript-style articles. **Not removed, to be handled in
    post-processing:** weak cases such as "systems3" (an ordinary word plus one number in an article not recognised as
    superscript-style) and surname plus number ("Hinton25"): 22,625 candidates, the first 20 per article listed in
    `LCS/superscript_residuals_v2.jsonl` (17,570 rows). Some of them are variables or equation fragments ("equation5", "pred2"), so a
    review is needed before deleting. Model names (GPT5, Llama3, gpt2) are protected by a name list built from articles that do not
    cite by superscript.
19. **Open decision: articles with headings but no Results, Discussion or Conclusion heading (2,484).** Stage F4 excludes them for now
    (`STRUCTURAL` in `build_exclusion_list.py`); the project owner has not decided whether they should be included. Headingless articles
    (1,234) and articles without an end marker (361) are excluded by the owner's rule that articles without headings are not analysed.
20. **Fold the author rules (F3) into `article_blacklist.py` (planned).** Until then, rerunning `article_blacklist.py` rebuilds the list
    without the 116 F3 rows, so the order is F1/F2, then F3, then F4. After the fold, the pending window-based text test (item 13) joins the same script.
21. **Stages still to be built or rerun on the whitelist (planned).** P1b, P1c and P2 v2 have been run (RL-060, RL-062, RL-064). P6 v2 (coreference, D13, RL-067) and P7 v2 (focal sentences and shared
    citation ids, D14, RL-068) have been run. G2 v2 (graphbrain, D15, RL-070 to RL-072) has been run on the whole corpus (702,948 sentences, 0 exceptions), then superseded by G2 v3 (sentence units and the other item-30 fixes, RL-074 to RL-077), also run on the whole corpus (710,869 units, 1 error line). Next: G3 after its revision (item 29), then the matrix builder M1; everything downstream of P2
    in the May run is stale (item 2). P3 to P5 are superseded by the P1 v2 dictionary. A decision is also open on removing URLs and DOIs from the text (about 500 lines in
    600 articles contain one).
22. **The fetch loop keeps the oldest version (raised 2026-09-25, not yet checked).** `fetch_delta_pmc.py` and `fetch_s3_pmc.sh` try
    versions 1 to 4 in order and stop at the first that exists, so an article that has several versions in the bucket would be fetched in its
    earliest one. How often this happens is unknown; the bucket folder `PMC10462176.1` was empty although that article's text is on disk,
    so older versions can disappear. Related to D4.
23. **Known limits of the article blacklist.** About 90% of the comment flags rest on a title pattern (23 on a PubMed type alone); paper
    types come from keyword rules and PubMed's labels, which can be wrong (a paper on a dermatology examination is typed "Case Reports");
    PubMed answers are live queries cached on 2026-09-25, and a refresh changed 79 type reasons in a test; duplicates are only found where titles
    are within a Levenshtein distance of 0.30; for copies both members are removed, so no version of a copied paper stays; a preprint whose
    published version is not in the corpus is lost entirely; group-author articles are removed (item 14).

24. **Known limits of P2 v2 (2026-09-26, RL-064).** (a) The flat `focal_words.txt` that P6 and G3 read is looser than P2: it holds bare GPT, BERT, Claude, RITA and the other
    guarded names and is matched case-insensitively without the guard rules, so a block P2 selected can be re-checked by P6 v1 against a term P2 would have rejected in another place (P6 v2 does not
    re-filter blocks and does not read the flat file); G3 should read the matched terms recorded per sentence in `focal_extractions_v2.jsonl` instead. (b) The guard rules were checked by reading samples, not against labelled data; the
    rejected hits are counted per article in `focal_status_v2.csv` (10,050 in all) and can be audited. (c) 4,637 whitelisted articles yield no focal sentence: 3,925 have no accepted hit
    (mostly AI-use disclosures and acknowledgements after the closing section, 316 with no focal word anywhere), 412 had only rejected hits, 68 only hits inside headings, and a
    few could not be placed in a sentence. They leave the corpus at this stage. (d) Evidence for ambiguous names is at article level, so a homonym inside an article about LLMs
    is caught only by its homonym pattern. (e) "Transformer model(s)" stays in the list as before but does not count as article evidence; 41% of its occurrences are in articles with
    no other LLM term. (f) Context sentences do not cross headings and do cross paragraph breaks inside a section.
25. **Known limits of P1c.** About 670 lines that look like abbreviation lists remain (mostly prose with inline glosses) and about 530 footnote-like lines that do not follow a table.

26. **Known limits of P6 v2 (built 2026-09-26, run 2026-09-27/28).** (a) Precision was judged by reading samples (15 hard sentences, 143 and 30 real blocks), not against
    labelled data. (b) A pronoun whose antecedent is more than two sentences before the block, or in an earlier section, stays unresolved; a block that opens its
    section has no context. (c) "The former" / "the latter" are not handled. (d) A participle name ("the model ..., called MedLLM") gives "the model". (e)
    Requiring both models to agree removes about 9% of the replacements spaCy alone would make. (f) Source defects pass through (for example a typo with a leftover
    superscript, "Tranformers1's", item 18). (g) CPU speed: about 76 s per article per process on incline and about 34 s per article on CSF (4-core task, RL-066); the full run took about 4 hours with 100 shards (RL-067). (h) Memory: the largest blocks need more than 16 GB (17.5 GB measured for a 2,009-word block); a task that is killed is rerun with `--mem=32G`.

27. **Cited works are not identified across articles by P1 v2 (found 2026-09-28; solved for the kept sentences by P7 v2, D14).** The old P3 to P5 gave the same cited paper the same global reference id in every article (Entrez lookups;
    numeric citations resolved by their position in the reference list). P1 v2 replaces them, but its tokens (`a<PMCID digits>r<n>`) are per article, so one paper cited in two
    articles has two unrelated tokens, and numeric citations, which P1 v2 deletes, give no link at all. P1 v2 is therefore a replacement for graphbrain parsing (graphbrain does not read
    the old tokens), not for cross-article citation links. `LCS/citation_dictionary_v2.jsonl` records 412,616 cited works (355,911 matched to their reference-list text); its `doi` field
    is empty in every entry because of a bug in `citation_masking_v2.py` line 66: the pattern `10\.\d{4,9}/[^[ \t]]+` is malformed (the character class ends early, so it needs a literal `]`
    after the DOI) and never matches. DOIs are present in the stored reference text of 263,012 of the 355,911 matched works (74%; the stored text is cut at 300 characters, so some
    are lost). Fixing it does not change the masked text, only the dictionary. Reference matching has a second known limit: 24,043 works (5.8%) are flagged `ambiguous_with` (two or
    more reference entries share the first author and year, e.g. two "Wang et al. 2021"); they get one token that stands for all candidates. 56,705 works (13.7%) are unresolved.
    P7 v2 does not read the empty `doi` field: it reads the full reference lines and extracts DOIs and PMCIDs itself. The pattern in P1 v2 stays broken; if P1 is ever rerun, the fix is `[^\s\[\]]+`.
    Whether the matrix stage needs shared cited works for all 412,616 works, not only those in kept sentences, is open; if it does, a step is needed that extracts DOIs or titles from the reference text (or resolves
    them through Entrez, as old P4 did) and unifies the works across articles.

28. **Known limits of P7 v2 and how it was validated (2026-09-28, RL-069).** The first run (RL-068) had a bug: the DOI cleaner cut every final ".NNNNNNN" from a DOI, so IEEE, ACM, Frontiers
    and Taylor & Francis DOIs became proceedings- or issue-level DOIs and unrelated papers were merged (963 of 12,644 tokens, 670 works). It was found by a structural check and fixed; outputs of the
    first run are kept as `*.before_doi_fix_20260928`. Checks on the corrected output (local diagnostics log D-033; labels are the author's reading of the reference lines, not independent ground truth):
    the same paper in 25/25 pairs sharing a DOI, 12/12 sharing a PMCID, 4/4 identifier chains, 35/35 title-linked pairs across identifier groups, 30/30 title-only pairs; 60/60 works drawn uniformly at random
    and the 21 works with the least similar members are single papers (95% lower bound 0.94 for the uniform sample); a detector for "identifier shared by clearly different papers" finds 43 works
    (415 pairs) in the first run and 1 work (3 pairs) now. **Remaining limits.** (a) Missed merges: of 35 sampled near-miss pairs (different works, same first author, year within one, title-word Jaccard 0.35
    to 0.6) 30 were the same paper, so one paper (BERT, GPT-3, ELMo, Singhal et al.) can carry several ids; word overlap cannot separate these from different papers by the same authors in the same year.
    (b) 2,215 tokens (P1 v2 could not match them to the reference list) keep an id of their own and are never shared. (c) One source-data error is known: a reference line whose PMC metadata names the
    PMCID of another paper of the same journal issue (REF000928 holds one wrong member). (d) DOIs ending in a glued 8-digit PMID (326 distinct strings, 157 with the stem also present elsewhere) stay in the
    identifier list as separate strings; they are linked through the PMCID when there is one. (e) Threshold sensitivity: title threshold 0.5 adds 526 grouped token pairs, 0.4 adds 1,085, 0.7 removes 1,750.
29. **G3 upgrade: agreed changes and open decisions (owner discussion 2026-09-29; test script built 2026-09-30, RL-078, see the last paragraph of this item).** The May `chunk_4h_hpc.py` is revised
    before G3 is run on the G2 output; it is not rerun as it is. (a) **Recursion crash (agreed):** raise the recursion limit and put the loop over
    `hg_raw.search(('source', '*', '*'))` under error handling, so one deep edge cannot end a task (the May G3 kept 359 of 22,601 parsed articles, RL-071).
    (b) **New input (needed):** read the G2 databases — now `PG/g2_v3/shards/db_provenance_NNN.sqlite` (item 30 complete as of RL-077, 2026-09-29;
    supersedes the `g2_v2` path this item pointed to before the rerun; the 3-digit shard ids already
    match G3's pattern), use the `occurrences` attribute (sid) as provenance instead of an md5 of the edge, and write to a new output folder instead of
    `PG/postprocessed_output/`. (c) **Focal-word filter (to review):** G3 keeps an edge only if one of its atoms is in the flat `focal_words.txt` (plus a
    4-entry normalisation map), so multi-word terms ("large language models") count only through an abbreviation in the same edge; in the RL-071 trial
    8 of 14 articles kept edges. Related to item 24(a). (d) **Hyphenated words — [SUPERSEDED, stale as a G3-level open item]: the rule was
    designed and shipped inside G2 v3 instead (item 30(l), RL-075/RL-077), not left for G3.** Originally: merge the pieces that the parser splits into
    one atom (ben-gurion → ben_gurion); observed in the May/G2 v2 output: "multi-device" → multi, -, device; "pre-training" → pre, -, training, while
    "gpt-4" stays whole. G2 v3 now joins these with "_" at parse time (except where a hyphenated word contains a focal term, item 30(l)) — G3 receives
    already-joined atoms and can split on "_" again if a future need requires it. (e) **Abbreviation periods:** the wrong sentence cuts are handled in G2 (item 30); the atoms left in edges (`e%2eg%2e`,
    `i%2ee%2e`, `vs%2e`, `etc`) are connectives, proposed to be dropped in G3. (f) **Mathematical signs and remnants of tables, lists and equations
    (owner request):** remove; rule to be designed (commonest symbols in the corpus: ± 3,818, × 1,715, ∑ 500, ∈ 450, ≥ 426; the parser drops ± and × but
    makes ≥ an atom; local diagnostics log D-021). (g) **Auxiliary verbs (owner's idea, verified 2026-09-29):** drop be, have and do only when they are
    auxiliaries and keep them as main verbs. The parser already marks the difference in the atom type: auxiliaries are `Mv` ("has/Mv managed", "was/Mv
    terminated", "is/Mv produced"), main verbs `P`/`Pd` ("An LLM **is/Pd** a type of artificial intelligence", "which **has/P** three layers",
    "ChatGPT/GPT-4 **has/Pd** a wide range of applications"). The current test `role.startswith(('P', 'Mv')) and lemma in AUXILIARY_ROOTS` drops both, so
    today "is" is deleted from "An LLM is a type of AI"; the change is to test `Mv` only. Main-verb "do" still to be spot-checked. (h) **Negation (no
    change):** not, never, no and n't are taken out of the stop list and kept as modifiers. (i) **Modal verbs (open decision):** can, could, will, would,
    should, might, must, may and shall have type `Mm`. Today only "will" is dropped (it is in `nltk_abridged_stopwords_list.txt`); the others survive as
    separate atoms because the auxiliary test checks `P`/`Mv`, not `Mm` (an accident, not a decision; "can", "may" and "shall" are listed in
    `AUXILIARY_ROOTS` and still survive). Options: drop them all; keep them as atoms (very frequent, low-content nodes); fuse them with the verb
    ("could_improve", like the existing phrasal-verb fusion; multiplies the verb vocabulary and splits a verb's counts); record the modality as a tag on
    the edge and keep the verb atom (the assistant's recommendation). (j) **Comparison with the toy-corpus version (pending):**
    `tensor_data_staging/toy_large/7.5postprocessing_4hbased_correct.py` is a corrected version of the toy postprocessing that was never used for the toy
    rebuild (CLAUDE.md §4.23); compare it with `chunk_4h_hpc.py` before revising. (k) Superscript citations that P1 v2 left inside words ("systems3",
    item 18) are still for post-processing; the glued numbers at sentence ends are handled in G2 (item 30). (l) **Hyphenated words and
    focal terms (owner decision 2026-09-29):** G2 v3 does not join a hyphenated word when its parts contain a focal term ("LLM-based",
    "AD-LLM", "SMILES-BERT", "ChatGPT-generated", "large-language-model-based"), so the focal term stays its own atom; these words are
    listed per sentence (`hyphens_kept_focal`; 118 of 886 hyphenated words in a 1,266-sentence trial). All other hyphenated words are
    joined with "_" ("pre_trained", "transformer_based"); G3 can split them on "_" if needed. (m) **Leftover symbols:** atoms made only of symbols or formula debris ("[", "]", "∈") are dropped in G3;
    "%" is kept (it is the atom "%", shown encoded as `%25`, not the word "percent").
    **Comparison (j), done 2026-09-30.** `chunk_4h_hpc.py` (2026-05-29) is the same code as the toy `postprocessing_4h.py`
    (2026-06-15, which built the toy `corpus_curated.sqlite`); `7.5postprocessing_4hbased_correct.py` (2026-06-16) corrects it and was never
    applied to any data. Differences: chunk_4h starts the search for a focal term's predicate one level too high (`len(path)-3`), so the
    simplest clause ("LLMs outperform X") yields nothing; it writes a fringe pool (`source_fringe`) that chunk12 never reads (chunk12 takes
    fringe atoms from the cousins); it strips trailing digits from every atom (`gpt2` -> `gpt`, undoing P1's protection of model names); it
    prunes appositions around the focal term; it lemmatises single words out of context with `en_core_web_sm`. Owner decision (2026-09-30):
    G3 is built on 7.5, with the fixes below. **Test script `PG/scripts/g3_curation_test.py` (RL-078)** — (a) input read line by line,
    recursion limit 20,000, every unit in its own try block; (b) input is the G2 v3 shard JSONL `g2_parsed_NNN.jsonl`, not the database
    (same content, one record per unit with uid, unit hash, sid, hash_final, lemma edges and atom-to-word positions; the database keeps
    one source edge per article and distinct edge); lemmas are G2's own `en_core_web_trf` lemma edges, so G3 no longer loads spaCy;
    (c) focal terms found in the unit text by P2's own matcher (`focal_terms.py`, guard rules included) and every atom of a mention
    replaced by one canonical atom `<canon>/Cp/focal`; merged: plural/singular, spelling variants (Chat GPT, Chat-GPT, ChatGPT; GPT4,
    GPT 4, GPT-4, GPT-4.0), abbreviation and expansion (LLM, large language model(s); BERT, Bidirectional encoder representations from
    transformers), brand prefixes (Google Gemini, Mistral AI), and (owner decision 2026-09-30, RL-079) ChatGPT-<version> into
    GPT-<version> (`chatgpt_4` -> `gpt_4`; bare `chatgpt` stays), model sizes and snapshot dates dropped (`mistral_7b` -> `mistral`,
    `gpt_4o_2024` -> `gpt_4o`; ESM-1b keeps its "1b"); kept apart: `llm`, `language_model` and `transformer_model`, every model version
    (`gpt_4`, `gpt_4o`, `gpt_3_5`), the bare family name (`gpt`); a version is a one-digit major ("GPT-44" is GPT-4 plus a glued
    citation and is read as `gpt`); (e) e.g., i.e., vs., cf., etc., viz., et al. dropped; (f) atoms without letters dropped (numbers, signs, brackets),
    "%" becomes the atom `percent/C/en`; (g) be/have/do dropped only as `Mv`, kept as main verbs; the stop list is not applied to main-verb
    be/have/do (the RDS list `PG/nltk_abridged_stopwords_list.txt` holds "been" and "am", the toy list holds "be"; the two lists differ only
    in be/also); modal verbs (`Mm`) recorded as `modality` on their verb group and dropped from the atoms (option `--modals tag`; `drop`
    and `keep` also available); (h) negation kept whenever it is a modifier, now including "no" as a determiner (`no/Md`, dropped by 7.5
    and chunk_4h); type `Cm` (noun used as a modifier, "cancer research") kept, dropped by 7.5 and chunk_4h; provenance: ids and hashes
    `sha1("<id>|<edge>")[:12]` for parents `<uid>.P<k>`, children `.D`/`.F`/`.S<j>`, cousins `<uid>.K<j>` with the parents they belong to
    (the unit is the grandparent edge), every output atom traced to its source atoms, words and token positions. Output JSONL per unit,
    optional database in chunk12's form (`pmcid::uid::hash`). **Open decisions:** stop-list additions (RL-079 evidence: "our", discourse adverbs, number
    words, "well"); leftover phrasal particles ("on/M"); modal option (tag recommended). **Found upstream:** "LLM" is a PLAIN focal term
    with no homonym guard; PMC8815195 uses it for lipid-lowering medication ("not on LLM (89.5%)") and passed P2.
30. **G2 sentence units: graphbrain's own re-splitting and glued citation numbers (found 2026-09-29; fix designed, tested, and run on the whole corpus as G2 v3, RL-077).** Each P7
    sentence is parsed by one graphbrain call, but graphbrain splits it again with its own model: 22,091 of 702,948 sentences (3.1%) became several units
    in G2 v2. In two samples of 120 cuts labelled by hand (the assistant's reading of each context), 72/120 and 68/120 cuts were real sentence boundaries that
    P2's splitter (`en_core_sci_sm`) had missed; the others cut where no sentence ends ("pre‖-trained", "(Figure 2)‖.", "SciBERT: ‖Developed from
    scratch …", "e.g.‖ …"). Separately, 2,032 sentences (2,164 places) hold a citation number glued to a sentence-final period ("… by OpenAI.5 To reduce
    …", "… by LLMs.9,10 ChatGPT …"), which hides a real boundary from both models; P1 v2's superscript rule (item 18) does not remove these because it judges
    the word before the number and refuses OpenAI, LLMs, AI (to protect model names such as GPT5). **Proposed fix: at parse time in G2, not upstream**, so
    P2, P6 and P7 are not rerun and every sentence id and hash stays: (1) a glued citation number is removed (as P1 v2 deletes superscripts) and a boundary
    is set there; (2) a cut proposed by graphbrain is kept only where one sentence ends and another begins (the text before ends with . ! ? or …, possibly
    followed by a citation number and closing quotes or brackets, and not with an abbreviation that never ends a sentence such as e.g., i.e., cf., vs.,
    Fig., Eq., No., Dr., U.S.; "etc." and "et al." only before a capitalised word; the text after begins with a capital letter, a digit, "(i)" or "(a)");
    other cuts are dropped and the text is parsed again with the kept boundaries set before `en_core_web_trf`'s parser runs. A first version kept a cut only
    when `en_core_sci_sm` also found it; that test is circular (that model made these sentences) and kept 3 of 72 and 4 of 68 real boundaries, so it was
    rejected; the earlier count of "271 of 271 successful merges" measured valid trees, not correct segmentation. **Tests** (prototype
    `diagnostics/g2_prototype/sentence_units_v2.py`, local, RL-073): the rule was refined on the first labelled sample, frozen, and scored on the second:
    real boundaries kept 58/68 (85%), false cuts dropped 50/52 (96%), 12 errors in 120 (keeping every cut: 52; the `en_core_sci_sm` test: 64). The
    remaining errors are missing periods, comma splices, a bullet, "et al. GPT…", a variable named "vs.", junk reference lines and one boundary inside quoted
    speech. Glued-citation detector: 59 of 60 sampled cuts correct before a fix for section numbers ("A.3"). Self-test 25/25. Against the G2 v2 output:
    1,131 of 1,147 random sentences give byte-identical edges (the other 16 are false cuts removed); in 300 split sentences 207 cuts are kept and 171
    dropped; all 300 sampled glued-citation sentences change as intended; 0 parse failures, 0 exceptions, every edge restored from its string, every preset
    boundary respected, maximum depth 20; no extra time for sentences graphbrain does not split (about 0.07 s per sentence on incline). Units keep their
    sentence's sid and hash; removed citation numbers are recorded per sentence. A unit may lack a focal term after a cut; G3 filters edges by focal terms anyway.
    **Scope agreed with the owner (2026-09-29) and built as `PG/scripts/graphbrain_parse_v3.py` (RL-074, RL-075; CSF test `check_csf_g2v3.sh`,
    full run `submit_g2_v3.sh`, output `PG/g2_v3/`; merging is a separate script, `merge_g2_v3.py`, RL-076 — parsing needs graphbrain and spaCy, merging
    needs neither):** unit ids `<sid>.U<k>` with hash sha1("<uid>|<parsed text>")[:12]; glued-citation
    removal; the sentence-end rule; removal of non-prose (all captions including focal ones, captions behind a DOI or figure id, caption
    bodies whose label P2 left on the previous sentence, captions glued after a sentence end, lone labels, reference-list lines, headings
    of at most 4 words ending with a colon, units without letters); URLs and DOIs replaced by tokens URL + 8 digits from a hash of the
    normalised address (same address, same token in every article; graphbrain otherwise makes each URL one escaped, never-repeated atom);
    "∼", "≈", "≥", "≤" and arrows turned into words; hyphenated words joined with "_" (parsed with the hyphen, verbs and objects are lost:
    "was pre-trained on PubMed abstracts" gave the predicate "-"), except words whose parts contain a focal term (item 29 l). The list of
    caption bodies is computed by each task from P2's output (no separate preparation step). Auxiliary verbs, modal verbs and leftover symbol atoms stay in G3.
    Precision of the sentence-level non-prose rules (25 random hits each, read by the assistant): captions 24/25 (the miss, "Table 2 (See
    App. B) shows ...", is now excluded), caption bodies 25/25, DOI captions 25/25, reference lines 25/25, headings 19/25 at up to 6 words
    (the six misses were clauses introducing a list; the rule now takes at most 4 words). Corpus-wide candidates: 3,889 captions, 1,398
    DOI captions, 694 caption bodies, 1,546 captions glued after a sentence end, 574 reference lines. Recall was not measured.
    **Full run and merge completed 2026-09-29 (RL-077), superseding G2 v2 (row above).** CSF: `submit_g2_v3.sh`, array job 21557322 —
    only shard 49 finished under the array job itself, the other 49 shards each finished under a separately submitted job id (cause not
    diagnosed, flagged for the owner). Merge run directly on incline (not `sbatch`: the merge needs neither graphbrain nor spaCy, so it
    does not need CSF; `merge_g2_v3.py`, RL-076). Output `PG/g2_v3/g2_parsed_v3.jsonl`: 34,662 articles, 702,948 sentences (696,351 ok,
    6,591 non-prose, 6 empty), 710,869 units (36 dropped), 1,674 distinct URL/DOI tokens, maximum edge depth 101, 362,204 hyphens joined
    (62,652 kept apart for a focal term), 2,164 glued citations removed, 2,947 URLs tokenised, 1,500 symbols normalised, 582 labels and
    1,542 captions removed. 1 error line total (`PMC12476623.r1.L92.S3`, a single sentence that is a long comma-separated list of terms,
    nested 101 levels deep — a structural edge case, not a defect). 0 problems found on the merge's own verification.

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
| `PP/focal_window_extraction.py`, `PP/exclusion_diagnostics.py`, `LCS/focal_extractions_v1.jsonl`, `PP/exclusion_report.csv` | P2 v1 and its diagnostic, superseded by P2 v2 (D12) |
| `PP/citation_resolution_1_inventory.py`, `_2_api.py`, `_3_translate.py`, `_4_coref.py`, `_4b_restore.py`, `PP/api_inventory_target.json`, `PP/global_translation_dictionary.json`, `LCS/focal_extractions_v2_resolved.jsonl`, `LCS/focal_extractions_v3_graphbrain_ready.jsonl` | May citation resolution (P3 to P5) and coreference (P6/P7 v1), superseded by P1 v2, P6 v2 and P7 v2 (D9, D13, D14) |
| `PP/citation_standartization_soft_masking.py`, `PP/citation_vault_light_masking_v1.jsonl`, `LCS/masked_corpus_v1/` | P1 v1 masking, superseded by P1 v2 (D9) |
| `LCS/coref_v2_trial/` | Output of the CSF trial of P6 v2 (RL-066, 20 articles); not part of the run |
| `PP/focal_citations_v2.before_doi_fix_20260928.py`, `LCS/*.before_doi_fix_20260928`, `PP/focal_words.before_v2_20260926.txt` | Versions replaced by a fix or a rewrite (RL-069, RL-063), kept for the record; do not use |
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
| P1b | `python3 pmc_preprocessing/nonprose_removal_v1.py --selftest`, then `python3 pmc_preprocessing/nonprose_removal_v1.py --workers 8` | `[LOG]` RL-056 to RL-060 |
| P1c | `python3 pmc_preprocessing/nonprose_extra_rules_v1.py --selftest`, then `python3 pmc_preprocessing/nonprose_extra_rules_v1.py --workers 8` | `[LOG]` RL-061, RL-062 |
| P2 (v1) | `nohup python3 pmc_preprocessing/focal_window_extraction.py > PP/extraction.log 2>&1 &` | `[LOG]` (superseded, D12) |
| P2 (v2) | `python3 pmc_preprocessing/focal_terms.py --selftest`, `python3 pmc_preprocessing/focal_terms.py --write-flat-list`, then `PYTHONPATH=<scratch numpy 1.26> python pmc_preprocessing/focal_extraction_v2.py --workers 8` | `[LOG]` RL-063, RL-064 |
| P2d | `python3 pmc_preprocessing/exclusion_diagnostics.py` | `[MTIME]` |
| P3–P5 | `python3 pmc_preprocessing/citation_resolution_{1_inventory,2_api,3_translate}.py` in order | `[LOG]` `phase2.log`, `phase3.log` (P4, P5) |
| P6, P7 (v1) | `python3 pmc_preprocessing/citation_resolution_4_coref.py`, then `..._4b_restore.py` | `[LOG]` `phase4.log`; P7 `[MTIME]` (superseded, D13) |
| P6 (v2) | `PYTHONPATH=$HOME/np1_for_spacy python pmc_preprocessing/coref_resolution_v2.py --selftest`; on CSF `sbatch pmc_preprocessing/submit_coref_v2.sh`; then `PYTHONPATH=$HOME/np1_for_spacy python pmc_preprocessing/coref_resolution_v2.py --merge --nshards 100` | `[LOG]` RL-065 to RL-067 |
| P7 (v2) | `python3 pmc_preprocessing/focal_citations_v2.py --selftest`, then `python3 pmc_preprocessing/focal_citations_v2.py` | `[LOG]` RL-068 (superseded), RL-069 |
| G1 | `python3 phase5_graphbrain/scripts/01_matrix_partition.py` | `[MTIME]` |
| G2 (v3) | `sbatch phase5_graphbrain/scripts/check_csf_g2v3.sh`; `sbatch phase5_graphbrain/scripts/submit_g2_v3.sh`; then `python3 phase5_graphbrain/scripts/merge_g2_v3.py --nshards 50` (merging needs neither graphbrain nor spaCy, so no `PYTHONPATH`) | `[LOG]` RL-074, RL-075 (script, CSF test), RL-076 (merge script) |
| G2 (v2) | `sbatch phase5_graphbrain/scripts/check_csf_g2.sh`; `sbatch phase5_graphbrain/scripts/submit_g2_v2.sh`; then `PYTHONPATH=$HOME/np1_for_spacy python phase5_graphbrain/scripts/graphbrain_parse_v2.py --merge --nshards 50` | `[LOG]` RL-070, RL-071 (tests), RL-072 (full run + merge) |
| G2 (May) | `sbatch phase5_graphbrain/scripts/submit_v2.sh` | `[LOG]` `PG/scripts/logs_v2/node_15772611_*` |
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
| `pmc_preprocessing/nonprose_extra_rules_v1.py` | d34e1c6f9d81 | 2026-09-26 |
| `pmc_preprocessing/build_content_regions.py` | 64e959d673a0 | 2026-09-25 20:33 |
| `pmc_preprocessing/citation_standartization_soft_masking.py` | 7acf961258e6 | 2026-05-16 01:03 |
| `pmc_preprocessing/citation_standartization_soft_masking.sh` | dfb1035f90b0 | 2026-05-16 01:05 |
| `pmc_preprocessing/focal_window_extraction.py` | a8b50a542e10 | 2026-09-23 13:31 |
| `pmc_preprocessing/exclusion_diagnostics.py` | bed02af9a327 | 2026-05-17 00:43 |
| `pmc_preprocessing/citation_resolution_1_inventory.py` | fa97765de2aa | 2026-05-18 19:21 |
| `pmc_preprocessing/citation_resolution_2_api.py` | 07ae408249a3 | 2026-05-18 19:26 |
| `pmc_preprocessing/citation_resolution_3_translate.py` | 84f9c722ad25 | 2026-05-18 19:35 |
| `pmc_preprocessing/citation_resolution_4_coref.py` | da9d09307d8f | 2026-05-19 20:01 |
| `pmc_preprocessing/coref_resolution_v2.py` | 5a1e21746d2b | 2026-09-26 |
| `pmc_preprocessing/submit_coref_v2.sh` | 77ef164cc5ee | 2026-09-26 |
| `pmc_preprocessing/focal_citations_v2.py` | e9fd4bf14862 | 2026-09-28 |
| `phase5_graphbrain/scripts/graphbrain_parse_v3.py` | a3d7f986d292 | 2026-09-29 |
| `phase5_graphbrain/scripts/merge_g2_v3.py` | 7cb4b1d7f1c9 | 2026-09-29 |
| `phase5_graphbrain/scripts/submit_g2_v3.sh` | a6ac2ec80db2 | 2026-09-29 |
| `phase5_graphbrain/scripts/check_csf_g2v3.sh` | 85838f5aa8e8 | 2026-09-29 |
| `phase5_graphbrain/scripts/graphbrain_parse_v2.py` | 3eec8b0b256c | 2026-09-28 |
| `phase5_graphbrain/scripts/submit_g2_v2.sh` | c279b391f6cf | 2026-09-28 |
| `phase5_graphbrain/scripts/check_csf_g2.sh` | 0523eec595e8 | 2026-09-28 |
| `pmc_preprocessing/focal_citations_v2.before_doi_fix_20260928.py` (the version of RL-068, kept for the record; do not use) | 49e9a3a64c5b | 2026-09-28 |
| `pmc_preprocessing/citation_resolution_4b_restore.py` | 861b71001a65 | 2026-05-20 21:58 |
| `pmc_preprocessing/focal_words.txt` (generated from `focal_terms.py`, 208 terms) | 87abdc416191 | 2026-09-26 |
| `pmc_preprocessing/focal_words.before_v2_20260926.txt` (the 185-term list of 2026-09-23) | 5a163624bf28 | 2026-09-23 13:31 |
| `pmc_preprocessing/focal_terms.py` | ae5962450379 | 2026-09-26 |
| `pmc_preprocessing/focal_extraction_v2.py` | 78e3e9e5599e | 2026-09-26 |
| `phase5_graphbrain/scripts/01_matrix_partition.py` | 89be16b155f0 | 2026-05-21 00:35 |
| `phase5_graphbrain/scripts/02_transformer_node.py` | cf8c8b02e28b | 2026-05-21 00:37 |
| `phase5_graphbrain/scripts/submit_array.sh` | 55b80cddd73e | 2026-05-21 00:51 |
| `phase5_graphbrain/scripts/parse_stage_v2.py` | 086a5d4237af | 2026-05-31 18:29 |
| `phase5_graphbrain/scripts/submit_v2.sh` | f8021d0db570 | 2026-05-23 20:27 |
| `phase5_graphbrain/scripts/chunk_4h_hpc.py` | dfc6cc5e9662 | 2026-05-29 23:36 |
| `phase5_graphbrain/scripts/g3_curation_test.py` | bf53c0404fb7 | 2026-09-30 |
| `phase5_graphbrain/scripts/submit_4h.sh` | cc13dba06e28 | 2026-05-30 19:19 |
| `phase5_graphbrain/nltk_abridged_stopwords_list.txt` | 2b6c7d9fdae9 | 2026-06-29 21:42 |
