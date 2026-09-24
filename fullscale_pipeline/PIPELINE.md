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
| P1 | Citation masking (regex; citations become `__CITE_<pmcid>_NNN__`) inside the content boundaries only | `nohup python3 citation_standartization_soft_masking.py` (an sbatch wrapper `.sh` exists but no SLURM output from it exists) | `PP/sequence_metadata_relaxed.csv` (to be switched to `PP/content_regions_v2.csv`, `include`=1), `LCS/pure_text_corpus/` | `LCS/masked_corpus_v1/` (28,284 files), `PP/citation_vault_light_masking_v1.jsonl` | 2026-05-16 | DONE-stale |
| P2 | Sentence split (sciSpaCy `en_core_sci_sm`), focal-word regex, keep hit sentence ±1 | `nohup python3 focal_window_extraction.py` | `LCS/masked_corpus_v1/` | `LCS/focal_extractions_v1.jsonl` (22,795 documents) | 2026-05-16; **script edited 2026-09-23, not re-run** | DONE-stale |
| P2d | Diagnostic: why documents produced no focal window | `python3 exclusion_diagnostics.py` | P0 csv, `LCS/pure_text_corpus/`, `LCS/focal_extractions_v1.jsonl` | `PP/exclusion_report.csv` (5,489 rows) | 2026-05-17 | DONE (diagnostic, not a chain input; uses an outdated hardcoded word list) |
| P3 | Inventory of citation tokens found inside the extracted windows | `python3 citation_resolution_1_inventory.py` | `LCS/focal_extractions_v1.jsonl`, `PP/citation_vault_light_masking_v1.jsonl` | `PP/api_inventory_target.json` (13,271 documents) | 2026-05-18 | DONE-stale |
| P4 | Resolve tokens through Entrez `efetch` (batches of 200, 0.35 s apart, no API key); positional index for numeric citations, year match for author-year; unresolved tokens get `__REF_HASH_<sha256[:10]>__` | `python3 citation_resolution_2_api.py` | `PP/api_inventory_target.json`, NCBI Entrez | `PP/global_translation_dictionary.json` (105,389 tokens), optional `PP/phase2_api_errors.log` | 2026-05-18 | DONE-stale |
| P5 | Replace per-document tokens with the global reference ids | `python3 citation_resolution_3_translate.py` | `PP/global_translation_dictionary.json`, `LCS/focal_extractions_v1.jsonl` | `LCS/focal_extractions_v2_resolved.jsonl` (22,795) | 2026-05-18 | DONE-stale |
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
    (`include`=1).

## 7. Environment

Conda environment `tensor_env` (`/mnt/hum01-home01/p91688di/miniconda3/envs/tensor_env`), read
2026-09-24: Python 3.10.20; graphbrain 0.7.0; spaCy 3.4.4 with `en_core_web_trf` 3.4.0 (G2),
`en_core_web_sm` 3.4.1 (G3), `en_core_sci_sm` 0.5.1 and scispacy 0.5.1 (P2); fastcoref 2.1.6 (P6);
torch 2.11.0; transformers 4.25.1; numpy 2.2.6; scipy 1.15.3; pandas 2.3.3; awscli 1.44.78 (R2/R3;
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
| `PG/scripts/02_transformer_node.py`, `submit_array.sh`, `PG/output_sqlite/` | v1 parse (75 GB), superseded by v2 |
| `PG/scripts/trial_run_v2.py`, `verify_provenance.py`, `verify_v2_provenance.py`, `trial_postprocess.py`, `trial_generational.py`, `PG/curated_sqlite_v2/` | Trial and diagnostic scripts and their outputs |

## Appendix B — commands (all run from `R`; no secrets are stored in any script)

| Stage | Command | Evidence |
|---|---|---|
| R1 | `export NCBI_API_KEY=<key>; nohup python3 -u query_pmc_entrez.py > <log> 2>&1 &` | `[LOG]` `pmc_metadata_log.txt` (May); session log 2026-09-23 |
| R2 | `nohup python3 fetch_s3_pmc.sh > LCS/modern_download.log 2>&1 &` | `[LOG]` |
| R3 | `python3 isolate_delta.py` then `sbatch submit_delta_fetch.sh` | `[LOG]` `slurm-21295326.out` |
| P0 | `python3 pmc_preprocessing/build_content_regions.py` (writes `PP/content_regions_v2.csv`; refuses to overwrite) | `[LOG]` RL-025 |
| P1 | `nohup python3 pmc_preprocessing/citation_standartization_soft_masking.py > PP/masking_execution.log 2>&1 &` | `[LOG]` |
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
| `pmc_preprocessing/build_content_regions.py` | 16b485d1f3ea | 2026-09-24 21:56 |
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
