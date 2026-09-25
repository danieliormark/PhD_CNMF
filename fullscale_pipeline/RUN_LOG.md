# Full-scale run log (append-only)

Every action on full-scale data or code is recorded here, newest at the bottom. Rules:
`SESSION_PROTOCOL.md` §J. Stage IDs refer to [`PIPELINE.md`](PIPELINE.md). Entries are never edited
afterwards; a correction is a new entry that cites the old one.

**Entry types.** `RUN` a stage was executed · `CODE-CHANGE` a script or list was edited ·
`AUDIT` read-only check. **Provenance.** `LIVE` written when the action happened ·
`RECONSTRUCTED` rebuilt on 2026-09-24 from logs and file times, with the evidence named.

**Script version at run time.** A script whose modification time precedes the run, and which has
not been changed since, is taken to be the version that ran (sha256 in PIPELINE.md, Appendix C).
Where a script was edited after the run, the run-time version is stated as *not recoverable*.

Exploratory diagnostics (searches, trials, sample scans) are not logged here; they go to a
separate diagnostics log (SESSION_PROTOCOL.md §J).

Paths use the abbreviations of PIPELINE.md (`R`, `LCS`, `PP`, `PG`).

---

## Reconstructed history (2026-05 to 2026-06)

**RL-001 · 2026-05-09 · R2 · RUN · RECONSTRUCTED**
`python3 fetch_s3_pmc.sh` under `nohup`. Input: a target list of **34,453** IDs (its generating
query is not preserved). Result: 34,398 downloaded, 55 not found. Evidence: `LCS/modern_download.log`.
Script version: mtime 2026-05-09 18:33, before the run.

**RL-002 · 2026-05-13 · R1 · RUN · RECONSTRUCTED**
`query_pmc_entrez.py` under `nohup`. Output `LCS/target_pmcids.txt` and `target_metadata.json`,
**34,800** IDs. Evidence: `R/pmc_metadata_log.txt`. Script version: *not recoverable* (edited
2026-09-23).

**RL-003 · 2026-05-14 · R3 · RUN · RECONSTRUCTED**
`isolate_delta.py`, then `fetch_delta_pmc.py`. The script states 402 missing IDs
(34,800 − 34,398). Texts on disk afterwards: 34,751 `[DERIVED]`. No log survives; evidence: script
mtimes 2026-05-14 and the comment in `fetch_delta_pmc.py`. Script version: as on disk.

**RL-004 · 2026-05-14/15 · P0 · RUN · RECONSTRUCTED · GAP**
`PP/sequence_metadata.csv` (05-14) and `PP/sequence_metadata_relaxed.csv` (05-15, 31,511 rows) were
produced by code that was not preserved. Evidence: file times only.

**RL-005 · 2026-05-16 · P1 · RUN · RECONSTRUCTED**
`nohup python3 citation_standartization_soft_masking.py`. 28,284 documents masked. Evidence:
`PP/masking_execution.log`. Script version: as on disk (mtime 05-16 01:03).

**RL-006 · 2026-05-16 · P2 · RUN · RECONSTRUCTED**
`nohup python3 focal_window_extraction.py`. 22,795 documents with focal windows, written to
`LCS/focal_extractions_v1.jsonl` (23:18). Evidence: `PP/extraction.log`. Script version:
*not recoverable* (edited 2026-09-23; the earlier hardcoded list lacked the chatbot names).

**RL-007 · 2026-05-17 · P2d · RUN · RECONSTRUCTED**
`exclusion_diagnostics.py` → `PP/exclusion_report.csv`, 5,489 rows. Evidence: script and file times.

**RL-008 · 2026-05-18 · P3, P4, P5 · RUN · RECONSTRUCTED**
`citation_resolution_1_inventory.py` (13,271 documents), `citation_resolution_2_api.py` (105,389
tokens mapped), `citation_resolution_3_translate.py` (22,795 documents to
`LCS/focal_extractions_v2_resolved.jsonl`). Evidence: `PP/phase2.log`, `PP/phase3.log`, output files.
Script versions: as on disk.

**RL-009 · 2026-05-19/20 · P6 · RUN · RECONSTRUCTED**
`citation_resolution_4_coref.py`: 22,795 processed, 22,572 kept, 223 dropped, 0 failed.
Evidence: `PP/phase4.log`. Script version: as on disk (mtime 05-19 20:01).

**RL-010 · 2026-05-20 21:58 · P7 · RUN · RECONSTRUCTED**
`citation_resolution_4b_restore.py`: 223 documents appended to v3, flag `restored_from_v2`.
v3 now 22,795 unique PMCIDs. Evidence: v3 file contents (verified 2026-09-24); no log.

**RL-011 · 2026-05-21 00:35 · G1 · RUN · RECONSTRUCTED**
`01_matrix_partition.py` → 50 shards, 22,795 lines. Evidence: file times and contents.

**RL-012 · 2026-05-21/22 · G2 (v1) · RUN · RECONSTRUCTED · SUPERSEDED**
`sbatch submit_array.sh` (array 1-50%15), job **15138115**, `02_transformer_node.py`. 50 databases in
`PG/output_sqlite/`, no provenance link. Evidence: `PG/scripts/logs/node_15138115_*`.

**RL-013 · 2026-05-30 19:22–19:50 · G3 · RUN · RECONSTRUCTED**
`sbatch submit_4h.sh` (array 1-500; tasks 1–50 worked), `chunk_4h_hpc.py`. 50 curated databases.
SLURM job ID not recorded in the logs. Evidence: `PG/logs/cluster_*.log`, file times. Note: the raw
v2 databases it read were regenerated on 2026-06-01 (RL-014); a first v2 parse before 05-30 is
implied but has no log `[INFERENCE]`.

**RL-014 · 2026-06-01 00:44–23:08 · G2 (v2) · RUN · RECONSTRUCTED**
`sbatch submit_v2.sh` (array 1-50%15), job **15772611**, `parse_stage_v2.py` (mtime 05-31 18:29).
50 databases `PG/output_sqlite_v2/db_provenance_NN.sqlite`. Evidence: `PG/scripts/logs_v2/`.

## Live entries

**RL-015 · 2026-09-23 · P2, P6, G3 word list · CODE-CHANGE · LIVE**
33 general LLM/chatbot names added to `focal_window_extraction.py` (hardcoded list) and
`PP/focal_words.txt` (both 13:31). Reason: the May list lacked ChatGPT, Claude, Grok and related
names; exact-phrase policy (bare GPT, Llama, PaLM, Perplexity collide with other biomedical uses).
Deviation D1. **P2 not re-run.**

**RL-016 · 2026-09-23 · R1 · CODE-CHANGE · LIVE**
The same 33 names added to `BASE_QUERY` in `query_pmc_entrez.py` (`[Title/Abstract]`).
Backups made first: `LCS/target_pmcids.pre_llm_terms_backup_20260923_133436.txt` and
`target_metadata.pre_llm_terms_backup_20260923_133436.json`.

**RL-017 · 2026-09-23 13:34–13:43 · R1 · RUN · LIVE**
`nohup python3 -u query_pmc_entrez.py` on the `incline` host, no SLURM, anonymous NCBI rate
(0.35 s). Result: **46,241** unique PMCIDs, `LCS/target_pmcids.txt` and `target_metadata.json`
(both 46,241 entries). About 9 minutes. Script version at run time: *not recoverable* (the
API-key edit of RL-018 followed). Deviation D1.

**RL-018 · 2026-09-23 13:47 · R1 · CODE-CHANGE · LIVE**
`query_pmc_entrez.py`: NCBI key read from `NCBI_API_KEY`, interval 0.11 s with a key. No key is
stored in the script. Deviation D2. Not yet used in a run.

**RL-019 · 2026-09-24 13:05 · R3 · RUN · LIVE**
`python3 isolate_delta.py` on `incline`. 11,555 IDs written to `LCS/delta_pmcids.txt`.
sha256 `8b3877541844`.

**RL-020 · 2026-09-24 13:15 · R3 · CODE-CHANGE · LIVE**
New `submit_delta_fetch.sh` (sha256 `69545e630577`). A first version requested four cores and was
rejected by SLURM on the `serial` partition; corrected to one core. Deviation D5.

**RL-021 · 2026-09-24 13:18–16:07 · R3 · RUN · LIVE**
`sbatch submit_delta_fetch.sh`, job **21295326**, `fetch_delta_pmc.py` (sha256 `b620c1b961d8`), CSF3
`serial`. 11,491 downloaded, 64 not found; empty error log. Evidence: `R/slurm-21295326.out`.

**RL-022 · 2026-09-24 ~19:30 · R2/R3 · AUDIT · LIVE**
Read-only checks: 46,242 texts on disk, no empty files; the 64 unfetched IDs equal the 64 failure lines
of the job; 65 files on disk are not in the current target list; `aws s3 ls` for all 64 unfetched IDs:
22 exist under versions outside 1–4, 42 have no folder. Deviation D4 opened. No file under `R`
was modified. Scratch files only in `/tmp`.

**RL-023 · 2026-09-24 · G1–G3 · AUDIT · LIVE**
Read-only: 50 shards (22,795 lines), 50 raw v2 and 50 curated databases; every curated database is
older than its raw database (50 of 50); v3 has 22,795 unique PMCIDs of which 223 are flagged
`restored_from_v2`; 22,756 of them are in the current 46,241 list. Recorded in PIPELINE.md §3, §6.

**RL-024 · 2026-09-24 21:56 · P0 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/build_content_regions.py` (sha256 `16b485d1f3ea`) replaces the lost May generator.
Rules and reasons: PIPELINE.md D8. Created in the RDS directory: this script and, by RL-025, one
output file. `sequence_metadata_relaxed.csv` was not touched. Deviation D8.

**RL-025 · 2026-09-24 21:56–21:58 · P0 · RUN · LIVE**
`python3 build_content_regions.py` on `incline32`, no SLURM, 1 min 43 s. Inputs: `LCS/target_pmcids.txt`
(46,241; 46,177 have a text) and `LCS/pure_text_corpus/`. Output `PP/content_regions_v2.csv`, 46,177 rows:
`ok` 41,243, `no_closing_heading` 2,796, `headingless` 1,635, `no_end_marker` 503; start rule
Introduction 38,179, divider 7,998; 9,807 articles with at least one extra region (Supplementary 3,216,
Ethics 7,269). Downstream stages P1–G3 not re-run. Deviation D8.

**RL-026 · 2026-09-25 13:46 · F1 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/build_article_blacklist.py` (sha256 `b6b7ccc67a19`). Created in the RDS directory:
this script and, by RL-027, `llm_corpus_staging/article_blacklist.csv`. Batch 1 removes four paper types
chosen by the project owner: correction/erratum/retraction, case report, unclassified, guideline/consensus.

**RL-027 · 2026-09-25 13:46–13:47 · F1 · RUN · LIVE**
`python3 build_article_blacklist.py` on `incline`, no SLURM, 20 s. 46,177 articles classified; 1,714
blacklisted (unclassified 1,259, case report 337, correction/erratum/retraction 97, guideline/consensus 21).
Output `LCS/article_blacklist.csv` (pmcid, title, reason, basis, added). No later stage reads it yet.

<!-- Append new entries below. Format: **RL-nnn · date/time · stage · TYPE · LIVE** then command,
host, job ID, script sha256, inputs, outputs, outcome, deviation reference. -->
