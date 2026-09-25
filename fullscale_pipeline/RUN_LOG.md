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

**RL-028 · 2026-09-25 13:57 · F1 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/append_blacklist_preprints.py` (sha256 `0b2adc269c48`). Rule: an article is a preprint
only if its header says `Article version: preprint` AND its journal is a preprint server. Reasons: avoids
duplicates of published papers; preprint repositories are excluded by agreement with colleagues.

**RL-029 · 2026-09-25 13:57 · F1 · RUN · LIVE**
`python3 append_blacklist_preprints.py` on `incline`, no SLURM, 35 s. Backup first:
`LCS/article_blacklist.before_preprints_20260925.csv`. Appended 1,475 rows with reason `preprint` (bioRxiv 657,
medRxiv 408, ArXiv 222, Research Square 188); none was already on the list. The header status, the journal
and (for the 1,144 articles with a May lookup) PubMed's Preprint type agree with no exceptions. List now
3,189 rows, 3,189 distinct articles. No later stage reads it yet.

**RL-030 · 2026-09-25 14:02 · F1 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/fetch_pubmed_types_unclassified.py` (sha256 `140dd1cccd80`): reads the PMID from each
text header and queries PubMed esummary for publication types. Key read from `NCBI_API_KEY`, not stored.
No contact e-mail is sent; only a tool name.

**RL-031 · 2026-09-25 14:02:41–14:02:52 · F1 · RUN · LIVE**
`python3 fetch_pubmed_types_unclassified.py` on `incline`, no SLURM, with the API key, 6 requests of 200.
Input: the 1,098 blacklisted "unclassified" articles never looked up in May. 1,006 had a PMID in the header and
all returned publication types; 92 had no PMID. Output `PP/pubmed_types_unclassified_20260925.csv`.

**RL-032 · 2026-09-25 14:03 · F1 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/refine_blacklist_unclassified.py` (sha256 `6c4defcb9e3e`). Rules by the project owner:
no PubMed record or type Address stay on the list; blacklisted categories stay with an updated reason; any other
type is admissible and is removed.

**RL-033 · 2026-09-25 14:03 · F1 · RUN · LIVE**
`python3 refine_blacklist_unclassified.py` on `incline`, 2 s. Backup first:
`LCS/article_blacklist.before_unclassified_refine_20260925.csv`. Of 1,259 unclassified: 952 admissible and removed
(research article 721, review 172, editorial 28, letter 13, systematic review 11, perspective/news 6, protocol 1;
written to `LCS/article_blacklist_removed_20260925.csv`); 252 stay as `no_pubmed_record` (160 from the May
lookup plus 92 with no PMID in the header); 40 stay as `case_report`; 14 as `correction_erratum_retraction`;
1 as `address`. List now 2,237 rows, all distinct articles. No later stage reads it yet.

**RL-034 · 2026-09-25 14:12 · F2 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/title_levenshtein_duplicates.py` (sha256 `a4bcd509b52c`); new package `rapidfuzz` 3.14.5
installed into `tensor_env` (`pip install --no-deps`). Created in the RDS directory: the script and the folder
`llm_corpus_staging/title_screen/`.

**RL-035 · 2026-09-25 14:13–14:14 · F2 · RUN · LIVE**
`python3 title_levenshtein_duplicates.py --workers 8` on `incline`, no SLURM, 63 s. 43,940 articles not on the
blacklist, 43,921 with a usable title. 825 pairs at normalised distance 0.30 or less (distance 0: 46; up to
0.05: 83; up to 0.10: 134; up to 0.15: 235; up to 0.20: 319; up to 0.25: 476). 101 clusters (219 articles) at 0.10
or less. Random pairs: median distance 0.76, lowest 1% at 0.66. No article was blacklisted from this result.

**RL-036 · 2026-09-25 14:36 · F2 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/resolve_duplicate_pairs.py` (sha256 `06b6bc5a0975`). Rules by the project owner: keep
originals and blacklist comments and replies; blacklist both members when authors are the same or partly
overlapping; blacklist articles with no PubMed record; erratum and retracted-and-republished links blacklisted;
for different authors, compare the openings and blacklist both members at low distance. "Blacklist" and
"remove" mean the same. Two dry runs preceded the apply run; they changed the rules once (a PubMed type alone
counts as a comment signal only when the titles are within 0.15) and added title patterns.

**RL-037 · 2026-09-25 14:37 · F2 · RUN · LIVE**
`python3 resolve_duplicate_pairs.py --apply --intro-threshold 0.5` on `incline`, no SLURM, 18 s, with the API key
(PubMed efetch, 6 requests). Backup first: `LCS/article_blacklist.before_duplicates_20260925.csv`. 825 pairs from
the title screen: comment or reply 232, no PubMed record 29, different authors 25, same authors 16, overlapping
authors 11, retracted-republished 1, no signal 511 (series and shared templates, left alone). Different-author
opening distances: 0.40, 0.41, then 0.70 and above; threshold 0.5 blacklisted two pairs (conference-abstract
editions). Appended 293 rows: comment_or_reply 194, no_pubmed_record 42, same_authors 29, overlapping_authors 22,
similar_introduction 4, retracted_republished_or_reprint 2. List now 2,530 rows, all distinct articles.
Decisions file: `LCS/title_screen/duplicate_decisions_v1.csv`. No later stage reads the blacklist yet.

**RL-038 · 2026-09-25 14:46 · F2 · CODE-CHANGE · LIVE**
`resolve_duplicate_pairs.py` edited (sha256 now `5a53f637e7eb`; RL-036 version `06b6bc5a0975`). The PubMed-link, no-record
and author rules now reach title distance 0.30 (before: 0.15); the introduction check and the PubMed-type comment
signal stay at 0.15; new `--tag` option for output names. Reason: the project owner asked for the 32 same or partly
overlapping-author pairs and the 15 no-record pairs in the 0.15 to 0.30 band to be blacklisted as well.

**RL-039 · 2026-09-25 14:48 · F2 · RUN · LIVE**
`python3 resolve_duplicate_pairs.py --apply --tag v2 --intro-threshold 0.5` on `incline`, no SLURM, 13 s, with the
API key. Backup first: `LCS/article_blacklist.before_duplicates_v2_20260925.csv`. Appended 68 rows (overlapping
authors 31, same authors 22, no PubMed record 15). List now 2,598 rows, all distinct articles. Decisions file:
`LCS/title_screen/duplicate_decisions_v2.csv`. No later stage reads the blacklist yet.

**RL-040 · 2026-09-25 18:24–18:34 · F1/F2 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/article_blacklist.py` applies every blacklist rule in one command (R1 preprints, R2 paper
types, R3 no usable type, R4 duplicate screen and pair decisions, R5 comment-like titles) and replaces the six
scripts of RL-026 to RL-039 and the heading screen. Rules: PIPELINE.md, stage rows F1 and F2. Differences from the
incremental build, all agreed with the project owner: a preprint is recognised by the header status alone; comment-like
titles are blacklisted with or without a partner; same or overlapping-author pairs are blacklisted only as definite
copies (50% shared phrases), the rest are left pending for a test on the focal windows; PubMed types now come from one
fresh esummary lookup of every article, cached. First version (sha256 `e5d422ede9c3`, 18:24): its output was discarded
because a generic PubMed "Journal Article" beat a specific Subjects label, contrary to the stated rule (79 type reasons
differed from the previous list). Corrected version sha256 `27f6f58fec51` (18:34).

**RL-041 · 2026-09-25 18:24–18:36 · F1/F2 · RUN · LIVE**
`python3 article_blacklist.py --out ../llm_corpus_staging/article_blacklist_consolidated.csv --compare
../llm_corpus_staging/article_blacklist.csv --workers 8` on `incline`, no SLURM, with the API key. The first run
built the PubMed cache (45,398 esummary records, about 9 minutes, and 951 efetch records); the corrected run took 92 s
from the cache. 46,177 articles, 824 title pairs at 0.30 or less, 2,683 blacklisted, 43,494 remaining. By reason:
preprint 1,475; case report 447; no PubMed record 256 (article level) and 57 (in pairs); comment or reply 194 (in
pairs) and 83 (unpaired); correction/erratum/retraction 112; guideline/consensus 39; definite copies 13; similar
introduction 4; retracted-republished 2; address 1. Against the previous list (2,598): 94 only in the old one (90
same or overlapping-author articles now pending, 4 PubMed changes), 179 only in the new one (83 unpaired comments, 72
case reports and 20 guideline/consensus items found through the Subjects line, 4 with no PubMed record). 49
same or overlapping-author pairs are `pending_window_check`. The output was then renamed to
`LCS/article_blacklist.csv`; the previous list is kept as `LCS/article_blacklist.previous_incremental_20260925.csv`.
Other outputs: `LCS/blacklist_decisions.csv`, `LCS/blacklist_summary.json`, `LCS/blacklist_diff.csv`,
`LCS/blacklist_cache/`. No later stage reads the blacklist yet.

**RL-042 · 2026-09-25 18:57 · F3 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/author_rules_expansion.py` (sha256 `54ac879abe13`): author-based rules appended to the existing
blacklist so that `article_blacklist.py` need not be rerun now. Rules by the project owner: articles with no authors in
PubMed, with only group or consortium authors, and Springer conference chapters whose author block lists the volume
editors are blacklisted; one-author articles are kept. Two additions of mine, both small: articles with no PubMed record
and no author line in the header (12), and the editor detector (header Journal ID is an ISBN, 33 articles). The PubMed
author cache (43,053 records from an esummary lookup made earlier the same day) was copied to
`LCS/blacklist_cache/pubmed_esummary_authors.json`.

**RL-043 · 2026-09-25 18:58 · F3 · RUN · LIVE**
`python3 author_rules_expansion.py` (dry run) then `--apply` on `incline`, no SLURM, 6 s, one PubMed request. Backup first:
`LCS/article_blacklist.before_author_rules_20260925.csv`. 43,494 whitelist articles; 116 blacklisted: no authors in
PubMed 50, volume-editor book chapters 33, group-only 21, no author line in the header 12. List now 2,799 rows, all
distinct articles; 43,378 remain. Decisions file `LCS/author_rules_decisions.csv`. No later stage reads the blacklist yet.

**RL-044 · 2026-09-25 19:06 · F4 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/build_exclusion_list.py` (sha256 `fb8362059530`). It combines the article blacklist with the P0
structural status into one exclusion list and a whitelist, so later stages read one file. Reason: the structural rule
existed only as the `status` and `include` columns of `content_regions_v2.csv`, which no other script read.

**RL-045 · 2026-09-25 19:07 · F4 · RUN · LIVE**
`python3 build_exclusion_list.py` on `incline`, no SLURM, 4 s. 46,177 articles: 6,878 excluded (blacklist 2,799; P0
structure 4,079: no closing heading 2,484, headingless 1,234, no end marker 361); whitelist 39,299. Outputs
`LCS/article_exclusions.csv` and `LCS/article_whitelist.txt`. Nothing later reads them yet.

**RL-046 · 2026-09-25 20:04 · P1 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/citation_masking_v2.py` (sha256 `4b1914d65aa6`), with a built-in self-test of 68 cases. Rules and
reasons: PIPELINE.md D9. It replaces `citation_standartization_soft_masking.py` and the citation resolution of P3 to P5
in the chain; the old scripts and their outputs are untouched.

**RL-047 · 2026-09-25 20:04 · P1 · RUN · LIVE (test only)**
`python3 citation_masking_v2.py --selftest`: 68 of 68 passed. `python3 citation_masking_v2.py --limit 200 --workers 8
--outdir <scratch folder>` on `incline`, no SLURM. 200 random whitelist articles; all outputs went to a scratch folder
outside the RDS directory. Deleted: 5,540 numeric brackets, 1,115 numeric parentheses, 1,419 parenthetical
author-year groups; 575 narrative citations became tokens; 1,710 cited works in the dictionary, 82% of them matched to
the visible reference list (17 of the 200 articles have no visible list); 250 matches ambiguous. Not run on the
whole whitelist.

**RL-048 · 2026-09-25 20:25 · P1 · CODE-CHANGE · LIVE**
`citation_masking_v2.py` edited (sha256 now `8e468a30f3ea`; RL-046 version `4b1914d65aa6`). New narrative forms become one
token: an author followed by a journal and a year ("Dong et al. (Eur J Radiol, 2026)", only when a journal-like word is
present), and "et al." or multi-author names followed by a reference number ("Dong et al. [12]", number = position in
the reference list). Self-test 81 of 81. A rerun on the same 200-article sample in a scratch folder gave 1,222 narrative
tokens (575 before), of which 477 cited works came from the "Author [n]" form. Still not run on the whole whitelist.

**RL-049 · 2026-09-25 20:33 · P0 · CODE-CHANGE · LIVE**
`build_content_regions.py` (sha256 now `64e959d673a0`; RL-024 version `16b485d1f3ea`): extra windows are capped (next heading of any
kind, 60 lines after their heading, or the References heading). Reason and effect: PIPELINE.md D10.

**RL-050 · 2026-09-25 20:33–20:36 · P0 · RUN · LIVE**
`python3 build_content_regions.py` on `incline`, 2 min 22 s. The previous output was first renamed to
`PP/content_regions_v2.before_region_cap.csv`. 46,177 rows; every column except `regions_json` is identical, so main windows, statuses and
the whitelist are unchanged (F4 not rerun). `regions_json` changed for 2,637 articles; extra windows 10,485 to 9,120; longest 61 lines.

**RL-051 · 2026-09-25 20:36–21:02 · P1 · CODE-CHANGE · LIVE**
`citation_masking_v2.py` (sha256 now `cee75d17e1d4`; RL-048 version `8e468a30f3ea`): superscript rule (evidence and vetoes in the
script's docstring), deletions that keep line breaks (before, 20 of 495 regions lost up to 8 lines), heading lines skipped by the
superscript rule, removal of empty parentheses left by deleted citations, and a fix for author names starting with letters outside
Latin-1 (the first full run crashed on "Šaltenis" after 4 seconds; its partial outputs were deleted with the project owner's approval).
Self-test 116 of 116. New input file `PP/citation_lexicon_v1.json` (sha256 `da5024dfdaa3`, built by `--build-lexicon` in 33 s).

**RL-052 · 2026-09-25 21:03–21:09 · P1 · RUN · LIVE**
`python3 citation_masking_v2.py --workers 8` on `incline`, no SLURM, 373 s. The previous complete run (20:54–21:00, script sha256
`81a6ccf60932`, without the empty-parenthesis cleanup) was renamed to `*.previous` and is kept until its deletion is approved. Input:
`LCS/article_whitelist.txt` (39,299) with `PP/content_regions_v2.csv`. Output in `LCS/`: `masked_corpus_v2/` (47,619 region files, exactly the
regions P0 defines; 0 regions changed their line count), `citation_dictionary_v2.jsonl` (412,616 cited works; 355,911 of 397,650 matched to the
visible reference list in articles that have one, 90%; 3,295 articles have no visible list), `citation_marks_v2.jsonl` (283,362 deleted
parenthetical citations), `superscript_residuals_v2.jsonl` (17,570 weak superscripts for post-processing), `citation_masking_v2_summary.json`.
Removed: 1,088,039 numeric brackets, 188,086 numeric parentheses, 283,362 parenthetical author-year groups, 320,707 superscripts (11,956
articles cite by superscript); 213,485 narrative citations became tokens. Old P1 outputs and P3 to P5 are untouched. P2 does not read
the new files yet.

**RL-053 · 2026-09-25 21:15–21:23 · P1 · CODE-CHANGE · LIVE**
`citation_masking_v2.py` (sha256 now `2d378caabeb6`; RL-051 version `cee75d17e1d4`): a token that touches a letter or digit in the raw text
gets a space, so it no longer fuses with the next word ("a9761198r25A new", "a13245614r71and"). Found by a line-by-line alignment test of the
RL-052 outputs (293 of 9.2 million lines were not the raw line with deletions only; 143 articles). Self-test 119 of 119.

**RL-054 · 2026-09-25 21:23–21:29 · P1 · RUN · LIVE**
`python3 citation_masking_v2.py --workers 8`, 370 s, on `incline`. Same counts as RL-052 (39,299 articles, 47,619 region files, 412,616 cited works;
deletions and tokens unchanged). The RL-052 outputs were renamed to `*.run2103`; the first run's outputs remain as `*.previous`. Alignment test of
every masked region against its raw region (read-only): line counts equal in all 47,619; no region ends after its reference heading; the first line
is identical in all 47,619 regions; the last line is identical in 47,610 and differs by deletions only in 9; of the last non-empty lines 44,111 are
identical and 3,508 differ by deletions only; 46 of 9.2 million lines are not deletions-only, all of them inserted commas (47) or one curly possessive,
from tokens joined for a citation with several years. This run supersedes RL-052.

<!-- Append new entries below. Format: **RL-nnn · date/time · stage · TYPE · LIVE** then command,
host, job ID, script sha256, inputs, outputs, outcome, deviation reference. -->
