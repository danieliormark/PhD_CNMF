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

**RL-055 · 2026-09-25 · P1 · CLEANUP · LIVE**
Deleted, with the project owner's approval, the two superseded complete runs of P1 v2 outputs: `LCS/masked_corpus_v2.previous/` (the 20:54 run) and
`LCS/masked_corpus_v2.run2103/` (the 21:03 run), about 11 GB each, together with the matching `.previous` and `.run2103` copies of
`citation_dictionary_v2.jsonl`, `citation_marks_v2.jsonl`, `superscript_residuals_v2.jsonl` and `citation_masking_v2_summary.json`. They were replaced
by the run of RL-054 (21:23) and had no diagnostic value. The final outputs are intact (47,619 masked files). The sizes named earlier in the session were wrong.

**RL-056 · 2026-09-25 21:40–22:14 · P1b · CODE-CHANGE · LIVE**
New `pmc_preprocessing/nonprose_removal_v1.py` (sha256 `098d19ecbfde`). Rules and reasons: the script's docstring and PIPELINE.md stage row P1b.
Built from a survey of 500 articles (16% of the main-window lines are table rows) and a small graphbrain probe (a parenthetical such as "(p < 0.05)"
adds a junk argument to the main relation; "L = a + b" pollutes the subject; a bare "≥" becomes an atom while "at least" parses cleanly).
Four audit rounds on scratch samples changed the rules: equation-like lines need a real operator ("2-13b-chat" and "Run 2/3" had been blanked),
scale legends such as "(1=extremely likely)" and "(correct=1)" are kept, decimals without a leading zero ("p < .001") and fit indices (CFI, TLI,
CMIN/DF, chi-square with degrees of freedom) are removed, and space-free tokens with a Unicode math operator are removed. Self-test 85 of 85.

**RL-057 · 2026-09-25 22:11–22:16 · P1b · RUN · LIVE (tests only)**
Runs on random samples of the whitelist (300 articles, then three of 600), all outputs in a scratch folder outside the RDS directory. Last sample:
726 files, 137,439 lines; no line count changed; 0 cleaned lines that are not the masked line with removals only; 17.7% of the lines that mention
ChatGPT, GPT-4, LLM or BERT changed (mostly table rows and statistics). Not run on the corpus.

**RL-058 · 2026-09-25 22:20–22:50 · P1b · CODE-CHANGE · LIVE**
`nonprose_removal_v1.py` edited (sha256 now `282bc1d24475`; RL-056 version `098d19ecbfde`), after the project owner asked why symbols and captions were kept. Added: caption lines
("Table 3", "Fig. 1", "Figure 2" followed by a capitalised word) are blanked while running sentences ("Table 5 shows ...") stay (in 600 articles 2,800 caption-like lines and
519 running sentences); symbols become words or disappear (degrees, micro-units, squared, cubed, times; "±" deleted); "=" deleted inside formula fragments and "equals" only
for plain statements; hexadecimal codes deleted. Order fixed so equation lines are recognised before symbols are rewritten; web addresses are protected. Self-test 107 of 107.

**RL-059 · 2026-09-25 22:40–22:50 · P1b · RUN · LIVE (tests only)**
Fresh random samples of 700 articles into scratch folders outside the RDS directory. Last: 849 files, 163,001 lines, no line count changed, no line that is not the masked
line with removals and inserted words; symbols left: "=" in 36 lines (inside web addresses); 268 "equals" conversions (539 before the last tightening). Not run on the corpus.

**RL-060 · 2026-09-25 22:49–22:51 · P1b · RUN · LIVE**
`python3 pmc_preprocessing/nonprose_removal_v1.py --selftest` (107 of 107), then `python3 pmc_preprocessing/nonprose_removal_v1.py --workers 8` (host incline, about 2 minutes;
console log `PP/nonprose_removal_v1_run.log`). Script sha256 `282bc1d24475`. Input: `LCS/masked_corpus_v2/` (47,619 files, 39,299 articles), raw texts, `PP/content_regions_v2.csv`.
Outputs: `LCS/clean_corpus_v1/` (47,619 files), `LCS/nonprose_removal_v1_summary.json`, `LCS/nonprose_examples_v1.jsonl`.
Checked afterwards: 0 files with a changed line count against the masked file, no missing file. Words 255,410,914 -> 231,282,656 (9.4% removed).
Removals by rule: T1 table rows 1,308,321; T2b captions 152,676; T2 label-only lines 5,296; T3 LaTeX 251,168; T4 dollar math 77; T5 math-symbol tokens 56,303, sub/superscripts and commands 12,053,
unspaced formulas 72,223; T6 statistics 395,487; T7 comparison words 27,518; T8 enumeration lines 32,154, equation-like lines 17,194, number lines 229; T9 symbols 195,160.
URLs and DOIs were kept (open decision, PIPELINE.md §6 item 21).

**RL-061 · 2026-09-26 · P1c · CODE-CHANGE · LIVE**
New `pmc_preprocessing/nonprose_extra_rules_v1.py` (sha256 `d34e1c6f9d81`), stage P1c, a complement to P1b (D11). Written after a survey of the cleaned corpus found
about 2,425 abbreviation-list lines, about 1,541 table-footnote-like lines and 246 "Alt text:" lines with no removal rule; most footnote lines sit directly after a table
block (839 of 1,171 "Note/Abbreviations" lines in 6,000 articles). Rules T10, T11, T12 are in the script's docstring. Self-test 18 of 18, including the cases that must stay
(an inline gloss "(C: hypercalcemia; R: renal failure)", "Category 1: mild; Category 2: moderate", the same footnote sentence away from a table).

**RL-062 · 2026-09-26 · P1c · RUN · LIVE**
Test first: 700 random articles into a scratch folder outside the RDS directory (97 abbreviation lists, 10 alt-text lines, 257 footnotes; the removed lines were read, one
mixed footnote paragraph was flagged and judged correct). Then `python3 pmc_preprocessing/nonprose_extra_rules_v1.py --workers 8` (host incline, 19:01 to 19:06, 4.5 minutes).
Input `LCS/clean_corpus_v1/` (47,619 files). Outputs `LCS/clean_corpus_v2/` (47,619 files), `LCS/nonprose_extra_v1_summary.json`, `LCS/nonprose_extra_examples_v1.jsonl`.
Removed: T10 5,655, T11 246, T12 17,327 lines. Checked afterwards: 0 files with a changed line count, no missing file, every changed line became empty (23,228 lines),
words 231,282,656 to 230,803,527 (0.21%). Lines still matching the survey patterns: abbreviation lists 2,425 to 670, table-footnote-like 1,541 to 529.

**RL-063 · 2026-09-26 · P2 · CODE-CHANGE · LIVE**
New `pmc_preprocessing/focal_terms.py` (sha256 `ae5962450379`; word list and guard rules; self-test 18 of 18) and `pmc_preprocessing/focal_extraction_v2.py` (sha256 `78e3e9e5599e`; sentence extraction), D12.
`focal_words.txt` regenerated from `focal_terms.py --write-flat-list`: 185 to 208 terms, sha256 `5a163624bf28` to `87abdc416191`; the old list is kept as `focal_words.before_v2_20260926.txt`.
This is a deviation from the list that P6 and G3 read. Built after a review of the original P2 script and list (findings in the local diagnostics log): no traceability, whole-text splitting that glued headings to sentences, blocks up to
8,996 words, silent skips, the wrong input text, and a list without bare BERT, GPT, Gemini, Llama and "language model". Test runs on 300 and 600 random articles in scratch folders outside the RDS directory (0 errors, 0 duplicate sentence
ids, 97 to 98% of sampled sentences found on their raw line by text, the rest edited by masking or cleaning).

**RL-064 · 2026-09-26 · P2 · RUN · LIVE**
`PYTHONPATH=<scratch numpy 1.26.4> python pmc_preprocessing/focal_extraction_v2.py --workers 8` (host incline, 19:08 to 19:22, 803 s; env `tensor_env` with numpy 1.26 from a scratch folder, item 17; console log `PP/focal_extraction_v2_run.log`).
Input `LCS/clean_corpus_v2/` (47,619 files, 39,299 articles). Outputs `LCS/focal_extractions_v2.jsonl` (34,662 articles, 268,532 blocks, 1,297,393 sentences of which 684,366 focal; 637 MB), `LCS/focal_status_v2.csv`, `LCS/focal_errors_v2.jsonl` (empty),
`LCS/focal_extraction_v2_summary.json`. Outcome: 34,662 ok, 4,637 without a focal sentence, 0 errors, 0 duplicate sentence ids. Blocks: median 83 words, 95% under 302, longest 2,009; 5.1% over 300 words; median 3 sentences, longest 98; 267,151 blocks in
main windows and 1,381 in extra windows. Hits inside headings skipped: 20,614; hits rejected by the guard rules: 10,050. Afterwards, on a 1% sample (360 articles): 5,810 of 5,899 sentences found on their raw line by text (98.5%), all 32 extra-window
sentences, and all 5,899 hashes verify. Most frequent terms: ChatGPT 177,079, LLMs 166,953, LLM 133,156, BERT 57,413, GPT-4 37,039, "language model" 35,795, Gemini 28,801, GPT 26,760.

**RL-065 · 2026-09-26 · P6 · CODE-CHANGE · LIVE (not yet run on the corpus)**
New `pmc_preprocessing/coref_resolution_v2.py` (sha256 `5a1e21746d2b`) and `pmc_preprocessing/submit_coref_v2.sh` (sha256 `77ef164cc5ee`, CSF array job of 100 shards); new folder
`pmc_preprocessing/logs_coref_v2/`. Rules and reasons in D13. Built from a review of P6 v1 and scratch experiments recorded in the local diagnostics log (D-027 to D-031): spaCy and LingMess
agree on 94 of 97 pronoun resolutions in 143 blocks; the apposition handling comes from the dependency parse and gives identical results with either model's clusters on 15 hard sentences.
Tests with scratch outputs outside the RDS directory: self-test 10 of 10 (including the owner's ChatGPT apposition example, a pronoun chain over four sentences, an antecedent found only in
the context, a partitive, two entities, a demonstrative, a noun phrase); 30 articles in 2 shards and merged: 0 errors, ids, hashes, raw lines and text identical to P2, every resolved sentence
equal to its original plus the listed replacements (no context text in the output), 175 of 243 pronouns replaced, 38 rejected (models disagree 16, no noun antecedent 11, number 10,
LingMess not clustered 1); 6 articles to test resume and the merge checks (missing shard refused, rerun without --resume refused, resume skipped finished articles, merge passed).
Context changed afterwards from two preceding paragraphs to two preceding sentences (owner decision); self-test rerun, not re-run on a sample. Environment check on a CSF compute
node (job 21401159, node1223): all models load and run with `PYTHONPATH=$HOME/np1_for_spacy` (item 17).

**RL-066 · 2026-09-27 · P6 · RUN (trial) · LIVE**
CSF trial of the array job: `sbatch --array=0-1 --export=ALL,NSHARDS=2,LIMIT=20,SAMPLE=1,OUTDIR=LCS/coref_v2_trial pmc_preprocessing/submit_coref_v2.sh` (job 21401371, node1235, 4 cores per task, CPU,
script sha256 `5a1e21746d2b`). Both tasks exited 0 (10 articles each, 180 s and 509 s). Merge: `python pmc_preprocessing/coref_resolution_v2.py --merge --nshards 2 --limit 20 --sample 1 --outdir LCS/coref_v2_trial`,
passed (20 articles, all present once, no errors). Outputs in `LCS/coref_v2_trial/` (3.9 MB; a test folder, not the run's output folder). 160 blocks, 730 sentences: 83 of 120 pronouns replaced
(6 with an antecedent in the context sentences), 26 rejected (models disagree 11, number 10, LingMess not clustered 3, no noun antecedent 2), 206 noun-phrase links, 5 demonstratives in clusters.
Verified against the P2 records: for all 730 sentences sid, hash, raw line, text, focal flag and terms are identical, hash_resolved verifies, and every text_resolved equals the original text plus the
listed replacements. Speed on CSF: 686 s for 20 articles (median 11 s, longest 214 s per article), about 34 s per article per 4-core task; the full run is therefore expected to take roughly 3 to 4 hours
per shard with 100 shards.

**RL-067 · 2026-09-27/28 · P6 · RUN · LIVE**
Full CSF array run: `sbatch pmc_preprocessing/submit_coref_v2.sh` (job 21401638, 100 tasks, 4 cores, 16 GB, 12 h, script sha256 `5a1e21746d2b`), started 2026-09-27 00:38, all 100 tasks exited by
2026-09-27 ~04:20. 99 of 100 exited 0; task 86 was OOM-killed (exit 137) partway through (125 of 346 articles done) on `PMC12211737`, whose largest block is 2,009 words (the largest in the corpus;
measured peak memory 17.5 GB on incline, of which 6.5 GB is the four loaded models). Resubmitted `sbatch --array=86 --mem=32G pmc_preprocessing/submit_coref_v2.sh` (job 21489427, 2026-09-28
13:57 to 15:56, 8,628 s), `--resume` skipped the 125 done articles and completed the remaining 221; exit 0.

Merge: `python pmc_preprocessing/coref_resolution_v2.py --merge --nshards 100 --outdir LCS/coref_v2` (2026-09-28), passed: 34,662 articles, all present once, no errors. Outputs in `LCS/coref_v2/`
(3.3 GB): `coref_resolved_v2.jsonl`, `coref_np_links_v2.jsonl` (340,271 rows), `coref_rejected_v2.jsonl` (48,906 rows), `coref_status_v2.jsonl`, `coref_errors_v2.jsonl` (empty),
`coref_v2_summary.json`. 268,532 blocks, 1,297,393 sentences; of 285,941 personal pronouns in blocks, 203,493 replaced (11,445 with an antecedent in the two context sentences), 48,906 rejected
(models disagree 20,182, number mismatch 18,002, no noun antecedent 7,694, LingMess did not cluster 3,028); 34,025 demonstratives left unresolved in clusters; 340,271 noun-phrase mentions linked
without changing the text.

Verified afterwards: a 1% random sample (355 articles, 13,685 sentences) checked against the P2 records — sid, hash, raw_line, text, focal flag and terms identical; hash_resolved verifies;
every text_resolved equals the original text plus the listed replacements (no context text in the output) — 0 problems found.

**RL-068 · 2026-09-28 · P7 · CODE-CHANGE + RUN · LIVE**
New `pmc_preprocessing/focal_citations_v2.py` (sha256 `49e9a3a64c5b`), stage P7 v2, D14. Built after a check that P1 v2's `doi` field is empty (malformed DOI pattern, item 27) and that only 12,644 citation tokens occur in
focal sentences. Tests before the run (scratch outputs outside the RDS directory, local diagnostics log D-032): self-test 21 of 21; audit of 15 random works with two or more DOIs (all one paper each);
three defects found and fixed (truncated DOIs merging unrelated papers, DOIs glued to PMIDs or to their own repeat, BERT split into two works by a same-year rule).
Run: `python3 pmc_preprocessing/focal_citations_v2.py` (host incline, 2026-09-28 19:22 to 19:23, 29 s; console log `PP/focal_citations_v2_run.log`). Inputs `LCS/coref_v2/coref_resolved_v2.jsonl`,
`LCS/citation_dictionary_v2.jsonl` (not modified), raw texts, `PP/content_regions_v2.csv`. Outputs `LCS/focal_sentences_v2.jsonl` (610 MB), `LCS/citation_works_v2.jsonl` (5.7 MB),
`LCS/focal_citations_v2_summary.json`. 34,662 articles; 702,948 of 1,297,393 sentences kept (18,582 only through a coreference replacement); 14,154 citation occurrences of 12,644 tokens; 10,350 works
(5,498 identified by DOI or PMCID, 2,637 by reference text only, 2,215 unresolved and given an id of their own), 1,008 cited in two or more articles; 510 title links joined 202 works; 23 truncated DOIs
not used for linking. Verified afterwards on all 702,948 sentences against P6: sid, hash, raw line, text and resolved text identical; hash_final verifies; text_final equals the resolved text with each
token replaced by its work's id; no per-article token left; every id exists in the works file; every kept sentence has a focal term. 0 problems found.
*[Superseded by RL-069: the figures in this entry (10,350 works, 1,008 cited in two or more articles, script `49e9a3a64c5b`) come from a run with a DOI-cleaning bug; its outputs are kept as `*.before_doi_fix_20260928` and must not be used.]*

**RL-069 · 2026-09-28 · P7 · CODE-CHANGE + RERUN · LIVE (supersedes RL-068)**
Validation of the RL-068 output (local diagnostics log D-033) found a bug in `focal_citations_v2.py`: the DOI cleaner cut a final ".NNNNNNN" from every DOI to remove PMIDs glued onto DOIs, but IEEE, ACM, Frontiers and
Taylor & Francis DOIs end that way themselves, so 963 of 12,644 tokens (670 works) had a proceedings- or issue-level DOI shared by unrelated papers and different papers were merged (for example six CHI papers in one
id). Fix: only a PMID glued to a bioRxiv/medRxiv DOI (10.1101/YYYY.MM.DD.NNNNNN) is cut. New script sha256 `e9fd4bf14862` (self-test 23 of 23, including ACM and IEEE DOIs); the version of RL-068 (`49e9a3a64c5b`)
is kept as `pmc_preprocessing/focal_citations_v2.before_doi_fix_20260928.py`. The three outputs and the console log of RL-068 were renamed with the suffix `.before_doi_fix_20260928` (no file deleted) and the script was rerun:
`python3 pmc_preprocessing/focal_citations_v2.py` (2026-09-28 19:40, 29 s). Outputs `LCS/focal_sentences_v2.jsonl`, `LCS/citation_works_v2.jsonl`, `LCS/focal_citations_v2_summary.json`. Same 702,948 sentences and 12,644 tokens as
before; 10,555 works (was 10,350: 5,696 identified by DOI or PMCID, 2,644 by reference text only, 2,215 unresolved), 964 cited in two or more articles (was 1,008, the difference being false merges), 563 title links
joining 226 works, 167 truncated DOIs not used for linking (was 23).
Checks repeated on the real output: all 702,948 sentences against P6 (ids, hashes, raw lines, text; text_final rebuilt from the resolved text; no per-article token left; every id in the works file; every kept
sentence has a focal term): 0 problems; output byte-identical to the scratch run on which the labelled samples were drawn; detector for "identifier shared by clearly different papers" (title-word Jaccard < 0.15
and different first author) with the old cleaner on the old output 43 works / 415 pairs, with the fixed cleaner on the new output 1 work / 3 pairs; structural checks (226 works joined by title across identifier groups,
21 least-similar works read, threshold sensitivity) and labelled samples as recorded in PIPELINE.md item 28.

**RL-070 · 2026-09-28 · G2 · CODE-CHANGE + TEST · LIVE**
New `phase5_graphbrain/scripts/graphbrain_parse_v2.py` (sha256 `e14d9ada997f`), `submit_g2_v2.sh` (`b4ef2b872a31`) and the CSF check `check_csf_g2.sh` (`0523eec595e8`), stage G2 v2, D15. New log folder `PG/scripts/logs_g2_v2/`; no existing file changed.
Tests on incline (tensor_env with `PYTHONPATH=$HOME/np1_for_spacy`; outputs in the session scratch folder, not in the RDS directory): graphbrain 0.7 with spaCy 3.4.4 / `en_core_web_trf` 3.4.0 loads; without `parser.atom2token = {}` every sentence fails inside graphbrain and comes back without an edge. 200 random focal sentences: 209 units (9 sentences split in two by graphbrain), all with an edge, every edge and lemma edge restored identically from its string, 0.08 s per sentence (4 threads), 3.3 GB peak. `check_csf_g2.sh` run directly on incline: ALL G2 CHECKS OK (150 sentences incl. the longest, 548 words, parsed as one unit; `REF000053` parses as `ref000053/Cp.s`). Script trial on the first 40 articles (770 sentences, 3 shards): all 770 sentences ok, 807 units, 0 error lines; a task killed mid-run with a torn last line resumed and gave output byte-identical to an uninterrupted run; the merge refused while a shard was unfinished and refused to overwrite its output. Full run on CSF: pending (to be submitted by the owner after `check_csf_g2.sh` passes on CSF).

**RL-071 · 2026-09-28 · G2 · CSF CHECK + CODE-CHANGE + TEST · LIVE**
CSF check `sbatch phase5_graphbrain/scripts/check_csf_g2.sh` (owner, job 21504977, node1221): ALL G2 CHECKS OK; 0.26 s per sentence, 3.3 GB peak, estimate about 1 h per task with 50 tasks.
Owner request: keep the graphbrain SQLite output that G3 reads. `graphbrain_parse_v2.py` (now sha256 `3eec8b0b256c`) builds per shard `PG/g2_v2/shards/db_provenance_NNN.sqlite` with `('source', PMCID, main_edge)` and the lemma edges; each source edge carries the attribute `occurrences` (sid, hash_final, unit, text, atom-to-word positions); built on the node's local disk in one transaction and copied, then its source-edge count is checked (also by --merge). The JSONL keeps the same content (and now atom2word). Recursion limit raised to 20,000, round-trip check protected, maximum edge depth recorded per shard, edges deeper than 100 levels noted. `submit_g2_v2.sh` comment updated (sha256 `c279b391f6cf`).
Checks of the May outputs (read-only): G2 (job 15772611) parsed 22,601 of 22,795 articles (all 456 in 49 shards; shard 48 262 of 456); all 826 lines of its error logs are SQLite `disk I/O error`s in shard 48 (195 articles). G3 (`db_curated_NN.sqlite`, 2026-05-30) holds only 359 articles in total (1 to 50 per shard). A scratch copy of `chunk_4h_hpc.py` (output path changed only) rerun on the finished `db_provenance_10.sqlite` crashed with a RecursionError inside `hg_raw.search(('source','*','*'))`, outside the per-sentence try block, and wrote 0 articles: the May source edges reach depth 383 (keys up to 15,070 characters) because whole blocks were parsed. G3 must raise the recursion limit and guard the loop before it is rerun (§3 G3 row).
Tests of the revised G2 (scratch folder): 40-article trial in 3 shards: 807 units, 806 source edges (one edge occurs twice in an article, both occurrences recorded), 0 errors; JSONL identical to the RL-070 trial; a task whose database was deleted rebuilt it on resubmission; merge refused while a database was missing. Unchanged G3 copy on the new shard-0 database: runs without error, 1,152 curated links in 8 of 14 articles (G3's focal filter decides the rest; to be reviewed at G3). 40 longest sentences of the corpus (197 to 548 words): maximum depth 93, 2 partial parses, no exception. Database size about 43 MB per 300 sentences, about 100 GB expected for the full run (7.9 TB free).

**RL-072 · 2026-09-28/29 · G2 · RUN · LIVE**
Full run of `graphbrain_parse_v2.py` (sha256 `3eec8b0b256c`) on the whole corpus, after the CSF check (RL-071, job 21504977, ALL G2 CHECKS OK) and the local trials (RL-070, RL-071).
CSF: `sbatch phase5_graphbrain/scripts/submit_g2_v2.sh`, job 21506341, array 0-49 (`multicore`, 4 CPUs, 8 GB, 6 h); all 50 tasks exited 0, running 2026-09-28 22:26 to 22:54 (about 25-35 min each, in parallel; 0.14-0.26 s per sentence on a CSF node). No task failed or needed a resubmission. Aggregated from the 50 `g2_done_NNN.json` files: 34,662 articles, 702,948 sentences, 730,292 parse units (701,237 ok, 1,711 partial, 0 no_edge, 0 exception), 725,513 database source edges, maximum edge depth 93 (May's whole-block parse reached 383, RL-071). 1,734 lines across the 50 error files (all partial-parse notes, not crashes). 50 `db_provenance_NNN.sqlite` databases, 118 GB total, in `PG/g2_v2/shards/`.
Merge (incline, tensor_env, `PYTHONPATH=$HOME/np1_for_spacy python phase5_graphbrain/scripts/graphbrain_parse_v2.py --merge --nshards 50`, 32m35s): checked all 50 shards done, every database's source-edge count matching its done file, every input article present exactly once, and every sentence id and hash matching `LCS/focal_sentences_v2.jsonl`. Output `PG/g2_v2/g2_parsed_v2.jsonl` (34,662 articles, 1.8 GB, sha256 `ad8a4bcc630c`), `PG/g2_v2/g2_v2_summary.json` (totals matching the shard aggregate exactly, including all 50 job ids). 0 problems found.

**RL-073 · 2026-09-29 · G2, G3 · DESIGN + TEST (prototype, no pipeline file changed) · LIVE**
G3 revision discussed with the owner and recorded as PIPELINE.md item 29 (agreed: recursion guard, new input and provenance, auxiliary verbs dropped only as `Mv`, hyphenated-word merge, removal of mathematical signs and table/list/equation remnants; open: focal-word filter, modal verbs, comparison with `toy_large/7.5postprocessing_4hbased_correct.py`). Checked on the G2 v2 output (read-only): main-verb be/have are typed `P`/`Pd` and auxiliaries `Mv`, so the May test `role.startswith(('P', 'Mv'))` deletes "is" from "An LLM is a type of AI"; of the modal verbs (type `Mm`) only "will" is dropped, through the stop list.
G2 sentence units (PIPELINE.md item 30): 22,091 of 702,948 sentences were split by graphbrain in G2 v2; 2,032 sentences (2,164 places) hold a citation number glued to a sentence-final period. Prototype `diagnostics/g2_prototype/sentence_units_v2.py` (local, not in the RDS directory; v1 kept as the record of a rejected first rule). Tests: two samples of 120 graphbrain cuts labelled by hand (72 and 68 real); the first rule (keep a cut only if `en_core_sci_sm` also finds it) kept 3/72 and 4/68 real boundaries and was rejected as circular; this also withdraws the "271 of 271 successful merges" reported in the session on 2026-09-29, which counted valid trees rather than correct segmentation. The punctuation rule, refined on sample 1 and frozen, scored on sample 2: real kept 58/68, false dropped 50/52, 12 errors (keep-all 52, `en_core_sci_sm` rule 64). Glued-citation detector: 59/60 sampled cuts correct, then section numbers ("A.3") excluded. Self-test 25/25. Against G2 v2 on 1,147 random, 300 split and 300 glued-citation sentences: 1,131/1,147 random sentences byte-identical, 207 kept and 171 dropped cuts in the split sample, all glued samples changed, 0 parse failures, 0 exceptions, 0 round-trip failures, all preset boundaries respected, maximum depth 20. Nothing was written to the RDS directory. Next step (G2 v3 script and CSF rerun) awaits the owner's go-ahead.

**RL-074 · 2026-09-29 · G2 · CODE (new script) + TEST · LIVE**
New `phase5_graphbrain/scripts/graphbrain_parse_v3.py` (sha256 `7a9ff453771e`), `submit_g2_v3.sh` (`09da9d8ea668`) and the CSF test `check_csf_g2v3.sh` (`24f91cd5b287`); new log folder `PG/scripts/logs_g2_v3/`. Scope as agreed with the owner (PIPELINE.md item 30, scope paragraph; G3 parts in item 29). Placeholder test: URL, WEB, LINK and REF-style tokens all parse as ordinary concept atoms (C/Cc/Cp); URL + 8 digits chosen. Hyphen test on 150 sentences: no parse failures; hyphenated verbs lose their object when not joined. URL test on 150 sentences with a URL or DOI: no failures either way; unjoined URLs become one escaped atom each. Self-test 54/54. Trial on incline (shards 0 and 1 of 50, 40 articles each, scratch folder): 1,266 sentences, 14 non-prose, 6 glued captions and 1 glued label removed, 4 glued citations, 3 URLs, 6 symbols, 886 hyphen joins, 0 errors, all ids, hashes and database counts verified, 0.08-0.09 s per sentence, 3.2 GB peak. The CSF test script was also run directly on incline (ALL G2 v3 CHECKS OK); its output folder `PG/g2_v3_csftest/job_manual/` was deleted afterwards. Next: `check_csf_g2v3.sh` on a CSF node (owner), then `--prepare` and the full run.

**RL-075 · 2026-09-29 · G2 · CODE-CHANGE + TEST · LIVE**
Owner decision: hyphenated words whose parts contain a focal term (any term of `PP/focal_words.txt`, single or multi-word) are not joined. `graphbrain_parse_v3.py` now sha256 `0e23963b5360` (also: each task computes the caption-body list itself; `--prepare` only writes it for inspection); `submit_g2_v3.sh` `a6ac2ec80db2`, `check_csf_g2v3.sh` `85838f5aa8e8` (preparation step removed). Self-test 57/57. Trial (shards 0 and 1, 40 articles each, scratch folder): 768 hyphenated words joined, 118 kept because of a focal part, 1,022 caption bodies found, verification OK, 0 errors.

**RL-076 · 2026-09-29 · G2 · CODE-CHANGE (new script) · LIVE**
Owner request: keep the shard-merge step as a separate script from the parser, for pipeline reproducibility (parsing needs graphbrain and spaCy; merging needs neither). Moved `merge()` out of `graphbrain_parse_v3.py` into a new, self-contained `merge_g2_v3.py` (sha256 `7cb4b1d7f1c9`) that re-derives every check from the raw input and the shard files (article/sentence counts, sid/hash_final match, unit id and hash verification, database source-edge count) rather than trusting the shards' own done files beyond a basic consistency check; `--verify` (used only by the CSF test) stays in `graphbrain_parse_v3.py`, now sha256 `a3d7f986d292` (the `--merge` option and its function were removed). Tests (scratch folder): a 4-article, 2-shard run (no `--limit`) merged cleanly (4 articles, 44 sentences, all checked), correctly refused to overwrite its own output on a second run, correctly refused a shard run with `--limit` (mismatched with the merge's own full-corpus expectation), correctly refused an unfinished shard (missing done file), and correctly refused a shard file with a tampered sentence id. One bug found and fixed before any of this: `max_depth` was summed into the total AND passed separately to the summary, a duplicate-keyword crash on the very first successful merge attempt. Self-test of `graphbrain_parse_v3.py` still 57/57 after removing `merge()`. Nothing in the RDS directory was touched other than the two script files and the scratch test folders (removed after). The full run (job 21557322, submitted after RL-075's CSF test passed) is unaffected by this change; it was still running when this edit was made.

**RL-077 · 2026-09-29 · G2 · RUN · LIVE**
Full run of `graphbrain_parse_v3.py` (sha256 `0e23963b5360`) on the whole corpus, after RL-075's CSF test passed.
CSF: `sbatch phase5_graphbrain/scripts/submit_g2_v3.sh`, array job 21557322 (0-49, `multicore`, 4 CPUs, 8 GB, 6 h). Only shard 49 finished under this array job itself (its own `g2_done_049.json` records job `21557322`); the other 49 shards each finished under a separate, individually-submitted job id instead (`21557693`-`21558717`, one id per shard, spread from 2026-09-29 shortly after submission to the evening — the full list is in `g2_v3_summary.json`'s `jobs` field). Cause not diagnosed here (most of the original array's tasks did not themselves produce the done file later found) — flagged for the owner rather than guessed at; every shard's own done file nonetheless matches this run's input and nshards with no `--limit`, so the merge's checks below are unaffected either way.
Merge: extracted into its own script per the owner's reproducibility request (RL-076); run directly on incline rather than through `sbatch` (the owner's own attempt failed with `sbatch: error: This does not look like a batch script` — `sbatch` needs a real script file, and the merge needs neither graphbrain nor spaCy, so it does not need CSF at all): `python3 phase5_graphbrain/scripts/merge_g2_v3.py --nshards 50` (incline, 2026-09-29 21:56 to 22:34, about 38 min). Checked all 50 shards done with matching input/nshards and no `--limit`, every database's source-edge count against its done file, every input article present exactly once with unchanged sid/hash_final, every unit's id and hash, and the url tables' agreement on shared tokens.
Output `PG/g2_v3/g2_parsed_v3.jsonl` (34,662 articles, 2.2 GB, matches `LCS/focal_sentences_v2.jsonl`'s article count exactly), `PG/g2_v3/g2_v3_summary.json`, `PG/g2_v3/url_table_v3.json` (1,674 distinct URL/DOI tokens).
Totals: 702,948 sentences (696,351 ok, 6,591 dropped as non-prose, 6 empty), 710,869 units (36 dropped), maximum edge depth 101, 2,164 glued citations removed, 2,947 URLs tokenised, 1,500 symbols normalised, 362,204 hyphenated words joined (62,652 kept apart for containing a focal term), 582 labels removed, 1,542 captions removed. 1 line across all 50 error files: `PMC12476623.r1.L92.S3.U1`, a single sentence that is a long comma-separated list of AI-related search terms, nested 101 levels deep — a genuine structural edge case, not a parsing defect. 0 problems found on verification.
G2 v3 (item 30) is now complete on the whole corpus; next is G3's revision (item 29).
*[Corrected by RL-081: all 50 shards ran in array job 21557322 (the 50 different ids are each task's own `SLURM_JOB_ID`, which the done files record), with no resubmission; the script of the run is sha256 `a3d7f986d292` (recorded in all 50 done files), not `0e23963b5360`, which is the version the CSF test job 21557049 ran.]*

**RL-078 · 2026-09-30 · G3 · CODE (new test script) + TEST · LIVE**
Comparison of `chunk_4h_hpc.py`, the toy `postprocessing_4h.py` and `7.5postprocessing_4hbased_correct.py` (PIPELINE.md item 29(j)); owner decision: G3 is built on 7.5.
New `phase5_graphbrain/scripts/g3_curation_test.py` (sha256 `2ba55f6100dc`) with the item-29 fixes (a, b, c, e, f, g, h) and provenance ids/hashes; scope in PIPELINE.md item 29, last paragraph. Checks before writing it (scratch, read-only on RDS): the two stop lists differ only in be (toy) / also (RDS); the RDS list holds "been", "am", "will"; in 3,000 articles of G2 v3, "no" is `Md` 995 times, "%" is an atom of its own, "e.g."/"i.e." are `M` and "vs." `Br`; chunk12 never reads `source_fringe`.
Tests (tensor_env with `PYTHONPATH=$HOME/np1_for_spacy`, outputs in the session scratch folder): self-test 34/34 (28 canonical-name cases, 6 synthetic units: main-verb "is" kept, auxiliaries dropped, negation, modal tag, "no"/%, stop-listed "been", multi-word focal term). Shard 0, 200 articles: 4,103 units, 0 errors, 0 provenance problems, 4,871 parents, 16,769 cousins, 5,755 focal mentions (118 surface forms -> 68 canonical atoms), 0.013 s per unit. Two changes after reading its report: the stop-list exemption narrowed to main-verb be/have/do (a wider exemption let `where/P`, `that/P`, `a/P` through), and type `Cm` kept (1,040 content atoms such as human, cancer, research were dropped). Rerun: self-test 34/34; the deepest unit of the corpus (depth 101, PMC12476623) 0 problems; shard 25, 300 articles: 5,605 units, 0 errors, 0 problems, database read back in chunk12's way. Found upstream: "LLM" (PLAIN, no guard) used for lipid-lowering medication in PMC8815195. Nothing written to the RDS data folders.

**RL-079 · 2026-09-30 · G3 · CODE-CHANGE + TEST · LIVE**
Owner decisions: ChatGPT-<version> merged into GPT-<version>; model sizes and snapshot dates dropped. `g3_curation_test.py` now sha256 `bf53c0404fb7`. Sizes and dates are removed from the surface form before separators are normalised (a first version on the normalised form turned "Llama-2-70B" into `llama`); ESM-1b, ESM-MSA-1b and ProtST-ESM-1b keep their "1b"; "Chat GPT-4.0" now merges; a token is extended beyond the matched name only by a one-digit-major version or a size, so "GPT-44", "GPT-418" (GPT-4/GPT-2 with a glued citation) read as `gpt`. Self-test 52/52. Shard 0, 700 articles: 14,225 units, 0 errors, 0 problems, 191 surface forms -> 91 canonical atoms, no two-digit version left.
Evidence for the owner (scratch): atom provenance — of 264,726 atom records, 10.7% have a source atom whose word occurs more than once in the unit, so the record lists every position of that word (graphbrain edges do not carry positions); 731 records merge several source atoms (all listed). Stop lists — the RDS and toy lists are NLTK's 198 English stop words minus 77 (negations, be/have/do and modal forms, more/most/few/same/both/each, above/below/against/under/through/until/before, we/our/ours/she) plus also/whilst; the RDS list holds "also", the toy list "be". On otherwise-kept atoms the list decides mainly and, such, also, other, as, when, only; frequent kept low-content atoms not in the list: our (778), well (444), however (391), two (342), three (294), first (282), on (220, a phrasal particle), therefore, additionally, respectively, furthermore.

**RL-080 · 2026-09-30 · G3 · CODE-CHANGE + TEST · LIVE**
Owner decision on modal verbs: an atom of the compound verb group (`dummy_sibling`, `dummy_cousin`), not a tag; verbs like "allows" stay independent predicates. `g3_curation_test.py` now sha256 `2a5e01342bf4`: `--modals keep` is the default; a modal is type `Mm` or a modal word the parser typed as another modifier ("can/M" 29, "ca/Mm" from "can't"); kept modal atoms count as part of the verb group when verb groups are joined upward or collected in the periphery. In G2 v3 (700 articles) modals are `Mm` (can 1,173, may 603, could 350, should 232) and allow/enable are `P`/`Pd`. Self-test 54/54 (new cases: "should be approached", "allows" as an independent predicate, a modal inside a `dummy_cousin`). Shard 0, 700 articles: 14,225 units, 0 errors, 0 problems; modal atoms: 1,510 in `dummy_sibling`, 1,542 in `dummy_cousin`, 349 in argument hyperedges, of which 341 also hold the verb of the embedded clause they belong to ("features of ChatGPT that a novice author can use"). Checked earlier in the session (scratch, same sample): with `keep`, distinct `dummy_sibling` hyperedges rise from 1,808 to 2,381, distinct parents from 16,043 to 16,057; focal-term replacement reached a non-focal occurrence of the same word in 3 of 19,971 mentions.

**RL-081 · 2026-09-30 · G2, G3 · DOC + CORRECTION · LIVE**
New stage documents `fullscale_pipeline/G2_PARSING.md` (G2 v3: rules, output format, provenance, run, checks, results, limitations, history) and `fullscale_pipeline/G3_POSTPROCESSING.md` (G3: target schema, choice of the 7.5 base, agreed decisions, canonical focal atoms, provenance, open questions, test script and results); PIPELINE.md links them and gains deviations D16 (G2 v3) and D17 (new G3), a stage-table row for the new G3 and an updated M1 contract. No code changed.
Checked from the files on disk while writing them (read-only):
- **CSF test of G2 v3, not recorded before:** job 21557049, `PG/scripts/logs_g2_v3/g2v3_check_21557049.out`, finished 2026-09-29 16:36: self-test, shards 0 and 1 limited to 40 articles each (617 and 649 sentences, 0 error lines, 0.38 and 0.15 s per sentence, 3.21 GB peak), `--verify`: ALL G2 v3 CHECKS OK; script `graphbrain_parse_v3.py` at that time sha256 `0e23963b5360` (edited to `a3d7f986d292` at 16:44, RL-076). Test output `PG/g2_v3_csftest/job_21557049/` is still on disk and can be deleted.
- **RL-077 corrected:** the logs `g2v3_21557322_0..49.out` show every task as "task i of job 21557322" and 50 × "exit 0" (start 16:53-17:29, end 17:20-17:59, about 0.1 s per sentence); the done files record `SLURM_JOB_ID`, SLURM's per-task id, so RL-077's "only shard 49 finished under the array job; the other 49 under separately submitted jobs" was a misreading. All 50 done files record script `a3d7f986d292` (saved 16:44, before the first task started at 16:53); RL-077 named `0e23963b5360`. The two versions differ only in the removed merge code.
- Breakdown of the merged G2 v3 output: non-prose sentences 6,591 = captions 3,865, DOI captions 1,398, caption bodies 694, reference lines 574, headings 60; dropped units 36 = debris 30, heading 3, reference line after prose 2, caption after prose 1; 2,032 sentences cut at a glued citation; units per sentence: 1 for 683,675 sentences, 2 for 11,325, 3 or more for 1,351; 708,631 database source edges; output 2.5 GB JSONL and 117 GB of shard files.
- G3 test rerun for the document (scratch): `g3_curation_test.py` `2a5e01342bf4`, shard 0 (all 694 articles), `--db`: 14,225 units, 0 errors, 0 problems, 16,983 parents, 56,989 cousins, 280 units with a focal term but no parent, database read back in chunk12's way.

**RL-082 · 2026-10-07/08 · G3 · TEST (read-only) + DOC · LIVE**
Evidence runs for the owner's decisions on G3_POSTPROCESSING.md §6 items 1–3 (passive voice, embedded clauses, lexical
modal verbs). Host incline, `tensor_env`; nothing written to the RDS data folders; no pipeline script changed.
- `g3_curation_test.py` (sha256 `2a5e01342bf4`, unchanged) on shard 0 with `--limit 60` and `--limit 300`
  (1,236 and 5,829 units, 0 problems), outputs in the session scratch folder. Used to re-verify the §6 item 2 example
  (the earlier quoted fragment was truncated; corrected in commit 857d8ba) and to count how lexical modal verbs come out
  today [LOG: this session's output].
- Lexical modal verbs (§6 item 3): first detectors (scratch scripts, shard 0, 300 articles; files lost from the session
  scratch folder before they were saved, figures as reported in §6 item 3) [INFERENCE from the session record], then
  the rule implemented in `fullscale_pipeline/g3_modal_check/modal_rules.py` (final sha256 `3f993e0ff164`; edited during
  the session as guards were added and narrowed). It reads `PG/g2_v3/shards/g2_parsed_NNN.jsonl` and recomputes every
  decision with each guard switched off. Runs: `pool` on shards 0; 20,30 (seed 7); 1–4 (all candidates); 40–42 (seed
  31); all 50 shards for force, empower, instruct, help in V-ing, allow/enable V-ing (seeds 11, 23) and whole-corpus
  sweeps for need, let and the gerund groups (`WORDS` / `SAMPLE_POOLS` filters); `hard` on the 40 constructed cases
  (`hard_cases.json`, parsed with graphbrain `create_parser(lang="en", lemmas=True)` with `PYTHONPATH=$HOME/np1_for_spacy`,
  as G2): 33/40 with the final rule. Outputs (contain corpus text, so local only):
  `fullscale_pipeline/diagnostics/g3_modal_check/` [LOG].
- Results, labels and accepted residual errors: G3_POSTPROCESSING.md §6 item 3. Decisions recorded there and in §6 items
  1–2 (commits d7d3f4d, 857d8ba and this one).

**RL-083 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Evidence for G3_POSTPROCESSING.md §6 item 4 (stop list). Host incline, `tensor_env`:
`python g3_curation_test.py --shard 0 --limit 700 --outdir fullscale_pipeline/diagnostics/g3_stoplist/shard0 --show 0`
(script sha256 `2a5e01342bf4`, unchanged): 694 articles, 14,225 units, 0 problems, 116,938 structures, 45,746 distinct
argument hyperedges. Counts per candidate class (structure occurrences, hyperedges made only of candidate words, distinct
hyperedges lost if the class were stop-listed) computed in a scratch pass over `g3_test_000.jsonl` [LOG]. Also found:
in the periphery "not" is left as a `cousin_he` of its own (355 in shard 0: "will not be known" →
`(dummy_cousin know will) (cousin_he not)`), while in parents negation joins the verb group [LOG]. Output local only
(`fullscale_pipeline/diagnostics/g3_stoplist/`); nothing written to the RDS data folders.
Follow-up tests the same day, on modified **copies** of the script in that folder (the RDS script is unchanged):
`g3_negfix.py` (negation counted as a verb-group member in `sweep`, as `extract` already does): self-test 54/54; shard 0:
parents unchanged, standalone negation cousins 360 → 17, negation in `dummy_cousin` 12 → 553 [LOG]. `g3_q4.py`
(`ONLY_MODE=keep`: "only" off the stop list; `vg`: also a verb-group member like negation): "only" alone as a hyperedge
111 (keep) vs 12 (vg); in verb groups 130 (vg) [LOG]. Scratch passes over shard 0 for "overall / finally / notably /
similarly" (60 labelled occurrences), "as well (as)" (217 occurrences by context) and single-adverb hyperedges (3,894;
50 labelled with the verb they would attach to) [LOG].
Second round (2026-10-08, same folder): `hard_g3.py` runs the current script, `g3_negfix.py` and `g3_q4.py`
(`ONLY_MODE=vg2`: "only" joins the verb group only when a predicate follows it, directly or after one adverb, and not
after "if") on 24 constructed sentences parsed with graphbrain as G2 (`hard_neg_only.json`): current script 5/10 negation
cases, negfix 9/10 (the 10th an expectation error, meaning kept), vg2 14/14 "only" cases [LOG]. Negation fix on shards
20 and 30 (`--limit 400`, 16,727 units) against the current script: parents unchanged, standalone negation cousins
411 → 26, no other change [LOG]. "only" guard on shard 0: 21 removals from verb groups, all restrictions on numbers or
nouns [LOG]. `adverb_check.py` on shards 20 and 30: 6,257 single-adverb hyperedges categorised (attach 3,000, connective
1,215, opener 755, stance opener 45, left 1,242) [LOG]. "as well (as)" contexts on G2 shards 20, 30, 40: 343 [LOG].
Third round (2026-10-08): `g3_q4full.py` = `g3_q4.py` plus the question-4 rules switched on by `Q4=1` (connectives incl.
although/thereby/though/since/whereas and "even though/if"; opener drop list; "as well (as)" with the degree guard; "not
only/just/merely/simply/solely" fused into one non-negating atom `not_only`; plain adverbs (type M) as verb-group members;
position rules applied only when every occurrence of an identical atom qualifies). Self-test 54/54; shards 0 and 20
(`--limit 400`, 16,526 units, 0 problems): single-modifier hyperedges 7,556 → 1,375, distinct verb groups about +40%,
`not_only` 139, 12 ambiguous positions [LOG]; samples read: 15 stance/time openers (14 correct; error: "Historically Black
Colleges"), 25 attached adverbs (22 correct; errors: subordinators since/though, then added to the connectives), 8 not_only
(8 correct) [LOG]. Hard cases `hard_q4.json` (25): 23 pass, A10 an over-specified expectation, A12 a parse with no
predicate in every version; a duplicate-atom leak of "only" (A25) found and fixed (all occurrences must qualify) [LOG].
Fourth round (2026-10-08): "but also" fusion added to `g3_q4full.py` (`also` → `but_also` when "but" stands at most three
tokens before it with only auxiliaries, modals, be/have/do forms or pronouns in between and no punctuation; when the parser
attaches "also" to the conjunction, `(also/M but/J) A B`, it is moved into the predicate of the conjunct after "but").
Self-test 54/54; hard cases `hard_but.json` 7/8 (B06: no predicate path in the parse of an inverted "Not only did…");
shards 0 and 20, 0 problems: not_only/but_also 238 (verb group 130, argument 101, alone 7), 32 sampled, about 29
attributed correctly [LOG]. Parents whose verb group holds 2+ verbs (upward climbing now also passes adverbs): shard 0
4.9% → 5.6% [LOG]. `hard_q4.json` unchanged at 22/25 (the three known non-defects) [LOG].

**RL-084 · 2026-10-08 · G3 · WORD LIST (new) · LIVE**
New G3 word list `PG/g3_word_lists/adverb_roles.tsv` (sha256 `486043050b18`, 499 words; copy of the tracked
`fullscale_pipeline/g3_word_lists/adverb_roles.tsv`, built by `build_adverb_roles.py`, sha256 `7aced43f9054`, from the
single adverbs the tested question-4 rule attaches in G2 v3 shards 0 and 20, RL-083). Columns: rank, word, attachments,
role, action, flag, example. Roles by the assistant from the owner's decisions of 2026-10-08 (G3_POSTPROCESSING.md §6
item 4, to be written): keep (attach to the verb group) degree, frequency, manner, focus, likelihood hedges; drop time,
stance/attitude, certainty boosters, other hedges, in-clause discourse words, subordinators; leave (no attachment, no drop)
nouns the parser typed as modifiers; particles deferred to §6 item 5. Share of attachments: keep 73%, drop 25%, leave 1%,
particles 0.5%; 24 rows flagged for the owner's review. Not yet read by any G3 script. Host incline. [LOG]

**RL-085 · 2026-10-08 · G3 · WORD LIST (revision) · LIVE**
`PG/g3_word_lists/adverb_roles.tsv` revised after the owner's review (commit 03d29d8, comments now in the `owner_note`
column): sha256 `f109f358b15b`, 513 words, built by `build_adverb_roles.py` (sha256 `f8f291e2a29d`). Owner's changes: drop often,
consistently, typically, even, mainly, closely, automatically; keep in-clause first and instead; similarly kept only before
"to"; above dropped at the end of a clause, else kept. Applied by analogy (assistant, flagged in the table): the other
high-frequency generalisers and scalar focusers dropped (frequently, commonly, always, usually, continually, continuously,
repeatedly, regularly, constantly, normally; specifically, primarily, particularly, especially, largely, mostly,
predominantly); rather kept like instead; below conditional like above; unexpectedly as stance; 14 negative or
low-frequency forms added and kept (counts from the text of G2 v3 shards 0-9). Share of attachments: keep 58%, drop 40%.
Host incline. [LOG]

**RL-086 · 2026-10-08 · G3 · WORD LIST (revision) · LIVE**
`PG/g3_word_lists/adverb_roles.tsv` revised after the owner's second review round: sha256 `f66d153f5efa`, 513 words, built by
`build_adverb_roles.py` (sha256 `4b6f4ae541e0`). Owner's decisions: keep yet, drop still, drop generally, drop actually, keep
periodically (unchanged, already kept), simply left to the assistant's judgement (not sure), otherwise left to the
assistant's judgement (not sure). Assistant's recommendation, applied: keep otherwise (120 of 127 occurrences in G2 v3
shards 0-9 carry a conditional/counterfactual meaning, e.g. "unless specified otherwise"; moved from discourse to a new
contrast role beside instead/rather) and keep simply (8 of 10 sampled occurrences are the exclusive/minimising sense, the
same class as merely/purely/solely, already kept; 1 booster sense accepted as a residual, as elsewhere in this table).
Host incline. [LOG]

**RL-087 · 2026-10-08 · G3 · DOC · LIVE**
G3_POSTPROCESSING.md §6 item 4 (stop list) written up as decided, consolidating RL-083 to RL-086 and the owner's two
review rounds on `g3_word_lists/adverb_roles.tsv`: "our" kept off the list; numbers unchanged; "only" a guarded
verb-group member; the negation fix; "not only"/"but also" fused into non-negating atoms (flagged for a later
drop decision); core connectives plus the subordinator extension dropped everywhere; the opener rule for
overall/finally/notably/similarly and a matching discourse/sequence opener list; the single-adverb role table;
"as well as"/"as well" with the degree-word guard. §1 schema row and §3 g2 row updated; §8 and the status line
mark items 1-4 decided. No code changed; `g3_curation_test.py` is unchanged on disk. Host incline. [LOG]

**RL-088 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Evidence for G3_POSTPROCESSING.md §6 item 5 (leftover phrasal particles). Host incline, `tensor_env`,
`PYTHONPATH=$HOME/np1_for_spacy`: `python g3_curation_test.py --shard 0 --limit 700 --outdir
fullscale_pipeline/diagnostics/g3_stoplist/q5 --show 0` (script sha256 `2a5e01342bf4`, unchanged): 694 articles,
14,225 units, 0 problems. Scratch pass over `g3_test_000.jsonl` [LOG]: standalone (unfused) `PHRASAL_PARTICLES`
atoms sitting alone in a `cousin_he`/`sibling_he`/`focal_he` — on 60, up 36, out 36, over 13, in 6, off 5, down 5
(161 total); distinct fused phrasal-verb lemmas (`stand_out`, `break_down`, `set_up`, `point_out`, `follow_up`,
`carry_out`, `feed_in`, `bake_in`, `go_on`, `speed_up`, `scale_up`, `pick_up`, `skip_over`, `come_out`, etc.) mostly
correct. 5 of ~35 distinct fused `_on`/`_in` lemmas are wrong attachments (`employ_on`, `embed_on`, `improve_in`,
`present_in`, `provide_on`), traced to the particle being far from its real verb in the text (1 case, a token-
adjacency guard would catch it) or to an upstream G2 parse-attachment error / garbled source sentence (4 cases, not
fixable by a G3-side guard). Output local only (`fullscale_pipeline/diagnostics/g3_stoplist/q5/`); nothing written
to the RDS data folders.
*[Corrected by RL-089: the five wrong fusions are all particles typed `Mt` or `C` and are removed by a type guard; on
four shards adjacency would catch 2 wrong fusions and lose 9 correct separable ones, so it was not adopted.]*

**RL-089 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Q5 follow-up after the owner's request to check the flip for false positives/negatives and to assess a token-adjacency
guard. Host incline, `tensor_env`. Copies in `fullscale_pipeline/diagnostics/g3_stoplist/q5/` (RDS script unchanged):
`g3_q5flip.py` (sha256 `4c543643ef94`; particles no longer exempt from the stop list; a particle survives only inside a fused verb) and
`g3_q5fix.py` (sha256 `e02344cdb57e`; flip + type guard: only particles typed `Ml` or plain `M` fuse, not `Mt`/`C` + chain fusion: an `Ml`
particle wrapping a predicate under a chain of one-argument modifiers, e.g. `(out/Ml (should/Mm (be/Mv carried/P)))`,
fuses with it). Both: self-test 54/54; shards 0, 20, 30, 40, `--limit 700` each (2,780 articles, 56,126 units), 0
problems [LOG]. Particle types from the G2 v3 shard files (`q5_types.py`) [LOG]. Results (orig -> fix): `Ml` 216 fused /
200 left over -> 400 fused / 0 left (16 unfused, parse errors, now dropped); `Mt` 16 fused (16/16 wrong: "based on",
"depending on" glued to a participle) / 1,096 left -> all dropped; plain `M` 9 fused (mostly right, kept) / 315 left
(quantity phrases: over N, up to N, N out of M) -> dropped. Fusions: 269 kept, 185 gained (40/40 sampled correct), 17
lost (17/17 wrong). Side effects: 13 cousins merged by de-duplication; 8 units where a fused verb-only argument now joins
the verb group below (existing join rule; about 5 wrong). Hard cases (`hard_q5.json`, 24 constructed sentences parsed as
in G2): fix recovers make_up, look_up, switch_off, turn_on, speed_up; loses feed_in where "in" is typed `C`. Token
distance verb->particle over the fix's 454 fusions: 1 for 394, 2-4 for 11 (9 correct separable: "made this information
up", "breaking DNA sequences down"), -1 for 3, unknown for 46 (single-token compounds such as follow_up). Outputs local
only; nothing written to the RDS data folders.

**RL-090 · 2026-10-08 · G3 · TEST (read-only) + DOC · LIVE**
Owner decisions on G3_POSTPROCESSING.md §6 item 5: fusion-conditional particle exemption, type guard (`Ml` or plain `M`
only) and chain fusion approved; no token-adjacency guard; `feed_in` loss and the 8-unit verb-group residual accepted
(unit ids listed in §6 item 5). Written up in §6 item 5; status line, §8 and §9 updated. Tracked:
`fullscale_pipeline/g3_phrasal_check/` (`g3_q5fix.patch` sha256 `d7c40968b4cb`, diff of
`diagnostics/g3_stoplist/q5/g3_q5fix.py` `e02344cdb57e` against `PG/scripts/g3_curation_test.py` `2a5e01342bf4`;
`hard_q5.json` `69873d864054`; `hard_q5.py` `b1428685f3e2`; `q5_types.py` `5b105b053aed`; `q5_compare.py`
`47dfbc208df2`). Read-only count, host incline, over `PG/g2_v3/shards/g2_parsed_000.jsonl`: atoms typed `C#` with
letters 733 (257 spelled-out numbers, mostly "one"; 476 alphanumeric labels such as "8A", "7e-6"), all dropped by the
`KEEP_TYPES` filter; `M#` with letters 1,524, kept. Recorded as a correction under §6 item 4 and flagged with the
number-comparison question for the owner after items 5–7. No pipeline script changed; nothing written to RDS. [LOG]

**RL-091 · 2026-10-08 · G3 / P2 · TEST (read-only) · LIVE**
Evidence for G3_POSTPROCESSING.md §6 item 6 ("LLM" and other focal-name homonyms). Host incline, `tensor_env`; scripts in
`fullscale_pipeline/diagnostics/g3_homonym/` (local; outputs contain corpus text): `llm_scan.py` (every "LLM(s)" in all
50 `PG/g2_v3/shards/g2_parsed_NNN.jsonl`, per article: forms, definitions, evidence), `llm_classify2.py` (a definition
counts only if its word initials spell L-L-M), `anchor_scan.py` (P2's `focal_terms.Matcher` over every article: anchor
and accepted hits per term), `name_scan.py` (ProGen, BioBridge, PaLM 2, ESM-2 with exact case). Results [LOG]: 34,662
articles; "LLM" in 15,215 (329,474 occurrences). Defined as language model 13,424 articles (306,332 occ.); undefined
but "language model" in the article 1,379 (40/40 sampled language model); undefined with other evidence 230 (30/30);
defined as something else 73 (66 homonyms by hand, 5 misspelt "large language model", 1 uncertain, 1 both senses);
undefined with no evidence 109 (hand-labelled: 66 homonym, 43 language model). Homonyms: 133 articles, 4,863
occurrences (1.5%); 128 of them have no other accepted focal term. Rule R1 (accept "LLM" only if the article has another
accepted focal term besides LLM/LLMs/transformer model(s), and does not define LLM as something else unless it also
defines it as a language model): removes 132 of 133 homonym articles (4,826 occ.), loses 115 language-model articles
(590 occ., 0.18%). Cue rules ("LLMs", "LLM-based", ...) recover about 100 occurrences but readmit 8-11 homonym
articles. Other names: "transformer model(s)" is the only focal term in 6,349 articles (29/30 sampled neural
transformers, mostly non-LLM; 1 electrical); homonyms found for ProGen (9 of 87 articles: company Progen), BioBridge (6
of 8), PaLM 2 (1 of 307: protein "Palm 2" domain), ESM-2 (1 of 454: climate model); the same evidence rule removes
7/9, 6/6, 1/1, 1/1 of them and loses 1, 0, 1, 6 model-sense articles. Nothing written to RDS; no script changed.

**RL-092 · 2026-10-08 · G3 / P2 / R1 · TEST (read-only) · LIVE**
Follow-up to RL-091 after the owner's first decisions on §6 item 6 (remove the 73 articles defining LLM otherwise, the 109
with no definition and no evidence, and the ProGen/BioBridge/PaLM 2/ESM-2 homonym articles; asked about "transformer
model"). Host incline. Read-only. (1) Why the transformer-only articles reached G2: R1's query
(`query_pmc_entrez.py`, sha256 `1d057b072f50`) includes `"Transformer model"[Text Word]`, `"Transformer models"[Text
Word]` and `"LLM"[Title/Abstract]`; P2 accepts "transformer model(s)" as a PLAIN focal term and only excludes it from
article evidence (`NOT_ANCHOR`), so its sentences pass to P7 and G2. (2) `transformer_fulltext.py`: full raw texts
(`LCS/pure_text_corpus/`) of the 6,349 articles whose only accepted focal term is "transformer model(s)": 4,710 have no
LLM-related word anywhere; 1,639 have one somewhere (754 "generative AI"/chatbot, 634 "language model", 355 BERT-family,
175 ChatGPT, 49 LLM; 290 also an electrical-transformer word); 25 sampled: 23 peripheral (generative-AI statements,
references, AI-use disclosures, passing mentions), 2 borderline (an mBERT/XLM-RoBERTa study; one sentence on future
multimodal LLMs). (3) Of the 73 articles defining LLM otherwise, 7 are about language models: 5 misspell the expansion
("Large Languge Models"), 1 uses both senses (PMC12647564), 1 uncertain (PMC10967767). (4) The 74 articles in the
LLM classes with no other accepted focal term are language-model uses; several write "large-language model(s)", which
P2's PLAIN list does not match (a P2 matching gap). (5) PMC12405693 ("LLM-MK2", a cell line) was kept by rule R1
only because a ProGen homonym counted as evidence; it is in the ProGen homonym set. Exclusion set sizes: 6,548 articles
(18.9% of 34,662) with all 73 and all 6,349; 6,541 keeping the 7; 4,902 (14.1%) keeping the 7 and the 1,639. [LOG]

**RL-093 · 2026-10-08 · G3 / P2 · TEST (read-only) · LIVE**
Owner decisions on §6 item 6 (second round): remove 67 articles defining LLM otherwise (the 66 homonyms plus the uncertain
PMC10967767); keep 6 (5 misspelt "large language model", PMC12647564 with both senses) and correct their spelling by
hand; remove the 109 undefined/no-evidence articles and the 17 name homonyms; remove the 6,349 transformer-only articles
unless they refer to LLM-related transformers such as BERT; exclusion by a separate short script, ideally part of
preprocessing. Host incline. `transformer_regions.py`: the 6,349 searched in P0's analysed regions
(`PP/content_regions_v2.csv`, raw text in `LCS/pure_text_corpus/`) for BERT-family, GPT-family, T5, LLaMA/Gemini/Claude/
Mistral/PaLM/Qwen/DeepSeek and "language model"/LLM: 6,094 none; 255 some. All 255 read and labelled by hand
(`transformer_255_labels.json`, one context per article): 76 use or compare such a model (text models: DistilBERT,
ClinicalBERT, XLM-RoBERTa, BERTopic, SBERT; and BERT/GPT-style models of other data: scGPT, scBERT, ProtBERT, HuBERT,
SleepGPT, TimeGPT, BERT4Rec), 122 mention one in passing (examples in an introduction, "similar to BERT", a related-work
list), 31 only vision-language models (CLIP, BLIP, LLaVA), 26 false hits or peripheral (Gemini as a scanner, virus,
camera or serum supplier; a stress-tensor symbol mT5; ASR n-gram language models; AI-use statements; a reviewer
comment). Misspellings in the 5 kept articles: 6 occurrences in 6 G2 units ("Large Languge Models", "Large Langue
Model", "large langaue model's", "Large learning models (LLMs)" x3). Nothing written to RDS; no script changed. [LOG]

**RL-094 · 2026-10-08 · G3 / P2 · CODE (new) + OUTPUT + DOC · LIVE**
Owner decisions on G3_POSTPROCESSING.md §6 item 6 (third round): also exclude the 122 passing-mention and 31
vision-language-only transformer articles; correct the 3 "large learning models" misnomers; implement and document.
Host incline, `tensor_env`. New tracked folder `fullscale_pipeline/g3_scope_exclusions/`: `decisions/` (hand-checked
lists: `llm_defined_otherwise.csv` 6589528cfbfb, 67 exclude / 6 keep; `llm_no_definition_no_evidence.csv` 54539bf05cdb,
109; `name_homonyms.csv` c47f861aad32, 17; `transformer_only.csv` 823a90e9142c, 6,273 exclude / 76 keep), `evidence/`
(the scans of RL-091 to RL-093: `llm_scan.py` 5859451cb840, `llm_classify2.py` 57eb6cb4f935, `anchor_scan.py`
3e10ca288ac9, `name_scan.py` 2a9ef07da2a1, `transformer_fulltext.py` e604967ebcc3, `transformer_regions.py`
c882a11f5a54; their outputs hold corpus text and stay in `diagnostics/g3_homonym/`). `python build_scope_exclusions.py`
(sha256 `3a5de5167371`): checks every listed article is in G2 v3 and every correction matches its unit once; wrote
`scope_exclusions.csv` (`5b71d844d9af`, 6,466 articles: llm_defined_otherwise 67, llm_no_definition_no_evidence 109,
name_homonym 17, transformer_only_not_llm 6,273) and `spelling_corrections.csv` (`d0a70f6c1a20`, 6 units). A first
version corrected only the unit text; `test_corrections.py` (`dd18ca3aca3e`, runs `g3_curation_test.py`
`2a5e01342bf4` unchanged) found the misspelt word left as an atom (`languge/C/en`) in 5 of 6 units, because G3 maps
tokens through `atom2word`; the hook now corrects the same token in `atom2word` too: 6 of 6 units give one
`llm/C/focal` atom and no stray atom. Copied to RDS for CSF: `PG/g3_scope_exclusions/` (`scope_exclusions.csv`,
`spelling_corrections.csv`, `build_scope_exclusions.py`; sha256 identical to the tracked files; new folder, nothing
overwritten). Documented in G3_POSTPROCESSING.md §6 item 6 (status line, §8, §9 updated), with dated notes in
G2_PARSING.md and PIPELINE.md item 24. No pipeline script changed; P2 and G2 not rerun. [LOG]

**RL-095 · 2026-10-08 · P2 word list / G3 · CODE-CHANGE + TEST · LIVE**
Owner: add hyphenated "large-language model(s)" to the word list if needed. Checked first (`hyphen_scan.py`, sha256
`a0320bbc44ed`, over all G2 v3 units of the kept corpus): 87 mentions in 85 units of 76 articles, none matched by
P2's matcher (`large-language-model` 37, `large‐language models` 21, ...); 12 of the articles would gain article
evidence. Edited `PP/focal_terms.py` (sha256 `ae5962450379` -> `892f98bb986e`; backup
`PP/focal_terms.before_hyphen_20261008.py`, `ae5962450379`): `EXTRA_PLAIN` gains `large[-‐‑]language[-‐‑ ]models?`;
new self-test case; `--selftest` 19 of 19. Rerun: 87 of 87 matched; `g3_curation_test.py` `canonical()` maps every
variant to `llm`. `focal_words.txt` not regenerated (the pattern is a regex, not a surface form; G3 uses the matcher).
P2, P6, P7 not rerun: the change acts in G3 only. Among the 6,466 excluded articles only PMC13486626 (class "no
definition, no evidence") contains the form and so now has evidence; left excluded pending the owner. Host incline. [LOG]

**RL-096 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Evidence for G3_POSTPROCESSING.md §6 item 7 (focal terms with no governing verb). Host incline, `tensor_env`. Scratch
passes over the RL-089 outputs of `g3_curation_test.py` (`2a5e01342bf4`, shards 0, 20, 30, 40, 700 articles each), with
the 6,466 articles of `g3_scope_exclusions/scope_exclusions.csv` left out, and over the matching G2 v3 edges [LOG]:
54,620 units, 53,735 with a focal term, 908 (1.7%) with no parent and no cousin. By G2 edge: no predicate anywhere 411
(abbreviation glosses 76, short labels/heading-like 82, captions/footnotes 36, "X: ..." list items 34, reference lines
21, other verbless 162); predicate present but the top edge a conjunction/colon and the focal term in a verbless
conjunct 341; predicate present, focal term in a modifier outside it 156. A fallback (nearest ancestor with a clause
child; that clause's verb), 40 sampled units with a predicate, hand-labelled: verb found 38, correct 26, wrong 10
(3 speaker labels "ChatGPT response:", 2 junk predicates "’"/"–", wrong clause), 2 unclear. Outputs local only
(`diagnostics/g3_stoplist/q5/q7_*.json`); nothing written to RDS; no script changed.

**RL-097 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Owner on §6 item 7: PMC13486626 stays excluded; the 411 verbless units (group A) are dropped (as now: no structure);
test the group-B fallback with its guards. Copy `fullscale_pipeline/diagnostics/g3_q7/g3_q7.py` (final sha256 `b863c03286be`; the
RDS script is unchanged): when no predicate lies above a focal term, the nearest ancestor holding a clause beside the
focal branch gives the verb group (`Q7=1`); guards by environment: `Q7_JUNK` (reject a clause whose predicate atoms
have no letters), `Q7_LABEL` (verbless focal branch before a ":" edge), `Q7_LABEL2` (focal words in a prefix of <= 5
words before the first ":"), `Q7_LABEL3` (narrow: such a prefix with a response word, or a quotation after ":").
Self-test 54/54 each time. Shards 0, 20, 30, 40, `--limit 700`, 0 problems; baseline `Q7=0` with the current
`focal_terms.py` (so the RL-095 word-list change is not counted) [LOG]. Results, excluded articles left out: no guards
+648 parents in 622 units, 356 of 908 empty focal units recovered, no baseline parent lost; colon guards +614; colon and
prefix guards +601 (removed 47 parents: 17 correctly, 30 wrongly, mostly "Model: description" definition lists such as
"GPT-4o (URL): Introduced by OpenAI..."); narrow guard + junk guard +640 parents in 614 units, 348 recovered, removed 8
parents, 8/8 correctly ("ChatGPT response: '...'"); the junk guard fired 4 times and changed no parent. Hand labels: 50
random added parents (colon and prefix guards) 38 correct, 11 wrong (verbs invented by the parser, wrong clause), 1
unclear; the 39 parents the narrow guard keeps back are 30 correct, 9 wrong. Constructed cases (`hard_q7.json`, 24,
parsed as in G2, `hard_q7.py`): original 21, no guards 21, colon guards 21, prefix 22, narrow 22 of 24 (the 2 failures
fail in every version: "Prompt to ChatGPT:" parsed with "prompt" as verb; a "’" junk predicate with no other clause).
Outputs local only; nothing written to RDS. Host incline.

**RL-098 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Owner on §6 item 7: drop label-colon units ("BERT encoder: ...") where the fallback would apply, rather than recover
them; delete text reproducing LLM output, so that it is not conflated with researchers' text about LLMs. Host incline,
`tensor_env`, all 50 G2 v3 shards, kept corpus (689,197 units). `response_scan.py` (sha256 `a182e5950bef`): units whose
<= 6-word prefix before ":" holds a focal term: with a response word 147 (86 articles; also researcher headings such as
"LLM invocation and output handling:"), with a quotation after ":" 153 (86; also researchers' prompts), other labels
3,252 (1,743; definitions, headings, a few transcript turns). `llm_output_units.py` (`0c8b9e6a92ab`): a unit is LLM
output if its prefix holds a model name plus only a response word and filler, or only the model name with a quotation
after ":"; prompt labels (prompt, asked, instructed, query, input, question) are kept; a curly double quotation left
open continues into the next unit only if it is the next unit of the same sentence or the next sentence (G2 holds only
focal sentences, so non-adjacent units were wrongly joined in two earlier versions: 458, then 224 false
continuations). Result: 155 label units + 7 continuations = 162 units in 55 articles; 30 sampled labels all model
output; known misses: outputs introduced by "asked"/"questions" ("When asked why ..., Gemini says: '...'"). On shards 0,
20, 30, 40 with the fallback and the colon/prefix guards (`guard2`, RL-097): +601 fallback parents, 327 empty units
recovered, 572 focal units without parent; 24 LLM-output units dropped (15 baseline parents, 19 cousins). Outputs
local only (`diagnostics/g3_q7/`); nothing written to RDS. [LOG]

**RL-099 · 2026-10-08 · G3 · TEST (read-only) · LIVE**
Re-examined RL-097's 50-case fallback sample against the actual G2 parse tree (not just the sentence text), after the
owner asked for misattribution detail. Host incline. Of 50, 36 correct (72%), 14 wrong (28%, revised down from the
earlier 38/50 eyeball read). Five causes identified, each with a traced example: (1) wrong-conjunct choice — the
fallback returns the first predicate sibling in left-to-right order, not the one nearest the focal term, when an
"and"/"but" ancestor has more than one ("Web_interface access was chosen... but... parents... chose a commercial
LLM" attaches LLM to the wrong, same-lemma "choose"; "featured... and was developed based on GPT-4's..." attaches
GPT-4 to "feature" not "develop"); (2) verb-lemma collision, a special case of (1) where two separate clauses share
one verb root and the dummy_sibling groups them by lemma; (3) a present participle used adjectivally but typed `Mv`
by the parser ("existing embeddings", "as evidenced in") is accepted by `is_pred()` as a real predicate; (4) the
sibling check looks only at the immediate head of each candidate (`has_pred(c[0])`), so a true predicate one level
down under a bare adverb ("llms (often (generate ...))") is missed and the fallback climbs further, often onto one
of the (3)-type false predicates; (5) upstream G2 attachment ambiguity independent of the fallback (a passive agent
or an em-dash clause attached to the wrong head). A sixth, softer pattern: "kitchen-sink" parents whose `focal_he`
absorbs many unrelated words from a loosely-related clause, correlated with wrongness (mean 5.4 words in the 14
wrong vs 2.9 in the 36 correct); a `focal_he` >= 8-word cap would catch 4 of 14 wrong cases with 0 false positives on
this sample, tested but not adopted pending the owner. Separately verified the LLM-output continuation rule (RL-098):
tightened to require true text adjacency (same sentence, i.e. the unit split at a clause boundary, or the next
sentence by raw line), which rejects one real multi-turn dialogue continuation 5 lines later in the same article
(G2 keeps only focal sentences, so "next focal sentence" skips non-focal ones in between and is not the same as
"next sentence") in favour of never joining two unrelated sentences; 7 of the 162 LLM-output units are continuations
under the strict rule. Outputs local only; nothing written to RDS. [LOG]

**RL-100 · 2026-10-08 · G3 · CODE (new) + OUTPUT + DOC · LIVE**
Owner's final decision on §6 item 7: drop the fallback entirely (persistent ~28% error rate on recovered parents,
RL-099) and leave both groups (verbless units; predicate-elsewhere units) without a parent, as today; keep the
LLM-output unit deletion. Host incline, `tensor_env`. No change to `g3_curation_test.py` or any pipeline script for
this item: the fallback was never adopted, so there is nothing to revert. New tracked folder
`fullscale_pipeline/g3_llm_output/`: `build_llm_output_units.py` (sha256 `cbceaf621848`; cleaned, deterministic
version of `diagnostics/g3_q7/llm_output_units.py`, checked it loads `g3_scope_exclusions/scope_exclusions.csv` and
excludes those articles) run over all 50 G2 v3 shards: 162 units, 55 articles (155 label, 7 continuation), matching
the diagnostic run exactly; wrote `llm_output_units.csv` (`b71d5717e250`). Copied to RDS for CSF:
`PG/g3_llm_output/` (same two files, identical sha256). Documented in G3_POSTPROCESSING.md §6 item 7 (all 7 items
now decided); status line, §8 and §9 updated. [LOG]

**RL-101 · 2026-10-09 · G3 · TEST (read-only) · LIVE**
Owner: commit the synonym-merging decision (done, 2e650bc); identify where in the rule sequence to drop the fused
`not_only` / `but_also` atoms (§6 item 4 flag), accuracy first, speed second. Host incline, `tensor_env`. Copy
`diagnostics/g3_notonly/g3_q4drop.py` (sha256 `3f3eb2a404b5`; from `diagnostics/g3_stoplist/g3_q4full.py`, RDS script unchanged),
switch `NOTONLY_DROP`: 0 keep, 1 drop both atoms in `Ctx.curate()` (option C), `paired` drop `but_also` always and
`not_only` only when "but", "also" or "as well" follows it in the unit. Option D (keep, then strip the atoms from the
finished structures and remove emptied hyperedges) computed from the keep output as the reference for pure removal.
Text level (A) not possible without re-running G2; edge level (B) not built: it would rebuild modifier edges for no
speed gain, since `curate()` decisions are cached per atom. Self-test 54/54 in all three settings. Shards 0, 20, 30,
40, `--limit 700` (56,126 units), 0 problems; run times within noise (31-36 s per shard in every setting) [LOG]. C
equals D in all 359 units with a fused atom and changes no other unit; no parent invalidated (`compare_cd.py`). Hard
cases of question 4 (57, `hard_cd.py`): C equals D in 57/57; no "not" left as negation. Pairing (`pairing.py`): 297
fused not-only occurrences, 273 paired in their unit, 24 unpaired, 0 paired only across units of the sentence; in most
unpaired cases dropping reverses or distorts the claim ("risks are not merely theoretical", "not solely relying on
ChatGPT's responses", "not simply due to ...", "an adjunct, not simply a replacement"). `paired` setting: equals keep
minus exactly the intended atoms in all 56,126 units (`but_also` removed in 291 units, `not_only` kept in the 27
unpaired units); hard cases as intended. Pre-existing limitation, unchanged: 3 of 319 units with a not-only
construction are not fused because the unit has another identical "not" atom (all positions must qualify). Outputs
local only; nothing written to RDS. [LOG]

**RL-102 · 2026-10-09 · G3 · TEST (read-only) · LIVE**
Owner questions dropping "not only"/"but also" ("help clinicians but also mislead patients": "but" is contrastive) and
asks whether the cases that read as "and" can be told apart deterministically. Scratch pass over the RL-101 keep
outputs (shards 0, 20, 30, 40) [LOG]: "but (...) also" 313 occurrences, 248 after a "not only/just/merely/simply/
solely", 65 bare. Hand labels: 20 sampled "not only X but also Y" all additive, whatever the valence (both positive,
"improved learning efficiency... but have also expanded applicability"; both negative, "not only misleading, but also
deceptive"); 30 sampled bare "X but also Y" 26 contrastive (a positive and a negative property, "can greatly improve
academic work but also brings up ethical issues", "highest AUC but also cost the longest training time"), 4 additive.
The valence of a conjunct ("reduced costs" good, "reduced accuracy" bad) is not needed to tell the two apart, and could
not be read deterministically anyway. Nothing written to RDS; no script changed.

**RL-103 · 2026-10-09 · G3 · TEST (read-only) · LIVE**
Owner on the "not only"/"but also" question: paired "not only X but also Y" reads as "and" and may be dropped, but a
bare contrastive "X but also Y" (no "not only") does not and must keep `but_also`; checked whether the pairing rule
holds on hard cases where the two halves are far apart in complex clauses, before adopting it. Host incline,
`tensor_env`. Same copy as RL-101/102, `diagnostics/g3_notonly/g3_q4drop.py` (sha256 `980b89fd4682`, fixed during this round,
see below): `NOTONLY_DROP=paired` drops `but_also` always and `not_only` only when paired.
`q4_paired()` found a partner anywhere later in the unit, with no distance limit, which is correct for distance (true
pairs up to 25 words apart, heavily embedded, are all still found) but open to a later, unrelated "but"/"also" in a
different independent clause of the same unit. 10 constructed adversarial cases (`hard_pairing.json`) built to
trigger exactly that: 5 of 5 wrongly paired on the first version. Checked against real data (300 fused `not_only`
occurrences, shards 0/20/30/40): only 1 of 273 "paired" cases is a genuine error, and it is exactly this kind
("does not simply vanish...; Its F1 increment is null... but significantly positive again...", a semicolon-separated,
unrelated "but"); every other sampled case (29 "but" not immediately followed by "also"/an auxiliary, 23 relying on
"but" alone with no "also" in the unit) read by hand is a genuine additive pair, including a 21-word gap with no
"also" at all ("Not only will future versions of ChatGPT supersede GPT-4, but the current GPT-4 sits behind a
paywall."). Fix: `q4_paired` now works on `ctx.text` directly (punctuation has no token position of its own, unlike
words, so the position-based version never saw the ';' it was built to detect) and stops the search for a partner
at the next ';'. Self-test 54/54. Re-run on the real one error: fixed. Only 2 of 56,126 test-shard units change
between the old and fixed version (the one real error, and one real true pair that also happens to have a ';',
which the fix now keeps rather than drops — safe, conservative). 10 constructed cases re-checked: 10/14 pass (7 true
pairs across embedded clauses, the real fixed error, 2 correctly-kept unpaired cases); 3 of 5 adversarial cases still
wrongly paired — a comma-spliced independent clause with a new subject, no ';' and no "also" anywhere ("ChatGPT is
not only widely used..., but hospital administrators, who also track budget constraints, raised separate... concerns
entirely unrelated to ChatGPT."). This residual pattern has 0 confirmed occurrences in the ~300 real cases sampled.
Outputs local only; nothing written to RDS. [LOG]

**RL-104 · 2026-10-09 · G3 · AUDIT + CODE-CHANGE (test copy) + TEST · LIVE**
Owner: audit the RL-103 tests and results (made with Sonnet 5) and, if valid and reliably good, vote on option 2 with
the fix. Host incline, `tensor_env`. Findings on RL-103 (copy kept as `diagnostics/g3_notonly/g3_q4drop_rl103.py`,
sha256 `980b89fd4682`): (1) it did not implement option 2 -- in `paired` mode `but_also` was dropped always, so the bare
contrastive "X but also Y" the owner asked to keep was deleted, and no test checked that half of the rule; (2) the
text-based `q4_paired` located "not" with `\bn't\b`, which never matches inside "doesn't"/"isn't", so contraction
pairs never paired; (3) it located the n-th "not" by counting tokens while the regex cannot see the "not" of
"cannot" (tokenized "can" + "not"), so a "cannot" earlier in the unit shifted the count; (2) and (3) fail in the safe
direction and are rare (whole corpus: 3 contraction pairs, 3 "cannot" cases); (4) "273/273, 100%" overstated: 52 of
273 paired cases were read, 240 were classed safe by a regex. The hard-case harness and the semicolon diagnosis were
otherwise sound. Fixed in `diagnostics/g3_notonly/g3_q4drop.py` (sha256 `02b35e6278f8`): `Ctx.spans` from the script's own
`token_spans()`; `q4_paired` finds a partner after the not-only word, before the next ';'; new `q4_also_paired`
drops `but_also` only when a not-only stands before its "but", after the last ';' (option 2). Self-test 54/54 in all
settings. Hard cases `hard_option2.json` (23: the RL-103 set, contractions, "cannot", bare contrastive, mixed;
`hard_option2.py`): fixed 19/23, RL-103 11/23; failures: the 3 constructed comma-spliced new-subject clauses (also in
RL-103) and one constructed unit with two "but ... also" (identical "also" atoms are decided together). Shards 0, 20,
30, 40 (56,126 units), 0 problems: structure equals keep apart from the two atoms in every unit; `not_only` 271
dropped, 29 kept (same decisions as RL-103 in these shards); `but_also` 236 dropped, 55 kept. Hand-read: 30 of 30
sampled dropped `but_also` are not-only partners; of the 55 kept, about 43 contrastive, about 12 additive (marker kept,
no meaning lost). Corpus has no observed unit with two "but ... also" after a not-only. Outputs local only; nothing
written to RDS. [LOG]

**RL-105 · 2026-10-09 · G3 · CODE-CHANGE (test copy) + DOC · LIVE**
Owner: adopt option 2 and document the not-only/but-also mechanisms and guards. Host incline, `tensor_env`.
`diagnostics/g3_notonly/g3_q4drop.py` (sha256 `c6d3edf8682a`): option 2 is now the default (`NOTONLY_DROP` unset =
`paired`; `0` and `1` remain for tests). Self-test 54/54; shard 0 (`--limit 700`, 14,225 units, 0 problems)
identical to the explicit `paired` run of RL-104. New tracked folder `fullscale_pipeline/g3_notonly/`:
`g3_q4_option2.patch` (`99ec4a8c84fa`, the whole question-4 rule set with option 2 as a diff against
`PG/scripts/g3_curation_test.py` `2a5e01342bf4`; supersedes `diagnostics/g3_stoplist/g3_q4full.py` as the reference
for porting), `hard_option2.json` (`66b11a9a5144`) and `hard_option2.py` (`b000aae90c5b`, re-run from the tracked
folder: 19/23), `compare_cd.py` (`fa7300335dc2`). G3_POSTPROCESSING.md §6 item 4: the not-only/but-also bullet
rewritten as decided (fusion, attribution, identical-atom guard, option-2 drop with the ';' boundary and token-span
alignment, why `curate()`, evidence, history, accepted limits); §8 item 1 and 1a and §9 updated. Nothing written to
RDS; the RDS script is unchanged. [LOG]

<!-- Append new entries below. Format: **RL-nnn · date/time · stage · TYPE · LIVE** then command,
host, job ID, script sha256, inputs, outputs, outcome, deviation reference. -->

**RL-106 · 2026-10-09 · G3 · AUDIT (read-only) · LIVE**
Owner's four points (traceability and identity, junk atoms, stop list vs adverb table, duplicates) checked on the
existing shard-0 output of `diagnostics/g3_notonly/g3_q4drop.py` (`c6d3edf8682a`, `vdef_0/`, 694 articles, 14,225
units). Host incline32. `python3 diagnostics/g3_audit/g3_audit.py vdef_0/g3_test_000.jsonl PG/g2_v3/shards/g2_parsed_000.jsonl`
(sha256 `55ca4ef2aec3`), plus inline follow-up counts (duplicate categories, P6 reflexive replacements over all 50 G2
shards; units listed in `diagnostics/g3_audit/emph_units.json`). Findings: 0 traceability problems; content identity
parent 95% / child 50% / cousin 44% distinct; a token in two children of one parent in 647 parents (3.8%), all one
pattern, a focal term inside an "X-based/-generated/-assisted ..." modifier that is both clause head and focal
argument; tokens in two cousins of one unit 725, of which 2 within one parent's cousin set (the rest are different
parents' views); emphatic reflexives replaced by their antecedent in P6 ("the model the model") in 1,675 P6 sentences,
887 G2 units (17 in shard 0: focal mention counted twice in 5, word in two children in 2); junk atom occurrences:
letters+digits 2,103 (mostly legitimate: f1, model sizes 8b/13b/70b, o1, r1), inner symbols 1,157 (escaped ".", "(",
")", "&", "@", U+2010 hyphen, glued citation digits), layout words 1,035 (table, fig, figure, supplementary,
appendix), single letters 534, REF citation tokens 341, URL tokens 29; 1,112 structures made only of such atoms;
38 units repeated within an article (abbreviation glossaries, table notes). Stop list vs adverb table: one shared
word ("only", handled by its own rule); 12 table rows still say "open question 5". Nothing written to RDS or the
scripts. [LOG]

**RL-107 · 2026-10-09 · G3 · SCRATCH TEST (copy) · LIVE**
Owner's question on "X-based" focal phrases (4a after RL-106). Host incline32, `tensor_env`. Copy
`diagnostics/g3_audit/g3_xbased.py` (sha256 `0f0c5bcc0a22`) = `g3_notonly/g3_q4drop.py` plus `XBASED` (default 1):
an edge counts as a clause only if graphbrain types it as a relation (`R`); a noun phrase holding a participle
("LLM-based chatbots", typed `Mv`) is not a clause and does not split cousins; edges that cannot be typed after focal
replacement (`(llm/Cp/focal based/Mv)`) count as non-clauses (first run without this guard: 725 unit errors, discarded).
Self-test 54/54. `Q4=1 ONLY_MODE=vg2 XBASED=0|1 python g3_xbased.py --shard 0 --limit 700 --outdir xb0_0|xb_0`; the
`XBASED=0` control is byte-identical to `g3_notonly/vdef_0`. Shard 0, 14,225 units, 0 errors, 0 problems: parents
16,985 -> 16,888; parents with a token in two children 647 -> 8; cousins 53,609 -> 47,061; units changed 2,198 (616
with an X-based word); 44 units lose all parents, none with a predicate at the top of the parse (headings, fragments).
Samples: the X-based phrase now sits in the focal argument of the clause's real verb ("LLM-based assessment tools ...
could promote ..." -> `(dummy_sibling could promote) (focal_he assessment base llm tool) ...`). [LOG]

**RL-108 · 2026-10-09 · G3 · AUDIT (read-only) · LIVE**
Owner's questions 4a (headings), 4b (reflexive pronouns), 4c (several parents per clause). Host incline32,
`tensor_env`. `python diagnostics/g3_audit/refl_dep.py` (sha256 `68e708eb7bf2`): every reflexive replaced by P6 (3,157, from
`LCS/coref_v2/coref_resolved_v2.jsonl`) parsed in its original text with spaCy `en_core_web_trf` (49 s); output
`refl_dep.json`. Dependency labels: appos 1,914, dobj 597, pobj 307 (by 111, in 48, for 41, among 25, of 21), npadvmod
248, nsubj 46, attr 22, dative 9, conj 6, other 4. Hand-read samples: appos 8/8, npadvmod 8/8, attr 8/8 and "in itself"
10/10 emphatic; dobj 8/8, dative 8/8 object; nsubj 7/8 object; "by itself" 8/10 adverbial ("alone"), 2/10 passive
agent. Inline counts on `diagnostics/g3_audit/xb_0` (shard 0, units with a parent 13,676): heading-like or non-prose
candidates 73 (21 numbered and <= 15 words, 12 without end punctuation and <= 15 words, 40 longer without end
punctuation; they include reference entries, pseudocode lines and figure-panel captions as well as real numbered
sentences); units with 2+ parents 2,674; clauses whose verb tokens give 2+ parents 836. [LOG]

**RL-109 · 2026-10-09 · G3 · AUDIT (read-only) · LIVE**
Inline counts for the owner's 4c and 5.2 questions. Host incline32, `tensor_env`. Shard 0, units with a parent in
`diagnostics/g3_audit/xb_0`: clause levels down to the focal term (graphbrain relations with a predicate connector,
approximate focal list of 12 names; 14,726 occurrences): 1 level 59.7%, 2 levels 30.2%, 3 levels 7.7%, 4 or more 1.8%,
none 0.6%. Participle properties typed as verbs inside arguments (source type `Mv`, output `/P`): focal_he 1,659,
sibling_he 1,542, cousin_he 1,529; most frequent base 527, generate 305, exist 137, propose 128, increase 69;
19 structures consist only of such words. Searched the docs, scripts and this project's transcripts for a decision
to write numbers as words ("ninety percent"): none recorded. [LOG]

**RL-110 · 2026-10-09 · G3 · SCRATCH TEST (copy) + DOC · LIVE**
Owner decisions after RL-106 to RL-109: hyphen-only participle rule, cousin grouping, no parent merging, no kinship
types, content key, numbers option (c), junk and non-prose rules, emphatic reflexives restored. Host incline32,
`tensor_env`. Test copy `diagnostics/g3_audit/g3_rules2.py` (sha256 `53f3162ddb83`; from `g3_xbased.py`):
`XBASED=hyphen` (a participle hyphen-joined to the token before it, or a G2-joined hyphen word containing "_", is a
noun property: its phrase is not a clause, the word is kept as written and typed M, and it is never a predicate;
the hyphen's own token position is skipped and a hyphen glued to the previous token, "LLM- based", is found),
`GROUP=1` (a phrase cut open to reach a nested clause keeps its other words as one cousin). Self-test 54/54. Shard 0
(`Q4=1 ONLY_MODE=vg2 ... --limit 700`), 0 errors, 0 problems; `XBASED=0 GROUP=0` byte-identical to
`g3_notonly/vdef_0`. Base → hyphen → group → both: parents 16,985 → 16,939 → 16,985 → 16,939; parents with a token
in two children 647 → 27 → 647 → 27; tokens in 2+ cousins of a unit 725 → 595 → 212 → 191; one-word cousins 12,398
→ 11,474 → 10,916 → 10,183; units losing all parents 0 → 29 → 0 → 29 (focal phrase outside any clause in the parse).
Rejected variant `GROUP=2` (loose single words only): 725 → 697, 13/15 hard cases. Hard cases
`g3_audit_rules/hard_rules2.json` (`51f577bd1bfe`), runner `hard_rules2.py` (`d77b9859b754`): before 4/15, final
15/15. Intermediate runs during development (no hyphen found because the hyphen has its own token position; a
first patch that failed and left the code unchanged) were discarded. G3_POSTPROCESSING.md §6 item 8 written; status
line, §8 items 1 and 1a, §9 updated. Nothing written to RDS. [LOG]

**RL-111 · 2026-10-09 · G3 · AUDIT (read-only) · LIVE**
Owner's follow-up questions on §6 item 8. Host incline32. (1) Cousin grouping on real units: shard 0, `r_base` vs
`r_gr1` (RL-110), 1,570 units changed (538 with 2+ parents, 1,032 with one); 30 sampled (15 + 15, seed 21, uids in
`diagnostics/g3_audit/grp_sample.json`) read by hand: 26 correct (the pieces of one phrase joined, or the rendering
another parent already had adopted), 4 join items that are separate (3 coordinated list items, 1 run-in label
"Fragility of key conclusions:" with the clause after it). Over all changed units: 643 only adopt or share an
existing cousin, 927 create a larger cousin. (2) The 29 units that lose every parent under the participle rule,
read by hand: 9 headings, titles or labels; 20 real sentences in which the parser attached the focal phrase outside
every clause (e.g. the subject phrase parsed as the head of the whole sentence, or the focal phrase under a top-level
"on"/"than"/"in" connector); before the rule the participle acted as a fake verb for them. The RL-110 log line saying
all 29 were headings or captions was wrong. (3) Numbers: `python3 diagnostics/g3_audit/numbers_scan.py
PG/g2_v3/shards/g2_parsed_000.jsonl r_new/g3_test_000.jsonl` (sha256 `5a56dba58c10`): 10,401 numbers in 13,688
kept units (4,537 units with a number): part of a focal name 2,864; percentage 1,564; integer 0-10 1,390; integer
> 10 1,362; decimal 1,123; label (Table 2, Section 2.3) 685; year 375; range 326; inside square brackets 244;
statistic notation 196; bracketed list label 195; enumerator at unit start 77. [LOG]

**RL-112 · 2026-10-09 · G3 · CODE (new test script, stages 1-3) · LIVE**
Owner: start building the G3 test script with all decided rules (§8 item 1a). Host incline32, `tensor_env`. New tracked
folder `fullscale_pipeline/g3_v2/`: `g3_curation_v2.py` (from `diagnostics/g3_audit/g3_rules2.py`, i.e. the RDS test
script + item 4 + item 8 participle and cousin rules), stage by stage:
(1) item 5 merged by hand from `g3_phrasal_check/g3_q5fix.patch` (particles off the stop-list exemption, type guard,
chain fusion); decided defaults fixed (`Q4=1`, `ONLY_MODE=vg2`, `GROUP=1`, `XBASED=hyphen`; switches kept for comparison
runs only); the adverb role table implemented for the first time in a G3 script (it had been built and reviewed but no
G3 copy applied it; the item-4 copy attached every plain adverb): drop / keep / leave (mistyped nouns: kept, never
attached) / "similarly" only before "to" / "above", "below" dropped at a clause end; the 12 rows marked "open question
5" (forward(s), behind, ahead, throughout, back, under, before, around, despite, beside, away) are not covered by item
5's decision and are kept, provisional, for the owner. (2) items 6 and 7: input lists read from RDS with sha256 checks
(`scope_exclusions.csv` 5b71d844d9af, `spelling_corrections.csv` d0a70f6c1a20, `llm_output_units.csv` b71d5717e250,
`adverb_roles.tsv` f66d153f5efa), excluded articles and LLM-output units skipped before focal matching (each list's
unit hash checked), corrections applied to text and atom2word, the unit hash verified on the original text, article
evidence (`anchored`) taken from the kept, corrected units; test input of 18 affected articles: 5 excluded, 23
LLM-output units skipped, 6 corrections, each now matching `llm`, 0 problems. (3) item 1: measured first, as the
decision requires (`g3_v2/measure_be.py`, sha256 `8cbfe1c89631`, shard 0, auxiliary-typed "be" by the first content
word after it): passive main verb 3,594 (86.5%), other main verb 450 (10.8%), non-verb 110 (2.7%; mostly degree words
or coordination before a verb, about 15 real mistyped copulas such as "are capable"). Rule: an auxiliary "be" whose
next content word (adverbs, negation, auxiliaries, modals, "to", degree words, coordinators skipped) is a main verb
with the passive-subject role is kept as `be/M/en` (typed M, so it is not the copula `be/P/en`); every occurrence of
the same atom must qualify. A bug found by the self-test and fixed: a type without a role field ('Pd.<pf--') had its
feature field read as roles. Self-test 54/54 (one case updated: "should be approached" now keeps be, as decided).
Hard cases, runner `g3_v2/run_hard.py` (sha256 `80b9ca527b8f`, parses cached in `g3_v2/hard/parsed_cache.json`):
v2 copies of the item-4 sets with expectations updated where they predate later decisions (role table, option 2,
item 1), each marked with a note and its old expectation kept: item 4 adverbs 22/25, but-also 7/8, negation/only
22/24, option 2 19/23, item 8 15/15, new passive set `hard_passive.json` 9/10 (V08 a parser limitation); every
remaining failure is a known one, also failing in the item-4 reference copy. Shard 0 (`--limit 700`): 694 articles,
132 excluded, 13,845 units, 0 errors, 0 problems, 13,332 units with a parent, 16,557 parents, 47,731 cousins. Script
sha256 `eb990f331bb9`. Outputs local only. [LOG]

**RL-113 · 2026-10-09 · G3 · CODE (test script v2, stage 4: §6 item 3) · LIVE**
Host incline32, `tensor_env`. New module `g3_v2/modal_merge.py` (sha256 `f398b16b0042`): the tested decision rule copied from
`g3_modal_check/modal_rules.py` (3f993e0ff164; decisions now return atoms) plus the edge rewrite that had not existed:
all decisions are taken on the parse as G2 gave it and applied in sentence order, before focal replacement (graphbrain
cannot type sub-edges such as `(llm/Cp/focal generated/Mv)`: a first version applied after replacement raised 8 unit
errors on shard 0); the trigger's clause and its complement clause(s) become one clause (connector = trigger +
complement verbs, arguments = both clauses' arguments; participle guard: an -ing trigger keeps no subject-role
argument); "be able / unable": the target is the copula clause holding able, the copula leaves the connector, modals
and negation stay, `able/M/en` is a verb-group member, and a complement the parser attached to another clause is moved
into it (anchor verb after "able to"); a decision whose complement verb has no clause of its own in the parse is not
applied (counted); any exception inside the merge leaves the unit unmerged (counted, never lost). Bugs found and fixed
on the way: be-able first merged into a neighbouring clause (the smallest clause holding both), and dropped the modal
in "may be able to". `g3_curation_v2.py` (sha256 `c304a9dfcb5d`): `be/M` (item 1) and `able/M` counted as verb-group atoms
(this restores the pull-up of verb-only levels such as "which is detailed as follows" that applied before item 1 kept
"be"). Tests: decision hard cases `g3_modal_check/hard_cases.json` through the copy 33/40, identical to the original;
self-test 54/54 (one case updated: "LINS allows users to integrate LLMs" now gives `(dummy_sibling allow integrate)`,
as decided); new G3-level set `g3_v2/hard/hard_modal_g3.json` (sha256 `7abb48991a41`) 13/14 (M09: hortative "let us" is
blocked by the guard but pulled into the verb group by the pre-existing rule for verb-only levels); all other sets
unchanged (107/119 in total, every failure a known one). Shard 0: 621 merges (allow group 293, able 66, prompt 46, fail
46, ask 41, need 38, tend 33, instruct 26, have to 12, help in 5, force 3, gerund 3, empower 2), 38 not applied, 11 be-able
complements moved, 0 errors, 0 problems; parents 16,557 → 16,555; 358 units with changed parents. 20 changed units
read by hand: 14 correct merges, 2 be-able errors (both fixed above and re-checked), 4 changes from the restored
pull-up, not from the merge. [LOG]

**RL-114 · 2026-10-09 · G3 · CODE (test script v2, stage 5: §6 item 8 atom, number and unit rules; content key) · LIVE**
Host incline32, `tensor_env`. `g3_v2/g3_curation_v2.py` (sha256 `fad8e5e394fb`): layout words dropped when they name a non-text
object by a label, with the label ("Table 2", "Fig. 1b", "Supplementary Table S1"; the period of "Fig." can be its own
token); numbers: non-text ones dropped (labels incl. Section/Question/Step/Phase/Version N, inside square brackets,
"(1)", an enumerator opening the unit, statistical notation after p, CI, ±, OR/HR/r/n/t =, <, >, and the second number
of such an interval), all others kept as written (years included), integers 0-10 as words; spelled-out numbers typed
C# kept; comparison words next to a kept number kept (over, under, above, below, up, more, less, fewer, least, most,
nearly, approximately, about, around, almost, roughly); URL tokens and model sizes (8b, 70b) dropped; atoms cleaned of
glued citation digits and edge punctuation, Unicode hyphens inside atoms written "_", single letters dropped (checked
only where an atom would otherwise be kept, so other drop reasons keep their counts); a version written as the next
token joins a versioned focal name ("DeepSeek-V3" → deepseek_v3, "ChatGPT 4o" → gpt_4o); non-prose units skipped
(headings: no end punctuation and Title Case, or a section number such as "9.5"; reference entries; pseudocode; panel
captions), written to `g3_nonprose_NNN.jsonl` for review; glossary parents dropped (verb equal/indicate/denote/stand/
refer, a focal argument and at least one more argument made only of focal terms; first version without the
two-sided condition also dropped "the websites that GPT-4 referred to", 2 of 13 wrong); content key on every child,
cousin (sha1 of the sorted atoms, 16 hex) and parent (of its children's keys), verified by the checks. Self-test 54/54
(one case updated: a plain-text number is now kept). New hard set `g3_v2/hard/hard_item8.json` (sha256 `e57dafa3d091`) 11/11;
all sets 118/130, every failure a known one. Shard 0: 694 articles, 132 excluded, 13,820 units, 0 errors, 0 problems,
13,311 units with a parent, 16,529 parents, 47,104 cousins; drops: label 2,255, non-text number 563, model size 286, URL
80, single letter or symbol 555; non-prose units 25 (14 headings, 6 pseudocode, 3 panel captions, 2 reference entries),
all 25 read and correct; glossary parents 9. [LOG]

**RL-115 · 2026-10-10 · G3 · CODE + DATA (§6 item 8: emphatic reflexives) · LIVE**
Host incline32, `tensor_env`. `g3_v2/reflexive_restore.py` (sha256 `92fa72c383de`): every reflexive pronoun P6 replaced
(3,157, from `LCS/coref_v2/coref_resolved_v2.jsonl`) classified emphatic or object by the dependency label of the
pronoun in P6's original sentence (spaCy `en_core_web_trf`): emphatic 2,324, object (kept) 833. For each emphatic
sentence reaching G2 v3 (1,283 of 2,324 distinct sentences), the replacement is located in the G2 unit text (right to
left per sentence, matched by the longest shared suffix before it, since several units can repeat the same
replacement string) and the original pronoun restored, then the unit is re-parsed alone with graphbrain as G2 parses
its pieces. Run (2,560 s = 43 min; background, not detached from the harness): located 1,231, not located 55,
re-parse gave other than one sentence 19, units restored 1,211. Output `g3_reflexive/reflexive_restored.jsonl`
(sha256 `d121a203daec`, mirrored from the RDS path it was built at). Checked: 30 random units read by hand, all correct
(pronoun present, no duplicated phrase left); 1 unit has two restorations. `g3_curation_v2.py` (sha256 `8a153640fd6b`) reads
the table before the exclusion/correction hooks, on the uid; checks its hash and original text against the unit; on a
mismatch records a problem and leaves the unit as G2 gave it. Self-test 54/54; all hard sets unchanged (the sentences
they test are not in the table). [LOG]

**RL-116 · 2026-10-10 · G3 · CODE (test script v2 complete: §6 items 1-8) · LIVE**
Host incline32, `tensor_env`. The 12-adverb decision (option 3, owner 2026-10-09/10) and the M1 verb-group flag added
to `g3_curation_v2.py` (same sha256 `8a153640fd6b` as RL-115, one script): direction words (forward(s), ahead, behind, back,
away) fuse with their verb like the item-5 particles, including through a modal/auxiliary chain ("must move forward"
-> `move_forward`), else stay a modifier in their phrase; before/throughout dropped; around/under kept only before a
number; despite/beside dropped. New hard set `g3_v2/hard/hard_adverbs12.json` (sha256 `665276a1e3bb`) 10/11 (D04: the parser
attaches "back" to "to the user", not to the verb). Regression found and fixed while adding these: the glossary rule
(RL-114) matched a verb by its first part after "_", so "GPT-4 **stands out** among LLMs" lost all its parents (the
rule read "stand" instead of "stand_out"); fixed to match the whole verb; new set `hard_glossary.json` (sha256
`0ef122e3b622`) 4/4, including this sentence. All hard sets 127/139 total pass (the one new "pass" over RL-114's 118/130 is
the 10th adverbs-12 case; the four glossary cases were not counted there), every failure a known one. Full run on
shard 0 (`--limit 700`, 694 articles) and shard 20 (`--limit 0`, the whole shard, 693 articles): both 0 errors, 0
problems. Shard 0: 132 articles excluded, 13,820 units, 13,311 with a parent, 16,525 parents, 47,099 cousins, 24 units
reflexive-restored, 25 non-prose units removed, 9 glossary parents dropped. Shard 20: 121 excluded, 14,483 units,
14,013 with a parent, 17,130 parents, 49,258 cousins, 2 LLM-output units, 34 reflexive-restored, 15 non-prose, 5
glossary. `g3_curation_v2.py` now implements all of §6 items 1-8; nothing written to RDS or the production script.
[LOG]

**RL-117 · 2026-10-10 · G3 · CODE FIX (test script v2: run-to-run nondeterminism) · LIVE**
Host incline32, `tensor_env`. Found while re-checking shard 0 after adding the reflexive table to the input hash checks
(`INPUT_SHA`, sha256 `d121a203daec`; the table was the only input read without a check): 2 of 13,820 records differed
from the RL-116 run, both in the lexical-modal merge. Cause: the decision rule copied from `g3_modal_check/modal_rules.py`
takes "the first verb" of a connector from graphbrain's `atoms()`, which is a set, so the chosen verb followed Python's
per-process hash seed (one unit merged under seeds 2, 3, 5 and was skipped under 1, 4, 6). The tested original has the
same latent flaw. Fix in `g3_v2/modal_merge.py` (sha256 `75c7022ac187`): atoms taken in edge order (`leaves()`), also
for the be-able connector; membership tests left as they were. Both affected units are coordinated complements the
parser shaped oddly ("allows us to use models ... to infer and visualize"; "enables GPT-4 to better comprehend and
generate language"): the first verb in edge order has no clause of its own, so the merge is now always skipped (the
state before the decision, as the rule says for such parses). Checks: decision hard cases 33/40 (unchanged); self-test
54/54; all hard sets 132/145 (RL-116 gave 127/139, a miscount: 145 cases are judged, and the 13 failures are the known
ones listed in RL-112 to RL-116); determinism: shard 0 (`--limit 700`) run under PYTHONHASHSEED 1 and 777, and shard 20
(`--limit 0`) under 3 and 4242, each pair byte-identical. `g3_curation_v2.py` sha256 `2f7e8f356692`. Final counts:
shard 0 — 694 articles, 132 excluded, 13,820 units, 13,311 with a parent, 16,525 parents, 47,100 cousins, 281,852 atoms
kept, 178,048 dropped, 621 modal merges (38 skipped), 24 reflexive-restored units, 25 non-prose units, 9 glossary
parents, 0.0033 s per unit; shard 20 — 693 articles, 121 excluded, 14,483 units, 14,013 with a parent, 17,130 parents,
49,265 cousins, 731 modal merges (54 skipped), 34 reflexive-restored, 2 LLM-output, 15 non-prose, 5 glossary; both 0
errors, 0 problems. `fullscale_pipeline/g3_v2/runs/` (373 MB, corpus-derived) added to `.git/info/exclude`. [LOG]

**RL-118 · 2026-10-10 · G3 · CHECK (§6 item 8: two promised labelled checks) · LIVE**
Host incline32. Labelled by the assistant from reading each case in context; samples, labels and the drawing code in
`diagnostics/g3_checks/` (local; untracked like the rest of `diagnostics/`).
(1) Emphatic vs object reflexives (item 8 promised "about 100" before use; the table had been built and applied first,
RL-115): 60 drawn from the 1,211 restored units (seed 1010) — 60/60 emphatic, correctly restored; 40 drawn from the
reflexives the rule keeps as objects (dobj, dative, other prepositions, small-clause subjects, conj; "by" phrases left
out because the cached parse cannot tell a passive agent from "by itself") — 40/40 correctly kept, 2 borderline and
harmless ("decides for itself", "without themselves relying"). The split stands.
(2) Integers 0-10 written as words (item 8 promised a precision check before use; it had been applied, RL-114): 100
drawn from the 2,206 converted digits in the RL-117 outputs of shards 0 and 20 (seed 2020): 59 are counts or values in
plain text ("2 reviewers", "4 dimensions", "up to 4 times", "a score of 5"); 41 are not: labels the label rule misses 22
("Textbox 3", "Online Resource 5", "Supplementary Note 3", "Surgeon 2", "Topic 7", "Tier 1", "round 1", "group 0 and 3",
"prompt variant 0", "readers 1 and 2", the later items of "Tables 4, 6, 7, and 8", and labels sharing an atom with a
plain occurrence of the same digit in the unit), model versions outside the focal folding 7 ("Claude Sonnet 4", "DALL-E
3", "ChatGPT 4 and 5", "transformer 3 (GPT-3)"), dates 4 ("October 9, 2023", "2024-04-09"), products or ids 3 ("AMD Ryzen
7", "ROS 2"), citation or list debris 3, a formula 1, a statistic 1. So the conversion turns labels, versions and dates
into counts ("Table 3" and "three models" become the same atom). The non-text number rules leak the same cases also
when digits are kept as written. Not changed yet: for the owner. [LOG]

**RL-119 · 2026-10-10 · G3 · CODE (test script v2: 0-10 as words rolled back) · LIVE**
Owner, after RL-118: writing integers 0-10 as words is not important enough to keep, given that 41 of 100 converted digits
were labels, versions, dates or debris. Host incline32, `tensor_env`. `g3_v2/g3_curation_v2.py` (sha256 `b78ee9b98a40`):
`number_word()` removed; every plain-text number is kept as written; spelled-out numbers stay as written (so "2 models"
and "two models" are different atoms again). Hard-case expectations updated, old ones kept with a note: J05
(`hard_item8.json`) and D09 (`hard_adverbs12.json`). Self-test 54/54; hard cases 132/145 (unchanged; the 13 known
failures). Shard 0 (`--limit 700`) and shard 20 (`--limit 0`): 0 errors, 0 problems; parents and cousins unchanged
(16,525 / 47,100; 17,130 / 49,265); 725 and 757 units changed against RL-117, all of them a digit written as a word
before and as the digit now, except 3 units per shard where a digit and the same number spelled out ("5" and "five")
had been merged into one atom and are now two. Still open from RL-118: the non-text number rules miss labels such as
"Textbox 3", "Surgeon 2", "Topic 7", list items after ", and", model versions outside focal folding, dates, products;
these now leak as digits. [LOG]

**RL-120 · 2026-10-10 · G3 · CODE + CHECK (test script v2: non-text numbers, owner's plain-text rule) · LIVE**
Owner: remove what is not plain text, keep what is. Applied to the RL-118 misses (classification by the assistant,
stated to the owner): references to non-text objects are dropped with their number ("Textbox 3", "Online Resource 5",
"Supplementary Note 3", "Box 1", "Algorithm 1"; a preceding supplementary / supplemental / online / additional / extended
goes too); index numbers on things in the prose are dropped and the word kept ("Topic 7", "Surgeon 2", "group 0", "round
1", "readers 1 and 2", "Challenge 1"; 34 index nouns and their plurals); citation and list debris, numbers inside
formulas ("|", "~", "^", "*", "="), snapshot-date ids, numbers after SD / SE / IQR / U / Z, intervals in parentheses right
after a value ("4.5 (4.0–5.0)"), and citation years ("et al. (2020)") are dropped; numbers inside names ("Claude Sonnet
4", "DALL-E 3") and dates in prose ("October 9, 2023") are plain text and kept. Guards: the label must stand directly
before its number (space or period only); a number followed by "%" or a unit is a value; after an index noun only a
whole number is an index; label lists continue over connective tokens (", and", "or"). Host incline32, `tensor_env`.
`g3_v2/g3_curation_v2.py` sha256 `389123552561`. New hard set `g3_v2/hard/hard_numbers_rl120.json` (sha256 `402544dba99d`) 22/22; all sets
154/167 (the 13 known failures); self-test 54/54. Checks (labels in `diagnostics/g3_checks/rl120_labels.json`):
round 1 — 40 newly removed tokens, 31 correct, 9 real counts lost ("topics, 13 (9.5%)", "27 patients (90%)", "model,
90%"), all from the missing directly-before and % guards, which were then added; the 100-number kept sample of round 1
was invalid (it drew digits from focal atoms' sources: 12 of its 15 "version leaks" were inside `gpt_3_5` etc.).
Round 2 (fresh samples, seed 1201): removed 37/40 correct (wrong: "Grade Level 17.4", and two "+" read as a formula);
both causes fixed, after which those 3 are kept and the other 37 still removed; kept 95/100 plain text (leaks: a
command-line argument list, "μ Human 11.79", "Martinson 2023)", a "24—" enumerator, citation digits after names
"LLaMA 22"). Shards 0 and 20: 0 errors, 0 problems; parents 16,523 / 17,130, cousins 47,077 / 49,224. Also found, not
changed (focal matching, §4): the matcher reads a score after a name as a version ("ChatGPT 8.0 [7.0–10.0]" →
`gpt_8`). [LOG]

**RL-121 · 2026-10-10 · G3 · AUDIT + CODE (input checks, adverb table) + MEASUREMENT + DOC · LIVE**
Owner: harmonise G3 code, decisions and documentation; run any test a decision needs first. Host incline32, `tensor_env`.
Audit: every sha256 cited in G3_POSTPROCESSING.md §7.1/§9 checked against the local and RDS files (38 files, all match);
the four RDS input mirrors match git. Found and fixed: (1) `g3_word_lists/adverb_roles.tsv` still gave the 12 rows decided
in §6 item 8 (option 3) the action "open question 5" and the test script overrode them in code; `build_adverb_roles.py`
(sha256 `f05285f9bb65`) now writes the decided actions (direction words: fuse with the verb; before, throughout, despite,
beside: drop; around, under: kept before a number, else drop), the table (sha256 `79466ff78dfe`, exactly the 12 rows
changed) is copied to `PG/g3_word_lists/`, and `g3_v2/g3_curation_v2.py` (sha256 `efbf44b09683`) reads them from the table
with a start-up check instead of the override; (2) the script imported P2's `focal_terms.py` and read the stop list
without a hash check; both are now in `INPUT_SHA` (892f98bb986e, 2b6c7d9fdae9). Verification: self-test 54/54, hard
cases 154/167 (the 13 known failures), shard 0 (`--limit 700`) and shard 20 (`--limit 0`, PYTHONHASHSEED 99) records and
non-prose files byte-identical to RL-120. Measurements for two items left without an owner answer: (a) units that lose
every parent under the hyphen-participle rule, whole v2 script against the same with `XBASED=0`: shard 0 27 of 13,820,
shard 20 14 of 14,483, none gains (RL-111 counted 29 on shard 0 with the earlier copy); about half headings, captions or
bullet fragments; (b) a score after a model name read as a version (RL-120): in the focal-form tables of shards 0 and 20,
3 forms in 2 units ("ChatGPT 8.0/9.0 [..]" in a score table, PMC12821988; "GPT 0.78 to 0.65", PMC13257059); other
name + number forms are real versions (ChatGPT 5.2–5.5, Grok 3/4, Gemma 2/3, Qwen 2.5). The glossary count of shard 0
rose from 9 (RL-117) to 11 at RL-120 because the number rules now remove the stray "4" in "Language Model 4; LLM equals
large language model." (one unit, 2 parents). Documents: dated notes in G3_POSTPROCESSING.md (status, §1, §6 items 1, 3,
4, 5, 8, §7.1 commands, inputs, results and earlier runners, §8 item 4 with the two questions, §9), PIPELINE.md (G2 and
G3 rows, flow diagram, focal_words readers, items 21, 24, 29, command and file tables) and G2_PARSING.md (copula count).
[LOG]

**RL-122 · 2026-10-10 · G3 · CODE (test script v2: units and articles without a parent; scores after model names) · LIVE**
Owner, on the two questions of RL-121: (1) a unit that ends up without a parent is removed; an article none of whose units
has a parent is marked invalid; (2) "can we just remove all values in such brackets [7.0–10.0]?" — those values were
already dropped (every number inside square brackets, RL-114); the false version came from the number before the bracket,
so the item-8 version step of `focal_mentions` now refuses a number followed by a bracketed interval or below 1. A first
guard that also refused a number followed by "to <number>" removed 9 real versions in shard 0 ("from 91.4% for Gemini 3 to
98.1%", "DeepSeek-V3.2 to 89.0%", "from ChatGPT 3.5 to 4o"), so it was narrowed. Host incline32, `tensor_env`.
`g3_v2/g3_curation_v2.py` sha256 `266ddcbb00d2`: units without a parent go to `g3_noparent_NNN.jsonl` instead of
`g3_test_NNN.jsonl`; new `g3_articles_NNN.jsonl` with one status per article (excluded / valid / invalid_no_parent); report
counts `units_removed_no_parent`, `articles_valid`, `articles_invalid_no_parent`; all output files refuse to overwrite.
New hard set `g3_v2/hard/hard_versions_rl122.json` (sha256 `302cf7da0fe4`) 8/8; all sets 162/175 (the 13 known failures);
self-test 54/54. Shard 0 (`--limit 700`) and shard 20 (`--limit 0`, PYTHONHASHSEED 5): 0 errors, 0 problems; the written
units are exactly those that had a parent in RL-121 (13,310 and 14,013), removed 510 and 470 (290 and 252 with a focal
mention); records otherwise identical to RL-121 except 2 units of shard 0 (PMC12821988, "ChatGPT 8.0/9.0 [..]" →
`chatgpt`; PMC13257059, "GPT 0.78 to 0.65" → `gpt`, also Gemini and Claude); articles: shard 0 560 valid, 2 invalid, 132
excluded; shard 20 569 valid, 3 invalid, 121 excluded. The 5 invalid articles, read: an abbreviation list, an AI-use
declaration ("ChatGPT (Version 4.0) [Large language model]."), a "Method: ..." label, and two articles whose only focal
sentence the parser left without a governing clause. [LOG]

**RL-123 · 2026-10-10 · G3 · CODE (CSF test) + REFERENCE RUN · LIVE**
Owner: code, memory and git aligned on G3; make the CSF test. Production will run the test script v2 itself (no fork):
`g3_curation_v2.py` (266ddcbb00d2), `modal_merge.py` (75c7022ac187), `run_hard.py` (80b9ca527b8f) and `hard/` (all sets
and the parse cache, identical to git) copied to `PG/scripts/g3_v2/`. Reference run on incline37 (Xeon Gold 6326) from
the RDS copy, whole shards 0 and 20 (`--limit 0`) into `PG/g3_v1_csftest/reference_incline/`: 0 problems, 44-45 s per
shard, peak memory 57 MB; records identical to RL-122. New `g3_v2/check_csf_g3.sh` (tracked; copy at
`PG/scripts/g3_v2/check_csf_g3.sh`; SLURM `serial`, 1 core, 4 GB, 30 min): checks the three script hashes, self-test
54/54, hard cases TOTAL 162/175 (parses cached, so no parser is loaded), curates shards 0 and 20 into
`PG/g3_v1_csftest/job_<id>/`, compares the test, noparent, articles, nonprose and error files with the reference byte
for byte and the reports for 0 problems and 0 unit errors. Dry run on incline (bash, outside SLURM): ALL G3 CHECKS OK;
its output deleted. To submit on CSF: `sbatch /mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/g3_v2/check_csf_g3.sh`.
[LOG]

**RL-124 · 2026-10-10 · G3 · CODE (SLURM array, checked merge) + TEST · LIVE**
RL-123's CSF test (job 22539002) passed: "ALL G3 CHECKS OK" (its log was missing its opening lines, a `tee /dev/stderr`
buffering artefact of the check script, not of the test itself; fixed below). Owner: build the production array and
merge. Host incline37/incline32, `tensor_env`.
`g3_v2/check_csf_g3.sh`: the self-test and hard-case sections no longer pipe through `tee /dev/stderr`, so a rerun's log
is complete from the first line; re-verified with a fresh incline reference run and a bash dry run, both clean.
`g3_v2/g3_curation_v2.py` (sha256 `0f3146266f26`): the report also records `host`, `job` (`SLURM_ARRAY_JOB_ID` or
`SLURM_JOB_ID`) and `task` (`SLURM_ARRAY_TASK_ID`), mirroring G2 v3's done file; every record file unchanged (checked
against the RL-122 outputs of shards 0 and 20, byte for byte). `check_csf_g3.sh`'s pinned hash updated to match; the
incline reference run and the dry check were both redone and pass.
New `g3_v2/submit_g3_v1.sh` (sha256 `d0c8dad02783`): 50-task SLURM array, `serial`, 1 core, 4 GB, 30 min (no parser is
loaded in production G3 — confirmed by import: `focal_terms.py` imports only `re`/`sys`, `modal_merge.py` only
`graphbrain.hedge` — so this is ample); task *i* runs `g3_curation_v2.py --shard i --limit 0 --outdir PG/g3_v1/shards/`.
New `g3_v2/merge_g3_v1.py` (sha256 `a6921837ecf3`): for each shard, checks the report's script sha256 and input hashes
(read from the script itself, not hardcoded) agree across all 50, and 0 problems/0 unit errors; re-derives completeness
by reading the shard's own `g3_articles_NNN.jsonl` against the real G2 v3 shard file directly (every pmcid exactly once,
none missing, none extra, status counts sum to the article count); checks no pmcid is shared across shards; then
concatenates the five record files and writes `g3_v1_summary.json`. Refuses to overwrite; on any failure removes the
partial `.tmp` files it had started (checked: a corrupted report and a pre-existing output each stopped the merge
cleanly, no partial file left).
Tested end to end: all 50 G2 v3 shards curated into a scratch folder (`--limit 0` each, 10 at a time, all 0 problems, 0
tracebacks), then merged there. Merge totals: 34,662 articles (28,075 valid, 121 invalid_no_parent, 6,466 excluded — the
excluded count matches item 6's own figure exactly); 687,824 units, 816,075 parents, 2,266,190 cousins; the five output
files' line counts match the summary's own counts exactly (`g3_test_v1.jsonl` 664,725 = units_with_parent;
`g3_noparent_v1.jsonl` 23,099 = units_removed_no_parent; `g3_articles_v1.jsonl` 34,662 = every article;
`g3_nonprose_v1.jsonl` 1,211 = the four nonprose counts summed; `g3_errors_v1.jsonl` empty, matching 0 unit errors
everywhere). Scratch output deleted; `PG/g3_v1/` not yet produced. Both scripts mirrored to `PG/scripts/g3_v2/`.
To run on CSF (owner): `sbatch PG/scripts/g3_v2/submit_g3_v1.sh`; after all 50 tasks,
`python PG/scripts/g3_v2/merge_g3_v1.py --nshards 50`. [LOG]

**RL-125 · 2026-10-10 · G3 · PRODUCTION RUN · LIVE**
Owner: `sbatch PG/scripts/g3_v2/submit_g3_v1.sh` (job 22540333, 50-task array, `serial`). All 50 tasks finished in about
4.5 min (first report 17:54, last 17:58); every `g3_report_NNN.json` shows 0 problems and 0 unit errors; no failure
signature in any log. `python PG/scripts/g3_v2/merge_g3_v1.py --nshards 50`: every check passed; output written to
`PG/g3_v1/` (refused-to-overwrite guard confirms this is the first write). `g3_articles_v1.jsonl` 34,662 (one row per
article: 28,075 valid, 121 invalid_no_parent, 6,466 excluded — matches item 6's own count exactly); `g3_test_v1.jsonl`
664,725 units with a parent; `g3_noparent_v1.jsonl` 23,099; `g3_nonprose_v1.jsonl` 1,211; `g3_errors_v1.jsonl` empty;
816,075 parents, 2,266,190 cousins. Totals identical to the RL-124 scratch dry run. `g3_v1_summary.json` records
script_sha `0f3146266f26`, merge_script_sha `a6921837ecf3`, the 7 input hashes, and job `22540333`. G3 production is
complete. Next: the M1 input contract (G3_POSTPROCESSING.md §8 item 3). [LOG]

**RL-126 · 2026-10-10 · G3 · DOC · LIVE**
Owner: document the exclusion of the false-positive articles and state the new corpus size explicitly (not just the
component counts). Added a dated note at G3_POSTPROCESSING.md §6 item 6 and at PIPELINE.md's item 24 note and its
corpus-size milestone table (new row): **28,196 articles** (34,662 − 6,466 excluded) is the corpus G3 and M1 work
with; of those, 121 further have no usable parent for the separate reason decided in item 8 (RL-122), leaving
**28,075** articles with actual content in `g3_test_v1.jsonl`. Both figures were re-verified against the real
production run (job 22540333) before writing them, not taken from the plan. [LOG]

**RL-127 · 2026-10-10 · G3 · AUDIT (read-only) · LIVE**
Owner's question, following the "pre_traine" finding of the last session: would it be more reliable to rerun G2
and/or G3? Investigated before answering, not guessed at. Host incline37, `tensor_env`.
(1) G2's hyphen-joining (`join_hyphens`, `graphbrain_parse_v3.py`) is not reversible without a worse cost: tested
directly with `create_parser(lang="en", lemmas=True)` — "The model was pre-trained on PubMed abstracts." (hyphen
kept) parses as `(-/Pd ((was pre/Pd) (the model)))`, losing "on PubMed abstracts" and splitting "pre"/"trained"
apart; "...was pre_trained on PubMed abstracts." (joined) parses correctly, `(was pre_trained) ... (on (pubmed
abstracts))`. The join is a deliberate, already-justified fix for a structural parse failure (PIPELINE.md item
30(l)), not something to undo for the lemma's sake.
(2) The lemma defect is not a process-order bug: "GPT-2 was fine_tuned on the dataset." lemmatises correctly to
`fine_tune/P` through the identical mechanism (same call, same joining) that gives "pre_trained" the wrong
`pre_traine/P`. Re-running the same code on the same text is deterministic and would reproduce the same error
exactly; delaying the join relative to lemmatisation was tested too (lemmatising the unjoined hyphenated form) and
does not help — the unjoined parse is the broken one from (1), so there is no usable compound lemma to take from it
either.
(3) Measured the true scope directly (regex scan of every `_lemma` edge in all 34,662 articles' raw G2 v3 output,
not a sample): 35,201 occurrences across 4,934 distinct (root, lemma) pairs where the underscore-joined root's
lemma differs from the root. The large majority follow one identifiable pattern — a spurious trailing "e" added to
a verb stem that does not take one (`pre_trained`→`pre_traine` 2,198; `retrieval_augmented`→`retrieval_augmente`
803; `ai_powered`→`ai_powere` 377; `board_certified`→`board_certifie` 300; `question_answering`→`question_answere`
221; `self_reported`→`self_reporte` 197; `human_authored`→`human_authore` 152; `resource_constrained`→
`resource_constraine` 149; `english_speaking`→`english_speake` 121; `fastest_growing`→`fastest_growe` 88;
`instruction_following`→`instruction_followe` 83; `prompt_engineered`→`prompt_engineere` 75; about 30 more at lower
frequency), while others of the same shape already lemmatise correctly (`fine_tuned`→`fine_tune` 6,871;
`ai_generated`→`ai_generate` 1,622; `ai_based`→`ai_base` 676; `privacy_preserving`→`privacy_preserve` 299;
`self_supervised`→`self_supervise` 122; `time_consuming`→`time_consume` 89) — the transformer lemmatiser is simply
inconsistent on out-of-vocabulary joined compounds, not uniformly wrong. One outlier noted, a different-looking bug:
`second_best`→`second_b` (76), not the trailing-e pattern.
**Conclusion, given to the owner: rerunning G2 and/or G3 as they are would not help** — both are deterministic, and
nothing about an article's processing history changes their output. The surface parse is already correct (only the
lemma string is wrong), so no G2 rerun is needed at all; a fix belongs in G3, as a small, measured correction table
built the way every other G3 rule was (sample, hand-judge, hard cases, before/after counts on real shards) — not yet
built, pending the owner's decision to proceed. [LOG]

**RL-128 · 2026-10-10 · G3 · AUDIT (read-only) + CORRECTION of RL-127 · LIVE**
Owner asked whether the RL-127 diagnosis and proposal hold. Re-checked; host incline37, `tensor_env`. Corrections:
(1) **Mechanism.** The lemmatiser is not "inconsistent": `en_core_web_trf`'s lemmatiser runs in `rule` mode, and for an
out-of-vocabulary word (every underscore-joined word is one) it takes the first suffix rule without a lookup check
(-ed → -e, -ing → -e, -est → ""): `pre_trained`→`pre_traine`, `question_answering`→`question_answere`,
`second_best`→`second_b`, while the plain word "trained" gives `train`. It is right only by luck where the base ends
in "e" (`fine_tune`, `ai_generate`). Systematic and predictable.
(2) **Counts.** RL-127's 35,201 counted every joined word whose lemma differs from it, including correct cases
(`fine_tuned`→`fine_tune` 6,871) and plurals; its 7,426 / "88% wrong" came from grepping raw JSON, which repeats atoms
in the source lists. Recounted from the parent and cousin edges of `PG/g3_v1/g3_test_v1.jsonl`: `pre_traine/P`
2,823 vs `pre_train/P` 305 (about 90% of verb uses wrong, so the headline holds); the modifier uses are kept as written
under §6 item 8 (`pre_trained/M` 12,525) and are not affected. Over all raw G2 lemma edges, spaCy applied a suffix
rule to a joined word 34,830 times: 21,707 right by luck, 13,123 wrong (2,272 distinct words; types M 8,528,
P 3,308, C 1,286).
(3) **Remedy.** RL-127's hand table of the ~30 frequent pairs would leave about 2,240 distinct wrong words. A rule does
better: where spaCy's lemma of a joined word differs from the word, use the part before the last "_" plus the
corpus's own most frequent lemma of the last part as a plain word (lookup built from G2's lemma edges, so G3 still
loads no spaCy). Tested on the raw lemma edges: it corrects all 13,123 suffix-rule errors listed above that have a
plain-word lemma (371 occurrences have none and stay), leaves the 21,707 right ones unchanged, and fixes gerund nouns
too (`deep_learne`→`deep_learning`, `instruction_followe`→`instruction_following`). Errors of its own seen in the top
45: `second_best`→`second_good` (plain "best" lemmatises to "good"; needs an exception for comparatives and
superlatives) and `plug_ins` stays plural. Not built; the usual checks (labelled sample, hard cases, shards 0 and 20)
would come first, then a G3 rerun (about 5 min) and a new merge. G2 is untouched either way. Agreed with RL-127: no G2
rerun (deterministic, and the parse is right); the hyphen join stays. [LOG]

**RL-129 · 2026-10-10 · G3 · CODE (head-word lemma rule) + CHECK · LIVE**
Owner: proceed with the RL-128 plan. Host incline37, `tensor_env`.
New `g3_word_lists/build_head_lemmas.py` (sha256 `0dcfc42943ae` before the Unicode-hyphen extension; see §9 for the
final hash) writes `head_lemmas.tsv` (sha256 `34cee7ee73d0`, 2,106 rows, words only; copied to `PG/g3_word_lists/`):
for every last part of a joined word whose lemma spaCy changed, the most frequent lemma G2's parse gives that part as a
plain word (word, coarse type). Joined = "_" (G2's hyphen join) or a Unicode hyphen G2 left alone (U+2010/2011/2013;
found when one "pre‐trained" escaped the first version). `g3_v2/g3_curation_v2.py` (sha256 `b3eadb6d385c`), in the
per-unit lemma map: where spaCy's lemma of a joined word differs from the word, the lemma becomes the head (with its
joiner) plus the plain-word lemma of the last part; kept as written when that plain lemma is an irregular form not
sharing the word's first letters (best → good: `second_best` stays; led → lead: `investigator_led` stays); spaCy's
lemma kept when there is no plain-word lemma, or when spaCy only stripped a noun's final "s" and the table has nothing
better (`set_ups`→`set_up`, `plug_ins`→`plug_in`). Two guard fixes found by the check: the prefix test compares at most
as many letters as the shorter word (`ups`/`up`), and the plural guard. Counted as `lemma_head_rule`, examples under
`detail`. Input hash pinned in `INPUT_SHA`.
Checks: self-test 54/54; new hard set `g3_v2/hard/hard_lemmas_rl129.json` (13 cases: joined verbs, gerunds,
participles as modifiers, an irregular superlative, plurals with and without a plain-word lemma) 13/13; all sets
175/188 (the 13 known failures). Shards 0 and 20 (whole) against the RL-125 production shards: units, parents (16,523
/ 17,130), cousins (47,077 / 49,224), no-parent units, article statuses and non-prose units identical; 171 units
changed, every change a lemma; all 57 distinct rewrites (216 occurrences) read by hand — the whole population on the two
shards, not a sample: 56 correct (214 occurrences: `pre_traine`→`pre_train` 99, `meta_analyse`→`meta_analysis` 15,
`document_grounde`→`document_ground` 9, `self_hoste`→`self_host` 5, `sub_specialtie`→`sub_specialty`, `de_identifie`→
`de_identify`, `best_performe`→`best_perform`, `third_b`→`third_best`, ...), 1 wrong (`data_base`→`data_basis`, 2:
"bases" is the plural of both base and basis; accepted). No `pre_traine` left. Shard 20 byte-identical under hash seeds
default and 4242. Production: script and hard sets mirrored to `PG/scripts/g3_v2/`; `check_csf_g3.sh` re-pinned (script
`b3eadb6d385c`, TOTAL 175/188); incline reference rebuilt; dry run ALL G3 CHECKS OK. The RL-125 output was moved, not
deleted, to `PG/g3_v1_rl125/`, so the array can write `PG/g3_v1/` again. Owner to submit on CSF. [LOG]

**RL-130 · 2026-10-10 · G3 · PRODUCTION RUN · LIVE**
Owner: `sbatch check_csf_g3.sh` then `sbatch --dependency=afterok:<check id> submit_g3_v1.sh` (check job 22548003, array
22548004). Race, not a failure: the check job actually ran to completion (sacct: `COMPLETED 0:0`) while its log was read
mid-write and looked cut off after shard 0; the owner's `scancel 22548004` on that wrong reading lost the race against
SLURM releasing the array the moment the check's `afterok` dependency was satisfied — by the time the cancel reached the
scheduler, the array had already launched and (being fast, no parser loaded) finished most or all of its 50 tasks.
Confirmed clean regardless: all 50 `g3_v1_22548004_*.out` logs end `exit 0`; all 50 shard reports 0 problems, 0 unit
errors; no failure signature anywhere. A second, redundant check job (22548129, same `sbatch` line re-run per the
owner's literal instructions) also passed; its test output not touched, scratch only.
`python merge_g3_v1.py --nshards 50`: every check passed; output written to `PG/g3_v1/` (the RL-125 output first moved
to `PG/g3_v1_rl125/`, RL-129, so the refuse-to-overwrite guard allowed this write). Totals identical to RL-125 except
6,171 `lemma_head_rule` corrections and cousins 2,266,190 → 2,266,188. Both missing cousins traced by hand (not assumed
harmless): PMC11605461 and PMC11686211, each a unit where a `pre_traine`-spelled verb-group clause and a correctly
spelled `pre_train` clause used to be two distinct `dummy_cousin` entities and now collapse into one, since both
clauses hold no other atom — the intended effect of RL-129's fix, observed on real data for the first time. `g3_v1/
g3_v1_summary.json`: script_sha `b3eadb6d385c`, job `22548004`, article_status unchanged (28,075 / 121 / 6,466).
`PG/g3_v1_rl125/` kept as the pre-fix record. G3 production, with the head-word lemma rule, is complete. [LOG]

**RL-131 · 2026-10-10 · CORPUS-STATS · CODE (new) + OUTPUT + DOC · LIVE**
Owner: before moving to author counts, confine all further analysis to the 28,075 articles G3 marks
`valid` (not the 121 `invalid_no_parent` or 6,466 `excluded`), and get their types from
PMC/MEDLINE/Entrez/PubMed only — not OpenAlex, not the publisher's own self-classification — flagging
where that source lacks the information rather than filling in from elsewhere.
New `corpus_statistics/article_types_valid.py`. Scope: the 28,075 `valid` pmcids from
`PG/g3_v1/g3_articles_v1.jsonl` (verified against the real file, not assumed from RUN_LOG). Source: the
PMID printed in the article's own PMC header (`diagnostics/openalex_authors/headers.json`, PMC/Entrez
metadata, R2 stage — not OpenAlex) and MEDLINE's own `PublicationType` tag list for that PMID
(`diagnostics/openalex_authors/pubmed_pubtypes.json`, fetched by NCBI `efetch` on 2026-10-06; already
covers all 34,620 PMIDs of the full G2 corpus, a superset of the 28,075, so no new network fetch was
needed). Never falls back to the publisher's self-declared "Subjects:" line, unlike
CORPUS_STATISTICS.md §2 row (a) and the older `diagnostics/paper_type_table.py`.
Gaps reported, not filled in: 38 of the 28,075 have no PMID in their PMC header at all (no
PMC/MEDLINE/Entrez/PubMed type possible for them); of the remaining 28,037, every one has >=1
MEDLINE `PublicationType` tag — no further gaps. 39 distinct raw tags seen (most articles carry one,
up to six); a fixed priority order over these tags (most specific evidence-synthesis/study-design
type first) gives each article one primary type with no "Unclassified"/"Other" residue: research
article 22,672; review 3,802; systematic review/meta-analysis/scoping review 926; editorial/letter/
comment/news 360; clinical trial/protocol 218; no PMID 38; dataset 29; historical article 18;
conference proceedings 12. 46 articles also carry `Retracted Publication` (a status flag, not a
content type, so not its own row — left for a later decision, not acted on). Output:
`corpus_statistics/article_types_valid.csv` (28,075 rows: pmcid, pmid, raw tags, primary type).
Documented in CORPUS_STATISTICS.md §8 (new; §1-§7 kept, with a dated note at the top flagging the
scope/method difference). [LOG]
