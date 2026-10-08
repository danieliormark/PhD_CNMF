# Stage G2 — graphbrain parse of the focal sentences (v3, current)

Status: **DONE-current**, run on the whole corpus 2026-09-29 (RUN_LOG RL-074 to RL-077, corrected in RL-081).
Part of the full-scale pipeline described in [`PIPELINE.md`](PIPELINE.md) (stage table §1, item 30, deviation D16).
The script's own docstring (`PG/scripts/graphbrain_parse_v3.py`) is the authoritative short description; this
document explains it, records how it was run and checked, and lists what is known to be imperfect.
Paths use the abbreviations of PIPELINE.md (`LCS`, `PP`, `PG` under `R = /mnt/hum01-rds/Basov/p91688di`).

## 1. Purpose and place in the chain

G2 turns every focal sentence of the corpus into graphbrain hyperedges (semantic hypergraph notation), one edge per
sentence unit, each traceable to its article, sentence and unit. It reads the output of P7 v2 and is read by G3
(postprocessing, [`G3_POSTPROCESSING.md`](G3_POSTPROCESSING.md)).

```
P7 v2  LCS/focal_sentences_v2.jsonl  ──►  G2 v3  PG/g2_v3/  ──►  G3 (test stage)  ──►  M1 (not built)
```

G2 v3 reads P7's output directly and shards it itself; the earlier partition step G1 and the outputs of the May G2
and of G2 v2 are not used by it or by anything after it (PIPELINE.md §1 keeps them as superseded rows).

## 2. Input

`LCS/focal_sentences_v2.jsonl`: 34,662 articles, 702,948 focal sentences (sha256 of the file `0bc9d0f6f695`). Each
sentence has a stable id `sid` (e.g. `PMC12476623.r1.L92.S3` = article, region, raw line, sentence), a hash
`hash_final` and the text `text_final` in which pronouns are resolved (P6 v2) and cited works carry shared ids
`REF<6 digits>` (P7 v2). G2 never changes `sid`, `hash_final` or `text_final`; everything below happens inside one
sentence. The caption-body list (sentences whose caption label P2 left on the previous sentence) is computed by each
task from P2's `LCS/focal_extractions_v2.jsonl`.

## 3. What happens to each sentence

1. **Non-prose sentences are not parsed** (status `nonprose`, with the reason): a caption at the start ("Table 3 …",
   "Fig. 2 …"; "Table 2 (See App. B) shows …" is prose and kept), a caption behind a DOI or figure id, a lone label
   ("Fig 5"), a reference-list line (three or more author names with initials), a heading of at most 4 words ending
   with a colon ("GPT-4o response:"), and a caption body whose label P2 left at the end of the previous sentence.
2. **A caption label glued to the end** of the sentence ("…).Fig.") is removed.
3. **Glued citation numbers are removed** and the text is cut there into pieces: "… provided by OpenAI.5 To reduce …"
   becomes "… provided by OpenAI." + "To reduce …". A number counts as a glued citation when it follows a period
   attached to a word, "%", ")" or "]" and is followed by a capitalised word; it is not removed after fig, eq, table,
   no, ref, sec, vol, p/pp, step, phase, stage, type, grade, level, version and similar, nor after a lone letter
   (section numbers such as "A.3", initials).
4. **Normalisation inside each piece:**
   - URLs and DOIs become `URL` + 8 digits: `sha1(normalised address) mod 10^8`, where normalisation lower-cases and
     strips `http(s)://`, `www.`, `doi.org/`, `doi:` and a final `/`. The same address gets the same token in every
     article (graphbrain otherwise makes each URL one escaped atom that never recurs). The token table is written out.
   - "∼110"/"~110" and "≈" become "approximately", "≥" "at least", "≤" "at most"; an arrow between numbers becomes
     "to", any other arrow a comma. Other mathematical signs are left for G3.
   - Hyphenated words are joined with "_" ("pre-trained" → "pre_trained"): parsed with the hyphen, the parser loses
     verbs and objects ("was pre-trained on PubMed abstracts" gave the predicate "-"). **Exception (owner decision
     2026-09-29):** a hyphenated word whose parts contain a focal term (any term of `PP/focal_words.txt`, single or
     multi-word) is left as it is, so the focal term stays its own atom ("LLM-based", "AD-LLM", "SMILES-BERT",
     "large-language-model-based").
5. **Parse and sentence-unit rule.** Graphbrain (with spaCy `en_core_web_trf`) parses the piece. Its own splitter
   sometimes cuts a sentence in two; a cut is kept only where one sentence really ends and another begins: the text
   before ends with . ! ? or … (optionally followed by a citation number and closing quotes/brackets) and not with an
   abbreviation that never ends a sentence (e.g., i.e., cf., vs., approx., ca., resp., incl., Fig., Eq., No., Vol.,
   pp., Dr., initials such as U.S.; "etc." and "et al." count only before a capitalised word), and the text after
   begins with a capital letter, a digit, "(i)" or "(a)". If any proposed cut is rejected, the piece is parsed again
   with only the kept cuts preset as sentence starts on the tokens before the parser runs, so the parser respects
   exactly those boundaries. (A first version that kept a cut only when `en_core_sci_sm` also found it was rejected
   as circular; scores of the frozen rule on hand-labelled cuts are in PIPELINE.md item 30.)
6. **Unit clean-up:** a unit without letters is dropped (`debris`); a caption or reference line that starts a later
   unit is dropped together with every unit after it; a later unit that is only a short heading is dropped.
7. **Unit ids and hashes:** each kept unit gets the id `<sid>.U<k>` (k from 1) and the hash
   `sha1("<uid>|<parsed text>")[:12]`, where the parsed text is the unit's text after steps 2–4.

Every change is recorded per sentence under `transforms` (glued citations, URLs, symbols, hyphens joined and kept,
label or caption removed), so the parsed text can be traced back to `text_final`.

## 4. Output

In `PG/g2_v3/` (shard files in `shards/`, `NNN` = 000–049):

| File | Content |
|---|---|
| `shards/g2_parsed_NNN.jsonl` | one record per article (see below); the task's checkpoint for resuming |
| `shards/db_provenance_NNN.sqlite` | graphbrain database per shard: `('source', PMCID, main_edge)` plus lemma edges; each source edge carries the attribute `occurrences` = list of {uid, unit hash, sid, hash_final, text, atom2word} (one entry per unit with that edge in that article); 708,631 source edges in total |
| `shards/url_table_NNN.json` | token → normalised address |
| `shards/g2_errors_NNN.jsonl`, `shards/g2_done_NNN.json` | error notes; per-shard counts, script hash, input, host, SLURM job id |
| `g2_parsed_v3.jsonl` (2.5 GB) | all 50 shards joined, 34,662 article records in shard order |
| `g2_v3_summary.json` | totals, input and script hashes, all job ids |
| `url_table_v3.json` | the joined URL table, 1,674 tokens |

Total size about 120 GB, almost all of it the 50 databases.

Article record in the JSONL:

```
{"pmcid", "sentences": [ {"sid", "hash_final", "status", "reason",
    "pieces": [{"orig", "parsed"}], "transforms": {...},
    "units": [{"uid", "hash", "piece", "text", "main_edge", "extra_edges", "failed", "atom2word"}],
    "dropped": [{"piece", "text", "reason"}] } ]}
```

`status` is one of ok, partial (some units without an edge), no_edge, exception, nonprose (not parsed), empty (every
unit dropped). `main_edge` is the graphbrain edge as a string (`hedge(string)` restores it; checked for every edge);
`extra_edges` are graphbrain's lemma edges `(_lemma word/T/en lemma/T/en)`, lemmas from `en_core_web_trf` in context;
`atom2word` lists `[atom, word, token position]` for every atom of the unit.

**Provenance chain:** article (`pmcid`) → sentence (`sid`, `hash_final`, from P2/P7) → unit (`uid`, unit hash) →
edge and its atoms (with word and token position). The database keeps the same chain in `occurrences`, but merges
identical edges of one article into one source edge; the JSONL keeps one record per unit.

## 5. How it was run, and how to rerun it

Environment: conda `tensor_env` with `PYTHONPATH=$HOME/np1_for_spacy` (spaCy 3.4 needs numpy < 2), graphbrain 0.7.0,
spaCy 3.4.4, `en_core_web_trf` 3.4.0. Graphbrain 0.7 needs `parser.atom2token = {}` before each call (done in the
script).

```
python graphbrain_parse_v3.py --selftest                               # 57 cases
sbatch PG/scripts/check_csf_g2v3.sh                                    # CSF test: self-test, 2 limited shards, --verify
sbatch PG/scripts/submit_g2_v3.sh                                      # array 0-49, multicore, 4 CPUs, 8 GB, 6 h
python3 PG/scripts/merge_g2_v3.py --nshards 50                         # on incline; needs neither graphbrain nor spaCy
```

- Articles go to shard i by input position (`position % 50`). A task resumes after its last complete article if
  resubmitted (`sbatch --array=<i> …`); a finished task (with `g2_done_NNN.json`) does nothing.
- The database is built on the node's local disk in one transaction and then copied (in May all G2 errors were
  SQLite disk I/O errors on the RDS drive); its source-edge count is checked after the copy.
- `merge_g2_v3.py` is a separate script (owner decision, RL-076). It re-derives every check from the input and the
  shard files instead of trusting the done files: every shard done with this input, `nshards` and no `--limit`;
  every database's source-edge count equal to its done file; every input article present exactly once with
  unchanged sid and hash_final; every unit id and hash verified; URL tables agreeing on shared tokens. It refuses to
  overwrite its output.
- **Run of record (2026-09-29):** CSF test job 21557049 (16:36, ALL G2 v3 CHECKS OK, script `0e23963b5360`); array
  job 21557322, all 50 tasks exit 0 with no resubmission, started 16:53–17:29 and finished 17:20–17:59, about
  0.1 s per sentence, 3.2 GB peak, script `a3d7f986d292` in all 50 done files (the version without the merge code;
  the parsing code is the same as in the tested version). Each task has its own `SLURM_JOB_ID`
  (21557693 … 21558717; task 49 carries the array id 21557322); these are the ids in the done files and the summary.
  Merge on incline 21:56–22:34, 0 problems. Logs: `PG/scripts/logs_g2_v3/`.

## 6. Results of the full run

| Count | Value |
|---|---|
| articles / sentences | 34,662 / 702,948 |
| sentences parsed, status ok | 696,351 (no partial, no_edge or exception) |
| non-prose sentences not parsed | 6,591: captions 3,865, DOI captions 1,398, caption bodies 694, reference lines 574, headings 60 |
| sentences whose every unit was dropped | 6 |
| units | 710,869; 683,675 sentences give one unit, 11,325 two, 1,351 three or more |
| units dropped | 36: debris 30, heading 3, reference line after prose 2, caption after prose 1 |
| sentences cut at a glued citation | 2,032 (2,164 numbers removed) |
| captions / labels removed from the end of a sentence | 1,542 / 582 |
| URLs and DOIs tokenised | 2,947 (1,674 distinct tokens) |
| symbols normalised | 1,500 |
| hyphenated words joined / kept for a focal part | 362,204 / 62,652 |
| database source edges | 708,631 |
| maximum edge depth | 101 (one unit, below) |
| error lines | 1 |

## 7. Known limitations

- **One unit nested 101 levels deep** (`PMC12476623.r1.L92.S3.U1`, a sentence that is a long comma-separated list of
  search terms). It is parsed and kept; the error line only notes the depth. G3 raises the recursion limit for it.
- **Parser typing errors seen in G3 tests** (G2 keeps the parse as graphbrain gives it): a verb typed as a noun
  ("can automatically flag reports" → `flag/C`), a copula "be" typed as an auxiliary (`Mv`) in "cannot be
  absorbable", "can" typed `M` instead of `Mm` (29 of about 1,200 in 700 articles), and lemma artefacts of the
  transformer lemmatiser on joined words ("pretrained" → `pretraine`). Their frequency beyond these samples has not
  been measured.
- **Repeated words:** graphbrain edges do not carry token positions, so when a word occurs twice in a unit its atom
  string is the same for both occurrences; `atom2word` lists both positions. Attribution of atoms to sub-edges is
  exact; only the token index of such an atom is ambiguous (10.7% of the atom records in a G3 test).
- **Upstream, not G2:** the focal term "LLM" has no homonym guard in P2 (PMC8815195 uses it for lipid-lowering
  medication); pending owner decision, see G3_POSTPROCESSING.md §6.
  *[2026-10-08: decided in G3_POSTPROCESSING.md §6 item 6: 6,466 articles, homonym and off-topic ones, are excluded
  from G3 and M1 by `g3_scope_exclusions/scope_exclusions.csv`; G2 itself is not rerun.]*
- The sentence-unit rule misses some boundaries (missing periods, comma splices, bullets) and recall of the
  non-prose rules was not measured (precision per rule in PIPELINE.md item 30).

## 8. History

| Version | Script | What it did | Why it was replaced |
|---|---|---|---|
| G2 v1 (May) | `02_transformer_node.py` | bare edges, no link to the article | no provenance (D6) |
| G2 (May) | `parse_stage_v2.py` after G1 `01_matrix_partition.py` | whole blocks parsed, edges tied to the article only | no sentence provenance, silent skips, disk I/O errors in shard 48, whole-block edges nested up to 383 levels that crashed the May G3 (D15) |
| G2 v2 | `graphbrain_parse_v2.py` (RL-070 to RL-072) | one P7 sentence per parser call, statuses, checked merge | graphbrain's own re-splitting (22,091 sentences split, about 40% of cuts false) and glued citation numbers hiding real boundaries; no non-prose removal, URL or hyphen handling (item 30, D16) |
| **G2 v3** | `graphbrain_parse_v3.py` + `merge_g2_v3.py` (RL-074 to RL-077) | this document | — |

The outputs of the May G2 (`PG/output_sqlite_v2/`) and of G2 v2 (`PG/g2_v2/`) are kept but not used.

## 9. Files

| File (under `PG/scripts/`) | sha256 (12) |
|---|---|
| `graphbrain_parse_v3.py` | a3d7f986d292 |
| `merge_g2_v3.py` | 7cb4b1d7f1c9 |
| `submit_g2_v3.sh` | a6ac2ec80db2 |
| `check_csf_g2v3.sh` | 85838f5aa8e8 |
