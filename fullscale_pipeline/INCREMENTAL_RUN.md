# Incremental run after the corpus extension (ticket 107 P1, from 2026-10-21, RL-165 onward)

Goal: bring the 15,563 new articles and the matcher changes through P6 → P7 → G2 without redoing the current corpus,
keeping every existing sentence id, hash and citation id that does not have to change. Owner approval: 2026-10-19 (in
general), 2026-10-21 ("proceed with the incremental pipeline").

## Units

| Unit | Articles | What changes |
|---|---|---|
| New | 15,563 (`gap_stage/gap_article_list_after_ab.txt`) | everything from P6 on (P2 done in the staging tree) |
| Current, removed | 980 (`LCS/article_exclusions_stage_b.csv`) | dropped before M1 (no stage is rerun for them) |
| Current, removals only | to be counted | sentences dropped downstream: rejected hits (`stage_b_rejected_hits.tsv`, 255) and AI-use statement sentences (owner: statements are not analysed) |
| Current, additions | to be counted | the v3 matcher finds new focal sentences (CT-BERT, ChatGPT4, …): P6 rerun for the article |
| Current, unchanged | the rest | nothing |

Sentence ids (`PMCID.rN.L<raw line>.S<k>`) and P2 hashes are deterministic, so v2 and v3 P2 records are compared by id.

## Steps

1. **P2 v3 on the current corpus** → `LCS/p2_v3_current/` (existing P2 output untouched). Compare with v2 by article
   and sentence id: classify articles as unchanged / removals only / additions.
2. **Prepare** (`retrieval/inc_prepare.py`): P2 inputs for P6 = new articles + current articles with additions, with
   rejected hits and AI-use statement sentences un-flagged (focal = false, terms emptied; blocks are not rebuilt, so
   sentence ids stay). A drop list of sentence ids for the removals-only articles.
3. **P6** (coreference) on the prepared inputs, as two runs (the new articles' cleaned texts and regions live in the
   staging tree): CSF array with `submit_coref_v2.sh`-style launchers, output in a separate folder. Unchanged current
   articles keep their existing P6 records.
4. **P7** (`focal_citations_v2.py`) rerun on the combined P6 records (existing records for unchanged articles + new
   ones), with the combined citation dictionary, regions and raw texts; then **REF id remapping**: a work that contains
   tokens of an existing work keeps that work's id; a new work gets the next free number; a new link between two
   existing works is counted and reported (the smaller id kept, the other recorded as an alias). The drop list of step 2
   is applied here.
5. **G2** (`graphbrain_parse_v3.py --input …`) only on sentences whose (sid, hash_final) is not in the existing G2 v3
   output, plus the 876 curly-apostrophe re-parses (RL-152) and re-parses where G2 joined an unlisted model name. CSF
   array; merged with the existing parse per article.
6. **Checks**: every existing sid, hash, hash_final, unit id and REF id unchanged except where listed with a reason;
   counts per unit and per year.

Then, as planned: one G3 production rerun (D4.1, B06, D6), the D5 dash count, re-verification of the P1 trial checks,
and P2 weighting for the chunk12v3 trial matrices.

## Where CSF is needed
P6 and G2 (array jobs, launchers as `.sh` files for the owner to submit). Steps 1, 2, 4 and 6 run locally.

## Traceability and uniqueness when old and new material are merged (owner, 2026-10-21)

Provenance chain, unchanged by the merge: word (atom2word: atom → word position in the unit text) → unit (`<sid>.U<k>`,
unit hash) → sentence (`PMCID.rN.L<raw line>.S<k>`, hash, hash_final) → block (`PMCID.rN.B<k>`) → article (PMCID);
hyperedges carry their unit id, so every hyperedge and its parents trace to a sentence and an article.
Rules for the merge:
1. **One record per article, in one place.** The merged G2 output is written as a fresh shard set (`PG/g2_v3_1/`), every
   article exactly once: an existing article keeps its old parse for untouched sentences, and its re-parsed or new
   sentences replace or join them inside the same record. No article is read from two shard sets.
2. **Ids are never reused or renumbered.** Sentence and unit ids are built from the PMCID and raw line, so new articles
   cannot collide with old ones (the two article sets are disjoint: 0 shared PMCIDs). Existing REF ids are frozen; new
   works are numbered from the next free id; a merge of two existing works keeps the smaller id and records the other
   as an alias (no id points to two works).
3. **Hashes are recomputed, never copied** for anything whose text changed; an unchanged sentence keeps its hash.
4. **Checks before G3** (written to a report): every sid, uid and REF id unique across the merged output; every sid
   belongs to the article its id names; every REF token in a text_final exists in the works file; every unit hash and
   hash_final recomputes from its id and text; every existing (sid, hash_final) and unit id is unchanged or listed with
   its reason (re-parse, dropped sentence, REF remap); no sentence of an excluded article or a dropped sentence remains;
   counts per corpus, class and year.
5. URL tokens (`URL` + 8 digits) are deliberately shared: the same address gets the same token in every article.

## Status (2026-10-21): done — merged output `PG/g2_v3_1/` passes all checks (RUN_LOG RL-165 to RL-169)

| Step | Result |
|---|---|
| 1 P2 v3 on the current corpus | 34,777 articles with a focal sentence (v2: 34,662) |
| 2 Prepare | current: 26,034 unchanged, 2,178 removals only, 5,470 additions, 101 new under v3 (+1 excluded), 13 nothing left; new: 15,563 |
| 2b Statement correction | broad sentence-level rule was ~10% precise inside kept articles; 6,205 sentences checked by Haiku readers + assistant (`retrieval/STAGE_AB_METHOD.md` §6): 1,222 statements removed, 4,983 research sentences restored |
| 3 P6 (CSF 23477301, 23477302) | 15,563 + 5,571 articles, all present once, no errors |
| 4 P7 with frozen REF ids (`p7b`) | 10,499 existing ids kept, 5 new links (aliases), 2 splits, 2,191 new works; existing sentences 696,948 unchanged, 482 changed (rerun P6 antecedents, REF links/splits), 5,518 not carried |
| 5 G2 (CSF 23488211 + local round 2) | 120,582 + 2,606 sentences parsed, 0 errors; 704 hyphenated-v3 and 878 apostrophe re-parses |
| 6 Merge and checks | `PG/g2_v3_1/`: 49,346 articles (33,783 current, 15,563 new), 816,678 sentences, 825,693 units, 12,690 works; all checks passed; 5 homonym sentences re-entered through coreference removed after review, 4 genuine ones kept |

Inputs of record beside the shards: `focal_sentences_v2.jsonl` (P7, mapped REF ids, statements removed),
`citation_works_v2.jsonl`, `url_table_v3_1.json`, `check_report.json`. The existing `PG/g2_v3/` and `LCS/` outputs are
untouched; the current corpus's article exclusions (`LCS/article_exclusions_stage_b.csv`, 980: 789 AI-use statement only, 104 all hits rejected, 87 F2 other sense) are applied in this
output, not yet in `LCS/article_whitelist.txt` (only P0–P2 read the whitelist; nothing downstream needs it changed).

## A3 part 2 (2026-10-26/27, RUN_LOG RL-203): sentences lost upstream by the v3 matcher

Same principles, one more round: P2 with `focal_terms_v3_1.py` (v3 + G3's hub spellings) on both corpora, compared by
sentence id (`retrieval/a3p2_prepare.py`): 424 additions, 0 lost; read by the assistant, 45 rejected → 379 sentences in
180 articles. P6 locally (`a3p2_p6.py`), P7 with frozen ids (`a3p2_p7.py`; reproduction check first; only the accepted
additions added, existing sentences untouched), G2 for the 378 new sentences with the hub spellings kept whole
(`a3p2_g2.py`), positions, merge → **`PG/g2_v3_3/` + `g2_v3_3_pos/`** (817,056 sentences, 826,081 units; checks passed).
G3 reads g2_v3_3 from RL-203 on.
