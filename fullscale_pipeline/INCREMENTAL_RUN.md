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
