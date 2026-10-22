# G3 inputs for the merged corpus (RL-171, 2026-10-22)

G3 (`g3_v2/g3_curation_v2.py`) reads, besides the G2 parse, lists that were built in October for the old corpus
(G3_POSTPROCESSING.md §6 items 6–8). For the rerun on `PG/g2_v3_1/` (old and new articles, RL-169) they are rebuilt
here with the same rules; the RL-130 lists stay in `g3_scope_exclusions/`, `g3_llm_output/`, `PG/g3_reflexive/`.
Data copies on RDS: `PG/g3_inputs_v3_1/`; G3 checks each file by sha256 (`INPUT_SHA`).

| File | What | Built by |
|---|---|---|
| `scope_exclusions.csv` | §6 item 6: articles G3 and M1 leave out. Old list (6,466) − 5 restored + 606 new (585 new articles, 21 current) = 7,067 (206 of them no longer in g2_v3_1) | `screen_item6.py` → `write_item6_decisions.py` (decisions, `decisions/item6_v3_1.csv`) → `build_inputs_v3_1.py` |
| `spelling_corrections.csv` | the old 6 + 1 ("large linguistic model (LLM)" → language) | `build_inputs_v3_1.py` |
| `llm_output_units.csv` | §6 item 7: units quoting model output; 169 (old 162 all kept, 7 added through v3 names) | `g3_llm_output/build_llm_output_units.py --shards … --excluded … --matcher v3` |
| `reflexive_restored.jsonl` (RDS only, corpus text) | §6 item 8: emphatic reflexives restored and re-parsed; 1,393 units (1,202 identical to the old table, 3 re-done on their new parse, 188 new; 6 old ones belong to dropped articles) | `g3_v2/reflexive_restore.py --coref inc_run/p7b/lcs/coref_v2/… --shards g2_v3_1` |
| `term_rejections.tsv` | focal hits Stage B rejected, so that G3 (which re-finds terms in the text, now with the v3 matcher) does not bring them back: 677 sentence-level homonym-rule rows, 29 article-level prerelease rows; F2 verdicts give none (every "no" fell on a dropped article); 5 rows lifted for the misnomer article | `build_stage_b_rejections.py` (the calls of `retrieval/inc_prepare.clean`) → `build_inputs_v3_1.py` |

Item 6 screen (`screen_summary.json`): the new articles and the current articles whose focal terms changed in the
incremental run (additions, removals_only, new_in_v3); old decisions stand for all others. Tests copied from
`g3_scope_exclusions/evidence/`. One reading of rule 2 differs from a literal replay: an article whose only "LLM" is
undefined but which carries another accepted model name (Mistral, GPT-2, XLM-RoBERTa, BioBERT) counts as having
evidence (20 articles kept; the old evidence regex misses these spellings).
