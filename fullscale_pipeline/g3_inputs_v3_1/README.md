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

## RL-172 additions (2026-10-22)

| File | What | Built by |
|---|---|---|
| `decisions/g4_decisions.csv` (copy `PG/g3_inputs_v3_1/g4_decisions.csv`) | group-4 model names (v3 matcher, class G4): 234 spellings rejected (1,334 occurrences: genes, plasmids, peptide/palmitate "Palm" tokens, palmitoylation tools, Geant4 physics lists, scales, trial names, devices, person names, tools), ROBERT decided per article (2 kept, 9 rejected), 11 read mappings (typos, Med-PaLM spellings) and 1,200 glued-citation mappings (SpikeGPT28 → SpikeGPT; of three digits the first is kept as a version when name + digit is a known name: DNABERT217 → DNABERT2) | `g4_review.py` (lists with contexts, RDS only) → reading → `write_g4_decisions.py` |
| `statement_sentences.csv` | 138 AI-use statements found by the second statement check (`retrieval/stmt_check2.py`, STAGE_AB_METHOD.md §7); G3 skips them; the 91 articles with no other focal sentence are in `scope_exclusions.csv` (reason `ai_use_statement_only`) | `retrieval/stmt_check2.py` + reading |
| `scope_exclusions.csv` | now 7,158 (RL-171's 7,067 + 91 statement-only) | `build_inputs_v3_1.py` |

**Group-4 review, how it was done** (owner 2026-10-22: false positives must not become focal atoms; a sentence whose only
focal term is one is removed — G3 does this, since such a sentence gets no parent).
1. Precision sample of 100 G4 occurrences (60 from spellings in one article, 40 from the rest): 4–5 false among the 60, 0–1
   among the 40; real names often carried a glued citation number (5 of 60).
2. Candidate rules (gene-style GPT names, "Palm" chemistry, `_BERT_` physics lists, capitalised words, glued digits)
   applied to all 5,900 spellings of the kept articles; every hit of the first four read (118 spellings), the glued-digit
   hits (1,169) checked by stem (all real model names, mostly BioBERT/PubMedBERT/SciBERT/DNABERT + citation).
3. Every spelling (431) of the 700 articles whose only focal terms are G4 names read with title and context.
4. Residual sample of 60 occurrences from the 4,180 spellings neither under a rule nor read: 2–3 false (HMW-GS-GPT, a
   glutenin hydrolysate; GeminiMol, a molecular model; LLaMA-Factory, a tool), added. **Residual estimate: about 3–5% of
   the ~29,000 unreviewed occurrences**, mostly in articles that also carry other focal terms.
5. Checked on shards 0, 2, 20: 16, 10 and 136 mentions rejected, 47, 40 and 34 mapped, no rejected spelling left as a
   focal atom.

**Checkpoint names** (owner 2026-10-22, in `g3_curation_v2.canonical`): a family name followed by a configuration
(size, base/large, cased/uncased, multilingual, language, instruct/it/chat, pro/flash/lite, turbo, exp/preview, snapshot
codes, "-based") is written as family + version, like its prose form: bert-base-uncased → `bert`, gemma-3-12b-it →
`gemma_3`, gemini-2.5-pro → `gemini_2_5`, BERT-based → `bert` (the property word "based" of these 109 occurrences is not
kept as an atom); GPT-35 (Azure's name) → `gpt_3_5`. 292 spellings, 1,561 occurrences change.
