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

## RL-173: Haiku review of the remaining group-4 spellings (2026-10-22)

**Why.** Owner, for rigour: the 4,180 group-4 spellings that neither a rule nor the assistant's reading covered in RL-172
(estimated 3–5% false) are each judged by a reader. 4,173 remained once RL-172's residual sample was decided.

**Readers.** Agent type `general-purpose`, model Haiku (`haiku`), no effort level set. Rubric `G4_READER_RUBRIC.md`
(model / not_model / mixed / unsure, with examples of each). Items (`g4_haiku.py build`): one spelling, its occurrences and
articles in the kept corpus, up to three contexts from different articles with the article title; 17 batches (250, last
173). Prompt (verbatim, NN = batch number; calibration: `calib.txt`, "72 in total", `calib_haiku.jsonl`, single write):

> You are a careful reader doing a narrow classification task. Work only with the files named here; do not search,
> explore, or open any other file, and do not run code.
> 1. Read the rubric: …/g3_inputs_v3_1/g4/haiku/RUBRIC.md
> 2. Read the batch: …/g3_inputs_v3_1/g4/haiku/batch_NN.txt — it is long: read it in pieces with offset/limit until you
>    have seen every line.
> 3. Answer every QUESTION line by the rubric, judging the spelling from its contexts and article titles. Use "unsure"
>    when the shown text does not decide it, and "mixed" when the contexts disagree; do not force a choice.
> 4. Write all answers to …/g3_inputs_v3_1/g4/haiku/batch_NN_haiku.jsonl — one JSON object per line, exactly the
>    rubric's output format, one line per question, no other text. Use the Write tool (you may write the answers in
>    several steps, each time rewriting the file with all answers so far).
>
> Finish by replying with one line: the number of answers written and how many of each verdict.

(… = `/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain`.)

**Calibration** on 72 spellings the assistant had decided in RL-172, none among the rubric's examples (36 model, 36
not_model): 71 of 71 decisive answers agreed; the 72nd (AMBERT) was answered "mixed", and the reader was right where the
gold label was not ("AMBERT 14" in one article is a typo for the AMBER software).

**Full run.** 4,173 of 4,173 answered, no format errors: model 3,683 spellings (27,311 occurrences), not_model 364
(1,036), unsure 119 (497), mixed 7 (46). Every verdict with its note: `g4_haiku_verdicts.csv`.

**Assistant's check.** All 490 not_model, unsure and mixed verdicts read; 50 random model verdicts read (50 correct). Three
lines applied consistently (`g4_haiku_decisions.py`): (1) a model's own code class (TFBertModel, GPT2LMHeadModel,
DistilBertTokenizer …) names that model and is mapped to it (21 spellings; the rubric had listed code identifiers as
not_model); (2) "<model>-extracted / -encoded / -assisted / -generated / -guided" and glued sentence fragments
("ChatGPTas", "settings.GPT4TS56") refer to the model and are mapped to it (34); (3) a named LLM application, assistant
or agent is a model (MemGPT, HuggingGPT, ZotGPT, FraudGPT, …: 25 kept), a named method, library, metric, framework,
dataset, label, variable or repository is not (StructGPT, SparseGPT, TopicGPT, SelfCheckGPT, notation such as hiBERT: the
other unsure/mixed rejected). The citation rule was widened to every kept spelling (a trailing ".NN" or 2–3 glued digits
whose removal leaves a spelling of the corpus: 138 more, AntiBERTy19 → AntiBERTy, TCR-BERT.31 → TCR-BERT; one digit never
removed).

**Result** (`decisions/g4_decisions.csv`, 5,619 rows): 644 spellings rejected (2,623 occurrences), 1,404 mapped, the rest
kept. Shards 0 and 20: 46 and 145 mentions rejected (RL-172: 16 and 136), 0 problems. Every group-4 spelling of the kept
corpus now carries a decision from a rule, the assistant's reading or a reader checked as above.

**Cost.** Agents' completion reports: calibration 61,177 tokens; batches 89,345–119,281 each; 1,995,419 in all; about 3–4
minutes per batch, all 17 in parallel.

## RL-174: one line for the model / method boundary; residual homonyms (2026-10-22)

**Why.** Owner's questions on RL-173: the boundary between models and methods had been drawn inconsistently (same name
decided both ways in different spellings, e.g. SBERT-k-means kept but SBERT-K-Means rejected, LLaMA-adapter kept but
LLaMA-Adapter69 rejected; similar LLM tools decided differently, GPTCelltype kept but OntoGPT and TopicGPT rejected).
Owner: apply one explicit line to the whole boundary set; map "model + algorithm" pipelines to their model.

**The line** (`g4_boundary_decisions.py`; "framework" vs "app" and "method" vs "model" could not be applied
consistently, because the same system is described as both):
- **keep** — the name denotes something whose outputs are produced by a language model: a trained or fine-tuned model,
  or an LLM-backed app, agent, assistant, framework or pipeline that answers or generates (UroGPT, TrialGPT, OntoGPT,
  GPTCelltype, TopicGPT, StructGPT, FrugalGPT);
- **map to the underlying model** — a use or configuration of a specific model: run, arm, condition and answer labels
  (BERT_FT, GPT4x3, Llama70BCombinedAnswer), "-based/-driven/-powered/-aligned/-optimized", compounds of a model with
  other components (SBERT-k-means, BiLSTM-CRF-BERT, DeBERTa-GCN, GCN_ProtBert), adapters (LLaMA-Adapter → `llama`),
  modules of a named system (TrialGPT-Ranking → `trialgpt`), glossary letters glued to the name, misspellings;
- **reject** — outputs not produced by a language model: procedures that transform models (SparseGPT, QuantGPT, KnGPT,
  oBERT, bert2BERT), detectors, metrics and benchmarks (DetectGPT, BERTScore, MedGPTEval), non-generative libraries and
  tools (LlamaIndex, GPTCache, BERTopic, KeyBERT, exBERT, visBERT), datasets, files, variables, notation, non-language
  networks (GPTNeXt).
Variants that differ from a reviewed spelling only in separators or a glued citation number get its decision; case is
not ignored, because it separates names (ExBERT a model, exBERT a tool; GBERT German BERT, gBERT a glossary entry;
minGPT a repository, MinGPT a model). Case variants of one name were decided explicitly (12).

**Reading.** 479 boundary spellings read by the assistant with a context each (kept spellings whose notes say framework,
system, app, method, tool, agent, pipeline, platform or library; rejected ones whose notes say method, tool, framework
and the like; all name groups decided both ways). Result: 151 mapped, 24 kept (formerly rejected: OntoGPT, TopicGPT,
StructGPT, AugGPT, FrugalGPT, GPTArticleExtractor, ZoteroGPT, tpGPT, GPT4Designer, IoTGemini, GPTSwarm, ViperGPT,
GPTCelltype …), 14 rejected (bert2BERT, GPTNeXt, GPTZip, InfoBERT, VisBERT …), plus 12 variants harmonised. Table now
5,623 rows: 613 spellings rejected (2,387 occurrences), 1,546 mapped, 3,464 kept. Remaining case-differing groups were
listed and read: all differ in meaning (above) or are citation variants of their base name.

**Residual homonyms of listed names** (owner's question 1). ESM-2, BioBridge, ProGen and PaLM 2 are names of P2's list,
not group 4, and item 6 removed only articles resting on them alone. Every kept mention was searched for the homonym
senses found in item 6 (climate models MRI-/FIO-ESM-2-0, Electronic Supplementary Material "Table/see ESM-2", the BioBridge
scaffold and implant, the Progen supplier, the Palm 2 domain): ESM-2 5,235 mentions, 66 pattern hits, all the protein model;
PaLM 2 896, 9 hits, all the language model; ProGen 257, 8 hits, all the protein model; BioBridge 6, of which 2 the rib
plate in PMC11320226. That article's other term, "Bard", is C. R. Bard (Marlex mesh): the article is excluded
(`decisions/manual_homonyms_rl174.csv`). The same Bard sense, searched in all 1,026 kept articles with Bard: 2 more
sentences ("BD Bard" in a disclosure, PMC12492233; "Bard Electrophysiology Division", PMC11066324), rejected at sentence
level; the other 52 pattern hits are the chatbot. Exclusions 7,159; term rejections 703.
