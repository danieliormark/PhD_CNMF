# Corpus extension: Stage A and Stage B cleaning — method (2026-10-20/21)

Ticket 107 P1. Run log entries RL-157 to RL-163. This file records the method in one place, including the exact
instructions given to the reader agents. Lists of dropped and kept articles are in `stage_b_lists/` (below).

Terms: a *list* is a sequence of article ids (PMCIDs); a *corpus* is a set of text files. Nothing was deleted from any
corpus; every removal is a row in a list, with a reason.

## 1. Input

- Article list: the overnight full-text candidate search (RL-155), 116,783 PMCIDs not in R1, of which 111,552 have an
  allowed type (research article, historical article, any review; RL-156).
- Downloaded corpus: `LCS/pure_text_corpus_gap/`, 118,590 files (the eligible list plus the earlier title/abstract set).

## 2. Stage A — the original preprocessing on the new files (`gap_stage_a.py`)

The stage scripts of the existing pipeline, unchanged, run in a staging tree (`LCS/gap_stage/`) by redirecting their root
paths: cohort (type, year, preprint flag) → P0 content regions → whitelist (P0 success, allowed type, not a preprint) →
P1 citation masking → P1b/P1c non-prose removal → P2 focal sentences with the new matcher `focal_terms_v3.py`.

Keep rule (owner): an article stays only if P2 finds at least one focal sentence in its body text.

`focal_terms_v3.py` keeps every decision of the existing matcher (v2) and adds only hits on tokens v2 did not judge:
spellings of listed names (LLaMA, CT-BERT, ChatGPT4, plurals, checkpoints such as bert-base-uncased,
"language-model"), unlisted model names accepted by the group-4 rule (ClinicalBERT, scGPT, MedGemma …), and BERT with a
glued citation number. Two bugs were found from the outputs and fixed, each followed by a P2 rerun:
1. v3 re-examined tokens v2 had *rejected*, so "glutamyl pyruvic transaminase (GPT)" became a model name and then
   served as evidence for v2's own GPT hits (1,843 articles). Now v3 never touches a token v2 judged, and bare listed
   names never enter as new names.
2. All-capital GPT compounds (CGPT, GPTMS, R-GPTS, HAE-ANGPT) and runs of 15+ capitals (peptides) are refused.

## 3. Stage B — article and hit flags (`gap_stage_b.py`)

Runs on the new corpus and, with the same rules, on the current corpus (to measure cost and find false positives there).

**Per hit, applied first** (a removed hit is listed with its reason in `stage_b/stage_b_rejected_hits.tsv`; a sentence
left without a term is no longer focal):
- *Release date* (owner): a named model cannot be meant before its release year (e.g. ProGen 2020, BERT 2018, Llama
  2023, Gemini 2023, Gemma 2024); general terms (language model, LLM, transformer model) are exempt.
- *Abbreviation definitions*: an article that spells out LLM / GPT only as something else ("lipid-lowering medications
  (LLMs)", "Logic Learning Machine (LLM)", "Grooved Pegboard Test (GPT)") loses its LLM / GPT hits. A definition counts
  only if the initials of its last words spell the abbreviation; never applied in an article that mentions "large
  language model".
- *Sentence patterns*: Progen the supplier, Gemini instruments, Geant4 physics lists (QGSP_BERT_HP), electrical
  transformers, linguistic "language model", GPT the enzyme, ESM-2 in climate-model lists, palm domains, explicit
  LLM other senses (LLM-1xx explosives, lean mass, lipid-lowering, Rulex). No sentence pattern applies when the
  sentence has a language-model cue (prompt, token, embedding, fine-tuned, ChatGPT, GPT-n …).

**Per article** (first that applies):
- **F0 hits rejected**: every hit was removed by the rules above → dropped.
- **F1 AI-use statement only**: every focal sentence is a statement that an AI tool was used for the manuscript, its
  figures or code ("We used ChatGPT for language editing"; also negative statements, "No AI tools were used in the
  production of this work") → dropped (owner: AI-use statements are not analysed).
- **F2 ambiguous only**: all remaining terms are ambiguous (LLM, language model, BERT, ProGen, GPT, Gemini …) and the
  article has no machine-learning context word, or a confound marker and fewer than 5 → manual check (section 4).
- **kept**: otherwise.

The statement rule's wording and the homonym rules were tuned in rounds; after each round a fresh random sample was
checked by hand by the assistant (400 articles in two formal rounds, about 300 further rule decisions), with the
errors fixed and re-tested. The position of a statement in the article (heading above, sections below) was measured as
extra evidence (owner's idea) and found not usable as a veto: journals often put the AI disclosure at the end of Methods.

## 4. Manual check of F2 (owner: option B)

**Readers.** Haiku agents read compact excerpts, not whole articles (a raw article is about 15,000 tokens at the median;
an excerpt about 600). Hard cases went to the assistant (Opus 5.5).

- Agent type `general-purpose`, model pinned to Haiku; the transcripts record `claude-haiku-5-5`. No effort level was
  set, so the model's default was used. Each agent was told to read only the rubric and its two batch files, not to
  search or run code, and to write only its answer file. These limits were instructions, not a technical sandbox.
- **Excerpt** (`gap_f2_excerpts.py`): title, journal, year; the abstract (first 150 words) and keywords from the PMC
  header; every P2 block whose non-statement focal sentences hold a surviving ambiguous hit, the hit marked ⟦…⟧, with
  its section heading (P2 already merges touching ±1-sentence windows; passages are chosen so every asked term is shown,
  and a long passage is cut to ±250-character windows around every mark); the whole-text count of machine-learning
  context words; the article's own expansions of LLM / GPT / BERT. One question per article and term.
- **Rubric**: `F2_READER_RUBRIC.md` (verdicts yes / no / statement / unsure with reason codes).
- **Scoring**: `gap_f2_score.py`.

**Exact prompt, calibration run** (one agent, files `calib_v1_part1/2.txt`):

> You are a careful reader doing a narrow word-sense task. Work only with the files named here; do not search, explore,
> or open any other file, and do not run code.
>
> 1. Read the rubric: …/manual/RUBRIC.md
> 2. Read the batch, part 1: …/manual/calib_v1_part1.txt (69 questions). If the file is long, read it in pieces with
>    offset/limit until you have seen every line.
> 3. Read part 2: …/manual/calib_v1_part2.txt (60 questions), the same way.
> 4. Answer every QUESTION line (129 in total) by the rubric, judging only from the text shown for that article. Use
>    "unsure" when the shown text does not decide it; do not force a choice.
> 5. Write all answers to …/manual/calib_v1_haiku.jsonl — one JSON object per line, exactly in the rubric's output
>    format ({"qid": ..., "verdict": "yes|no|statement|unsure", "reason": <code>, "note": <at most 12 words>}), one line
>    per question, no other text in the file. Write the file with the Write tool (you may write part 1's answers first
>    and then rewrite the file with all 129).
>
> Finish by replying with one line: the number of answers written and how many of each verdict.

**Exact prompt, batches 01–12** (one agent per batch; NN and the question count varied):

> You are a careful reader doing a narrow word-sense task. Work only with the files named here; do not search, explore,
> or open any other file, and do not run code.
> 1. Read the rubric: …/manual/RUBRIC.md
> 2. Read the batch, part 1: …/manual/f2_bNN_part1.txt — if it is long, read it in pieces with offset/limit until you
>    have seen every line.
> 3. Read part 2: …/manual/f2_bNN_part2.txt, the same way.
> 4. Answer every QUESTION line (N in total) by the rubric, judging only from the text shown for that article. Use
>    "unsure" when the shown text does not decide it (including when no marked passage for the term is shown); do not
>    force a choice.
> 5. Write all answers to …/manual/f2_bNN_haiku.jsonl — one JSON object per line, exactly the rubric's output format, one
>    line per question, no other text. Use the Write tool (you may write part 1's answers first, then rewrite the file
>    with all of them).
>
> Finish by replying with one line: the number of answers written and how many of each verdict.

(… = `/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/gap_stage/stage_b/manual`.) Batches 10 and 11 were blocked at the
first launch by the auto-mode classifier and relaunched with the same prompt after the owner's approval.

**Procedure and results.**
1. *Gold set*: 115 articles (129 questions) labelled by the assistant: 32 earlier-judged articles still in F2, 68 new
   F2 articles, 15 positive controls (kept articles whose terms are all ambiguous).
2. *Calibration*: Haiku agreed on all 123 decisive questions; 2 passing mentions answered unsure; both AI-use
   statements and both hard cases matched.
3. *Full run*: 12 batches, 1,352 articles, 1,443 questions; 1,441 answered (2 skipped, decided by the assistant);
   verdicts no 1,145, yes 202, statement 35, unsure 59.
4. *Assistant checks*: spot check of 76 random Haiku decisions (yes and no), 76 correct; every unsure decided by the
   assistant.
5. *Owner rule* (2026-10-21): the focal concept is the **neural** language model. A computational language model whose
   neural nature the text does not show (BCI spellers, speech decoders, n-gram or unspecified surprisal, distributional
   models) is not focal. Of 40 such cases, 2 were kept (Markov vs RNN language models; Whisper).
6. *Outcome* (keep if any term is focal): new corpus keep 185 of 1,325 (drop 1,111 other sense, 29 statement only);
   current corpus keep 34 of 127 (drop 87, 6).

**Cost.** The agents' completion reports gave about 150–163k tokens per run (≈2.1M in total). The per-turn usage
records in the transcripts, which also count the cache re-read on every turn, sum to about 6.9M tokens for the 13 runs,
mostly cache reads. The recorded output counts look too small for runs that each wrote about 120 JSON lines and may be
under-recorded.

| run | model | turns | input | cache write | cache read | output | total |
|---|---|---|---|---|---|---|---|
| calibration | claude-haiku-5-5 | 5 | 10 | 161,375 | 349,415 | 43 | 510,843 |
| batch 01 | claude-haiku-5-5 | 5 | 10 | 161,044 | 355,030 | 29 | 516,113 |
| batch 02 | claude-haiku-5-5 | 5 | 10 | 156,130 | 335,821 | 43 | 492,004 |
| batch 03 | claude-haiku-5-5 | 4 | 8 | 138,663 | 213,883 | 25 | 352,579 |
| batch 04 | claude-haiku-5-5 | 5 | 10 | 154,356 | 355,385 | 49 | 509,800 |
| batch 05 | claude-haiku-5-5 | 5 | 10 | 159,538 | 345,701 | 43 | 505,292 |
| batch 06 | claude-haiku-5-5 | 6 | 12 | 159,933 | 428,221 | 35 | 588,201 |
| batch 07 | claude-haiku-5-5 | 5 | 10 | 156,470 | 361,880 | 44 | 518,404 |
| batch 08 | claude-haiku-5-5 | 8 | 16 | 153,320 | 671,374 | 328 | 825,038 |
| batch 09 | claude-haiku-5-5 | 5 | 10 | 156,516 | 346,228 | 42 | 502,796 |
| batch 10 | claude-haiku-5-5 | 5 | 10 | 150,608 | 337,455 | 44 | 488,117 |
| batch 11 | claude-haiku-5-5 | 5 | 10 | 161,937 | 355,132 | 22 | 517,101 |
| batch 12 | claude-haiku-5-5 | 6 | 12 | 152,893 | 457,847 | 44 | 610,796 |
| **all 13 runs** | | | 138 | 2,022,783 | 4,913,372 | 791 | 6,937,084 |

## 5. Results and lists (`stage_b_lists/`)

| | New corpus | Current corpus |
|---|---|---|
| Kept | **15,563** (`kept_new_corpus_after_stage_ab.txt`) | 33,682 of 34,662 with a focal sentence |
| No focal sentence in the body text (P2) | 82,158 (`dropped_no_focal_sentence_p2_gap.txt`) | (already outside the current corpus) |
| All hits rejected (F0) | 2,639 | 104 |
| F2 other sense (manual check) | 1,111 | 87 |
| AI-use statement only (F1 + F2) | 6,996 | 789 |
| P0 structure / type / preprint | 7,990 / 139 / 1,994 | — |

- `dropped_no_focal_terms_left.csv`: articles that had focal hits but none survived — F0 (with every rejected hit and
  its reason) and F2 other sense (with every term's verdict). Both corpora.
- `dropped_ai_use_statement_only.csv`: articles whose only focal sentences are AI-use statements, with an example.
- The full removal list of the new corpus with reasons is `LCS/gap_stage/gap_removals_stage_ab.csv` (on RDS); the
  current corpus's exclusions are `LCS/article_exclusions_stage_b.csv` (980), not yet applied: they and the 255 rejected
  hits inside kept current articles are to be removed in the incremental run, before M1.
- 2026 articles are kept for now (owner, 2026-10-21; 5,821 of the 15,563); they can be dropped later.

## 6. Sentence-level AI-use statements inside kept articles (2026-10-21, RL-169)

**The error.** The broad statement rule of Stage B (`gap_stage_b.is_statement`) was validated on whole articles: an
article whose only focal sentences are AI-use statements is dropped (F1; 98 of 100 hand-checked flagged articles
correct). In the incremental run (`inc_prepare.py`) the same rule was also applied to single sentences inside *kept*
articles, to remove their statements (owner: statements are not analysed). There it was wrong most of the time: of 40
random sentences it had removed, 4 were statements; 36 were research (survey results such as "LLMs were used for
scientific writing (29/60)", methods such as "We used GPT-4 to extract key findings", recommendations to authors). About
6,200 research sentences had been removed. It surfaced when the merge check found removed sentences that P7 had kept
again through coreference.

**Rule-based repair, measured and rejected.** A strict rule (disclosure formula; negation about the paper; a use word
with an editing task; no research markers — `statement_strict.py`) reached about 83% precision at about 65% recall after
three rounds; the remaining errors were research *about* AI-assisted writing, which uses the same words.

**Manual check (owner decision).** Every candidate checked by a reader, calibrated first.
- Candidates (`stmt_check.py build`): every sentence the broad rule had un-flagged inside a kept article — 6,205 (5,088
  current corpus, 1,117 new). Each item: the article title, the section heading above the sentence (the strongest cue:
  disclosures sit under "Acknowledgements", "Use of AI tools", "Declaration of generative AI" or at the end of Methods),
  the sentence marked ⟦…⟧, and the sentences before and after it (from the P2 block).
- Rubric (`STMT_READER_RUBRIC.md`): **statement** = about how *this* paper's text, figures or code were produced with AI
  tools, or a declaration that they were / were not used (including a sentence that only completes such a declaration);
  **research** = everything else, also AI use inside the research process (coding qualitative data, deduplicating
  records, drafting survey items, study materials), other papers' disclosures, reviewers' comments, recommendations;
  **unsure** when the shown text does not decide.
- Readers: agent type `general-purpose`, model pinned to Haiku (`claude-haiku-5-5` in every transcript), no effort level
  set (default). Instructions limited reading to the rubric and one batch file and writing to one answer file (not a
  technical sandbox).
- Gold set: 70 sentences already judged by the assistant (the two reproducible samples of 40 and 30): 29 statements,
  41 research.

**Exact prompt, calibration** (one agent):

> You are a careful reader doing a narrow classification task. Work only with the files named here; do not search,
> explore, or open any other file, and do not run code.
> 1. Read the rubric: …/inc_run/stmt_check/RUBRIC.md
> 2. Read the batch: …/inc_run/stmt_check/calib.txt — if it is long, read it in pieces with offset/limit until you have
>    seen every line.
> 3. Answer every QUESTION line (70 in total) by the rubric, judging the marked sentence with the help of its section
>    heading, the article title and the sentences around it. Use "unsure" when the shown text does not decide it; do
>    not force a choice.
> 4. Write all answers to …/inc_run/stmt_check/calib_haiku.jsonl — one JSON object per line, exactly the rubric's output
>    format, one line per question, no other text. Use the Write tool.
>
> Finish by replying with one line: the number of answers written and how many of each verdict.

**Exact prompt, batches** (one agent per half-batch of about 250 sentences, `batch_NN_partK.txt`):

> You are a careful reader doing a narrow classification task. Work only with the files named here; do not search,
> explore, or open any other file, and do not run code.
> 1. Read the rubric: …/inc_run/stmt_check/RUBRIC.md
> 2. Read the batch: …/inc_run/stmt_check/batch_NN_partK.txt — it is long: read it in pieces with offset/limit until you
>    have seen every line.
> 3. Answer every QUESTION line by the rubric, judging the marked sentence with the help of its section heading, the
>    article title and the sentences around it. Use "unsure" when the shown text does not decide it; do not force a
>    choice.
> 4. Write all answers to …/inc_run/stmt_check/batch_NN_partK_haiku.jsonl — one JSON object per line, exactly the
>    rubric's output format, one line per question, no other text. Use the Write tool (you may write the answers in
>    several steps, each time rewriting the file with all answers so far).
>
> Finish by replying with one line: the number of answers written and how many of each verdict.

(… = `/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging`.) Batches of 500 were split into halves because a whole batch
(about 100k tokens) would have come close to the reader's context limit with its output; at most 20 agents ran at
once (the concurrency limit), the rest when slots freed. A 4-agent pilot (1,000 sentences) ran before the other 22.

**Procedure and results.**
1. Calibration: 69 of 69 decisive answers agreed with the gold set (29/29 statements, 40/41 research; 1 research
   sentence answered unsure).
2. Full run, 26 agents: 6,135 sentences; statement 1,184, research 4,899, unsure 52; every question answered, no
   format errors.
3. Assistant: every unsure decided (11 statement, 41 research; dividing line: the tool used on this paper's text,
   figures or code → statement; inside the research process → research); spot check of 30 random statement verdicts
   (27 correct; errors: study materials generated under a disclosure formula, and papers *about* drafting with ChatGPT)
   and 30 random research verdicts (29 correct; error: AI-written analysis code, a statement by the rubric); the 4 errors
   corrected; calibration sentences take the gold labels.
4. Final (`stmt_check/final_verdicts.json`, copy `retrieval/stmt_check_verdicts.json`: sentence id → verdict, with the
   sources of each decision): **1,222 statements removed, 4,983 research sentences restored**; F1 article
   decisions unchanged.

**Cost.** Agents' completion reports: about 120k tokens per half-batch run, about 57k for the calibration and each
small batch-13 run; about 3.2M in total. Transcript usage records of the 27 runs: 15.9M tokens (cache read 13.3M, cache
write 2.6M); their output counts (9,730) look too small for runs that each wrote about 250 JSON lines and are probably
under-recorded. Wall time: about 3 minutes per run, most runs in parallel.

**Applied** (`inc_reflag.py`, P7 rerun `p7b`, second G2 round, merge — see `../INCREMENTAL_RUN.md`): restored
sentences got their focal terms back through the same hit filters as all others; genuine statements are removed from
the merged output whatever their P7 source.

## 7. Second statement check: statements the broad rule did not flag (2026-10-22, RL-172)

**Why.** The RL-169 check (§6) read every sentence the broad statement rule had flagged inside kept articles. The item-6
screen of RL-171 showed AI-use statements still present as focal sentences ("No LLM or other AI-assisted tool is listed
as an author") that the rule had not flagged. Owner: identify them; Haiku readers allowed if more cost-effective.

**Rule alone measured and not used.** A tight pattern (use word + editing purpose + this paper's own text, or a
disclosure formula) found 303 candidates; 13 of a random 30 were statements (about 45%): the same words occur in
research sentences ("the BERT model employed in this work", "readability of the text generated by DeepSeek").

**Candidates** (`stmt_check2.py build`): focal sentences of `PG/g2_v3_1/` in articles not excluded
(`g3_inputs_v3_1/scope_exclusions.csv`) and not judged in RL-169, matching any of five patterns — AB (use word followed by
an editing purpose, and a reference to this paper's own text), C (disclosure formulas: "is not listed as an author",
"during the preparation of this work", "declaration of AI"), D (loose: "authorship", "credited as author", "no
generative AI … was used", "not used in the writing"), E ("the authors reviewed", "take full responsibility"): 822
sentences (D 496, AB 311, C 8, E 7). Items exactly as in §6 (title, section heading, sentence, neighbours), four batches
(250, 250, 250, 72), the RL-169 rubric copied unchanged (`stmt_check2/RUBRIC.md`).

**Readers.** Agent type `general-purpose`, model Haiku (`haiku`), no effort level set; the batch prompt of §6 verbatim,
with the paths `…/inc_run/stmt_check2/batch_NN.txt` → `batch_NN_haiku.jsonl` (… = `/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging`).
No new calibration: the rubric and prompt are the ones calibrated in §6 (69/69); every statement verdict was read instead.

**Results.** 822 of 822 answered, no format errors: 140 statement, 680 research, 2 unsure. Assistant: all 140 statement
verdicts and both unsure read, 30 random research verdicts read (30 correct). Changed: 3 statement → research by the
rubric's research-process line (Claude as qualitative coder; GPT-4 to draft guideline sections in a protocol; ChatGPT to
pick search terms for the literature search); unsure → statement (S03.229, "LLMs such as ChatGPT do not currently satisfy
our authorship criteria", followed by "It was used to refine language only") and unsure → research (S01.134). Final
(`stmt_check2/final_verdicts.json`, copy `retrieval/stmt_check2_verdicts.json`): **138 statements** in 135 articles; 91
of these articles have no other focal sentence and are excluded as statement-only (owner's F1 decision), in the other
44 the statement sentences are skipped by G3 (`g3_inputs_v3_1/statement_sentences.csv`).

**Cost.** Agents' completion reports: 121,004, 120,029, 118,993 and 58,291 tokens (418k in all); about 1.5 minutes per
run, all four in parallel.

## 8. Second, independent reading of the three manual checks (2026-10-23, RL-181)

**Why.** Owner: how reliable are the assistant's manual decisions and the gold sets the Haiku readers were calibrated on?
The gold labels had a single annotator (the assistant); Haiku agreement with them shows that the readers follow that
reading, not that it is right. A reader that never sees the assistant's labels re-reads a random sample.

**Reader.** Agent type `general-purpose`, model Sonnet (`sonnet`, not used for any gold set or reading before), no effort
level set; the same rubrics and item blocks the Haiku readers had (copied exactly, renumbered R001 …); the final
verdicts kept in a separate folder the reader is not pointed to. Sampler `retrieval/second_reader_sample.py` (seed
20261023): per list a trial of 5 + 5 and a main sample of 50 + 50, stratified by the final decision (F2: focal / not
focal; statements: statement / research; group-4 names: model / not model), disjoint. Items: `PG/second_reader/`
(corpus text, RDS only); verdicts and keys (ids and labels only): `retrieval/second_reader/`.
Prompt (verbatim apart from list name and count): "You are a careful reader doing a narrow classification task. Work only
with the files named here; do not search, list directories, explore, or open any other file, and do not run code or shell
commands. Judge only from the text shown for each item. … 1. Read the rubric … 2. Read the items … 3. Answer every
QUESTION line by the rubric. Use "unsure" when the shown text does not decide it; do not force a choice. 4. Write all
answers to <list>_main_sonnet.jsonl …". One breach: the F2 reader wrote its answer file with a short Python script run
through the shell instead of the Write tool; its transcript shows the script held the reader's own verdicts inline and
opened no other file, and its only file reads were the rubric and the items, so the answers stay blind.

**Results.** Trial: 29 of 30 agree (one figure caption answered unsure). Main samples:

| List | Decisive answers | Agreement | Cohen's κ | Unsure |
|---|---|---|---|---|
| F2 word sense | 97 | 95 (97.9%) | 0.959 | 3 |
| AI-use statements | 100 | 99 (99.0%) | 0.980 | 0 |
| Group-4 model names | 97 | 95 (97.9%) | 0.959 | 3 |

The 5 disagreements and 6 unsure answers were read by the assistant (cases and reading in RUN_LOG RL-181) and go to the
owner: likely errors of the final labels R061 (F2: "language models" of an EFL narrative, probably models of language use,
not neural) and R040 (group 4: UTR+BERT-derived, inconsistent with the "<model>-derived" mapping); borderline R005, R080
(F2), R054 (RegulatoryGPT, a proposed tool). Rough error rate of the final labels on items drawn evenly from both
decisions: 0.5–1%; rule-of-three upper bound about 3% per list. Cost: agents' reports 54k (trial) and 120k, 65k, 61k
tokens; about 1 minute each, run in parallel.
