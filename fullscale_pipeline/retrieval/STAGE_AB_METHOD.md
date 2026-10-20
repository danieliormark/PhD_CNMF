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
