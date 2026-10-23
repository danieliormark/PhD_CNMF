# Rolling notes — where to resume

> **Rule (owner, 2026-10-19):** whenever the owner says they want to compact the dialogue, this file is updated
> *before* compacting: the "Latest output" section is replaced by the assistant's latest substantive reply (verbatim),
> and "State" by a short pointer to where work stands and what is waiting for the owner. Then commit and push.
> Older contents are not kept here (git history has them); the durable record stays in the design docs, RUN_LOG and
> CORPUS_STATISTICS.


## State (2026-10-21, after RL-169)

**Current work:** ticket 107 P1 (data quality) — corpus extension and incremental pipeline **done**; next is the G3
production rerun. Records: RUN_LOG RL-155 to RL-169; method `fullscale_pipeline/retrieval/STAGE_AB_METHOD.md`
(Stage A/B rules, F2 and statement manual checks with the exact reader prompts); run plan and results
`fullscale_pipeline/INCREMENTAL_RUN.md`; lists `fullscale_pipeline/retrieval/stage_b_lists/`.

**Done (2026-10-19 to 10-21):**
- Full-text candidate search → downloaded corpus `LCS/pure_text_corpus_gap/` (118,590 files).
- Stage A (original preprocessing on the new files, P0–P2 with the new matcher `focal_terms_v3.py`, two matcher bugs
  found and fixed) and Stage B (release-date and homonym hit rules; F0 all hits rejected, F1 AI-use statement only, F2
  ambiguous only). F2 manual check: 12 Haiku batches + assistant (neural-LM rule: only *neural* language models are
  focal). **New articles kept: 15,563** (`gap_stage/gap_article_list_after_ab.txt`; 5,821 from 2026, kept for now by
  owner decision).
- Current corpus cleaned by the same rules: 980 articles excluded (`LCS/article_exclusions_stage_b.csv`: 789 statement
  only, 104 all hits rejected, 87 F2 other sense) and false hits inside kept articles removed.
- Incremental pipeline (P2 v3 on the current corpus → P6 on CSF → P7 with frozen REF ids → G2 on CSF + local → merge):
  **`PG/g2_v3_1/` — 49,346 articles (33,783 current, 15,563 new), 816,678 sentences, 825,693 units, 12,690 citation
  works; all traceability checks passed** (`check_report.json`).
- Error found and corrected (RL-169): the broad statement rule had removed ~6,200 research sentences inside kept
  articles; all 6,205 checked by Haiku readers + assistant: 1,222 statements removed, 4,983 restored.
- Design decisions recorded: G3 item 11 (sentence-scoped cousins; typed links contains / co-occurs / embedded; no
  parent–parent links) and the ticket-107 addendum (separate nested vs co-occurring relations; frequency-based
  down-weighting of generic material, to be tested).

**Remaining, in order:**
1. **G3 production rerun — DONE (RL-175, 2026-10-22).** CSF check 23664260 passed; array 23664753, 50/50 tasks exit 0,
   0 problems; merged to `PG/g3_v2/`: 41,778 valid articles, 6,953 excluded, 615 without a parent; 804,702 units,
   905,585 parents, 2,082,569 cousins. Inputs: RL-171–RL-174 (lists for the merged corpus, v3 matcher, checkpoint fold,
   group-4 decisions incl. Haiku review and the model/method line, homonyms, statement check).
1b. **G3 final rerun — DONE (RL-176 to RL-182, 2026-10-23).** Model names one atom per model at tier level (vendor names to
   the family), word clean-up, notation units removed; second reading by Sonnet (κ 0.96–0.98, STAGE_AB_METHOD.md §8);
   incline run: `PG/g3_v2/` 41,752 valid, 641 invalid, 6,953 excluded; audit 0 problems. Kept: `PG/g3_v2_rl175/`, `_rl178/`.
   Open, small: ~60 checkpoint settings in reversed or glued forms (base-BERT, Mistral7B); formula fragments with Latin
   variables inside prose (salign, cctx) not measured; lower-case hub spellings (xlm-roberta-base) not matched by the v3
   matcher (upstream). **Next: M1 merge discussion (ticket 107)**; to do there: count semantic sparsity under different
   model-merging levels (design doc addendum 2026-10-22).
   RL-183 (D5 read, open points measured): asides misattach a focal term in about 8–12% of 2,832 aside units; words glued
   across an em dash become junk atoms (ai_the, sam_is …; 1,129 occurrences incl. legitimate en-dash compounds and ranges);
   numbers to be left out of the modelling in M1 (owner).
1c. **G3 RL-185 rerun — DONE (2026-10-23).** G2 dash re-parse (13,050 sentences) merged to `PG/g2_v3_2/` (G3's input
   now); A1 name rules, formula-token rule, lower-case hub spellings (730 mentions), GPT-OSS/J/Neo joined (917). Incline run:
   `PG/g3_v2/` 41,754 valid, 639 invalid, 6,953 excluded; audit 0 problems; kept `PG/g3_v2_rl182/`, `_rl185_try/`.
   CSF check renewed, not yet run on CSF (optional: `sbatch PG/scripts/g3_v2/check_csf_g3_v2.sh`).
   **Open, waiting for the owner:** A3 part 2 — 688 distinct sentences (833 spellings) in 307 valid articles lost upstream
   because the v3 matcher missed lower-case spellings; needs P2 with the extended matcher -> P6 -> P7 (frozen REF ids) ->
   G2 -> merge `g2_v3_3` -> G3. Then B3 (author dedup stats, 2026 decision), B5 housekeeping, M1 discussion.
1d. **G3 RL-186 rerun — DONE (2026-10-24).** G2 positions pass (`PG/g2_v3_2_pos/`, 825,554 of 825,698 units exact):
   provenance per hyperedge; focal names replaced and placed by their own positions (two pre-existing errors fixed);
   contrast words kept (without, versus, than, unlike, in_contrast). `PG/g3_v2/` 41,753 valid, 640 invalid, 6,953
   excluded; audit 0 problems; CSF check renewed and passed on incline. Kept: `PG/g3_v2_rl185/`, `_rl186_first/`,
   `g3_v2_rl186_trial2/` (owner may delete the last two).
   **Next, owner decisions 2026-10-24:** first M1 trial without the temporal extension, on a subset (one discipline or
   one model family, cleaner data). Data look and proposal: design doc addendum 2026-10-24; waiting for the owner on
   field vs family, discipline source, OpenAlex/NLM fetch, size. A3 part 2 (lost sentences) still to do, after the
   whitelist cleanup (B5).
2. Small known item: the open RL-118 leak (citation numbers left as words). ("BERTH" checked in RL-172: a genuine
   model, BERT for hydrology; no exclusion.)
3. D5 count of dash-enclosed asides (approved), on the G3 output.
4. Author-deduplication statistics by year for the new articles (OpenAlex/PubMed author fetch; OpenAlex key from the
   environment only), then the owner's decision on 2026 articles.
5. Re-verify the P1 trial checks; P2 weighting; the chunk12v3 trial matrices (slices 2020/2021).
6. Housekeeping: RUN_LOG has two entries numbered RL-155 (owner to say how to merge); `LCS/article_whitelist.txt` still
   lists the 980 excluded current articles (only P0–P2 read it; the exclusions are applied in `g2_v3_1`).

## Owner decisions (2026-10-20/21, summary; details in RUN_LOG)

- Keep the overnight high-recall candidate list; clean it rather than hunt more 2026 articles; ignore the 591
  morning-only and 161 unfetchable candidates.
- Drop articles with no focal sentence; drop articles whose only focal sentences are AI-use statements (statements are
  not analysed); drop obvious false positives (release date, homonyms, repository venues).
- F2: manual check with Haiku readers for clear cases, the assistant for hard ones; only *neural* language models
  count as focal.
- Keep 2026 articles for now; decide after the author-deduplication statistics.
- Apostrophe re-parse rule applied to new sentences as well as the 876 old ones.
- Traceability: one record per article, no reused ids, hashes recomputed, check report before G3.
- Cousins scoped by sentence; nested and co-occurring ties are different relations; generic material to be
  down-weighted by a tested frequency rule; parent–parent links not needed (traced via sentence and article).
- Sentence-level statements inside kept articles: every candidate checked (Haiku + assistant), not removed by rule.

## Owner input (2026-10-19, verbatim decisions, recorded on request)

- **D4.1:** approved; implement (done in the test script; production run in step 5). Citations with apostrophes:
  checked, names and REF ids kept (RL-150).
- **Failed hard cases:** B06 "Not only did GPT-4 pass the exam, but it also outperformed the residents" — the verb "did"
  is visible and special not-only/but-also rules exist, so the failure is surprising; A12 "GPT-4 as well as BERT was
  evaluated on the exam" — "was evaluated" is a compound predicate. Investigate whether the right post-processing code
  was applied and at which stage the error occurs; fix it if feasible; if not feasible or a trade-off, estimate how
  frequent the error is and report.
- **D2 groups 1–3:** approved ("more elegant to add these model aliases into the list"); CT-BERT → `ct_bert` approved.
- **D2 group 4:** approved with one addition: consider only articles, historical articles and all types of reviews;
  other types are disregarded. Start with it; while it runs, do work that does not depend on it (e.g. the failed hard
  cases; except for the D5 incidence count, new sentences are not needed for that).
- **New papers:** check whether new papers should be added — only those with missed models, not instruments.
- **Incremental run:** a good idea; approved in general, but be systematic: first agree on the other things, then launch.
- **D6:** sentence-centric approach agreed — one parent per clause holding a focal term; each argument holding a focal
  word is a focal hyperedge (several allowed); a focal hyperedge may hold several focal words; a clause with a focal term
  is never a cousin; the rule changes types and removes no words or hyperedges. Document it.
- **D5:** count of dash-enclosed asides approved; the corpus to count on is defined after the hyphenated model names
  are solved.
- **Clitic main verbs** → be/have/will/would: approved ("a great point").
- **Second round (2026-10-19):** G2 apostrophe fix — test on real and hard cases before any re-parse; O08 ("not only
  matched but exceeded human scores") — output acceptable: "exceeded human scores" is a legitimate clause related to the
  parent but not part of it; clinical trials / protocols excluded (a separate genre, rarely included in such corpora), so
  the allowed types are research articles, historical articles and reviews.
- **Third round (2026-10-19):** re-parse of the 876 sentences with an apostrophe parsed as a verb approved for the to-do
  list; the sequence approved in general; new papers: retrieve the ids and the papers following the group-4 proposal,
  check whether they are already in the list, run the preprocessing on them (P0 regions, ..., blocks); existing
  articles also need their block extraction updated where they contain the missed aliases. **Keep rule for new papers
  (owner):** the same rule as originally — an article is kept if at least one focal term (new matcher) is found in its
  plain text (P2: a focal sentence), not only in title/abstract. Also: say where CSF or external APIs are needed.
