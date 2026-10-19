# Rolling notes — where to resume

> **Rule (owner, 2026-10-19):** whenever the owner says they want to compact the dialogue, this file is updated
> *before* compacting: the "Latest output" section is replaced by the assistant's latest substantive reply (verbatim),
> and "State" by a short pointer to where work stands and what is waiting for the owner. Then commit and push.
> Older contents are not kept here (git history has them); the durable record stays in the design docs, RUN_LOG and
> CORPUS_STATISTICS.

## State (2026-10-19, after RL-151)

- **Current work:** ticket 107 P1 (data quality): G3 and focal-term fixes before the trial matrices. Records: design doc
  §8.3 ("P1 owner decisions and follow-ups"), RUN_LOG RL-149/RL-150, G3_POSTPROCESSING.md §6 items 9 (D4.1) and 10 (D6).
- **Sequential plan (in this order):**
  1. D2 groups 1–3: add the spellings of listed terms to `focal_terms.py` (fix the "LLaMA" pattern bug); hyphenated names
     as one focal atom (CT-BERT → `ct_bert`, assistant's proposal, owner approved).
  2. D2 group 4: extend the list with unlisted model names (not tools, not noise), checked for false positives
     (llama the animal, LLM as a biomedical abbreviation); only articles of type research article, historical article
     or any review count (other types disregarded).
  3. New papers: whitelist articles that gain a focal sentence only through missed *models* (not tools), of the
     allowed types; list for the owner before adding.
  4. Incremental run (approved in general; details to agree first): P2 with the new matcher → P6 only on new blocks →
     P7 with existing REF ids frozen → G2 only on new sentences (plus re-parse where G2 joined an unlisted name) → checks
     that every existing id/hash is unchanged.
  5. G3 changes, all in one production rerun: D4.1 (implemented, tested), B06 inverted-auxiliary fix (done; script
     `c8b46c840f74`), D6 sentence-centric parents (decided, to implement).
  6. D5 count of dash-enclosed asides with a focal term (approved), on the corpus as fixed after step 3.
  7. Re-verify the trial P1 checks, then P2 (weighting, chunk12v3 trial matrices).

## Open points for the owner (2026-10-19, after RL-151)

- Curly apostrophe after a plural ("LLMs’ responses") derails spaCy in 880 units (692 articles). Owner: test on real
  and hard cases first (does the fix work; does it introduce new problems) before committing to re-parsing. Tested (RL-152):
  recommended to re-parse only the 876 sentences whose quote mark was parsed as a verb; awaiting owner.
- Group 4 draft rule and exclusion lists (RUN_LOG RL-151) to confirm; the 50 candidate new articles are reviewed after the
  matcher is final.

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
