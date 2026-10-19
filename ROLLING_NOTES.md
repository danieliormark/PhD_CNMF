# Rolling notes — where to resume

> **Rule (owner, 2026-10-19):** whenever the owner says they want to compact the dialogue, this file is updated
> *before* compacting: the "Latest output" section is replaced by the assistant's latest substantive reply (verbatim),
> and "State" by a short pointer to where work stands and what is waiting for the owner. Then commit and push.
> Older contents are not kept here (git history has them); the durable record stays in the design docs, RUN_LOG and
> CORPUS_STATISTICS.

## State (2026-10-19, after RL-149)

- **Current work:** ticket 107 P1 (data quality) — design doc §8.3, "P1 owner decisions and follow-ups"; RUN_LOG RL-149;
  G3_POSTPROCESSING.md §6 item 9.
- **Done:** D4.1 + figure-panel legends implemented in `g3_v2/g3_curation_v2.py` (sha256 `43ae81d2c33a`), tested (self-test
  54/54, hard 187/200, shards 0 and 20 compared); not yet run on the corpus. D2 measured corpus-wide (scan scripts
  `corpus_statistics/focal_gap_scan.py`, `focal_gap_classify.py`); D5 traced to spaCy; D6 rule origin traced (no
  recorded reason).
- **Waiting for the owner:** D2 decisions (fix spellings of listed terms + LLaMA bug + Bert/Roberta; hyphenated name as
  one atom or listed part; adopt unlisted model names; incremental rerun route); D6 confirm/revise; D5 whether to
  measure/apply a dash→bracket G2 rule.
- **Next:** implement D2 in `focal_terms.py`, incremental P2→P6→P7→G2 run, one G3 production rerun, re-verify trial P1
  checks; then P2 (weighting, chunk12v3 trial matrices).

## Latest output (assistant, 2026-10-19, verbatim)

All of this is recorded (design doc §8.3, RUN_LOG RL-148; commit `6a14454`, pushed).

## A. Author linking — applied

The current method had not been applied before (only tested). Applied now to the 74 trial bylines without an ID:
**23 linked** to a person whose ID we know (each with a shared institution or shared co-author on other articles; all
look sound — e.g. eight University of Edinburgh authors of one paper, Tsinghua and Mayo Clinic authors), **2
ambiguous** (left unlinked), **49 stay their own person**. The table with every link's evidence is
`corpus_statistics/author_links_trial.csv`.

## D2. Focal terms typed as ordinary words — investigated

177 of 10,160 focal-term occurrences in the trial (1.7%), in 68 articles. Three causes, from every case:
1. **Hyphenated model names containing a focal term:** FARM-BERT, CT-BERT, Med-BERT, Bioclinical-BERT, Sentence-BERT,
   bert-base-uncased, roberta-large. The step that splits words keeps such hyphenated words apart, and the detector of
   focal terms does not list the "BERT" part, so it is typed as an ordinary word.
2. **Other spellings:** "Bert", "bert", "Roberta" — possibly deliberate, to avoid matching people named Bert or Roberta;
   to be checked.
3. **Lists where only some occurrences were detected.**

A consequence beyond the type: an undetected occurrence also gets **no parent of its own** (a sentence whose only
focal term is "CT-BERT" produces no parent).

**Fix — your decision:** treat a hyphenated name containing a focal term, and the other spellings when the article
elsewhere uses the focal term, as focal occurrences ("Bioclinical-BERT" → `bert/C/focal` + `bioclinical`, with its
own parent). This changes G3, so G3 would be rerun (about 5 minutes on CSF).

## D4.1. Numbers and special characters — investigated

**Legitimate, kept:** numbers in running text — sizes, counts, scores ("768 hidden nodes", "F1 score of 0.78"),
already kept by your 2026-10-10 decision ("keep what is plain text"); names containing digits (f1, word2vec, T5, i2b2);
years; accented names (Fréchet).

**Leftovers from formatting, to fix:**

| What | Example | Proposed |
|---|---|---|
| PDF ligatures | `ﬁne`, `classiﬁcation` | **normalise** to "fine", "classification" (real words) |
| words glued to brackets | `lucene(a`, `rnn)-based` | strip the bracket, keep the word |
| formula fragments | `O(n2`, `p(st+1\|s0`, `t′`, `Xα` | drop |
| stray quote marks as atoms | `’` | drop |
| footnote / citation numbers | "PubMed 1 abstracts", "KeyBERT 12, 28" | drop (the known gap recorded in RL-118; roughly 1 in 6 small integers in my sample) |
| a figure-panel caption treated as a sentence | "a, HNRNPA1 locus showing: predicted CAGE expression in K562; …" | drop |

**A finding worth knowing:** the `REFnnnnnn` tokens (citations replaced by placeholders) are **global IDs of cited
works** — the same work has the same token in every article citing it (REF000001 appears in 22 articles). You decided
earlier to keep them ("sources of authority"). So when a sentence names a cited work as part of its text ("REF009941
proposed the pretraining…"), our hyperedges already link articles that cite the same work: 293 occurrences in the
trial. A partial citation signal is already inside the data.

## D5. "Focal term inside a 'that'-clause" — a parser error

The sentence: *"In order to use just_in_time AIG in conjunction with CAT, it is imperative that the item generating
method—in our case language models—reliably produce items that represent a requested construct …"*

The focal term ("language models") belongs to the clause whose verb is "produce". By G3's rule that clause should give
the parent: verb group `reliably produce`, focal hyperedge "the item generating method / language models", sibling
"items that represent a requested construct …", and "it is imperative …" a cousin.

The parser did not build "the item generating method … reliably produce items" as a clause. It left two loose pieces —
the noun phrase with the focal term, and "reliably produce items …" — inside the clause "it is imperative that …". So
G3 found "it is imperative" as the clause governing the focal term: verb group `be`, focal hyperedge = the noun phrase,
sibling = "reliably produce items …". G3 applied its rule correctly to a wrong parse. One case in 30 sentences; its
frequency cannot be measured without checking parses by hand.

## D6. A lone modifier as a cousin — follows G3's rule

G3 removes focal-term words from cousins (G3_POSTPROCESSING §3, kept from the earlier version). The sentence had two
focal occurrences, so two parents. For the second parent, the first parent's clause — "the first stage was fine_tuning
the pretrained language model BERT" — is another clause, so it becomes a cousin; with the focal words ("language
model", "BERT") removed, its non-verb part is just `pretrained`. Expected by the rule, not an error; single-word cousins
of this kind are its side effect.

## D7. Long hyperedges — legitimate except one

68 of 49,867 trial hyperedges have 20 or more words. The largest are plain-text enumerations — lists of models
("BERT, Bioclinical-BERT, PubMedBERT, RoBERTa, and DeBERTa"), of methods, of workflow steps ("text collection
(Reddit), text preprocessing (removal of URLs, lowercasing, …)") — legitimate as you say. The one exception is the
figure-panel caption above (a 40-word hyperedge that is caption text, not prose).

**Decisions needed before P2:** the D2 fix (and G3 rerun), and the D4.1 cleaning list. Both can go into one G3 patch
and rerun, documented like earlier G3 decisions.
