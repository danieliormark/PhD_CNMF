# Stage G3 — postprocessing (curation) of the G2 parse: decisions, open questions, test status

Status (2026-09-30): **design agreed in part; test script built and tested on real data; not yet run on the corpus.**
Test script `PG/scripts/g3_curation_test.py` (RUN_LOG RL-078 to RL-080). The May G3 (`chunk_4h_hpc.py`) is
superseded and must not be rerun (deviation D17 in [`PIPELINE.md`](PIPELINE.md); discussion history in PIPELINE.md
item 29). Input: G2 v3, see [`G2_PARSING.md`](G2_PARSING.md). Path abbreviations as in PIPELINE.md.

## 1. Purpose and target schema

G3 reduces each graphbrain edge that contains a focal term to the structures the matrix builder (M1, the full-scale
analogue of the toy `chunk12.py`) turns into relation matrices:

| Structure | What it holds |
|---|---|
| atom | one word as a lemma with a coarse type: `/C` concept, `/P` predicate (verb), `/M` modifier; a focal term is one atom `<canonical>/C/focal` |
| `dummy_sibling` | the compound verb group of the clause that governs the focal term: its verbs, negations and modals (plus verbs of levels above that hold nothing but verbs, negations or modals) |
| `focal_he` | the argument of that clause that contains the focal term, flattened to a set of atoms |
| `sibling_he` | each other argument of that clause, and the non-verb modifiers of its head, flattened |
| **parent** | `(parent dummy_sibling focal_he sibling_he …)`, one per focal occurrence (identical ones merged) |
| `dummy_cousin` | the verb group of any other clause of the same unit |
| `cousin_he` | the non-verb part of any other clause of the same unit |
| grandparent | the higher-order, predicate-governed hyperedge that holds a parent and its cousins (owner's term, 2026-09-30). It is not written as a structure of its own; in the test script it is the unit's main edge, and each cousin lists the parents it belongs to |

Real example (G2 v3 shard 0, test output, `--modals keep`):

> *ChatGPT is a language model based on the transformer architecture, and after a large amount of text data
> training, ChatGPT has produced human_like language text that allows users to receive information intuitively.*

```
P1 (parent (dummy_sibling be/P/en) (focal_he chatgpt/C/focal) (sibling_he language_model/C/focal))
P3 (parent (dummy_sibling produce/P/en) (focal_he chatgpt/C/focal)
           (sibling_he amount/C/en datum/C/en large/M/en text/C/en training/C/en) (sibling_he human_like/M/en language/C/en text/C/en))
K  (dummy_cousin base/P/en)   (cousin_he architecture/C/en transformer/C/en)   (dummy_cousin allow/P/en)
   (dummy_cousin receive/P/en)   (cousin_he user/C/en)   (cousin_he information/C/en) …   each listing its parents
```

In the toy `chunk12.py` the parent edges become `parent_he`, their children `core_child_he`, the children's atoms
`core_atom`; the cousins become `cousin_he` and their atoms `fringe_atom` (relations `M_Atom_Child`,
`M_Child_Parent`, `M_Fringe_Cousin`, `M_Cousin_Parent`, `M_Cousin_Child` and the three article anchors).

## 2. Which script the new G3 is built on

| Script | Date | Status |
|---|---|---|
| `PG/scripts/chunk_4h_hpc.py` | 2026-05-29 | the May full-scale G3; same code as the toy `postprocessing_4h.py`; crashed on deep edges and curated 359 of 22,601 articles (RL-071) |
| `tensor_data_staging/toy_large/postprocessing_4h.py` | 2026-06-15 | built the toy `corpus_curated.sqlite` (all toy results rest on it) |
| `tensor_data_staging/toy_large/7.5postprocessing_4hbased_correct.py` | 2026-06-16 | corrected version, never applied to any data |

**Decision (owner, 2026-09-30): build G3 on 7.5, not on chunk_4h.** Defects of chunk_4h that 7.5 does not have:
the search for the governing clause starts one level too high (`len(path)-3`), so the simplest clause ("LLMs
outperform X") yields nothing; a fringe pool (`source_fringe`) that chunk12 never reads (chunk12 takes fringe atoms
from the cousins); trailing digits stripped from every atom (`gpt2` → `gpt`, undoing P1's protection of model
names); apposition pruning around the focal term; single-word lemmatisation out of context with `en_core_web_sm`.

## 3. Agreed decisions (all implemented in the test script)

Item letters follow PIPELINE.md item 29.

| # | Decision | Date |
|---|---|---|
| a | **Recursion:** limit raised to 20,000; each unit in its own error block; input read line by line, never through `hgraph.search` (where the May G3 died). The deepest unit of the corpus (depth 101) passes. | 2026-09-29 |
| b | **Input:** G2 v3 (`PG/g2_v3/shards/g2_parsed_NNN.jsonl`, one record per unit with uid, unit hash, sid, hash_final, text, edge, lemma edges, token positions). The original plan named the G2 databases with their `occurrences` attribute; the JSONL has the same content from the same run, keeps one record per unit (the database merges identical edges of an article) and carries the lemma edges per unit. Lemmas come from G2's `en_core_web_trf` lemma edges, so G3 runs no spaCy. | 2026-09-29/30 |
| c | **Focal terms:** found in the unit text by P2's own matcher (`PP/focal_terms.py`, guard rules and article evidence included), mapped to token positions, and every atom of a mention replaced by one canonical atom `<canon>/Cp/focal` (see §4). | 2026-09-30 |
| e | **Connective abbreviations** dropped: e.g., i.e., vs., cf., etc., viz., et al. (Sentence cuts at abbreviation periods are handled in G2.) | 2026-09-29 |
| f | **Atoms without letters** dropped (numbers, signs, brackets, formula debris); "%" becomes the separate atom `percent/C/en`. | 2026-09-30 |
| g1 | **Auxiliaries:** be, have, do dropped only when the parser types them as auxiliaries (`Mv`); kept as main verbs (`be/P`: "An LLM **is** a type of AI"). The stop list is not applied to main-verb be/have/do (the RDS list holds "been" and "am", the toy list "be", which silently deleted main verbs before). | 2026-09-29/30 |
| g2 | **Modal verbs:** an atom of the compound verb group, in `dummy_sibling` and `dummy_cousin` alike: "should be approached" → `(dummy_sibling approach/P/en should/M/en)`. A modal is type `Mm`, or a modal word the parser typed as another modifier ("can/M", "ca" from "can't"). Verbs that express modality lexically (allow, enable) are ordinary predicates. A modal of a clause embedded in an argument stays with that clause's verb inside the argument (see open question 2). Options `--modals tag` and `drop` remain only for comparison. | 2026-09-30 |
| h | **Negation** (not, no, never, n't, nor) kept whenever it is a modifier, including "no" as a determiner ("no difference", dropped by 7.5 and chunk_4h). | 2026-09-29/30 |
| — | **Nouns used as modifiers** (type `Cm`: "*cancer* research", "*human* evaluation") kept; 7.5 and chunk_4h dropped them (about 1,000 content atoms per 200 articles). | 2026-09-30 |
| — | **Hyphenated words** are joined in G2 (focal parts kept apart); nothing to do in G3. | 2026-09-29 |
| — | **No fringe pool; no digit stripping** (from the 7.5 base). | 2026-09-30 |
| — | **Provenance:** see §5. Tracing to the sentence is sufficient; tracing each atom to an individual token is not needed for now (owner). | 2026-09-30 |

Kept from 7.5 without change: determiners, pronouns, prepositions/triggers, conjunctions and builders dropped by
type (only types `Cc`, `Cp`, `Ca`, `Cm`, `M`, `P` survive); phrasal verbs fused with their particle ("carry_out");
focal atoms left out of the cousins; each structure is a set of atoms (a word twice in one structure counts once).

## 4. Focal-term canonical atoms

Canonical form: lower case, separators → "_", a final ".0" dropped, a glued version split ("gpt4o" → `gpt_4o`).

- **Merged:** plural and singular; spelling variants (Chat GPT, Chat-GPT, ChatGPT → `chatgpt`; GPT4, GPT 4, GPT-4,
  GPT-4.0 → `gpt_4`); abbreviation and expansion (LLM, LLMs, large language model(s) → `llm`; BERT, Bidirectional
  encoder representations from transformers → `bert`); brand prefixes (Google Gemini → `gemini`, Mistral AI →
  `mistral`); **ChatGPT-<version> into GPT-<version>** (ChatGPT-4 → `gpt_4`, ChatGPT-3.5 → `gpt_3_5`; bare ChatGPT
  stays `chatgpt`); **model sizes and snapshot dates dropped** (Mistral-7B → `mistral`, gpt-4o-2024 → `gpt_4o`,
  GPT-4-0613 → `gpt_4`) — owner decisions 2026-09-30.
- **Kept apart:** `llm`, `language_model` and `transformer_model`; every model version (`gpt_4`, `gpt_4o`,
  `gpt_3_5`, `llama_3_1`); the bare family name (`gpt`, `llama`); ESM-1b keeps its "1b" (part of the name).
- A token is extended beyond the matched name only by a version with a one-digit major (4, 3.5, 4.1) or a size, so
  "GPT-44" (GPT-4 plus a glued citation 4) reads as `gpt`.
- Shard 0 (694 articles): 19,997 mentions, 191 surface forms → 91 canonical atoms; the test report lists every
  surface form with its canonical atom and count, for review.

## 5. Provenance

- Ids and hashes follow G2's unit-hash form `sha1("<id>|<edge>")[:12]`: parent `<uid>.P<k>`; its children
  `<uid>.P<k>.D` (dummy_sibling), `.F` (focal_he), `.S<j>` (sibling_he); cousins `<uid>.K<j>` with the list of
  parents they belong to. With the unit id this gives: atom → child → parent (and cousin → parents) → unit (uid,
  unit hash) → sentence (sid, hash_final) → article (pmcid). Every id and hash is recomputed and checked by the script.
- Every output atom lists the source atoms, words and token positions it comes from (for a focal atom: the tokens of
  the mention; for a fused phrasal verb: the verb and its particle).
- **Attribution of atoms to structures is exact**: it follows the edge structure. A word used twice in a sentence
  appears in each structure whose branch holds it. Only the token index is ambiguous when the same word occurs more
  than once in a unit (10.7% of atom records list several positions); this does not affect the matrices.
- The one place positions decide attribution is focal-term replacement when the same word also occurs outside the
  mention: it reached a non-focal occurrence in 3 of 19,971 mentions.
- For M1: the database written with `--db` uses chunk12's id form `pmcid::uid::hash`. **chunk12 keeps only the last
  parent per sentence id** (`hash_to_parent[sent_hash] = parent_idx`); M1 must use each cousin's parent list instead.

## 6. Open questions (owner decisions needed before the production run)

1. **Passive voice.** Dropping auxiliary "be" loses the difference between "should approach" and "should be
   approached". The parser marks voice without it: in 3,379 "be + verb" constructions, 3,010 (89%) give the main verb a
   passive-subject role. Options: (i) a marker atom in the verb group, e.g. `passive/M/en`; (ii) keep "be" only in
   passives; (iii) record the focal term's role in its clause (subject, object, passive subject, agent), which also
   answers who acts on whom, since `focal_he` does not say whether the focal term is subject or object. A related
   parser error: a copula "be" is occasionally typed `Mv` and dropped ("cannot be absorbable"); frequency not measured.
2. **Clauses embedded in arguments.** The periphery splits every clause into `dummy_cousin` and `cousin_he`, but
   the parent's own arguments are flattened whole, so a relative, infinitival or participial clause inside them keeps
   its verbs in `focal_he` / `sibling_he`: 11% of `focal_he` and 28% of `sibling_he` contain a verb (shard 0). Example:
   "features of ChatGPT that a novice author can use" → `(sibling_he academic/M/en author/C/en can/M/en novice/C/en
   use/P/en writing/C/en)`. Should such clauses be split out into their own verb group and hyperedge?
3. **Lexical modality verbs** (allow, enable, permit, help): now separate verb groups ("allows users to receive" →
   `(dummy_cousin allow)`, `(dummy_cousin receive)`). Join them with their complement verb? (raised, not decided)
4. **Stop list** (deferred by the owner). Base: the RDS list `PG/nltk_abridged_stopwords_list.txt` = NLTK's 198
   English stop words minus 77 (negations, be/have/do and modal forms, more/most/few/same/both/each,
   above/below/against/under/through/until/before, we/our/ours/she) plus also/whilst; the toy list differs only in be
   (toy) / also (RDS). Candidates to add, by frequency among kept atoms (700 articles): our/ours/us (778), discourse
   adverbs (however 391, therefore 161, additionally 140, respectively 125, furthermore 124, moreover, thus, hence),
   well (444, mostly "as well as"), number words (two 342, three 294, one; ordinals undecided). Borderline: only.
5. **Leftover phrasal particles** ("on/M", 220 in 700 articles): particles are exempt from the stop list so that they
   can be fused with a verb; proposal: exempt them only when fused.
6. **"LLM" homonym guard in P2** (deferred; upstream of G2): "LLM" is a PLAIN focal term with no guard; PMC8815195
   uses it for lipid-lowering medication ("not on LLM (89.5%)") and passed P2.
7. **Focal terms with no governing verb:** 280 of 14,225 units (2%) contain a focal term but yield no parent (no verb
   above the term: title-like fragments, lists). Accept, or give them a structure of their own?

## 7. Test script

```
python g3_curation_test.py --selftest                                           # 54 cases
python g3_curation_test.py --shard 0 --limit 200 --outdir DIR [--modals keep|tag|drop] [--db] [--show N]
```

Writes to `--outdir` (refuses to overwrite): `g3_test_NNN.jsonl` (one record per unit: pmcid, sid, hash_final,
uid, unit hash, text, focal mentions with surface form, canonical atom and positions, parents with children and
atom sources, cousins with parents and atom sources), `g3_errors_NNN.jsonl`, `g3_report_NNN.json` (counts, every
focal surface form → canonical atom, dropped atoms by reason and word, problems) and with `--db` a graphbrain
database in chunk12's form. Checks before exit (exit code 1 on any problem): input unit hashes, every id and hash,
children rebuild their parent, every cousin has parents that exist, every atom traced to a token position, and the
database read back in chunk12's way. Environment as G2 (`tensor_env`, `PYTHONPATH=$HOME/np1_for_spacy`).

Results with the current version (sha256 `2a5e01342bf4`), shard 0, all 694 articles: 14,225 units, 0 errors,
0 problems; 13,716 units with a parent, 16,983 parents, 56,989 cousins; 287,524 atoms kept, 187,125 dropped (stop
list 68,529, no letters 47,321, determiners 31,943, other types 24,575, auxiliaries 7,386, pronouns 3,164,
connectives 1,254, focal atoms in cousins 2,953); lemma missing for 115 atoms; about 0.012 s per unit (the whole
corpus in about 2.5 h on one core, minutes across 50 shards). Earlier runs: shard 25 (300 articles) and the deepest
unit of the corpus, 0 errors and 0 problems (RL-078).

## 8. Before the production run

1. Owner decisions on §6 (at least 1, 2 and 4).
2. Production script from the test script: output folder `PG/g3_v1/` (not `PG/postprocessed_output/`), one task per
   G2 shard, a SLURM wrapper, a CSF test, and a separate checked merge (as for G2).
3. The M1 input contract (JSONL and/or database, cousin–parent links as in §5).

## 9. Files

| File | sha256 (12) |
|---|---|
| `PG/scripts/g3_curation_test.py` | 2a5e01342bf4 |
| `PG/scripts/chunk_4h_hpc.py` (superseded, May) | dfc6cc5e9662 |
| `PG/nltk_abridged_stopwords_list.txt` (RDS stop list) | 2b6c7d9fdae9 |
| `tensor_data_staging/nltk_abridged_stopwords_list.txt` (toy stop list) | f5893e962fcd |
