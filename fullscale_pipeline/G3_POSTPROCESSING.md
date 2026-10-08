# Stage G3 — postprocessing (curation) of the G2 parse: decisions, open questions, test status

Status (2026-09-30): **design agreed in part; test script built and tested on real data; not yet run on the corpus.**
*[2026-10-08: §6 items 1–7 decided; items 1, 3, 4 and 5 not yet implemented in the test script; items 6 and 7
are built (`g3_scope_exclusions/`, `g3_llm_output/`) and G3 must read both.]*
Test script `PG/scripts/g3_curation_test.py` (RUN_LOG RL-078 to RL-080). The May G3 (`chunk_4h_hpc.py`) is
superseded and must not be rerun (deviation D17 in [`PIPELINE.md`](PIPELINE.md); discussion history in PIPELINE.md
item 29). Input: G2 v3, see [`G2_PARSING.md`](G2_PARSING.md). Path abbreviations as in PIPELINE.md.

## 1. Purpose and target schema

G3 reduces each graphbrain edge that contains a focal term to the structures the matrix builder (M1, the full-scale
analogue of the toy `chunk12.py`) turns into relation matrices:

| Structure | What it holds |
|---|---|
| atom | one word as a lemma with a coarse type: `/C` concept, `/P` predicate (verb), `/M` modifier; a focal term is one atom `<canonical>/C/focal` |
| `dummy_sibling` | the compound verb group of the clause that governs the focal term: its verbs, negations and modals (plus verbs of levels above that hold nothing but verbs, negations or modals); since 2026-10-08 also a lexical modal verb together with its complement verb(s), §6 item 3 (decided, not yet implemented) |
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
| g2 | **Modal verbs:** an atom of the compound verb group, in `dummy_sibling` and `dummy_cousin` alike: "should be approached" → `(dummy_sibling approach/P/en should/M/en)`. A modal is type `Mm`, or a modal word the parser typed as another modifier ("can/M", "ca" from "can't"). Verbs that express modality lexically (allow, enable) are ordinary predicates *[superseded 2026-10-08 by §6 item 3: they now form one verb group with their complement verb]*. A modal of a clause embedded in an argument stays with that clause's verb inside the argument (see open question 2). Options `--modals tag` and `drop` remain only for comparison. | 2026-09-30 |
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

1. **Passive voice — DECIDED 2026-10-07 (owner): option (ii), keep auxiliary "be" in passive constructions.**
   Dropping auxiliary "be" loses the difference between "a model should approach this problem" and "a model should
   be approached" — the owner's own example, and the reason given: the two mean different things (who acts on
   whom), so collapsing them is not an acceptable simplification. Detection reuses the parser's own, already-measured
   signal: of 3,379 "be + verb" constructions, 3,010 (89%) give the main verb a passive-subject role — that marking,
   not a separate heuristic, is what "passive" means here. **Rule:** when an auxiliary "be" (type `Mv`) sits in a
   clause whose main verb carries the passive-subject role, keep it in the verb group, exactly as modals already are
   (g2); when it does not (the other ~11%, e.g. ordinary progressive "is approaching"), continue to drop it as g1
   already does. Options (i) (a `passive/M/en` marker atom) and (iii) (recording the focal term's full clause role:
   subject, object, passive subject, agent) are not adopted for this question — (iii) remains open separately as a
   richer, not-yet-decided layer (would also answer who acts on whom for *active* constructions, which keeping "be"
   does not).

   **Left open by this decision, not solved by it: the copula-mistyping failure mode.** A copula "be" (the already-kept
   main-verb case, g1 — "an LLM **is** a type of AI") is sometimes mistyped by the parser as auxiliary `Mv` and dropped
   ("cannot be **absorbable**"). A mistyped copula's clause has no other main verb to carry a passive-subject role, so
   it will not pass this rule's keep-condition either — the new rule does not rescue it, and does not make it worse.
   Frequency still not measured. Before implementing, measure how often a `Mv`-typed "be" has no passive-subject-marked
   main verb in its own clause (the mistyped-copula signature) versus a genuine non-passive auxiliary use, so the
   production script's behaviour on that residual case is known rather than assumed.
2. **Clauses embedded in arguments — DECIDED 2026-10-07 (owner): leave flattened (status quo).** The periphery
   splits every clause into `dummy_cousin` and `cousin_he`, but the parent's own arguments are flattened whole, so a
   relative, infinitival or participial clause inside them keeps its verbs in `focal_he` / `sibling_he`: 11% of
   `focal_he` and 28% of `sibling_he` contain a verb (shard 0). Example (shard 0, PMC10788737, re-verified 2026-10-07
   against the current test script, corrected from an earlier truncated quote): *"This review paper aims to explore
   some features of ChatGPT that a novice author can use for academic writing."* → `(parent (dummy_sibling
   explore/P/en) (focal_he chatgpt/C/focal feature/C/en) (sibling_he academic/M/en author/C/en can/M/en novice/C/en
   use/P/en writing/C/en))`. "Features of ChatGPT" (the parent's own argument) is correctly `focal_he`; the embedded
   relative clause "that a novice author can use for academic writing" is flattened into `sibling_he`'s bag, where
   its own subject ("author"), modal ("can") and verb ("use") sit at the same status as every other atom in the bag.

   **Reasons given (owner):** (1) a cousin hyperedge is, by this schema's own definition, *not* part of any parent —
   cousins hold whatever other clause the unit contains, structurally separate from a parent's arguments (§1).
   Splitting an embedded clause out as a cousin would misrepresent it as something it is not: it genuinely is part
   of the parent's own argument, not a sibling structure beside it. (2) The finer partition buys little
   interpretively — splitting `sibling_he`'s bag into "a novice author can use" and "academic writing" barely
   changes what can be read off it; the words were already held together as one argument-level bag, so dividing
   that bag further is a difference of degree, not of substance, for this case.

   **Accepted cost, not hidden by this decision:** the embedded clause's own internal subject-verb relation (which
   word acts on which) stays unrecovered from the flattened bag, same as before. No code change for this question.
3. **Lexical modality verbs — DECIDED 2026-10-07/08 (owner): a lexical modal verb and its complement verb(s) form one
   verb group**, under the rule and guards below. Not yet implemented in `g3_curation_test.py`; the rule is implemented
   and tested in the stand-alone script `fullscale_pipeline/g3_modal_check/modal_rules.py` (RUN_LOG RL-082).
   Originally raised as: allow, enable, permit, help are separate verb groups ("allows users to receive" →
   `(dummy_cousin allow)`, `(dummy_cousin receive)`); join them with their complement verb?

   **Reason (owner).** "allow demonstrate" is qualitatively different from both "allow" and "demonstrate"; sparsity
   must not be reduced at the cost of accuracy. **Before the decision** the same construction came out in three
   ways, depending on details of the parse (shard 0, 300 articles): 22 parents already merged (when nothing but a
   dropped pronoun sat beside the lexical verb, "This allows BERT to excel" → `(dummy_sibling allow excel)`), 56
   parents with the complement verb inside a sibling bag, 116 separate `dummy_cousin`s.

   **Relation to decision 2.** A narrow exception: the complement verb leaves the argument bag. It stays inside the
   parent (it moves into the verb group), so the reason for decision 2 (a cousin is not part of a parent) is
   untouched; a lexical modal verb without its complement says almost nothing ("enable LLMs").

   **Rule.** The trigger and its complement verb(s) form one verb group, in `dummy_sibling` and `dummy_cousin` alike.
   The arguments of both clauses become arguments of the merged clause, including the causer and the enabled entity
   (option c, chosen over leaving the complement verb's clause as the parent, a, or moving only the enabled entity
   into it, b: under a and b, 15 of 31 parents with the focal term inside the complement lost the modal verb, and
   whether the causer appeared depended on which of two parse shapes the parser chose). **Participle guard:** when
   the trigger is an -ing participial adjunct ("These domains are well represented in the training data, allowing
   LLMs to provide…"), no causer is taken from the parse; the parser then gives the whole preceding clause, which
   is already in the cousins.

   | Group | Triggers | Complement |
   |---|---|---|
   | enabling, causing | allow, enable, permit, help, let, force, empower | (object +) infinitive, with or without "to" |
   | directives | ask, prompt, instruct | object + to-infinitive |
   | semi-modals | need, have/has/had to, be able / unable to, fail to, tend to | to-infinitive |
   | gerund complements | help **in** V-ing; allow / enable / permit + V-ing | gerund |

   **Not included: require.** 19 of its 24 "require … to V" cases were purpose clauses ("deploying LLMs requires
   careful monitoring to avoid…", "the manual effort required to analyse…", "further research is required to
   determine…"); complement and purpose readings share one parse in active, passive and participial uses alike.
   Other directive verbs seen in the corpus (guide, invite) were not tested.

   **"be able / unable to":** the copula is dropped and `able` / `unable` is kept as a modal-like atom, as for "can":
   "ChatGPT was able to offer…" → `(dummy_sibling able offer)`.

   **Guards** (each can be switched off in the test script):
   - passive infinitives count ("allowed the plugin functionalities to be tailored"); a complement wrapped in a
     modifier or a coordination is unwrapped;
   - **word order:** the complement verb must follow the trigger. Removes fronted purpose clauses: "To test this, we
     prompted GPT-3.5 to advise…" → `prompt advise`, not `prompt test`;
   - **anchor and positional fallback** (semi-modals): the verb right after "to" (or "to be"), if the parser typed it
     as a predicate, has priority over an infinitive found elsewhere in the parse, and is used when the parse
     attaches no complement at all ("ChatGPT was able to score…" parsed as `(was chatgpt able)`);
   - **need:** no merge for passive or participial "needed" ("further research is needed to explore…", "the
     resources needed to train LLMs…": purpose), nor when a noun phrase stands between "need" and "to" ("we need
     more data to train the model");
   - **let:** no merge for hortative or imperative "let" (followed by *us* or *'s*, or the first word of the unit:
     "Let us consider…", "Let's think step by step"), for definitions ("let X denote / be / represent / consist", also
     after an opening phrase: "Formally, let H be the dimension…") or for "let me / us know";
   - **gerunds:** help in V-ing takes its verb only from the "in" phrase ("Improving performance may help in
     overcoming…" → `help overcoming`); for allow / enable / permit the gerund must directly follow the trigger
     (removes participial adjuncts: "…, thereby allowing", "hence improving"), and there is no gerund reading after a
     passive trigger ("was enabled using RAG": a means adjunct);
   - **precedence:** when a token has an infinitive and a gerund reading, the gerund reading is tried first ("Bard
     can help in preparing articles, summarize the evidence and provide ideas" → `help preparing`, not the
     coordinated main verbs).

   **Evidence** (labels by the assistant from reading each sentence; a complement and a purpose clause are told apart
   by whether "in order to" fits without changing the meaning). Shard samples are random draws from G2 v3:

   | Group | Sample | Merges | Correct | Missed |
   |---|---|---|---|---|
   | allow, enable, permit, help, let | shard 0, 300 articles, 247 tokens (first rule, before the later guards) | 125 | 121 real constructions (114 complete; 6 caught only the first of several coordinated verbs; 1 wrong verb); the 4 non-modal merges were 3 hortative "let" (now guarded) and 1 "allows for … to formulate" | 20 (14 parser errors) |
   | force | 60, whole corpus | 50 | 50 (1 with extra verbs from a relative clause) | 2 |
   | empower | 60, whole corpus | 53 | 52 | 4 |
   | instruct | 60, whole corpus | 54 | 54 (also correctly unmerged as part of a model name, "Mistral-7B Instruct") | 2 |
   | ask / prompt | shard 0, 300 articles | 20 / 16 | 20 / 16 (one wrong verb each from a fronted purpose clause, removed by the word-order guard) | 0 / 0 |
   | need | shards 20, 30; 40 tokens, final guards | 14 | 14 | 1 |
   | have to | three samples | 55 | 51 obligation, 4 possession (accepted, below) | — |
   | fail to / tend to | shards 20, 30; 40 tokens each | 40 / 38 | 40 / 38 | 0 / 2 (broken text) |
   | be able / unable to | shards 20, 30; 40 tokens | 40 | 39 (1 wrong verb) | 0 |
   | help in V-ing | 60, whole corpus | 54 | 54 | 2 |
   | allow / enable / permit + V-ing | 80, whole corpus, final rule | 23 | 23 | 11 (unchanged from before) |

   Guard effects, each read case by case: **word order** (shards 1–4, 1,819 merges) changed 65 decisions: 23 false
   merges removed, 33 verbs corrected, 9 without effect, none harmed; **positional fallback** recovered 34 of 34
   correct constructions on shard 0 and 19 of 19 on shards 20 and 30 (5 took only the first of several coordinated
   verbs); **anchor** (whole corpus) changed 12, 10 improved, 2 neutral; **need guard** blocked 8 of 8 purpose
   clauses in the fresh sample (1 active "DeepSeek also needed to provide…" wrongly blocked, the parser having marked
   it as a participle); **need-object guard** (whole corpus, acts on 149 of 5,449 "need" candidates) 42 of 43 sampled
   blocks correct after it was narrowed to an intervening noun phrase (a first version wrongly blocked 7 "need to V"
   with a spurious parser object); **let guard** (whole corpus, 437 "let") 38 of 40 sampled blocks correct ("These
   observations let us hypothesize" wrongly blocked, "Let ChatGPT give you a starting point" debatable), its
   definitional extension 25 of 26 correct; **adjacency guard** blocked 7 wrong verbs in 80 V-ing tokens; **passive
   trigger** 2 of 2 correct; **gerund-first precedence** 5 of 5 correct (whole corpus).

   **Interaction of the guards.** Every decision was recomputed with each guard switched off in turn: in 803 sampled
   candidates no decision depended on two guards; in the whole-corpus "let" sweep two did, both correct with the two
   guards agreeing ("To formalize the problem, let Xi represent…"). 40 constructed hard cases, each combining two or
   more guard situations, with expected results fixed before running (`g3_modal_check/hard_cases.json`, parsed with
   graphbrain as G2 does): 30 of 40 at first, 33 of 40 with the final rule. The one real interaction found, the
   "be able" rule taking a later infinitive of another clause while the word-order guard kept it ("To be able to
   answer, the model needs to retrieve…"), led to the anchor. When a guard errs, the result is a missing merge, i.e.
   the state before this decision, not a changed meaning.

   **Accepted residual errors.**
   - **have to with a fronted object** is merged as obligation: "the potential [that] LLMs have to transform
     medicine" → `have transform` (4 of 55 have-to merges; nouns potential ×2, facts, opportunities). Owner's choice
     2026-10-08, option a (accept, for simplicity), over a lexical guard on such nouns (would have caught all 4 in the
     samples, at the risk of skipping obligations like "the data that the model had to process"), dropping "have
     to" (losing about 93% correct merges) and skipping relative clauses (losing many correct obligations such as
     "challenges that have to be addressed").
   - **noun object + purpose infinitive** read as a complement: "users can enable logging to monitor the system",
     "LLMs allow fine_tuning of models to improve performance" (about 1 in 100 allow merges; the ambiguity that
     excluded require).
   - **coordinated main verbs** read as complements: "LLMs can empower patients and clinicians, eliminate barriers…"
     (1 of 53 empower merges).
   - **coordination:** a few percent of merges catch only the first of several coordinated complement verbs; the
     others stay in the argument bag, as before.
   - **misses**, unchanged from before: mostly parser errors ("help" typed as a noun in "can help generate", a
     complement attached elsewhere), and about 30% of the V-ing pattern.
4. **Stop list — DECIDED 2026-10-08 (owner), across several rounds.** Base unchanged: the RDS list
   `PG/nltk_abridged_stopwords_list.txt` = NLTK's 198 English stop words minus 77 (negations, be/have/do and modal
   forms, more/most/few/same/both/each, above/below/against/under/through/until/before, we/our/ours/she) plus
   also/whilst. On top of it, the decisions below either drop a word outright, or let it join the verb group it
   belongs to (the same place modals already go, §3 g2) instead of being either kept as a separate hyperedge or
   dropped as a bare stop word. None of this is yet implemented in `g3_curation_test.py`; it is implemented and
   tested in `g3_word_lists/` and `diagnostics/g3_stoplist/` (RUN_LOG RL-083 to RL-086).

   - **"our"/"us": kept, not added.** Owner's reason: distinguishes the authors' own models or approaches from
     others', which the analysis should not discard.
   - **Numbers.** Numeric forms ("2", "2024") were already dropped by decision f (no letters). Spelled-out numbers
     are not added: cardinals carry study-design information ("two models", "three groups"); ordinals split between
     content ("the first model of this kind") and a discourse use ("First, …"), handled by the opener rule below.
     *[Correction 2026-10-08, found while testing item 5 (RL-090): spelled-out numbers do not all survive today.
     As a modifier they are typed `M#` and kept ("two models" → `two/M`); as the head of a noun phrase they are typed
     `C#`, which is not in `KEEP_TYPES`, and are dropped by type ("the first **one**", "**one** of the largest",
     "**twelve** out of 20"): 257 occurrences in shard 0. The same type filter drops 476 alphanumeric labels typed
     `C#` ("Figure 8A", "7e-6"), which is harmless. Flagged for the owner after items 5–7, with the comparison loss
     described under item 5.]*
   - **"only" — a guarded verb-group member, like negation**, not a stop word and not always dropped. "Only LLMs can
     generate…" keeps "only" in `focal_he`; "LLMs can only generate…" puts it in the verb group with "can" and
     "generate". The guard: "only" joins the verb group only when a predicate follows it, directly or after one
     adverb, and never after "if" ("if only because…"). Without the guard, 5 of 40 sampled verb-group placements were
     wrong ("only four qualified", "only 20 amino acids"); the guard removed exactly those 21 wrong placements
     corpus-wide and no correct one. 14 constructed hard cases distinguishing the two positions all pass.
   - **Negation, fixed to stay with its predicate.** `g3_curation_test.py` already keeps negation in the verb group
     when building a parent (§3 h), but not when building the periphery (`sweep`), which used a narrower test
     (predicate or modal only) than `extract` (predicate, modal or negation). "The results will **not** be known"
     gave `(dummy_cousin know will)` + a separate `(cousin_he not)`; the fix makes `sweep` use the same test as
     `extract`. Self-test 54/54; on shards 0, 20 and 30 (31,000 units) no parent changed, and standalone negation
     hyperedges fell from 771 to 43 (the rest elliptical, "but not others", with no verb to join). 9 of 10
     constructed hard cases pass; the 10th keeps its correct meaning regardless (a case of decision 2, an embedded
     clause).
   - **"not only" / "just" / "merely" / "simply" / "solely", and "but also": fused into one non-negating atom each,
     `not_only` / `but_also`, not treated as negation.** Detected deterministically: "not" or "n't" directly
     followed by one of the five words; "also" with "but" standing at most three tokens before it, with only
     auxiliaries, modals, be/have/do forms or pronouns in between and no punctuation ("but also", "but can also",
     "but it also"). Attributed to the clause's verb group or argument exactly as any other atom, reusing the same
     predicate-finding logic (one parser quirk needed a fix: when the parser attaches "also" to the conjunction
     itself rather than to either clause, it is moved into the predicate of the clause after "but"). Reason: the
     polarity of "not only … but also …" depends on the words it scopes over, not on the construction itself
     ("not only **incapable**" stays negative; "not only **cheaper**" does not), so no polarity list is needed, and
     the construction must not be read as a plain negation of its first clause. Self-test 54/54; sampled corpus
     placements (238 cases across shards 0 and 20, split 130 verb group / 101 argument / 7 alone): about 29 of 32
     correct; 7 of 8 and then 7 of 8 constructed hard cases pass (the two failures are a parser reading with no
     predicate path, common to every version tested). **Flagged, not solved now, to revisit once all of §6 is
     decided:** once correctly fused and attributed, "not only"/"but also" read substantively as "and" (owner);
     whether and at which stage of the rule sequence to drop them without disturbing how anything else was attached
     is left open.
   - **Core discourse connectives, and subordinators found to cause the same problem: dropped everywhere**, not
     only as stop words but from the kept-adverb rule below too. List: however, therefore, additionally,
     furthermore, moreover, thus, hence, consequently, nevertheless, nonetheless, accordingly, conversely,
     meanwhile, likewise, indeed, respectively, although, thereby, since, whereas, though, and "even though"/"even
     if". The subordinators were added after the parser typed them as plain adverbs and the rule below would
     otherwise have attached them to a verb group ("since our input will be…" → `(be since will)`). Owner's
     reason for not keeping any of these (2026-10-08): synonymous connectives would create distinct hyperedges for
     the same relation ("although" vs "despite", "since" vs "because"), imposing a sparsity the data does not
     really have, since the model cannot see that the words mean the same thing; dropping removes the problem for
     connectives entirely (prepositions such as "despite" are already dropped by type).
   - **"Overall" / "finally" / "notably" / "similarly": dropped only as sentence openers.** Rule: the word is the
     first word of its unit, or follows ";" or ":", and is directly followed by a comma. 60 labelled occurrences:
     every connective use caught ("Finally, …" 14/14, "Notably, …" 14/14, "Similarly, …" 11/11), no content use lost
     ("the **overall** accuracy", "performed **similarly** to"). A plain word-type test cannot tell the two apart,
     and dropping the word wherever it stands alone would also drop "performed similarly".
   - **A matching opener-drop list** (discourse and sequence words acting as openers only): finally, first, second,
     third, fourth, fifth, firstly, secondly, thirdly, fourthly, lastly, last, next, overall, similarly,
     specifically, notably, yet (opener use only — see below), instead, together, collectively, rather, besides,
     herein, hereafter, subsequently, thereinto, regardless, otherwise, together with the single words above. Same
     rule and same evidence (sampled 40/40 correct, shards 20 and 30). In-clause "first" ("we first investigated")
     is a separate, kept role (next item): the two uses are told apart by position, not confused.
   - **Single adverbs, attached to their own clause's verb group** wherever they stand in the sentence, **selectively
     by role**, per a word-by-word table built from real attachment data and revised twice by the owner:
     `g3_word_lists/adverb_roles.tsv` (513 words, sha256 `f66d153f5efa`; built by `build_adverb_roles.py`, sha256
     `4b6f4ae541e0`; mirrored on RDS at `PG/g3_word_lists/adverb_roles.tsv` for CSF). Columns: rank, word,
     attachments (count in a corpus sample), role, action, flag (the assistant's reasoning), owner_note (the
     owner's own comments, kept verbatim), example. Of 4,962 sampled attachments, 59% keep (attach) and 39% drop;
     1% are nouns the parser mistyped as modifiers, neither attached nor dropped. Roles kept: degree (greatly,
     marginally), low frequency (rarely, occasionally — "often implies no special claim, rarely changes the
     meaning" is the owner's distinction from high-frequency words), a schedule sense (periodically, daily),
     manner (the open majority of -ly words), exclusive focus (only, merely, simply, solely), likelihood hedges
     (likely, possibly, necessarily), negative or low forms added from corpus counts and kept even where the
     positive is dropped (incorrectly, inconsistently, insufficiently, indirectly — 26 words), contrast (instead,
     rather, otherwise), and concessive "yet". Roles dropped: high-frequency generalisers (often, always, usually,
     typically, consistently), scalar focusers that do not change the claim (even, mainly, especially,
     particularly), stance/attitude (interestingly, surprisingly, unexpectedly), certainty boosters (clearly,
     definitely, actually), other hedges (generally, apparently), time (recently, currently, still — owner: "does
     not add new information" — simultaneously, traditionally). Two conditional, position-dependent exceptions
     mirror the opener rule: "similarly" is kept only directly before "to" (52 of 896 occurrences in 3 shards; the
     bare connective use is dropped); "above"/"below" are dropped only at the end of a clause ("discussed above.")
     and kept as a comparison otherwise ("above average"; 400 vs 186 "above", shards 20/30/40 and 0/10). On real
     data (shards 0 and 20, 400-unit limit, 16,526 units, 0 problems): single-modifier hyperedges fell from 7,556 to
     a count consistent with the smaller kept set; 25 constructed hard cases pass 22/25 (two are a parser reading
     with no predicate anywhere in the sentence, one an over-specific test expectation); 58 of 60 sampled real
     placements correct (one error: a proper name, "Historically Black Colleges", not an adverb at all). Owner's
     own review (two rounds, commits `03d29d8` and the one after, kept verbatim in `owner_note`) corrected or
     confirmed about 50 of the highest-frequency words; the remaining rows followed the same stated logic by
     analogy (keep what turns a claim towards negation, contrast or comparison; drop generic generalisers and
     focusers that do not), each flagged as "analogy/added" rather than the owner's own word.
   - **Flagged, not solved now:** the same argument that justified dropping connectives — synonymous words
     fragmenting otherwise-identical hyperedges — applies to the kept adverbs themselves ("significantly",
     "substantially" and "markedly" before the same verb are now three different verb groups; distinct verb groups
     rose from 4.9% to 5.6% of parents in the sampled shards). Proposed, not run: compare G3 variants with more and
     less merging (attach every adverb; drop every single adverb; the role table above; a WordNet-synonym grouping)
     by the sparsity of the article × child-hyperedge and article × cousin-hyperedge incidence computed directly
     from G3 output, once this and the "not only"/"but also" question above are both revisited together.
   - **"as well as" / "as well": the connective use dropped, the comparative use kept.** "GPT-4 **as well as** BERT
     was evaluated" and "used **as well**" (meaning "too") are dropped; "performed **as well as** the residents" and
     "**at least** as well as…" are kept. Rule: "well" after "as" is dropped unless the word two tokens before is a
     verb or one of a short list of degree words (at least, not, about, nearly, almost, just, equally, roughly,
     approximately), or the comparison continues "or better/worse than". 560 cases across four shards: the simple
     rule alone missed 5 comparisons with a degree word in that position; the extended rule fixed all 5, leaving
     about 0.3% of drops wrong. Chosen over substituting "and" (already redundant, since "and" is itself a stop
     word) as more direct and no less accurate.
   - Residual, left as parser error and not addressed: a handful of sentences the parser reads with no predicate at
     all ("GPT-4 as well as BERT was evaluated on the exam" in isolation), which every rule set, including the
     unmodified script, fails identically.
5. **Leftover phrasal particles — DECIDED 2026-10-08 (owner).** Before: the particles out, up, down, in, on, off,
   over (`PHRASAL_PARTICLES`) were exempt from the stop list unconditionally, so that `fuse_phrasal` could fuse them
   with their verb ("carried out" → `carry_out`). When fusion failed, the particle stayed as a content atom of its
   own (`(cousin_he on/M/en)`; "look up or clarify information" → `(cousin_he information lecture up)`). Not yet
   implemented in `g3_curation_test.py`; implemented and tested as a copy, kept as the patch
   `g3_phrasal_check/g3_q5fix.patch` against the RDS script (RUN_LOG RL-088 to RL-090).

   - **The exemption becomes fusion-conditional.** Particles are ordinary stop words; one survives only as part of
     a fused verb. Fusion runs before the stop list, so this changes nothing else.
   - **Type guard: only particles the parser types `Ml` (its particle tag, spaCy `prt`) or plain `M` may fuse;**
     not `Mt` (a preposition turned modifier) or `C`. All 16 `Mt` fusions in the test sample were wrong ("based
     **on**" / "depending **on**" glued to a nearby participle: `employ_on`, `use_on`, `provide_on`), and both `C`
     fusions came from garbled text. Plain-`M` fusions were mostly right ("go on", "start over", "creep in").
   - **Chain fusion: an `Ml` particle fuses with the verb under a chain of one-argument modifiers** ("to",
     auxiliaries, modals, negation, adverbs): `(out/Ml (should/Mm (be/Mv carried)))` → `carry_out`. Without it,
     fusion missed 200 of 416 `Ml` particles, 184 of them in exactly this pattern ("to look up", "should be carried
     out", "often make up", "was switched off"), and the flip alone would then have turned "make up" into `make`
     and "rule out" into `rule`.
   - **Evidence** (shards 0, 20, 30, 40, 700 articles each: 2,780 articles, 56,126 units; self-test 54/54; 0
     problems). `Ml`: 216 fused and 200 left over before, 400 fused and none left over after (16 unfused because of
     parse errors, now dropped). Fusions: 269 unchanged, 185 gained (`carry_out` 36, `point_out` 17, `break_down`,
     `rule_out`, `make_up`, `turn_off`; 40 of 40 sampled correct), 17 lost (17 of 17 wrong). Dropped leftovers:
     1,096 `Mt` (prepositions, mostly "based on", which G3 drops everywhere else by type) and 315 plain `M` (quantity
     phrases whose number decision f already removes: "over 90%", "up to 3 times", "35 out of 50"); 40 of 40
     sampled drops lose no meaning that the remaining atoms still carried. 24 constructed sentences parsed as in G2
     (`g3_phrasal_check/hard_q5.json`): `make_up`, `look_up`, `switch_off`, `turn_on`, `speed_up` recovered, false
     `employ_on` removed, separable forms kept ("carried the assessment out", "followed the patients up").
   - **Accepted residuals (owner).** About 5 of 454 fusions remain wrong (`fine_tune_on`, `set_over`, `improve_in`,
     `depict_in`, `clarify_over`: garbled source text or G2 attachment errors). "fed our own data in" loses
     `feed_in` because the parser types "in" as `C` (constructed case; the two `C` fusions in the corpus sample were
     both wrong). In 8 units the fused verb now joins the verb group of the clause below through the existing rule
     that a verb-only argument joins the verb group, about 5 of them wrongly ("It has been pointed out that ChatGPT
     can provide…" → `(can point_out provide)`); the leftover particle had blocked that rule by accident. For manual
     checking: PMC11572215.r1.L144.S6.U1, PMC12015923.r1.L148.S5.U1, PMC12232492.r1.L166.S2.U1,
     PMC12415252.r1.L58.S2.U1, PMC12457457.r1.L461.S4.U1, PMC12628953.r1.L134.S4.U1, PMC12988595.r1.L192.S1.U1,
     PMC13143990.r1.L161.S2.U1.
   - **Considered and not built: a token-adjacency guard** (the particle must be the word right after its verb).
     Graphbrain already links a separated particle to its own verb, so separable phrasal verbs fuse correctly when
     the particle comes several words later: 9 correct fusions at distance 2–4 ("made this information **up**",
     "breaking DNA sequences **down**", "turn this option **off**") would be lost, against 2 wrong ones caught. The
     wrong fusions that matter are mostly adjacent ("based **on** embedding") and are caught by the type guard,
     which adjacency would not do. Distance is also undefined for 46 single-token compounds ("follow_up"), and
     ambiguous wherever a word occurs twice in a unit, since atoms carry every position of their word (§5). A
     weaker check, particle after its verb, would catch about 3 more wrong fusions in 2,780 articles and was not
     judged worth the extra rule.
   - **Flagged, for the owner after items 5–7 (numbers).** Decision f drops every atom without letters, so the
     number in a comparison goes and only its frame can remain ("over 90% agreement" → `percent agreement`; "35 out
     of 50" → nothing). If comparisons with numbers should survive, one option is to keep "over" / "up to" /
     "under" when they modify a `percent` atom; not tested. Related: spelled-out numbers as noun heads are dropped
     by type (item 4, correction of 2026-10-08).
6. **Focal-name homonyms and corpus scope — DECIDED 2026-10-08 (owner), in three rounds.** First raised as an "LLM"
   homonym guard in P2: "LLM" is a PLAIN focal term (no guard, any capitalisation) and P2's article evidence, so a
   homonym both creates false focal atoms and unlocks the guarded names. Checked over the whole corpus (all 50 G2 v3
   shards, 34,662 articles; full raw texts where P2's sentences were not enough; RUN_LOG RL-091 to RL-094). Decided:
   **exclude 6,466 articles (18.7%) from G3 and M1 altogether**, and correct 6 misspelt definitions. Built as
   `g3_scope_exclusions/` (reasons and lists below); P2 and G2 are not rerun. Ideally the exclusion belongs to
   preprocessing, beside `PP/article_blacklist.py` (owner).

   - **How "LLM" is used.** 15,215 articles, 329,474 mentions. Classified by the article's own definition (an
     expansion before "(LLM)" counts only if its word initials spell L-L-M): defined as a language model 13,424
     articles (93% of mentions); undefined but "language model" in the article 1,379 (40 of 40 sampled are the
     language model); undefined with other LLM evidence 230 (30 of 30); defined as something else 73; undefined and
     no evidence 109 (read by hand: 66 homonyms, 43 language model). Homonyms found: lipid-lowering medication, low
     or leg lean mass, lipid-laden macrophages, logic learning machine, log-linear and linear logistic models,
     LLM-105 (an explosive), long lateral mass screws, the leucine–leucine–methionine domain, the cell line LLM-MK2.
     In all, 133 articles and 4,863 mentions (1.5%); 128 of these articles have no other focal term.
   - **Excluded: the 73 that define LLM as something else, except 6** (67 excluded, including PMC10967767,
     "Linguistic Landscape Model", as ambiguous). Kept: 5 articles about GPT models whose definition is misspelt or a
     misnomer, and PMC12647564, an LLM review in which one passage uses "logic learning model (LLM)".
   - **Excluded: all 109 with no definition and no evidence**, including the 43 language-model uses among them (owner:
     the corpus is large enough to trade them for fewer false positives).
   - **Excluded: 17 homonym articles of four other PLAIN names**, found by checking which articles rest on one focal
     term alone: ProGen 9 of 87 (the company Progen: antibodies, ELISA kits, a supplement), BioBridge 6 of 8 (a
     tendon implant, a surgical technique, a school programme), PaLM 2 1 of 307 (the Palm 2 domain of Csm1), ESM-2 1
     of 454 (the climate model FIO-ESM-2-0). The bare names P2 already guards (GPT, Claude, Bard, Llama, Gemma) and
     the other PLAIN names were clean in samples.
   - **Excluded: 6,273 of the 6,349 articles whose only focal term is "transformer model(s)".** R1's PMC query asks
     for `"Transformer model"[Text Word]` (and `"LLM"[Title/Abstract]`); P2 accepts the term as focal and only
     denies it the role of article evidence (`NOT_ANCHOR`), so these articles passed to P7 and G2. In P0's analysed
     regions of their raw texts, 6,094 name no LLM-related transformer at all; the 255 that do were read by hand: 76
     use or compare a BERT- or GPT-type model (DistilBERT, ClinicalBERT, XLM-RoBERTa, BERTopic, SBERT; scGPT,
     scBERT, ProtBERT, HuBERT, SleepGPT, TimeGPT), **kept**; 122 mention one only in passing ("such as BERT and GPT",
     "similar to BERT's class token"), 31 only vision-language models (CLIP, BLIP, LLaVA), and 26 are false hits or
     peripheral (Gemini as a scanner, virus, camera or serum supplier; the symbol mT5; n-gram "language models" in
     speech recognition; AI-use statements), all **excluded** (owner).
   - **Spelling corrections in the 6 kept articles.** 6 units: "Large Languge Models", "Large Langue Model", "large
     langaue model's", and three "Large learning models (LLMs)", a misnomer corrected to "language" (owner). Applied
     to the unit's text *and* to the same token in G2's `atom2word` before G3 matches focal terms: correcting the
     text alone left the misspelt word as an atom of its own (`languge/C/en`) because positions are mapped through
     `atom2word`. Tested with `g3_curation_test.py` (unchanged): 6 of 6 units give one `llm/C/focal` atom for the
     definition and no stray atom.
   - **How G3 and M1 use it.** G3 skips every article in `scope_exclusions.csv` and calls `apply_corrections` from
     `build_scope_exclusions.py` on each unit before focal matching; M1 leaves the same articles out of every
     relation, social ones included (otherwise they would still enter article–author and article–journal ties).
     Mirrored for CSF at `PG/g3_scope_exclusions/`.
   - **Considered, not adopted:** a mention-level rule (accept "LLM" only with other focal evidence and no other
     definition) removed 132 of 133 homonym articles at a cost of 590 genuine mentions; the owner preferred removing
     whole articles. Cue words ("LLMs", "LLM-based") would have recovered about 100 genuine mentions but let 8–11
     homonym articles back in.
   - **Flagged, not fixed:** P2's PLAIN list does not match hyphenated "large-language model(s)", which several
     articles use; they stay in the corpus through their "LLM" mentions. If P2 is ever rerun, its list and R1's
     query should be revised to match these decisions.
     *[Fixed 2026-10-08 (owner: "if needed"), RL-095: `PP/focal_terms.py` gains the pattern
     `large[-‐‑]language[-‐‑ ]models?` (PLAIN, so also article evidence; self-test 19/19; the previous version kept as
     `focal_terms.before_hyphen_20261008.py`). In the kept corpus 87 such mentions in 85 units of 76 articles were not
     matched and now are; every spelling maps to `llm`. P2, P6 and P7 are not rerun, so the change acts in G3 only (G3
     matches focal terms with this file). One excluded article now has evidence, PMC13486626 (no definition of LLM, a
     "large‐language‐model‐assisted" GPT-4o screening): left excluded, for the owner to confirm.]*
7. **Focal terms with no governing verb — DECIDED 2026-10-08 (owner).** 908 of 53,735 focal units in the test
   sample (1.7%, excluded articles left out) have no parent and no cousin, in two groups: 411 with no predicate
   anywhere in the G2 parse (abbreviation glosses, headings, captions, "X: ..." list items, reference lines), and
   497 with a predicate elsewhere in the sentence that the focal term does not sit under (elided verbs in a
   coordination, "Open Evidence produced X and ChatGPT Y"; a fronted adjunct, "Similar to ChatGPT, ..."). Checked
   over the whole kept corpus and tested on copies; neither `g3_curation_test.py` nor the production script needs
   any change for this item. RUN_LOG RL-096 to RL-099.

   - **Both groups are left without a parent, as today.** A fallback was built and tested (climb to the nearest
     ancestor holding a clause beside the focal branch, with guards) and recovered about two-thirds of the 497: on
     four test shards, 601 added parents in 587 units, 327 to 348 of the 908 empty units filled, depending on the
     guard. Checked against the real G2 parse tree, not just the sentence text, its accuracy was 72% (36 of 50
     correct) and did not improve enough across rounds of guarding. Five traced causes: when a coordinated clause
     has more than one predicate sibling, the fallback returns the first in document order rather than the one
     nearest the focal term (two "choose" clauses in one sentence merged into one parent; "featured... and was
     developed based on GPT-4's..." attached to "feature" instead of "develop"); a present participle used as an
     adjective is typed `Mv` by the parser and accepted by `is_pred()` as a real verb ("exist**ing** embeddings",
     "as evidenc**ed** in..."); the sibling check looks only at the immediate head of each candidate, missing a
     predicate one level down under a bare adverb ("LLMs often **generate**..."); and some errors are upstream G2
     attachment mistakes independent of the fallback (a passive agent attached to the wrong head). **Owner: given
     the persistent error rate, drop the fallback rather than guard it further** — both groups stay as they are.
   - **Units that reproduce LLM-generated text are deleted outright, so model output is not conflated with
     researchers' own statements about LLMs (owner).** A unit is LLM output if its short prefix (<= 8 words) before
     a ":" holds a focal term plus only a response word ("ChatGPT response:", "Answer from GPT-4:") or holds only
     the focal term with a quotation following ("ChatGPT: "Sure!..."); a prefix with a prompt/question word
     ("ChatGPT prompt:", "was asked:") is a researcher's prompt and is kept. A unit continuing an open quotation is
     deleted too, but only when it is truly adjacent (the next unit cut from the same sentence, or the next
     sentence by raw line) — G2 keeps only focal sentences, so two consecutive G2 records can be pages apart in the
     real article; an earlier, looser version joined 224 to 458 unrelated sentences this way before the adjacency
     check was added. Built as `g3_llm_output/build_llm_output_units.py` (deterministic; no model, no randomness):
     162 units in 55 articles across the whole kept corpus (155 labels, 7 continuations; 30 sampled labels all
     genuine model output), mirrored to `PG/g3_llm_output/` for CSF. Known miss, not fixed: output introduced by a
     question word ("When asked why..., Gemini says: '...'") reads as a prompt and is kept.
   - **How G3 must use it:** skip every `uid` in `g3_llm_output/llm_output_units.csv`, the same way it skips the
     articles in `g3_scope_exclusions/scope_exclusions.csv` (§6 item 6) — before focal matching, since a deleted
     unit is removed whether or not it already parses to a parent.

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

1. Owner decisions on §6 — DONE, all 7 items decided 2026-10-08. *[Remaining: the numbers question flagged
   under items 4 and 5, and the two flags under item 4 (the not_only/but_also drop question, and synonym
   fragmentation of kept adverbs), both deferred to after all 7 questions, per the owner.]*
1a. Implement the decided rules in the test script, then the production script: §6 item 1 (auxiliary "be" kept in
   passives), item 3 (lexical modal verbs, ported from `g3_modal_check/modal_rules.py`, with its hard cases as
   a regression test), and item 4 (negation fix, the "only"/"not only"/"but also" rules, connectives, the opener
   rule and the adverb role table, ported from `g3_word_lists/` and `diagnostics/g3_stoplist/g3_q4full.py`, hard
   cases included), and item 5 (fusion-conditional particle exemption, type guard, chain fusion, from
   `g3_phrasal_check/g3_q5fix.patch`, with `hard_q5.json` as a regression check), item 6 (skip the articles in
   `g3_scope_exclusions/scope_exclusions.csv`; apply `spelling_corrections.csv` through `apply_corrections` before focal
   matching; `test_corrections.py` as a regression check; M1 must leave the same articles out of every relation), and
   item 7 (skip every `uid` in `g3_llm_output/llm_output_units.csv`; item 7 needs no other change).
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
| `fullscale_pipeline/g3_modal_check/modal_rules.py` (rule of §6 item 3, test implementation) | 3f993e0ff164 |
| `fullscale_pipeline/g3_modal_check/hard_cases.json` (40 hard cases with expected results) | 25bc5059068e |
| `fullscale_pipeline/g3_word_lists/adverb_roles.tsv` (§6 item 4 adverb role table, 513 words) | f66d153f5efa |
| `fullscale_pipeline/g3_word_lists/build_adverb_roles.py` (builds the table from corpus attachment data) | 4b6f4ae541e0 |
| `fullscale_pipeline/g3_phrasal_check/g3_q5fix.patch` (§6 item 5, diff against `g3_curation_test.py`) | d7c40968b4cb |
| `fullscale_pipeline/g3_phrasal_check/hard_q5.json` (24 constructed particle cases) | 69873d864054 |
| `fullscale_pipeline/g3_phrasal_check/hard_q5.py` (parses them; runs the RDS script and the test copies) | b1428685f3e2 |
| `fullscale_pipeline/g3_phrasal_check/q5_types.py`, `q5_compare.py` (particle types from G2; orig vs fix) | 5b105b053aed, 47dfbc208df2 |
| `fullscale_pipeline/g3_llm_output/build_llm_output_units.py` (§6 item 7; also on `PG/g3_llm_output/`) | cbceaf621848 |
| `fullscale_pipeline/g3_llm_output/llm_output_units.csv` (162 units, 55 articles; also on `PG/g3_llm_output/`) | b71d5717e250 |
| `fullscale_pipeline/g3_scope_exclusions/build_scope_exclusions.py` (§6 item 6: builds the two lists; G3 correction hook) | 3a5de5167371 |
| `fullscale_pipeline/g3_scope_exclusions/scope_exclusions.csv` (6,466 excluded articles with reasons; also on `PG/`) | 5b71d844d9af |
| `fullscale_pipeline/g3_scope_exclusions/spelling_corrections.csv` (6 unit corrections; also on `PG/`) | d0a70f6c1a20 |
| `fullscale_pipeline/g3_scope_exclusions/test_corrections.py` (the 6 units through `g3_curation_test.py`) | dd18ca3aca3e |
| `fullscale_pipeline/g3_scope_exclusions/decisions/*.csv` (hand-checked lists), `evidence/*.py` (the scans) | see RUN_LOG RL-094 |
