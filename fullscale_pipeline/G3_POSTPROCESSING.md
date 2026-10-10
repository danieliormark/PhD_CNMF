# Stage G3 — postprocessing (curation) of the G2 parse: decisions, open questions, test status

Status (2026-09-30): **design agreed in part; test script built and tested on real data; not yet run on the corpus.**
*[2026-10-08: §6 items 1–7 decided; items 1, 3, 4 and 5 not yet implemented in the test script; items 6 and 7
are built (`g3_scope_exclusions/`, `g3_llm_output/`) and G3 must read both.]*
*[2026-10-09: §6 item 8 (audit of the curated output: identifiers, junk and non-prose pieces, duplicates, numbers)
decided; its participle and cousin rules are implemented in a test copy, the rest is to be built.]*
*[2026-10-10: every rule of §6 items 1–8 is implemented in the test script v2, `g3_v2/g3_curation_v2.py` (§7.1),
and tested on shards 0 and 20; the production script (§8 item 2) is not yet built.]*
*[2026-10-10, RL-123/RL-124: the production script is the test script itself (mirrored, hash-pinned); CSF test passed (job 22539002); the SLURM array and checked merge are written and tested; not yet submitted for the full corpus.]*
*[2026-10-10, RL-118 to RL-121: the promised checks are done, 0–10 as words rolled back, non-text numbers widened, and
code, tables and documents harmonised. Two small questions are open for the owner (§8 item 4).]*
*[2026-10-10, RL-122: both answered and implemented: units without a parent are removed and articles left without any
parent are marked invalid; a score written after a model name is no longer read as a version.]*
Test script `PG/scripts/g3_curation_test.py` (RUN_LOG RL-078 to RL-080) *[superseded by `g3_v2/g3_curation_v2.py`,
§7.1; the RDS script stays as the record, §7.2]*. The May G3 (`chunk_4h_hpc.py`) is
superseded and must not be rerun (deviation D17 in [`PIPELINE.md`](PIPELINE.md); discussion history in PIPELINE.md
item 29). Input: G2 v3, see [`G2_PARSING.md`](G2_PARSING.md). Path abbreviations as in PIPELINE.md.

## 1. Purpose and target schema

G3 reduces each graphbrain edge that contains a focal term to the structures the matrix builder (M1, the full-scale
analogue of the toy `chunk12.py`) turns into relation matrices:

| Structure | What it holds |
|---|---|
| atom | one word as a lemma with a coarse type: `/C` concept, `/P` predicate (verb), `/M` modifier; a focal term is one atom `<canonical>/C/focal` |
| `dummy_sibling` | the compound verb group of the clause that governs the focal term: its verbs, negations and modals (plus verbs of levels above that hold nothing but verbs, negations or modals); since 2026-10-08 also a lexical modal verb together with its complement verb(s), §6 item 3 (decided; *[implemented in the test script v2, §7.1]*) |
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
- *[2026-10-10, owner (RL-122): a version written as the next token (§6 item 8) is not taken when the number is a
  score: followed by a bracketed interval ("ChatGPT 8.0 [7.0–10.0]" → `chatgpt`, the 8.0 kept as a plain value) or
  below 1 ("GPT 0.78 to 0.65"). The numbers inside such brackets were already dropped (§6 item 8). Real versions
  starting a range ("from ChatGPT 3.5 to 4o") or followed by a citation number or a score in round brackets ("Grok 3
  (0.78)") stay versions.]*

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

*[2026-10-10: items 1–8 are all decided and implemented in the test script v2 (§7.1); two small follow-ups are in §8
item 4.]*

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
   *[Measured 2026-10-09, RL-112 (`g3_v2/measure_be.py`, shard 0, auxiliary-typed "be" by the first content word after
   it): passive main verb 3,594 (86.5%), other main verb 450 (10.8%), non-verb 110 (2.7%, mostly degree words or
   coordination before a verb; about 15 real mistyped copulas such as "are capable"). The mistyped copula is therefore
   rare, and is dropped as before. Implemented in the test script v2, §7.1.]*
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
   *[2026-10-10: implemented in the test script v2 through `g3_v2/modal_merge.py`, §7.1 (RL-113, RL-117).]*
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
   *[2026-10-10: implemented in the test script v2, §7.1, the adverb role table included (RL-112).]*

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
   - **"not only" / "just" / "merely" / "simply" / "solely", and "but also": fused, then dropped only as a pair —
     DECIDED 2026-10-08/09 (owner), option 2.** Implemented in `g3_notonly/g3_q4_option2.patch` (the full question-4
     rule set against `g3_curation_test.py`; RUN_LOG RL-083, RL-101 to RL-104). Five mechanisms, in the order they
     act; all of them sit in `Ctx.curate()`, which decides each atom once:
     1. **Fusion of "not only".** "not" or "n't" directly followed by one of the five words becomes one non-negating
        atom `not_only`; the second word is dropped (`q4_fused_not_only`). Reason: the polarity of the construction
        depends on the words it scopes over ("not only **incapable**" stays negative, "not only **cheaper**" does not),
        and it must not be read as a plain negation of its clause.
     2. **Fusion of "but also".** "also" with "but" at most three tokens before it, only auxiliaries, modals,
        be/have/do forms or pronouns in between and no punctuation ("but also", "but can also", "but it also"),
        becomes `but_also`. A parser quirk is corrected first: when "also" is attached to the conjunction itself
        (`(also/M but/J) A B`), it is moved into the predicate of the clause after "but" (`q4_move_also`).
     3. **Attribution.** Both atoms count as verb-group members (`pred_or_neg`), so they reach the clause's verb group
        or argument by the same predicate-finding logic as any other atom. About 29 of 32 sampled placements correct.
     4. **Guard for identical atoms.** Atoms carry no position of their own, so a rule tied to a position holds only
        if every occurrence of the same atom string qualifies (`q4_all`). Consequence: in 3 of 319 test units with a
        not-only construction the unit has another "not", nothing is fused, and the construction reads as a negation.
     5. **Dropping, option 2.** `not_only` is dropped when a partner — "but", "also" or "as well" — follows the
        not-only word in the unit before the next ";" (`q4_paired`); `but_also` is dropped when a not-only stands
        before its "but" in the unit after the last ";" (`q4_also_paired`). So the additive pair "**not only** reduced
        costs **but also** improved accuracy" becomes `reduce` + `improve`, which reads as "and"; a bare "X **but
        also** Y" keeps `but_also` as the contrast marker ("may help clinicians **but also** mislead patients", owner:
        "but" here is a contradiction, and without it the clause could be misread); an unpaired "not only/merely/
        simply" keeps `not_only`, the only trace of a denial ("risks are **not merely** theoretical", "an adjunct,
        **not simply** a replacement" — dropping would assert the opposite). The ";" stops the search because it opens
        an independent clause whose "but"/"also" is not this construction's partner ("does **not simply** vanish once
        labels are removed; Its F1 increment is null at 0.02 **but** significantly positive again..."). Words are
        located in the text through `token_spans()`: punctuation has no token position of its own, and the
        alignment finds "n't" and the "not" of "cannot" (tokenized "can" + "not") in order.

     **Why drop in `curate()`, not earlier or later.** Removing the words from the text would need G2 to be re-run
     (token positions shift); stripping them from the parse edge would mean rebuilding modifier edges such as
     `(not/Mn (only/M provide))`, an easy way to produce malformed edges, for no gain in speed; stripping them from
     the finished structures needs an extra pass and re-hashing. Dropping in `curate()` is the earliest safe point: on
     shards 0, 20, 30 and 40 (56,126 units) its output equals the strip-afterwards reference in every one of the 359
     units with a fused atom and changes no other unit, and all 57 question-4 hard cases agree; run time is the same
     within noise (31–38 s per shard), since `curate()` decides each atom once.

     **Evidence for option 2.** "but … also" (313 occurrences in the test shards): 248 follow a not-only and are
     additive (20 of 20 sampled, whatever the valence: "not only misleading, **but also** deceptive"); 65 are bare
     and mostly contrastive (26 of 30 sampled: "highest AUC **but also** cost the longest training time"). The
     valence of the halves ("reduced costs" good, "reduced accuracy" bad) is not needed and could not be read
     deterministically. "not only" (300 fused): 273 paired, 24 unpaired. Result of the final rule on the four
     shards: `not_only` dropped in 271 units, kept in 29; `but_also` dropped in 236, kept in 55; structure otherwise
     identical to keeping both atoms. Read by hand: 30 of 30 sampled dropped `but_also` are not-only partners; of the
     55 kept, about 43 are contrastive and about 12 additive ("programmers **but also** managers"; the marker is
     kept, no meaning lost). Distance between the halves does not matter: true pairs up to 25 words apart, across
     embedded clauses, are found. Hard cases `g3_notonly/hard_option2.json` (23, including contractions, "cannot",
     bare contrastive and mixed constructions): 19 of 23.

     **History.** The owner first proposed dropping both atoms (they read as "and"); the paired/unpaired difference
     for "not only" was found on corpus samples (RL-101), then the contrastive bare "but also" (owner, RL-102). A first
     implementation tested with another model (RL-103) still dropped `but_also` always and missed contractions and
     "cannot"; corrected after an audit (RL-104).

     **Known limits, accepted (no case found in the corpus, constructed sentences only):** a comma-spliced new
     clause with its own subject after a not-only, with no ";" and no "also" ("ChatGPT is not only widely used...,
     **but** hospital administrators... raised separate concerns"), wrongly drops `not_only` (3 of 3 constructed
     cases); a unit with two "but … also" after a not-only drops the contrastive second one with the first, because
     identical "also" atoms are decided together. Also: a true pair split by ";" keeps `not_only` (safe; 1 real case);
     "not be solely X but may also Y" is not fused at all, because "not" and "solely" are not adjacent.
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
     rose from 4.9% to 5.6% of parents in the sampled shards). **Decided 2026-10-09 (owner): no merging first.**
     Run M1 on G3's output as already decided (the role table, no synonym grouping) and measure the sparsity of the
     article × child-hyperedge and article × cousin-hyperedge incidence directly; only build a merging step (a
     WordNet-synonym grouping, or collapsing to the role classes of the adverb table) if that sparsity turns out to
     be a real problem. Superseded: comparing several G3 variants (attach every adverb; drop every single adverb;
     the role table; a WordNet grouping) up front, before M1 exists. This is an M1-stage question, not G3; revisit
     once M1 is built and run.
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
   *[2026-10-10: implemented in the test script v2, §7.1 (RL-112).]*

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

   *[2026-10-10, RL-125, confirmed against the real production run (job 22540333), not just the plan: exactly 6,466
   articles excluded — `scope_exclusions.csv` has 6,466 rows, `g3_articles_v1.jsonl` has exactly 6,466 rows with
   status `excluded`, and none of them contributes a single unit to `g3_test_v1.jsonl` (checked directly: two named
   examples, PMC10967767 and PMC8815195, are present only as an excluded row with 0 units/0 parents and absent from
   the whole output). **New corpus size for G3 and M1: 28,196 articles** (34,662 − 6,466). Of those, 121 further have
   no usable G3 parent at all, for the separate, unrelated reason decided in item 8 (units with no parent anywhere
   are removed, RL-122) — **28,075 articles carry actual content** in `g3_test_v1.jsonl`. This does not revise
   P1–G2's own article count (still 34,662 there): the exclusion is scoped to G3 and M1.]*

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

8. **Audit of the curated output — DECIDED 2026-10-09 (owner), several rounds.** The owner's four points: traceable
   ids, junk and non-prose pieces, the stop list beside the adverb table, and duplicated words. Checked on the shard-0
   output of the question-4 test copy (694 articles, 14,225 units) and, where stated, on the whole corpus (RUN_LOG
   RL-106 to RL-110). Two rules are implemented in a test copy; everything else is decided and still to be built.
   *[2026-10-10: all of item 8 is implemented in the test script v2, §7.1.]*

   - **Identifiers.** Provenance as in §5 holds (0 problems). Each structure gets three identifiers:

     | Identifier | Built from | Example | What it is for |
     |---|---|---|---|
     | occurrence id (exists) | position | `<uid>.P1.S2`, with `sha1(id\|edge)[:12]` as an integrity hash | tracing a structure to its unit, sentence and article |
     | entity id | role + set of words (the edge string) | `(sibling_he survey/C/en user/C/en)` | the matrix row: the role decides the facet, so a structure lands in the right matrix |
     | **content key** (new) | set of words only, without the role: `sha1(sorted atoms)[:16]` | {survey, user} | finding the same content in different roles or facets |

     The **content key**'s uses:
     - In v11 there are no atom facets (owner: single-word community attribution is not theoretically meaningful,
       since the theory works on associations, and it is unstable). So a child hyperedge and a cousin hyperedge with
       the same words, possibly from different sentences or articles, are otherwise unrelated in the matrices; the
       key is what can link them if M1 wants that link.
     - Counting how often content recurs across roles.
     - Storing each distinct content once.
     - Grouping identical content when building tables.

     16 hex characters keep the chance of any collision negligible at corpus size; 12 would give about 0.7% for
     roughly 2 million distinct structures. M1 must check that no two different word sets share a key. The role
     stays part of the entity id: (focal_he X), (sibling_he X) and (cousin_he X) are different entities.
   - **Spelling-corrected units (§6 item 6):** verify the unit hash on the original text, then correct, and record
     the correction (the check otherwise fails on the 6 corrected units).
   - **Junk and non-prose pieces.** Upstream removal (P1b, P1c, G2 v3) and decision f leave, in shard 0, these atom
     classes. Decided:
     - Layout words (table, fig, figure, appendix, supplementary, panel, equation) are dropped when they are not
       plain text, i.e. when they name a non-text object by a label ("Table 2", "Fig. 1b", "Supplementary Table
       S1"), together with the label. Label uses: table 343/368, fig 253/261, figure 207/234, appendix 28/36.
       Kept in prose: "the figure presents", "an expert panel", "a panel survey".
     - Letters-plus-digits atoms (2,113) are kept, because most are names (word2vec, T5, ESM2, 3D, CHA2DS2, F1,
       12th). Exceptions: figure labels (by the label rule); model sizes (8b, 13b, 70b) dropped; version tokens split
       off a focal name (764 directly after a focal term: "DeepSeek-V3" → `deepseek` + `v3`, "ChatGPT 4o") folded
       into the canonical focal atom (`deepseek_v3`, `gpt_4o`), consistent with §4.
     - A hyphen left inside an atom (U+2010 and similar, 273) becomes "_". The other symbol atoms (escaped ".",
       "(", ")", "&", "@"; glued citation digits "self_education,2") are reviewed and cleaned.
     - Citation tokens (`REF000053`, 341) are kept: cited works are sources of authority and objects of critique.
     - URL tokens (`URL12345678`, 29) are removed.
     - Single letters (534: variables, list labels) are dropped.
     - Plain text must contain at least a parent; units without one already contribute nothing. Lines that do
       parse to a parent but are not prose are dropped by four rules, precision to be checked on a labelled sample:
       headings (no end punctuation and Title Case, or a section number such as "4.2.2."); reference entries
       ("45 L. P. Argyle, Out of one, many: ..."); pseudocode ("B_output←Pooled_output(...)//..."); figure-panel
       captions ("A Accuracy evaluation ...; B Accuracy evaluation ..."). Glossary lines ("LLM equals large
       language model.", "LLM1 indicates GPT-4o (...)") are dropped by a glossary rule: a parent whose verb is
       equal/indicate/denote/stand for/refer to and whose arguments are only focal terms and their expansions.
       73 candidate units with a parent in shard 0.
       *[Checked RL-114: the test script v2 removes 25 non-prose units in shard 0 (14 headings, 6 pseudocode, 3 panel
       captions, 2 reference entries), all 25 read and correct; the glossary rule drops 9 parents there (after a fix
       for 2 of 13 wrong drops in its first version; 11 since RL-120, whose number rules remove the stray "4" that had
       hidden the glossary line "Language Model 4; LLM equals large language model.", 2 parents).
       Recall was not measured.]*
   - **Stop list and adverb table: kept as separate files**, loaded through one place with a consistency check at
     start-up. The NLTK list drops a word whatever its role; the corpus-built table applies only to single adverbs
     and decides attach or drop. They share only "only", which has its own rule. 12 table rows still read "open
     question 5" and are to be updated to item 5's decision. How the table was built is documented in item 4.
     *[Updated 2026-10-10, RL-121: the 12 rows carry the actions decided below (option 3), written by
     `build_adverb_roles.py`; the table's sha256 is now `79466ff78dfe` (RDS copy updated), and the test script checks the
     table against the decision at start-up instead of overriding it in code. Output unchanged.]*
   - **Duplicated words.** Within one structure duplicates are impossible (a structure is a set). Found elsewhere:
     - **Participle properties, now one rule (implemented in the test copy).** A participle joined by a hyphen to
       the word before it ("LLM-based chatbots", "ChatGPT-generated text", "LLM- based", en dash, or a hyphen word
       G2 already joined, `expert_written`) is a property of the noun, not a predicate (owner): its phrase is not
       a clause, the word is kept as written and typed as a modifier (`based/M/en`, not `base/P/en`), and it never
       counts as a verb-group member. Without a hyphen the participle is treated as before ("text generated by
       ChatGPT" keeps `generate` as its verb). Before: G3 made the modifier itself a clause, putting the focal term
       in `focal_he` and `sibling_he` and the participle in the verb group and `focal_he` (647 parents, 3.8%).
       After: the phrase sits in the focal argument of the clause's real verb ("LLM-based assessment tools ...
       could promote ..." → `(dummy_sibling could promote) (focal_he assessment based llm tool) ...`). Shard 0:
       parents with a token in two children 647 → 27 (the rest are other parse patterns); 971 parents changed;
       836 hyphen properties; 29 units lose every parent: in all of them the parser attached the focal phrase
       outside every clause, and before the rule the participle served as a fake verb. 9 are headings, titles or
       labels; 20 are real sentences (subject phrase parsed as the head of the sentence; focal phrase under a
       top-level "on"/"than"/"in" connector). Handling under discussion (RL-111). Hard cases
       `g3_audit_rules/hard_rules2.json`: 10 of 10 (4 of 10 before). "llm_based" is not made a focal term of its
       own (it would split the `llm` entity).
       *[2026-10-10, RL-121: re-measured with the whole test script v2 against the same script with the rule off
       (`XBASED=0`): 27 of 13,820 units lose every parent in shard 0 (0.20%), 14 of 14,483 in shard 20 (0.10%), none
       gains one; about half are headings, captions or bullet fragments ("• BERT-based monolingual model."). The
       script leaves them without a parent (option A of RL-111, the same treatment as item 7); the owner's
       confirmation is open, §8 item 4.]* *[Decided 2026-10-10 (owner), RL-122: every unit that ends up without a
       parent, for this or any other reason, is removed; an article none of whose units has a parent is marked invalid.
       See §7.1, step 9.]*
     - **Emphatic reflexives (P6).** P6 replaced every reflexive pronoun by its antecedent; emphatic uses then
       duplicate the noun ("even for the model the model", "just like LLMs LLMs"). Told apart by the dependency
       label of the reflexive in the original sentence (spaCy `en_core_web_trf`, all 3,157 replacements; samples
       read by hand): emphatic = apposition (1,914; 8/8), adverbial noun phrase (248; 8/8), predicate attribute
       (22; 8/8), "in itself" (48; 10/10), "by itself/themselves" except a passive agent (about 90; 8/10 "alone");
       object = direct or indirect object (606; 16/16), other prepositions (about 150), small-clause subjects (46;
       7/8). Decided: object uses stay ("LLMs cannot evaluate LLMs" is a statement; lexicalised "presents itself
       as" stays too); emphatic uses get the original pronoun back from P6's record (original text and character
       offsets are kept), which G3 then drops as a pronoun, and the affected units are re-parsed with G2's parser
       and given to G3 through a replacement table with the original unit hash kept (as for spelling corrections).
       A larger labelled sample (about 100) confirms the split before use. *[Done 2026-10-10, RL-118: 60 of 60
       restored units emphatic and correctly restored, 40 of 40 kept object uses correct. Table built by
       `g3_v2/reflexive_restore.py` (RL-115).]*
     - **Cousins seen differently by different parents (implemented in the test copy).** Each parent's cousins are
       the unit minus that parent's own clause. When the sweep had to cut a phrase open to reach a clause nested in
       it, it split the phrase word by word, while a parent whose own clause was that nested clause saw the phrase
       whole: "Those who find ChatGPT valuable are more likely to use ChatGPT regularly" gave `(cousin_he more)` +
       `(cousin_he likely)` for one parent and `(cousin_he likely more)` for the other. Rule: a phrase cut open
       keeps all its words outside the nested clauses as one cousin, which is exactly how the other parent sees it,
       so a phrase has one rendering in every view and identical cousins are one entity with a parent list. A
       narrower variant (group only loose single words) left the views inconsistent (13/15 hard cases) and was
       rejected. Real units (RL-111): of 30 changed units read by hand, 26 correct, 4 join items that are separate
       (3 coordinated list items, 1 run-in label), the way G3 already joins coordinated items in any phrase with no
       clause inside. Shard 0: tokens in two cousins of a unit 725 → 191 with both rules (the rest: a verb group
       absorbing a verb-only clause in one view only, by the verb-group rule); one-word cousins 12,398 → 10,183;
       cousins 53,609 → 48,809; parents unchanged by this rule. Hard cases: 5 of 5 (0 of 5 before).
     - **Not changed (owner):** one parent per focal hyperedge, also when one clause holds two focal terms (836
       clauses in shard 0; the approach is centred on focal terms and their hyperedges, so parents are not merged).
       Cousins are not typed by kinship degree; the current breadth is sufficient. For the record, clause levels
       down to the focal term (shard 0 only, approximate: relations with a predicate connector, 12 focal names,
       14,726 occurrences): 1 level 59.7%, 2 levels 30.2%, 3 levels 7.7%, 4 or more 1.8%.
   - **Numbers — DECIDED 2026-10-09 (owner, revised the same day; supersedes option (c), which used one placeholder
     atom for every value and so made all numbers look alike).** Non-text numbers are removed by rule; every other
     number stays as written, years included; integers 0–10 become words, applied only after the removal rules, so
     that a label is never turned into a count. *[Rolled back 2026-10-10 (owner, RUN_LOG RL-118/RL-119): the promised
     precision check found 41 of 100 converted digits were labels, model versions, dates or debris that the removal
     rules miss ("Textbox 3", "Claude Sonnet 4", "October 9"), so the conversion made them look like counts; and the
     conversion was judged not important. Every plain-text number now stays as written, digits included; spelled-out
     numbers stay as written too, so "2 models" and "two models" remain different atoms. The removal rules still
     miss those 41 kinds of cases, which now leak as digits: open, see RL-118.]*
     *[2026-10-10, owner: remove what is not plain text, keep what is (RL-120). Dropped now: references to non-text
     objects with their number ("Textbox 3", "Online Resource 5", "Supplementary Note 3", "Box 1"); index numbers on
     things in the prose, the word kept ("Topic 7", "Surgeon 2", "group 0", "round 1"); citation and list debris,
     formula numbers, snapshot-date ids, numbers after SD/SE/IQR, intervals right after a value, citation years ("et
     al. (2020)"). Kept as plain text: numbers inside names ("Claude Sonnet 4") and dates ("October 9, 2023"). Guards:
     the label directly before its number; "%" or a unit after a number makes it a value; an index is a whole number.
     Checked on fresh samples: 37/40 removals correct before the last two fixes (all 40 after), 95/100 kept numbers
     plain text.]* Removed: labels (Table/Figure/Section/Question/Step N, "Table 2",
     "Section 2.3"); bracketed list labels ("(1)"); enumerators and numbers at the start of a unit ("[21] A
     clinician…"); every number inside square brackets (intervals such as "[5.8–7.0]" and citation leftovers such
     as "(PAL)[18]"; owner: they bear little relevant meaning); statistical notation (a number after p, CI, OR, HR,
     ±, =, <, >, n =). *[Rolled back, see the note above:]* integers 0–10 as words remove a style artefact (journals differ on spelling out numbers
     below ten), not a meaning; their precision is checked on a hand-labelled sample before use. Shard 0 (RL-111):
     10,401 numbers in kept units, of which 2,864 are already part of a focal name, about 1,400 fall under the
     removal rules (labels 685, square brackets 244, statistics 196, list labels 195, enumerators 77), and the rest
     (percentages 1,564, integers 0–10 1,390, integers above 10 1,362, decimals 1,123, years 375, ranges 326) are
     kept, minus statistics among decimals and ranges. Accepted cost: hyperedges that differ only by a value are
     different entities (measured at M1). Comparison words next to a number (over, under, up to, more/less than,
     at least, nearly, approximately) are kept. Spelled-out numbers heading a noun phrase (typed `C#`, dropped by
     type today) are kept as written. Supersedes the open flag under items 4 and 5. To be implemented and tested.
     *[Implemented and tested in the test script v2: RL-114, RL-119, RL-120.]*
   - **The 12 adverb-table rows left by item 5 — DECIDED 2026-10-09 (owner, option 3).** forward, forwards, ahead,
     behind, back, away, throughout, before, under, around, despite, beside were given the role "particle" when the
     table was built and deferred to item 5, which decided only the seven fusable particles. Most of their uses are
     prepositions (types T, Br, Jr; dropped by type, about 400 in shard 0); the table acts on the plain-adverb uses
     only (43 in shard 0, about 1,000 in the corpus). Rule, following decisions already taken: direction words
     (forward(s), ahead, behind, back, away) fuse with their verb like the item-5 particles, including a plain
     modifier on a modal/auxiliary chain ("Looking ahead" → `look_ahead`, "must move forward" → `move_forward`, "left
     behind" → `leave_behind`), and otherwise stay a modifier in their phrase ("a significant step forward"); time uses
     (before, throughout) are dropped like the table's time adverbs; around and under are kept only before a number
     (numbers rule: "around 30%"), otherwise dropped; despite and beside are dropped. Hard cases
     `g3_v2/hard/hard_adverbs12.json` 10/11 (D04: the parser attaches "back" to "to the user", not to the verb).
   - **Flag for M1 (owner 2026-10-09): verb groups are deliberately specific.** Keeping passive "be" (`be/M`, item 1)
     makes "X was trained" and "X trained" different verb-group entities, and the same holds for the other atoms G3
     puts into `dummy_sibling` / `dummy_cousin`: modals (can, should; item 2 of §3), negation, `only`, `not_only` /
     `but_also` when kept, the adverbs the role table keeps (item 4), `able` / `unable` and the complement verbs of a
     lexical modal (item 3), fused phrasal verbs and direction words (item 5 and above), and verbs pulled up from
     verb-only levels. Each makes one core verb appear in several entities, which raises the sparsity of every
     relation that holds verb groups. This is intended (owner: these distinctions carry meaning), but M1 should
     measure it (how many verb-group entities per core verb, and their article counts) and decide whether a
     back-off is needed. A cheap option, not built: a second key on verb groups over the predicate lemmas alone
     (modifiers left out), written like the content key, so M1 could group or compare verb groups by their core verb.
     Related: the synonym question of item 4 (deferred to M1) concerns the same relations.
   - **Implemented** in the test script v2 (§7.1), together with every other rule of this item. The earlier copy:
   - **Test copy and checks** *[the record of the decision; superseded by §7.1]*: `diagnostics/g3_audit/g3_rules2.py` (sha256 `53f3162ddb83`; `XBASED=hyphen`,
     `GROUP=1`; `XBASED=0 GROUP=0` reproduces the question-4 copy byte for byte); hard cases
     `g3_audit_rules/hard_rules2.json` and runner `hard_rules2.py` (15/15; 4/15 before); audit scripts in
     `diagnostics/g3_audit/`.

## 7. Test script

### 7.1 Test script v2 (current, 2026-10-10): every rule of §6 items 1–8

```
cd fullscale_pipeline/g3_v2
python g3_curation_v2.py --selftest                                             # 54 cases
python g3_curation_v2.py --shard N [--limit A | --limit 0] --outdir DIR [--db] [--show K]
python run_hard.py hard/hard_*.json                                             # 199 constructed cases, 175 with expectations
```

`g3_v2/g3_curation_v2.py` (sha256 `0f3146266f26`; `2f7e8f356692` at RL-117, `389123552561` at RL-120, `efbf44b09683` at
RL-121, which changed only input checks with outputs byte-identical) with its module `g3_v2/modal_merge.py` (`75c7022ac187`; it must sit
beside the script). Built stage by stage from the RDS test script of §7.2 (RUN_LOG RL-112 to RL-117); the RDS script is
unchanged. `--limit` counts articles (default 100; 0 = the whole shard). Environment as G2 (`tensor_env`,
`PYTHONPATH=$HOME/np1_for_spacy`). The switches kept from the test copies (`Q4`, `ONLY_MODE`, `NOTONLY_DROP`, `XBASED`,
`GROUP`, `HYPH_JOINED`, `Q5_CHAIN`, `Q5_TYPE`) default to the decided settings and exist for comparison runs only.

**What it does, in order.**
1. **Article:** skipped if listed in `scope_exclusions.csv` (§6 item 6).
2. **Unit**, each checked against its list's unit hash:
   - skipped if it reproduces LLM output (item 7);
   - skipped if it is not prose: heading, reference entry, pseudocode or panel caption (item 8); the skipped units
     are written to `g3_nonprose_NNN.jsonl` for review;
   - replaced by its restored, re-parsed version if it holds an emphatic reflexive (item 8);
   - spelling-corrected (item 6).
   The unit hash is always verified on the original G2 text, and a changed unit records `text_original`.
3. **Article evidence** for the guarded focal names is taken from the kept, corrected units.
4. **Lexical modal verbs (item 3)** are merged on the parse as G2 gave it, before any replacement.
5. **Focal terms** are found and replaced, with a version token after a versioned name folded into its atom (item 8).
6. **Extraction:**
   - phrasal particles and direction words are fused with their verb (items 5 and 8);
   - a phrase holding a hyphen-joined participle is not a clause (item 8);
   - the parent clause and its verb group are found as before;
   - the cousins are swept, a phrase cut open keeping its other words as one cousin (item 8);
   - glossary parents are dropped (item 8).
7. **Atoms**, each decided once per unit (`Ctx.curate`):
   - focal atoms, and the hyphen properties written as `based/M`;
   - labels and their numbers, non-text numbers dropped, other numbers kept as written (the 0–10-as-words step was
     rolled back, RL-119);
   - `percent`; atoms without letters, connectives, URL tokens, model sizes;
   - comparison words before a number;
   - the item-4 rules (not-only / but-also option 2, connectives, openers, "as well");
   - the adverb role table, including the 12 rows decided in item 8;
   - negation, pronouns, determiners;
   - passive "be" kept as `be/M` (item 1), other auxiliaries dropped;
   - modals, and `able` of a merged "be able to";
   - the stop list (particles no longer exempt, item 5), the type filter, single letters;
   - the remaining atoms written as lemma and type, symbols cleaned and Unicode hyphens written as "_".
8. **Identifiers:** ids and hashes as in §5, plus a content key on every child, cousin and parent (item 8).
9. **Units and articles without a parent** (owner 2026-10-10, RL-122): a unit with no parent is removed from the output
   and written to `g3_noparent_NNN.jsonl` for review; an article none of whose units has a parent is marked
   `invalid_no_parent` in `g3_articles_NNN.jsonl`, which gives every article of the shard a status (`excluded`, `valid`,
   `invalid_no_parent`). M1 uses only `valid` articles, in every relation, social ones included.

**Inputs**, read from RDS, each checked against its sha256 before the run:

| File | sha256 (12) |
|---|---|
| `PG/g3_scope_exclusions/scope_exclusions.csv` | 5b71d844d9af |
| `PG/g3_scope_exclusions/spelling_corrections.csv` | d0a70f6c1a20 |
| `PG/g3_llm_output/llm_output_units.csv` | b71d5717e250 |
| `PG/g3_word_lists/adverb_roles.tsv` | 79466ff78dfe (f66d153f5efa before RL-121: only the 12 rows of item 8 changed) |
| `PG/g3_reflexive/reflexive_restored.jsonl` (1,211 units; corpus text, so on RDS only) | d121a203daec |
| `PG/nltk_abridged_stopwords_list.txt` (checked since RL-121) | 2b6c7d9fdae9 |
| `PP/focal_terms.py` (P2's matcher, imported; checked since RL-121) | 892f98bb986e |

**Outputs** (refuses to overwrite) as in §7.2:
- `g3_test_NNN.jsonl`: per record, also `content_key`, and `text_original`, `corrections`, `reflexive_restored` where
  they apply; *[since RL-122 only units with a parent]*;
- `g3_noparent_NNN.jsonl` and `g3_articles_NNN.jsonl` (RL-122, step 9);
- `g3_errors_NNN.jsonl`;
- `g3_report_NNN.json`: also the input hashes, and counts of every new rule;
- `g3_nonprose_NNN.jsonl`;
- with `--db`, the chunk12-form database.

**Checks**, before exit (exit code 1 on any problem): those of §7.2, plus that every content key matches its edge and
that each list entry matches its unit.

**Test suite.**
- **Self-test:** 54/54. Four cases were updated where a decision changed the expected output: passive "be", the modal
  merge, a plain-text number now kept, and "allows … to integrate".
- **Hard cases:** `g3_v2/hard/`, run with `run_hard.py` on the parse cache `hard/parsed_cache.json`, 162/175 (RL-122; 154/167 at RL-120, 132/145 before):

  | Set | Pass |
  |---|---|
  | item 4 adverbs | 22/25 |
  | but also | 7/8 |
  | negation and only | 22/24 |
  | option 2 | 19/23 |
  | item 8 participle and cousin | 15/15 |
  | passive | 9/10 |
  | lexical modal, G3 level | 13/14 |
  | item 8 atom and number | 11/11 |
  | the 12 adverbs | 10/11 |
  | glossary | 4/4 |
  | non-text numbers (RL-120) | 22/22 |
  | scores after model names (RL-122) | 8/8 |

  The 13 failures are all known and documented in RUN_LOG and in the cases' notes: parser misreadings, option 2's four
  accepted limits, and "let us" pulled up by the existing rule for verb-only levels.
- **Earlier runners** (`g3_notonly/hard_option2.py`, `g3_phrasal_check/hard_q5.py`, `g3_audit_rules/hard_rules2.py`) load
  test copies that live only in the untracked `diagnostics/` folder; they are the record of their decisions and are not
  needed any more: `run_hard.py` runs the same sets on the v2 script. Of the copied sets only `hard_rules2.json` differs
  from its original (one expectation updated for item 1, with a note).
- **Copied sets:** sets copied from earlier work whose expectations predate a later decision keep the old expectation
  beside the new one, with a note.
- **Item-5 particle cases:** they carry no expectations. They are compared with the item-5 reference copy: 14/24
  identical, and the 10 differences all come from later decided rules.
- **Lexical-modal decision cases:** through the copied decision module, 33/40, the same as the original.

**Results** (RL-121; identical to RL-120), both shards with 0 errors and 0 problems, about 0.003 s per unit
*[RL-122: the same, except that the units without a parent (shard 0: 510, of which 290 hold a focal mention; shard 20:
470 and 252) are removed, 2 units of shard 0 lose a false version, and the articles are 560 valid, 2 invalid (shard 0)
and 569 valid, 3 invalid (shard 20)]*
*[RL-117 values, before the number changes of RL-119/RL-120, in brackets where they differ]*:

| | Shard 0 (`--limit 700`) | Shard 20 (whole shard) |
|---|---|---|
| Articles | 694 | 693 |
| Excluded articles | 132 | 121 |
| Units | 13,820 | 14,483 |
| Units with a parent | 13,310 (13,311) | 14,013 |
| Parents | 16,523 (16,525) | 17,130 |
| Cousins | 47,077 (47,100) | 49,224 (49,265) |
| Atoms kept / dropped | 281,606 / 178,266 (281,852 / 178,048) | 294,136 / 185,398 (294,517 / 185,017) |
| Modal merges (not applied) | 621 (38) | 731 (54) |
| Reflexive-restored units | 24 | 34 |
| LLM-output units | 0 | 2 |
| Non-prose units | 25 | 15 |
| Glossary parents | 11 (9) | 5 |

**Determinism.** Shard 0 and shard 20 were each run twice under different Python hash seeds; each pair is
byte-identical. Before RL-117, the lexical-modal decision took "the first verb" from graphbrain's `atoms()`, which is a
set, so 2 units of shard 0 changed between runs. The decision now reads atoms in edge order.

**Flags for M1** (§6 item 8):
- the content key;
- the specificity of verb groups;
- the article exclusions of item 6, which M1 must apply to every relation.

### 7.2 Earlier test script (2026-09-30, superseded by 7.1; kept as the record)

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

1. Owner decisions on §6 — DONE, all 7 items decided 2026-10-08. *[2026-10-09: the two flags under item 4 are
   settled (not-only/but-also option 2; synonym merging deferred to M1). Remaining: the numbers question flagged
   under items 4 and 5.]* *[2026-10-09: numbers decided in item 8 (option c); item 8 added and decided.]*
1a. *[DONE in the test script 2026-10-10: `g3_v2/g3_curation_v2.py`, §7.1; the production script remains, item 2.]*
   Implement the decided rules in the test script, then the production script: §6 item 1 (auxiliary "be" kept in
   passives), item 3 (lexical modal verbs, ported from `g3_modal_check/modal_rules.py`, with its hard cases as
   a regression test), and item 4 (negation fix, the "only"/"not only"/"but also" rules, connectives, the opener
   rule and the adverb role table, ported from `g3_word_lists/` and `diagnostics/g3_stoplist/g3_q4full.py`, hard
   cases included), and item 5 (fusion-conditional particle exemption, type guard, chain fusion, from
   `g3_phrasal_check/g3_q5fix.patch`, with `hard_q5.json` as a regression check), item 6 (skip the articles in
   `g3_scope_exclusions/scope_exclusions.csv`; apply `spelling_corrections.csv` through `apply_corrections` before focal
   matching; `test_corrections.py` as a regression check; M1 must leave the same articles out of every relation), and
   item 7 (skip every `uid` in `g3_llm_output/llm_output_units.csv`; item 7 needs no other change). For item 4, the
   reference implementation of the whole rule set, including the not-only/but-also option 2, is
   `g3_notonly/g3_q4_option2.patch` (it supersedes `diagnostics/g3_stoplist/g3_q4full.py`), with
   `g3_notonly/hard_option2.json` as a regression check. Item 8: content key, junk and non-prose rules, glossary rule, model-version
   folding, emphatic-reflexive restoration (with the re-parse of affected units), numbers; the participle and
   cousin rules from `diagnostics/g3_audit/g3_rules2.py`, with `g3_audit_rules/hard_rules2.json` as a regression check.
2. Production script from the test script: output folder `PG/g3_v1/` (not `PG/postprocessed_output/`), one task per
   G2 shard, a SLURM wrapper, a CSF test, and a separate checked merge (as for G2).
   *[2026-10-10, RL-123: production uses the test script v2 itself, mirrored to `PG/scripts/g3_v2/` and pinned by
   sha256 (no second copy that could drift). CSF test written: `PG/scripts/g3_v2/check_csf_g3.sh` (script hashes,
   self-test, hard cases, whole shards 0 and 20 compared byte for byte with an incline reference run of the same script
   in `PG/g3_v1_csftest/reference_incline/`); passed on CSF (job 22539002).]*
   *[2026-10-10, RL-125: run for real (job 22540333). All 50 tasks finished in ~4.5 min, 0 problems, 0 unit errors
   anywhere. `PG/g3_v1/` now holds the corpus-scale G3 output: 34,662 articles (28,075 valid, 121 invalid_no_parent,
   6,466 excluded, matching item 6's own count); 664,725 units with a parent; 816,075 parents; 2,266,190 cousins;
   identical to the RL-124 scratch dry run. G3 production is complete. Next: the M1 input contract (item 3).]*
   *[2026-10-10, RL-124: the SLURM array and the checked merge are written and tested. `g3_v2/submit_g3_v1.sh`
   (sha256 `d0c8dad02783`; tracked copy of `PG/scripts/g3_v2/submit_g3_v1.sh`): 50-task array on `serial` (1 core, 4 GB,
   30 min; no parser is loaded, so this is ample), task *i* curates G2 v3 shard *i* with `g3_curation_v2.py` into
   `PG/g3_v1/shards/`. `g3_v2/merge_g3_v1.py` (sha256 `a6921837ecf3`): re-derives completeness from each shard's own
   `g3_articles_NNN.jsonl` against its real G2 v3 shard file (every pmcid present exactly once, article status counts
   match, no pmcid shared across shards), checks every shard's report for the same script sha256 and input hashes and
   0 problems/0 unit errors, then concatenates the five record files and writes `g3_v1_summary.json`. Tested end to end
   on all 50 shards in a scratch folder: every shard 0 problems, the merge's own checks all passed, totals
   self-consistent (34,662 articles = the known corpus size; 6,466 excluded = the known item-6 count; 28,075 valid +
   121 invalid + 6,466 excluded = 34,662; the five output files' line counts match the summary's counts exactly) and
   match the known corpus totals; a corrupted report and a pre-existing output were each separately confirmed to stop
   the merge with no partial file left. Scratch output deleted; nothing produced in `PG/g3_v1/` yet. To submit on CSF
   (owner): `sbatch PG/scripts/g3_v2/submit_g3_v1.sh`, then after all 50 tasks `python PG/scripts/g3_v2/merge_g3_v1.py
   --nshards 50`.]*
3. The M1 input contract (JSONL and/or database, cousin–parent links as in §5). *[2026-10-09: M1 must also handle the
   flags in §6 item 8: content keys, the specificity of verb groups, and the article exclusions of item 6.]*
4. *[Added 2026-10-10, RL-121.]* Two small questions for the owner; the test script's current behaviour is given:
   - **Units that lose every parent under the hyphen-participle rule** (§6 item 8): 0.10–0.20% of units, about half of
     them headings or fragments. Left without a parent (option A of RL-111). To confirm.
   - **A score after a model name read as a version** (§4; P2's matcher): "ChatGPT 8.0 [7.0–10.0]" → `gpt_8`,
     "GPT 0.78 to 0.65" → `gpt_0_78`. 3 surface forms in 2 units of shards 0 and 20 (about 28,000 units); real versions
     written the same way ("ChatGPT 5.3", "Grok 4", "Gemma 2") are read correctly. Left as is.
   *[Both decided 2026-10-10 (owner) and implemented, RL-122: units without a parent are removed and articles without
   any parent marked invalid (§7.1 step 9); scores are no longer read as versions (§4).]*
5. *[Added 2026-10-10, RL-127.]* **Lemmatiser artefact on hyphen-joined compounds — a new question, not yet
   decided.** G2's transformer lemmatiser is inconsistent on out-of-vocabulary underscore-joined words (G2_PARSING.md):
   `fine_tuned`→`fine_tune` is right, `pre_trained`→`pre_traine` is not. Measured corpus-wide, not sampled: 35,201
   occurrences, 4,934 distinct (root, lemma) pairs differ; most of the frequent ones add a spurious trailing "e" to a
   verb that does not take one (`pre_trained` 2,198, `retrieval_augmented` 803, `ai_powered` 377, `board_certified`
   300, `question_answering` 221, `self_reported` 197, and about 25 more down to low frequency). Not a process-order
   bug: tested directly, delaying the lemma step relative to the hyphen-join does not help, and undoing the join to
   let the lemmatiser see the hyphenated form loses the clause structure entirely (the reason the join exists, item
   30(l)). **Recommended: a small, measured correction table in G3** (`lemma_of()`), built the way every other G3 rule
   was — sample the frequent pairs, judge by hand, hard cases, before/after counts — not a G2 rerun (deterministic;
   would reproduce the same lemma). Not built; waiting on the owner's decision to proceed.
   *[Corrected 2026-10-10, RL-128: the lemmatiser is not inconsistent but rule-based (an out-of-vocabulary word gets the
   first suffix rule, -ed/-ing → -e, -est → ""); 35,201 included correct cases. The real scope: 13,123 wrong suffix-rule
   lemmas on joined words (2,272 distinct), e.g. `pre_traine/P` 2,823 vs `pre_train/P` 305 in the G3 output; modifier
   uses are kept as written and unaffected. Proposed instead of a hand table: a head-word rule (prefix + the corpus's own
   lemma of the last part as a plain word, applied only where spaCy's suffix rule fired), with an exception for
   comparatives/superlatives (`second_best` → `second_good` otherwise).]*

## 9. Files

| File | sha256 (12) |
|---|---|
| `PG/scripts/g3_curation_test.py` | 2a5e01342bf4 |
| `PG/scripts/chunk_4h_hpc.py` (superseded, May) | dfc6cc5e9662 |
| `PG/nltk_abridged_stopwords_list.txt` (RDS stop list) | 2b6c7d9fdae9 |
| `tensor_data_staging/nltk_abridged_stopwords_list.txt` (toy stop list) | f5893e962fcd |
| `fullscale_pipeline/g3_modal_check/modal_rules.py` (rule of §6 item 3, test implementation) | 3f993e0ff164 |
| `fullscale_pipeline/g3_modal_check/hard_cases.json` (40 hard cases with expected results) | 25bc5059068e |
| `fullscale_pipeline/g3_word_lists/adverb_roles.tsv` (§6 item 4 adverb role table, 513 words) | 79466ff78dfe (f66d153f5efa before RL-121) |
| `fullscale_pipeline/g3_word_lists/build_adverb_roles.py` (builds the table from corpus attachment data in the untracked `diagnostics/g3_stoplist/adverb_attachments.json`) | f05285f9bb65 (4b6f4ae541e0 before RL-121) |
| `fullscale_pipeline/g3_phrasal_check/g3_q5fix.patch` (§6 item 5, diff against `g3_curation_test.py`) | d7c40968b4cb |
| `fullscale_pipeline/g3_phrasal_check/hard_q5.json` (24 constructed particle cases) | 69873d864054 |
| `fullscale_pipeline/g3_phrasal_check/hard_q5.py` (parses them; runs the RDS script and the test copies) | b1428685f3e2 |
| `fullscale_pipeline/g3_phrasal_check/q5_types.py`, `q5_compare.py` (particle types from G2; orig vs fix) | 5b105b053aed, 47dfbc208df2 |
| `fullscale_pipeline/g3_notonly/g3_q4_option2.patch` (§6 item 4 rule set incl. not-only/but-also option 2, diff against `g3_curation_test.py`) | 99ec4a8c84fa |
| `fullscale_pipeline/g3_notonly/hard_option2.json`, `hard_option2.py` (23 hard cases; runner) | 66b11a9a5144, b000aae90c5b |
| `fullscale_pipeline/g3_notonly/compare_cd.py` (drop in `curate()` vs strip afterwards) | fa7300335dc2 |
| `fullscale_pipeline/g3_llm_output/build_llm_output_units.py` (§6 item 7; also on `PG/g3_llm_output/`) | cbceaf621848 |
| `fullscale_pipeline/g3_llm_output/llm_output_units.csv` (162 units, 55 articles; also on `PG/g3_llm_output/`) | b71d5717e250 |
| `fullscale_pipeline/g3_scope_exclusions/build_scope_exclusions.py` (§6 item 6: builds the two lists; G3 correction hook) | 3a5de5167371 |
| `fullscale_pipeline/g3_scope_exclusions/scope_exclusions.csv` (6,466 excluded articles with reasons; also on `PG/`) | 5b71d844d9af |
| `fullscale_pipeline/g3_scope_exclusions/spelling_corrections.csv` (6 unit corrections; also on `PG/`) | d0a70f6c1a20 |
| `fullscale_pipeline/g3_scope_exclusions/test_corrections.py` (the 6 units through `g3_curation_test.py`) | dd18ca3aca3e |
| `fullscale_pipeline/g3_scope_exclusions/decisions/*.csv` (hand-checked lists), `evidence/*.py` (the scans) | see RUN_LOG RL-094 |
| `fullscale_pipeline/g3_audit_rules/hard_rules2.json`, `hard_rules2.py` (§6 item 8: 15 hard cases for the participle and cousin rules; runner) | 51f577bd1bfe, d77b9859b754 |
| `fullscale_pipeline/diagnostics/g3_audit/g3_rules2.py` (test copy with the item-8 participle and cousin rules; untracked working copy) | 53f3162ddb83 |
| **Test script v2 (§7.1)** | |
| `fullscale_pipeline/g3_v2/g3_curation_v2.py` (the G3 test script with every rule of §6 items 1–8; its report also records host/job/task since RL-124) | 0f3146266f26 |
| `fullscale_pipeline/g3_v2/hard/hard_versions_rl122.json` (8 hard cases for the score-not-a-version guard) | 302cf7da0fe4 |
| `fullscale_pipeline/g3_v2/check_csf_g3.sh` (CSF test, RL-123; mirrored to `PG/scripts/g3_v2/`) | baa9f73046ad |
| `fullscale_pipeline/g3_v2/submit_g3_v1.sh` (50-task SLURM array, RL-124; mirrored) | d0c8dad02783 |
| `fullscale_pipeline/g3_v2/merge_g3_v1.py` (checked merge, RL-124; mirrored) | a6921837ecf3 |
| `fullscale_pipeline/g3_v2/modal_merge.py` (§6 item 3: decision rule copied from `g3_modal_check/modal_rules.py`, edge-order fix, and the clause merge; imported by the script) | 75c7022ac187 |
| `fullscale_pipeline/g3_v2/reflexive_restore.py` (§6 item 8: builds the reflexive table; 43 min on incline) | 92fa72c383de |
| `fullscale_pipeline/g3_v2/measure_be.py` (§6 item 1: the measurement required before implementing passive "be") | 8cbfe1c89631 |
| `fullscale_pipeline/g3_v2/run_hard.py` (runs every hard-case set on the script; parses cached) | 80b9ca527b8f |
| `fullscale_pipeline/g3_v2/hard/hard_*.json` (13 sets, 199 cases, 175 with expectations) and `hard/parsed_cache.json` (their parses) | see RUN_LOG RL-112 to RL-117 |
| `PG/g3_reflexive/reflexive_restored.jsonl` (1,211 restored and re-parsed units; corpus text, RDS only) | d121a203daec |
| `fullscale_pipeline/g3_v2/runs/` (test outputs, corpus-derived; local only, excluded from git) | — |
