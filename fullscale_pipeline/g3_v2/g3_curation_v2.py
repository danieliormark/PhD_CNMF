"""
Stage G3 test script (2026-09-30): curation of the G2 v3 parse into the structures the matrix builder (M1) reads, with provenance.
Built on the toy-corpus version tensor_data_staging/toy_large/7.5postprocessing_4hbased_correct.py (the corrected one; chunk_4h_hpc.py is not
used: it skips a focal term's immediate clause, keeps a fringe pool nothing reads and strips digits from model names), plus PIPELINE.md item 29:

  (a) recursion: limit raised, every unit in its own try block; the input is read line by line, not through hgraph.search (where May's G3 died)
  (b) input: G2 v3 shard file g2_parsed_NNN.jsonl (one record per unit: uid, unit hash, sid, hash_final, text, main edge, lemma edges,
      atom-to-word positions); lemmas come from G2's own en_core_web_trf lemma edges, so G3 no longer runs spaCy
  (c) focal terms: found in the unit text with P2's own matcher (focal_terms.py, guard rules included), mapped to token positions, and every
      atom of the mention replaced by one canonical atom <canon>/Cp/focal; spelling variants, plurals and abbreviation/expansion pairs are
      merged, model versions are kept apart, sizes and snapshot dates dropped, ChatGPT-<version> merged into GPT-<version> (owner
      2026-09-30; canonical() below; the report lists every surface form and its canonical atom)
  (e) connective abbreviations (e.g., i.e., vs., cf., etc., viz., et al.) dropped
  (f) atoms without letters (numbers, signs, brackets, formula debris) dropped, except "%", which becomes the atom percent/C/en
  (g) be/have/do dropped only as auxiliaries (type Mv), kept as main verbs; modal verbs (type Mm, or can/may/... typed as another
      modifier) by --modals: keep (default, owner 2026-09-30: the modal is an atom of the compound verb group, "should be controlled" ->
      (dummy_sibling control/P/en should/M/en), in dummy_sibling and dummy_cousin alike; verbs such as allow/enable are ordinary
      predicates), tag (recorded on the verb group as "modality", atom dropped), drop. The stop list is not applied to main-verb be/have/do
      (the RDS list holds "been" and "am", the toy list "be") and does not decide the modal question ("will")
  types kept: Cc, Cp, Ca, Cm, M, P (Cm, a noun used as a modifier, "cancer research", was dropped by 7.5 and chunk_4h)
  (h) negation (not, no, never, n't, nor) kept whenever it is a modifier, including "no" as a determiner
  D4.1 (owner 2026-10-19, RL-149): formatting remnants inside words cleaned (tidy_word: ligatures, quotes, glued brackets,
      formula and table debris), flattened superscript citation numbers dropped (Ctx.citation_numbers), figure-panel legends
      opening with lower-case panel letters removed as non-prose (PANEL_LEGEND)
  D6 (owner 2026-10-19, RL-170): one parent per clause that governs a focal term; every argument holding a focal word is a focal_he;
      a clause holding a focal term is never a cousin (no rotated parents, no focal clause stripped into another parent's cousins)
  item 11 (owner 2026-10-21, RL-170): cousins scoped by sentence (linked to every parent of the sentence, identical content folded
      into one occurrence, a unit without a parent kept for its cousins when its sentence has a parent); typed links in
      g3_occurrences_NNN.jsonl: contains (child -> parent), co_occurs (cousin -> parent), embedded (parent -> outer parent)
  provenance: every structure has an id and hash sha1("<id>|<edge>")[:12] (the G2 unit-hash form): parent <uid>.P<k>, its children
      <uid>.P<k>.D (dummy_sibling), .F<j> (focal_he), .S<j> (sibling_he), cousins <uid>.K<j> with every parent of their sentence (also_in:
      the same cousin in the sentence's later units, with their atoms); every output atom lists the source atoms, words and token
      positions it comes from. Unit -> sentence (sid, hash_final) -> article (pmcid) as in G2.

    python g3_curation_test.py --selftest
    python g3_curation_test.py --shard 0 --limit 200 --outdir DIR [--modals tag|drop|keep] [--db]

Writes DIR/g3_test_NNN.jsonl (one record per unit with a parent, owner 2026-10-10, or with cousins of a sentence that has one, item 11), g3_noparent_NNN.jsonl (the removed units without a
parent), g3_articles_NNN.jsonl (per article: excluded, valid, or invalid_no_parent when no unit has a parent; M1 keeps only valid ones),
g3_nonprose_NNN.jsonl, g3_errors_NNN.jsonl, g3_occurrences_NNN.jsonl (item 11), g3_report_NNN.json and, with --db, g3_test_NNN.sqlite in the form
chunk12 reads (('source_core', 'pmcid::uid::hash', parent), ('source_periphery', 'pmcid::uid::hash', cousin)). Refuses to overwrite.
Checks at the end (exit 1 on any problem): input unit hashes, every output id and hash, every atom traced to a token position, every
cousin's parents present, and with --db the database read back in chunk12's way.
"""
import argparse, collections, contextlib, csv, hashlib, itertools, json, os, random, re, sys, time, unicodedata, urllib.parse

sys.setrecursionlimit(20000)
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
PP = "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing/"
POSITIONS = PG + "g2_v3_2_pos/"                                       # RL-186: exact token position of every atom occurrence (g2_positions.py)
SHARDS = PG + "g2_v3_2/shards/"                                       # RL-185: g2_v3_1 with the dash re-parses (RL-171: g2_v3_1)
STOPWORDS = PG + "nltk_abridged_stopwords_list.txt"
sys.path.insert(0, PP)
import focal_terms as focal_terms_v2                                   # noqa: E402  v2, loaded first: v3 extends it
import focal_terms_v3 as focal_terms                                   # noqa: E402  RL-171: P2's v3 matcher (v2 hits unchanged + v3 additions),
                                                                       # the matcher of the merged corpus; the single source of the focal terms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal_merge                                                     # noqa: E402  §6 item 3 (decision rule and edge rewrite)
from graphbrain import hedge                                           # noqa: E402

STOP = {w.strip() for w in open(STOPWORDS, encoding="utf-8") if w.strip()}
CUSTOM_NOISE = {'such', 'other', 'many', 'some', 'any', 'own', 'same', 'few', 'several', 'this', 'that', 'these', 'those', 'whilst'}
NEGATION = {'not', 'no', 'never', "n't", 'nor'}
AUXILIARIES = {'be', 'have', 'do'}
MODAL_WORDS = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ca', 'wo', "'ll", "'d"}   # 'd: RL-180   # ca/wo: can't, won't
CONNECTIVES = {'e.g.', 'eg', 'e.g', 'i.e.', 'ie', 'i.e', 'vs.', 'vs', 'v.', 'cf.', 'cf', 'etc.', 'etc', 'viz.', 'viz', 'et', 'al.', 'al',
               'et al.', 'resp.'}
PHRASAL_PARTICLES = {'out', 'up', 'down', 'in', 'on', 'off', 'over'}
ACTIVE_STOP = (STOP | CUSTOM_NOISE) - NEGATION   # §6 item 5: particles survive only when fuse_phrasal attaches them to a verb
ONLY_MODE = os.environ.get('ONLY_MODE', 'vg2')   # §6 item 4: 'only' joins the verb group only before a predicate
Q4 = os.environ.get('Q4', '1') == '1'             # §6 item 4 rule set (switch kept for comparison runs only)
NOTONLY_MODE = os.environ.get('NOTONLY_DROP', 'paired')  # §6 item 4 (owner 2026-10-09, option 2): not_only and but_also dropped only as a pair; 0 keep all, 1 drop all (tests only)
NOTONLY_DROP = NOTONLY_MODE == '1'


NOT_ONLY_RX = re.compile(r"(?i)(?:\bnot|n't)\s+(?:only|just|merely|simply|solely)\b")
PARTNER_RX = re.compile(r"(?i)\bbut\b|\balso\b|\bas well\b")


def q4_paired(ctx, p):
    """the 'not only/just/...' whose 'not' is token p has a partner ('but', 'also', 'as well') later in the unit, before
    the next ';' (a ';' starts an independent clause whose 'but'/'also' is not this construction's partner). Works on the
    text through token spans (punctuation has no token position; 'n't' and the 'not' of 'cannot' are found in order)."""
    sp = ctx.spans.get(p + 1) or ctx.spans.get(p)
    if sp is None:
        return False
    after = ctx.text[sp[1]:].split(";", 1)[0]
    return bool(PARTNER_RX.search(after))


def q4_also_paired(ctx, p):
    """the 'but (...) also' whose 'also' is token p closes a 'not only/just/...' standing before its 'but' in the unit,
    after the last ';' -- the additive pair; a bare 'X but also Y' (no not-only) is contrastive and is not paired"""
    q = next((q for q in range(p - 1, p - 4, -1) if ctx.words.get(q) == "but"), None)
    sp = ctx.spans.get(q) if q is not None else None
    if sp is None:
        return False
    before = ctx.text[:sp[0]].rsplit(";", 1)[-1]
    return bool(NOT_ONLY_RX.search(before))


Q4_CONNECTIVES = set("however therefore additionally furthermore moreover thus hence consequently nevertheless nonetheless "
                     "accordingly conversely meanwhile likewise indeed respectively although thereby though since whereas".split())
Q4_OPENER_DROP = set("finally first second third fourth fifth sixth seventh firstly secondly thirdly fourthly lastly last next "
                     "overall similarly specifically notably yet instead together collectively rather besides herein hereafter "
                     "briefly concretely relatedly correspondingly complementarily namely plus essentially basically "
                     "subsequently thereinto regardless otherwise".split())
Q4_NOT_ONLY = {"only", "just", "merely", "simply", "solely"}
Q4_DEGREE = {"least", "not", "about", "nearly", "almost", "just", "equally", "roughly", "approximately", "quite", "half", "twice"}
if ONLY_MODE != 'drop':
    ACTIVE_STOP = ACTIVE_STOP - {'only'}
# §6 item 4: the adverb role table (built by g3_word_lists/build_adverb_roles.py, owner-reviewed), RDS copy for CSF
ADVERB_ROLES_PATH = PG + "g3_word_lists/adverb_roles.tsv"
ADVERB_ROLES_SHA = "79466ff78dfe"
ADV_ROLE = {r["word"]: r["action"] for r in csv.DictReader(open(ADVERB_ROLES_PATH, encoding="utf-8"), delimiter="\t")}
# the 12 "particle" rows (owner 2026-10-09, option 3, written into the table): direction words fuse with their verb like the
# item-5 particles ("look_ahead", "move_forward", "leave_behind") and otherwise stay a modifier in their phrase ("a step
# forward"); time uses (before, throughout) dropped like the table's time adverbs; around/under kept only before a number
# (the comparison-word rule keeps them there), otherwise dropped; despite/beside dropped
DIRECTION_PARTICLES = {'forward', 'forwards', 'ahead', 'behind', 'back', 'away'}
TABLE_ACTION_CODE = {"fuse with its verb like a phrasal particle, else leave in its phrase": 'direction',
                     "keep before a number (numbers rule), else drop": 'drop'}
assert {w for w, a in ADV_ROLE.items() if a == "fuse with its verb like a phrasal particle, else leave in its phrase"} \
    == DIRECTION_PARTICLES, "adverb table out of step with §6 item 8 (direction words)"
ADV_ROLE = {w: TABLE_ACTION_CODE.get(a, a) for w, a in ADV_ROLE.items()}
CLAUSE_END = re.compile(r"^\s*(?:[.,;:)]|$)")
# §6 items 6 and 7: articles out of scope, units reproducing LLM output, word-for-word spelling corrections; RL-171: the
# lists rebuilt for the merged corpus in PG/g3_inputs_v3_1/ (fullscale_pipeline/g3_inputs_v3_1/; the RL-130 lists stay in
# g3_scope_exclusions/, g3_llm_output/, g3_reflexive/)
INPUTS = PG + "g3_inputs_v3_1/"
SCOPE_EXCLUSIONS = INPUTS + "scope_exclusions.csv"
SPELLING_CORRECTIONS = INPUTS + "spelling_corrections.csv"
LLM_OUTPUT_UNITS = INPUTS + "llm_output_units.csv"
REFLEXIVE_TABLE = INPUTS + "reflexive_restored.jsonl"               # §6 item 8, built by g3_v2/reflexive_restore.py
TERM_REJECTIONS = INPUTS + "term_rejections.tsv"                    # RL-171: focal hits Stage B rejected (sentence / article)
G4_DECISIONS = INPUTS + "g4_decisions.csv"                          # RL-172: group-4 false hits (reject) and glued citations (map)
STATEMENT_SENTENCES = INPUTS + "statement_sentences.csv"            # RL-172: AI-use statements found by the second check
COMPOUND_FORMS = INPUTS + "compound_forms.tsv"
HUB_FAMILIES = INPUTS + "hub_families.txt"                          # RL-185: model families for lower-case hub spellings                      # RL-180: spellings of one compound -> its most frequent form
# §8 item 5 (owner 2026-10-10, RL-129): spaCy's rule lemmatiser gives an underscore-joined (out-of-vocabulary) word its first
# suffix rule unchecked (pre_trained -> pre_traine, question_answering -> question_answere, second_best -> second_b). Where it
# changed a joined word, the lemma is rebuilt as the part before the last "_" plus the lemma G2's parse gives the last part as a
# plain word (table built by g3_word_lists/build_head_lemmas.py); where that plain lemma is an irregular form not sharing the
# word's first three letters (best -> good), the joined word is kept as written; with no plain-word lemma, spaCy's stays.
HEAD_LEMMAS = PG + "g3_word_lists/head_lemmas.tsv"
JOINED = re.compile('(.+[_\u2010\u2011\u2013])([^_\u2010\u2011\u2013]+)')   # joined by G2 ("_") or a Unicode hyphen G2 left alone
HEAD_LEMMA = {(r["word"], r["type"]): r["lemma"] for r in csv.DictReader(open(HEAD_LEMMAS, encoding="utf-8"), delimiter="\t")}
INPUT_SHA = {SCOPE_EXCLUSIONS: "838f50923616", SPELLING_CORRECTIONS: "f7a5277bd31e", LLM_OUTPUT_UNITS: "cb79e798185f",   # RL-174 scope list (RL-172 f50826614f4a; RL-171 0b4ba5c2b2b1; RL-130: 5b71d844d9af, d0a70f6c1a20, b71d5717e250)
             ADVERB_ROLES_PATH: ADVERB_ROLES_SHA}
INPUT_SHA[REFLEXIVE_TABLE] = "573be717ee21"                                # RL-130: d121a203daec
INPUT_SHA[HEAD_LEMMAS] = "34cee7ee73d0"
INPUT_SHA[STOPWORDS] = "2b6c7d9fdae9"
INPUT_SHA[PP + "focal_terms.py"] = "892f98bb986e"     # P2's matcher with the RL-095 hyphen pattern (acts in G3 only)
INPUT_SHA[PP + "focal_terms_v3.py"] = "49f38796e655"                      # RL-171: the v3 matcher and its group-4 rule
INPUT_SHA[focal_terms.RULES + "focal_group4_rule.py"] = "70d88dcdf8e4"
INPUT_SHA[TERM_REJECTIONS] = "1dcd13352f6a"
INPUT_SHA[G4_DECISIONS] = "08b623b537a7"                     # RL-181: RegulatoryGPT kept (RL-174 7e3732cb0cdb, RL-173 05a32cbad7ce, RL-172 326130aeba60)
INPUT_SHA[STATEMENT_SENTENCES] = "bfcb97d7b479"
INPUT_SHA[HUB_FAMILIES] = "c1cb350b89b0"                    # RL-185
INPUT_SHA[COMPOUND_FORMS] = "1d1bc8c84b8a"                 # RL-180 (built from the RL-178 output)
# §6 item 8 (owner 2026-10-09): junk atoms, labels, numbers
LAYOUT_WORDS = {'table', 'tables', 'tab', 'fig', 'figs', 'figure', 'figures', 'appendix', 'appendices', 'supplementary',
                'panel', 'panels', 'equation', 'equations', 'eq', 'eqs'}
# RL-120 (owner 2026-10-10: remove what is not plain text): more references to non-text objects ("Textbox 3", "Online
# Resource 5", "Supplementary Note 3", "Box 1", "Algorithm 1"), dropped with their number like "Table 2"
LAYOUT_WORDS |= {'textbox', 'textboxes', 'box', 'boxes', 'resource', 'resources', 'note', 'notes', 'algorithm',
                 'algorithms', 'scheme', 'schemes', 'chart', 'charts', 'listing', 'listings', 'file', 'files', 'video', 'videos'}
LAYOUT_PREFIX = {'supplementary', 'supplemental', 'online', 'additional', 'extended'}
NUM_LABEL_WORDS = LAYOUT_WORDS | {'section', 'sections', 'question', 'questions', 'step', 'steps', 'phase', 'phases',
                                  'chapter', 'chapters', 'item', 'items', 'version', 'versions', 'experiment', 'study'}
# RL-120: index numbers on things in the prose ("Topic 7", "Surgeon 2", "group 0", "round 1"): the word is plain text and
# stays, the number is a pointer and goes
INDEX_NOUNS = set('topic tier round group reader review variant arm cohort wave session iteration scenario case task prompt '
                  'level model rater reviewer annotator surgeon participant patient respondent physician expert judge '
                  'evaluator student condition dataset run challenge rule hypothesis aim objective criterion'.split())
NUM_LABEL_WORDS |= INDEX_NOUNS | {w + 's' for w in INDEX_NOUNS}
UNIT_AFTER = re.compile(r'^\s*(?:%|(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?|percent|times?|fold)\b)', re.I)
LABEL_RX = re.compile(r'^(?:s?\d+(?:\.\d+)*[a-z]?|[a-z]|[ivx]+)$')
NUMBER_RX = re.compile(r'^\d+(?:[.,]\d+)*$')
SMALL_NUMBERS = 'zero one two three four five six seven eight nine ten'.split()
NUMBER_WORDS = set(SMALL_NUMBERS) | set('eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty '
                                       'thirty forty fifty sixty seventy eighty ninety hundred thousand million billion '
                                       'dozen'.split())
STAT_BEFORE = re.compile(r'(?i)(?:\bp\s*[<=>≤≥]\s*|\bci\b[\s:,]*|±\s*|\b(?:or|hr|rr|r|n|t|f|sd|se|iqr|auc|κ|kappa|α|β)\s*[=:<>≤≥]\s*|[=<>≤≥]\s*'
                         r'|\b(?:sd|se|iqr|u|z)\s+|\bet al\.?,?\s*\(?)$')     # RL-120: "SD 18.11", "U 116", "et al. (2020)"
COMPARE_WORDS = {'over', 'under', 'above', 'below', 'up', 'more', 'less', 'fewer', 'least', 'most', 'nearly',
                 'approximately', 'about', 'around', 'almost', 'roughly'}
URL_RX = re.compile(r'^url\d{8}$')
SIZE_RX = re.compile(r'^\d+(?:\.\d+)?[bm]$')                     # model sizes left beside a name: 8b, 13b, 70b
# a number followed by a bracketed interval is a score, not a version ("ChatGPT 8.0 [7.0–10.0]"; owner 2026-10-10), and so is a
# number below 1 ("GPT 0.78 to 0.65"). Not used: a following "to <number>" (real versions start ranges: "from ChatGPT 3.5 to 4o"),
# a bare bracketed number (a citation), a score in round brackets (real versions take one: "Grok 3 (0.78)")
VALUE_AFTER = re.compile(r'^\s*\[\s*[−-]?\d+(?:\.\d+)?\s*(?:[–—-]|to\b|,)')
CITE_LIST = re.compile(r'(?<=[A-Za-z)\]%] )(\d{1,3}(?:\s?[,–-]\s?\d{1,3})*)(?=\s+[.,;:)\]](?:\s|$))')     # D4.1
CITE_VALUE_WORDS = set('of to was were is are be been equals equal than by cost costs from at out times about approximately '
                       'around nearly over under with and or between top only all first last every each per page pages line '
                       'lines version n'.split())
SCORE_TOKEN = re.compile(r'^0\.\d')
GENERIC_NAMES = {'llm', 'language_model', 'transformer_model', 'small_language_model', 'chatbot', 'foundation_model'}   # RL-179
SNAPSHOT_CODE = re.compile(r'^(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])$')              # RL-177: GPT-4 0613, 1106, 0125 (month + day)
VERSION_TOKEN = re.compile(r'^(?:v\d+(?:\.\d+)*|[ro]\d+|\d(?:\.\d{1,2})?[a-z]{0,2})$', re.I)
HYPHEN_CHARS = re.compile('[\u2010\u2011\u2013\u2014]')


# D4.1 (owner 2026-10-19, RL-149): formatting remnants inside words. PDF ligatures are written out; curly quotes and the prime
# used as an apostrophe become "'"; a clitic left as a word of its own ("’s", "’m", "’ve", from a curly apostrophe the
# tokeniser did not split; as a main verb it becomes be / have / will / would) and a possessive or clitic glued to a
# word go, as the straight forms do; formula and table
# debris (|, =, ^, ~, <, >, mathematical operators; "o(n2", "p(x"; a Greek letter or prime with at most one Latin letter,
# "xα", "fθ", "δr"; a Greek letter inside brackets) is dropped; brackets glued to words go ("(AI)-based" -> ai_based,
# "author(s)" -> author, "Lucene(a)" -> lucene, "[18F]FDG" -> 18f_fdg); "+" and "·" are plain text (C++, Command R+,
# CNN+LSTM, DALL·E) and stay; Aβ stays
LIGATURES = {'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'st', 'ﬆ': 'st'}
SINGLE_QUOTES = re.compile("[‘’‚‛`´′]")
# RL-180 (owner 2026-10-22): character clean-up of word atoms, applied after the model-name rules (focal atoms are not touched)
INVISIBLE = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]')            # zero-width characters, soft hyphen
GREEK = re.compile('[\u0370-\u03ff\u00b5\u2206]')                      # with ∆ (increment sign)
GREEK_SPELLED = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon', 'ϵ': 'epsilon', 'ζ': 'zeta', 'η': 'eta',
                 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu', 'µ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'π': 'pi',
                 'ρ': 'rho', 'σ': 'sigma', 'ς': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi', 'ϕ': 'phi', 'χ': 'chi', 'ψ': 'psi',
                 'ω': 'omega'}
GREEK_FIRST_NAMES = {'α_helix', 'π_stacking', 'nf_κb', 'ifn_γ', 'α_kg', 'β_chain', 'α_lipoic', 'α_ketoglutaric', 'α_pinene',
                     'γh2ax', 'αβtcr', 'αβtcr_peptide_hla_i', 'αcdr3', 'βcdr3', 'αmhc', 'βmhc', 'tgfβr1', 'αbgtx', 'αcbtx',
                     'δorf1', 'δorf3', 'δδg'}   # Greek letter first or inside, but a biomedical or chemical name (read, RL-180)
GREEK_MICRO = {'μct': 'microct', 'µct': 'microct', 'μema': 'microema', 'µema': 'microema'}
GREEK_END_NOTATION = {'scoreβ', 'oncologistχ', 'aucμ', 'prθ', 'casφ'}    # Greek letter last, but a variable (read, RL-180)
COMPOUND_EXCEPT = {'pro_state', 'co_pilot', 're_agent', 'co_star', 're_present', 're_turn'}   # RL-180: read, a different word
WORD_MAP = {'w/o': 'without'}
# RL-186 (owner 2026-10-23): words of opposition, absence and comparison are kept as atoms (the parser types them as
# connectors, T/J/B, which G3 drops), so "GPT-4 versus clinicians" and "GPT-4 with clinicians" are different word sets;
# versus/vs/against are one form. "in contrast" (owner) is one atom, with the verb group when it opens a clause ("In
# contrast, GPT-4 failed") and with its object otherwise ("in contrast to rule-based systems"). Not kept: between, of,
# with, in, for (neutral); rather and instead were kept already (modifiers).
CONTRAST_WORDS = {'without': 'without', 'versus': 'versus', 'vs': 'versus', 'vs.': 'versus', 'against': 'versus',
                  'than': 'than', 'unlike': 'unlike'}
CONTRAST_ATOMS = {f"{w}/M/en" for w in set(CONTRAST_WORDS.values())}
# They are carried along, but they do not make a group of their own: a group of only these words is empty, and they do
# not stop a verb group ("limit rather than"), so the clause structure stays as without them (RL-186 trial: 226 lone
# hyperedges such as (cousin_he than), whose object was dropped)
MATH_CHARS = re.compile('[∥⟨⟩∝∑∏√∫≈≠∈∉⊂⊆∪∩⊕⊗∀∃∇∂]')                     # RL-185: formula debris in an atom
FORMULA_SPAN = re.compile(r"\b[A-Za-z][A-Za-z0-9]{0,8}\((?:[a-z][a-z0-9]?)(?:,\s*[a-z][a-z0-9]?)+\)"
                          r"|\S*[⟨⟩∥∝∑∏√∫∈⊂⊆⊕⊗]\S*|\b\S+\s*[<>≤≥=]\s*\S+|\b\S+\s+at most\s+\S+")
LATIN_PLAIN = str.maketrans({'ı': 'i', 'ł': 'l', 'ø': 'o', 'đ': 'd', 'ð': 'd', 'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'þ': 'th'})   # no NFKD form
CLITIC = re.compile(r"^'(?:s|m|d|t|ve|re|ll)\d*$")
CLITIC_TAIL = re.compile(r"(?<=[a-z0-9])'(?:s|m|d|ve|re|ll)$")
CLITIC_VERB = {"'s": 'be', "'m": 'be', "'re": 'be', "'ve": 'have', "'ll": 'will', "'d": 'would'}
FORMULA_SIGNS = re.compile(r"[|=^~<>∑∏√∂∫×÷∈∀∃≠±∗∘⊂⊆∪∩→←↔⇒⊕⊗⋅∣¬∼]")
SYMBOL_MARKS = re.compile("[Ͱ-Ͽ″‴̂]")
FORMULA_CALL = re.compile(r"^[a-z]{1,2}\(")                                   # o(n2, p(x, f(x
GROUP_NAME = re.compile(r"^([a-z]{1,2})\((\d)\)[-_]([a-z]{3,}.*)$")          # se(3)-equivariant -> se3_equivariant
ENUM_TAIL = re.compile(r"\((?:[a-z]|\d{1,2})\)?$")                            # lucene(a, icc(2, 3(a
PLURAL_TAIL = re.compile(r"\((?:s|es)\)?$")                                   # author(s
BRACKETS = "()[]{}"
DQUOTES = '“”„‟"'


def tidy_word(w):
    """D4.1: the word with formatting remnants removed, or None when it is formula or table debris (applied in clean_root)"""
    for k, v in LIGATURES.items():
        w = w.replace(k, v)
    w = SINGLE_QUOTES.sub("'", w.replace(';', ';').replace('\\', ''))   # U+037E: the Greek question mark (";")
    if CLITIC.match(w) or FORMULA_SIGNS.search(w):
        return None
    w = CLITIC_TAIL.sub('', re.sub(r"(?<=[a-z])'s\d+$", "'s", w))       # hsu’s22 (citation 22) -> hsu
    g = GROUP_NAME.match(w)
    if g:
        w = f"{g.group(1)}{g.group(2)}_{g.group(3)}"
    elif FORMULA_CALL.match(w):
        return None
    if SYMBOL_MARKS.search(w) and (any(c in w for c in BRACKETS) or 'approximately' in w):
        return None                                                            # "μθ(xt", "κapproximately" (κ ≈)
    w = PLURAL_TAIL.sub('', w)
    w = ENUM_TAIL.sub('', w)
    w = re.sub(f"[{DQUOTES}]" + r"\d+$", '', w)                                 # perfect”15 -> perfect (a citation)
    w = re.sub(r"[)\]}]-|-[(\[{]", '_', w)                                      # (AI)-based -> ai_based
    w = re.sub(r"(?<=[a-z0-9])[()\[\]{}" + DQUOTES + r"]+(?=[a-z0-9])", '_', w)    # godel(grounded, models“bert
    w = re.sub(r"[()\[\]{}" + DQUOTES + "]", '', w)
    w = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", '', w).rstrip('−')
    if SYMBOL_MARKS.search(w) and len(re.sub(r"[^a-z]", '', w)) <= 1 and not w.startswith('aβ'):
        return None
    return re.sub(r'_+', '_', w).strip('_')


def clean_root(word):
    """§6 item 8: an atom's word cleaned of glued citation digits ('self_education,2', 'nccn)3'), edge punctuation, and with a
    Unicode hyphen inside it written as '_'; then D4.1's tidy_word; None if no letter is left"""
    w = HYPHEN_CHARS.sub('_', INVISIBLE.sub('', word))
    w = re.sub(r'[,;.)\]]+\d+$', '', w)
    w = tidy_word(w)
    if w is None:
        return None
    w = w.strip('.,;:()[]\'"-_•·')                                         # RL-180: also hanging hyphens and bullets
    return w if re.search(r'[a-z]', w) else None


def norm_word(w, forms=None):
    """RL-180: final form of a word atom's root (after clean_root): Greek letters spelled out in names and the atom dropped (None)
    when it is notation; accents removed; a hyphen inside written as '_'; spellings of one compound merged into its most frequent
    form (forms: {letters without joiners: form}, from compound_forms.tsv) except the read exceptions"""
    if GREEK.search(w):
        if w in GREEK_MICRO:
            w = GREEK_MICRO[w]
        elif w in GREEK_FIRST_NAMES or ((re.search(r'[a-z0-9][\u0370-\u03ff\u00b5]+\d*$', w) or w.startswith('aβ'))
                                        and w not in GREEK_END_NOTATION and not re.search(r'[,−+*;]', w)):
            w = ''.join(GREEK_SPELLED.get(c, c) for c in w)
        else:
            return None
        if GREEK.search(w):
            return None
    w = ''.join(c for c in unicodedata.normalize('NFKD', w) if not unicodedata.combining(c)).translate(LATIN_PLAIN).lower()
    w = re.sub(r'[\s〈〉⟨⟩‹›«»†‡?]', '', w)                                  # no spaces (h˜i), angle brackets (〈sep〉), daggers
    w = re.sub(r':\d+(?:\.\d+)?[bmk]$', '', w).strip('/')                    # size tags (vision:90b), slashes at the edges
    w = re.sub(r'_+', '_', w.replace('-', '_')).strip('_')
    if forms and w not in COMPOUND_EXCEPT:
        w = forms.get(w.replace('_', ''), w)
    return w or None


# §6 item 8: units that parse to a parent but are not prose; glossary parents
END_PUNCT = re.compile(r'[.?!;:"”’\')\]]\s*$')
TITLE_SMALL = {'of', 'the', 'and', 'in', 'for', 'with', 'on', 'to', 'a', 'an', 'by', 'from', 'vs', 'versus', 'or', 'at', 'as', 'via'}
SECTION_NO = re.compile(r'^\s*\d+\.\d+(?:\.\d+)*\.?\s+[A-Z]')
REFERENCE_ENTRY = re.compile(r"^\s*\[?\d+\]?\.?\s+(?:(?:[A-Z]\.\s?){1,3}\s*[A-Z][A-Za-z'\-]+|[A-Z][A-Za-z'\-]+,?\s+(?:[A-Z]\.\s?){1,3})[,.]")
PSEUDOCODE = re.compile(r'←|//|:=')
PANEL_CAPTION = re.compile(r'^[A-H]\s+[A-Z][a-z]+[^;]*;\s*[B-H]\s+[A-Z][a-z]+')
# D4.1 (owner 2026-10-19): a figure-panel legend in the journal style that opens with lower-case panel letters and a comma
# ("a, Enformer is trained to …", "d, e Heatmaps showing …", "a, b: rows with matching superscripts …"); all 125 such units
# of the G3 v1 output were read: figure legends, and 3 table notes
PANEL_LEGEND = re.compile(r'^[a-h]\s*,\s|^[a-h](?:\s*,\s*[a-h])+\s*[:\s]')
GLOSS_VERBS = {'equal', 'indicate', 'denote', 'stand', 'refer'}


# RL-181 (owner 2026-10-23): sentences that define the symbols of a removed formula ("where μLLM is the estimated grand mean")
SYM_LABELS = {'text', 'img', 'image', 'final', 'low', 'high', 'input', 'output', 'query', 'local', 'global', 'mask', 'judge',
              'tokens', 'token', 'matched', 'ground', 'train', 'test', 'max', 'min', 'mean', 'total', 'base', 'head', 'enc', 'dec',
              'cls', 'question', 'answer', 'pred', 'true', 'model', 'loss', 'layer'}
SYM_OPENERS = {'this', 'that', 'these', 'those', 'here', 'there', 'then', 'thus', 'hence', 'they', 'one', 'two', 'all', 'each',
               'both', 'such', 'also', 'its', 'our', 'his', 'her', 'the', 'and', 'but', 'for', 'not', 'now', 'yet', 'who', 'how',
               'why', 'see', 'use', 'data', 'fig', 'table', 'next', 'last', 'first', 'second', 'third'}
SYM_STOP = {'a', 'i', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or',
            'so', 'to', 'up', 'us', 'we', 'ai', 'ml', 'dl'}


def is_symbol(tok, loose=True):
    """a formula symbol: a Greek letter (with a label), one or two letters with digits (L, d, xi, M1, SE1), a capital with a short
    or known label (Fesm, Xtext, Hfinal); loose (after 'where'/'let'): also a capital + letters + digits (HC2, Lk1)"""
    tok = tok.strip("(),.;:′'")
    if re.match(r'^[\u0370-\u03ff\u2206]', tok):
        return True
    if re.fullmatch(r'[A-Za-z]{1,2}\d*', tok):
        return tok.lower() not in SYM_STOP
    if tok.lower() in ACTIVE_STOP or tok.lower() in SYM_OPENERS:
        return False
    m = re.fullmatch(r'([A-Z])([a-z]+)', tok)
    if m:
        return len(m.group(2)) <= 3 or m.group(2) in SYM_LABELS
    return loose and bool(re.fullmatch(r'[A-Z][A-Za-z]{0,3}\d+|[A-Z][a-z]?[A-Z][a-z]{0,6}', tok))


NOTATION_START = re.compile(r'^\W*(where|let)\s+(\S+?)[,\s]', re.I)
NOTATION_HERE = re.compile(r'^\W*here\s+(\S+?)[,\s]', re.I)
NOTATION_LOST = re.compile(r'^\W*(?:(?:here|where)\s*,?\s+(?:denotes?|represents?|refers to)\b|where\s*,?\s+(?:is|are)\b(?!\s+(?:a|an|the|some|no|it|we)\b))', re.I)
NOTATION_SUBJ = re.compile(r'^\W*(\S+?)\s+(?:denotes|represents|is defined as|indicates(?!\s+that))\b')
NOTATION_CLAUSE = re.compile(r'\bwhere\s+(\S+?)\s+(?:is|are|denotes?|represents?|indicates?(?!\s+that)|refers to)\b'
                             r'|\bdenoted\s+(?:by|as)\s+(\S+?)[\s.,)]|\blet\s+(\S+?)\s+(?:be|denote)\b'
                             r'|[,;]\s*(\S+?)\s+(?:denotes?|represents?)\b', re.I)


def notation_definition(text):
    """RL-181: the unit is a definition of formula symbols (where/let + symbol, here + a short symbol, or symbol + denotes)"""
    m = NOTATION_START.match(text)
    if m:
        return is_symbol(m.group(2))
    if NOTATION_LOST.match(text):                                      # 'Here, denotes ...': the symbol went with the formula
        return True
    m = NOTATION_HERE.match(text)
    if m:
        tok = m.group(1).strip("(),.;:")
        return bool(re.fullmatch(r'[A-Za-z]{1,2}\d*|[\u0370-\u03ff\u2206]\w*', tok)) and is_symbol(tok)
    m = NOTATION_SUBJ.match(text)
    return bool(m) and is_symbol(m.group(1), loose=False)


def defined_symbols(text):
    """RL-181: symbols a definition clause names inside other prose (', where L is the length of ...', 'denoted by EG')"""
    out = set()
    for m in NOTATION_CLAUSE.finditer(text):
        tok = next(g for g in m.groups() if g)
        if is_symbol(tok):
            out.add(tok.strip("(),.;:′'").lower())
    return out


def nonprose_kind(text):
    """§6 item 8: 'heading', 'reference', 'pseudocode', 'panel_caption', (RL-181) 'notation' or None"""
    words = re.findall(r"[A-Za-z][A-Za-z\-_']*", text)
    if PSEUDOCODE.search(text):
        return 'pseudocode'
    if REFERENCE_ENTRY.match(text):
        return 'reference'
    if PANEL_CAPTION.match(text) or PANEL_LEGEND.match(text):
        return 'panel_caption'
    if len(words) <= 20 and (SECTION_NO.match(text) and (not END_PUNCT.search(text) or text.rstrip().endswith('?'))):
        return 'heading'
    content = [w for w in words if w.lower() not in TITLE_SMALL]
    if len(words) <= 20 and content and not END_PUNCT.search(text) and sum(w[0].isupper() for w in content) / len(content) >= 0.6:
        return 'heading'
    if notation_definition(text):
        return 'notation'
    return None


def content_key(atoms):
    """§6 item 8: the content key, the same for the same set of words whatever the role or place (sha1, 16 hex)"""
    return hashlib.sha1(" ".join(sorted(atoms)).encode("utf-8")).hexdigest()[:16]


KEEP_TYPES = ('Cc', 'Cp', 'Ca', 'Cm', 'M', 'P')    # Cm (noun used as a modifier: "cancer research") added; 7.5 and chunk_4h dropped it
KINDS = ('parent', 'dummy_sibling', 'focal_he', 'sibling_he', 'dummy_cousin', 'cousin_he')

# ------------------------------------------------------------------ focal terms: canonical atoms
SYNONYMS = {   # merged after spelling normalisation; everything else keeps its own canonical form
    'llms': 'llm', 'large_language_model': 'llm', 'large_language_models': 'llm',
    'language_models': 'language_model', 'transformer_models': 'transformer_model',
    'bidirectional_encoder_representations_from_transformers': 'bert',
    'chat_gpt': 'chatgpt', 'google_gemini': 'gemini', 'google_bard': 'bard', 'mistral_ai': 'mistral',
    'gpt_35': 'gpt_3_5',            # RL-172: Azure's name for GPT-3.5 (GPT-35-turbo)
}
VERSIONED = r'(chatgpt|gpt|llama|qwen|claude|gemini|grok|palm|gemma|mistral|deepseek)'
VERSION_SUFFIX = re.compile(r'^[-‐– ]?(?:v?\d(?:\.\d{1,2})?[a-z]?|\d+(?:\.\d+)?[bm])$', re.I)   # one-digit major: "GPT-44" is GPT-4 + citation 4
SIZE_IN_NAME = {'esm-1b', 'esm-msa-1b', 'protst-esm-1b'}       # focal-list names whose "1b" is not a model size


# RL-172 (owner 2026-10-22): a checkpoint name is written like its prose form, family + version: the configuration after it
# (size, base/large, cased/uncased, multilingual, language, instruct/it/chat, snapshot codes, '-based') is dropped, as sizes and
# dates already were (2026-09-30): bert-base-uncased -> bert, gemma-3-12b-it -> gemma_3, llama-7b -> llama. Only when the first
# part after the name and its versions is such a setting (BERT-CRF stays).
# RL-177 (owner 2026-10-22, "different models of different power can be treated differently"): product tiers are part of the
# name and kept (gemini-2.5-pro -> gemini_2_5_pro, gpt-4o-mini -> gpt_4o_mini, GPT-4 Turbo -> gpt_4_turbo); Claude is written
# family + tier + version whatever the order (Claude 3.5 Sonnet, Claude Sonnet 3.5 -> claude_sonnet_3_5); the fold applies to
# every model name, not only 15 families (xlm-roberta-base -> xlm_roberta, deberta-v3-large -> deberta_v3), also to settings
# glued to a BERT-family name (BERTbase, RoBERTalarge -> bert, roberta); a leading 'meta' before llama is dropped (vendor).
CKPT_TIER = {'sonnet', 'opus', 'haiku', 'pro', 'flash', 'lite', 'mini', 'nano', 'turbo', 'ultra', 'max'}
CKPT_SETTING = re.compile(r'^(?:base|large|small|tiny|medium|xl|xxl|xlarge|xxlarge|uncased|cased|multilingual|instruct|it|chat|exp|'
                          r'preview|latest|based|hf|\d+(?:\.\d+)?(?:x\d+)?[bmk]|e\d+b|\d{3,}|(?:19|20)\d\d\S*)$')
CKPT_VERSION = re.compile(r'^(?:v?\d{1,2}(?:\.\d{1,2})*[a-z]{0,2}|[ro]\d+)$')
CKPT_LEAD = {'base', 'large', 'small', 'medium', 'cased', 'uncased'}    # RL-184 (owner): a setting written before the family;
# not tiny (TinyBERT is a model), instruct (InstructGPT) or multilingual (Multilingual BERT)
CKPT_GLUED_SIZE = re.compile(r'^(mistral|mixtral|gemma|llama|qwen|falcon|phi|deepseek|vicuna|baichuan)(\d+(?:\.\d+)?[bm])$')   # RL-184
CKPT_GLUED = re.compile(r'^(.*b(?:er)?t|.*berta)(base|large|uncased|cased)$')     # bertbase, robertalarge, phobertbase, albertbase


def fold_checkpoint(s):
    """family + versions + tiers of a lower-case model name; settings and everything after the first one dropped; None when the
    name holds no setting or tier after its family (left to the other rules) or something unknown follows (BERT-CRF)"""
    toks = []
    for t in re.split(r'[\s_-]+', s.strip()):
        m = CKPT_GLUED.match(t)
        toks += [m.group(1), m.group(2)] if m and m.group(1) not in ('', 'ro') else [t]
    known = lambda t: CKPT_SETTING.match(t) or CKPT_VERSION.match(t) or t in CKPT_TIER
    i = next((k for k in range(1, len(toks)) if known(toks[k])), None)
    if i is None:
        return None
    fam, vers, tiers, cut = toks[:i], [], [], False
    for t in toks[i:]:
        if CKPT_SETTING.match(t) or (t == 'mini' and re.search(r'bert[a]?$', fam[-1])):   # a size ('7b'; BERT-mini) is a setting
            cut = True; break
        if CKPT_VERSION.match(t):
            if tiers and re.fullmatch(r'\d{2,}', t) and vers:                # RL-185: claude-3-5-sonnet-20… : a cut snapshot code
                cut = True; break
            vers.append(t)
        elif t in CKPT_TIER:
            tiers.append(t)
        else:
            return None
    if not cut and not tiers:
        return None
    if len(fam) > 1 and fam[0] in ('meta', 'openai', 'google', 'anthropic') and re.match(r'(gpt|gemini|claude|llama|bert|palm|gemma|bard)', fam[1]):
        fam = fam[1:]                                                     # RL-179: vendor before the family dropped
    parts = fam + (tiers + vers if fam == ['claude'] else vers + tiers)    # RL-179: Claude tier first, every other family version first
    return "_".join(parts)


def canonical(surface):
    """one canonical atom root per focal term: lower case, separators -> '_', '.0' dropped, a glued version split ('gpt4o' -> 'gpt_4o'),
    SYNONYMS applied. 'GPT-4', 'GPT4', 'GPT 4', 'GPT-4.0' -> gpt_4; 'LLMs', 'large language models' -> llm; 'GPT-4o' stays gpt_4o."""
    s = surface.lower().replace('‐', '-').replace('–', '-').rstrip('.')
    org = re.match(r'^([a-z0-9][a-z0-9._-]*)/(?=[a-z0-9])', s)
    if org and not re.search(r'gpt|bert|llama|t5|roberta|mistral|gemma|qwen|claude|gemini|deepseek|\d', org.group(1)):
        s = s[org.end():]                                                 # RL-185: a hub organisation prefix (cardiffnlp/…) dropped
    lead = re.match(r'^(' + '|'.join(CKPT_LEAD) + r')[\s_-]+(?=(?:bert|roberta|gpt|llama|t5|electra|xlnet|bart|mistral|gemma|qwen)\b)', s)
    if lead:                                                              # RL-184 (owner): base-BERT -> bert; only before a model
        s = s[lead.end():]                                                # family ('large language models' is untouched)
    f = fold_checkpoint(s) if s not in SIZE_IN_NAME else None             # RL-172/RL-177: settings dropped, tiers kept
    if f:
        s = f
    s = re.sub(r'[\s\-_](?:(?:19|20)\d\d(?:-\d\d){0,2}|\d{4})$', '', s)   # snapshot date (owner 2026-09-30): gpt-4o-2024 -> gpt-4o
    if s not in SIZE_IN_NAME:
        s = re.sub(r'[\s\-_]\d+(?:\.\d+)?(?:x\d+)?[bm]$', '', s)         # model size (owner 2026-09-30): mistral-7b -> mistral
    s = s.replace('+', '_plus')
    s = re.sub(r'(\d)\.0(?!\d)', r'\1', s)
    s = re.sub(r'[\s\-./]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s).strip('_')
    s = re.sub(r'_+', '_', s)
    s = re.sub(r'^chat_gpt(?=_|$)', 'chatgpt', s)
    # RL-179 (owner 2026-10-22): a vendor name before a model is replaced by the model family: OpenAI o1 / ChatGPT o1 / GPT-o1 ->
    # gpt_o1; Google Gemini, Anthropic Claude, Meta Llama, OpenAI GPT, Google BERT -> the family alone (Microsoft Copilot is the
    # product's own name and stays)
    s = re.sub(r'^(?:openai|chatgpt|gpt)_?(o\d)(?=_|$)', r'gpt_\1', s)     # also glued: GPTo1, ChatGPTo1 (owner: individual rules)
    s = re.sub(r'^biobert_v1_(\d)\d+(?=_|$)', r'biobert_v1_\1', s)           # BioBERT has v1.0-1.2 only: v1.18 = v1.1 + citation 8
    s = re.sub(r'^(?:openai|google|anthropic|meta)_(?=(?:gpt|gemini|claude|llama|bert|palm|gemma|bard)(?:_|\d|$))', '', s)
    s = re.sub(r'^gpt_35(?=_|$)', 'gpt_3_5', s)                           # RL-172: Azure's GPT-35(-turbo)
    s = re.sub(r'^claude_(\d+(?:_\d+)?)_(sonnet|opus|haiku)$', r'claude_\2_\1', s)   # RL-177: tier before version
    s = CKPT_GLUED_SIZE.sub(r'\1', s)                                    # RL-184 (owner): mistral7b -> mistral
    s = {'gpt_4b': 'gpt_4', 'gpt4b': 'gpt_4', 'gpt_2m': 'gpt_2', 'gpt2m': 'gpt_2'}.get(s, s)   # RL-184: read, GPT4b = the "biased" GPT-4
    s = SYNONYMS.get(s, s)                                                # grader, GPT2M = GPT-2 Medium
    m = re.fullmatch(VERSIONED + r'(\d[a-z0-9_]*)', s)
    if m:
        s = f"{m.group(1)}_{m.group(2)}"
    return re.sub(r'^chatgpt_(?=\d)', 'gpt_', s)                         # owner 2026-09-30: chatgpt_4 -> gpt_4 (bare chatgpt stays)


def token_spans(text, a2w):
    """char span of each token position (from atom2word), aligned in order; tokens that cannot be found are left out"""
    spans, cur = {}, 0
    for pos, word in sorted({(p, w) for _, w, p in a2w}):
        i = text.find(word, cur)
        if i >= 0:
            spans[pos] = (i, i + len(word)); cur = i + len(word)
    return spans


# RL-185 (owner 2026-10-23): model names written in lower-case Hugging Face style (xlm-roberta-base, gemini-2.5-pro,
# cardiffnlp/twitter-roberta-base-sentiment-latest) that the v3 matcher misses; families from HUB_FAMILIES (set by run())
HUB_FAMS = set()
HUB_TOKEN = re.compile(r"(?<![\w./-])((?:[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*[a-z0-9])(?![\w/])")
HUB_SIGNAL_SEG = re.compile(r"\d|^(?:base|large|small|medium|xl|xxl|tiny|mini|instruct|chat|hf|uncased|cased|it|pro|flash|lite|turbo|"
                            r"sonnet|opus|haiku|nano|ultra)$")


def hub_hits(text, spans_taken):
    """RL-185: lower-case hub spellings whose segments include a listed family as a whole segment (or the family + a version:
    gemma3, qwen2.5) and a hub signal (a digit or a checkpoint word); not inside a matcher hit"""
    out = []
    if not HUB_FAMS:
        return out
    for m in HUB_TOKEN.finditer(text):
        tok, a, b = m.group(1), m.start(1), m.end(1)
        if '-' not in tok and '/' not in tok or any(sa < b and sb > a for sa, sb in spans_taken):
            continue
        segs = [x for x in re.split(r'[-_/]', tok) if x]
        if not re.match(r'[a-z]', segs[-1 if '/' not in tok else 0]) and not re.match(r'[a-z]', segs[0]):
            continue                                                     # RL-185: a number sequence (1-2-3-gpt-6-7-8, a plasmid)
        glued = lambda x: re.fullmatch(r'([a-z]+)\d+(?:\.\d+)?', x)
        fam = [x for x in segs if x in HUB_FAMS or glued(x) and glued(x).group(1) in HUB_FAMS - {'opt'}]   # opt4 (Geant4) is not OPT
        if fam and any(HUB_SIGNAL_SEG.search(x) for x in segs if x not in fam):
            out.append((f"hub:{tok}", (a, b), tok))
    return out


def focal_mentions(matcher, text, a2w, anchored):
    """[(term, surface, canonical, token positions)] for every accepted focal match; a match inside a token is extended to the whole token
    only when the rest is a version ('ChatGPT' in 'ChatGPT-3.5' -> chatgpt_3_5), otherwise the matched text decides the canonical form"""
    spans = token_spans(text, a2w)
    out = []
    hits = matcher.hits(text, anchored)[0]
    hits = hits + hub_hits(text, [h[1] for h in hits])                 # RL-185
    for term, (a, b), surf in hits:
        pos = sorted(p for p, (s, e) in spans.items() if s < b and e > a)
        if not pos:
            out.append((term, surf, canonical(surf), [], None)); continue
        ts, te = spans[pos[0]][0], spans[pos[-1]][1]
        if ts == a and te > b and VERSION_SUFFIX.match(text[b:te]):
            surf = text[a:te]
        # §6 item 8: a version written as the next token ("DeepSeek-V3", "ChatGPT 4o", "GPT 2") is part of the name, when the
        # name is a versioned family and has no version yet; a size token ("8B") is not (dropped as a model size)
        base = None                                                       # RL-177: the mention as the RL-130 rule forms it
        # RL-177: also a product tier ("Gemini 2.5 Pro", "GPT-4 Turbo", "GPT-4o mini", "Claude Sonnet 4.5": tier, then version) and a
        # snapshot code after a version ("GPT-4 0613", dropped by canonical); a hyphen the parser made its own token is stepped over
        # ("DeepSeek-R1" parsed as deepseek - r1); up to three tokens
        for _ in range(3):
            later = [p for p in sorted(spans) if p > pos[-1]]
            c = canonical(surf)
            versioned = re.match(VERSIONED + r'(?:_|$)', c)
            if not later or c in GENERIC_NAMES or (not versioned and a > 0 and (text[a - 1].isalnum() or text[a - 1] in '-\u2010')):
                break             # RL-179: tiers and v-versions after any model name that is a word of its own ('BERT' inside
                                  # 'BioBERT v1.1' is not), plain numbers only after a versioned family
            q, end = later[0], spans[pos[-1]][1]
            gap, tokw = text[end:spans[q][0]], text[spans[q][0]:spans[q][1]]
            if tokw in ('-', '\u2010', '\u2013', '\u2014') and gap in ('', ' ') and len(later) > 1 \
                    and text[spans[q][1]:spans[later[1]][0]] in ('', ' '):        # RL-179: also "Gemini 1.5 – Pro", the dash a token
                q = later[1]; gap, tokw = '-', text[spans[q][0]:spans[q][1]]
            if gap not in ('-', ' ', '\u2010', '\u2013', '- ', ' - ', ' \u2013 ', ' \u2010 '):   # RL-179: "Gemini 1.5 – Pro"
                break
            has_version = bool(re.search(r'\d', c[len(versioned.group(0)):] if versioned else c))
            is_version = VERSION_TOKEN.match(tokw) and not SIZE_RX.match(tokw.lower()) and not VALUE_AFTER.match(text[spans[q][1]:]) \
                and not SCORE_TOKEN.match(tokw) and (versioned or re.fullmatch(r'v\d{1,2}(?:\.\d{1,2})*', tokw, re.I))   # 'V324': v3 + citation
            t_low, last = tokw.lower(), c.split('_')[-1]
            is_tier = (t_low in CKPT_TIER - ({'max'} if not c.startswith('qwen') else set())
                       and (last not in CKPT_TIER or (last == 'flash' and t_low == 'lite'))
                       and not (t_low == 'nano' and re.match(r'\s*banana', text[spans[q][1]:], re.I)))   # Gemini "Nano Banana": a name
            gpt_cont = c == 'gpt' and re.fullmatch(r'(?i)(?:oss|j|neo|neox)(?:[-_.][\w.]+)*', tokw)   # RL-185: GPT-OSS-120B, GPT-J
            if not ((is_version and not has_version) or is_tier or gpt_cont or (versioned and has_version and SNAPSHOT_CODE.match(tokw))):
                break
            old_rule = base is None and is_version and re.fullmatch(VERSIONED, c) and q == later[0] and gap in ('-', ' ', '\u2010', '\u2013', '- ')
            if base is None and not old_rule:
                base = (surf, pos)
            pos = pos + [q]; surf = text[a:spans[q][1]]
        out.append((term, surf, canonical(surf), pos, base))
    return out


# ------------------------------------------------------------------ edge helpers
def is_focal(atom):
    p = atom.parts()
    return len(p) > 2 and p[2] == 'focal'


def at_barrier(edge, barrier):
    """barrier: None, one edge string, or a set of edge strings (D6: every parent clause of the unit)"""
    if not barrier:
        return False
    return str(edge) == barrier if isinstance(barrier, str) else str(edge) in barrier


def terminal(edge, barrier=None):
    if edge is None or at_barrier(edge, barrier):
        return []
    if edge.is_atom():
        return [edge]
    return [a for sub in edge for a in terminal(sub, barrier)]


def replace_all(edge, mapping):
    if edge.is_atom():
        new = mapping.get(str(edge), edge)
        return new(edge) if callable(new) else new          # RL-186: a factory makes one atom per occurrence
    return hedge(tuple(replace_all(c, mapping) for c in edge))


def tight_cover(edge, need):
    """RL-177: the smallest sub-edge holding every atom of need holds nothing else but connectors (a joined tier, version or
    code belongs to the name's own phrase, not to a list or another clause)"""
    path = smallest_cover(edge, {str(a) for a in need})
    if path is None:
        return False
    sub = edge
    for i in path:
        sub = sub[i]
    return all(str(a) in need or a.type()[:1] in ('B', 'J') or not re.search(r'[a-z0-9]', a.root(), re.I) for a in terminal(sub))


def remove_atoms(edge, drop):
    """RL-179: edge without the atoms in drop. A one-argument modifier in drop ("(4.6 claude_opus)") gives way to its argument;
    another connector in drop leaves the edge as it is. An edge whose arguments are all removed keeps its connector when the
    connector carries words (a modifier, predicate or nested edge) and goes when it is a pure builder or conjunction; a builder
    left with one argument becomes that argument. Returns (edge or None, atoms removed)."""
    if edge.is_atom():
        return (None, 1) if str(edge) in drop else (edge, 0)
    head = edge[0]
    if head.is_atom() and str(head) in drop:
        if head.type()[:1] == 'M' and len(edge) == 2:
            k, m = remove_atoms(edge[1], drop)
            return k, m + 1
        return edge, 0
    kids, n = [head], 0
    for c in edge[1:]:
        k, m = remove_atoms(c, drop); n += m
        if k is not None:
            kids.append(k)
    if not n:
        return edge, 0
    if len(kids) == 1:
        return (None, n) if head.is_atom() and head.type()[:1] in ('B', 'J') else (head, n)
    if len(kids) == 2 and head.is_atom() and head.type()[:1] == 'B':
        return kids[1], n
    return hedge(tuple(kids)), n


def smallest_cover(edge, need):
    """index path to the deepest sub-edge whose atoms include all of need"""
    if edge.is_atom():
        return None
    for i, c in enumerate(edge):
        r = smallest_cover(c, need)
        if r is not None:
            return [i] + r
    return [] if need <= {str(a) for a in terminal(edge)} else None


def smallest_cover_pos(edge, ps, apos):
    """RL-186: index path to the deepest sub-edge holding the atom occurrences at the token positions ps (by position,
    not by atom string: "natural language processing … ability of language model" has two 'language' atoms)"""
    if edge.is_atom():
        return None
    for i, c in enumerate(edge):
        r = smallest_cover_pos(c, ps, apos)
        if r is not None:
            return [i] + r
    got = set().union(*(apos.get(id(a), set()) for a in terminal(edge)))
    return [] if ps <= got else None


def replace_at(edge, path, mapping):
    if not path:
        return replace_all(edge, mapping)
    ch = list(edge)
    ch[path[0]] = replace_at(edge[path[0]], path[1:], mapping)
    return hedge(tuple(ch))


def is_modal(a):
    """modal verb: type Mm, or a modal word the parser typed as another modifier ('can/M'); 'May' the month is C and not a modal"""
    t = a.type()
    return t.startswith('Mm') or (t.startswith('M') and not t.startswith(('Mv', 'Mn'))
                                  and SINGLE_QUOTES.sub("'", urllib.parse.unquote(a.root()).lower()) in MODAL_WORDS)   # RL-180: ’ll


def is_pred(a):
    if a.type().startswith('Mv') and hyphen_prop(a):          # 2026-10-09 audit: a hyphen property is never a predicate
        return False
    return a.type().startswith(('P', 'Mv')) or is_modal(a)


def is_neg(a):
    return urllib.parse.unquote(a.root()).lower() in NEGATION and a.type().startswith('M')


CURRENT_CTX = None


def only_before_verb(src):
    """'only' (source atom string) directly before a predicate, or before one adverb and then a predicate; not 'if only'"""
    ctx = CURRENT_CTX
    if ONLY_MODE == 'vg':
        return True
    def ok(p):
        if ctx.words.get(p - 1) == 'if':
            return False
        t1 = ctx.bypos.get(p + 1, '/x/').split('/')[1]
        t2 = ctx.bypos.get(p + 2, '/x/').split('/')[1]
        return t1.startswith(('P', 'Mv', 'Mm')) or (t1.startswith('M') and not t1.startswith(('Md', 'M#', 'Mn')) and t2.startswith(('P', 'Mv')))
    ps = [p for _, p in ctx.pos.get(str(src), [])]
    return bool(ps) and all(ok(p) for p in ps)      # every occurrence of an identical atom must qualify


def is_adverb(a):
    """Q4: a plain modifier (type M, an adverb) that joins the verb group; particles (open question 5) and 'only' excluded"""
    w = urllib.parse.unquote(a.root()).lower()
    return Q4 and a.type() == 'M' and w not in PHRASAL_PARTICLES | DIRECTION_PARTICLES and w != 'only' and ADV_ROLE.get(w) != 'leave'


def is_vg_mod(a):
    """negation, 'able/unable' of a merged 'be able to' (§6 item 3), and 'only' when it modifies a predicate (ONLY_MODE vg /
    vg2): members of the verb group"""
    if is_neg(a) or is_adverb(a) or (CURRENT_CTX is not None and str(a) in CURRENT_CTX.able):
        return True
    if ONLY_MODE in ('vg', 'vg2') and urllib.parse.unquote(a.root()).lower() == 'only' and a.type().startswith('M'):
        return only_before_verb(a)
    return False


def flat_is_verb_group(f):
    """a flattened group made only of verb-group atoms; an 'only' counts only if each of its sources passes only_before_verb"""
    for out in f.atoms():
        if str(out) in CONTRAST_ATOMS:                                 # RL-186
            continue
        if not (pred_or_neg(out) or (Q4 and all(is_adverb(hedge(src)) for src in f.items[str(out)][1]))):
            return False
        if urllib.parse.unquote(out.root()).lower() == 'only' and not all(only_before_verb(src) for src in f.items[str(out)][1]):
            return False
    return True


def has_pred(head):
    return any(is_pred(a) for a in terminal(head))


# 2026-10-09 audit (owner): a participle joined to the word before it by a hyphen ("LLM-based chatbots", "GPT-4-generated text")
# is a property of the noun, not a predicate: its phrase is not a clause, and the word is kept as written, typed M ("based/M").
# Without a hyphen the participle is treated as before. XBASED: 'hyphen' (default), 'all' (every participle phrase, RL-107), '0' off.
XBASED = os.environ.get('XBASED', 'hyphen')
HYPHENS = re.compile('^[-\u2010\u2011\u2013\u2014] ?$')   # hyphen, Unicode hyphens, en/em dash; 'LLM- based' (spaced) included
# cousin grouping (owner 2026-10-09): words of a phrase that is cut open to reach a clause inside it stay one cousin_he
GROUP = os.environ.get('GROUP', '1')   # '1' (decided): all words outside nested clauses; '2': only loose single words (rejected); '0' off
HYPH_JOINED = os.environ.get('HYPH_JOINED', '1') == '1'


def hyphen_prop(a):
    """a participle (type Mv) hyphen-joined to the token before it at every position of its atom"""
    if XBASED != 'hyphen' or not a.is_atom() or not a.type().startswith('Mv'):
        return False
    return str(a) in CURRENT_CTX.hyph or (HYPH_JOINED and '_' in a.root())   # G2 joined 'expert-written' as expert_written


def is_clause(edge):
    if XBASED == '0' or edge.is_atom():
        return True
    try:
        if edge.type().startswith('R'):
            return True
    except RuntimeError:          # a modifier phrase whose modifier became a focal atom, (llm/Cp/focal based/Mv): not typable, not a clause
        pass
    if XBASED == 'all':
        return False
    return not any(hyphen_prop(a) for a in terminal(edge[0]))


def deep_pred(edge, barrier):
    if edge is None or at_barrier(edge, barrier):
        return False
    if edge.is_atom():
        if XBASED == 'all':
            return is_pred(edge) and not edge.type().startswith('Mv')
        return is_pred(edge) and not hyphen_prop(edge)
    return any(deep_pred(c, barrier) for c in edge)


def split_clauses(edge, barrier, acc, clauses, phrases):
    """a phrase cut open to reach the clauses nested in it: its loose words (acc, one cousin), its sub-phrases without a
    clause (phrases, one cousin each; GROUP '1' puts their words in acc too) and the nested clauses"""
    if edge is None or at_barrier(edge, barrier):
        return
    if edge.is_atom():
        acc.append(edge); return
    if has_pred(edge[0]) and is_clause(edge):
        clauses.append(edge); return
    if GROUP == '2' and not deep_pred(edge, barrier):
        phrases.append(edge); return
    for c in edge:
        split_clauses(c, barrier, acc, clauses, phrases)


def class_collapse(atom, override=None):
    p = atom.parts()
    if len(p) > 1:
        r = p[1]
        p[1] = override or ('C' if r.startswith('C') else 'M' if r.startswith('M') and not r.startswith('Mv') else
                            'P' if r.startswith(('P', 'Mv')) else r)
    return hedge('/'.join(p))


def h12(i, edge_str):
    return hashlib.sha1(f"{i}|{edge_str}".encode("utf-8")).hexdigest()[:12]


def q4_opener(ctx, word, p):
    """token p opens the unit or a clause after ; or :, and a comma follows it in the text"""
    nth = sum(1 for q, w in ctx.words.items() if q < p and w == word)
    ms = list(re.finditer(r"\b" + re.escape(word) + r"\b", ctx.text, re.I))
    if nth >= len(ms):
        return False
    m = ms[nth]
    before = ctx.text[:m.start()].rstrip()
    return (re.fullmatch(r"[\W\d_]*", before) is not None or before.endswith((";", ":"))) and ctx.text[m.end():m.end() + 1] == ","


def q4_as_well(ctx, p):
    """'well' at p is the connective 'as well (as)': after 'as', not after a verb or a degree word, not 'or better/worse'"""
    if ctx.words.get(p - 1) != "as":
        return False
    t2 = ctx.bypos.get(p - 2, "/x/").split("/")[1]
    if t2.startswith(("P", "Mv")) or ctx.words.get(p - 2) in Q4_DEGREE:
        return False
    follow = " ".join(ctx.words.get(q, "") for q in range(p + 1, p + 5))
    return not re.search(r"\bor (better|worse)\b", follow)


BE_HAVE_DO = {"is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did"}


def q4_but_before(ctx, p):
    """'but also', or 'but' + up to two auxiliaries, modals, forms of be/have/do or pronouns + 'also' ('but can also',
    'but it also'); no punctuation (no missing token position) in between"""
    for q in range(p - 1, p - 4, -1):
        if q not in ctx.words:
            return False
        if ctx.words[q] == "but":
            return True
        t = ctx.bypos.get(q, "/x/").split("/")[1]
        if not (t.startswith(("Ci", "Mv", "Mm")) or ctx.words[q] in BE_HAVE_DO):
            return False
    return False


def q4_move_also(edge):
    """the parser can attach 'also' to the conjunction ('(also/M but/J) A B'): move it into the predicate of the conjunct
    that follows 'but' (the last one), so that it reaches that clause's verb group"""
    if edge.is_atom():
        return edge
    kids = [q4_move_also(c) for c in edge]
    conn = kids[0]
    if (Q4 and not conn.is_atom() and len(conn) == 2 and len(kids) >= 3 and all(a.is_atom() for a in conn)
            and {urllib.parse.unquote(a.root()).lower() for a in conn} == {"also", "but"}):
        also = next(a for a in conn if a.root() == "also")
        but = next(a for a in conn if a.root() == "but")
        last = kids[-1]
        if not last.is_atom() and has_pred(last[0]):
            last = hedge(tuple([hedge((also, last[0]))] + list(last[1:])))
            return hedge(tuple([but] + kids[1:-1] + [last]))
    return hedge(tuple(kids))


def q4_all(ctx, atom, test):
    """a position rule holds for every position of this source atom (atoms with the same string are not told apart)"""
    ps = [p for _, p in ctx.pos.get(str(atom), [])]
    if not ps:
        return False
    hits = [test(p) for p in ps]
    if any(hits) and not all(hits):
        ctx.stats['q4_ambiguous_position'] += 1
    return all(hits)


# ------------------------------------------------------------------ per-unit context and atom curation
class Ctx:
    def __init__(self, unit, modals):
        self.modals = modals
        self.lemma, self.lemma_head = {}, set()
        for x in unit["extra_edges"]:
            m = re.fullmatch(r'\(_lemma (\S+) (\S+)\)', x)
            if m:
                key, lem = m.group(1), m.group(2).split('/')[0]
                root, t = urllib.parse.unquote(key.split('/')[0]), key.split('/')[1][:1]
                jm = JOINED.fullmatch(root)
                spacy = urllib.parse.unquote(lem).lower()
                if jm and spacy != root.lower():                          # §8 item 5: spaCy's suffix rule fired on a joined word
                    head, last = jm.group(1), jm.group(2)                 # head keeps its joiner ("pre_", "pre\u2010")
                    plain = HEAD_LEMMA.get((last.lower(), t))
                    if plain is not None and not (t == 'C' and plain == last.lower() and spacy == root.lower()[:-1]):
                        # (a noun whose final "s" spaCy stripped, with no plain-word lemma to go on, keeps spaCy's: set_ups -> set_up)
                        n = min(3, len(plain), len(last))
                        new = f"{head}{plain}" if plain[:n] == last.lower()[:n] else root
                        if new.lower() != spacy:
                            lem = new; self.lemma_head.add(key)
                self.lemma[key] = lem
        self.pos = collections.defaultdict(list)                        # source atom -> [(word, position)]
        self.apos, self.keep = None, []                                 # RL-186: id(atom object) -> its own positions (None: unknown)
        self.word_at = {p: w for at, w, p in unit["atom2word"]}
        self.bypos = {p: at for at, w, p in unit["atom2word"]}
        self.text = unit["text"]
        self.spans = token_spans(unit["text"], unit["atom2word"])
        att, order = collections.defaultdict(list), sorted(self.spans)
        for a, w, p in unit["atom2word"]:
            if a.split('/')[1].startswith('Mv') if '/' in a else False:
                i = order.index(p) if p in self.spans else 0      # the hyphen itself has a token position but no atom
                prev, cur = (self.spans[order[i - 1]] if i else None), self.spans.get(p)
                gap = unit["text"][prev[1]:cur[0]] if prev and cur else None
                att[a].append(gap is not None and bool(HYPHENS.match(gap) or (gap.strip() == '' and HYPHENS.match(unit["text"][prev[1] - 1]))))
        self.hyph = {a for a, v in att.items() if v and all(v)}
        self.words = {p: w.lower() for at, w, p in unit["atom2word"]}
        self.label_pos, self.nontext_num = self.find_labels_and_numbers()
        global CURRENT_CTX
        CURRENT_CTX = self
        for a, w, p in unit["atom2word"]:
            self.pos[a].append((w, p))
        self.decisions = {}                                             # (atom, periphery) -> (reason or None, out atom, modal)
        self.able = set()                                               # 'able'/'unable' atoms merged as modal-like (§6 item 3)
        self.stats = collections.Counter(); self.detail = collections.defaultdict(collections.Counter)

    def init_positions(self, edge, unit):
        """RL-186: each atom occurrence of the unit edge gets its own token position (G2 positions file); without one, or
        when the edge was rebuilt from another text (reflexive table, spelling correction), positions stay by atom string"""
        pos = unit.get("atom_pos")
        atoms = edge.all_atoms()
        if pos is None or len(pos) != len(atoms):
            self.stats['positions_by_string_unit'] += 1
            return
        self.apos = {}
        for a, p in zip(atoms, pos):
            if p is not None:
                self.apos[id(a)] = {p}
        self.keep.append(edge)
        self.stats['positions_exact_unit'] += 1

    def inherit(self, new, srcs):
        """RL-186: a new atom made from source atoms carries their positions (if every source's is known)"""
        if self.apos is None:
            return new
        got = [self.apos.get(id(x)) for x in srcs]
        if all(g is not None for g in got):
            self.apos[id(new)] = set().union(*got); self.keep.append(new)
        return new

    def lemma_of(self, atom):
        p = atom.parts()
        return self.lemma.get(f"{p[0]}/{p[1][:1]}/{p[2] if len(p) > 2 else 'en'}", p[0]) if len(p) > 1 else p[0]

    def find_labels_and_numbers(self):
        """§6 item 8: token positions of layout words used as labels with their labels ("Table 2", "Fig. 1b", "Supplementary
        Table S1"), and of numbers that are not plain text (labels, inside square brackets, list labels "(1)", an enumerator
        opening the unit, statistical notation)"""
        order = sorted(self.words)
        nxt = {p: order[i + 1] for i, p in enumerate(order[:-1])}
        labels, nontext = set(), set()
        for p in order:
            w = self.words[p].rstrip('.')                                  # "Fig." is one token
            if w in NUM_LABEL_WORDS and p in nxt:
                q, got = nxt[p], []
                while q is not None and self.words.get(q) in ('.', ':'):      # "Fig . 1b": the period can be a token
                    q = nxt.get(q)
                if q is None or p not in self.spans or q not in self.spans \
                        or not re.fullmatch(r'[\s.]*', self.text[self.spans[p][1]:self.spans[q][0]]):
                    continue                                           # RL-120: the label stands directly before its number
                index_noun = w in INDEX_NOUNS or (w.endswith('s') and w[:-1] in INDEX_NOUNS)
                while q is not None and LABEL_RX.match(self.words.get(q, '')) and len(got) < 6:
                    if index_noun and not re.fullmatch(r'\d+[a-z]?', self.words.get(q, '')):
                        break                                          # "Grade Level 17.4": an index is a whole number
                    if q in self.spans and UNIT_AFTER.match(self.text[self.spans[q][1]:]):
                        break                                          # "patients 65 years", "model (80%)": a value, not an index
                    got.append(q)
                    r = nxt.get(q)
                    while r is not None and self.words.get(r) in (',', 'and', 'or', '&', '-', '–'):
                        r = nxt.get(r)                                 # the connectives can be tokens of their own
                    gap = self.text[self.spans[q][1]:self.spans[r][0]] if r in self.spans and q in self.spans else 'x'
                    q = r if re.fullmatch(r'[\s,&\-–]*(?:and|or)?[\s,&\-–]*', gap) else None    # "Tables 4, 6, 7, and 8"
                if got:
                    labels.update(got); nontext.update(got)
                    if w in LAYOUT_WORDS:
                        labels.add(p)
                        prev = order[order.index(p) - 1] if order.index(p) else None
                        if prev is not None and self.words.get(prev).rstrip('.') in LAYOUT_PREFIX:
                            labels.add(prev)                           # "Supplementary Note 3", "Online Resource 5"
        first = order[0] if order else None
        for p in order:
            w = self.words[p]
            if not NUMBER_RX.match(w) or p not in self.spans:
                continue
            a, b = self.spans[p]
            before, after = self.text[:a], self.text[b:]
            if (before.rfind('[') > before.rfind(']') and ']' in after) \
                    or (before.endswith('(') and after.startswith(')') and len(w) <= 2) \
                    or (p == first and re.match(r'^\s*[(\[]?\d+(?:\.\d+)*[.)\]]?\s', self.text)) \
                    or STAT_BEFORE.search(before[-14:]) or re.match(r'\s*±', after) or re.match(r'\s*%\s*CI\b', after) \
                    or (re.search(r'\d\s*[–-]\s*$', before[-6:]) and any(q in nontext for q in (p - 1, p - 2))) \
                    or self.debris_number(p, a, b, before, after):
                nontext.add(p)
        nontext |= self.citation_numbers()
        return labels, nontext

    def citation_numbers(self):
        """D4.1 (owner 2026-10-19): superscript citations flattened into the text, a number or list of numbers after a word and
        followed by a space and punctuation ("based on BERT 18 .", "hallucinations 7– 9 ,", "domains 8 , 9 , 10 , 11 ,");
        every number of the list goes. Not: after a value word ("would cost 0 .", "of 2 : 1"), a ratio, a thousands figure
        split by a space ("7 , 380 seconds"), or the end of the unit without punctuation ("out of 100")"""
        out = set()
        for m in CITE_LIST.finditer(self.text):
            prev = re.findall(r"[A-Za-z]+", self.text[:m.start()][-30:])
            items = re.findall(r"\d+", m.group(1))
            if (prev and prev[-1].lower() in CITE_VALUE_WORDS) or re.match(r"\s+:\s*\d", self.text[m.end():]) \
                    or (len(items) == 2 and len(items[1]) == 3 and ',' in m.group(1)):
                continue
            out |= {p for p, (s, e) in self.spans.items() if m.start() <= s and e <= m.end() and NUMBER_RX.match(self.words.get(p, ''))}
        return out

    def debris_number(self, p, a, b, before, after):
        """RL-120: a number that is citation or list debris ("REF009518 3", "2020.1,2", "6., 7., 8"), sits inside a formula
        ("(1 | Word)"), or belongs to a snapshot-date id ("gpt-4-turbo-2024-04-09")"""
        order = sorted(self.words)
        i = order.index(p)
        if i and re.fullmatch(r'ref\d+', self.words[order[i - 1]]):
            return True
        if re.search(r'[A-Za-z)\]\d][.,]$', before) or re.match(r'\.\s*,', after):
            return True
        o, c = before.rfind('('), after.find(')')
        if o > before.rfind(')') and c >= 0 and re.search(r'[|~^*=]', before[o:] + after[:c]):     # not "+": "ChatGPT + top 5"
            return True
        if o > before.rfind(')') and c >= 0 and re.search(r'[\d%]\s*$', before[:o]) and \
                re.fullmatch(r'[−\-]?\d+(?:\.\d+)?%?\s*(?:to|–|-|,)\s*[−\-]?\d+(?:\.\d+)?%?', (before[o + 1:] + self.text[a:b] + after[:c]).strip()):
            return True                                                # an interval right after a value: "4.5 (4.0–5.0)"
        for m in re.finditer(r'\d{4}-\d{2}-\d{2}', self.text[max(0, a - 11):b + 11]):
            s0 = max(0, a - 11) + m.start()
            if s0 <= a and b <= s0 + len(m.group(0)):
                return True
        return False

    def compares_number(self, p):
        """a comparison word ("over 90%", "more than 20", "up to 5") whose next content word is a kept number"""
        for q in range(p + 1, p + 4):
            w = self.words.get(q)
            if w is None or w in ('than', 'to'):
                continue
            return bool(NUMBER_RX.match(w)) and q not in self.nontext_num
        return False

    def passive_after(self, p):
        """§6 item 1: the first content word after the auxiliary at token p (adverbs, negation, other auxiliaries, modals,
        'to', degree words and coordinators skipped) is a main verb whose argument roles carry the passive subject ('p')"""
        for q in range(p + 1, p + 12):
            at = self.bypos.get(q)
            if at is None:
                continue
            t = at.split('/')[1] if '/' in at else ''
            if t.startswith(('M', 'J')):
                continue
            f = t.split('.')
            roles = f[1] if len(f) > 1 and f[1][:1] not in '<|-' else ''      # 'Pd.<pf--' has no role field
            return t.startswith('P') and 'p' in roles
        return False

    def adverb_dropped(self, atom, word):
        """§6 item 4 adverb role table: drop, keep, leave (a mistyped noun: kept, not attached), and two conditional rows"""
        act = ADV_ROLE[word]
        if act == 'drop':
            return True
        if word == 'similarly':                 # kept only directly before "to" ("performed similarly to")
            return not q4_all(self, atom, lambda p: self.words.get(p + 1) == 'to')
        if word in ('above', 'below'):          # dropped only at the end of a clause ("discussed above.")
            return q4_all(self, atom, lambda p: p in self.spans and bool(CLAUSE_END.match(self.text[self.spans[p][1]:])))
        return False

    def curate(self, atom, periphery):
        """(out atom or None, modal lemma or None); counted once per unit and atom"""
        key = (str(atom), periphery)
        if key in self.decisions:
            return self.decisions[key]
        out, modal, reason = None, None, None
        root, t = atom.root(), atom.type()
        word = INVISIBLE.sub('', urllib.parse.unquote(root).lower())          # RL-180: invisible characters removed
        lemma = self.lemma_of(atom)
        if word in WORD_MAP:                                                  # RL-180: w/o -> without (then the stop list decides)
            word = lemma = WORD_MAP[word]
        if str(atom) in self.able:
            out = hedge(f"{word}/M/en"); self.detail['modal_able'][word] += 1    # §6 item 3: "was able to offer" -> able offer
        elif is_focal(atom):
            reason = 'focal_in_periphery' if periphery else None
            out = None if periphery else class_collapse(atom, 'C')
        elif hyphen_prop(atom):
            out = hedge(f"{root.lower()}/M/en"); self.detail['hyphen_property'][word] += 1
        elif word == '%':
            out = hedge('percent/C/en'); self.detail['percent'][t] += 1
        elif t.startswith('P') and CLITIC.match(SINGLE_QUOTES.sub("'", word)):
            v = CLITIC_VERB.get(SINGLE_QUOTES.sub("'", word).rstrip('0123456789'))     # D4.1: "I’m a language model", the
            out = hedge(f"{v}/P/en") if v else None; reason = None if v else 'clitic'  # main verb of a curly clitic
            self.detail['clitic_verb'][word] += 1
        elif q4_all(self, atom, lambda p: p in self.label_pos):
            reason = 'label'; self.detail['label'][word] += 1                 # §6 item 8: "Table 2", "Fig. 1b"
        elif NUMBER_RX.match(word) or (t.startswith(('C#', 'M#')) and word in NUMBER_WORDS):
            if q4_all(self, atom, lambda p: p in self.nontext_num):
                reason = 'number_nontext'; self.detail['number_nontext'][word] += 1
            else:                                  # §6 item 8: plain-text numbers kept as written (0-10 as words rolled back, RL-119)
                nroot = re.sub(r'(?<=\d)(?:,|%2c)(?=\d{3}(?!\d))', '', root.lower())   # graphbrain's encoding kept ('8%2e0')
                out = hedge(f"{nroot}/{'C' if t.startswith('C') else 'M'}/en"); self.detail['number_kept'][word] += 1   # RL-180: 10,000 -> 10000
        elif not re.search(r'[a-z]', word):
            reason = 'no_letters'; self.detail['no_letters'][word] += 1
        elif URL_RX.match(word):
            reason = 'url'                                                    # §6 item 8
        elif SIZE_RX.match(word):
            reason = 'model_size'; self.detail['model_size'][word] += 1       # §6 item 8
        elif word in COMPARE_WORDS and q4_all(self, atom, self.compares_number):
            out = hedge(f"{word}/M/en"); self.detail['comparison_kept'][word] += 1   # §6 item 8: "over 90%" keeps "over"
        elif word in CONTRAST_WORDS and not (t.startswith('C') and word == 'against'):    # "favor, against, none": a label
            out = hedge(f"{CONTRAST_WORDS[word]}/M/en"); self.detail['contrast_kept'][word] += 1           # RL-186
        elif word == 'in' and q4_all(self, atom, lambda p: self.words.get(p + 1) == 'contrast'):
            out = hedge('in_contrast/M/en'); self.detail['contrast_kept']['in contrast'] += 1           # RL-186
            self.contrast_clause = q4_all(self, atom, lambda p: self.words.get(p + 2) not in ('to', 'with'))
        elif word == 'contrast' and q4_all(self, atom, lambda p: self.words.get(p - 1) == 'in'):
            reason = 'fused_in_contrast'
        elif word in CONNECTIVES or word == 'and/or':                         # RL-180: and/or
            reason = 'connective'; self.detail['connective'][word] += 1
        elif Q4 and word in NEGATION and t.startswith('M') and q4_all(self, atom, lambda p: self.words.get(p + 1) in Q4_NOT_ONLY):
            drop = NOTONLY_DROP or (NOTONLY_MODE == 'paired' and q4_all(self, atom, lambda p: q4_paired(self, p)))
            out = None if drop else hedge('not_only/M/en'); reason = 'q4_not_only_dropped' if drop else None; self.detail['q4_not_only'][f"{word} {'/'.join(sorted({self.words.get(p + 1) for _, p in self.pos.get(str(atom), [])}))}"] += 1
        elif Q4 and word in Q4_NOT_ONLY and q4_all(self, atom, lambda p: self.words.get(p - 1) in ("not", "n't")):
            reason = 'q4_fused_not_only'
        elif Q4 and word == 'also' and q4_all(self, atom, lambda p: q4_but_before(self, p)):
            drop = NOTONLY_DROP or (NOTONLY_MODE == 'paired' and q4_all(self, atom, lambda p: q4_also_paired(self, p)))
            out = None if drop else hedge('but_also/M/en'); reason = 'q4_but_also_dropped' if drop else None; self.detail['q4_but_also']['but (…) also'] += 1
        elif Q4 and word == 'even' and q4_all(self, atom, lambda p: self.words.get(p + 1) in ('though', 'if')):
            reason = 'q4_connective'; self.detail['q4_connective']['even (though/if)'] += 1
        elif Q4 and word in Q4_CONNECTIVES:
            reason = 'q4_connective'; self.detail['q4_connective'][word] += 1
        elif Q4 and word in Q4_OPENER_DROP and q4_all(self, atom, lambda p: q4_opener(self, word, p)):
            reason = 'q4_opener'; self.detail['q4_opener'][word] += 1
        elif Q4 and word == 'well' and q4_all(self, atom, lambda p: q4_as_well(self, p)):
            reason = 'q4_as_well'
        elif Q4 and t == 'M' and word in ADV_ROLE and self.adverb_dropped(atom, word):
            reason = 'adverb_role'; self.detail['adverb_role_dropped'][word] += 1
        elif word in NEGATION and t.startswith('M'):
            out = class_collapse(hedge(f"{lemma}/{t}/en"), 'M'); self.detail['negation'][f"{word}/{t}"] += 1
        elif t.startswith('Ci'):
            reason = 'pronoun'
        elif t.startswith(('Md', 'Mi')):
            reason = 'determiner'
        elif t.startswith('Mv') and lemma == 'be' and q4_all(self, atom, self.passive_after):
            out = hedge('be/M/en'); self.detail['passive_be_kept'][word] += 1     # §6 item 1: "should be approached"
        elif t.startswith('Mv') and lemma in AUXILIARIES:
            reason = 'auxiliary'; self.detail['auxiliary_dropped'][lemma] += 1
        elif is_modal(atom):
            lemma = {'ca': 'can', 'wo': 'will', "'ll": 'will', "'d": 'would'}.get(SINGLE_QUOTES.sub("'", word), lemma)   # RL-180: word, curly
            self.detail['modal'][lemma] += 1
            if self.modals == 'keep':
                out = hedge(f"{lemma}/M/en")
            else:
                reason = 'modal_' + self.modals; modal = lemma if self.modals == 'tag' else None
        else:
            main_bhd = t.startswith('P') and lemma in AUXILIARIES     # main-verb be/have/do: the stop list must not decide these
            if word in ACTIVE_STOP or lemma in ACTIVE_STOP:
                if main_bhd:
                    self.detail['stoplist_bypassed_verb'][f"{word}/{t}"] += 1
                else:
                    reason = 'stoplist'
            if reason is None and not t.startswith(KEEP_TYPES):
                reason = 'type_' + t
            if reason is None and word in getattr(self, 'notation_syms', ()):
                reason = 'notation_symbol'; self.detail['notation_symbol'][word] += 1            # RL-181
            ps = {q for _, q in self.pos.get(str(atom), [])}
            if reason is None and (MATH_CHARS.search(word) or (ps and ps <= getattr(self, 'formula_pos', set()) and is_symbol(word)
                                                             and not re.fullmatch(r'o\d', word))):   # o1, o3: models
                reason = 'formula_token'; self.detail['formula_token'][word] += 1                # RL-185
            if reason is None and (clean_root(word) is None or len(clean_root(word)) == 1):
                reason = 'single_letter_or_symbol'; self.detail['single_letter_or_symbol'][word] += 1     # §6 item 8
            if reason is None:
                if t.startswith('Mv'):
                    self.detail['mv_kept'][lemma] += 1
                if not self.lemma.get(f"{root}/{t[:1]}/en"):
                    self.stats['lemma_missing'] += 1
                if f"{root}/{t[:1]}/en" in self.lemma_head:
                    self.stats['lemma_head_rule'] += 1; self.detail['lemma_head_rule'][f"{root} -> {lemma}"] += 1
                lc = clean_root(INVISIBLE.sub('', urllib.parse.unquote(lemma).lower())) or lemma
                lc = norm_word(lc, FORMS)                                     # RL-180: Greek, accents, joiners, compounds
                if lc is None or len(lc) == 1 or lc in ACTIVE_STOP:      # RL-180: the stop list again on the final form (öf -> of)
                    reason = 'greek_notation' if lc is None else 'single_letter_or_symbol' if len(lc) == 1 else 'stoplist'
                else:
                    lc = urllib.parse.quote(lc, safe="_-'&@+")               # §6 item 8: symbols cleaned, hyphen -> _
                    out = class_collapse(hedge(f"{lc}/{t}/en"))
        self.stats['atom_kept' if out is not None else 'atom_dropped'] += 1
        if reason:
            self.stats['drop_' + (reason if not reason.startswith('type_') else 'type')] += 1
            if reason.startswith('type_'):
                self.detail['dropped_type'][t] += 1
        self.decisions[key] = (out, modal)
        return out, modal


class Flat:
    """a flattened group of atoms: out atom -> source atoms, plus modal tags"""
    def __init__(self):
        self.items, self.tags, self.where = {}, [], {}     # where: out atom string -> {(source string, position or None)}

    def add(self, out, src, pos=None):
        self.items.setdefault(str(out), [out, set()])[1].add(str(src))
        w = self.where.setdefault(str(out), set())                   # RL-186: the source occurrence's own positions
        w.update((str(src), p) for p in pos) if pos else w.add((str(src), None))

    def merge(self, o):
        for k, (a, s) in o.items.items():
            self.items.setdefault(k, [a, set()])[1].update(s)
            self.where.setdefault(k, set()).update(o.where.get(k, ()))
        self.tags += [t for t in o.tags if t not in self.tags]
        return self

    def atoms(self):
        return [self.items[k][0] for k in sorted(self.items)]

    def __bool__(self):
        return any(k not in CONTRAST_ATOMS for k in self.items)      # RL-186: contrast words alone are no group


def flatten_atoms(atoms, periphery, ctx):
    f = Flat()
    for a in atoms:
        out, modal = ctx.curate(a, periphery)
        if out is not None:
            f.add(out, a, ctx.apos.get(id(a)) if ctx.apos is not None else None)
        if modal and modal not in f.tags:
            f.tags.append(modal)
    return f


def flatten(branch, barrier, periphery, ctx):
    return flatten_atoms(terminal(branch, barrier), periphery, ctx)


def mk(kind, flat):
    return hedge((kind,) + tuple(flat.atoms()))


MODAL_LEMMAS = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}


def pred_or_neg(a):
    """a curated atom that belongs to a compound verb group: verb, negation or modal"""
    w = urllib.parse.unquote(a.root()).lower()
    return a.type().startswith(('P', 'Mv')) or w in NEGATION or (a.type().startswith('M') and w in MODAL_LEMMAS) \
        or (ONLY_MODE in ('vg', 'vg2') and w == 'only') or w in ('not_only', 'but_also') \
        or (w == 'in_contrast' and getattr(CURRENT_CTX, 'contrast_clause', False)) \
        or (a.type().startswith('M') and w in ('be', 'able', 'unable'))       # passive be (§6 item 1), be able (§6 item 3)


# ------------------------------------------------------------------ 7.5 logic: phrasal verbs, periphery sweep, core circuits
PARTICLE_TYPES = ('Ml',)    # Q5: the parser's particle type, or a plain modifier (type exactly 'M'); not Mt (preposition) or C


def is_particle(a):
    if not a.is_atom():
        return False
    t = a.type()
    return urllib.parse.unquote(a.root()).lower() in PHRASAL_PARTICLES | DIRECTION_PARTICLES and (t.startswith(PARTICLE_TYPES) or t == 'M')


def chain_pred(e):
    """Q5: the predicate under a chain of one-argument modifiers ('to', auxiliaries, modals, negation, adverbs), or None"""
    while not e.is_atom():
        if len(e) != 2 or not e[0].is_atom() or not e[0].type().startswith('M'):
            return None
        e = e[1]
    return e if e.type().startswith('P') else None


def replace_atom(e, old, new):
    if e.is_atom():
        return new if e is old else e
    return hedge(tuple(replace_atom(c, old, new) for c in e))


def fuse_phrasal(edge, ctx):
    if edge.is_atom():
        return edge
    kids = [fuse_phrasal(c, ctx) for c in edge]
    if os.environ.get("Q5_CHAIN", "1") == "1" and len(kids) == 2 and is_particle(kids[0]) \
            and (kids[0].type().startswith('Ml') or urllib.parse.unquote(kids[0].root()).lower() in DIRECTION_PARTICLES) \
            and not kids[1].is_atom():         # direction words (§6 item 8, option 3): a plain modifier on a verb chain too
        v = chain_pred(kids[1])
        if v is not None:                      # (out/Ml (should/Mm (be/Mv carried/P))) -> (should (be carry_out))
            fused = ctx.inherit(hedge(f"{ctx.lemma_of(v)}_{kids[0].root()}/{v.parts()[1]}/en"), (v, kids[0]))
            for src in (v, kids[0]):
                ctx.pos[str(fused)].extend(ctx.pos.get(str(src), []))
            return replace_atom(kids[1], v, fused)
    verbs, parts, others = [], [], []
    for a in kids:
        if not a.is_atom():
            others.append(a); continue
        if a.type().startswith('P'):
            verbs.append(a)
        elif is_particle(a) if os.environ.get("Q5_TYPE", "1") == "1" else urllib.parse.unquote(a.root()).lower() in PHRASAL_PARTICLES:
            parts.append(a)
        else:
            others.append(a)
    if verbs and parts:
        v = verbs[0]
        fused = ctx.inherit(hedge(f"{ctx.lemma_of(v)}_{'_'.join(p.root() for p in parts)}/{v.parts()[1]}/en"), [v] + parts)
        for src in [v] + parts:
            ctx.pos[str(fused)].extend(ctx.pos.get(str(src), []))
        return hedge(tuple([fused] + others))
    return hedge(tuple(kids))


def focal_paths(edge, path=None, idx=()):
    """[(edges from the unit edge down to a focal atom, index of each step's sub-edge in the edge above)]"""
    path = (path or []) + [edge]
    if edge.is_atom():
        return [(path, idx)] if is_focal(edge) else []
    return [p for i, sub in enumerate(edge) for p in focal_paths(sub, path, idx + (i,))]


def sweep(edge, barrier, ctx):
    if edge is None or at_barrier(edge, barrier):
        return []
    if edge.is_atom():
        f = flatten(edge, barrier, True, ctx)
        return [('cousin_he', f)] if f else []
    head = edge[0]
    if has_pred(head) and is_clause(edge):
        out, rec = [], []
        head_atoms = terminal(head, barrier)
        preds = flatten_atoms([a for a in head_atoms if is_pred(a) or is_vg_mod(a)], True, ctx)
        for arg in edge[1:]:
            if at_barrier(arg, barrier):
                continue
            t = flatten(arg, barrier, True, ctx)
            if t and flat_is_verb_group(t):
                preds.merge(t)
            else:
                rec.append(arg)
        if preds:
            out.append(('dummy_cousin', preds))
        mods = flatten_atoms([a for a in head_atoms if not (is_pred(a) or is_vg_mod(a))], True, ctx)
        if mods:
            out.append(('cousin_he', mods))
        for arg in rec:
            out += sweep(arg, barrier, ctx)
        return out
    if deep_pred(edge, barrier):
        if GROUP == '0':
            return [x for c in edge for x in sweep(c, barrier, ctx)]
        acc, clauses, phrases = [], [], []
        for c in edge:
            split_clauses(c, barrier, acc, clauses, phrases)
        f = flatten_atoms(acc, True, ctx)
        out = [('cousin_he', f)] if f else []
        for ph in phrases:
            g = flatten(ph, barrier, True, ctx)
            if g:
                out.append(('cousin_he', g))
        for c in clauses:
            out += sweep(c, barrier, ctx)
        return out
    f = flatten(edge, barrier, True, ctx)
    return [('cousin_he', f)] if f else []


def aux_only_head(head, ctx):
    """RL-151: a head made only of auxiliaries (be/have/do typed Mv) is not a clause of its own: subject-auxiliary
    inversion ("Not only did GPT-4 pass the exam") is parsed as (did/Mv gpt-4) inside the clause of "pass"; without this,
    that pair was taken as the governing clause, its verb group came out empty and the focal term got no parent"""
    atoms = terminal(head)
    return bool(atoms) and all(a.type().startswith('Mv') and ctx.lemma_of(a) in AUXILIARIES for a in atoms)


def extract(edge, ctx):
    """D6 (owner 2026-10-19): one parent per clause that governs a focal term; each argument of the clause that holds a focal
    word is a focal_he, the other arguments (and the head's non-verb modifiers) sibling_he. Cousins are the clauses of the
    unit outside every parent clause, linked to every parent (§6 item 11: to every parent of the sentence, link_sentence);
    a clause holding a focal term is never a cousin. Returns (circuits {parent edge: parts}, cousins {cousin edge: [kind,
    flat]}, embedded {parent edge: [edges of the parents in whose argument its clause lies]}, item 11)"""
    norm = fuse_phrasal(q4_move_also(edge), ctx)
    clauses = {}                                        # clause position -> [path, clause index in path, focal argument indices]
    for path, idx in focal_paths(norm):
        if len(path) < 2:
            continue
        pidx = next((i for i in range(len(path) - 2, -1, -1) if has_pred(path[i][0]) and is_clause(path[i])
                     and not aux_only_head(path[i][0], ctx)), -1)
        if pidx == -1:
            ctx.stats['focal_path_no_predicate'] += 1
            continue
        clauses.setdefault(idx[:pidx], [path, pidx, set()])[2].add(idx[pidx])
    circuits, where, barriers = {}, collections.defaultdict(list), set()
    for cpos, (path, pidx, fargs) in clauses.items():
        hi, tele = pidx, Flat()
        for i in range(pidx - 1, -1, -1):
            step = flatten(path[i], str(path[i + 1]), False, ctx)
            if not step:
                continue
            if flat_is_verb_group(step):
                tele.merge(step); hi = i
            else:
                break
        parent = path[pidx]
        head_atoms = terminal(parent[0])
        cp = [a for a in head_atoms if is_pred(a) or is_vg_mod(a)]
        cn = [a for a in head_atoms if not (is_pred(a) or is_vg_mod(a))]
        dummy = Flat().merge(tele).merge(flatten_atoms(cp, False, ctx))
        if not dummy:
            ctx.stats['focal_clause_empty_verb_group'] += 1
            continue
        focals, sibs = [], []
        if 0 in fargs:                                  # a focal word in the head: the head is the focal_he (as before D6)
            f = flatten(parent[0], None, False, ctx)
            if f:
                focals.append(f)
        else:
            nf = flatten_atoms(cn, False, ctx)
            if nf:
                sibs.append(nf)
        for j, arg in enumerate(parent[1:], 1):
            f = flatten(arg, None, False, ctx)
            same = [g for g in focals if str(mk('focal_he', f)) == str(mk('focal_he', g))] if f and j in fargs else []
            if same:                                    # one name split by the parser over two arguments, or a glossary line
                same[0].sides = getattr(same[0], 'sides', 1) + 1; ctx.stats['focal_he_identical_merged'] += 1
                continue
            if f and j not in fargs and all(pred_or_neg(a) and a.root() == 'in_contrast' for a in f.atoms()):
                dummy.merge(f); ctx.stats['in_contrast_to_verb_group'] += 1    # RL-186: "In contrast, X failed"
                continue
            if f:
                (focals if j in fargs else sibs).append(f)
        if not focals:
            ctx.stats['focal_clause_empty_focal_he'] += 1
            continue
        parts = [('dummy_sibling', dummy)] + [('focal_he', f) for f in focals] + [('sibling_he', s) for s in sibs]
        key = str(hedge(('parent',) + tuple(mk(k, f) for k, f in parts)))
        ctx.stats['focal_clauses_identical_merged'] += key in circuits
        circuits.setdefault(key, parts)
        where[key].append(cpos)
        barriers.add(str(path[hi]))
    embedded = {}                                       # item 11: inside an argument (not the head) of an outer parent clause
    for a, pas in where.items():
        outer = [b for b, pbs in where.items() if b != a and any(len(pb) < len(pa) and pa[:len(pb)] == pb and pa[len(pb)] >= 1
                                                               for pa in pas for pb in pbs)]
        if outer:
            embedded[a] = outer
    cousins = {}
    for kind, f in sweep(norm, frozenset(barriers), ctx):   # no parent: the whole unit (its cousins may serve the sentence)
        cousins.setdefault(str(mk(kind, f)), [kind, f])
    return circuits, cousins, embedded


# ------------------------------------------------------------------ one unit
def atom_record(flat, ctx):
    """src: [source atom, word, position] of the occurrences inside this hyperedge (RL-186); a source occurrence whose own
    position is unknown lists every position of its atom string in the unit, as before RL-186 (counted)"""
    out = []
    for a in flat.atoms():
        src = set()
        for s, p in flat.where.get(str(a), ()):
            if p is not None:
                src.add((s, ctx.word_at.get(p), p))
            else:
                ctx.stats['positions_by_string_atom'] += 1
                src.update((s, w, q) for w, q in ctx.pos.get(s, []))
        out.append(dict(atom=str(a), src=[list(x) for x in sorted(src, key=lambda x: (x[2] is None, x[2] or 0, x[0]))] or
                        [[s, None, None] for s in sorted(flat.items[str(a)][1])]))
    return out


def curate_unit(art, s, u, matcher, anchored, modals):
    ctx = Ctx(u, modals)
    ctx.notation_syms = defined_symbols(u["text"])                    # RL-181: symbols defined inside prose ('where L is ...')
    fspans = [m.span() for m in FORMULA_SPAN.finditer(u["text"])]      # RL-185 (owner): symbol tokens inside a formula span
    ctx.formula_pos = {p for p, (a, b) in token_spans(u["text"], u["atom2word"]).items() if any(fa <= a and b <= fb for fa, fb in fspans)}
    edge = hedge(u["main_edge"])
    ctx.init_positions(edge, u)                                       # RL-186
    try:                                    # §6 item 3: lexical modal + complement = one clause, on the parse as G2 gave it
        edge, ctx.able = modal_merge.merge_unit(edge, u, ctx.stats)
    except Exception as e:                  # never lose a unit to the merge: it stays unmerged, and this is counted
        edge = hedge(u["main_edge"]); ctx.stats['modal_merge_error_' + type(e).__name__] += 1
        ctx.apos = None; ctx.init_positions(edge, u)
    by_pos0 = collections.defaultdict(list)
    for a, w, p in u["atom2word"]:
        by_pos0[p].append(a)
    ments = []
    for term, surf, canon, pos, base in focal_mentions(matcher, u["text"], u["atom2word"], anchored):
        extra = []
        if base and not tight_cover(edge, {a for p in pos for a in by_pos0.get(p, [])}):   # RL-177: joined token elsewhere in
            extra = [p for p in pos if p not in base[1]]; pos = base[1]; base = None        # the parse: the name's own atoms are
            ctx.stats['mention_join_not_phrase'] += 1                                        # renamed, the stray token deleted (RL-179)
        ments.append((term, surf, canon, pos, base, extra))
    kept_m = [m for m in ments if not stage_b_rejected(m[0], s["sid"], art)]   # RL-171: hits Stage B rejected stay rejected
    ctx.stats['mention_rejected_stage_b'] += len(ments) - len(kept_m); ments = []
    for term, surf, canon, pos, base, extra in kept_m:                 # RL-172: group-4 decisions
        d = g4_decision(term, art)
        if d and d[0] == 'reject':
            ctx.stats['mention_rejected_g4'] += 1; continue
        if d and d[0] == 'map':
            canon = canonical(d[1]); ctx.stats['mention_mapped_g4'] += 1
        ments.append((term, surf, canon, pos, base, extra))
    covered = {p for m in ments for p in m[3]}
    by_pos = collections.defaultdict(list)
    for a, w, p in u["atom2word"]:
        by_pos[p].append(a)
    mrec = []
    for term, surf, canon, pos, base, extra in ments:
        need = {a for p in pos for a in by_pos.get(p, [])}
        how = 'no_atom'
        if base and need and smallest_cover(edge, need) is None and not all({p for _, p in ctx.pos[a]} <= covered for a in need):
            extra = [p for p in pos if p not in base[1]]; pos = base[1]; base = None        # RL-177/RL-179: placed as the old
            need = {a for p in pos for a in by_pos.get(p, [])}; ctx.stats['mention_join_not_placed'] += 1   # form, full name kept
        if base:
            ctx.stats['mention_joined_rl177'] += 1
        if need:
            catom = hedge(f"{canon}/Cp/focal")
            made = []                                                  # RL-186: one focal atom per replaced occurrence
            mine = set(pos) | set(extra)
            def make(old, made=made, canon=canon, mine=mine):
                if ctx.apos is not None and not ctx.apos.get(id(old), {None}) <= mine:
                    ctx.stats['mention_token_of_other_name_kept'] += 1     # RL-186: the same word in another name stays
                    return old                                             # for that name (GPT 3,5 and GPT 4)
                new = ctx.inherit(hedge(f"{canon}/Cp/focal"), [old]); made.append(new); return new
            mapping = {a: make for a in need}
            if all({p for _, p in ctx.pos[a]} <= covered for a in need):
                edge = replace_all(edge, mapping); how = 'global'
            else:
                ps = {p for p in pos if by_pos.get(p)}
                path = smallest_cover_pos(edge, ps, ctx.apos) if ctx.apos is not None and ps else None
                if path is None:                                       # no position known (a rebuilt atom): by string
                    path = smallest_cover(edge, need)
                if path is None:
                    how = 'not_in_edge'
                else:
                    edge = replace_at(edge, path, mapping); how = 'scoped'
            if how != 'not_in_edge':
                ctx.pos[str(catom)].extend((w, p) for a in need for w, p in ctx.pos.get(a, []) if p in pos)
                # RL-179 (owner 2026-10-22: a model and its tier are one atom): a joined token that lies outside the name's phrase
                # is deleted where its word occurs only there in the unit; it stays (and the name is still the full one) otherwise
                drop = {a for p in extra for a in by_pos.get(p, []) if ctx.pos.get(a) and {q for _, q in ctx.pos[a]} <= set(extra)}
                if drop:
                    new, n = remove_atoms(edge, drop)
                    n = n if new is not None else 0
                    if n:
                        edge = new
                        ctx.pos[str(catom)].extend((w, p) for a in drop for w, p in ctx.pos.get(a, []))
                        if ctx.apos is not None:                       # RL-186: the deleted token's place goes with the name
                            for x in made:
                                if id(x) in ctx.apos:
                                    ctx.apos[id(x)] |= {p for a in drop for _, p in ctx.pos.get(a, [])}
                        ctx.stats['mention_stray_token_deleted'] += 1
                    else:
                        ctx.stats['mention_stray_token_kept'] += 1
                elif extra:
                    ctx.stats['mention_stray_token_kept'] += 1
        mrec.append(dict(term=term, surface=surf, canonical=canon, positions=sorted(set(pos) | set(extra)), replaced=how))
    circuits, cousins, embedded = extract(edge, ctx)
    gloss = [k for k, parts in circuits.items()                         # §6 item 8: "LLM equals large language model."
             if all(urllib.parse.unquote(a.root()) in GLOSS_VERBS or not pred_or_neg(a)
                    for kind, f in parts if kind == 'dummy_sibling' for a in f.atoms())
             and any(urllib.parse.unquote(a.root()) in GLOSS_VERBS for kind, f in parts if kind == 'dummy_sibling' for a in f.atoms())
             and all(is_focal(a) for kind, f in parts if kind != 'dummy_sibling' for a in f.atoms())
             and sum(getattr(f, 'sides', 1) for kind, f in parts if kind != 'dummy_sibling') >= 2]   # a term and its expansion: two sides (merged focal_he count both)
    for k in gloss:
        del circuits[k]; ctx.stats['parent_glossary_dropped'] += 1
    uid = u["uid"]
    parents, pid_of = [], {}
    for k, (key, parts) in enumerate(circuits.items(), 1):
        pid = f"{uid}.P{k}"; pid_of[key] = pid
        kids, n = [], collections.Counter()
        for kind, f in parts:
            n[kind] += 1
            cid = f"{pid}.{ {'dummy_sibling': 'D', 'focal_he': f'F{n[kind]}'}.get(kind, f'S{n[kind]}') }"   # D6: .F1, .F2 ...
            e = str(mk(kind, f))
            kids.append(dict(id=cid, hash=h12(cid, e), kind=kind, edge=e, content_key=content_key(str(a) for a in f.atoms()),
                             atoms=atom_record(f, ctx),
                             **({'modality': f.tags} if f.tags else {})))
        parents.append(dict(id=pid, hash=h12(pid, key), edge=key, content_key=content_key(k["content_key"] for k in kids),
                            children=kids))
    for p in parents:                                   # item 11: embedded link to the outer parent(s)
        outer = [pid_of[b] for b in embedded.get(p["edge"], []) if b in pid_of]
        if outer:
            p["embedded_in"] = outer
    cous = []
    for j, (ck, (kind, f)) in enumerate(cousins.items(), 1):
        kid = f"{uid}.K{j}"
        cous.append(dict(id=kid, hash=h12(kid, ck), kind=kind, edge=ck, content_key=content_key(str(a) for a in f.atoms()),
                         parents=[p["id"] for p in parents], atoms=atom_record(f, ctx),     # link_sentence widens to the sentence
                         **({'modality': f.tags} if f.tags else {})))
    rec = dict(pmcid=art, sid=s["sid"], hash_final=s["hash_final"], uid=uid, unit_hash=u["hash"], text=u["text"],
               focal_mentions=mrec, parents=parents, cousins=cous)
    return rec, ctx


def link_sentence(recs):
    """§6 item 11 (owner 2026-10-21): cousins are scoped by sentence. Every cousin of a sentence is linked (co-occurs) to every
    parent of that sentence, also those of its other units; identical cousin content in several units of a sentence is one
    occurrence, kept in the first unit, the others listed in also_in with their atoms (no word is lost); cousin ids are
    renumbered per unit. recs: the curated units of one sentence, in order. Returns the number of folded cousins."""
    pids = [p["id"] for r in recs for p in r["parents"]]
    first, folded = {}, 0
    for r in recs:
        kept = []
        for c in r["cousins"]:
            if c["edge"] in first:
                first[c["edge"]].setdefault("also_in", []).append(dict(uid=r["uid"], atoms=c["atoms"])); folded += 1
            else:
                first[c["edge"]] = c; kept.append(c)
        for j, c in enumerate(kept, 1):
            c["id"] = f"{r['uid']}.K{j}"; c["hash"] = h12(c["id"], c["edge"]); c["parents"] = list(pids)
        r["cousins"] = kept
    return folded


def occurrences(rec):
    """§6 item 11: one row per hyperedge occurrence (id, hash, entity key = content key, role, uid/sid/pmcid) with typed
    links: contains (child -> its parent), co_occurs (cousin -> each parent of its sentence), embedded (parent -> the outer
    parent in whose argument its clause lies). No parent-parent links of the same sentence (owner)."""
    base = dict(uid=rec["uid"], sid=rec["sid"], pmcid=rec["pmcid"])
    for p in rec["parents"]:
        yield dict(occ=p["id"], hash=p["hash"], key=p["content_key"], role="parent", **base,
                   links=[["embedded", o] for o in p.get("embedded_in", [])])
        for c in p["children"]:
            yield dict(occ=c["id"], hash=c["hash"], key=c["content_key"], role=c["kind"], **base, links=[["contains", p["id"]]])
    for c in rec["cousins"]:
        yield dict(occ=c["id"], hash=c["hash"], key=c["content_key"], role=c["kind"], **base,
                   links=[["co_occurs", q] for q in c["parents"]])


# ------------------------------------------------------------------ checks
# ------------------------------------------------------------------ §6 items 6 and 7: exclusions and corrections
def load_reflexives():
    """{uid: record} of the units whose emphatic reflexive pronouns are restored and re-parsed (§6 item 8)"""
    return {r["uid"]: r for r in (json.loads(l) for l in open(REFLEXIVE_TABLE, encoding="utf-8"))}


REJECT = ({}, {})       # RL-171: ({sid: {term}}, {pmcid: {term}}) from TERM_REJECTIONS, set by run(); empty in the self-test
FORMS = {}              # RL-180: {letters without joiners: written form} from COMPOUND_FORMS, set by run(); empty in the self-test
G4D = {}                # RL-172: {(spelling, pmcid or '' or '*other'): (action, target)} from G4_DECISIONS, set by run()


def g4_decision(term, pmcid):
    """(action, target) for a v3 group-4 mention: an article-specific row first, then '*other' (every other article of a
    spelling decided per article), then the corpus-wide row; None if the spelling is not in the table"""
    if not term.startswith('v3:G4:'):
        return None
    sp = term.split(':', 2)[2]
    return G4D.get((sp, pmcid)) or G4D.get((sp, '*other')) or G4D.get((sp, ''))


def load_g4():
    return {(r['spelling'], r['pmcid']): (r['action'], r['target']) for r in csv.DictReader(open(G4_DECISIONS, encoding='utf-8'))}


def stage_b_rejected(term, sid, art):
    """RL-171/RL-177: a hit Stage B rejected for this sentence or article, compared without case ('BERT' = 'bert')"""
    return term.lower() in REJECT[0].get(sid, ()) or term.lower() in REJECT[1].get(art, ())


def load_rejections():
    """term_rejections.tsv: scope sentence (homonym rule, that sentence) or article (prerelease, F2 verdict, every sentence)"""
    sent, art = collections.defaultdict(set), collections.defaultdict(set)
    for r in csv.DictReader(open(TERM_REJECTIONS, encoding="utf-8"), delimiter="\t"):
        (sent if r["scope"] == "sentence" else art)[r["key"]].add(r["term"].lower())   # RL-177: compared without case
    return dict(sent), dict(art)


def load_lists():
    """(excluded pmcids, {uid: unit hash} of LLM-output units, {uid: (unit hash, [(wrong, right)])}); input hashes checked"""
    for path, want in INPUT_SHA.items():
        if sha(path) != want:
            sys.exit(f"{path}: sha256 {sha(path)}, expected {want}")
    excluded = {r["pmcid"] for r in csv.DictReader(open(SCOPE_EXCLUSIONS, encoding="utf-8"))}
    llm_out = {r["uid"]: r["unit_hash"] for r in csv.DictReader(open(LLM_OUTPUT_UNITS, encoding="utf-8"))}
    corr = {}
    for r in csv.DictReader(open(SPELLING_CORRECTIONS, encoding="utf-8")):
        corr.setdefault(r["uid"], (r["unit_hash"], []))[1].append((r["wrong"], r["right"]))
    return excluded, llm_out, corr


def correct_unit(u, fixes):
    """the unit with each (wrong, right) fixed in its text and in atom2word: the one occurrence of `wrong` standing before
    'model(s)' (same rule as g3_scope_exclusions/build_scope_exclusions.py); one word for one word, so positions hold"""
    u = dict(u)
    for wrong, right in fixes:
        hits = list(re.finditer(r"(?<![A-Za-z])" + re.escape(wrong) + r"(?=\s+[Mm]odel)", u["text"]))
        words = {p: w for _, w, p in u["atom2word"]}
        pos = [p for p, w in words.items() if w == wrong and words.get(p + 1, "").lower().startswith("model")]
        if len(hits) != 1 or len(pos) != 1:
            raise ValueError(f"correction {wrong!r} -> {right!r} not found exactly once in {u['uid']}")
        u["text"] = u["text"][:hits[0].start()] + right + u["text"][hits[0].end():]
        u["atom2word"] = [[at, right if p == pos[0] else w, p] for at, w, p in u["atom2word"]]
    return u


def check_record(rec, u, sentence_pids=None):
    """provenance problems of one output record against its input unit; sentence_pids: the parent ids of its whole sentence
    (item 11; None = the unit's own)"""
    bad = []
    if hashlib.sha1(f"{u['uid']}|{u['text']}".encode("utf-8")).hexdigest()[:12] != u["hash"]:
        bad.append("input unit hash does not verify")
    pids = set()
    for k, p in enumerate(rec["parents"], 1):
        if p["id"] != f"{rec['uid']}.P{k}" or p["hash"] != h12(p["id"], p["edge"]):
            bad.append(f"parent id/hash {p['id']}")
        pids.add(p["id"])
        if str(hedge(('parent',) + tuple(hedge(c["edge"]) for c in p["children"]))) != p["edge"]:
            bad.append(f"children of {p['id']} do not rebuild the parent")
        kinds = [c["kind"] for c in p["children"]]
        if kinds[:1] != ['dummy_sibling'] or 'focal_he' not in kinds:
            bad.append(f"parent {p['id']} without a verb group or a focal_he")
        n = collections.Counter()
        for c in p["children"]:
            n[c["kind"]] += 1
            want = f"{p['id']}." + {'dummy_sibling': 'D', 'focal_he': f"F{n['focal_he']}"}.get(c["kind"], f"S{n['sibling_he']}")
            if c["id"] != want or c["hash"] != h12(c["id"], c["edge"]):
                bad.append(f"child id/hash {c['id']}")
    for p in rec["parents"]:
        if not set(p.get("embedded_in", [])) <= pids - {p["id"]}:
            bad.append(f"embedded link of {p['id']}")
    every = set(sentence_pids) if sentence_pids is not None else pids
    for j, c in enumerate(rec["cousins"], 1):
        if c["id"] != f"{rec['uid']}.K{j}" or c["hash"] != h12(c["id"], c["edge"]) or not c["parents"] or set(c["parents"]) != every:
            bad.append(f"cousin {c['id']}")
        if any(x.is_atom() and is_focal(x) for x in hedge(c["edge"])[1:]):
            bad.append(f"cousin {c['id']} holds a focal term")
    for st in [c for p in rec["parents"] for c in p["children"]] + rec["cousins"]:   # content key = the edge's words, role left out
        if st["content_key"] != content_key(str(x) for x in hedge(st["edge"])[1:]):
            bad.append(f"content key {st['id']}")
    for p in rec["parents"]:
        if p["content_key"] != content_key(c["content_key"] for c in p["children"]):
            bad.append(f"parent content key {p['id']}")
    for struct in [c for p in rec["parents"] for c in p["children"]] + rec["cousins"]:
        for a in struct["atoms"]:
            if a["atom"] != "percent/C/en" and any(x[2] is None for x in a["src"]):
                bad.append(f"atom {a['atom']} in {struct['id']} has no token position")
    return bad


# ------------------------------------------------------------------ run
def run(a):
    tag = f"{a.shard:03d}"
    inp = a.input or SHARDS + f"g2_parsed_{tag}.jsonl"
    os.makedirs(a.outdir, exist_ok=True)
    out_p, err_p, rep_p = (os.path.join(a.outdir, f"g3_{x}_{tag}.{e}") for x, e in (("test", "jsonl"), ("errors", "jsonl"), ("report", "json")))
    np_p, nop_p, art_p, occ_p = (os.path.join(a.outdir, f"g3_{x}_{tag}.jsonl") for x in ("nonprose", "noparent", "articles", "occurrences"))
    db_p = os.path.join(a.outdir, f"g3_test_{tag}.sqlite")
    for p in (out_p, err_p, rep_p, np_p, nop_p, art_p, occ_p) + ((db_p,) if a.db else ()):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    matcher = focal_terms.Matcher()
    excluded, llm_out, corrections = load_lists()
    global REJECT, G4D, FORMS
    REJECT = load_rejections()
    G4D = load_g4()
    FORMS = {r['key']: r['form'] for r in csv.DictReader(open(COMPOUND_FORMS, encoding='utf-8'), delimiter='\t')}
    HUB_FAMS.update(l.strip() for l in open(HUB_FAMILIES, encoding='utf-8') if l.strip() and not l.startswith('#'))
    statements = {r['sid']: r['hash_final'] for r in csv.DictReader(open(STATEMENT_SENTENCES, encoding='utf-8'))}   # RL-172
    reflexives = load_reflexives()
    pos_p = None if a.no_positions else POSITIONS + f"pos_{tag}.jsonl"     # RL-186: exact atom positions of this shard
    positions = {}
    if pos_p:
        for l in open(pos_p, encoding="utf-8"):
            x = json.loads(l)
            if x["status"] == "exact":
                positions[x["uid"]] = (x["hash"], x["pos"])
    nonprose, noparent, articles = [], [], []
    stats, detail = collections.Counter(), collections.defaultdict(collections.Counter)
    surfaces, problems, samples = collections.Counter(), [], []
    hg, stack = None, contextlib.ExitStack()
    if a.db:
        from graphbrain import hopen
        hg = stack.enter_context(hopen(db_p))
    t0 = time.time()
    with open(inp, encoding="utf-8") as f, open(out_p, "w", encoding="utf-8") as fo, open(err_p, "w", encoding="utf-8") as fe, \
            open(occ_p, "w", encoding="utf-8") as fx:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            r = json.loads(line)
            stats['articles'] += 1
            if r["pmcid"] in excluded:                                  # §6 item 6: out of scope, left out of G3 and M1
                stats['articles_excluded'] += 1
                articles.append(dict(pmcid=r["pmcid"], status="excluded", units=0, parents=0))
                continue
            art_units = art_parents = 0
            units = []
            for s in r["sentences"]:
                if s["sid"] in statements:                            # RL-172: an AI-use statement (second check)
                    stats['sentences_ai_use_statement'] += 1
                    if statements[s["sid"]] != s["hash_final"]:
                        problems.append(f"{s['sid']}: statement list hash_final {statements[s['sid']]} != {s['hash_final']}")
                    continue
                for u in s["units"]:
                    if not u["main_edge"]:
                        continue
                    if u["uid"] in llm_out:                             # §6 item 7: reproduces LLM output
                        stats['units_llm_output'] += 1
                        if llm_out[u["uid"]] != u["hash"]:
                            problems.append(f"{u['uid']}: LLM-output list hash {llm_out[u['uid']]} != unit hash {u['hash']}")
                        continue
                    kind = nonprose_kind(u["text"])
                    if kind:                                            # §6 item 8: not plain text
                        stats['units_nonprose_' + kind] += 1; nonprose.append((kind, u["uid"], u["text"]))
                        continue
                    if u["uid"] in positions:                       # RL-186
                        if positions[u["uid"]][0] == u["hash"]:
                            u = dict(u, atom_pos=positions[u["uid"]][1])
                        else:
                            problems.append(f"{u['uid']}: positions file hash {positions[u['uid']][0]} != unit hash {u['hash']}")
                    cu = u
                    if u["uid"] in reflexives:                         # §6 item 8: emphatic reflexive restored, re-parsed
                        rr = reflexives[u["uid"]]
                        if rr["unit_hash"] != u["hash"] or rr["text_g2"] != u["text"]:
                            problems.append(f"{u['uid']}: reflexive table does not match the unit")
                        else:
                            cu = dict(u, text=rr["text_restored"], main_edge=rr["main_edge"], extra_edges=rr["extra_edges"],
                                      atom2word=rr["atom2word"], atom_pos=None); stats['units_reflexive_restored'] += 1
                    if u["uid"] in corrections:                         # §6 item 6: corrected before focal matching
                        h, fixes = corrections[u["uid"]]
                        if h != u["hash"]:
                            problems.append(f"{u['uid']}: correction list hash {h} != unit hash {u['hash']}")
                        else:
                            cu = correct_unit(cu, fixes); stats['units_corrected'] += 1
                    units.append((s, u, cu))
            anchored = any(matcher.anchor_hits(cu["text"]) for _, _, cu in units)    # article evidence from the kept units
            for sid, group in itertools.groupby(units, key=lambda x: x[0]["sid"]):  # item 11: one sentence at a time
                done = []
                for s, u, cu in group:
                    stats['units'] += 1
                    try:
                        rec, ctx = curate_unit(r["pmcid"], s, cu, matcher, anchored, a.modals)
                        if cu is not u:
                            rec["text_original"] = u["text"]
                            if u["uid"] in corrections:
                                rec["corrections"] = corrections[u["uid"]][1]
                            if u["uid"] in reflexives:
                                rec["reflexive_restored"] = reflexives[u["uid"]]["restorations"]
                    except Exception as e:                              # RecursionError included
                        stats['unit_errors'] += 1; stats['error_' + type(e).__name__] += 1
                        fe.write(json.dumps(dict(uid=u["uid"], error=type(e).__name__, message=str(e)[:500])) + "\n")
                        continue
                    done.append((u, rec, ctx))
                stats['cousins_folded_in_sentence'] += link_sentence([rec for _, rec, _ in done])
                spids = [p["id"] for _, rec, _ in done for p in rec["parents"]]
                stats['sentences_with_parent'] += bool(spids)
                stats['sentences_parents_in_2plus_units'] += sum(bool(rec["parents"]) for _, rec, _ in done) > 1
                for u, rec, ctx in done:
                    # owner 2026-10-10: a unit without a parent is removed; item 11: unless its sentence has a parent and it
                    # holds cousins of that sentence
                    if rec["parents"] or (spids and rec["cousins"]):
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        for o in occurrences(rec):
                            fx.write(json.dumps(o, ensure_ascii=False) + "\n"); stats['occurrence_rows'] += 1
                        art_units += 1; art_parents += len(rec["parents"])
                        stats['units_cousins_only'] += not rec["parents"]
                    else:
                        stats['units_removed_no_parent'] += 1
                        noparent.append(dict(uid=u["uid"], focal=bool(rec["focal_mentions"]), text=rec["text"]))
                    stats.update(ctx.stats)
                    for k, c in ctx.detail.items():
                        detail[k].update(c)
                    for m in rec["focal_mentions"]:
                        surfaces[(m["surface"], m["canonical"])] += 1; stats['mention_' + m["replaced"]] += 1
                    stats['focal_mentions'] += len(rec["focal_mentions"])
                    stats['units_with_parent'] += bool(rec["parents"]); stats['parents'] += len(rec["parents"])
                    if rec["parents"] or spids:
                        stats['cousins'] += len(rec["cousins"])
                        stats['parents_embedded'] += sum(bool(p.get("embedded_in")) for p in rec["parents"])
                        stats['parents_2plus_focal_he'] += sum(sum(c["kind"] == 'focal_he' for c in p["children"]) > 1
                                                               for p in rec["parents"])
                    stats['units_focal_no_parent'] += bool(rec["focal_mentions"]) and not rec["parents"]
                    kept = bool(rec["parents"] or (spids and rec["cousins"]))   # a removed unit's cousins belong to no one
                    bad = check_record(rec if kept else dict(rec, cousins=[]), u, spids)   # unit hash verified on the original text
                    problems += [f"{u['uid']}: {b}" for b in bad]
                    if rec["parents"] and len(samples) < 400:
                        samples.append(rec)
                    if hg is not None and (rec["parents"] or spids):
                        for p in rec["parents"]:
                            hg.add(hedge(('source_core', f"{r['pmcid']}::{u['uid']}::{p['hash']}", hedge(p["edge"]))))
                        for c in rec["cousins"]:
                            hg.add(hedge(('source_periphery', f"{r['pmcid']}::{u['uid']}::{c['hash']}", hedge(c["edge"]))))
            # owner 2026-10-10: an article none of whose units has a parent is invalid (M1 leaves it out of every relation)
            status = "valid" if art_parents else "invalid_no_parent"
            stats['articles_' + status] += 1
            articles.append(dict(pmcid=r["pmcid"], status=status, units=art_units, parents=art_parents))
    secs = time.time() - t0
    stack.close()
    if hg is not None:
        from graphbrain import hopen
        with hopen(db_p) as hg2:
            n_core = n_per = 0
            for link in hg2.search(('source_core', '*', '*')):
                parts = str(link[1]).split('::')
                n_core += 1
                if len(parts) != 3 or not all(not c.is_atom() for c in link[2][1:]):
                    problems.append(f"db: source_core link not readable in chunk12's way: {str(link[1])}")
            for link in hg2.search(('source_periphery', '*', '*')):
                n_per += 1
                if len(str(link[1]).split('::')) != 3:
                    problems.append(f"db: source_periphery id not readable in chunk12's way: {str(link[1])}")
        stats['db_source_core'], stats['db_source_periphery'] = n_core, n_per
    with open(np_p, "w", encoding="utf-8") as fn:                    # §6 item 8, for review
        for kind, uid, text in nonprose:
            fn.write(json.dumps(dict(kind=kind, uid=uid, text=text), ensure_ascii=False) + "\n")
    with open(nop_p, "w", encoding="utf-8") as fn:                   # removed units without a parent, for review
        for x in noparent:
            fn.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(art_p, "w", encoding="utf-8") as fn:                   # one status row per article of the shard, for M1
        for x in articles:
            fn.write(json.dumps(x) + "\n")
    report = dict(stage="G3 test v2", input=inp, script=os.path.abspath(__file__), script_sha=sha(__file__), modals=a.modals,
                  host=os.uname().nodename, job=os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
                  task=os.environ.get("SLURM_ARRAY_TASK_ID"), inputs={p: h for p, h in INPUT_SHA.items()},
                  positions=dict(file=pos_p, sha=sha(pos_p) if pos_p else None, units_exact=len(positions)),
                  limit=a.limit, seconds=round(secs, 1), seconds_per_unit=round(secs / max(stats['units'], 1), 4), counts=dict(stats),
                  focal_surface_to_canonical=[[s, c, n] for (s, c), n in surfaces.most_common()],
                  detail={k: dict(v.most_common(60)) for k, v in detail.items()}, problems=problems[:200], n_problems=len(problems))
    json.dump(report, open(rep_p, "w"), indent=1, ensure_ascii=False)
    random.Random(1).shuffle(samples)
    print(json.dumps(dict(stats), indent=0))
    for rec in samples[:a.show]:
        print("\n" + "=" * 100 + f"\n{rec['uid']}  {rec['text'][:400]}")
        for m in rec["focal_mentions"]:
            print(f"   focal: {m['surface']!r} -> {m['canonical']} ({m['replaced']})")
        for p in rec["parents"]:
            print(f"   {p['id']}  {p['edge']}")
            for c in p["children"]:
                if c.get("modality"):
                    print(f"      {c['id']} modality {c['modality']}")
        for c in rec["cousins"][:6]:
            print(f"   {c['id']} <- {','.join(x.rsplit('.', 1)[1] for x in c['parents'])}  {c['edge']}")
    print(f"\n{stats['units']} units in {secs:.0f} s; problems: {len(problems)}; report {rep_p}")
    for p in problems[:20]:
        print("  PROBLEM", p)
    return 1 if problems else 0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ self-test
CANON_CASES = [('LLMs', 'llm'), ('large language models', 'llm'), ('Large Language Model', 'llm'), ('LLM', 'llm'),
               ('language models', 'language_model'), ('Transformer models', 'transformer_model'), ('Chat-GPT', 'chatgpt'),
               ('Chat GPT', 'chatgpt'), ('ChatGPT', 'chatgpt'), ('GPT-4', 'gpt_4'), ('GPT4', 'gpt_4'), ('GPT 4', 'gpt_4'),
               ('GPT-4.0', 'gpt_4'), ('GPT-4o', 'gpt_4o'), ('GPT4o', 'gpt_4o'), ('GPT-3.5', 'gpt_3_5'), ('gpt3.5', 'gpt_3_5'),
               ('Llama-2', 'llama_2'), ('Llama2', 'llama_2'), ('LLaMA 2', 'llama_2'), ('Llama-3.1', 'llama_3_1'), ('LLaMA', 'llama'),
               ('Bidirectional encoder representations from transformers', 'bert'), ('BioT5+', 'biot5_plus'),
               ('ChatGPT-3.5', 'gpt_3_5'), ('ChatGPT-4', 'gpt_4'), ('ChatGPT-4o', 'gpt_4o'), ('ProstT5', 'prostt5'), ('ESM-2', 'esm_2'),
               ('Google Gemini', 'gemini'), ('Mistral-7B', 'mistral'), ('Llama-2-70B', 'llama_2'), ('Llama 3.1 8B', 'llama_3_1'),
               ('Mixtral 8x7B', 'mixtral'), ('gpt-4o-2024', 'gpt_4o'), ('GPT-4-0613', 'gpt_4'), ('gpt-4o-2024-05-13', 'gpt_4o'),
               ('Qwen2.5', 'qwen_2_5'), ('GPT-3', 'gpt_3'), ('BioMedGPT-10B', 'biomedgpt'), ('Qwen2.5-1.5B', 'qwen_2_5'),
               ('Llama-3.2-1B', 'llama_3_2'), ('Chat GPT-4.0', 'gpt_4'), ('ESM-1b', 'esm_1b'), ('ESM-MSA-1b', 'esm_msa_1b'),
               ('ESM-1v', 'esm_1v'),
               # RL-172: checkpoint names written like their prose form
               ('bert-base-uncased', 'bert'), ('bert-base-chinese', 'bert'), ('roberta-large-mnli', 'roberta'), ('llama-7b', 'llama'),
               ('gemma-3-12b-it', 'gemma_3'), ('gemini-2.5-pro', 'gemini_2_5_pro'), ('gemini-1.5-flash-002', 'gemini_1_5_flash'),
               ('mistral-7b-instruct', 'mistral'), ('BERT-based', 'bert'), ('LLaMA-based', 'llama'), ('gemma-3n-e4b-it', 'gemma_3n'),
               ('GPT-4-turbo', 'gpt_4_turbo'), ('gpt-4o-mini', 'gpt_4o_mini'), ('Llama-3.1-8B-Instruct', 'llama_3_1'), ('BERT-CRF', 'bert_crf'),
               ('bert-large', 'bert'), ('llama-2', 'llama_2'), ('GPT-3.5-turbo', 'gpt_3_5_turbo'), ('gpt-35-turbo', 'gpt_3_5_turbo'),
               # RL-177: tiers kept, every model name folded, glued BERT-family settings, Claude family + tier + version
               ('BERTbase', 'bert'), ('RoBERTalarge', 'roberta'), ('phobertbase', 'phobert'), ('xlm-roberta-base', 'xlm_roberta'),
               ('deberta-v3-large', 'deberta_v3'), ('distilbert-base-uncased', 'distilbert'), ('albert-base-v2', 'albert'),
               ('meta-llama-3-8b-instruct', 'llama_3'), ('Claude 3.5 Sonnet', 'claude_sonnet_3_5'),
               ('Claude Sonnet 4.5', 'claude_sonnet_4_5'), ('claude-3-5-sonnet-20241022', 'claude_sonnet_3_5'),
               ('Claude-3-Opus', 'claude_opus_3'), ('Gemini 2.5 Pro', 'gemini_2_5_pro'), ('GPT-4 Turbo', 'gpt_4_turbo'),
               ('GPT-4-1106-preview', 'gpt_4'), ('acegpt-7b-chat', 'acegpt'), ('hermes-2-pro-mistral', 'hermes_2_pro_mistral'),
               ('e5-mistral-7b-ablation', 'e5_mistral'), ('multilingual BERT', 'multilingual_bert'), ('Robert', 'robert'),
               ('Gemini Pro 2.5', 'gemini_2_5_pro'), ('Claude 4 Opus', 'claude_opus_4'),
               # RL-179: vendor names replaced by the model family
               ('OpenAI o1', 'gpt_o1'), ('OpenAI o1 pro', 'gpt_o1_pro'), ('ChatGPT o3-mini', 'gpt_o3_mini'), ('GPT-o1', 'gpt_o1'),
               ('Google Gemini 2.5 Pro', 'gemini_2_5_pro'), ('Anthropic Claude 3.5 Sonnet', 'claude_sonnet_3_5'), ('Meta Llama 3', 'llama_3'),
               ('OpenAI GPT-4.1 mini', 'gpt_4_1_mini'), ('BERT-mini', 'bert'), ('RoBERTa-Mini', 'roberta'), ('ProkBERT-mini', 'prokbert'), ('Microsoft Copilot', 'microsoft_copilot'), ('HuatuoGPT-o1', 'huatuogpt_o1'),
               ('ChatGPTo1 pro', 'gpt_o1_pro'), ('GPTo1', 'gpt_o1'), ('base-BERT', 'bert'), ('Base-BERT', 'bert'), ('large-BERT', 'bert'), ('uncased-BERT', 'bert'),
               ('Tiny-BERT', 'tiny_bert'), ('Instruct-GPT', 'instruct_gpt'), ('Mistral7B', 'mistral'), ('Gemma27B', 'gemma'),
               ('GPT4b', 'gpt_4'), ('GPT2M', 'gpt_2'), ('Base-BiFormer-GPT', 'base_biformer_gpt'),
               ('large language models', 'llm'), ('small language model', 'small_language_model'),
               ('GPT-OSS-120B.', 'gpt_oss'), ('gpt-oss-20b', 'gpt_oss'), ('GPT-J-6B', 'gpt_j'), ('claude-3-5-sonnet-20', 'claude_sonnet_3_5'),
               ('cardiffnlp/twitter-roberta-base-sentiment', 'twitter_roberta'), ('biobert-v1.18', 'biobert_v1_1'), ('BioBERT v1.123', 'biobert_v1_1'), ('BioBERT v1.2', 'biobert_v1_2')]


def unit(text, edge, words, lemmas):
    """synthetic G2 v3 unit; words = [(atom, word)] in text order, lemmas = {atom: lemma}"""
    a2w = [[a, w, i] for i, (a, w) in enumerate(words)]
    simple = lambda a: f"{a.split('/')[0]}/{a.split('/')[1][:1]}/en"
    extra = [f"(_lemma {simple(a)} {lemmas.get(a, a.split('/')[0])}/{a.split('/')[1][:1]}/en)" for a, _ in words]
    uid = "PMCX.r1.L1.S1.U1"
    return dict(uid=uid, hash=hashlib.sha1(f"{uid}|{text}".encode()).hexdigest()[:12], text=text, main_edge=edge, extra_edges=extra,
                atom2word=a2w)


UNIT_CASES = [
    ("main-verb 'is' kept, focal LLM",
     unit("An LLM is a type of AI.", "(is/Pd.sc/en (an/Md/en llm/Cp.s/en) (a/Md/en (of/Br.ma/en type/Cc.s/en ai/Cp.s/en)))",
          [("an/Md/en", "An"), ("llm/Cp.s/en", "LLM"), ("is/Pd.sc/en", "is"), ("a/Md/en", "a"), ("type/Cc.s/en", "type"),
           ("of/Br.ma/en", "of"), ("ai/Cp.s/en", "AI")], {"is/Pd.sc/en": "be"}),
     dict(parent_contains=["be/P/en", "llm/C/focal", "type/C/en", "ai/C/en"], parent_lacks=["an/", "a/M"])),
    ("auxiliaries have/been dropped, main verb kept",
     unit("LLMs have been widely used.", "((have/Mv.<f---/en (been/Mv.<pf--/en (widely/M/en used/Pd.<pf--/en))) llms/Cp.p/en)",
          [("llms/Cp.p/en", "LLMs"), ("have/Mv.<f---/en", "have"), ("been/Mv.<pf--/en", "been"), ("widely/M/en", "widely"),
           ("used/Pd.<pf--/en", "used")], {"have/Mv.<f---/en": "have", "been/Mv.<pf--/en": "be", "used/Pd.<pf--/en": "use"}),
     dict(parent_contains=["use/P/en", "llm/C/focal", "widely/M/en"], parent_lacks=["have/", "be/"])),
    ("negation kept, auxiliary do dropped",
     unit("LLMs do not reduce errors.", "((do/Mv.<f---/en (not/Mn/en reduce/Pd.so/en)) llms/Cp.p/en errors/Cc.p/en)",
          [("llms/Cp.p/en", "LLMs"), ("do/Mv.<f---/en", "do"), ("not/Mn/en", "not"), ("reduce/Pd.so/en", "reduce"),
           ("errors/Cc.p/en", "errors")], {"reduce/Pd.so/en": "reduce", "errors/Cc.p/en": "error"}),
     dict(parent_contains=["not/M/en", "reduce/P/en", "error/C/en"], parent_lacks=["do/"])),
    ("modal inside the verb group, determiner 'no' kept, percent kept, connective dropped, plain-text number kept (§6 item 8)",
     unit("GPT-4 can reach no 50 % error, e.g. here.",
          "((can/Mm/en reach/Pd.so/en) gpt-4/Cp.s/en (no/Md/en (+/B.ma/. (50/M#/en %25/Cc.s/en) error/Cc.s/en)) e%2eg%2e/M/en)",
          [("gpt-4/Cp.s/en", "GPT-4"), ("can/Mm/en", "can"), ("reach/Pd.so/en", "reach"), ("no/Md/en", "no"), ("50/M#/en", "50"),
           ("%25/Cc.s/en", "%"), ("error/Cc.s/en", "error"), ("e%2eg%2e/M/en", "e.g.")], {}),
     dict(parent_contains=["(dummy_sibling can/M/en reach/P/en)", "gpt_4/C/focal", "no/M/en", "percent/C/en", "error/C/en", "50/M/en"],
          parent_lacks=["e%2eg"])),
    ("compound verb group: should + passive be + main verb (§6 item 1: be kept in passives)",
     unit("The use of LLMs should be approached with caution.",
          "((should/Mm/en (be/Mv.-i-----/en approached/Pd.xpx.<pf----/en)) (of/Br.ma/en (the/Md/en use/Cc.s/en) llms/Cp.p/en) "
          "(with/T/en caution/Cc.s/en))",
          [("the/Md/en", "The"), ("use/Cc.s/en", "use"), ("of/Br.ma/en", "of"), ("llms/Cp.p/en", "LLMs"), ("should/Mm/en", "should"),
           ("be/Mv.-i-----/en", "be"), ("approached/Pd.xpx.<pf----/en", "approached"), ("with/T/en", "with"), ("caution/Cc.s/en", "caution")],
          {"approached/Pd.xpx.<pf----/en": "approach", "be/Mv.-i-----/en": "be"}),
     dict(parent_contains=["(dummy_sibling approach/P/en be/M/en should/M/en)", "llm/C/focal"], parent_lacks=["be/P"])),
    ("lexical modal 'allows' merged with its complement (§6 item 3); modal inside a dummy cousin",
     unit("LINS allows users to integrate LLMs and can adapt.",
          "(and/J/en (allows/Pd.sx/en lins/Cp.s/en ((to/Mi/en integrate/P.so/en) users/Cc.p/en llms/Cp.p/en)) "
          "((can/Mm/en adapt/Pd/en) lins/Cp.s/en))",
          [("lins/Cp.s/en", "LINS"), ("allows/Pd.sx/en", "allows"), ("users/Cc.p/en", "users"), ("to/Mi/en", "to"),
           ("integrate/P.so/en", "integrate"), ("llms/Cp.p/en", "LLMs"), ("and/J/en", "and"), ("can/Mm/en", "can"),
           ("adapt/Pd/en", "adapt")], {"allows/Pd.sx/en": "allow", "users/Cc.p/en": "user"}),
     dict(parent_contains=["(dummy_sibling allow/P/en integrate/P/en)", "llm/C/focal", "user/C/en", "lins/C/en"],
          cousins_contain=["(dummy_cousin adapt/P/en can/M/en)"])),
    ("stop-listed main verb 'been' kept (the RDS stop list has 'been')",
     unit("LLMs have been instrumental.", "((have/Mv.<f---/en been/Pd.sc/en) llms/Cp.p/en instrumental/Ca/en)",
          [("llms/Cp.p/en", "LLMs"), ("have/Mv.<f---/en", "have"), ("been/Pd.sc/en", "been"), ("instrumental/Ca/en", "instrumental")],
          {"been/Pd.sc/en": "be"}),
     dict(parent_contains=["be/P/en", "llm/C/focal", "instrumental/C/en"], parent_lacks=["have/"])),
    ("multi-word focal term merged into one atom inside its builder",
     unit("Large language models outperform the baseline.",
          "(outperform/Pd.so/en (+/B.am/. large/Ma/en (+/B.am/. language/Ma/en models/Cc.p/en)) (the/Md/en baseline/Cc.s/en))",
          [("large/Ma/en", "Large"), ("language/Ma/en", "language"), ("models/Cc.p/en", "models"), ("outperform/Pd.so/en", "outperform"),
           ("the/Md/en", "the"), ("baseline/Cc.s/en", "baseline")], {}),
     dict(parent_contains=["outperform/P/en", "llm/C/focal", "baseline/C/en"], parent_lacks=["large/", "language/", "models/"])),
]


WORD_CASES = [   # RL-180: (root as G2/lemma gives it, final root or None); compounds use the small FORMS below
    ('bio\u00admedical', 'biomedical'), ('\u200b\u200bopenai', 'openai'), ('follows:•', 'follows'), ('study.•', 'study'),
    ('〈sep〉', 'sep'), ('h˜i', 'hi'), ('𝕏s', 'xs'), ('†note', 'note'), ('corpus?†', 'corpus'), ('/group', 'group'),
    ('vision:90b', 'vision'), ('cdk4/6', 'cdk4/6'), ('naïve', 'naive'), ('yılmaz', 'yilmaz'), ('łódź', 'lodz'),
    ('tcrβ', 'tcrbeta'), ('aβ42', 'abeta42'), ('hif_1α', 'hif_1alpha'), ('α_helix', 'alpha_helix'), ('μct', 'microct'),
    ('δauc', None), ('μhuman', None), ('∆mcc', None), ('scoreβ', None), ('pre-', 'pre'), ('covid-19', 'covid_19'),
    ('pretrained', 'pre_trained'), ('pre_trained', 'pre_trained'), ('pro_state', 'pro_state'), ('co_pilot', 'co_pilot'),
    ('re_present', 're_present')]
WORD_FORMS = {'pretrained': 'pre_trained', 'prostate': 'prostate', 'copilot': 'copilot', 'represent': 'represent'}
NOTATION_CASES = [   # RL-181: (unit text, removed as notation?)
    ('where H is the diagnostic hypothesis; x is the clinical input vector', True),
    ('M denotes the total number of transformer encoder layers in BERT.', True), ('Let Ri denote the ith response by LLM.', True),
    ('Here, denotes the vocabulary of ClinicalBERT.', True), ('Post denotes whether a month occurs after ChatGPT.', True),
    ('Saprot represents the inaugural universal PLM.', False), ('Enformer represents a significant advance.', False),
    ('M3 indicates that the fusion module works.', False), ('This represents a significant knowledge gap given LLMs.', False),
    ('Here we present GeneInsight, a tool.', False), ('Here are ten simple rules for using large language models:', False),
    ('Where ChatGPT is used, results improve.', False)]


def selftest():
    bad = 0
    global REJECT
    saved, REJECT = REJECT, ({'S1': {'bert'}}, {'A1': {'gpt-5'}})                # RL-177: rejections compared without case
    for term, sid, art, want in (('BERT', 'S1', 'A9', True), ('bert', 'S1', 'A9', True), ('BERT', 'S2', 'A9', False),
                                 ('GPT-5', 'S2', 'A1', True), ('GPT-4', 'S2', 'A1', False)):
        ok = stage_b_rejected(term, sid, art) == want; bad += not ok
        print("PASS " if ok else "FAIL ", f"stage_b_rejected({term!r}, {sid}, {art}) = {not want if not ok else want}")
    REJECT = saved
    for raw, want in WORD_CASES:
        c = clean_root(raw); got = norm_word(c, WORD_FORMS) if c else None; ok = got == want; bad += not ok
        print("PASS " if ok else "FAIL ", f"norm_word({raw!r}) = {got!r}" + ("" if ok else f", expected {want!r}"))
    for text, want in NOTATION_CASES:
        got = nonprose_kind(text) == 'notation'; ok = got == want; bad += not ok
        print("PASS " if ok else "FAIL ", f"notation({text[:50]!r}) = {got}")
    for s, want in CANON_CASES:
        got = canonical(s); ok = got == want; bad += not ok
        print("PASS " if ok else "FAIL ", f"canonical({s!r}) = {got!r}" + ("" if ok else f", expected {want!r}"))
    matcher = focal_terms.Matcher()
    for name, u, want in UNIT_CASES:
        try:
            rec, _ = curate_unit("PMCX", dict(sid="PMCX.r1.L1.S1", hash_final="0" * 12), u, matcher, True, 'keep')
        except Exception as e:
            bad += 1; print("FAIL ", name, "exception", repr(e)); continue
        txt = " ".join(p["edge"] for p in rec["parents"])
        errs = [f"missing {x}" for x in want.get("parent_contains", []) if x not in txt]
        errs += [f"unexpected {x}" for x in want.get("parent_lacks", []) if x in txt]
        ctxt = " ".join(c["edge"] for c in rec["cousins"])
        errs += [f"missing cousin {x}" for x in want.get("cousins_contain", []) if x not in ctxt]
        mods = [m for p in rec["parents"] for c in p["children"] for m in c.get("modality", [])]
        if "modality" in want and mods != want["modality"]:
            errs.append(f"modality {mods}")
        errs += check_record(rec, u)
        if not rec["parents"]:
            errs.append("no parent")
        bad += bool(errs)
        print("PASS " if not errs else "FAIL ", name, "" if not errs else f"{errs}\n      {txt}")
    n = len(CANON_CASES) + len(UNIT_CASES) + 5 + len(WORD_CASES) + len(NOTATION_CASES)
    print(f"\n{n - bad} of {n} passed")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100, help="articles (0 = the whole shard)")
    ap.add_argument("--outdir")
    ap.add_argument("--input", help="a G2 v3 JSONL file instead of the shard file")
    ap.add_argument("--no-positions", action="store_true", help="RL-186: positions by atom string (tests without a positions file)")
    ap.add_argument("--modals", choices=("keep", "tag", "drop"), default="keep")
    ap.add_argument("--db", action="store_true", help="also write the graphbrain database in chunk12's form")
    ap.add_argument("--show", type=int, default=8, help="random curated units printed for reading")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(1 if selftest() else 0)
    if not a.outdir:
        sys.exit("--outdir is required")
    sys.exit(run(a))


if __name__ == "__main__":
    main()
