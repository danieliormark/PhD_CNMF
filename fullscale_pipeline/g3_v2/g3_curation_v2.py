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
  provenance: every structure has an id and hash sha1("<id>|<edge>")[:12] (the G2 unit-hash form): parent <uid>.P<k>, its children
      <uid>.P<k>.D (dummy_sibling), .F (focal_he), .S<j> (sibling_he), cousins <uid>.K<j> with the parents they were swept against (the unit is
      the grandparent edge that holds a parent and its cousins); every output atom lists the source atoms, words and token positions it
      comes from. Unit -> sentence (sid, hash_final) -> article (pmcid) as in G2.

    python g3_curation_test.py --selftest
    python g3_curation_test.py --shard 0 --limit 200 --outdir DIR [--modals tag|drop|keep] [--db]

Writes DIR/g3_test_NNN.jsonl (one record per unit), g3_errors_NNN.jsonl, g3_report_NNN.json and, with --db, g3_test_NNN.sqlite in the form
chunk12 reads (('source_core', 'pmcid::uid::hash', parent), ('source_periphery', 'pmcid::uid::hash', cousin)). Refuses to overwrite.
Checks at the end (exit 1 on any problem): input unit hashes, every output id and hash, every atom traced to a token position, every
cousin's parents present, and with --db the database read back in chunk12's way.
"""
import argparse, collections, contextlib, csv, hashlib, json, os, random, re, sys, time, urllib.parse

sys.setrecursionlimit(20000)
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
PP = "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing/"
SHARDS = PG + "g2_v3/shards/"
STOPWORDS = PG + "nltk_abridged_stopwords_list.txt"
sys.path.insert(0, PP)
import focal_terms                                                     # noqa: E402  P2's matcher, the single source of the focal terms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal_merge                                                     # noqa: E402  §6 item 3 (decision rule and edge rewrite)
from graphbrain import hedge                                           # noqa: E402

STOP = {w.strip() for w in open(STOPWORDS, encoding="utf-8") if w.strip()}
CUSTOM_NOISE = {'such', 'other', 'many', 'some', 'any', 'own', 'same', 'few', 'several', 'this', 'that', 'these', 'those', 'whilst'}
NEGATION = {'not', 'no', 'never', "n't", 'nor'}
AUXILIARIES = {'be', 'have', 'do'}
MODAL_WORDS = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'ca', 'wo', "'ll"}   # ca/wo: can't, won't
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
ADVERB_ROLES_SHA = "f66d153f5efa"
ADV_ROLE = {r["word"]: r["action"] for r in csv.DictReader(open(ADVERB_ROLES_PATH, encoding="utf-8"), delimiter="\t")}
# the 12 rows marked "open question 5" (owner 2026-10-09, option 3): direction words fuse with their verb like the item-5
# particles ("look_ahead", "move_forward", "leave_behind") and otherwise stay a modifier in their phrase ("a step forward");
# time uses (before, throughout) dropped like the table's time adverbs; around/under kept only before a number (numbers
# rule), otherwise dropped; despite/beside dropped
DIRECTION_PARTICLES = {'forward', 'forwards', 'ahead', 'behind', 'back', 'away'}
ADV_ROLE.update({w: 'drop' for w in ('before', 'throughout', 'around', 'under', 'despite', 'beside')})
ADV_ROLE.update({w: 'direction' for w in DIRECTION_PARTICLES})
CLAUSE_END = re.compile(r"^\s*(?:[.,;:)]|$)")
# §6 items 6 and 7: articles out of scope, units reproducing LLM output, word-for-word spelling corrections (RDS copies)
SCOPE_EXCLUSIONS = PG + "g3_scope_exclusions/scope_exclusions.csv"
SPELLING_CORRECTIONS = PG + "g3_scope_exclusions/spelling_corrections.csv"
LLM_OUTPUT_UNITS = PG + "g3_llm_output/llm_output_units.csv"
REFLEXIVE_TABLE = PG + "g3_reflexive/reflexive_restored.jsonl"      # §6 item 8, built by g3_v2/reflexive_restore.py
INPUT_SHA = {SCOPE_EXCLUSIONS: "5b71d844d9af", SPELLING_CORRECTIONS: "d0a70f6c1a20", LLM_OUTPUT_UNITS: "b71d5717e250",
             ADVERB_ROLES_PATH: ADVERB_ROLES_SHA}
INPUT_SHA[REFLEXIVE_TABLE] = "d121a203daec"
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
VERSION_TOKEN = re.compile(r'^(?:v\d+(?:\.\d+)*|[ro]\d+|\d(?:\.\d{1,2})?[a-z]{0,2})$', re.I)
HYPHEN_CHARS = re.compile('[\u2010\u2011\u2013\u2014]')


def clean_root(word):
    """§6 item 8: an atom's word cleaned of glued citation digits ('self_education,2', 'nccn)3'), edge punctuation, and with a
    Unicode hyphen inside it written as '_'; None if no letter is left"""
    w = HYPHEN_CHARS.sub('_', word)
    w = re.sub(r'[,;.)\]]+\d+$', '', w)
    w = w.strip('.,;:()[]\'"')
    return w if re.search(r'[a-z]', w) else None


# §6 item 8: units that parse to a parent but are not prose; glossary parents
END_PUNCT = re.compile(r'[.?!;:"”’\')\]]\s*$')
TITLE_SMALL = {'of', 'the', 'and', 'in', 'for', 'with', 'on', 'to', 'a', 'an', 'by', 'from', 'vs', 'versus', 'or', 'at', 'as', 'via'}
SECTION_NO = re.compile(r'^\s*\d+\.\d+(?:\.\d+)*\.?\s+[A-Z]')
REFERENCE_ENTRY = re.compile(r"^\s*\[?\d+\]?\.?\s+(?:(?:[A-Z]\.\s?){1,3}\s*[A-Z][A-Za-z'\-]+|[A-Z][A-Za-z'\-]+,?\s+(?:[A-Z]\.\s?){1,3})[,.]")
PSEUDOCODE = re.compile(r'←|//|:=')
PANEL_CAPTION = re.compile(r'^[A-H]\s+[A-Z][a-z]+[^;]*;\s*[B-H]\s+[A-Z][a-z]+')
GLOSS_VERBS = {'equal', 'indicate', 'denote', 'stand', 'refer'}


def nonprose_kind(text):
    """§6 item 8: 'heading', 'reference', 'pseudocode', 'panel_caption' or None"""
    words = re.findall(r"[A-Za-z][A-Za-z\-_']*", text)
    if PSEUDOCODE.search(text):
        return 'pseudocode'
    if REFERENCE_ENTRY.match(text):
        return 'reference'
    if PANEL_CAPTION.match(text):
        return 'panel_caption'
    if len(words) <= 20 and (SECTION_NO.match(text) and (not END_PUNCT.search(text) or text.rstrip().endswith('?'))):
        return 'heading'
    content = [w for w in words if w.lower() not in TITLE_SMALL]
    if len(words) <= 20 and content and not END_PUNCT.search(text) and sum(w[0].isupper() for w in content) / len(content) >= 0.6:
        return 'heading'
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
}
VERSIONED = r'(chatgpt|gpt|llama|qwen|claude|gemini|grok|palm|gemma|mistral|deepseek)'
VERSION_SUFFIX = re.compile(r'^[-‐– ]?(?:v?\d(?:\.\d{1,2})?[a-z]?|\d+(?:\.\d+)?[bm])$', re.I)   # one-digit major: "GPT-44" is GPT-4 + citation 4
SIZE_IN_NAME = {'esm-1b', 'esm-msa-1b', 'protst-esm-1b'}       # focal-list names whose "1b" is not a model size


def canonical(surface):
    """one canonical atom root per focal term: lower case, separators -> '_', '.0' dropped, a glued version split ('gpt4o' -> 'gpt_4o'),
    SYNONYMS applied. 'GPT-4', 'GPT4', 'GPT 4', 'GPT-4.0' -> gpt_4; 'LLMs', 'large language models' -> llm; 'GPT-4o' stays gpt_4o."""
    s = surface.lower().replace('‐', '-').replace('–', '-')
    s = re.sub(r'[\s\-_](?:(?:19|20)\d\d(?:-\d\d){0,2}|\d{4})$', '', s)   # snapshot date (owner 2026-09-30): gpt-4o-2024 -> gpt-4o
    if s not in SIZE_IN_NAME:
        s = re.sub(r'[\s\-_]\d+(?:\.\d+)?(?:x\d+)?[bm]$', '', s)         # model size (owner 2026-09-30): mistral-7b -> mistral
    s = s.replace('+', '_plus')
    s = re.sub(r'(\d)\.0(?!\d)', r'\1', s)
    s = re.sub(r'[\s\-./]+', '_', s)
    s = re.sub(r'[^a-z0-9_]', '', s).strip('_')
    s = re.sub(r'_+', '_', s)
    s = re.sub(r'^chat_gpt(?=_|$)', 'chatgpt', s)
    s = SYNONYMS.get(s, s)
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


def focal_mentions(matcher, text, a2w, anchored):
    """[(term, surface, canonical, token positions)] for every accepted focal match; a match inside a token is extended to the whole token
    only when the rest is a version ('ChatGPT' in 'ChatGPT-3.5' -> chatgpt_3_5), otherwise the matched text decides the canonical form"""
    spans = token_spans(text, a2w)
    out = []
    for term, (a, b), surf in matcher.hits(text, anchored)[0]:
        pos = sorted(p for p, (s, e) in spans.items() if s < b and e > a)
        if not pos:
            out.append((term, surf, canonical(surf), [])); continue
        ts, te = spans[pos[0]][0], spans[pos[-1]][1]
        if ts == a and te > b and VERSION_SUFFIX.match(text[b:te]):
            surf = text[a:te]
        # §6 item 8: a version written as the next token ("DeepSeek-V3", "ChatGPT 4o", "GPT 2") is part of the name, when the
        # name is a versioned family and has no version yet; a size token ("8B") is not (dropped as a model size)
        later = [p for p in sorted(spans) if p > pos[-1]]
        if later and re.fullmatch(VERSIONED, canonical(surf)) and not re.search(r'\d', canonical(surf)):
            q = later[0]
            gap, tokw = text[spans[pos[-1]][1]:spans[q][0]], text[spans[q][0]:spans[q][1]]
            if gap in ('-', ' ', '\u2010', '\u2013', '- ') and VERSION_TOKEN.match(tokw) and not SIZE_RX.match(tokw.lower()):
                pos = pos + [q]; surf = text[a:spans[q][1]]
        out.append((term, surf, canonical(surf), pos))
    return out


# ------------------------------------------------------------------ edge helpers
def is_focal(atom):
    p = atom.parts()
    return len(p) > 2 and p[2] == 'focal'


def terminal(edge, barrier=None):
    if edge is None or (barrier and str(edge) == barrier):
        return []
    if edge.is_atom():
        return [edge]
    return [a for sub in edge for a in terminal(sub, barrier)]


def replace_all(edge, mapping):
    if edge.is_atom():
        return mapping.get(str(edge), edge)
    return hedge(tuple(replace_all(c, mapping) for c in edge))


def smallest_cover(edge, need):
    """index path to the deepest sub-edge whose atoms include all of need"""
    if edge.is_atom():
        return None
    for i, c in enumerate(edge):
        r = smallest_cover(c, need)
        if r is not None:
            return [i] + r
    return [] if need <= {str(a) for a in terminal(edge)} else None


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
                                  and urllib.parse.unquote(a.root()).lower() in MODAL_WORDS)


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
    if edge is None or str(edge) == barrier:
        return False
    if edge.is_atom():
        if XBASED == 'all':
            return is_pred(edge) and not edge.type().startswith('Mv')
        return is_pred(edge) and not hyphen_prop(edge)
    return any(deep_pred(c, barrier) for c in edge)


def split_clauses(edge, barrier, acc, clauses, phrases):
    """a phrase cut open to reach the clauses nested in it: its loose words (acc, one cousin), its sub-phrases without a
    clause (phrases, one cousin each; GROUP '1' puts their words in acc too) and the nested clauses"""
    if edge is None or str(edge) == barrier:
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
        self.lemma = {}
        for x in unit["extra_edges"]:
            m = re.fullmatch(r'\(_lemma (\S+) (\S+)\)', x)
            if m:
                self.lemma[m.group(1)] = m.group(2).split('/')[0]
        self.pos = collections.defaultdict(list)                        # source atom -> [(word, position)]
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
        return labels, nontext

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
        word = urllib.parse.unquote(root).lower()
        lemma = self.lemma_of(atom)
        if str(atom) in self.able:
            out = hedge(f"{word}/M/en"); self.detail['modal_able'][word] += 1    # §6 item 3: "was able to offer" -> able offer
        elif is_focal(atom):
            reason = 'focal_in_periphery' if periphery else None
            out = None if periphery else class_collapse(atom, 'C')
        elif hyphen_prop(atom):
            out = hedge(f"{root.lower()}/M/en"); self.detail['hyphen_property'][word] += 1
        elif word == '%':
            out = hedge('percent/C/en'); self.detail['percent'][t] += 1
        elif q4_all(self, atom, lambda p: p in self.label_pos):
            reason = 'label'; self.detail['label'][word] += 1                 # §6 item 8: "Table 2", "Fig. 1b"
        elif NUMBER_RX.match(word) or (t.startswith(('C#', 'M#')) and word in NUMBER_WORDS):
            if q4_all(self, atom, lambda p: p in self.nontext_num):
                reason = 'number_nontext'; self.detail['number_nontext'][word] += 1
            else:                                  # §6 item 8: plain-text numbers kept as written (0-10 as words rolled back, RL-119)
                out = hedge(f"{root.lower()}/{'C' if t.startswith('C') else 'M'}/en"); self.detail['number_kept'][word] += 1
        elif not re.search(r'[a-z]', word):
            reason = 'no_letters'; self.detail['no_letters'][word] += 1
        elif URL_RX.match(word):
            reason = 'url'                                                    # §6 item 8
        elif SIZE_RX.match(word):
            reason = 'model_size'; self.detail['model_size'][word] += 1       # §6 item 8
        elif word in COMPARE_WORDS and q4_all(self, atom, self.compares_number):
            out = hedge(f"{word}/M/en"); self.detail['comparison_kept'][word] += 1   # §6 item 8: "over 90%" keeps "over"
        elif word in CONNECTIVES:
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
            lemma = {'ca': 'can', 'wo': 'will', "'ll": 'will'}.get(lemma, lemma)
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
            if reason is None and (clean_root(word) is None or len(clean_root(word)) == 1):
                reason = 'single_letter_or_symbol'; self.detail['single_letter_or_symbol'][word] += 1     # §6 item 8
            if reason is None:
                if t.startswith('Mv'):
                    self.detail['mv_kept'][lemma] += 1
                if not self.lemma.get(f"{root}/{t[:1]}/en"):
                    self.stats['lemma_missing'] += 1
                lc = clean_root(urllib.parse.unquote(lemma).lower()) or lemma
                lc = urllib.parse.quote(lc, safe="_-'&@+")                   # §6 item 8: symbols cleaned, hyphen -> _
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
        self.items, self.tags = {}, []

    def add(self, out, src):
        self.items.setdefault(str(out), [out, set()])[1].add(str(src))

    def merge(self, o):
        for k, (a, s) in o.items.items():
            self.items.setdefault(k, [a, set()])[1].update(s)
        self.tags += [t for t in o.tags if t not in self.tags]
        return self

    def atoms(self):
        return [self.items[k][0] for k in sorted(self.items)]

    def __bool__(self):
        return bool(self.items)


def flatten_atoms(atoms, periphery, ctx):
    f = Flat()
    for a in atoms:
        out, modal = ctx.curate(a, periphery)
        if out is not None:
            f.add(out, a)
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
            fused = hedge(f"{ctx.lemma_of(v)}_{kids[0].root()}/{v.parts()[1]}/en")
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
        fused = hedge(f"{ctx.lemma_of(v)}_{'_'.join(p.root() for p in parts)}/{v.parts()[1]}/en")
        for src in [v] + parts:
            ctx.pos[str(fused)].extend(ctx.pos.get(str(src), []))
        return hedge(tuple([fused] + others))
    return hedge(tuple(kids))


def focal_paths(edge, path=None):
    path = (path or []) + [edge]
    if edge.is_atom():
        return [path] if is_focal(edge) else []
    return [p for sub in edge for p in focal_paths(sub, path)]


def sweep(edge, barrier, ctx):
    if edge is None or str(edge) == barrier:
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
            if str(arg) == barrier:
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


def extract(edge, ctx):
    norm = fuse_phrasal(q4_move_also(edge), ctx)
    circuits, cousins = {}, {}
    for path in focal_paths(norm):
        if len(path) < 2:
            continue
        pidx = next((i for i in range(len(path) - 2, -1, -1) if has_pred(path[i][0]) and is_clause(path[i])), -1)
        if pidx == -1:
            ctx.stats['focal_path_no_predicate'] += 1
            continue
        hi, tele = pidx, Flat()
        for i in range(pidx - 1, -1, -1):
            step = flatten(path[i], str(path[i + 1]), False, ctx)
            if not step:
                continue
            if flat_is_verb_group(step):
                tele.merge(step); hi = i
            else:
                break
        barrier = str(path[hi])
        parent, farg = path[pidx], path[pidx + 1]
        head_atoms = terminal(parent[0])
        cp = [a for a in head_atoms if is_pred(a) or is_vg_mod(a)]
        cn = [a for a in head_atoms if not (is_pred(a) or is_vg_mod(a))]
        dummy = Flat().merge(tele).merge(flatten_atoms(cp, False, ctx))
        if not dummy:
            ctx.stats['focal_path_empty_verb_group'] += 1
            continue
        focal = flatten(farg, None, False, ctx)
        if not focal:
            continue
        sibs = []
        nf = flatten_atoms(cn, False, ctx)
        if nf:
            sibs.append(nf)
        for arg in parent[1:]:
            if arg == farg:
                continue
            s = flatten(arg, None, False, ctx)
            if s:
                sibs.append(s)
        parts = [('dummy_sibling', dummy), ('focal_he', focal)] + [('sibling_he', s) for s in sibs]
        circ = hedge(('parent',) + tuple(mk(k, f) for k, f in parts))
        key = str(circ)
        circuits.setdefault(key, parts)
        for kind, f in sweep(norm, barrier, ctx):
            ck = str(mk(kind, f))
            cousins.setdefault(ck, [kind, f, []])
            if key not in cousins[ck][2]:
                cousins[ck][2].append(key)
    return circuits, cousins


# ------------------------------------------------------------------ one unit
def atom_record(flat, ctx):
    return [dict(atom=str(a), src=sorted([s, w, p] for s in flat.items[str(a)][1] for w, p in ctx.pos.get(s, [])) or
                 [[s, None, None] for s in sorted(flat.items[str(a)][1])]) for a in flat.atoms()]


def curate_unit(art, s, u, matcher, anchored, modals):
    ctx = Ctx(u, modals)
    edge = hedge(u["main_edge"])
    try:                                    # §6 item 3: lexical modal + complement = one clause, on the parse as G2 gave it
        edge, ctx.able = modal_merge.merge_unit(edge, u, ctx.stats)
    except Exception as e:                  # never lose a unit to the merge: it stays unmerged, and this is counted
        edge = hedge(u["main_edge"]); ctx.stats['modal_merge_error_' + type(e).__name__] += 1
    ments = focal_mentions(matcher, u["text"], u["atom2word"], anchored)
    covered = {p for *_, pos in ments for p in pos}
    by_pos = collections.defaultdict(list)
    for a, w, p in u["atom2word"]:
        by_pos[p].append(a)
    mrec = []
    for term, surf, canon, pos in ments:
        need = {a for p in pos for a in by_pos.get(p, [])}
        how = 'no_atom'
        if need:
            catom = hedge(f"{canon}/Cp/focal")
            mapping = {a: catom for a in need}
            if all({p for _, p in ctx.pos[a]} <= covered for a in need):
                edge = replace_all(edge, mapping); how = 'global'
            else:
                path = smallest_cover(edge, need)
                if path is None:
                    how = 'not_in_edge'
                else:
                    edge = replace_at(edge, path, mapping); how = 'scoped'
            if how != 'not_in_edge':
                ctx.pos[str(catom)].extend((w, p) for a in need for w, p in ctx.pos.get(a, []) if p in pos)
        mrec.append(dict(term=term, surface=surf, canonical=canon, positions=pos, replaced=how))
    circuits, cousins = extract(edge, ctx)
    gloss = [k for k, parts in circuits.items()                         # §6 item 8: "LLM equals large language model."
             if all(urllib.parse.unquote(a.root()) in GLOSS_VERBS or not pred_or_neg(a)
                    for kind, f in parts if kind == 'dummy_sibling' for a in f.atoms())
             and any(urllib.parse.unquote(a.root()) in GLOSS_VERBS for kind, f in parts if kind == 'dummy_sibling' for a in f.atoms())
             and all(is_focal(a) for kind, f in parts if kind != 'dummy_sibling' for a in f.atoms())
             and sum(kind != 'dummy_sibling' for kind, _ in parts) >= 2]          # a term and its expansion: two sides
    for k in gloss:
        del circuits[k]; ctx.stats['parent_glossary_dropped'] += 1
    if gloss:
        cousins = {ck: [kind, f, [p for p in ps if p not in gloss]] for ck, (kind, f, ps) in cousins.items()}
        cousins = {ck: v for ck, v in cousins.items() if v[2]}
    uid = u["uid"]
    parents, pid_of = [], {}
    for k, (key, parts) in enumerate(circuits.items(), 1):
        pid = f"{uid}.P{k}"; pid_of[key] = pid
        kids, j = [], 0
        for kind, f in parts:
            if kind == 'sibling_he':
                j += 1
            cid = f"{pid}.{ {'dummy_sibling': 'D', 'focal_he': 'F'}.get(kind, f'S{j}') }"
            e = str(mk(kind, f))
            kids.append(dict(id=cid, hash=h12(cid, e), kind=kind, edge=e, content_key=content_key(str(a) for a in f.atoms()),
                             atoms=atom_record(f, ctx),
                             **({'modality': f.tags} if f.tags else {})))
        parents.append(dict(id=pid, hash=h12(pid, key), edge=key, content_key=content_key(k["content_key"] for k in kids),
                            children=kids))
    cous = []
    for j, (ck, (kind, f, pkeys)) in enumerate(cousins.items(), 1):
        kid = f"{uid}.K{j}"
        cous.append(dict(id=kid, hash=h12(kid, ck), kind=kind, edge=ck, content_key=content_key(str(a) for a in f.atoms()),
                         parents=[pid_of[p] for p in pkeys], atoms=atom_record(f, ctx),
                         **({'modality': f.tags} if f.tags else {})))
    rec = dict(pmcid=art, sid=s["sid"], hash_final=s["hash_final"], uid=uid, unit_hash=u["hash"], text=u["text"],
               focal_mentions=mrec, parents=parents, cousins=cous)
    return rec, ctx


# ------------------------------------------------------------------ checks
# ------------------------------------------------------------------ §6 items 6 and 7: exclusions and corrections
def load_reflexives():
    """{uid: record} of the units whose emphatic reflexive pronouns are restored and re-parsed (§6 item 8)"""
    return {r["uid"]: r for r in (json.loads(l) for l in open(REFLEXIVE_TABLE, encoding="utf-8"))}


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


def check_record(rec, u):
    """provenance problems of one output record against its input unit"""
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
        for c in p["children"]:
            if not c["id"].startswith(p["id"] + ".") or c["hash"] != h12(c["id"], c["edge"]):
                bad.append(f"child id/hash {c['id']}")
    for c in rec["cousins"]:
        if c["hash"] != h12(c["id"], c["edge"]) or not c["parents"] or not set(c["parents"]) <= pids:
            bad.append(f"cousin {c['id']}")
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
    db_p = os.path.join(a.outdir, f"g3_test_{tag}.sqlite")
    for p in (out_p, err_p, rep_p) + ((db_p,) if a.db else ()):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    matcher = focal_terms.Matcher()
    excluded, llm_out, corrections = load_lists()
    reflexives = load_reflexives()
    nonprose = []
    stats, detail = collections.Counter(), collections.defaultdict(collections.Counter)
    surfaces, problems, samples = collections.Counter(), [], []
    hg, stack = None, contextlib.ExitStack()
    if a.db:
        from graphbrain import hopen
        hg = stack.enter_context(hopen(db_p))
    t0 = time.time()
    with open(inp, encoding="utf-8") as f, open(out_p, "w", encoding="utf-8") as fo, open(err_p, "w", encoding="utf-8") as fe:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            r = json.loads(line)
            stats['articles'] += 1
            if r["pmcid"] in excluded:                                  # §6 item 6: out of scope, left out of G3 and M1
                stats['articles_excluded'] += 1
                continue
            units = []
            for s in r["sentences"]:
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
                    cu = u
                    if u["uid"] in reflexives:                         # §6 item 8: emphatic reflexive restored, re-parsed
                        rr = reflexives[u["uid"]]
                        if rr["unit_hash"] != u["hash"] or rr["text_g2"] != u["text"]:
                            problems.append(f"{u['uid']}: reflexive table does not match the unit")
                        else:
                            cu = dict(u, text=rr["text_restored"], main_edge=rr["main_edge"], extra_edges=rr["extra_edges"],
                                      atom2word=rr["atom2word"]); stats['units_reflexive_restored'] += 1
                    if u["uid"] in corrections:                         # §6 item 6: corrected before focal matching
                        h, fixes = corrections[u["uid"]]
                        if h != u["hash"]:
                            problems.append(f"{u['uid']}: correction list hash {h} != unit hash {u['hash']}")
                        else:
                            cu = correct_unit(cu, fixes); stats['units_corrected'] += 1
                    units.append((s, u, cu))
            anchored = any(matcher.anchor_hits(cu["text"]) for _, _, cu in units)    # article evidence from the kept units
            for s, u, cu in units:
                stats['units'] += 1
                try:
                    rec, ctx = curate_unit(r["pmcid"], s, cu, matcher, anchored, a.modals)
                    if cu is not u:
                        rec["text_original"] = u["text"]
                        if u["uid"] in corrections:
                            rec["corrections"] = corrections[u["uid"]][1]
                        if u["uid"] in reflexives:
                            rec["reflexive_restored"] = reflexives[u["uid"]]["restorations"]
                except Exception as e:                                  # RecursionError included
                    stats['unit_errors'] += 1; stats['error_' + type(e).__name__] += 1
                    fe.write(json.dumps(dict(uid=u["uid"], error=type(e).__name__, message=str(e)[:500])) + "\n")
                    continue
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.update(ctx.stats)
                for k, c in ctx.detail.items():
                    detail[k].update(c)
                for m in rec["focal_mentions"]:
                    surfaces[(m["surface"], m["canonical"])] += 1; stats['mention_' + m["replaced"]] += 1
                stats['focal_mentions'] += len(rec["focal_mentions"])
                stats['units_with_parent'] += bool(rec["parents"]); stats['parents'] += len(rec["parents"])
                stats['cousins'] += len(rec["cousins"])
                stats['units_focal_no_parent'] += bool(rec["focal_mentions"]) and not rec["parents"]
                bad = check_record(rec, u)                              # the unit hash is verified on the original text
                problems += [f"{u['uid']}: {b}" for b in bad]
                if rec["parents"] and len(samples) < 400:
                    samples.append(rec)
                if hg is not None:
                    for p in rec["parents"]:
                        hg.add(hedge(('source_core', f"{r['pmcid']}::{u['uid']}::{p['hash']}", hedge(p["edge"]))))
                    for c in rec["cousins"]:
                        hg.add(hedge(('source_periphery', f"{r['pmcid']}::{u['uid']}::{c['hash']}", hedge(c["edge"]))))
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
    with open(os.path.join(a.outdir, f"g3_nonprose_{tag}.jsonl"), "w", encoding="utf-8") as fn:   # §6 item 8, for review
        for kind, uid, text in nonprose:
            fn.write(json.dumps(dict(kind=kind, uid=uid, text=text), ensure_ascii=False) + "\n")
    report = dict(stage="G3 test v2", input=inp, script=os.path.abspath(__file__), script_sha=sha(__file__), modals=a.modals,
                  inputs={p: h for p, h in INPUT_SHA.items()},
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
               ('ESM-1v', 'esm_1v')]


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


def selftest():
    bad = 0
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
    n = len(CANON_CASES) + len(UNIT_CASES)
    print(f"\n{n - bad} of {n} passed")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100, help="articles (0 = the whole shard)")
    ap.add_argument("--outdir")
    ap.add_argument("--input", help="a G2 v3 JSONL file instead of the shard file")
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
