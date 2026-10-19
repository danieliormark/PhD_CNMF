# D4.1: final form of the word-level clean-up for G3, simulated with G3's own clean_root steps after it.
import json, re, sys, collections

LIGATURES = {'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'st', 'ﬆ': 'st'}
SINGLE_QUOTES = re.compile("[‘’‚‛`´′]")       # curly, grave, acute quotes and the prime -> '
DOUBLE_QUOTES = '“”„‟"'
CLITIC = re.compile(r"^'(?:s|m|d|t|ve|re|ll)\d*$")
CLITIC_TAIL = re.compile(r"(?<=[a-z0-9])'(?:s|m|d|ve|re|ll)$")
FORMULA_SIGNS = re.compile(r"[|=^~<>∑∏√∂∫×÷∈∀∃≠±∗∘⊂⊆∪∩→←↔⇒⊕⊗⋅]")
SYMBOL_MARKS = re.compile("[Ͱ-Ͽ″‴̂]")
FORMULA_CALL = re.compile(r"^[a-z]{1,2}\(")
GROUP_NAME = re.compile(r"^([a-z]{1,2})\((\d)\)[-_]([a-z]{3,}.*)$")        # se(3)-equivariant -> se3_equivariant
ENUM_TAIL = re.compile(r"\((?:[a-z]|\d{1,2})\)?$")
PLURAL_TAIL = re.compile(r"\((?:s|es)\)?$")
HYPHEN_CHARS = re.compile('[‐‑–—]')


def tidy(w):
    for k, v in LIGATURES.items():
        w = w.replace(k, v)
    w = w.replace(';', ';').replace('\\', '')                           # Greek question mark; escaped quotes
    w = SINGLE_QUOTES.sub("'", w)
    if CLITIC.match(w):
        return None
    w = CLITIC_TAIL.sub('', w)                                               # openai's, students'll -> openai, students
    if FORMULA_SIGNS.search(w):
        return None
    g = GROUP_NAME.match(w)
    if g:
        w = f"{g.group(1)}{g.group(2)}_{g.group(3)}"
    elif FORMULA_CALL.match(w):
        return None
    if SYMBOL_MARKS.search(w) and re.search(r"[()\[\]{}]", w):
        return None                                                          # μθ(xt, φ(microgram
    w = PLURAL_TAIL.sub('', w)
    w = ENUM_TAIL.sub('', w)
    w = re.sub(r"[“”„‟\"]\d+$", '', w)                   # perfect”15 -> perfect
    w = re.sub(r"(?<=[a-z0-9])[()\[\]{}“”„‟\"]+(?=[a-z0-9])", '_', w)   # godel(grounded, models“bert
    w = re.sub(r"[()\[\]{}“”„‟\"]", '', w)
    w = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", '', w)
    w = w.rstrip('−')
    if SYMBOL_MARKS.search(w) and len(re.sub(r"[^a-z]", '', w)) <= 1 and not w.startswith('aβ'):
        return None
    return w


def clean_root(word):
    """G3's clean_root with the D4.1 clean-up first"""
    w = tidy(word)
    if w is None:
        return None
    w = HYPHEN_CHARS.sub('_', w)
    w = re.sub(r'[,;.)\]]+\d+$', '', w)
    w = w.strip('.,;:()[]\'"')
    w = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])', '_', w)
    w = re.sub(r'_+', '_', w).strip('_')
    return w if re.search(r'[a-z]', w) else None


if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))
    for kind, forms in D["counts"].items():
        ch = collections.Counter(); same = 0
        for w, n in forms:
            t = clean_root(w)
            if t == w:
                same += n
            else:
                ch[(w, t)] += n
        print(f"\n== {kind}: {sum(ch.values()):,} occurrences changed, {same:,} unchanged")
        print("  changed:  ", "; ".join(f"{w} -> {t if t is not None else 'DROP'} ({n})" for (w, t), n in ch.most_common(110)))
        print("  unchanged:", ", ".join([f"{w}:{n}" for w, n in forms if clean_root(w) == w][:30]))
