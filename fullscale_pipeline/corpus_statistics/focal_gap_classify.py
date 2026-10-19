# D2 exploration: classify the unmatched candidate tokens of d2_scan.json and count their upstream effect.
import collections, json, re, sys

R = json.load(open(sys.argv[1]))
FAM = r"(?:BERT|RoBERTa|GPT|LLaMA|Llama|LLAMA|PaLM|Gemini|Mistral|Gemma|Bard)"
LOWFAM = r"(?:bert|roberta|gpt|llama|gemini|mistral|gemma|palm|bard)"
H = r"[-‐‑]"
RULES = [
    ("A1 hyphen prefix + listed family name (CT-BERT, Med-PaLM)", re.compile(rf"^[A-Za-z0-9.+]+{H}{FAM}(?:{H}.*|\d.*)?$")),
    ("A2 version glued to a listed name (ChatGPT4, Gemini2.5)", re.compile(rf"^(?:ChatGPT|Chat-GPT|GPT|Gemini|PaLM|Gemma|Mistral|LLaMA|Llama|Claude|Grok)\d[0-9.]*[a-zA-Z]{{0,6}}$")),
    ("A3 plural of a listed name (GPTs, BERTs)", re.compile(rf"^(?:{FAM}|ChatGPT|LLM)s$")),
    ("A4 listed name in another case (bert, Bert, LLAMA)", re.compile(rf"(?i)^{LOWFAM}$|^roberta$")),
    ("A5 checkpoint name: listed name + suffix (bert-base-uncased)", re.compile(rf"(?i)^(?:{LOWFAM}|roberta|chatgpt){H}[A-Za-z0-9.]+(?:{H}[A-Za-z0-9.]+)*$")),
    ("A6 hyphenated 'language-model'", re.compile(r"(?i)^(?:.*[-‐‑_])?language[-‐‑_]model(?:s|ing|ling|led)?(?:[-‐‑_].*)?$")),
    ("B glued derivative (DistilBERT, scGPT, MedGemma)", re.compile(rf"(?:[A-Za-z0-9]{FAM}|{FAM}[a-z])")),
]


def cls(tok):
    for name, rx in RULES:
        if rx.search(tok):
            return name
    return "C other (mostly noise: Hilbert, puberty, palmitate)"


S = collections.defaultdict(lambda: dict(occ=0, occ_no_hit_line=0, arts=set(), arts_new=set(), forms=collections.Counter()))
for a, r in R.items():
    new_art = r["status"] != "ok"
    for o in r["occ"]:
        c = cls(o["tok"]); s = S[c]
        s["occ"] += 1; s["forms"][o["tok"]] += 1; s["arts"].add(a)
        s["occ_no_hit_line"] += not o["line_has_hit"]
        if new_art:
            s["arts_new"].add(a)
for c in sorted(S):
    s = S[c]
    print(f"\n{c}\n  occurrences {s['occ']:,}; in lines without an accepted focal hit {s['occ_no_hit_line']:,}; "
          f"articles {len(s['arts']):,}; of which articles with no focal sentence at P2 {len(s['arts_new']):,}")
    print("  ", ", ".join(f"{t}:{n}" for t, n in s["forms"].most_common(45)))
json.dump({c: dict(occ=s["occ"], occ_no_hit_line=s["occ_no_hit_line"], arts=len(s["arts"]), arts_new=sorted(s["arts_new"]),
                   forms=s["forms"].most_common()) for c, s in S.items()}, open(sys.argv[2], "w"), indent=0)
