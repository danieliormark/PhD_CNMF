"""
Focal terms v3 (2026-10-20, ticket 107 P1, D2; owner decisions 2026-10-19/20, RUN_LOG RL-157). Extends focal_terms.py (v2,
unchanged and still used by the existing pipeline) without changing any v2 match: every v2 hit is kept as it is; v3 only
adds hits on tokens that no v2 hit covers.

Added (classes as in corpus_statistics/focal_gap_classify.py and the group-4 rule, corpus_statistics/focal_group4_rule.py):
  A0  the bare official spelling "LLaMA" / "LLAMA" (v2's GENERIC pattern L[Ll]a[Mm]a missed it): as Llama, with Llama's
      guard rules (homonym patterns, article evidence)
  A1  a hyphen prefix before a listed family name: CT-BERT, Med-BERT, Sentence-BERT, XLM-RoBERTa, Med-PaLM, PMC-LLaMA
  A2  a version glued to a listed name: ChatGPT4, ChatGPT3.5, ChatGPT4o, PaLM2, Gemma2, Gemini2.5
  A3  plurals of listed names: GPTs, BERTs, ChatGPTs, LLaMAs
  A4  other capitalisation of BERT / RoBERTa: Bert, bert, Roberta, roberta, ROBERTA, RoBERTA ... (article evidence required;
      rejected before a capitalised surname or after a title — "Bert Vogelstein", "Dr. Roberta")
  A5  checkpoint names: a listed family name + configuration suffix: bert-base-uncased, roberta-large, gemini-2.5-pro,
      mistral-7b-instruct, gemma-2-9b-it, LLaMA-7B
  A6  "language-model(s)" written with a hyphen or underscore
  G4  unlisted model names accepted by the group-4 rule (DistilBERT, ClinicalBERT, scGPT, MedGemma, BioMistral ...), not
      tools (BERTopic, BERTScore, GPTZero, Ollama ...), code identifiers, URLs, organisations or look-alikes
  CG  a listed name with a glued citation number (BERT21, RoBERTa37): as the listed name
Guards: a token whose family needs article evidence in v2 (GPT, Gemini, Bard, Llama, Mistral, PaLM, Gemma) needs it here too,
and v2's homonym patterns for that family are applied around the token; A2/A3 forms of PLAIN names (ChatGPT4, ChatGPTs)
and G4 names count as article evidence, like PLAIN names.
    python3 focal_terms_v3.py --selftest
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import focal_terms as v2

RULES = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/"
_ns = {}
exec(open(RULES + "focal_group4_rule.py").read().split('if __name__')[0], _ns)
is_model = _ns["is_model"]

TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_]*(?:[-‐‑][A-Za-z0-9.+_]+)*")
H = r"[-‐‑]"
FAM = r"(?:BERT|RoBERTa|GPT|LLaMA|Llama|LLAMA|PaLM|Gemini|Mistral|Gemma)"
EVID_FAM = {"gpt": "GPT", "llama": "Llama", "palm": "PaLM", "gemini": "Gemini", "mistral": "Mistral", "gemma": "Gemma", "bard": "Bard"}
A0 = re.compile(r"^(?:LLaMA|LLAMA|LlaMA)$")
A1 = re.compile(rf"^[A-Za-z0-9.+]+{H}{FAM}(?:{H}[A-Za-z0-9.]+)*$")
A2 = re.compile(r"^(?:ChatGPT|GPT|Gemini|PaLM|Gemma|Mistral|LLaMA|Llama)\d(?:\.\d)?[a-zA-Z]{0,5}$")
A3 = re.compile(r"^(?:BERT|RoBERTa|GPT|ChatGPT|LLaMA|Llama)s$")
A4 = re.compile(r"^(?:Bert|bert|BERt|Roberta|roberta|ROBERTA|RoBERTA|ROBERTa|RoBerta|roBERTa)$")
A5 = re.compile(rf"(?i)^(?:bert|roberta|gpt|llama|gemini|mistral|gemma){H}(?:base|large|small|tiny|mini|uncased|cased|multilingual|\d)[A-Za-z0-9.\-‐‑_]*$")
A6 = re.compile(r"^(?:[Ll]anguage[-‐‑_][Mm]odels?)$")
CG = re.compile(r"^(BERT|RoBERTa)(\d{2})$")
PERSON = re.compile(r"^\s+(?:[A-Z][a-z]+|[A-Z]\.)")              # "Bert Vogelstein", "Roberta M."
TITLE = re.compile(r"(?:Dr|Prof|Mr|Mrs|Ms|Sir)\.?\s+$")
PLAIN_A2A3 = re.compile(r"^(?:ChatGPT\d|ChatGPTs$)")
CAPS_GPT = re.compile(r"^[A-Z0-9]*GPT[A-Z0-9]*$")    # CGPT, GPTMS, GPTS, ANGPT: abbreviations, not model names


def g4_ok(tok):
    """group-4 guard added after the Stage B hand check (2026-10-20): no all-caps GPT compound (CGPT test, GPTMS silane,
    R-GPTS scale, HAE-ANGPT), no part of 15+ capitals (peptide sequences)"""
    for part in re.split(r"[-‐‑]", re.sub(r"s$", "", tok)):
        if len(part) > 3 and CAPS_GPT.match(part):
            return False
        if len(part) >= 15 and part.isupper():
            return False
    return True


LISTED = {"gpt", "chatgpt", "bert", "roberta", "llama", "palm", "gemini", "mistral", "gemma", "bard"}   # bare names: v2's guarded path only


def family_of(tok):
    low = tok.lower()
    for k in ("chatgpt", "roberta", "bert", "gpt", "llama", "palm", "gemini", "mistral", "gemma", "language"):
        if k in low:
            return k
    return None


class Matcher(v2.Matcher):
    """v2's matcher plus the v3 additions; the interface (anchor_hits, hits) is v2's"""

    def _extra(self, line, anchored, spans):
        acc, rej, anchors = [], [], []
        for m in TOK.finditer(line):
            tok = m.group(0).rstrip(".")
            a, b = m.start(), m.start() + len(tok)
            if any(pa <= a < pb or pa < b <= pb for pa, pb in spans):
                continue
            kind = None
            if A0.match(tok): kind = "A0"
            elif CG.match(tok): kind = "CG"
            elif A2.match(tok): kind = "A2"
            elif A3.match(tok): kind = "A3"
            elif A4.match(tok): kind = "A4"
            elif A6.match(tok): kind = "A6"
            elif A5.match(tok): kind = "A5"
            elif A1.match(tok): kind = "A1"
            elif tok.lower().rstrip("s") not in LISTED and g4_ok(tok) and is_model(tok) == "model": kind = "G4"
            if not kind:
                continue
            fam = family_of(tok)
            term = f"v3:{kind}:{tok}"
            near = line[max(0, a - 80):b + 80]
            gfam = EVID_FAM.get(fam) if kind != "G4" and not PLAIN_A2A3.match(tok) else None
            if gfam and gfam in self.neg and self.neg[gfam].search(near):
                rej.append((term, (a, b), tok, "homonym")); continue
            if kind == "A4" and (PERSON.match(line[b:]) or TITLE.search(line[max(0, a - 6):a])):
                rej.append((term, (a, b), tok, "person name")); continue
            needs = (gfam in v2.NEED_EVIDENCE) or kind == "A4"
            if needs and not anchored:
                rej.append((term, (a, b), tok, "no evidence")); continue
            acc.append((term, (a, b), tok))
            if kind == "G4" or PLAIN_A2A3.match(tok):
                anchors.append((a, b))
        return acc, rej, anchors

    def anchor_hits(self, text):
        out = list(super().anchor_hits(text))
        for line in text.split("\n"):
            if not line.strip():
                continue
            acc, rej = super().hits(line, True)
            out += self._extra(line, True, [h[1] for h in acc + rej])[2]
        return out

    def hits(self, line, anchored):
        acc, rej = super().hits(line, anchored)
        e_acc, e_rej, _ = self._extra(line, anchored, [h[1] for h in acc + rej])   # tokens v2 accepted or rejected are v2's
        return acc + e_acc, rej + e_rej


SELFTEST = [   # (text, anchored, accepted surfaces expected among the v3 additions)
    ("We fine-tuned CT-BERT and Med-BERT on the corpus.", False, ["CT-BERT", "Med-BERT"]),
    ("Responses of ChatGPT4 and ChatGPT3.5 were compared.", False, ["ChatGPT4", "ChatGPT3.5"]),
    ("LLaMA was evaluated against BERTs.", True, ["LLaMA", "BERTs"]),
    ("LLaMA was evaluated.", False, []),
    ("We used bert-base-uncased and roberta-large.", True, ["bert-base-uncased", "roberta-large"]),
    ("We used DistilBERT and ClinicalBERT for classification.", False, ["DistilBERT", "ClinicalBERT"]),
    ("BERTopic and BERTScore were applied.", False, []),
    ("Gemmatimonadetes and Geminiviridae were abundant.", True, []),
    ("Bert Vogelstein described the pathway.", True, []),
    ("The Bert model was fine-tuned.", True, ["Bert"]),
    ("MedGemma and BioMistral were compared with Med-PaLM.", True, ["MedGemma", "BioMistral", "Med-PaLM"]),
    ("A language-model approach was used.", False, ["language-model"]),
    ("as shown by BERT21 and RoBERTa37", False, ["BERT21", "RoBERTa37"]),
    ("Serum ALT (GPTs) was measured.", True, []),
    ("Glutamyl pyruvic transaminase (GPT) activity was determined.", False, []),
    ("Glutamyl pyruvic transaminase (GPT) activity was determined.", True, []),
    ("Gemini and PaLM were compared.", False, []),
    ("A comprehensive genomic profiling test (CGPT) was used; CGPTs were feasible.", True, []),
    ("The GPTMS-t-SiO2 NPs were reacted with DOPO.", True, []),
    ("Paranoia was measured with the R-GPTS and the GPTS-B.", True, []),
    ("HAE-ANGPT is positioned in both endotypes.", True, []),
    ("FITC-Tat-p62DN (VDPSTGELQSLQMPESEGPSSLDPSQEGPTGLKEAALYPHLPP) peptides.", True, []),
    ("We compared scGPT and GeneGPT.", False, ["scGPT", "GeneGPT"]),
]


def selftest():
    m = Matcher(); bad = 0
    for text, anch, exp in SELFTEST:
        got = sorted(s for t, _, s in m.hits(text, anch)[0] if t.startswith("v3:"))
        ok = got == sorted(exp); bad += not ok
        print("PASS " if ok else "FAIL ", repr(text[:60]), "" if ok else f"got {got} expected {sorted(exp)}")
    # v2 behaviour unchanged on v2's own self-test
    for text, anch, exp in v2.SELFTEST:
        got = sorted(t for t, _, _ in v2.Matcher.hits(m, text, anch)[0])
        ok = got == sorted(exp); bad += not ok
        if not ok: print("FAIL v2 case", repr(text[:60]), got, exp)
    n = len(SELFTEST) + len(v2.SELFTEST)
    print(f"\n{n - bad} of {n} passed")
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
