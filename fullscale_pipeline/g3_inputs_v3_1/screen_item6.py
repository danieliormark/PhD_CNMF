"""G3 input for the merged corpus (RL-171): G3_POSTPROCESSING.md §6 item 6 (owner decisions 2026-10-08, RL-091 to RL-094)
applied to the articles of PG/g2_v3_1/ that the old screen did not see or whose focal terms changed.

Screened: the new articles (15,563) and the current articles whose focal sentences changed in the incremental run
(inc_run/classification_current.tsv: additions, removals_only, new_in_v3); the decisions of the old screen stand for
every other current article (they were made on the same text). The tests are the old scans', unchanged:
  rule 1/2  every "LLM(s)" in the article's G2 sentences (pieces), classified as in evidence/llm_classify2.py: a
            definition counts only if its word initials spell L-L-M; classes 1 both, 2 defined: language model,
            3 defined: other, 4 undefined with "language model", 5 undefined with other LLM evidence, 6 undefined, no
            evidence. Class 6 -> exclude (owner: all of them); class 3 -> read by hand (owner: exclude unless a
            misspelt or misnamed language-model definition, which is corrected instead).
  rule 3    articles whose only accepted focal term (P7 terms) is ProGen, BioBridge, PaLM 2 or ESM-2 -> read by hand
            (owner: exclude the homonym articles).
  rule 4    articles whose only accepted focal terms are "transformer model(s)": searched in P0's analysed regions of the
            raw text for LLM-related transformers (evidence/transformer_regions.py, same patterns) -> none: exclude;
            some: read by hand (owner: keep only articles that use or compare such a model; exclude passing mentions,
            vision-language models only, false hits).
Output in PG/g3_inputs_v3_1/item6/: screen.json (per screened article: class, terms, evidence with contexts; corpus
text, RDS only) and screen_summary.json.
    python3 screen_item6.py
"""
import collections, csv, glob, json, os, re, sys
csv.field_size_limit(10 ** 9)
R = "/mnt/hum01-rds/Basov/p91688di/"
PG, L = R + "phase5_graphbrain/", R + "llm_corpus_staging/"
OUT = PG + "g3_inputs_v3_1/item6/"
EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g3_scope_exclusions", "evidence")

# ---- rule 1/2: evidence/llm_scan.py and evidence/llm_classify2.py, patterns copied unchanged
LLM = re.compile(r'(?i)(?<![a-zA-Z0-9])LLMs?(?![a-zA-Z0-9])')
DEF_BEFORE = re.compile(r'((?:[\w\-‐’\']+[\s\-‐]+){1,6})\(\s*LLMs?\s*[),;]')
DEF_AFTER = re.compile(r'(?<![a-zA-Z0-9])LLMs?\s*\(\s*([^()]{3,60})\)')
LANGM = re.compile(r'(?i)language[\s\-]+models?')
OTHER_EVID = re.compile(r'(?i)(?<![a-zA-Z0-9])(chat[ -]?gpt|gpt[- ]?[345]|bert|gemini|claude|llama|deepseek|qwen|generative ai|chatbot|transformer|prompt)')
SKIP = {"of", "and", "the", "a", "an", "for", "in", "on", "with", "to"}
LM = re.compile(r"(?i)language")


def acronym(exp):
    ws = [w for w in re.split(r"[\s\-‐/]+", exp) if w]
    for k in range(2, min(6, len(ws)) + 1):
        tail = ws[-k:]
        ini = "".join(w[0].lower() for w in tail if w.lower() not in SKIP)
        if ini == "llm" or ini == "llms":
            return " ".join(tail)
    return None


def llm_class(full):
    occ = [(m.group(0), full[max(0, m.start() - 90):m.end() + 60]) for m in LLM.finditer(full)]
    if not occ:
        return None
    defs = [("before", m.group(1).strip()) for m in DEF_BEFORE.finditer(full)] + [("after", m.group(1).strip()) for m in DEF_AFTER.finditer(full)]
    exps = [e for e in (acronym(d) for t, d in defs if t == "before") if e]
    lm = [e for e in exps if LM.search(e)]
    oth = [e for e in exps if not LM.search(e)]
    if lm and oth: c = "1 both"
    elif lm: c = "2 defined: language model"
    elif oth: c = "3 defined: other"
    elif len(LANGM.findall(full)) > 0: c = "4 undefined, 'language model' in article"
    elif len(OTHER_EVID.findall(full)) > 0: c = "5 undefined, other LLM evidence"
    else: c = "6 undefined, no evidence"
    return dict(c=c, n=len(occ), other_defs=oth, lm_defs=lm, ex=occ[:4])


# ---- rule 4: evidence/transformer_regions.py, patterns copied unchanged
TERMS = {
    "BERT family": r"(?<![A-Za-z])(?:[A-Za-z]*BERT[a-z]*|RoBERTa|ALBERT|DeBERTa|ELECTRA|XLNet)(?![A-Za-z])",
    "GPT family": r"(?<![A-Za-z])(?:Chat[ -]?GPT|[A-Za-z]*GPT(?:[- ]?\d[\w.]*)?)(?![A-Za-z])",
    "T5/BART": r"(?<![A-Za-z])(?:mT5|Flan-T5|ByT5|ProtT5|ProstT5|T5[- ](?:model|base|large|small))(?![A-Za-z])",
    "LLaMA/Gemini/Claude/Mistral/PaLM": r"(?<![A-Za-z])(?:LLaMA|LLaMa|Llama[- ]?\d|Gemini|Claude|Mistral|Mixtral|PaLM|Qwen|DeepSeek)(?![A-Za-z])",
    "language model / LLM": r"(?i)(?<![a-z])(?:large[- ]language[- ]models?|(?:pre-?trained |masked |protein |causal |neural )?language models?|LLMs?)(?![a-z])",
}
NEG_GPT = re.compile(r"\b(?:SGPT|SGOT|ALT|AST|aminotransferase|transaminase|alanine|U/L|IU/L)\b")
RX = {k: re.compile(v) for k, v in TERMS.items()}
TRANSFORMER = {"transformer model", "transformer models"}
NAMES = {"progen", "biobridge", "palm 2", "esm-2"}


def regions_hits(pm, corpus, regs):
    raw = (L + "pure_text_corpus/" if corpus == "current" else L + "pure_text_corpus_gap/") + pm + ".txt"
    try:
        lines = open(raw, errors="ignore").read().split("\n")
    except FileNotFoundError:
        return None
    content = "\n".join("\n".join(lines[max(0, g["start"] - 1):g["end"]]) for g in regs)
    rec = {}
    for name, rx in RX.items():
        hits = []
        for m in rx.finditer(content):
            ctx = content[max(0, m.start() - 90):m.end() + 70].replace("\n", " ")
            if name == "GPT family" and NEG_GPT.search(ctx):
                continue
            hits.append((m.group(0), ctx))
        if hits:
            rec[name] = hits[:3] + [len(hits)]
    return rec


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    assert not os.path.exists(OUT + "screen.json"), "outputs exist"
    new_ids = set(open(L + "gap_stage/gap_article_list_after_ab.txt").read().split())
    cls = {r["pmcid"]: r["class"] for r in csv.DictReader(open(L + "inc_run/classification_current.tsv"), delimiter="\t")}
    old_dec = {}
    for fn in ("llm_defined_otherwise.csv", "llm_no_definition_no_evidence.csv", "name_homonyms.csv", "transformer_only.csv"):
        for r in csv.DictReader(open(os.path.join(EV, "..", "decisions", fn))):
            old_dec[r["pmcid"]] = (fn, r["decision"])
    terms = {}
    for line in open(PG + "g2_v3_1/focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        terms[r["pmcid"]] = sorted({t.lower() for s in r["sentences"] for t in s.get("terms", [])})
    screened = {pm for pm in terms if pm in new_ids or cls.get(pm) in ("additions", "removals_only", "new_in_v3")}
    out, c = {}, collections.Counter()
    for f in sorted(glob.glob(PG + "g2_v3_1/shards/g2_parsed_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            a = json.loads(line)
            pm = a["pmcid"]
            if pm not in screened:
                continue
            corpus = "gap" if pm in new_ids else "current"
            full = " ".join(p.get("orig") or p.get("parsed") or "" for s in a["sentences"] for p in s.get("pieces", []))
            ts = set(terms[pm])
            rec = dict(corpus=corpus, cls=cls.get(pm, "new"), terms=sorted(ts), old_decision=old_dec.get(pm))
            lc = llm_class(full)
            if lc:
                rec["llm"] = lc
            if ts and ts <= TRANSFORMER:
                rec["transformer_only"] = True
            if len(ts) == 1 and next(iter(ts)) in NAMES:
                rec["name_only"] = next(iter(ts))
            out[pm] = rec
    regs = {}
    want = {pm for pm, r in out.items() if r.get("transformer_only")}
    for src in (R + "pmc_preprocessing/content_regions_v2.csv", L + "gap_stage/pmc_preprocessing/content_regions_v2.csv"):
        for r in csv.DictReader(open(src)):
            if r["pmcid"] in want:
                regs[r["pmcid"]] = json.loads(r["regions_json"] or "[]")
    for pm in want:
        out[pm]["regions"] = len(regs.get(pm, []))
        out[pm]["region_hits"] = regions_hits(pm, out[pm]["corpus"], regs.get(pm, []))
    for pm, r in out.items():
        g = r["corpus"]
        c[f"{g}: screened"] += 1
        if "llm" in r:
            c[f"{g}: llm class {r['llm']['c']}"] += 1
        if r.get("transformer_only"):
            c[f"{g}: transformer only, " + ("region hits" if r["region_hits"] else "no region hits")] += 1
        if r.get("name_only"):
            c[f"{g}: name only {r['name_only']}"] += 1
        if r["old_decision"]:
            c[f"{g}: has an old decision {r['old_decision'][0]} {r['old_decision'][1]}"] += 1
    json.dump(out, open(OUT + "screen.json", "w"), ensure_ascii=False)
    json.dump(dict(sorted(c.items())), open(OUT + "screen_summary.json", "w"), indent=1)
    print(json.dumps(dict(sorted(c.items())), indent=1))
