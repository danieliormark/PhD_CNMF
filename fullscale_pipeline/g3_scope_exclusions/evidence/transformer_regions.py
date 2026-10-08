"""Q6: do the 6,349 transformer-only articles refer to LLM-related transformers (BERT family, GPT, T5, LLaMA, language models)?
Searched in P0's analysed regions of the raw text (references and back matter excluded) and, separately, in the whole text."""
import json, re, csv, collections, sys
csv.field_size_limit(10**9)
T = set(json.load(open("transformer_only_articles.json")))
D = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus/"
REG = {}
for r in csv.DictReader(open("/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing/content_regions_v2.csv")):
    if r["pmcid"] in T: REG[r["pmcid"]] = json.loads(r["regions_json"] or "[]")
TERMS = {
    "BERT family": r"(?<![A-Za-z])(?:[A-Za-z]*BERT[a-z]*|RoBERTa|ALBERT|DeBERTa|ELECTRA|XLNet)(?![A-Za-z])",
    "GPT family": r"(?<![A-Za-z])(?:Chat[ -]?GPT|[A-Za-z]*GPT(?:[- ]?\d[\w.]*)?)(?![A-Za-z])",
    "T5/BART": r"(?<![A-Za-z])(?:mT5|Flan-T5|ByT5|ProtT5|ProstT5|T5[- ](?:model|base|large|small))(?![A-Za-z])",
    "LLaMA/Gemini/Claude/Mistral/PaLM": r"(?<![A-Za-z])(?:LLaMA|LLaMa|Llama[- ]?\d|Gemini|Claude|Mistral|Mixtral|PaLM|Qwen|DeepSeek)(?![A-Za-z])",
    "language model / LLM": r"(?i)(?<![a-z])(?:large[- ]language[- ]models?|(?:pre-?trained |masked |protein |causal |neural )?language models?|LLMs?)(?![a-z])",
}
NEG_GPT = re.compile(r"\b(?:SGPT|SGOT|ALT|AST|aminotransferase|transaminase|alanine|U/L|IU/L)\b")
RX = {k: re.compile(v) for k, v in TERMS.items()}
out = {}
for k in sorted(T):
    try: lines = open(D + k + ".txt", errors="ignore").read().split("\n")
    except FileNotFoundError: out[k] = dict(missing=True); continue
    regs = REG.get(k, [])
    content = "\n".join("\n".join(lines[max(0, g["start"] - 1):g["end"]]) for g in regs)
    rec = {}
    for name, rx in RX.items():
        hits = []
        for m in rx.finditer(content):
            ctx = content[max(0, m.start() - 90):m.end() + 70].replace("\n", " ")
            if name == "GPT family" and NEG_GPT.search(ctx): continue
            hits.append((m.group(0), ctx))
        if hits: rec[name] = hits[:3] + [len(hits)]
    out[k] = dict(regions=len(regs), hits=rec)
json.dump(out, open("transformer_regions.json", "w"))
c = collections.Counter(); anyc = 0
for k, v in out.items():
    if v.get("missing"): c["missing"] += 1; continue
    if not v["regions"]: c["no regions"] += 1
    if v["hits"]: anyc += 1
    for n in v["hits"]: c[n] += 1
print("articles", len(out), "| with any LLM-related transformer term in analysed text:", anyc); print(c.most_common())
