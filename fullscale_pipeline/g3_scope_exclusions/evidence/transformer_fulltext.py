"""Q6: full raw text of the 6,349 transformer-only articles: any LLM-related word anywhere (not only in P2's sentences)."""
import json, re, collections, os
T = json.load(open("transformer_only_articles.json"))
D = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus/"
W = {"GPT (bare)": r"\bGPT\b", "ChatGPT": r"(?i)chat[ -]?gpt", "LLM": r"\bLLMs?\b", "language model": r"(?i)language models?",
     "BERT family": r"BERT", "Gemini/Llama/Claude/Mistral/PaLM": r"\b(?:Gemini|L[Ll]a[Mm]a|Claude|Mistral|PaLM)\b",
     "generative AI / chatbot": r"(?i)generative (?:ai|artificial)|chatbot", "electrical transformer": r"(?i)power transformer|distribution transformer|transformer (?:winding|oil|substation|fault)|substation|kVA|dissolved gas"}
RX = {k: re.compile(v) for k, v in W.items()}
cnt = collections.Counter(); none = []; elec = []
for k in T:
    p = D + k + ".txt"
    if not os.path.exists(p): cnt["missing"] += 1; continue
    t = open(p, errors="ignore").read()
    hits = {w for w, rx in RX.items() if rx.search(t)}
    for h in hits: cnt[h] += 1
    llm = hits - {"electrical transformer"}
    if not llm: none.append(k)
    if "electrical transformer" in hits: elec.append(k)
print("articles", len(T)); print(cnt.most_common())
print("no LLM-related word anywhere in full text:", len(none))
json.dump(dict(none=none, elec=elec), open("transformer_fulltext.json", "w"))
