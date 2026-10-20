# D2 group 4 (owner 2026-10-19), draft rule: an unlisted token is a model name when it carries a listed family name in its
# model spelling (case-sensitive) glued to other letters or digits, and is not on the exclusion lists (tools, code
# identifiers, URLs and organisations, genes and other look-alikes). Counted only in articles of the allowed types.
# RL-155 (owner 2026-10-20): a title-case family name (Llama, Mistral, Gemma, Gemini) followed by more lower-case letters is
# a different word (Gemmatimonadetes, Geminiviridae, Geminin), except a plural s or a version v + digit (SynLlamas,
# Mistralv0.1). Prints accepted and excluded forms and a random sample of accepted occurrences in context, for reading.
import collections, json, random, re, sys

FAMILY = re.compile(r"BERT|RoBERTa|Roberta(?=$|[-_])|(?<=[A-Za-z])Bert(?![a-z])|(?<=[a-z][A-Z])bert(?![a-z])|(?<=[a-z])Albert(?![a-z])|GPT|LLaMA|LLAMA|Llama(?![a-rtuw-z]|s[a-z]|v(?!\d))|LlaMA|Mistral(?![a-rtuw-z]|s[a-z]|v(?!\d))|Gemma(?![a-rtuw-z]|s[a-z]|v(?!\d))|Gemini(?![a-rtuw-z]|s[a-z]|v(?!\d))|PaLM|(?<=[A-Za-z-])Palm(?=$|[-_\d])")
ABBREV = {"gptu"}   # "ChatGPT use (GPTU)": a survey variable, not a model
TOOLS = {   # tools, methods, software, datasets built around a model, not models
    "bertopic", "bertscore", "bertscores", "keybert", "keybertinspired", "gptzero", "zerogpt", "detectgpt", "selfcheckgpt",
    "ollama", "llamaindex", "llama.cpp", "llama-factory", "gptq", "autogpt", "metagpt", "hugginggpt", "sharegpt", "bertviz",
    "berttokenizer", "bertmodel", "bertadam", "gpt4all", "llamaparse", "bertology", "bertsum"}
LOOKALIKE = re.compile(r"^(?:ANGPT|SGPT|BARD\d|Luria-Bertani|GeminiSEM|CIBERTSORT|Palm\d|RGPT$|MGPT$|GPTMS|CSS-Palm)", re.I)   # genes, a broth, a microscope, a tool misspelt, a polymer, a device, a variable,
# a silane (GPTMS), a palmitoylation tool (CSS-Palm)
CODE = re.compile(r"^(?:Bert|Roberta|GPT2?|Llama|Mistral|Gemma)(?:For[A-Z]|Tokenizer|Model|Config|Adam)")
URL_ORG = re.compile(r"\.(?:com|ai|net|org|io|cpp)\b|^www\.|^(?:meta-llama|mistralai|google-bert|MistralAI)$", re.I)


def is_model(tok):
    base = re.sub(r"[-_](?:based|derived|driven|like|style|powered|assisted|generated|only)$", "", tok)
    if not FAMILY.search(base):
        return None
    low = base.lower()
    if any(x in TOOLS or x.rstrip("s") in TOOLS for x in (low, re.sub(r"[-_].*", "", low))):     # plural tool names too (BERTopics)
        return "tool"
    if base.lower() in ABBREV:
        return "excluded"
    if LOOKALIKE.match(base) or CODE.match(base) or URL_ORG.search(base):
        return "excluded"
    if re.fullmatch(r"(?:BERT|GPT|RoBERTa)\d{1,2}", base):
        return "citation-glued (group 1)"
    return "model"


if __name__ == "__main__":
    F = json.load(open(sys.argv[1]))            # g4_forms.json
    R = json.load(open(sys.argv[2]))            # scan.json, for contexts
    C = collections.defaultdict(list)
    first = {f["form"]: is_model(f["form"]) for f in F}
    lower_models = {k.lower() for k, v in first.items() if v == "model"}
    def decide(form):
        v = first[form]
        if v is None:
            parts = re.split(r"[-_/](?=base|large|small|tiny|uncased|cased|multilingual|v\d|\d)", form)
            head = parts[0]
            if form.islower() and len(parts) > 1 and head in lower_models:
                return "model (lowercase checkpoint)"
        return v
    for f in F:
        C[decide(f["form"]) or "no family spelling (noise)"].append(f)
    for k, v in sorted(C.items()):
        print(f"\n== {k}: {len(v)} forms, {sum(x['occ'] for x in v):,} occurrences, "
              f"{len({a for x in v for a in [x['form']]}):,}")
        print("   ", ", ".join(f"{x['form']}:{x['articles']}" for x in v[:70]))
    acc = {x["form"] for x in C["model"] + C["model (lowercase checkpoint)"]}
    occ = [(a, o) for a, r in R.items() for o in r["occ"] if o["tok"] in acc]
    random.seed(4)
    print(f"\n== random sample of accepted occurrences ({len(occ):,} in all)")
    for a, o in random.sample(occ, 150):
        print(f"  [{o['tok']}] {o['ctx'].replace(chr(10), ' ')}")
