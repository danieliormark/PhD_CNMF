"""Stage B on the gap corpus (2026-10-20, ticket 107 P1, RL-157): article-level flags for articles whose focal sentences
(P2, v3 matcher) do not show that the article is about language models. Counts only: nothing is removed from any list or
corpus; the owner decides on removal after seeing the numbers. The same rules run on the current corpus to measure what
they would cost there (the current corpus was selected through title/abstract, so its flags are mostly errors of the rule).

Each focal sentence is labelled:
    statement  an AI-use statement about the manuscript or the analysis ("we used ChatGPT for language editing",
               "with help from Copilot for the R code"): a true mention of an LLM, but not research on or with LLMs
    focal      any other focal sentence
Each term is either AMBIGUOUS (it also names something else: ProGen the supplier, LLM as an abbreviation, BERT the
Bertini physics list, "language model" in linguistics, the evidence-guarded names GPT/Gemini/... ) or STRONG.
Article flags (one per article, first that applies):
    F1 statement_only   every focal sentence is an AI-use statement
    F2 ambiguous        the focal (non-statement) sentences hold only AMBIGUOUS terms, and the article either has no context
                        word in its cleaned text, or a confound marker of one of its terms and fewer than 5
                        context words
    kept                otherwise
    python3 gap_stage_b.py [--workers N]
"""
import csv, json, os, re, sys, collections
from multiprocessing import Pool

L = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
GR = L + "gap_stage/"
OUT = GR + "stage_b/"
CORPORA = {   # name: (P2 extractions, cleaned texts, year source)
    "gap": (GR + "llm_corpus_staging/focal_extractions_v2.jsonl", GR + "llm_corpus_staging/clean_corpus_v2/"),
    "current": (L + "focal_extractions_v2.jsonl", L + "clean_corpus_v2/"),
}

AMBIG_V2 = {"gpt", "bert", "roberta", "gemini", "bard", "llama", "mistral", "palm", "gemma", "language model", "llm", "llms",
            "progen", "transformer model", "transformer models", "claude", "grok", "rita", "gpn", "keap", "grover",
            "galactica", "kpgt", "agbt", "momu", "gimlet", "enbed", "pmlm", "molgen", "x-mol", "transmol", "go-pro",
            "gpt2", "palm 2"}                   # hand check 2026-10-20: GPT2 the gene, "palm 2" the polymerase domain
AMBIG_V3 = {"A0", "A2", "A3", "A4", "A6", "CG"}  # LLaMA, GPT1-type glued versions (not ChatGPT4: A2C), plurals, Bert, language-model, BERT21

CONTEXT = re.compile(r"(?i)\b(?:transformer-based|transformer architecture|natural language processing|NLP|pre-?trained|fine-?tun\w*|"
                     r"deep learning|neural networks?|machine learning|embeddings?|attention mechanism|tokeni[sz]\w*|"
                     r"artificial intelligence|generative AI|chatbots?|prompts?|prompting)\b")   # no focal term itself
CONFOUND = {   # per term: patterns in the whole article that show the other meaning
    "progen": r"Progen\s*(?:Biotechnik|Biotech|Pharmaceuticals)|(?:PROGEN|Progen)[,;]?\s*(?:Heidelberg|Germany|Wayne|PA\b|Australia)|"
              r"(?:Progen|PROGEN)\)|Nycodenz|Polymorph|(?:Progen|PROGEN)\s*[,;#]|\((?:Progen|PROGEN)\b",
    "llm": r"\bLLM-105\b|LLM\s+spectr|Auger|leaf lamina|logic learning machine|Rulex|lipid[- ]lowering|lean mass|lipid[- ]laden|\(\s*[A-Z]{2,4}(?:,\s*[A-Z]{2,4})*,\s*LLM\b|\bLLM(?:,\s*[A-Z]{2,4})+\)",
    "bert": r"(?:QGSP|FTFP|QBBC|QGSC)[ _]BERT|Bertini",
    "language model": r"(?i)sign language|language models? for (?:their )?child|(?:adult|parent\w*|caregiver\w*|teacher\w*) (?:as )?"
                      r"(?:good )?language models?|aphasia|Broca|Wernicke|psycholinguist\w*",
    "gpt2": r"\bGPT1\b|\bGOT[12]?\b|transaminase|aminotransferase|alanine",
    "palm 2": r"(?i)thumb|palm domain|fingers? (?:domain|subdomain)|polymerase|RdRp",
    "gpt": r"\b(?:SGPT|SGOT|ALT|AST|aminotransferase|transaminase|glutamic[- ]pyruvic)\b",
    "gemini": r"(?i)Gemini (?:Observatory|North|South|telescope|surveillance|twin|study|cohort|trial)|geminivir|gemini of (?:Cajal|coiled)",
    "palm": r"(?i)palm oil|palmar|palm kernel|oil palm|date palm",
}
CONFOUND["llms"] = CONFOUND["llm"]
CONF_MAX_CONTEXT = 5      # a confound marker counts only in an article with fewer context words than this
CONFOUND = {k: re.compile(v) for k, v in CONFOUND.items()}

AUTHOR = r"(?:\bthe authors?\b|\bwe\b|\bI\b|\bour\b|\bmy\b|the (?:first|corresponding|senior) author)"
MANU = (r"(?:manuscript|this (?:work|paper|article|study|review|report|text|research)|the (?:final )?(?:text|draft|abstract|title)|"
        r"\barticle\b|\bpaper\b|writing|rephras\w*|paraphras\w*|draft\w*|translat\w*|figures?\b|graphics?\b|graphical|"
        r"schematic|illustrations?|layouts?|syntax|punctuation|typograph\w*|wording|phrasing|clarity|coherence|cohesion|"
        r"precision|key arguments|structuring the content|reference list|citations|formulation\w* of (?:the )?text|"
        r"literature (?:discovery|search)|search terms|"
        r"\bEnglish\b|readability|grammar|grammatical|proof-?read\w*|spelling|copy-?edit\w*|language (?:editing|polishing|"
        r"refinement|improvement|correction|check\w*|quality|support|assistance)|(?:polish|refin|edit|improv)\w* (?:the )?"
        r"(?:language|writing|wording|text|English|clarity|flow|style)|writing (?:process|assistance|support)|word (?:count|limit))")
CODE = r"(?:\bcode\b|scripts?\b|coding|debug\w*|programming|Python|\bR\b (?:code|scripts?))"
AIUSE = (r"(?:\bused?\b|\busing\b|use of|usage|utili[sz]\w*|employ\w*|assist\w*|help\w*|aid\b|support\w*|acknowledge\w*|declare\w*|"
         r"leverag\w*|rel(?:y|ied|ying) on|appreciation|incorporat\w*|thank\w*|grateful|revised|edited|formatt?\w*|"
         r"refin\w*|prepar\w*)")
FORMULA = re.compile(r"(?i)during the preparation of this|generative AI (?:and AI-assisted )?technolog|AI-assisted technolog|"
                     r"take(?:s)? full responsibility for the content")
HELP = r"(?:assist\w*|help\w*|\baids?\b|support\w*|generated by|written (?:by|with))"
CODE_TASK = (r"(?:prepar|writ|generat|debug|develop|review|clean|draft|refin)\w* (?:the |of |and refine )?(?:analysis |R |Python |the )?"
             r"(?:code|scripts?)\b|code generation")
STATEMENT = re.compile(rf"(?i){AIUSE}.{{0,200}}(?:{MANU}|{CODE_TASK})|{HELP}.{{0,100}}{CODE}|{CODE}.{{0,100}}{HELP}|"
                       rf"(?:improv|enhanc|refin|polish|edit|correct|proofread)\w*\s+(?:the\s+)?(?:overall\s+)?{MANU}.{{0,120}}{AIUSE}|"
                       rf"{MANU}.{{0,60}}(?:with|by) (?:the )?{HELP}|"
                       rf"{MANU}.{{0,40}}(?:was|were) (?:written|generated|created|edited|revised|refined|polished|proofread|translated|"
                       rf"assisted|prepared|reviewed for grammar).{{0,40}}(?:with|by|using|via)\b")   # reverse order: "to improve the manuscript, we used"; "graphics ... with the help of"
NEGATED = re.compile(r"(?i)\b(?:no|not|never)\b.{0,150}\b(?:used|utili[sz]ed|employed|generate|alter|modify)\b.{0,120}"
                     r"(?:manuscript|paper|article|writing|this (?:work|study|research)|images?|data|results|analys\w*|content)")
RESEARCH = re.compile(r"(?i)evaluat|assess(?:ed|ing|ment)? (?:the |its |their )?(?:performance|accuracy|ability|quality of)|compar(?:ed|ing|ison)|"
                      r"\brated\b|ratings?\b|scores?\b|benchmark|participants|patients|respondents|students|questions|answers|"
                      r"generated? (?:patient|educational)|outputs?\b")
STATEMENT_NEED_AUTHOR = re.compile(rf"(?i){AUTHOR}|writers?\b|co-?authors?|author\(s\)|(?:was|were|has been|have been) "
                                   r"(?:used|employed|utili[sz]ed|created|revised|edited)|is acknowledged|played .{0,30}role|"
                                   r"appreciation|acknowledge|grateful|thank|(?:was|were) (?:assisted|made|written|generated)|"
                                   r"this (?:manuscript|article|paper)")


def term_key(t):
    if t.startswith("v3:"):
        kind, surf = t.split(":")[1].upper(), t.split(":", 2)[2]
        return "v3:A2C" if kind == "A2" and surf.lower().startswith("chatgpt") else "v3:" + kind
    return t.lower()


def is_ambig(k):
    return (k[3:] in AMBIG_V3) if k.startswith("v3:") else (k in AMBIG_V2)


def is_statement(text):
    """an AI-use statement: the disclosure formula, or a use of the tool for the manuscript or the code, said of the authors,
    without research words (a study that rates ChatGPT's readability is research, not a statement)"""
    if FORMULA.search(text) or NEGATED.search(text):
        return True
    return bool(STATEMENT.search(text)) and bool(STATEMENT_NEED_AUTHOR.search(text)) and not RESEARCH.search(text)


def files_by_article(d):
    idx = collections.defaultdict(list)
    for f in os.listdir(d):
        if f.endswith(".txt"):
            idx[re.match(r"(PMC\d+)", f).group(1)].append(d + f)
    return idx


def one(args):
    corpus, line, files = args
    r = json.loads(line)
    sents = [s for b in r["blocks"] for s in b["sentences"] if s["focal"]]
    lab = ["statement" if is_statement(s["text"]) else "focal" for s in sents]
    sub = collections.Counter(term_key(t) for s, l in zip(sents, lab) if l == "focal" for t in s["terms"])
    allt = collections.Counter(term_key(t) for s in sents for t in s["terms"])
    text = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    n_ctx = len(CONTEXT.findall(text))
    conf = sorted(k for k in sub if k in CONFOUND and CONFOUND[k].search(text))
    amb_only = bool(sub) and all(is_ambig(k) for k in sub)
    if sents and not sub:
        flag = "F1_statement_only"
    elif amb_only and (n_ctx == 0 or (conf and n_ctx < CONF_MAX_CONTEXT)):
        flag = "F2_ambiguous"
    else:
        flag = "kept"
    return dict(corpus=corpus, pmcid=r["pmcid"], n_focal=len(sents), n_statement=lab.count("statement"),
                terms=";".join(f"{k}:{v}" for k, v in sorted(allt.items())),
                terms_focal=";".join(sorted(sub)), ambiguous_only=int(amb_only), n_context=n_ctx,
                confound=";".join(conf), flag=flag,
                example=(sents[0]["text"][:300] if sents else ""))


def jobs(corpus):
    ext, clean = CORPORA[corpus]
    idx = files_by_article(clean)
    for line in open(ext, encoding="utf-8"):
        pm = line[11:line.index('"', 11)]
        yield corpus, line, idx.get(pm, [])


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 24
    os.makedirs(OUT, exist_ok=True)
    out = OUT + "stage_b_articles.tsv"
    assert not os.path.exists(out), out
    year = {r["pmcid"]: r["year"] for r in csv.DictReader(open(GR + "cohort.tsv"), delimiter="\t")}
    cols = ["corpus", "pmcid", "year", "flag", "n_focal", "n_statement", "terms_focal", "ambiguous_only", "n_context",
            "confound", "terms", "example"]
    summ = {}
    with open(out, "w", newline="", encoding="utf-8") as f, Pool(w) as pool:
        wr = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); wr.writeheader()
        for corpus in ("gap", "current"):
            c, cy = collections.Counter(), collections.defaultdict(collections.Counter)
            for row in pool.imap(one, jobs(corpus), chunksize=50):
                row["year"] = year.get(row["pmcid"], "") if corpus == "gap" else ""
                wr.writerow(row); c[row["flag"]] += 1; cy[row["year"]][row["flag"]] += 1
            summ[corpus] = dict(total=sum(c.values()), flags=dict(c), by_year={y: dict(v) for y, v in sorted(cy.items())})
            print(corpus, json.dumps(summ[corpus]["flags"]), flush=True)
    json.dump(summ, open(OUT + "stage_b_summary.json", "w"), indent=1)
