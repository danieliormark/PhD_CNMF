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
Release-date rule (owner, 2026-10-20), applied first and per term hit: a named model cannot be meant before it was
released, so a hit in an article published before the model's release year (RELEASE, year of the first public release
or paper; same year allowed) is removed; a sentence left without a term is no longer focal. General terms ("language
model", LLM, "transformer model") have no release year and are untouched. Removed hits are listed per sentence in
stage_b_prerelease_hits.tsv (also for articles that keep other focal sentences).
Article flags (one per article, first that applies):
    F0 hits_rejected    every focal sentence lost its terms to the release-date rule or a homonym rule (RL-160: the
                        hit's sentence shows the other meaning, e.g. "Progen, Heidelberg", "Zeiss Gemini450 SEM",
                        "QGSP_BERT", a transformer "winding", or the article expands LLM as something else)
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
MANU = (r"(?:manuscript|this (?:work|paper|article|review|report|text)|the (?:final )?(?:text|draft|abstract|title)|"
        r"\barticle\b|\bpaper\b|writing|rephras\w*|paraphras\w*|draft\w*|translat\w*|figures?\b|graphics?\b|graphical|"
        r"schematic|illustrations?|layouts?|syntax|punctuation|typograph\w*|wording|phrasing|clarity|coherence|cohesion|"
        r"precision|key arguments|structuring the content|reference list|citations|formulation\w* of (?:the )?text|"
        r"literature (?:discovery|search)|search terms|language optimi[sz]ation|"
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
                       rf"{MANU}.{{0,60}}(?:was|were) (?:written|generated|created|edited|revised|refined|polished|proofread|translated|"
                       rf"assisted|prepared|reviewed for grammar).{{0,40}}(?:with|by|using|via)\b")   # reverse order: "to improve the manuscript, we used"; "graphics ... with the help of"
NEGATED = re.compile(r"(?i)\b(?:no|not|never)\b.{0,150}\b(?:use|used|utili[sz]ed|employed|generate|alter|modify)\b.{0,120}"
                     r"(?:manuscript|paper\b|article\b|writing|submitted work|production|preparation|this (?:work|review|study)|any aspect|the conduct|"
                     r"image (?:creators|generators)|(?:the |patient |original )images|(?:study|original) data)")
NEGATED_LOOSE = re.compile(r"(?i)\b(?:no|not|never)\b.{0,150}\b(?:use|used|utili[sz]ed|employed|generate|alter|modify|"
                           r"analy[sz]e|interpret|create|produce)\b")   # companion sentence: only next to a statement
RESEARCH = re.compile(r"(?i)evaluat|assess(?:ed|ing|ment)? (?:the |its |their )?(?:performance|accuracy|ability|quality of)|compar(?:ed|ing|ison)|"
                      r"\brated\b|ratings?\b|scores?\b|benchmark|participants|patients|respondents|students|questions|answers|"
                      r"generated? (?:patient|educational)|outputs?\b")
STATEMENT_NEED_AUTHOR = re.compile(rf"(?i){AUTHOR}|writers?\b|co-?authors?|author\(s\)|(?:was|were|has been|have been) "
                                   r"(?:used|employed|utili[sz]ed|created|revised|edited)|is acknowledged|played .{0,30}role|"
                                   r"appreciation|acknowledge|grateful|thank|(?:was|were) (?:assisted|made|written|generated)|"
                                   r"this (?:manuscript|article|paper)")


RELEASE = [(re.compile(p), y) for p, y in [   # first match wins: specific names before families
    (r"chat\s?-?gpt", 2022), (r"instructgpt", 2022), (r"gpt[- ]?5", 2025), (r"gpt[- ]?4\.5", 2025), (r"gpt[- ]?4o", 2024),
    (r"gpt[- ]?4", 2023), (r"gpt[- ]?3\.5", 2022), (r"gpt[- ]?3", 2020), (r"gpt[- ]?2", 2019), (r"openai o[13]", 2024),
    (r"llama[- ]?3", 2024), (r"llama[- ]?2", 2023), (r"llama", 2023), (r"palm[- ]?2|palm-e", 2023), (r"gemini", 2023),
    (r"bard", 2023), (r"gemma", 2024), (r"mixtral", 2023), (r"mistral", 2023), (r"claude[- ]?3|claude (?:opus|sonnet|haiku)", 2024),
    (r"claude", 2023), (r"grok", 2023), (r"deepseek", 2023), (r"qwen", 2023), (r"galactica", 2022), (r"progen2", 2022),
    (r"progen", 2020), (r"esm-?2", 2022), (r"esm-1v", 2021), (r"esm-1b|esm-msa-1b", 2019), (r"enformer", 2021),
    (r"pubmedbert", 2020), (r"biobert|scibert|clinicalbert", 2019), (r"roberta", 2019), (r"bert", 2018),
    (r"github copilot", 2021), (r"microsoft copilot", 2023), (r"perplexity ai", 2022), (r"gpt", 2018), (r"palm", 2022)]]
GENERAL = {"language model", "large language model", "large language models", "llm", "llms", "transformer model",
           "transformer models", "bidirectional encoder representations from transformers"}


# Homonym rules per hit (2026-10-20, RL-160): the hit's own sentence (or the article's abbreviation definitions) shows the
# other meaning. Each returns a reason; the hit is removed like a pre-release hit.
ABBR = re.compile(r"((?:[A-Za-z][\w’'-]*\s+){1,6})\(\s*(LLMs?|GPT|BERT)\s*\)")
LM_SENSE = re.compile(r"(?i)language\b.*\bmodel|generative pre-?train|bidirectional encoder")
LM_CUE = re.compile(r"(?i)language models?|ChatGPT|GPT-?\d|prompt|tokens?\b|embedding|fine-?tun|pre-?train|"
                    r"transformer-based|transformer architecture|neural|deep learning|\bPLMs?\b|generative AI")   # not the terms themselves   # a sentence rule never applies next to these
SENT_RULES = {   # term key -> (reason, pattern in the hit's sentence)
    "progen": ("progen_supplier", None),                       # special: ProGen the model is written "ProGen"
    "gemini": ("gemini_instrument", re.compile(r"Gemini\s?\d{3}|GeminiSEM|Gemini SEM|Zeiss[^.]{0,40}Gemini|Gemini[^.]{0,20}(?:SEM|microscop)")),
    "bert": ("bert_geant4", re.compile(r"(?:QGSP|FTFP|QBBC|QGSC)_?BERT|BERT_HP|Bertini|FreeSurfer[^.]{0,60}\bbert\b|\bbert\b[^.]{0,60}FreeSurfer", re.I)),
    "transformer model": ("transformer_electrical", re.compile(r"(?i)winding|voltage|\bcoils?\b|inductor|inductance|eddy[- ]current|"
                          r"magnetic core|primary and secondary|power transformer|transformer oil|\bkV\b|\bMVA\b|tokamak|"
                          r"three[- ]phase|3-phase|\bRLC\b|differential transformer|air[- ]core")),
    "language model": ("language_model_linguistic", re.compile(r"(?i)sign language|for (?:their )?child|children'?s? (?:second )?language|"
                       r"caregiver|(?:adult|parent|teacher)s? (?:as |serve as )?(?:good )?language models?|aphasia|Broca|Wernicke")),
    "gpt2": ("gpt_enzyme", re.compile(r"\bGPT1\b|\bGOT[12]?\b|transaminase|aminotransferase|alanine|isoform|mitochondrial")),
    "gpt": ("gpt_enzyme", re.compile(r"\bGPT1\b|\bGOT[12]?\b|transaminase|aminotransferase|alanine|\bSGPT\b")),
    "esm-2": ("esm_supplement_or_climate", re.compile(r"(?i)electronic supplementary|ESM-?2[,;)]\s*(?:tables?|figs?)|CMIP|ACCESS-ESM|"
                                                r"CESM\d|earth system|climate|\bGCMs?\b")),
    "palm 2": ("palm_domain", re.compile(r"(?i)palm (?:1|2|I|II)\b.{0,20}domain|palm domain|thumb|fingers? (?:sub)?domain|polymerase|RdRp|cyclase")),
    "llm": ("llm_other_sense", re.compile(r"LLM-1\d\d\b|lean mass|lipid[- ]lowering|lipid[- ]laden|log[- ]layer mismatch|Rulex|"
                                          r"logic learning machine|laminated limestone|\bLLm\b|MCAN/")),
}
SENT_RULES["transformer models"] = SENT_RULES["transformer model"]
ML_CUE = re.compile(r"(?i)network|trained|training|learning|neural|deep|attention|encoder|decoder|predict\w*|classif\w*|"
                    r"embedding|token|self-attention|architecture|Vision Transformer|ViT|BERT|GPT|LSTM|CNN")
SENT_RULES["llms"] = SENT_RULES["llm"]


def abbr_senses(text):
    """the article's own expansions of LLM/GPT/BERT: {abbr: set of 'lm' / 'other'} from "long form (ABBR)"; a long form
    counts only if the initials of its last words spell the abbreviation (hyphenated parts count as words), the usual
    abbreviation-definition test; an article that mentions "large language model" anywhere gets no 'other' sense"""
    out = collections.defaultdict(set)
    llm_text = bool(re.search(r"(?i)large[- ]language[- ]models?", text))
    for m in ABBR.finditer(text):
        ab = m.group(2).upper().rstrip("S")
        words = re.findall(r"[A-Za-z]+", m.group(1))                    # hyphen parts as words: lipid-lowering = L L
        joined = re.findall(r"[A-Za-z][A-Za-z-]*", m.group(1))           # or as one word: pre-trained = P
        if not any(len(w) >= len(ab) and "".join(x[0] for x in w[-len(ab):]).upper() == ab for w in (words, joined)):
            continue
        if LM_SENSE.search(m.group(1)):
            out[ab].add("lm")
        elif not llm_text:
            out[ab].add("other")
    return out


def homonym_reason(t, text, senses):
    """reason to reject hit t in a sentence with this text, or None"""
    k = t.lower() if not t.startswith("v3:") else t.split(":", 2)[2].lower()
    if k in ("llm", "llms") and senses.get("LLM") == {"other"} and not LM_CUE.search(text):
        return "llm_defined_otherwise"
    if k == "gpt" and senses.get("GPT") == {"other"} and not LM_CUE.search(text):
        return "gpt_defined_otherwise"
    if k == "progen":
        model = re.search(r"ProGen|(?i:generat\w* (?:\w+ )?(?:proteins?|sequences?)|protein generation|language model)", text)
        return None if (model or LM_CUE.search(text)) else "progen_supplier"
    rule = SENT_RULES.get(k) or (SENT_RULES.get("gemini") if t.startswith("v3:") and "gemini" in k else None)
    if rule and rule[1] is not None and rule[1].search(text) and not LM_CUE.search(text):
        if rule[0] == "transformer_electrical" and ML_CUE.search(text):
            return None                      # a machine-learning transformer applied to an electrical or physical problem
        return rule[0]
    return None


def release_year(t):
    """release year of the model a term names, None for general terms; v3 terms by their surface form"""
    name = t.split(":", 2)[2] if t.startswith("v3:") else t
    name = name.lower()
    if name in GENERAL or (t.startswith("v3:A6")):
        return None
    for rx, y in RELEASE:
        if rx.search(name):
            return y
    return None


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
    corpus, line, files, year = args
    r = json.loads(line)
    sents0 = [s for b in r["blocks"] for s in b["sentences"] if s["focal"]]
    text = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    senses = abbr_senses(text)
    pre, sents = [], []
    for s in sents0:
        keep = []
        for t in s["terms"]:
            ry = release_year(t)
            why = f"prerelease<{ry}" if (year and ry and year < ry) else homonym_reason(t, s["text"], senses)
            if why:
                pre.append((s["sid"], t, why))
            else:
                keep.append(t)
        if keep:
            sents.append(dict(s, terms=keep))
    lab = ["statement" if is_statement(s["text"]) else "focal" for s in sents]
    if "statement" in lab:          # "... was not used to generate content or analyse data" completes a disclosure
        lab = ["statement" if l == "statement" or NEGATED_LOOSE.search(s["text"]) else l for s, l in zip(sents, lab)]
    sub = collections.Counter(term_key(t) for s, l in zip(sents, lab) if l == "focal" for t in s["terms"])
    allt = collections.Counter(term_key(t) for s in sents for t in s["terms"])
    n_ctx = len(CONTEXT.findall(text))
    conf = sorted(k for k in sub if k in CONFOUND and CONFOUND[k].search(text))
    amb_only = bool(sub) and all(is_ambig(k) for k in sub)
    if sents0 and not sents:
        flag = "F0_hits_rejected"
    elif sents and not sub:
        flag = "F1_statement_only"
    elif amb_only and (n_ctx == 0 or (conf and n_ctx < CONF_MAX_CONTEXT)):
        flag = "F2_ambiguous"
    else:
        flag = "kept"
    return dict(corpus=corpus, pmcid=r["pmcid"], n_focal=len(sents), n_statement=lab.count("statement"),
                terms=";".join(f"{k}:{v}" for k, v in sorted(allt.items())),
                terms_focal=";".join(sorted(sub)), ambiguous_only=int(amb_only), n_context=n_ctx,
                confound=";".join(conf), flag=flag, n_prerelease_hits=len(pre),
                prerelease=";".join(sorted({f"{t}:{why}" for _, t, why in pre})), _pre=pre,
                example=((sents or sents0)[0]["text"][:300] if sents0 else ""))


def jobs(corpus, year):
    ext, clean = CORPORA[corpus]
    idx = files_by_article(clean)
    for line in open(ext, encoding="utf-8"):
        pm = line[11:line.index('"', 11)]
        yield corpus, line, idx.get(pm, []), year.get(pm)


def years():
    """publication year per article: the gap cohort's year; for the current corpus the first 19xx/20xx of R1's pubdate"""
    gy = {r["pmcid"]: int(r["year"]) for r in csv.DictReader(open(GR + "cohort.tsv"), delimiter="\t") if r["year"].isdigit()}
    cy = {}
    for p, d in json.load(open(L + "target_metadata.json")).items():
        m = re.search(r"\b(19|20)\d\d\b", (d or {}).get("pubdate") or "")
        if m:
            cy[p] = int(m.group(0))
    return {"gap": gy, "current": cy}


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 24
    os.makedirs(OUT, exist_ok=True)
    out = OUT + "stage_b_articles.tsv"
    assert not os.path.exists(out), out
    Y = years()
    cols = ["corpus", "pmcid", "year", "flag", "n_focal", "n_statement", "terms_focal", "ambiguous_only", "n_context",
            "confound", "n_prerelease_hits", "prerelease", "terms", "example"]
    summ = {}
    with open(out, "w", newline="", encoding="utf-8") as f, open(OUT + "stage_b_rejected_hits.tsv", "w") as fp, Pool(w) as pool:
        wr = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); wr.writeheader()
        fp.write("corpus\tpmcid\tyear\tsid\tterm\treason\n")
        for corpus in ("gap", "current"):
            c, cy = collections.Counter(), collections.defaultdict(collections.Counter)
            for row in pool.imap(one, jobs(corpus, Y[corpus]), chunksize=50):
                row["year"] = Y[corpus].get(row["pmcid"], "")
                for sid, t, ry in row.pop("_pre"):
                    fp.write(f"{corpus}\t{row['pmcid']}\t{row['year']}\t{sid}\t{t}\t{ry}\n")
                wr.writerow(row); c[row["flag"]] += 1; cy[row["year"]][row["flag"]] += 1
            summ[corpus] = dict(total=sum(c.values()), flags=dict(c), by_year={str(y): dict(v) for y, v in sorted(cy.items(), key=lambda x: str(x[0]))})
            print(corpus, json.dumps(summ[corpus]["flags"]), flush=True)
    json.dump(summ, open(OUT + "stage_b_summary.json", "w"), indent=1)
