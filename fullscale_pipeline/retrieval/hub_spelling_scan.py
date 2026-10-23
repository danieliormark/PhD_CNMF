"""RL-184 (owner 2026-10-23): how many model names written in lower-case Hugging Face style (org/name-size-variant, a prefix
before the family: xlm-roberta-base, flan-t5-large, mistral-7b-instruct, dmis-lab/biobert-v1.1) does the v3 matcher miss, and
what does that cost the corpus? Read-only. "Language model" is defined by our own list, not by Hugging Face: the families
are the parts of the canonical focal names G3 wrote (at least MIN_MENTIONS mentions), without generic words.
Steps: (1) families from the G3 output; (2) a regex finds lower-case hub spellings containing a family in every cleaned full
text (current corpus and the downloaded candidates); (3) the sentence of each candidate goes through the v3 matcher, and a
spelling counts as missed only if no matcher hit covers it; (4) tallies: missed spellings in sentences that have another
focal hit (kept, the name an ordinary word), in sentences with no focal hit (lost sentences), and articles outside the
corpus whose only evidence would be such spellings.
    python3 hub_spelling_scan.py [--workers 16] [--out PG/hub_spelling_scan.json]
"""
import argparse, collections, glob, json, os, re, sys
from multiprocessing import Pool

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
TEXT_DIRS = {"current": LCS + "clean_corpus_v2/", "candidate": LCS + "gap_stage/llm_corpus_staging/clean_corpus_v2/"}
MIN_MENTIONS = 20
GENERIC = {"llm", "language", "model", "models", "large", "small", "transformer", "chatbot", "foundation", "pro", "mini", "flash",
           "turbo", "nano", "lite", "ultra", "max", "sonnet", "opus", "haiku", "microsoft", "copilot", "openai", "google", "meta",
           "ai", "chat", "med", "bio", "clinical", "base"}
STEMS = re.compile(r"bert|gpt|llama|roberta|mistral|mixtral|gemma|qwen|palm|bart|electra|xlnet|claude|gemini|deepseek|"
                   r"falcon|bloom|vicuna|alpaca|grok|bard|ernie|baichuan|olmo|galactica|minicpm|internlm|longformer|bigbird|"
                   r"megatron|gatortron|flan|t5|esm|opt|phi|glm|yi\b")
SHORT = re.compile(r"^(t5|esm|opt|phi|glm|yi|bard|palm)$")                   # short stems: only with a digit after them
HUB_SIGNAL = re.compile(r"\d|/|(?:^|[-_.])(base|large|small|medium|xl|xxl|tiny|mini|instruct|chat|hf|uncased|cased|v\d)(?:$|[-_.])")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def families():
    cnt = collections.Counter()
    for line in open(PG + "g3_v2/g3_test_v2.jsonl", encoding="utf-8"):
        for m in json.loads(line)["focal_mentions"]:
            cnt[m["canonical"]] += 1
    fam = collections.Counter()
    for c, n in cnt.items():
        for part in c.split("_"):
            if re.fullmatch(r"[a-z]{2,}\d*[a-z]*", part) and part not in GENERIC and not part.isdigit() and STEMS.search(part):
                fam[part] += n
    return sorted((f for f, n in fam.items() if n >= MIN_MENTIONS), key=len, reverse=True)


def hub_regex(fams):
    alt = "|".join(re.escape(f) + (r"(?=[-_.]?\d)" if SHORT.match(f) else "") for f in fams)
    # lower-case only, at least one separator: (org/)? (prefix-)* family [chars] (-segment)+  or  (org/)family[chars]
    return re.compile(r"(?<![A-Za-z0-9_./-])((?:[a-z0-9][a-z0-9_.]*/)?(?:[a-z0-9]+[-_.])*(?:" + alt + r")[a-z0-9.]*(?:[-_][a-z0-9.]+)+"
                      r"|[a-z0-9][a-z0-9_.-]*/(?:[a-z0-9]+[-_.])*(?:" + alt + r")[a-z0-9.-]*)(?![A-Za-z0-9_/])")


def scan_file(args):
    corpus, path, rx_src = args
    rx = re.compile(rx_src)
    global MATCHER
    out = []
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return os.path.basename(path)[:-4], corpus, []
    for para in text.split("\n"):
        if not rx.search(para):
            continue
        for sent in SENT.split(para):
            found = [(m.start(1), m.end(1), m.group(1).rstrip(".")) for m in rx.finditer(sent) if HUB_SIGNAL.search(m.group(1))]
            if not found:
                continue
            hits = MATCHER.hits(sent, False)[0]
            spans = [h[1] for h in hits]
            for a, b, tok in found:
                covered = any(s < b and e > a for s, e in spans)
                if not covered:
                    out.append(dict(tok=tok, other_hit=bool(hits), sent=sent[:300]))
    return os.path.basename(path)[:-4], corpus, out


def init():
    global MATCHER
    sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
    import focal_terms_v3
    MATCHER = focal_terms_v3.Matcher()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=PG + "hub_spelling_scan.json")
    ap.add_argument("--limit", type=int, default=0, help="files per corpus (0 = all; for a quick test)")
    a = ap.parse_args()
    fams = families()
    rx = hub_regex(fams)
    status = {}
    for line in open(PG + "g3_v2/g3_articles_v2.jsonl"):
        r = json.loads(line); status[r["pmcid"]] = r["status"]
    jobs = []
    for corpus, d in TEXT_DIRS.items():
        files = sorted(glob.glob(d + "PMC*.txt"))
        jobs += [(corpus, f, rx.pattern) for f in (files[:a.limit] if a.limit else files)]
    T = collections.Counter(); toks = collections.Counter(); lost_art = collections.defaultdict(list); samples = collections.defaultdict(list)
    with Pool(a.workers, initializer=init) as pool:
        for pm, corpus, found in pool.imap_unordered(scan_file, jobs, chunksize=64):
            T["files_" + corpus] += 1
            st = status.get(pm, "not_in_corpus")
            for f in found:
                key = ("in corpus (" + st + ")" if st != "not_in_corpus" else "not in corpus") + (", sentence has another focal hit" if f["other_hit"] else ", sentence has no focal hit")
                T[key] += 1; toks[f["tok"]] += 1
                if len(samples[key]) < 15:
                    samples[key].append([pm, f["tok"], f["sent"]])
                if st == "not_in_corpus" and not f["other_hit"]:
                    lost_art[pm].append(f["tok"])
    rep = dict(families=len(fams), family_examples=fams[:60], counts=dict(T), distinct_spellings=len(toks),
               top_spellings=toks.most_common(80), articles_not_in_corpus_with_unmatched_only_sentences=len(lost_art),
               lost_article_examples=dict(list(lost_art.items())[:40]), samples=dict(samples))
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("families", "counts", "distinct_spellings", "articles_not_in_corpus_with_unmatched_only_sentences")}, indent=1))


if __name__ == "__main__":
    main()
