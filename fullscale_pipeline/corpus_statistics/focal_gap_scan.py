# D2 exploration (read-only): tokens that contain a model-family name but are not accepted by the current focal matcher.
# Scans every whitelist article's cleaned region files (P2's input). Output: JSON with per-token counts, examples, and for each
# occurrence whether its line already has an accepted hit and whether the article has a P2 focal sentence.
import collections, csv, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
from focal_terms import Matcher

P = "/mnt/hum01-rds/Basov/p91688di/"
LCS = P + "llm_corpus_staging/"
CLEAN = LCS + "clean_corpus_v2/"
OUT = sys.argv[1]
TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_]*(?:[-‐‑][A-Za-z0-9.+_]+)*")
NAMES = re.compile(r"(?i)bert|gpt|llama|palm|gemini|mistral|gemma|bard|language[-‐‑_]models?")
_M = None


def init():
    global _M
    _M = Matcher()


def one(item):
    art, files = item
    texts = {}
    for f in files:
        rid = 1 if f == art + ".txt" else int(f.split(".")[1][1:])
        texts[rid] = open(CLEAN + f, encoding="utf-8", errors="replace").read().split("\n")
    anchored = bool(_M.anchor_hits("\n".join("\n".join(v) for v in texts.values())))
    occ = []
    for rid, lines in texts.items():
        for i, l in enumerate(lines):
            if not l.strip() or not NAMES.search(l):
                continue
            acc, _ = _M.hits(l, anchored)
            spans = [sp for _, sp, _ in acc]
            for m in TOK.finditer(l):
                tok = m.group(0)
                for n in NAMES.finditer(tok):
                    a, b = m.start() + n.start(), m.start() + n.end()
                    if any(pa <= a and b <= pb for pa, pb in spans):
                        continue
                    occ.append(dict(rid=rid, line=i, tok=tok.strip("."), name=n.group(0), line_has_hit=bool(spans),
                                    ctx=l[max(0, m.start() - 70):m.end() + 70]))
                    break
    return art, anchored, occ


if __name__ == "__main__":
    white = set(open(LCS + "article_whitelist.txt").read().split())
    files = collections.defaultdict(list)
    for f in os.listdir(CLEAN):
        if f.endswith(".txt") and f.split(".")[0] in white:
            files[f.split(".")[0]].append(f)
    status = {r["pmcid"]: r["status"] for r in csv.DictReader(open(LCS + "focal_status_v2.csv"))}
    res = {}
    with Pool(24, initializer=init) as pool:
        for art, anch, occ in pool.imap_unordered(one, sorted(files.items()), chunksize=20):
            if occ:
                res[art] = dict(anchored=anch, status=status.get(art), occ=occ)
    json.dump(res, open(OUT, "w"))
    print("articles with candidates", len(res))
