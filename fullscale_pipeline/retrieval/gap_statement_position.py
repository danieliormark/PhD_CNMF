"""Where do AI-use statements sit in the article? (owner, 2026-10-20, RL-159; measurement only, no flag changes)
For every focal sentence that Stage B labels an AI-use statement (both corpora), from the raw text:
    heading       the nearest heading line above it (P0's is_head)
    head_back     that heading is back matter in P0's sense or names AI use (Acknowledgements, Declarations, Use of AI, ...)
    intro_before  an Introduction heading occurs above it
    body_after    a Results / Discussion / Conclusion heading occurs below it
Owner's reading: a statement under an AI-use or similar heading, or after the last body section, is a genuine
declaration; one between the Introduction and a later Discussion/Conclusion is unlikely to be one.
    python3 gap_statement_position.py [--workers N]     -> gap_stage/stage_b/position/statement_position.tsv
"""
import csv, json, os, re, sys, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import gap_stage_b as SB
import build_content_regions as P0

L = SB.L
RAW = {"gap": L + "pure_text_corpus_gap/", "current": L + "pure_text_corpus/"}
OUT = SB.OUT + "position/"
AIHEAD = re.compile(r"(?i)\b(?:AI|artificial intelligence|generative|large language|LLMs?|chatgpt)\b")


def one(args):
    corpus, line, flag = args
    r = json.loads(line)
    sents = [s for b in r["blocks"] for s in b["sentences"] if s["focal"]]
    lab = ["statement" if SB.is_statement(s["text"]) else "focal" for s in sents]
    if "statement" in lab:
        lab = ["statement" if l == "statement" or SB.NEGATED_LOOSE.search(s["text"]) else l for s, l in zip(sents, lab)]
    st = [s for s, l in zip(sents, lab) if l == "statement"]
    if not st:
        return []
    Lr = open(RAW[corpus] + r["pmcid"] + ".txt", encoding="utf-8", errors="replace").read().split("\n")
    heads = [(i, P0.norm(Lr[i])) for i in range(len(Lr)) if P0.is_head(Lr, i)]
    out = []
    for s in st:
        i = s["raw_line"] - 1
        above = [h for h in heads if h[0] < i]
        below = [h for h in heads if h[0] > i]
        h = above[-1][1] if above else ""
        out.append(dict(corpus=corpus, pmcid=r["pmcid"], flag=flag, sid=s["sid"], heading=h,
                        head_back=int(bool(h) and (bool(P0.BACK.match(h)) or bool(AIHEAD.search(h)))),
                        intro_before=int(any(P0.INTRO.match(x[1]) for x in above)),
                        body_after=int(any(P0.RES.match(x[1]) or P0.DIS.match(x[1]) or P0.CON.match(x[1]) for x in below)),
                        text=s["text"][:300]))
    return out


def jobs(flags):
    for corpus in ("gap", "current"):
        ext = SB.CORPORA[corpus][0]
        for line in open(ext, encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            f = flags.get((corpus, pm))
            if f in ("F1_statement_only", "kept", "F2_ambiguous"):
                yield corpus, line, f


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 24
    csv.field_size_limit(10 ** 8)
    flags = {(r["corpus"], r["pmcid"]): r["flag"] for r in csv.DictReader(open(SB.OUT + "stage_b_articles.tsv"), delimiter="\t")}
    os.makedirs(OUT, exist_ok=True)
    out = OUT + "statement_position.tsv"
    assert not os.path.exists(out), out
    cols = ["corpus", "pmcid", "flag", "sid", "heading", "head_back", "intro_before", "body_after", "text"]
    c = collections.Counter()
    with open(out, "w", newline="", encoding="utf-8") as f, Pool(w) as pool:
        wr = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); wr.writeheader()
        for rows in pool.imap(one, jobs(flags), chunksize=50):
            for row in rows:
                wr.writerow(row)
                c[(row["corpus"], row["flag"], row["head_back"], row["intro_before"] and row["body_after"])] += 1
    for k, v in sorted(c.items()):
        print(k, v)
