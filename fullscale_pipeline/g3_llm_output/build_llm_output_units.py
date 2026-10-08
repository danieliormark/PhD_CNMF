"""Units that reproduce LLM-generated text (G3_POSTPROCESSING.md §6 item 7, owner decision 2026-10-08): dropped so
that model output is not conflated with researchers' own statements about LLMs. Not a parsing change -- G3 simply
skips every uid listed here, the same way it skips the articles in g3_scope_exclusions/scope_exclusions.csv.

A unit is LLM output if:
  (a) its prefix before the first ":" (<= 8 words) holds a focal term plus only a response word (response, output,
      answer, reply, ...) and filler words, or holds only the focal term and the ":" is followed by a quotation; or
  (b) it continues an open quotation opened by (a) -- the next unit of the SAME sentence (G3 split a long sentence
      at a clause boundary), or the next sentence by raw line -- so two G2 units that are merely the next FOCAL
      sentence (G2 keeps only focal sentences, so consecutive G2 records can be pages apart in the real article)
      are never wrongly joined.
A prefix holding a prompt/question word (prompt, asked, instructed, query, ...) is a researcher's prompt, not model
output, and is kept.

    python build_llm_output_units.py [--outdir DIR]       # default: this folder; writes llm_output_units.csv
"""
import argparse, csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARDS = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"
EXCLUDED = os.path.join(HERE, "..", "g3_scope_exclusions", "scope_exclusions.csv")
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms

RESP = {"response", "responses", "answer", "answers", "reply", "replies", "output", "outputs", "responded", "replied", "answered"}
PROMPT = {"prompt", "prompts", "asked", "instructed", "query", "queried", "input", "instruction", "instructions", "question", "questions", "told", "request"}
FILLER = {"from", "of", "the", "by", "to", "a", "an", "and", "its", "s", "1", "2", "3", "4", "5", "final", "initial", "original", "corresponding",
          "sample", "example", "full", "model", "was", "is", "were", "verbatim", "excerpt", "generated"}


def label_kind(matcher, text):
    """'output', 'prompt' or None for the unit's prefix before its first ':'"""
    if ":" not in text:
        return None
    pre, post = text.split(":", 1)
    if len(re.findall(r"[A-Za-z0-9]+", pre)) > 8:
        return None
    hits = matcher.hits(pre, True)[0]
    if not hits:
        return None
    rest = pre
    for _, (a, b), _ in sorted(hits, key=lambda h: -h[1][0]):
        rest = rest[:a] + " " + rest[b:]
    rest = re.sub(r"\([^)]*\)", " ", rest)
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", rest)}
    if words & PROMPT:
        return "prompt"
    quoted = bool(re.match(r"\s*[“‘\"'«]", post))
    if words & RESP and words <= RESP | FILLER:
        return "output"
    if not (words - FILLER) and quoted:
        return "output"
    return None


def quote_open_after(text, start_open):
    """does a curly double quotation ("...") remain open at the end of text? (apostrophes and straight quotes,
    which also mark single words and contractions, are too ambiguous to track this way)"""
    bal = (1 if start_open else 0) + text.count("“") - text.count("”")
    return bal > 0


def adjacent(a, b):
    """is sentence b the next unit of sentence a, or truly the next sentence? sid format PMCID.rN.L<line>.S<k>"""
    if not a:
        return False
    if a == b:                                    # the next unit cut from the same sentence
        return True
    ma, mb = re.match(r".*\.(r\d+)\.L(\d+)\.S(\d+)$", a), re.match(r".*\.(r\d+)\.L(\d+)\.S(\d+)$", b)
    if not (ma and mb) or ma.group(1) != mb.group(1):
        return False
    la, sa, lb, sb = int(ma.group(2)), int(ma.group(3)), int(mb.group(2)), int(mb.group(3))
    return (la == lb and sb == sa + 1) or (lb == la + 1 and sb == 1)


def find_units(excluded_articles=None):
    """[(pmcid, uid, unit_hash, reason, text)] for every LLM-output unit in the kept corpus"""
    matcher = focal_terms.Matcher()
    excluded_articles = excluded_articles or set()
    out = []
    for f in sorted(glob.glob(SHARDS)):
        for line in open(f):
            art = json.loads(line)
            if art["pmcid"] in excluded_articles:
                continue
            open_q, ncont, prev_sid = False, 0, None
            for s in art["sentences"]:
                for u in s.get("units", []):
                    text = u["text"]
                    kind = label_kind(matcher, text)
                    if kind == "output":
                        out.append((art["pmcid"], u["uid"], u["hash"], "label", text))
                        open_q = quote_open_after(text.split(":", 1)[1], False)
                        ncont = 0
                    elif open_q and ncont < 10 and adjacent(prev_sid, s["sid"]):
                        ncont += 1
                        out.append((art["pmcid"], u["uid"], u["hash"], "continuation", text))
                        open_q = quote_open_after(text, True)
                    else:
                        open_q = False
                    prev_sid = s["sid"]
    return out


def load_excluded(path):
    return {r["pmcid"] for r in csv.DictReader(open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()
    units = find_units(load_excluded(EXCLUDED))
    path = os.path.join(a.outdir, "llm_output_units.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmcid", "uid", "unit_hash", "reason"])
        w.writerows((p, u, h, r) for p, u, h, r, _ in units)
    by = {}
    for _, _, _, r, _ in units:
        by[r] = by.get(r, 0) + 1
    print(f"{len(units)} units in {len({p for p, _, _, _, _ in units})} articles {by}; wrote {path}")


if __name__ == "__main__":
    main()
