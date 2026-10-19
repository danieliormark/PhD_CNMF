# D4.1 (read-only): collect suspect output atoms of the G3 v1 output, by kind, with source words and context.
import collections, glob, json, re, sys, urllib.parse
from multiprocessing import Pool

SH = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/shards/"
LIG = re.compile("[ﬀ-ﬆ]")
BRACKET = re.compile(r"[()\[\]{}]")
FORMULA = re.compile(r"[|=+^~<>′″‴∑∏√∂∫×÷·∈∀∃≠±∗∘⊂⊆∪∩→←↔⇒]")
GREEK = re.compile(r"[Ͱ-Ͽ]")
QUOTE = re.compile(r"[‘’‚‛“”„‟\"'`´]")


def kinds(word):
    k = []
    if LIG.search(word): k.append("ligature")
    if BRACKET.search(word): k.append("bracket")
    if FORMULA.search(word): k.append("formula_sign")
    if GREEK.search(word) and re.search(r"[a-z0-9]", word): k.append("greek_mixed")
    if QUOTE.search(word): k.append("quote")
    return k


def one(path):
    C = collections.defaultdict(collections.Counter); EX = collections.defaultdict(list); units = 0
    caption = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); units += 1
        t = r["text"]
        if re.match(r"^\(?[a-hA-H]\)?\s*[,|.:]\s", t) or re.match(r"^[a-h]\s*\|\s", t):
            caption.append((r["uid"], t[:200]))
        structs = [c for p in r["parents"] for c in p["children"]] + r["cousins"]
        for s in structs:
            for a in s.get("atoms", []):
                w = urllib.parse.unquote(a["atom"].split("/")[0])
                for k in kinds(w):
                    C[k][w] += 1
                    if len(EX[(k, w)]) < 2:
                        src = a["src"][0]; pos = src[2]
                        i = t.find(src[1])
                        EX[(k, w)].append((r["uid"], src[1], t[max(0, i - 60):i + len(src[1]) + 60] if i >= 0 else t[:150]))
    return units, {k: dict(v) for k, v in C.items()}, {f"{k}\t{w}": v for (k, w), v in EX.items()}, caption


if __name__ == "__main__":
    files = sorted(glob.glob(SH + "g3_test_*.jsonl"))
    TOT = collections.defaultdict(collections.Counter); EXS = {}; U = 0; CAP = []
    with Pool(25) as pool:
        for u, c, ex, cap in pool.imap_unordered(one, files):
            U += u; CAP += cap
            for k, v in c.items(): TOT[k].update(v)
            for k, v in ex.items(): EXS.setdefault(k, v)
    json.dump(dict(units=U, counts={k: v.most_common() for k, v in TOT.items()}, examples=EXS, caption_like=CAP),
              open(sys.argv[1], "w"), ensure_ascii=False)
    print("units", U)
    for k, v in TOT.items():
        print(f"\n== {k}: {sum(v.values()):,} atom occurrences, {len(v):,} forms")
        print("  ", ", ".join(f"{w}:{n}" for w, n in v.most_common(60)))
    print("\ncaption-like unit starts:", len(CAP))
