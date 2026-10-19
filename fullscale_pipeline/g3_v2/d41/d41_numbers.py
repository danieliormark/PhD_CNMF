# D4.1 (read-only): candidate signs of footnote / superscript citation numbers that survive as atoms in the G3 v1 output.
# (a) a number (or comma list of numbers) after a word and followed by a space and punctuation, as flattened superscripts are
#     written in the PMC text ("based on BERT 18 .", "TextRank 17 ,"); (b) a small number between a capitalised word and a
#     lower-case word ("PubMed 1 abstracts"). Counts only numbers that are kept atoms of the unit.
import glob, json, random, re, sys
from multiprocessing import Pool

SH = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v1/shards/"
A = re.compile(r"(?<=[A-Za-z)\]] )(\d{1,3}(?:\s?[,–-]\s?\d{1,3})*)(?=\s+[.,;:)\]](?:\s|$)|\s*$)")
B = re.compile(r"\b[A-Z][A-Za-z0-9\-]+ (\d{1,2})(?= [a-z]{3,})")


def one(path):
    out = {"a": [], "b": [], "units": 0}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); out["units"] += 1
        kept = set()
        for s in [c for p in r["parents"] for c in p["children"]] + r["cousins"]:
            for at in s.get("atoms", []):
                w = at["atom"].split("/")[0]
                if re.fullmatch(r"\d+", w):
                    kept.add(w)
        if not kept:
            continue
        t = r["text"]
        for k, rx in (("a", A), ("b", B)):
            for m in rx.finditer(t):
                nums = re.findall(r"\d+", m.group(1))
                if any(n in kept for n in nums):
                    out[k].append((r["uid"], m.group(1), t[max(0, m.start() - 70):m.end() + 40]))
    return out


if __name__ == "__main__":
    tot = {"a": [], "b": [], "units": 0}
    with Pool(25) as pool:
        for o in pool.imap_unordered(one, sorted(glob.glob(SH + "g3_test_*.jsonl"))):
            tot["a"] += o["a"]; tot["b"] += o["b"]; tot["units"] += o["units"]
    json.dump(tot, open(sys.argv[1], "w"), ensure_ascii=False)
    random.seed(41)
    for k in ("a", "b"):
        print(f"\n== sign ({k}): {len(tot[k]):,} matches")
        for uid, n, ctx in random.sample(tot[k], 40):
            print(f"  [{n}] {ctx}")
