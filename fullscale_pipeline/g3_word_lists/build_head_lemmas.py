"""Build the head-word lemma table for G3 (G3_POSTPROCESSING.md §8 item 5, owner 2026-10-10, RUN_LOG RL-128/RL-129).

G2 joins hyphenated words with "_" (pre-trained -> pre_trained) so that graphbrain keeps the clause structure. spaCy's
lemmatiser (en_core_web_trf, rule mode) then sees an out-of-vocabulary word and applies its first suffix rule unchecked
(-ed/-ing -> -e, -est -> ""): pre_trained -> pre_traine, question_answering -> question_answere, second_best -> second_b.
G3 rebuilds such a lemma from the part before the last "_" plus the lemma that G2's own parse gives the last part when it
stands alone as a word. This script collects those plain-word lemmas from every `_lemma` edge of G2 v3 (word, coarse
type -> most frequent lemma), keeping only the last parts of joined words whose lemma spaCy changed.

    python build_head_lemmas.py            (writes head_lemmas.tsv beside this script; reads PG/g2_v3/shards/)

Output columns: word, type (coarse, first letter), lemma, count (plain occurrences with that lemma), total (plain
occurrences of word+type), joined (occurrences of joined words ending in this part whose lemma spaCy changed).
"""
import collections, glob, os, re, urllib.parse

G2 = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r'_lemma ([^\s/()]+)/([A-Za-z])[^\s()]* ([^\s/()]+)/')
JOINED = re.compile('(.+[_\u2010\u2011\u2013])([^_\u2010\u2011\u2013]+)')   # same as G3: "_" or a Unicode hyphen


def main():
    plain = collections.defaultdict(collections.Counter)
    needed = collections.Counter()
    for f in sorted(glob.glob(G2)):
        for line in open(f, encoding="utf-8"):
            for root, t, lem in PAT.findall(line):
                root, lem = urllib.parse.unquote(root), urllib.parse.unquote(lem)
                jm = JOINED.fullmatch(root)
                if jm:
                    if root.lower() != lem.lower():
                        needed[(jm.group(2).lower(), t)] += 1
                else:
                    plain[(root.lower(), t)][lem.lower()] += 1
    rows = ["word\ttype\tlemma\tcount\ttotal\tjoined"]
    for (w, t), n in sorted(needed.items()):
        if (w, t) in plain:
            (lem, c), tot = plain[(w, t)].most_common(1)[0], sum(plain[(w, t)].values())
            rows.append(f"{w}\t{t}\t{lem}\t{c}\t{tot}\t{n}")
    path = os.path.join(HERE, "head_lemmas.tsv")
    open(path, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print(path, len(rows) - 1, "rows;", sum(1 for k in needed if k not in plain), "last parts never seen alone")


if __name__ == "__main__":
    main()
