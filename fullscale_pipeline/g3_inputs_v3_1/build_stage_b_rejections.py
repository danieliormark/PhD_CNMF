"""G3 input for the merged corpus (RL-171): the focal hits Stage B rejected, so that G3, which re-finds focal terms in each
unit's text with P2's matcher (now v3), does not bring them back.

Stage B's hit rules were applied to P2 v3's sentences in retrieval/inc_prepare.py (clean): a term is dropped from a
sentence when the term's model was released after the article's year, when a homonym rule fires on the sentence
(gap_stage_b.homonym_reason, with the article's own abbreviation senses), or when the F2 manual check judged the term in
that article other-sense or statement. P7 then kept only the remaining terms, but G3 matches again from the text, so the
same rules are recomputed here, with the same calls, on every focal sentence of P2 v3 for every article of
PG/g2_v3_1/:
    sentence  sid    term   homonym_rule              (this sentence only: the rule reads the sentence)
    article   pmcid  term   prerelease | f2_verdict   (every sentence of the article: the rule reads only article and term)
Output PG/g3_inputs_v3_1/stage_b_rejections.tsv (scope, key, term, reason) and stage_b_rejections_summary.json.
    python3 build_stage_b_rejections.py [--workers N]
"""
import collections, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/retrieval")
import gap_stage_b as SB
import inc_prepare as IP

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
OUT = PG + "g3_inputs_v3_1/"
MERGED = PG + "g2_v3_1/focal_sentences_v2.jsonl"


def scan(args):
    corpus, line = args
    r = json.loads(line)
    pm = r["pmcid"]; year = IP._Y[corpus].get(pm)
    text = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in IP._FILES[corpus].get(pm, []))
    senses = SB.abbr_senses(text)
    rows, seen = [], set()
    for b in r["blocks"]:
        for s in b["sentences"]:
            if not s["focal"]:
                continue
            for t in s["terms"]:                                   # the order of inc_prepare.clean: first rule that fires
                ry = SB.release_year(t)
                if year and ry and year < ry:
                    why, scope, key = "prerelease", "article", pm
                elif SB.homonym_reason(t, s["text"], senses):
                    why, scope, key = "homonym_rule", "sentence", s["sid"]
                elif IP._F2.get((corpus, pm, IP.term_label(t))) in ("no", "statement"):
                    why, scope, key = "f2_verdict", "article", pm
                else:
                    continue
                if (scope, key, t) not in seen:
                    seen.add((scope, key, t)); rows.append((scope, key, t, why))
    return rows


def jobs(arts):
    for corpus in ("gap", "current"):
        for line in open(IP.SRC[corpus], encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if arts.get(pm) == corpus:
                yield corpus, line


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 24
    os.makedirs(OUT, exist_ok=True)
    assert not os.path.exists(OUT + "stage_b_rejections.tsv"), "outputs exist"
    new_ids = set(open(SB.GR + "gap_article_list_after_ab.txt").read().split())
    arts = {}
    for line in open(MERGED, encoding="utf-8"):
        pm = line[11:line.index('"', 11)]
        arts[pm] = "gap" if pm in new_ids else "current"
    c = collections.Counter()
    with Pool(w, initializer=IP.init) as pool, open(OUT + "stage_b_rejections.tsv", "w", encoding="utf-8") as fo:
        fo.write("scope\tkey\tterm\treason\n")
        for rows in pool.imap(scan, jobs(arts), chunksize=50):
            c["articles scanned"] += 1
            c["articles with a rejection"] += bool(rows)
            for scope, key, t, why in rows:
                fo.write(f"{scope}\t{key}\t{t}\t{why}\n"); c[f"{scope}:{why}"] += 1
    c["articles in g2_v3_1"] = len(arts)
    json.dump(dict(c), open(OUT + "stage_b_rejections_summary.json", "w"), indent=1)
    print(json.dumps(dict(c), indent=1))
