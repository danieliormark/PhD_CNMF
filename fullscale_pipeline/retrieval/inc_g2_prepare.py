"""Incremental run, step 5a (2026-10-21, RL-168): the G2 input — only the sentences that need a parse.
From the P7 output of the incremental run (inc_run/p7/out/focal_sentences_v2.mapped.jsonl), a sentence is parsed when
    new          its (sid, hash_final) is not in the existing G2 v3 parse (new article, new sentence, or changed text)
    apostrophe   its existing parse types a quote mark (’ or ‘) as a predicate (the 876 of RL-152, recounted here)
    hyphen_v3    its existing parse joined a hyphenated word whose parts now contain a v3 model name (BioMistral-7b …)
Everything else keeps its existing parse (merged per article in step 5c). Outputs in LCS/inc_run/g2_in/:
focal_sentences_inc.jsonl (P7 records holding only those sentences, input order = P7 order), reparse_reasons.tsv,
g2_prepare_summary.json.
    python3 inc_g2_prepare.py
"""
import collections, json, os, re, sys
R = "/mnt/hum01-rds/Basov/p91688di/"
LCS = R + "llm_corpus_staging/"; PG = R + "phase5_graphbrain/"
P7 = LCS + "inc_run/p7/out/focal_sentences_v2.mapped.jsonl"
G2 = PG + "g2_v3/g2_parsed_v3.jsonl"
OUT = LCS + "inc_run/g2_in/"
SURF = LCS + "inc_run/focal_surfaces_v3_hyphenated.txt"


def v3_seqs():
    return [tuple(re.split(r"[\s\-]+", w.strip().lower())) for w in open(SURF, encoding="utf-8") if w.strip()]


def has_v3_part(word, seqs):
    parts = word.lower().split("-")
    return any(tuple(parts[i:i + len(q)]) == q for q in seqs for i in range(len(parts) - len(q) + 1))


if __name__ == "__main__":
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    if arg("--p7"):                                   # RL-169: second round after the statement check
        P7 = LCS + f"inc_run/{arg('--p7')}/out/focal_sentences_v2.mapped.jsonl"
        OUT = LCS + f"inc_run/{arg('--out')}/"
    done_inc = {}
    for d in (arg("--parsed") or "").split(","):
        if d:
            import glob
            for p in glob.glob(PG + d + "/shards/g2_parsed_*.jsonl"):
                for line in open(p, encoding="utf-8"):
                    for s in json.loads(line)["sentences"]:
                        done_inc[s["sid"]] = s["hash_final"]
    os.makedirs(OUT, exist_ok=True)
    assert not os.path.exists(OUT + "g2_prepare_summary.json"), "outputs exist"
    seqs = v3_seqs()
    first = {q[0] for q in seqs}
    old = {}                                                         # sid -> (hash_final, reason to re-parse or None)
    for line in open(G2, encoding="utf-8"):
        r = json.loads(line)
        for s in r["sentences"]:
            why = None
            for u in s.get("units", []):
                if any(w in ("’", "‘") and "/" in a and a.split("/")[1].startswith("P") for a, w, i in u.get("atom2word", [])):
                    why = "apostrophe"; break
            if not why:
                for w in s.get("transforms", {}).get("hyphens_joined", []):
                    if set(w.lower().split("-")) & first and has_v3_part(w, seqs):
                        why = "hyphen_v3"; break
            old[s["sid"]] = (s["hash_final"], why)
    c = collections.Counter(); arts = 0
    with open(OUT + "focal_sentences_inc.jsonl", "w", encoding="utf-8") as fo, open(OUT + "reparse_reasons.tsv", "w") as fr:
        fr.write("pmcid\tsid\treason\n")
        for line in open(P7, encoding="utf-8"):
            r = json.loads(line); keep = []
            for s in r["sentences"]:
                o = old.get(s["sid"])
                why = "new" if (o is None or o[0] != s["hash_final"]) else o[1]
                if why and done_inc.get(s["sid"]) == s["hash_final"]:
                    why = None                                # parsed in an earlier incremental round
                c[why or "keep existing parse"] += 1
                if why:
                    keep.append(s); fr.write(f"{r['pmcid']}\t{s['sid']}\t{why}\n")
            if keep:
                fo.write(json.dumps(dict(r, sentences=keep, n_sentences=len(keep)), ensure_ascii=False) + "\n"); arts += 1
    summ = dict(sentences=dict(c), articles_with_a_parse=arts)
    json.dump(summ, open(OUT + "g2_prepare_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
