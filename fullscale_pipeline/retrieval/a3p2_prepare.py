"""A3 part 2, step 2 (2026-10-26, RL-203): which sentences does P2 with focal_terms_v3_1 (v3 + G3's hub spellings) add?
New P2 records (LCS/a3p2/p2_current/, p2_gap/) are compared by sentence id with the P2 v3 records the incremental run used
(LCS/p2_v3_current/ and the gap staging P2), for the articles of the merged corpus (PG/g2_v3_2/focal_sentences_v2.jsonl).
    addition   focal under v3.1, not focal under v3 (its terms are listed; hub terms are "hub:<spelling>")
    changed    focal under both, other terms (a hub term added to a sentence that already had a focal term: no new sentence,
               G3 already reads such hub spellings itself, RL-185)
    lost       focal under v3, not under v3.1 (must not happen: v3.1 only adds hits)
Outputs in LCS/a3p2/: additions.jsonl (pmcid, sid, terms, text, kinds of terms), p6_in_current/ and p6_in_gap/
focal_extractions_v2.jsonl (the new P2 records of the articles with an addition, for P6), prepare_summary.json.
Additions are read by the assistant before P6 (AI-use statements and false hits are un-flagged with --reject FILE, a TSV
sid<TAB>why; owner: statements are not analysed).
    python3 a3p2_prepare.py [--reject rejected.tsv]
"""
import collections, csv, json, os, sys

R = "/mnt/hum01-rds/Basov/p91688di/"
LCS, PG = R + "llm_corpus_staging/", R + "phase5_graphbrain/"
A = LCS + "a3p2/"
OLD = {"current": LCS + "p2_v3_current/focal_extractions_v2.jsonl", "gap": LCS + "gap_stage/llm_corpus_staging/focal_extractions_v2.jsonl"}
NEW = {"current": A + "p2_current/focal_extractions_v2.jsonl", "gap": A + "p2_gap/focal_extractions_v2.jsonl"}
FINAL = PG + "g2_v3_2/focal_sentences_v2.jsonl"


def focal(path, keep):
    out = {}
    for line in open(path, encoding="utf-8"):
        pm = line[11:line.index('"', 11)]
        if pm not in keep:
            continue
        r = json.loads(line)
        out[pm] = (r, {s["sid"]: s for b in r["blocks"] for s in b["sentences"]})
    return out


def main():
    rej = {}
    if "--reject" in sys.argv:
        rej = {r["sid"]: r["why"] for r in csv.DictReader(open(sys.argv[sys.argv.index("--reject") + 1]), delimiter="\t")}
    corpus = set()
    for line in open(FINAL, encoding="utf-8"):
        corpus.add(line[11:line.index('"', 11)])
    S = collections.Counter(); adds = []
    for unit in ("current", "gap"):
        old, new = focal(OLD[unit], corpus), focal(NEW[unit], corpus)
        S[f"{unit}: articles in corpus with an old record"] = len(old); S[f"{unit}: with a new record"] = len(new)
        touched = []
        for pm, (r, ns) in new.items():
            os_ = old.get(pm, (None, {}))[1]
            hit = False
            for sid, s in ns.items():
                o = os_.get(sid)
                if s["focal"] and not (o and o["focal"]):
                    kinds = sorted({t.split(":")[0] if ":" in t else "v2" for t in s["terms"]})
                    adds.append(dict(unit=unit, pmcid=pm, sid=sid, terms=s["terms"], kinds=kinds, text=s["text"],
                                     rejected=rej.get(sid, "")))
                    S[f"{unit}: addition"] += 1; S["additions with terms of kind " + "+".join(kinds)] += 1
                    if sid in rej:
                        s["focal"], s["terms"] = False, []; S[f"{unit}: addition rejected ({rej[sid]})"] += 1
                    else:
                        hit = True
                elif s["focal"] and o and o["focal"] and sorted(s["terms"]) != sorted(o["terms"]):
                    S[f"{unit}: focal under both, terms changed"] += 1
            for sid, o in os_.items():
                if o["focal"] and not (ns.get(sid) or {}).get("focal"):
                    S[f"{unit}: LOST (must be 0)"] += 1
            if hit:
                touched.append(r)
        os.makedirs(A + f"p6_in_{unit}", exist_ok=True)
        with open(A + f"p6_in_{unit}/focal_extractions_v2.jsonl", "w", encoding="utf-8") as fo:
            for r in touched:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        S[f"{unit}: articles for P6"] = len(touched)
    with open(A + "additions.jsonl", "w", encoding="utf-8") as fo:
        for d in adds:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    S["articles with an accepted addition"] = len({d["pmcid"] for d in adds if not d["rejected"]})
    json.dump(dict(S), open(A + "prepare_summary.json", "w"), indent=1)
    print(json.dumps(dict(S), indent=1))


if __name__ == "__main__":
    main()
