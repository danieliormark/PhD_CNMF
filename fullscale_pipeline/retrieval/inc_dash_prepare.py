"""RL-185 (owner 2026-10-23, B2): G2 re-parse of the sentences that hold an em dash or a spaced en dash, so that an aside is
parsed as an aside (written in brackets) and two words glued by an em dash are parsed apart (ai_particularly, sam_is).
Selects, from the G2 v3.1 input of record (PG/g2_v3_1/focal_sentences_v2.jsonl), every sentence whose text_final holds
"—" or " – ", except sentences of excluded articles (G3 scope list) and sentences with a unit on a G3 list keyed by unit
hash (reflexive table, LLM-output units, spelling corrections: a new parse would change the hash); writes the P7 records
holding only those sentences to LCS/inc_run/g2_in3/focal_sentences_dash.jsonl, with a summary.
    python3 inc_dash_prepare.py
"""
import collections, csv, json, os

R = "/mnt/hum01-rds/Basov/p91688di/"
PG, LCS = R + "phase5_graphbrain/", R + "llm_corpus_staging/"
INP = PG + "g3_inputs_v3_1/"
OUT = LCS + "inc_run/g2_in3/"


def main():
    os.makedirs(OUT, exist_ok=True)
    excl = {r["pmcid"] for r in csv.DictReader(open(INP + "scope_exclusions.csv", encoding="utf-8"))}
    listed = {json.loads(l)["uid"].rsplit(".U", 1)[0] for l in open(INP + "reflexive_restored.jsonl", encoding="utf-8")}
    for f in ("llm_output_units.csv", "spelling_corrections.csv"):
        listed |= {r["uid"].rsplit(".U", 1)[0] for r in csv.DictReader(open(INP + f, encoding="utf-8"))}
    C = collections.Counter()
    with open(OUT + "focal_sentences_dash.jsonl", "w", encoding="utf-8") as out:
        for line in open(PG + "g2_v3_1/focal_sentences_v2.jsonl", encoding="utf-8"):
            r = json.loads(line)
            keep = []
            for s in r["sentences"]:
                t = s["text_final"]
                if "—" not in t and " – " not in t:
                    continue
                C["dash sentences"] += 1
                if r["pmcid"] in excl:
                    C["skipped: excluded article"] += 1
                elif s["sid"] in listed:
                    C["skipped: unit on a hash-keyed G3 list"] += 1
                else:
                    keep.append(s)
            if keep:
                out.write(json.dumps(dict(r, sentences=keep, n_sentences=len(keep)), ensure_ascii=False) + "\n")
                C["articles"] += 1; C["sentences to re-parse"] += len(keep)
    json.dump(dict(C), open(OUT + "dash_prepare_summary.json", "w"), indent=1)
    print(dict(C))


if __name__ == "__main__":
    main()
