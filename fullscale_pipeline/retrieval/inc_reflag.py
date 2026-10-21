"""Incremental run, step 2 correction (2026-10-21, RL-169): restore the research sentences that the broad statement rule had
un-flagged inside kept articles. Input: stmt_check/final_verdicts.json (Haiku readers + assistant; 'statement' = about how
this paper was produced with AI tools; 'research' = everything else). Outputs in LCS/inc_run/:
  drop_sids_current_v2.tsv   the current-corpus drop list without the restored sentences (only genuine statements and
                             sentences left without a term are dropped)
  reflag_terms.json          sid -> focal terms for restored sentences inside the new P6 records (new articles and current
                             articles rerun from P6); terms pass the same hit filters as in inc_prepare.py (release date,
                             homonym rules, F2 verdicts)
    python3 inc_reflag.py
"""
import csv, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_stage_b as SB
import inc_prepare as IP

L = SB.L; INC = L + "inc_run/"
V = json.load(open(INC + "stmt_check/final_verdicts.json"))["verdicts"]
restore = {sid for sid, v in V.items() if v == "research"}

if __name__ == "__main__":
    assert not os.path.exists(INC + "reflag_terms.json"), "outputs exist"
    c = collections.Counter()
    with open(INC + "drop_sids_current_v2.tsv", "w") as fo:
        fo.write("pmcid\tsid\twhy\n")
        for r in csv.DictReader(open(INC + "drop_sids_current.tsv"), delimiter="\t"):
            if r["why"] == "statement" and r["sid"] in restore:
                c["current drop list: restored"] += 1; continue
            fo.write(f"{r['pmcid']}\t{r['sid']}\t{r['why']}\n"); c[f"current drop list: kept ({r['why']})"] += 1
    IP.init()
    reflag = {}
    new_p6 = {}
    for unit, p in (("gap", INC + "p6_new/coref_status_v2.jsonl"), ("current", INC + "p6_cur/coref_status_v2.jsonl")):
        for line in open(p, encoding="utf-8"):
            new_p6[(unit, json.loads(line)["pmcid"])] = True
    for corpus in ("gap", "current"):
        for line in open(IP.SRC[corpus], encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if (corpus, pm) not in new_p6:
                continue
            r = json.loads(line); year = IP._Y[corpus].get(pm)
            text = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in IP._FILES[corpus].get(pm, []))
            senses = SB.abbr_senses(text)
            for b in r["blocks"]:
                for s in b["sentences"]:
                    if s["sid"] not in restore or not s["focal"]:
                        continue
                    keep = []
                    for t in s["terms"]:
                        ry = SB.release_year(t)
                        if (year and ry and year < ry) or SB.homonym_reason(t, s["text"], senses) or \
                                IP._F2.get((corpus, pm, IP.term_label(t))) in ("no", "statement"):
                            continue
                        keep.append(t)
                    if keep:
                        reflag[s["sid"]] = keep; c[f"{corpus}: restored in new P6 record"] += 1
                    else:
                        c[f"{corpus}: restored sentence without a term left"] += 1
    json.dump(reflag, open(INC + "reflag_terms.json", "w"))
    print(json.dumps(dict(c), indent=1))
