"""Incremental run, steps 5c and 6 (2026-10-21, RL-169): merged G2 output and the check report.

5c  For every article of the incremental P7 output (inc_run/p7/out/focal_sentences_v2.mapped.jsonl), in P7's sentence order,
    each sentence takes its new parse (PG/g2_v3_inc/shards/) if there is one, otherwise its existing parse
    (PG/g2_v3/g2_parsed_v3.jsonl); the parse's hash_final must equal P7's. Written as a fresh shard set
    PG/g2_v3_1/shards/g2_parsed_NNN.jsonl (50 shards, article i to shard i % 50, the layout G3 reads), every article once;
    url tables joined (a token must map to one address). P7's mapped sentences and works are copied beside it as the
    inputs of record (PG/g2_v3_1/focal_sentences_v2.jsonl, citation_works_v2.jsonl).
6   Checks (PG/g2_v3_1/check_report.json; the merge refuses to finish with a failed check): sid, uid and REF id unique;
    every sid starts with its article's PMCID and every uid with its sid; every unit hash and hash_final recomputes from id
    and text; every REF token in a text_final is in the works file and every work id is unique; no sentence of an excluded
    article and no dropped sentence; every existing (sid, hash_final) and unit id unchanged or listed with its reason;
    counts per corpus, class and year.
    python3 inc_g2_merge_check.py
"""
import collections, csv, glob, hashlib, json, os, re, sys
R = "/mnt/hum01-rds/Basov/p91688di/"
LCS, PG = R + "llm_corpus_staging/", R + "phase5_graphbrain/"
INC = LCS + "inc_run/"
P7S, P7W = INC + "p7/out/focal_sentences_v2.mapped.jsonl", INC + "p7/out/citation_works_v2.mapped.jsonl"
OLD = PG + "g2_v3/g2_parsed_v3.jsonl"
NEWD = PG + "g2_v3_inc/shards/"
OUT = PG + "g2_v3_1/"
NSH = 50
REF = re.compile(r"REF\d{6}")


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def main():
    global P7S, P7W, NEWD
    arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    if arg("--p7"):                                   # RL-169 second round
        P7S = INC + f"{arg('--p7')}/out/focal_sentences_v2.mapped.jsonl"; P7W = INC + f"{arg('--p7')}/out/citation_works_v2.mapped.jsonl"
    newdirs = [PG + d + "/shards/" for d in (arg("--newdirs") or "g2_v3_inc").split(",")]   # first wins
    drop_file = INC + (arg("--drop") or "drop_sids_current.tsv")
    reason_files = [INC + d + "/reparse_reasons.tsv" for d in (arg("--reasons") or "g2_in").split(",")]
    assert not os.path.exists(OUT + "check_report.json"), "outputs exist"
    os.makedirs(OUT + "shards", exist_ok=True)
    fail, c = collections.Counter(), collections.Counter()
    # new parses
    new = {}
    for nd in newdirs:
        seen_here = set()
        for p in sorted(glob.glob(nd + "g2_parsed_*.jsonl")):
            for line in open(p, encoding="utf-8"):
                for s in json.loads(line)["sentences"]:
                    if s["sid"] in seen_here:
                        fail["sentence parsed twice in one new run"] += 1
                    seen_here.add(s["sid"])
                    new.setdefault(s["sid"], s)
    reasons = {}
    for rf in reason_files:
        for r in csv.DictReader(open(rf), delimiter="\t"):
            reasons.setdefault(r["sid"], r["reason"])
    need = set(reasons)
    fail["sentence to parse missing from the new run"] = len(need - set(new))
    # existing parses: only what is needed
    p7_sids = set()
    for line in open(P7S, encoding="utf-8"):
        p7_sids.update(s["sid"] for s in json.loads(line)["sentences"])
    old = {}
    for line in open(OLD, encoding="utf-8"):
        for s in json.loads(line)["sentences"]:
            if s["sid"] in p7_sids and s["sid"] not in new:
                old[s["sid"]] = s
    # inputs of record
    excl = {r["pmcid"] for r in csv.DictReader(open(LCS + "article_exclusions_stage_b.csv"))}
    drop = {r["sid"] for r in csv.DictReader(open(drop_file), delimiter="\t")}
    vf = INC + "stmt_check/final_verdicts.json"          # RL-169: genuine AI-use statements are removed whatever their P7 source
    stmts = {k for k, v in json.load(open(vf))["verdicts"].items() if v == "statement"} if os.path.exists(vf) and arg("--p7") else set()
    cls = {r["pmcid"]: r["class"] for r in csv.DictReader(open(INC + "classification_current.tsv"), delimiter="\t")}
    new_arts = set(open(LCS + "gap_stage/gap_article_list_after_ab.txt").read().split())
    works, wid = {}, collections.Counter()
    with open(OUT + "citation_works_v2.jsonl", "w", encoding="utf-8") as fw:
        for line in open(P7W, encoding="utf-8"):
            w = json.loads(line); works[w["ref"]] = w; wid[w["ref"]] += 1; fw.write(line)
    fail["REF id used by two works"] = sum(1 for v in wid.values() if v > 1)
    # merge + checks
    sids, uids, years, ex_kept = set(), set(), collections.Counter(), []
    shards = [open(OUT + f"shards/g2_parsed_{i:03d}.jsonl", "w", encoding="utf-8") for i in range(NSH)]
    fs = open(OUT + "focal_sentences_v2.jsonl", "w", encoding="utf-8")
    for k, line in enumerate(open(P7S, encoding="utf-8")):
        r = json.loads(line); pm = r["pmcid"]
        if stmts:
            kept = [x for x in r["sentences"] if x["sid"] not in stmts]
            c["AI-use statements removed after P7"] += len(r["sentences"]) - len(kept)
            r = dict(r, sentences=kept, n_sentences=len(kept)); line = json.dumps(r, ensure_ascii=False) + "\n"
        if not r["sentences"]:
            c["articles left without a sentence"] += 1; continue
        fs.write(line)
        corpus = "new" if pm in new_arts else "current"
        if pm in excl:
            fail["excluded article present"] += 1
        sents = []
        for s in r["sentences"]:
            sid = s["sid"]
            if sid in sids:
                fail["duplicate sid"] += 1
            sids.add(sid)
            if not sid.startswith(pm + "."):
                fail["sid not of its article"] += 1
            if sid in drop:
                if s.get("source") == "coref" and s.get("terms"):
                    c["dropped sentence kept through coreference with a focal term (listed)"] += 1
                    ex_kept.append(sid)
                else:
                    fail["dropped sentence present"] += 1
            if sha(sid + "|" + s["text_final"]) != s["hash_final"]:
                fail["hash_final does not recompute"] += 1
            for t in REF.findall(s["text_final"]):
                if t not in works:
                    fail["REF token without a work"] += 1
            g = new.get(sid) or old.get(sid)
            src = "new parse" if sid in new else "existing parse"
            if g is None:
                fail["sentence without a parse"] += 1; continue
            if g["hash_final"] != s["hash_final"]:
                fail["parse hash_final differs from P7"] += 1
            for u in g.get("units", []):
                if u["uid"] in uids:
                    fail["duplicate uid"] += 1
                uids.add(u["uid"])
                if not u["uid"].startswith(sid + ".U"):
                    fail["uid not of its sentence"] += 1
                if sha(u["uid"] + "|" + u["text"]) != u["hash"]:
                    fail["unit hash does not recompute"] += 1
            c[f"{corpus}: sentences from {src}"] += 1
            sents.append(g)
        c[f"{corpus}: articles ({cls.get(pm, 'new') if corpus == 'current' else 'new'})"] += 1
        shards[k % NSH].write(json.dumps(dict(pmcid=pm, sentences=sents), ensure_ascii=False) + "\n")
    for f in shards:
        f.close()
    fs.close()
    # url tables
    urls, conflict = {}, 0
    for p in [PG + "g2_v3/url_table_v3.json"] + [q for nd in newdirs for q in sorted(glob.glob(nd + "url_table_*.json"))]:
        for t, a in json.load(open(p)).items():
            if t in urls and urls[t] != a:
                conflict += 1
            urls[t] = a
    fail["URL token with two addresses"] = conflict
    json.dump(urls, open(OUT + "url_table_v3_1.json", "w"))
    # existing ids: unchanged or listed
    ex = collections.Counter()
    for line in open(OLD, encoding="utf-8"):
        for s in json.loads(line)["sentences"]:
            sid = s["sid"]
            if sid not in p7_sids:
                ex["existing sentence not carried (excluded article, dropped, or no longer kept)"] += 1
            elif sid in new:
                ex[f"existing sentence re-parsed: {reasons.get(sid, '?')}"] += 1
            else:
                ex["existing sentence and units unchanged"] += 1
    fail = {k: v for k, v in fail.items() if v}
    rep = dict(kept_through_coreference=ex_kept, failed_checks=fail, counts=dict(sorted(c.items())), existing=dict(ex), sentences=len(sids), units=len(uids),
               works=len(works), url_tokens=len(urls))
    json.dump(rep, open(OUT + "check_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if fail:
        sys.exit("CHECKS FAILED")


if __name__ == "__main__":
    main()
