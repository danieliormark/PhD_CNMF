"""RL-185 (owner 2026-10-23, B2): merged G2 output with the dash re-parses, PG/g2_v3_2/ (the layout G3 reads).
Every article of PG/g2_v3_1/shards/ is copied in order; a sentence re-parsed by inc_g2_dash.py (PG/g2_v3_dash/shards/)
replaces its old record when its status is ok or nonprose and its hash_final equals the old one (otherwise the old parse
stays and the case is reported). The inputs of record (focal_sentences_v2.jsonl, citation_works_v2.jsonl) are linked from
g2_v3_1; the URL table is g2_v3_1's joined with the new shard tables (a token must keep its address).
Checks (the merge refuses to finish with a failed one): every article once, every sid once and in its own article, every uid
starts with its sid and is unique, every unit hash recomputes from uid and text, every replaced record is a re-parse of
that sentence, the replaced count equals the new parses accepted. Report PG/g2_v3_2/check_report.json.
    python3 inc_g2_dash_merge.py
"""
import collections, glob, hashlib, json, os, sys

R = "/mnt/hum01-rds/Basov/p91688di/"
PG = R + "phase5_graphbrain/"
OLD, NEW, OUT = PG + "g2_v3_1/", PG + "g2_v3_dash/", PG + "g2_v3_2/"


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def main():
    if os.path.exists(OUT + "shards"):
        sys.exit(f"refusing to overwrite {OUT}shards")
    os.makedirs(OUT + "shards")
    new = {}
    for f in sorted(glob.glob(NEW + "shards/g2_parsed_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            for s in json.loads(line)["sentences"]:
                new[s["sid"]] = s
    C = collections.Counter(); problems = []; seen_art, seen_sid, seen_uid = set(), set(), set(); kept_old = []
    for f in sorted(glob.glob(OLD + "shards/g2_parsed_*.jsonl")):
        with open(OUT + "shards/" + os.path.basename(f), "w", encoding="utf-8") as out:
            for line in open(f, encoding="utf-8"):
                r = json.loads(line); C["articles"] += 1
                if r["pmcid"] in seen_art:
                    problems.append(f"article twice: {r['pmcid']}")
                seen_art.add(r["pmcid"])
                sents = []
                for s in r["sentences"]:
                    n = new.get(s["sid"])
                    if n is not None:
                        if n["hash_final"] == s["hash_final"] and n["status"] in ("ok", "nonprose"):
                            s = n; C["sentences re-parsed"] += 1
                            if n["transforms"].get("dash_rewritten"):
                                C["  of which text rewritten"] += 1
                        else:
                            kept_old.append([s["sid"], n["status"], n["hash_final"] == s["hash_final"]]); C["re-parse not used (old kept)"] += 1
                    if s["sid"] in seen_sid or not s["sid"].startswith(r["pmcid"] + "."):
                        problems.append(f"sid: {s['sid']}")
                    seen_sid.add(s["sid"]); C["sentences"] += 1
                    for u in s.get("units", []):
                        C["units"] += 1
                        if u["uid"] in seen_uid or not u["uid"].startswith(s["sid"] + ".U"):
                            problems.append(f"uid: {u['uid']}")
                        seen_uid.add(u["uid"])
                        if sha(f"{u['uid']}|{u['text']}") != u["hash"]:
                            problems.append(f"unit hash: {u['uid']}")
                    sents.append(s)
                out.write(json.dumps(dict(r, sentences=sents), ensure_ascii=False) + "\n")
    missing = [sid for sid in new if sid not in seen_sid]
    if missing:
        problems.append(f"{len(missing)} re-parsed sentences not found in g2_v3_1 (e.g. {missing[:3]})")
    table = json.load(open(OLD + "url_table_v3_1.json"))
    for f in glob.glob(NEW + "shards/url_table_*.json"):
        for tok, addr in json.load(open(f)).items():
            if tok in table and table[tok] != addr:
                problems.append(f"URL token {tok} with two addresses")
            table.setdefault(tok, addr)
    json.dump(table, open(OUT + "url_table_v3_2.json", "w"), indent=0)
    for x in ("focal_sentences_v2.jsonl", "citation_works_v2.jsonl"):
        if not os.path.exists(OUT + x):
            os.symlink(OLD + x, OUT + x)
    rep = dict(counts=dict(C), problems=problems[:200], n_problems=len(problems), reparse_not_used=kept_old[:200])
    json.dump(rep, open(OUT + "check_report.json", "w"), indent=1)
    print(json.dumps(dict(C)), "problems:", len(problems))
    if problems:
        sys.exit("MERGE CHECKS FAILED (see check_report.json)")


if __name__ == "__main__":
    main()
