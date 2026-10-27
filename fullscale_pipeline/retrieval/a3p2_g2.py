"""A3 part 2, step 5 (2026-10-26, RL-203): G2 for the added sentences and the merged G2 output PG/g2_v3_3/ (+ g2_v3_3_pos/).
    prepare   from the P7 output of a3p2_p7.py (LCS/a3p2/p7/out/focal_sentences_v2.mapped.jsonl) take the added sentences
              (added_sentences.json) as P7 records holding only them -> LCS/a3p2/g2_in/focal_sentences_inc.jsonl. Every other
              sentence of that P7 output must equal the merged corpus (PG/g2_v3_2/, sid and hash_final) or be one of the
              sentences the RL-169 merge removed after P7 (then it is not carried: not in g2_v3_2).
    parse     G2 v3 through inc_g2_dash.py (the wrapper of every existing parse: v3 hyphenated names, curly apostrophes, the
              RL-185 dash rewrite), 8 shards in parallel -> PG/g2_v3_a3p2/shards/
    positions g2_positions.py's run_shard (RL-186, unchanged) on those shards -> PG/g2_v3_a3p2/pos/
    merge     PG/g2_v3_3/: every article of g2_v3_2 in its own shard file and order; an article with additions gets its new
              sentences inserted in sentence-id order (region, raw line, k); g2_v3_3_pos/pos_NNN.jsonl = g2_v3_2_pos's plus
              the new units. Inputs of record: focal_sentences_v2.jsonl (g2_v3_2's records plus the added sentences),
              citation_works_v2.jsonl (the P7 works with frozen ids), url_table_v3_3.json (g2_v3_2's joined with the new
              shard tables; a token must keep its address).
              Checks (the merge refuses to finish with a failed one): every article once; every sid once and in its own
              article; every uid starts with its sid and is unique; every unit hash recomputes; no added sid was in g2_v3_2;
              every added sentence parsed (status ok or nonprose) and present; every REF token of an added text_final is in
              the works file; every new unit has a positions row. Report PG/g2_v3_3/check_report.json.
    python3 a3p2_g2.py prepare | parse | positions | merge
"""
import collections, glob, hashlib, importlib.util, json, os, re, subprocess, sys

R = "/mnt/hum01-rds/Basov/p91688di/"
LCS, PG = R + "llm_corpus_staging/", R + "phase5_graphbrain/"
A = LCS + "a3p2/"
P7 = A + "p7/out/"
GIN = A + "g2_in/"
NEW = PG + "g2_v3_a3p2/"
OLD, OLDPOS, OUT, OUTPOS = PG + "g2_v3_2/", PG + "g2_v3_2_pos/", PG + "g2_v3_3/", PG + "g2_v3_3_pos/"
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
NSH = 8


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def sid_key(sid):
    m = re.match(r"PMC\d+\.r(\d+)\.L(\d+)\.S(\d+)$", sid)
    return tuple(int(x) for x in m.groups())


def prepare():
    added = {s for v in json.load(open(P7 + "added_sentences.json")).values() for s in v}
    old = {}
    for line in open(OLD + "focal_sentences_v2.jsonl", encoding="utf-8"):
        for s in json.loads(line)["sentences"]:
            old[s["sid"]] = s["hash_final"]
    C = collections.Counter(); os.makedirs(GIN, exist_ok=True)
    with open(GIN + "focal_sentences_inc.jsonl", "w", encoding="utf-8") as fo:
        for line in open(P7 + "focal_sentences_v2.mapped.jsonl", encoding="utf-8"):
            r = json.loads(line); keep = []
            for s in r["sentences"]:
                if s["sid"] in added:
                    C["added: to parse"] += 1; keep.append(s)
                    if s["sid"] in old: C["ADDED SID ALREADY IN g2_v3_2 (must be 0)"] += 1
                elif s["sid"] not in old:
                    C["not added, not in g2_v3_2 (removed by the RL-169 merge, not carried)"] += 1
                elif old[s["sid"]] == s["hash_final"]:
                    C["existing, unchanged"] += 1
                else:
                    C["EXISTING CHANGED (must be 0)"] += 1
            if keep:
                fo.write(json.dumps(dict(r, sentences=keep, n_sentences=len(keep)), ensure_ascii=False) + "\n")
    C["added sentences not found in P7 output (must be 0)"] = len(added) - C["added: to parse"]
    json.dump(dict(C), open(GIN + "g2_prepare_summary.json", "w"), indent=1)
    print(json.dumps(dict(C), indent=1))


def parse():
    os.makedirs(NEW + "shards", exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.expanduser("~/np1_for_spacy"), OMP_NUM_THREADS="2")
    ps = [subprocess.Popen([sys.executable, "-u", HERE + "inc_g2_dash.py", "--shard", str(k), "--nshards", str(NSH),
                            "--input", GIN + "focal_sentences_inc.jsonl", "--outdir", NEW + "shards/"],
                           stdout=open(NEW + f"parse_{k}.log", "w"), stderr=subprocess.STDOUT, env=env) for k in range(NSH)]
    print("exit codes", [p.wait() for p in ps])


def positions():
    spec = importlib.util.spec_from_file_location("g2_positions", HERE + "g2_positions.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    g.INP, g.OUT = NEW + "shards/", NEW + "pos/"
    for k in range(NSH):
        g.run_shard(k)


def merge():
    if os.path.exists(OUT + "shards"):
        sys.exit(f"refusing to overwrite {OUT}shards")
    os.makedirs(OUT + "shards"); os.makedirs(OUTPOS)
    new_by_art = collections.defaultdict(list); new_status = {}
    for f in sorted(glob.glob(NEW + "shards/g2_parsed_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for s in r["sentences"]:
                new_by_art[r["pmcid"]].append(s); new_status[s["sid"]] = s["status"]
    newpos = {}
    for f in glob.glob(NEW + "pos/pos_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            p = json.loads(line); newpos[p["uid"]] = line
    C = collections.Counter(); problems = []; seen_art, seen_sid, seen_uid = set(), set(), set()
    works = {}
    for line in open(P7 + "citation_works_v2.mapped.jsonl", encoding="utf-8"):
        w = json.loads(line); works[w["ref"]] = w
    for f in sorted(glob.glob(OLD + "shards/g2_parsed_*.jsonl")):
        tag = re.search(r"_(\d+)\.jsonl$", f).group(1)
        pos_out = open(OUTPOS + f"pos_{tag}.jsonl", "w", encoding="utf-8")
        for line in open(OLDPOS + f"pos_{tag}.jsonl", encoding="utf-8"):
            pos_out.write(line)
        with open(OUT + "shards/" + os.path.basename(f), "w", encoding="utf-8") as out:
            for line in open(f, encoding="utf-8"):
                r = json.loads(line); C["articles"] += 1
                if r["pmcid"] in seen_art:
                    problems.append(f"article twice: {r['pmcid']}")
                seen_art.add(r["pmcid"])
                sents = r["sentences"]
                add = new_by_art.pop(r["pmcid"], [])
                if add:
                    C["articles with added sentences"] += 1
                    olds = {s["sid"] for s in sents}
                    for s in add:
                        if s["sid"] in olds: problems.append(f"added sid already in g2_v3_2: {s['sid']}")
                        if s["status"] not in ("ok", "nonprose"): problems.append(f"added sentence not parsed: {s['sid']} {s['status']}")
                        for tok in re.findall(r"REF\d{6}", s.get("text_final", "")):
                            if tok not in works: problems.append(f"REF token without work: {tok} in {s['sid']}")
                        for u in s.get("units", []):
                            if u["uid"] in newpos:
                                pos_out.write(newpos[u["uid"]]); C["new units with positions"] += 1
                            else:
                                problems.append(f"new unit without positions: {u['uid']}")
                        C["sentences added"] += 1; C["units added"] += len(s.get("units", []))
                    sents = sorted(sents + add, key=lambda s: sid_key(s["sid"]))
                for s in sents:
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
                out.write(json.dumps(dict(r, sentences=sents), ensure_ascii=False) + "\n")
        pos_out.close()
    if new_by_art:
        problems.append(f"{len(new_by_art)} articles with added sentences not in g2_v3_2 (e.g. {list(new_by_art)[:3]})")
    table = json.load(open(OLD + "url_table_v3_2.json"))
    for f in glob.glob(NEW + "shards/url_table_*.json"):
        for tok, addr in json.load(open(f)).items():
            if tok in table and table[tok] != addr:
                problems.append(f"URL token {tok} with two addresses")
            table.setdefault(tok, addr)
    json.dump(table, open(OUT + "url_table_v3_3.json", "w"), indent=0)
    # inputs of record: g2_v3_2's focal sentence records plus the added sentences (P7 records), the P7 works
    addrec = collections.defaultdict(list)
    for line in open(GIN + "focal_sentences_inc.jsonl", encoding="utf-8"):
        r = json.loads(line); addrec[r["pmcid"]] = r["sentences"]
    with open(OUT + "focal_sentences_v2.jsonl", "w", encoding="utf-8") as fo:
        for line in open(OLD + "focal_sentences_v2.jsonl", encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if pm in addrec:
                r = json.loads(line); ss = sorted(r["sentences"] + addrec[pm], key=lambda s: sid_key(s["sid"]))
                line = json.dumps(dict(r, sentences=ss, n_sentences=len(ss)), ensure_ascii=False) + "\n"
            fo.write(line)
    os.symlink(P7 + "citation_works_v2.mapped.jsonl", OUT + "citation_works_v2.jsonl")
    rep = dict(counts=dict(C), problems=problems[:200], n_problems=len(problems))
    json.dump(rep, open(OUT + "check_report.json", "w"), indent=1)
    print(json.dumps(dict(C)), "problems:", len(problems))
    if problems:
        sys.exit("MERGE CHECKS FAILED (see check_report.json)")


if __name__ == "__main__":
    dict(prepare=prepare, parse=parse, positions=positions, merge=merge)[sys.argv[1]]()
