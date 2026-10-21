"""Manual check of sentence-level AI-use statements inside kept articles (2026-10-21, RL-169; owner: restore, calibrate
Haiku, then check all). The broad statement rule (gap_stage_b.is_statement + companion rule) un-flagged ~6,000 sentences
inside kept articles in inc_prepare.py; inside genuine LLM papers it was ~10% precise (4 of 40). Every such sentence is
restored as a candidate and judged by a reader: statement (about how this paper itself was written, edited, translated,
illustrated or coded, or a declaration that AI was / was not used for it) or research (anything else).
    python3 stmt_check.py build        -> LCS/inc_run/stmt_check/: candidates.tsv, calib.txt/.json (gold sentences),
                                          batch_NN_part1/2.txt + batch_NN.json (the rest, gold excluded), RUBRIC.md
"""
import csv, json, os, random, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import gap_stage_b as SB
import build_content_regions as P0

L = SB.L
INC = L + "inc_run/"
OUT = INC + "stmt_check/"
RAW = {"gap": L + "pure_text_corpus_gap/", "current": L + "pure_text_corpus/"}
SRC = {"gap": SB.GR + "llm_corpus_staging/focal_extractions_v2.jsonl", "current": L + "p2_v3_current/focal_extractions_v2.jsonl"}
CLEANED = {"gap": INC + "p2_new.jsonl", "current": INC + "p2_current_additions.jsonl"}
BATCH = 500


def statement_sids(rec):
    sents = [s for b in rec["blocks"] for s in b["sentences"] if s["focal"]]
    lab = {s["sid"]: SB.is_statement(s["text"]) for s in sents}
    if any(lab.values()):
        lab = {s["sid"]: lab[s["sid"]] or bool(SB.NEGATED_LOOSE.search(s["text"])) for s in sents}
    return {k for k, v in lab.items() if v}


def candidates():
    """(corpus, pmcid, sid): broad-rule statements in kept articles"""
    out = []
    # current: the drop list says which sentences were un-flagged as statements
    for r in csv.DictReader(open(INC + "drop_sids_current.tsv"), delimiter="\t"):
        if r["why"] == "statement":
            out.append(("current", r["pmcid"], r["sid"]))
    # current articles rerun from P6 and new articles: statement sentences of the raw v3 record that the cleaned record un-flagged
    cur_add = {line[11:line.index('"', 11)] for line in open(CLEANED["current"], encoding="utf-8")}
    cleaned_focal = {}
    for c in ("gap", "current"):
        for line in open(CLEANED[c], encoding="utf-8"):
            r = json.loads(line)
            cleaned_focal[(c, r["pmcid"])] = {s["sid"] for b in r["blocks"] for s in b["sentences"] if s["focal"]}
    have = {x[2] for x in out}
    for c in ("gap", "current"):
        for line in open(SRC[c], encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if (c, pm) not in cleaned_focal or (c == "current" and pm not in cur_add):
                continue
            r = json.loads(line)
            for sid in statement_sids(r):
                if sid not in cleaned_focal[(c, pm)] and sid not in have:
                    out.append((c, pm, sid)); have.add(sid)
    return out


def contexts(cands):
    S = json.load(open(L + "gap_esummary_cache.json")); TM = json.load(open(L + "target_metadata.json"))
    want = collections.defaultdict(set)
    for c, pm, sid in cands:
        want[(c, pm)].add(sid)
    ctx = {}
    for c in ("gap", "current"):
        for line in open(SRC[c], encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if (c, pm) not in want:
                continue
            r = json.loads(line)
            rl = open(RAW[c] + pm + ".txt", encoding="utf-8", errors="replace").read().split("\n")
            heads = [(i, rl[i].strip()) for i in range(len(rl)) if P0.is_head(rl, i)]
            title = ((S.get(pm) or {}) if c == "gap" else (TM.get(pm) or {})).get("title") or ""
            for b in r["blocks"]:
                ss = b["sentences"]
                for k, s in enumerate(ss):
                    if s["sid"] in want[(c, pm)]:
                        h = next((x[1] for x in reversed(heads) if x[0] < s["raw_line"] - 1), "")
                        before = ss[k - 1]["text"] if k > 0 else ""
                        after = ss[k + 1]["text"] if k + 1 < len(ss) else ""
                        ctx[s["sid"]] = dict(corpus=c, pmcid=pm, title=title[:200], heading=h[:120], before=before[:400],
                                             text=s["text"], after=after[:400])
    return ctx


def render(items, path, title):
    lines = [f"# {title}: {len(items)} sentences", ""]
    for qid, x in items:
        lines += [f"## {qid} | section: {x['heading'] or 'unknown'}", f"Article title: {x['title']}"]
        if x["before"]:
            lines.append(f"Before: {x['before']}")
        lines.append(f"SENTENCE ⟦{x['text']}⟧")
        if x["after"]:
            lines.append(f"After: {x['after']}")
        lines.append(f"QUESTION {qid}: is the marked sentence an AI-use statement about this paper itself, or research content?")
        lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def build():
    os.makedirs(OUT, exist_ok=True)
    cands = candidates()
    ctx = contexts(cands)
    with open(OUT + "candidates.tsv", "w") as f:
        f.write("corpus\tpmcid\tsid\n"); [f.write(f"{c}\t{p}\t{s}\n") for c, p, s in cands]
    print("candidates", len(cands), collections.Counter(c for c, _, _ in cands), "with context", len(ctx))
    # gold: the assistant's judged samples that can be reproduced exactly
    import statement_strict as ST
    s1 = json.load(open(INC + "statement_check_sample.json"))
    gold = {sid: ("statement" if i in {0, 1, 16, 18} else "research") for i, (sid, t) in enumerate(s1)}
    drop = {r["sid"] for r in csv.DictReader(open(INC + "drop_sids_current.tsv"), delimiter="\t") if r["why"] == "statement"}
    K = []
    for line in open(L + "focal_extractions_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for b in r["blocks"]:
            for s in b["sentences"]:
                if s["sid"] in drop and ST.is_statement_strict(s["text"]):
                    K.append((s["sid"], s["text"]))
    random.seed(57)
    for i, (sid, t) in enumerate(random.sample(K, 30)):
        gold.setdefault(sid, "research" if i in {0, 3, 7, 14, 20} else "statement")
    gold = {k: v for k, v in gold.items() if k in ctx}
    json.dump(dict(judge="assistant (Opus 5.5), 2026-10-21", labels=gold), open(OUT + "calib_gold.json", "w"), indent=1)
    gids = sorted(gold); random.seed(3); random.shuffle(gids)
    calib = [(f"C{i:03d}", ctx[s]) for i, s in enumerate(gids, 1)]
    render(calib, OUT + "calib.txt", "Calibration")
    json.dump({q: dict(sid=s, gold=gold[s]) for (q, _), s in zip(calib, gids)}, open(OUT + "calib.json", "w"), indent=1)
    rest = [s for s in sorted(ctx) if s not in gold]
    random.seed(11); random.shuffle(rest)
    nb = -(-len(rest) // BATCH)
    for b in range(nb):
        part = rest[b * BATCH:(b + 1) * BATCH]
        items = [(f"B{b + 1:02d}.{i:03d}", ctx[s]) for i, s in enumerate(part, 1)]
        h = len(items) // 2
        render(items[:h], OUT + f"batch_{b + 1:02d}_part1.txt", f"Batch {b + 1:02d}, part 1")
        render(items[h:], OUT + f"batch_{b + 1:02d}_part2.txt", f"Batch {b + 1:02d}, part 2")
        json.dump({q: dict(sid=s, corpus=ctx[s]["corpus"], pmcid=ctx[s]["pmcid"]) for (q, _), s in zip(items, part)},
                  open(OUT + f"batch_{b + 1:02d}.json", "w"), indent=1)
    print("gold", len(gold), collections.Counter(gold.values()), "batches", nb, "of up to", BATCH)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
