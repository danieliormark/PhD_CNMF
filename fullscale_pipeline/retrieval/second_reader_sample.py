"""Second, independent reading of the three manual checks (owner 2026-10-23: a separate model reader, Sonnet, blind to the
assistant's labels). Draws stratified random samples of the items the Haiku readers answered, copies each item block exactly as
the readers saw it, renumbers it (R001 …) and writes:
  PG/second_reader/<list>_<run>.txt      the items, with the original question line (renumbered)
  PG/second_reader/<list>_RUBRIC.md      a copy of the rubric the Haiku readers used
  PG/second_reader_keys/<list>_<run>.json  {new id: {qid, final verdict, stratum}} — never shown to the reader
Lists: f2 (Stage B word sense, RL-161–163), stmt (AI-use statements, RL-169 and RL-172), g4 (group-4 model names, RL-173/174).
Runs: trial (5 + 5 per list) and main (50 + 50 per list), disjoint; seed fixed.
    python3 second_reader_sample.py
"""
import collections, csv, glob, json, os, random, re, shutil

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT, KEYS = PG + "second_reader/", PG + "second_reader_keys/"
SIZES = {"trial": 5, "main": 50}          # per stratum


def blocks(path):
    """{qid: block text} from a reader batch file ('## <qid> ...' up to the next '## ')"""
    out, cur, buf = {}, None, []
    for line in open(path, encoding="utf-8"):
        if line.startswith("## "):
            if cur:
                out[cur] = "".join(buf).rstrip() + "\n"
            cur, buf = line[3:].split(" | ")[0].strip(), [line]
        elif cur:
            buf.append(line)
    if cur:
        out[cur] = "".join(buf).rstrip() + "\n"
    return out


def f2_items():
    man = LCS + "gap_stage/stage_b/manual/"
    final = {(r["pmcid"], r["term"].lower()): r["verdict"] for r in csv.DictReader(open(man + "f2_decisions_final.tsv"), delimiter="\t")}
    items = {}
    for ids in sorted(glob.glob(man + "f2_b*_ids.tsv")):
        b = os.path.basename(ids)[:-8]
        order = [l.rstrip("\n").split("\t") for l in open(ids)]
        text = {}
        for part in sorted(glob.glob(man + b + "_part*.txt")):
            text.update(blocks(part))
        for aid, blk in text.items():
            pm = order[int(aid[1:]) - 1][1]
            for q in re.finditer(r"QUESTION (A\d+\.\d+): does ⟦(.+?)⟧", blk):
                v = final.get((pm, q.group(2).lower()))
                if v in ("yes", "no", "statement"):
                    head = blk.split("QUESTION")[0]          # the article block, with only this question
                    qline = [l for l in blk.splitlines() if l.startswith(f"QUESTION {q.group(1)}:")][0]
                    items[f"{b}:{q.group(1)}"] = (head + qline + "\n", v)
    return items, {"yes": "focal", "no": "not_focal", "statement": "not_focal"}


def stmt_items():
    items = {}
    for run in ("stmt_check", "stmt_check2"):
        d = LCS + "inc_run/" + run + "/"
        final = json.load(open(d + "final_verdicts.json"))["verdicts"]
        for bj in sorted(glob.glob(d + "batch_[0-9][0-9].json")):
            ids = json.load(open(bj))
            text = {}
            for part in sorted(glob.glob(bj[:-5] + "*.txt")):
                text.update(blocks(part))
            for qid, blk in text.items():
                v = final.get(ids.get(qid, {}).get("sid"))
                if v in ("statement", "research"):
                    items[f"{run}:{qid}"] = (blk, v)
    return items, {"statement": "statement", "research": "research"}


def g4_items():
    act = {}
    for r in csv.DictReader(open(PG + "g3_inputs_v3_1/g4_decisions.csv", encoding="utf-8")):
        if r["pmcid"] == "":
            act[r["spelling"]] = "not_model" if r["action"] == "reject" else "model"
    items = {}
    for bt in sorted(glob.glob(PG + "g3_inputs_v3_1/g4/haiku/batch_*.txt")):
        for qid, blk in blocks(bt).items():
            m = re.search(r"spelling: (.+?) \|", blk.splitlines()[0])
            v = act.get(m.group(1)) if m else None
            if v:
                items[f"{os.path.basename(bt)[:-4]}:{qid}"] = (blk, v)
    return items, {"model": "model", "not_model": "not_model"}


RUBRICS = {"f2": LCS + "gap_stage/stage_b/manual/RUBRIC.md", "stmt": HERE + "/STMT_READER_RUBRIC.md",
           "g4": os.path.join(HERE, "..", "g3_inputs_v3_1", "G4_READER_RUBRIC.md")}


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(KEYS, exist_ok=True)
    rng = random.Random(20261023)
    for name, fn in (("f2", f2_items), ("stmt", stmt_items), ("g4", g4_items)):
        items, stratum_of = fn()
        by = collections.defaultdict(list)
        for k, (blk, v) in sorted(items.items()):
            by[stratum_of[v]].append(k)
        used = set()
        for run, n in SIZES.items():
            pick = []
            for s in sorted(by):
                pool = [k for k in by[s] if k not in used]
                pick += [(k, s) for k in rng.sample(pool, min(n, len(pool)))]
            rng.shuffle(pick)
            used |= {k for k, _ in pick}
            key, txt = {}, [f"# Second reading, list {name}, {run}: {len(pick)} questions\n\n"]
            for i, (k, s) in enumerate(pick, 1):
                rid = f"R{i:03d}"
                blk, v = items[k]
                old = k.split(":")[1]
                blk = re.sub(r"^## \S+", "## " + rid, blk)
                blk = blk.replace(f"QUESTION {old}:", f"QUESTION {rid}:")
                txt.append(blk + "\n")
                key[rid] = dict(item=k, final=v, stratum=s)
            open(f"{OUT}{name}_{run}.txt", "w", encoding="utf-8").write("".join(txt))
            json.dump(key, open(f"{KEYS}{name}_{run}.json", "w"), indent=1)
            print(name, run, len(pick), collections.Counter(s for _, s in pick), "of", {s: len(v) for s, v in by.items()})
        shutil.copy(RUBRICS[name], f"{OUT}{name}_RUBRIC.md")


if __name__ == "__main__":
    main()
