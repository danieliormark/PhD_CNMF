"""A3 part 2, step 4 (2026-10-26, RL-203): P7 with frozen citation ids, the new sentences added.
inc_p7.py (RL-166/169) is reused unchanged: its assembly of the P6 records exactly as in the p7b run (drop list v2, restored
research sentences), P7 itself, its REF remap and its comparison. Two things differ:
  * in the articles with an addition (a3p2_prepare.py, P6 rerun by a3p2_p6.py) the new sentences are added to the existing
    P6 record; existing sentences are never touched (their text, resolved text, focal flag and block stay as they were).
    Added: a sentence of the new P6 record that is focal there and was either absent from the old record or not focal in it
    (made focal by a hub antecedent through coreference). Absent ones go into one extra block per region,
    "<PMCID>.r<N>.A3B1" (block ids are not read by G2 or G3); sentences rejected on reading (--reject) are not added.
  * ids are frozen against the merged corpus (PG/g2_v3_1/citation_works_v2.jsonl, the p7b ids), and the comparison is with
    PG/g2_v3_1/focal_sentences_v2.jsonl.
    python3 a3p2_p7.py --reproduce          (no sentence added: every existing sentence must come back unchanged)
    python3 a3p2_p7.py [--reject rejected.tsv]
Outputs in LCS/a3p2/p7[_reproduce]/out/ (as inc_p7.py) and added_sentences.json.
"""
import collections, csv, importlib.util, json, os, sys

R = "/mnt/hum01-rds/Basov/p91688di/"
LCS, PG, PP = R + "llm_corpus_staging/", R + "phase5_graphbrain/", R + "pmc_preprocessing/"
A = LCS + "a3p2/"
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def add_new(stage, reject):
    new = {}
    for unit in ("current", "gap"):
        f = A + f"p6_{unit}/coref_resolved_v2.jsonl"
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); new[r["pmcid"]] = r
    path = stage + "coref_v2/coref_resolved_v2.jsonl"
    c, added = collections.Counter(), collections.defaultdict(list)
    with open(path + ".tmp", "w", encoding="utf-8") as fo:
        for line in open(path, encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if pm not in new:
                fo.write(line); continue
            r = json.loads(line); old = {s["sid"]: s for b in r["blocks"] for s in b["sentences"]}
            extra = collections.defaultdict(list)
            for b in new[pm]["blocks"]:
                for s in b["sentences"]:
                    if not s["focal"] or s["sid"] in reject or (s["sid"] in old and old[s["sid"]]["focal"]):
                        continue
                    if s["sid"] in old:
                        old[s["sid"]].clear(); old[s["sid"]].update(s); c["made focal in an existing block"] += 1
                    else:
                        extra[b["region"]].append(s); c["added in an extra block"] += 1
                    added[pm].append(s["sid"])
                    for t in s["terms"]:
                        r["terms"][t.lower()] = r["terms"].get(t.lower(), 0) + 1
            for reg, ss in sorted(extra.items()):
                r["blocks"].append(dict(block_id=f"{pm}.r{reg}.A3B1", region=reg, raw_line_first=min(s["raw_line"] for s in ss),
                                        raw_line_last=max(s["raw_line"] for s in ss), n_sentences=len(ss), sentences=ss))
            r["n_blocks"] = len(r["blocks"])
            c["articles with an addition"] += bool(added[pm])
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    c["P6 records of the rerun not in the assembly (must be 0)"] = len(set(new) - set(json.loads(l)["pmcid"] for l in open(path)))
    return dict(c), added


if __name__ == "__main__":
    rep = "--reproduce" in sys.argv
    reject = set()
    if "--reject" in sys.argv:
        reject = {r["sid"] for r in csv.DictReader(open(sys.argv[sys.argv.index("--reject") + 1]), delimiter="\t")}
    ip = load("inc_p7", HERE + "inc_p7.py")
    ip.DROP = ip.INC + "drop_sids_current_v2.tsv"
    ip.REFLAG = json.load(open(ip.INC + "reflag_terms.json"))
    base = A + ("p7_reproduce/" if rep else "p7/"); stage, out = base + "lcs/", base + "out/"
    os.makedirs(stage, exist_ok=True)
    assert not os.path.exists(out + "inc_p7_summary.json"), "outputs exist"
    n, arts = ip.assemble(stage, False)
    na, added = ({}, {}) if rep else add_new(stage, reject)
    sys.path.insert(0, PP)
    load("focal_terms", PP + "focal_terms.py")
    load("focal_terms_v3", PP + "focal_terms_v3.py")
    sys.modules["focal_terms"] = load("focal_terms_v3_1", PP + "focal_terms_v3_1.py")
    m = load("focal_citations_v2", PP + "focal_citations_v2.py")
    m.LCS, m.PP = stage, stage
    sys.argv = ["focal_citations_v2.py", "--outdir", out, "--workers", "24"]
    m.main()
    ip.LCS = PG + "g2_v3_1/"                     # frozen ids and comparison: the merged corpus (p7b ids), not the May corpus
    rc = ip.remap(out, False)
    cmp, ex = ip.compare(out)
    json.dump(added, open(out + "added_sentences.json", "w"), indent=1)
    summ = dict(assembled=n, added=na, articles=len(arts), remap=rc, comparison=cmp, examples_changed=ex)
    json.dump(summ, open(out + "inc_p7_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "examples_changed"}, indent=1))
    for e in ex[:5]:
        print(e)
