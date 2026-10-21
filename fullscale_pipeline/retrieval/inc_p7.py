"""Incremental run, step 4 (2026-10-21, RL-166; plan in fullscale_pipeline/INCREMENTAL_RUN.md): P7 with frozen citation ids.

P7 (pmc_preprocessing/focal_citations_v2.py, unchanged) numbers citation works from scratch over the whole corpus. Here it
is run once on the combined corpus, then its works are mapped back to the existing ids:
  * a work that contains tokens of exactly one existing work keeps that work's id;
  * a work with tokens of several existing works (a new paper links them) keeps the smallest id; the others are recorded as
    aliases (counted, reported);
  * an existing work whose tokens now fall into several works keeps its id in the largest; the rest get new ids (counted);
  * a work with no existing tokens gets the next free id, in P7's own deterministic order.
text_final and hash_final are rewritten with the mapped ids. Inputs are assembled in a staging folder (P7's module-level
paths LCS and PP are pointed at it): coref_v2/coref_resolved_v2.jsonl = existing P6 records of the current articles carried
forward (sentences in inc_run/drop_sids_current.tsv un-flagged) + the new P6 records; citation_dictionary_v2.jsonl and
content_regions_v2.csv of both corpora; pure_text_corpus/ as links to both raw corpora.
    python3 inc_p7.py --test-current-only   (check: the current corpus alone must reproduce every existing id and hash)
    python3 inc_p7.py                        (the real run, after the P6 merges)
Outputs in LCS/inc_run/p7[_test]/out/: focal_sentences_v2.jsonl, citation_works_v2.jsonl, ref_remap.json, inc_p7_summary.json.
"""
import csv, collections, glob, hashlib, importlib.util, json, os, re, sys
R = "/mnt/hum01-rds/Basov/p91688di/"
PP, LCS = R + "pmc_preprocessing/", R + "llm_corpus_staging/"
INC = LCS + "inc_run/"
GR = LCS + "gap_stage/"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def assemble(stage, test):
    os.makedirs(stage + "coref_v2", exist_ok=True); os.makedirs(stage + "pure_text_corpus", exist_ok=True)
    cls = {r["pmcid"]: r["class"] for r in csv.DictReader(open(INC + "classification_current.tsv"), delimiter="\t")}
    drop = collections.defaultdict(set)
    for r in csv.DictReader(open(INC + "drop_sids_current.tsv"), delimiter="\t"):
        drop[r["pmcid"]].add(r["sid"])
    carry = {p for p, c in cls.items() if c in ("unchanged", "removals_only")} if not test else None
    n = collections.Counter(); arts = {}
    with open(stage + "coref_v2/coref_resolved_v2.jsonl", "w", encoding="utf-8") as fo:
        for line in open(LCS + "coref_v2/coref_resolved_v2.jsonl", encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if test or pm in carry:
                if not test and drop.get(pm):
                    r = json.loads(line)
                    for b in r["blocks"]:
                        for s in b["sentences"]:
                            if s["sid"] in drop[pm] and s["focal"]:
                                s["focal"], s["terms"] = False, []; n["sentences un-flagged"] += 1
                    line = json.dumps(r, ensure_ascii=False) + "\n"
                fo.write(line); n["current carried"] += 1; arts[pm] = "current"
        if not test:
            for unit, d in (("current", INC + "p6_cur/"), ("gap", INC + "p6_new/")):
                for line in open(d + "coref_resolved_v2.jsonl", encoding="utf-8"):
                    pm = line[11:line.index('"', 11)]
                    fo.write(line); n[f"{unit} from new P6"] += 1; arts[pm] = unit
    with open(stage + "citation_dictionary_v2.jsonl", "w", encoding="utf-8") as fo:
        for src in (LCS + "citation_dictionary_v2.jsonl",) + (() if test else (GR + "llm_corpus_staging/citation_dictionary_v2.jsonl",)):
            for line in open(src, encoding="utf-8"):
                if line[11:line.index('"', 11)] in arts:
                    fo.write(line)
    with open(stage + "content_regions_v2.csv", "w", encoding="utf-8") as fo:
        for k, src in enumerate((PP + "content_regions_v2.csv",) + (() if test else (GR + "pmc_preprocessing/content_regions_v2.csv",))):
            for j, line in enumerate(open(src, encoding="utf-8")):
                if j == 0 and k > 0:
                    continue
                fo.write(line)
    for pm, unit in arts.items():
        dst = stage + "pure_text_corpus/" + pm + ".txt"
        if not os.path.lexists(dst):
            os.symlink((LCS + "pure_text_corpus/" if unit == "current" else LCS + "pure_text_corpus_gap/") + pm + ".txt", dst)
    return dict(n), arts


def remap(out, test):
    old = {}
    for line in open(LCS + "citation_works_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for mbr in r["members"]:
            old[(mbr["pmcid"], mbr["token"])] = r["ref"]
    works = [json.loads(l) for l in open(out + "citation_works_v2.jsonl", encoding="utf-8")]
    nxt = max(int(v[3:]) for v in old.values()) + 1
    claimed, mapping, c, aliases = {}, {}, collections.Counter(), {}
    # pass 1: works that contain existing tokens, largest first, so a split old work keeps its id in its largest part
    for w in sorted(works, key=lambda w: -len(w["members"])):
        olds = sorted({old[(m["pmcid"], m["token"])] for m in w["members"] if (m["pmcid"], m["token"]) in old})
        if not olds:
            continue
        free = [o for o in olds if o not in claimed]
        if free:
            mapping[w["ref"]] = free[0]; claimed[free[0]] = w["ref"]
            c["kept existing id"] += 1
            if len(olds) > 1:
                c["new link between existing works"] += 1; aliases[free[0]] = [o for o in olds if o != free[0]]
        else:
            c["split of an existing work (new id)"] += 1
    # pass 2: everything else gets the next free ids, in P7's own order
    for w in sorted(works, key=lambda w: int(w["ref"][3:])):
        if w["ref"] not in mapping:
            mapping[w["ref"]] = f"REF{nxt:06d}"; nxt += 1; c["new id"] += 1
    rx = re.compile(r"REF\d{6}")
    with open(out + "citation_works_v2.mapped.jsonl", "w", encoding="utf-8") as fo:
        for w in works:
            w["ref_p7"] = w["ref"]; w["ref"] = mapping[w["ref"]]
            if w["ref"] in aliases:
                w["alias_refs"] = aliases[w["ref"]]
            fo.write(json.dumps(w, ensure_ascii=False) + "\n")
    with open(out + "focal_sentences_v2.mapped.jsonl", "w", encoding="utf-8") as fo:
        for line in open(out + "focal_sentences_v2.jsonl", encoding="utf-8"):
            r = json.loads(line)
            for s in r["sentences"]:
                s["text_final"] = rx.sub(lambda m: mapping[m.group(0)], s["text_final"])
                s["hash_final"] = sha(s["sid"] + "|" + s["text_final"])
                for ci in s["citations"]:
                    ci["ref"] = mapping[ci["ref"]]
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(mapping=mapping, aliases=aliases), open(out + "ref_remap.json", "w"))
    return dict(c)


def compare(out):
    """every sentence of the existing P7 output that is still kept: same sid, same hash_final?"""
    old = {}
    for line in open(LCS + "focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for s in r["sentences"]:
            old[s["sid"]] = (s["hash_final"], s["text_final"])
    c, ex = collections.Counter(), []
    seen = set()
    for line in open(out + "focal_sentences_v2.mapped.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for s in r["sentences"]:
            seen.add(s["sid"])
            if s["sid"] not in old:
                c["new sentence"] += 1
            elif old[s["sid"]][0] == s["hash_final"]:
                c["unchanged"] += 1
            else:
                c["changed"] += 1
                if len(ex) < 20:
                    ex.append(dict(sid=s["sid"], old=old[s["sid"]][1][:200], new=s["text_final"][:200]))
    c["existing sentence no longer kept"] = len(set(old) - seen)
    return dict(c), ex


if __name__ == "__main__":
    test = "--test-current-only" in sys.argv
    base = INC + ("p7_test/" if test else "p7/"); stage, out = base + "lcs/", base + "out/"
    os.makedirs(stage, exist_ok=True)
    assert not os.path.exists(out + "inc_p7_summary.json"), "outputs exist"
    n, arts = assemble(stage, test)
    sys.path.insert(0, PP)
    load("focal_terms", PP + "focal_terms.py")
    if not test:                                                    # the test reproduces the existing run: v2 matcher
        sys.modules["focal_terms"] = load("focal_terms_v3", PP + "focal_terms_v3.py")
    m = load("focal_citations_v2", PP + "focal_citations_v2.py")
    m.LCS, m.PP = stage, stage
    sys.argv = ["focal_citations_v2.py", "--outdir", out, "--workers", "24"]
    m.main()
    rc = remap(out, test)
    cmp, ex = compare(out)
    summ = dict(assembled=n, articles=len(arts), remap=rc, comparison=cmp, examples_changed=ex)
    json.dump(summ, open(out + "inc_p7_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "examples_changed"}, indent=1))
    for e in ex[:5]:
        print(e)
