"""Incremental run, step 2 (2026-10-21, RL-165; plan in fullscale_pipeline/INCREMENTAL_RUN.md): P6 inputs and drop lists.

Every P2 v3 record (new articles: staging P2; current corpus: LCS/p2_v3_current/) is cleaned with the Stage B hit rules
(gap_stage_b.py: release date, homonym rules), the F2 manual verdicts (a term judged other-sense or statement in a kept
F2 article) and the AI-use statement labels (owner: statements are not analysed). A focal sentence that loses all its
terms, or is a statement, becomes non-focal (focal = false, terms = []); blocks, sentence ids and hashes are not changed.
Current articles are then compared with their v2 records by sentence id:
    unchanged       same focal sentences as v2
    removals_only   v2 focal sentences dropped, none added -> their ids go to the drop list (applied in P7)
    additions       focal sentences added by v3 (with or without drops) -> the article is rerun from P6
    new_in_v3       a whitelisted current article with no focal sentence under v2 (not yet screened by Stage B's article flags)
    excluded        in LCS/article_exclusions_stage_b.csv (Stage B) -> not carried forward
Outputs in LCS/inc_run/: p2_new.jsonl (15,563 new articles), p2_current_additions.jsonl, drop_sids_current.tsv
(pmcid, sid, why), classification_current.tsv, prepare_summary.json.
    python3 inc_prepare.py [--workers N]
"""
import csv, json, os, sys, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap_stage_b as SB

L = SB.L
GR = SB.GR
OUT = L + "inc_run/"
SRC = {"gap": GR + "llm_corpus_staging/focal_extractions_v2.jsonl", "current": L + "p2_v3_current/focal_extractions_v2.jsonl"}
V2 = L + "focal_extractions_v2.jsonl"
_F2, _FILES, _Y = {}, {}, {}


def init():
    global _F2, _FILES, _Y
    for r in csv.DictReader(open(SB.OUT + "manual/f2_decisions_final.tsv"), delimiter="\t"):
        _F2[(r["corpus"], r["pmcid"], r["term"].lower())] = r["verdict"]
    _FILES = {c: SB.files_by_article(SB.CORPORA[c][1]) for c in ("gap", "current")}
    _Y = SB.years()


def term_label(t):
    k = SB.term_key(t) if not t.startswith("v3:") else t.split(":", 2)[2]
    import re
    return re.sub(r"(?i)^(llm|transformer model|language model)s$", r"\1", k).lower()


def clean(args):
    corpus, line = args
    r = json.loads(line)
    pm = r["pmcid"]; year = _Y[corpus].get(pm)
    text = "\n".join(open(f, encoding="utf-8", errors="replace").read() for f in _FILES[corpus].get(pm, []))
    senses = SB.abbr_senses(text)
    sents = [s for b in r["blocks"] for s in b["sentences"] if s["focal"]]
    lab = {s["sid"]: SB.is_statement(s["text"]) for s in sents}
    if any(lab.values()):
        lab = {s["sid"]: lab[s["sid"]] or bool(SB.NEGATED_LOOSE.search(s["text"])) for s in sents}
    dropped, why = [], collections.Counter()
    for b in r["blocks"]:
        for s in b["sentences"]:
            if not s["focal"]:
                continue
            keep = []
            for t in s["terms"]:
                ry = SB.release_year(t)
                if year and ry and year < ry:
                    why["prerelease"] += 1; continue
                if SB.homonym_reason(t, s["text"], senses):
                    why["homonym_rule"] += 1; continue
                if _F2.get((corpus, pm, term_label(t))) in ("no", "statement"):
                    why["f2_verdict"] += 1; continue
                keep.append(t)
            reason = "statement" if lab[s["sid"]] else ("no_term_left" if not keep else None)
            if reason:
                s["focal"], s["terms"] = False, []
                dropped.append((s["sid"], reason))
            else:
                s["terms"] = keep
    focal = {s["sid"] for b in r["blocks"] for s in b["sentences"] if s["focal"]}
    r["n_blocks"] = len(r["blocks"])
    return corpus, pm, json.dumps(r, ensure_ascii=False), focal, dropped, dict(why)


def jobs(corpus, keep):
    for line in open(SRC[corpus], encoding="utf-8"):
        pm = line[11:line.index('"', 11)]
        if pm in keep:
            yield corpus, line


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 24
    os.makedirs(OUT, exist_ok=True)
    assert not os.path.exists(OUT + "prepare_summary.json"), "outputs exist"
    new_ids = set(open(GR + "gap_article_list_after_ab.txt").read().split())
    excl = {r["pmcid"] for r in csv.DictReader(open(L + "article_exclusions_stage_b.csv"))}
    v2 = {}
    for line in open(V2, encoding="utf-8"):
        rr = json.loads(line)
        v2[rr["pmcid"]] = {s["sid"] for b in rr["blocks"] for s in b["sentences"] if s["focal"]}
    v3_ids = set()
    for line in open(SRC["current"], encoding="utf-8"):
        v3_ids.add(line[11:line.index('"', 11)])
    cur_ids = (set(v2) | v3_ids) - excl
    summ = dict(new=collections.Counter(), current=collections.Counter(), why=collections.Counter(), dropped=collections.Counter())
    v3_seen = set()
    with Pool(w, initializer=init) as pool, open(OUT + "p2_new.jsonl", "w") as fn, \
            open(OUT + "p2_current_additions.jsonl", "w") as fa, open(OUT + "drop_sids_current.tsv", "w") as fd, \
            open(OUT + "classification_current.tsv", "w") as fc:
        fd.write("pmcid\tsid\twhy\n"); fc.write("pmcid\tclass\tn_v2_focal\tn_v3_focal\tadded\tremoved\n")
        for c, pm, rec, focal, dropped, why in pool.imap(clean, jobs("gap", new_ids), chunksize=50):
            summ["why"].update(why); summ["dropped"].update(f"gap:{x}" for _, x in dropped)
            if focal:
                fn.write(rec + "\n"); summ["new"]["written"] += 1
            else:
                summ["new"]["no_focal_left"] += 1
        for c, pm, rec, focal, dropped, why in pool.imap(clean, jobs("current", cur_ids), chunksize=50):
            v3_seen.add(pm); summ["why"].update(why); summ["dropped"].update(f"current:{x}" for _, x in dropped)
            old = v2.get(pm, set()); added, removed = focal - old, old - focal
            cls = "additions" if added else ("removals_only" if removed else "unchanged")
            if pm not in v2 and focal:
                cls = "new_in_v3"              # whitelisted, no focal sentence under v2: not yet screened by Stage B
            if not focal:
                cls = "no_focal_left"
            summ["current"][cls] += 1
            fc.write(f"{pm}\t{cls}\t{len(old)}\t{len(focal)}\t{len(added)}\t{len(removed)}\n")
            if cls in ("additions", "new_in_v3"):
                fa.write(rec + "\n")
            reasons = dict(dropped)
            for sid in sorted(removed):
                fd.write(f"{pm}\t{sid}\t{reasons.get(sid, 'not_focal_in_v3')}\n")
        for pm in sorted((cur_ids & set(v2)) - v3_seen):           # v2 article that v3 P2 did not output (should not happen)
            summ["current"]["missing_in_v3"] += 1
            fc.write(f"{pm}\tmissing_in_v3\t{len(v2[pm])}\t0\t0\t{len(v2[pm])}\n")
    json.dump({k: dict(v) for k, v in summ.items()}, open(OUT + "prepare_summary.json", "w"), indent=1)
    print(json.dumps({k: dict(v) for k, v in summ.items()}, indent=1))
