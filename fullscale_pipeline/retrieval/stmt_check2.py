"""Second statement check (2026-10-22, RL-172; owner: identify the AI-use statements still left as focal sentences in kept
articles, Haiku readers allowed). The RL-169 check read every sentence the broad statement rule had flagged; statements
the rule did not flag are still in PG/g2_v3_1/. A rule cannot separate them from research sentences with the same words
(RL-169: ~83% precision at ~65% recall; here a tight pattern was ~45% precise on a sample of 30), so patterns only collect
candidates and readers decide, with the RL-169 rubric and prompt unchanged.
Candidates: focal sentences of g2_v3_1 in articles not excluded (g3_inputs_v3_1/scope_exclusions.csv) and not judged in
RL-169 (stmt_check/final_verdicts.json) that match any of
    A & B   a use word followed by an editing purpose (language, grammar, readability, proofreading, translation ...)
            and a reference to this paper's own text (this manuscript / paper / work ..., the manuscript, the writing)
    C       a disclosure formula: "is not listed as an author", "during the preparation of this work", "declaration of AI"
    D       loose: "authorship", "credited as author", "no generative AI ... was used", "not used in the writing"
    E       loose: "the authors reviewed", "take(s) full responsibility"
Item contents as in RL-169 (stmt_check.contexts: title, section heading, sentence, neighbours). Outputs in
LCS/inc_run/stmt_check2/: candidates.tsv (with the pattern), batch_NN.txt (about 250 questions each), batch_NN.json, RUBRIC.md.
    python3 stmt_check2.py build
"""
import csv, json, os, random, re, shutil, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stmt_check as SC

L = SC.L
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
OUT = L + "inc_run/stmt_check2/"
BATCH = 250
OWN = r"(?:this|the present|our|the current)\s+(?:manuscript|paper|article|work|review|text|draft)|the (?:manuscript|text)\b|\bthe writing\b"
PURPOSE = r"(?:language|grammar|grammatical|spelling|readability|proofread\w*|copy-?edit\w*|polish\w*|wording|phrasing|english|clarity|translat\w*|editing|editorial)"
A = re.compile(rf"(?i)\b(?:used|utili[sz]ed|employed|assist\w*|helped|applied|reliance)\b.{{0,140}}\b{PURPOSE}\b")
B = re.compile(rf"(?i){OWN}")
C = re.compile(r"(?i)\b(?:is|was|were|are)\s+not\s+(?:listed|credited|named)\s+as\s+(?:an?\s+)?(?:co-?)?author|during the preparation of this (?:work|manuscript)|declaration of (?:generative )?ai")
D = re.compile(r"(?i)\b(?:listed|credited|named)\s+as\s+(?:an?\s+)?(?:co-?)?author|\bauthorship\b|no\s+(?:generative\s+)?(?:ai|llm|artificial)[^.]{0,80}\bwas\s+used\b|\b(?:were|was)\s+not\s+used\s+(?:in|for)\s+(?:the\s+)?(?:writing|preparation)")
E = re.compile(r"(?i)the authors? (?:reviewed|take[s]? full responsibility)")


def pattern(t):
    if A.search(t) and B.search(t):
        return "AB"
    for name, rx in (("C", C), ("D", D), ("E", E)):
        if rx.search(t):
            return name
    return None


def build():
    os.makedirs(OUT, exist_ok=True)
    assert not os.path.exists(OUT + "candidates.tsv"), "outputs exist"
    judged = json.load(open(L + "inc_run/stmt_check/final_verdicts.json"))["verdicts"]
    excl = {r["pmcid"] for r in csv.DictReader(open(PG + "g3_inputs_v3_1/scope_exclusions.csv"))}
    new_ids = set(open(L + "gap_stage/gap_article_list_after_ab.txt").read().split())
    cands, pat = [], {}
    for line in open(PG + "g2_v3_1/focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["pmcid"] in excl:
            continue
        for s in r["sentences"]:
            if s["sid"] in judged:
                continue
            p = pattern(s["text"])
            if p:
                cands.append(("gap" if r["pmcid"] in new_ids else "current", r["pmcid"], s["sid"])); pat[s["sid"]] = p
    ctx = SC.contexts(cands)
    with open(OUT + "candidates.tsv", "w") as f:
        f.write("corpus\tpmcid\tsid\tpattern\tcontext\n")
        for c, p, s in cands:
            f.write(f"{c}\t{p}\t{s}\t{pat[s]}\t{int(s in ctx)}\n")
    shutil.copy(L + "inc_run/stmt_check/RUBRIC.md", OUT + "RUBRIC.md")
    ids = sorted(ctx); random.seed(22); random.shuffle(ids)
    nb = -(-len(ids) // BATCH)
    for b in range(nb):
        part = ids[b * BATCH:(b + 1) * BATCH]
        items = [(f"S{b + 1:02d}.{i:03d}", ctx[s]) for i, s in enumerate(part, 1)]
        SC.render(items, OUT + f"batch_{b + 1:02d}.txt", f"Batch {b + 1:02d}")
        json.dump({q: dict(sid=s, corpus=ctx[s]["corpus"], pmcid=ctx[s]["pmcid"], pattern=pat[s]) for (q, _), s in zip(items, part)},
                  open(OUT + f"batch_{b + 1:02d}.json", "w"), indent=1)
    print("candidates", len(cands), collections.Counter(pat.values()), "with context", len(ctx), "batches", nb)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
