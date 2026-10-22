"""G3 inputs for the merged corpus PG/g2_v3_1/ (RL-171), written to PG/g3_inputs_v3_1/:
  scope_exclusions.csv      §6 item 6: the old list (g3_scope_exclusions/scope_exclusions.csv) minus the restored articles,
                            plus the exclusions of decisions/item6_v3_1.csv (pmcid, reason, detail; reasons as before)
  spelling_corrections.csv  the old 6 corrections plus the new ones below, each checked against its g2_v3_1 unit
  term_rejections.tsv       stage_b_rejections.tsv (build_stage_b_rejections.py) without the lifted rows below
  RL-172: scope_exclusions.csv also holds the articles whose only focal sentences are AI-use statements found by the
  second statement check (statement_sentences.csv, retrieval/stmt_check2.py; reason ai_use_statement_only, as F1);
  g4_decisions.csv is copied from decisions/ (write_g4_decisions.py)
  RL-174: decisions/manual_homonyms_rl174.csv adds read homonyms: article exclusions and sentence rejections
    python3 build_inputs_v3_1.py
"""
import csv, glob, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
OUT = PG + "g3_inputs_v3_1/"
sys.path.insert(0, os.path.join(HERE, "..", "g3_scope_exclusions"))
import build_scope_exclusions as BSE

REASON = {"llm_no_definition_no_evidence": "llm_no_definition_no_evidence", "name_homonym": "name_homonym"}
NEW_CORRECTIONS = [("PMC13351568.r1.L79.S1.U1", "linguistic", "language")]     # 'large linguistic model (LLM)', a misnomer
LIFT = {"PMC13351568"}           # its LLM hits were rejected by the homonym rule only because of the misnomer


def reason_of(rule, label):
    if rule.startswith("1"):
        return "llm_defined_otherwise"
    if rule.startswith("2"):
        return "llm_no_definition_no_evidence"
    if rule.startswith("3"):
        return "name_homonym"
    return "transformer_only_not_llm"


if __name__ == "__main__":
    assert not os.path.exists(OUT + "scope_exclusions.csv"), "outputs exist"
    old = list(csv.DictReader(open(os.path.join(HERE, "..", "g3_scope_exclusions", "scope_exclusions.csv"))))
    dec = list(csv.DictReader(open(os.path.join(HERE, "decisions", "item6_v3_1.csv"))))
    restore = {r["pmcid"] for r in dec if r["decision"] == "restore"}
    rows = {r["pmcid"]: (r["pmcid"], r["reason"], r["detail"]) for r in old if r["pmcid"] not in restore}
    c = collections.Counter(old=len(old), restored=len(restore & {r["pmcid"] for r in old}))
    for r in dec:
        if r["decision"] == "exclude" and r["pmcid"] not in rows:
            rows[r["pmcid"]] = (r["pmcid"], reason_of(r["rule"], r["label"]), r["note"] or r["label"]); c["new: " + r["corpus"]] += 1
    corr_old = list(csv.DictReader(open(os.path.join(HERE, "..", "g3_scope_exclusions", "spelling_corrections.csv"))))
    want = {u for u, _, _ in NEW_CORRECTIONS} | {r["uid"] for r in corr_old}
    arts, units = set(), {}
    for p in sorted(glob.glob(PG + "g2_v3_1/shards/g2_parsed_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            a = json.loads(line); arts.add(a["pmcid"])
            for s in a["sentences"]:
                for u in s.get("units", []):
                    if u["uid"] in want:
                        units[u["uid"]] = u
    corr = []
    for r in corr_old:
        assert units[r["uid"]]["hash"] == r["unit_hash"], r["uid"]
        corr.append((r["uid"], r["unit_hash"], r["wrong"], r["right"]))
    for uid, wrong, right in NEW_CORRECTIONS:
        u = units[uid]
        BSE.correct_text(u["text"], wrong, right); BSE.correct_a2w(u["atom2word"], wrong, right)   # raise unless exactly once
        corr.append((uid, u["hash"], wrong, right))
    stm = list(csv.DictReader(open(OUT + "statement_sentences.csv")))          # RL-172
    nfocal = collections.Counter()
    for line in open(PG + "g2_v3_1/focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line); nfocal[r["pmcid"]] = len(r["sentences"])
    per = collections.Counter(r["pmcid"] for r in stm)
    for pm, k in sorted(per.items()):
        if k == nfocal[pm] and pm not in rows:
            rows[pm] = (pm, "ai_use_statement_only", "second statement check (RL-172)"); c["new: statement only"] += 1
    import shutil
    shutil.copy(os.path.join(HERE, "decisions", "g4_decisions.csv"), OUT + "g4_decisions.csv")
    manual = list(csv.DictReader(open(os.path.join(HERE, "decisions", "manual_homonyms_rl174.csv"))))   # RL-174
    for r in manual:
        if r["scope"] == "exclude" and r["key"] not in rows:
            rows[r["key"]] = (r["key"], "name_homonym", r["reason"]); c["new: manual homonym exclusion"] += 1
    c["excluded, in g2_v3_1"] = len(set(rows) & arts); c["excluded, not in g2_v3_1 (Stage B or earlier)"] = len(set(rows) - arts)
    c["total excluded"] = len(rows)
    with open(OUT + "scope_exclusions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pmcid", "reason", "detail"]); w.writerows(sorted(rows.values()))
    with open(OUT + "spelling_corrections.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["uid", "unit_hash", "wrong", "right"]); w.writerows(corr)
    n = 0
    with open(OUT + "stage_b_rejections.tsv", encoding="utf-8") as fi, open(OUT + "term_rejections.tsv", "w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            scope, key, term, why = line.rstrip("\n").split("\t")
            if i and key.split(".")[0] in LIFT:
                c["rejections lifted"] += 1; continue
            fo.write(line); n += i > 0
        for r in manual:                                                              # RL-174: read by the assistant
            if r["scope"] == "sentence":
                fo.write(f"sentence\t{r['key']}\t{r['term']}\tmanual_homonym\n"); n += 1; c["manual sentence rejections"] += 1
    c["term rejections"] = n
    print(json.dumps(dict(c), indent=1))
