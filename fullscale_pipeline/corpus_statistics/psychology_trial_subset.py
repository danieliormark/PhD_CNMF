# RL-198 (owner 2026-10-26): the article list of the first M1 trial, Psychology. Read-only; no network. Steps, each
# recorded per excluded article in psychology_trial_exclusions.csv (stage, reason):
#   0. working corpus v2 (working_corpus_v2_pmcids.txt, 41,640) articles whose OpenAlex primary topic is in the field
#      Psychology (subset_candidates_by_article.csv, RL-193; owner RL-194: primary topic only) -> 2,255
#   1. language-model demotion (RL-196): an article left with no focal unit after the targeted G3 rerun
#      (PG/g3_v2_lm_demotion/demoted/g3_articles_*.jsonl, status not "valid") leaves; its G3 records for M1 are those of
#      the demoted run, not of PG/g3_v2/ (the other 2,162 - 27 = 2,135 articles are unchanged by the rerun)
#   2. field review (RL-197 step 1 assistant, owner's decisions in psychology_field_review.csv, GitHub commit d8fcbae):
#      decision "drop" leaves; "keep" (and the owner's spelling "kep") stays; "flag" (not yet decided by the owner)
#      stays, listed as pending
#   3. author identity (RL-200; AUTHOR_IDENTITY.md R2/R6, decided): an article whose every byline is removed by R2 or R6
#      (diagnostics/author_coverage/author_table_v2.jsonl, author_linking_v2/author_table.py) leaves the working corpus.
#      Articles without a PubMed record (no PMID, so no byline backbone) are not removed here: listed as pending.
# Writes corpus_statistics/psychology_trial_pmcids.txt (the trial list), psychology_trial_exclusions.csv and
# psychology_trial_pending.csv.
import collections, csv, glob, json

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
HERE = FP + "corpus_statistics/"
DEM = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v2_lm_demotion/demoted/"

wc = {l.strip() for l in open(HERE + "working_corpus_v2_pmcids.txt") if l.strip()}
psy = sorted(r["pmcid"] for r in csv.DictReader(open(HERE + "subset_candidates_by_article.csv"))
             if r["field"] == "Psychology" and r["pmcid"] in wc)
status = {}
for f in glob.glob(DEM + "g3_articles_*.jsonl"):
    for l in open(f):
        r = json.loads(l); status[r["pmcid"]] = r["status"]
review = {r["pmcid"]: r for r in csv.DictReader(open(HERE + "psychology_field_review.csv"))}
KEEP = {"keep", "kep"}
unknown = {r["decision"] for r in review.values()} - KEEP - {"drop", "flag"}
assert not unknown, f"unexpected decisions {unknown}"

AT = collections.defaultdict(list)
for l in open(FP + "diagnostics/author_coverage/author_table_v2.jsonl"):
    r = json.loads(l); AT[r["pmcid"]].append(r["status"])
kept, excl, pending = [], [], []
for p in psy:
    if status.get(p, "valid") != "valid":
        excl.append((p, "1 language-model demotion", f"no focal unit left after demotion (G3 status {status[p]})"))
        continue
    r = review[p]
    if r["decision"] == "drop":
        excl.append((p, "2 field review", f"owner: drop (assistant tier {r['tier']}: {r['reason_or_theme']})"))
        continue
    if AT.get(p) and "kept" not in AT[p]:
        excl.append((p, "3 author identity", "every byline removed by R2/R6 (no author left)"))
        continue
    if r["decision"] == "flag":
        pending.append((p, r["tier"], r["reason_or_theme"], r["title"]))
    if not AT.get(p):
        pending.append((p, "", "no PubMed record (no PMID): no byline backbone, author handling undecided", r["title"]))
    kept.append(p)
assert set(review) == {p for p in psy if status.get(p, "valid") == "valid"}, "review covers a different article set"
open(HERE + "psychology_trial_pmcids.txt", "w").write("\n".join(kept) + "\n")
with open(HERE + "psychology_trial_exclusions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pmcid", "stage", "reason", "title"])
    for p, s, why in excl:
        w.writerow([p, s, why, (review.get(p) or {}).get("title", "")])
with open(HERE + "psychology_trial_pending.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pmcid", "assistant_tier", "theme", "title"]); w.writerows(pending)
print(f"Psychology {len(psy)}; demotion -{sum(e[1].startswith('1') for e in excl)}; field review "
      f"-{sum(e[1].startswith('2') for e in excl)}; author identity -{sum(e[1].startswith('3') for e in excl)}; trial list {len(kept)} (of which {len(pending)} pending owner decision)")
