# The working corpus on the RL-186 G3 output (owner 2026-10-24: retracted papers are removed), same rule as
# working_corpus.py (the RL-130 / G3 v1 corpus): the articles G3 marks "valid" (PG/g3_v2/g3_articles_v2.jsonl), minus
# every article MEDLINE tags "Retracted Publication" (pubmed_pubtypes.json, NCBI efetch, for the PMID of the article's
# own PMC header: headers.json for the current corpus, headers_new.json for the new articles, RL-187). Articles
# without a PMID are kept, their retraction status unknown, and counted. G3's output is not changed. Every excluded
# article and its reason: corpus_exclusions_post_g3_v2.csv. Retraction notices ("Retraction of Publication") are
# counted as well; none is expected among valid articles.
import csv, json

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain"
FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
HERE, D = FP + "corpus_statistics/", FP + "diagnostics/openalex_authors/"

valid = sorted(json.loads(l)["pmcid"] for l in open(f"{PG}/g3_v2/g3_articles_v2.jsonl") if json.loads(l)["status"] == "valid")
H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
PT = json.load(open(D + "pubmed_pubtypes.json"))
excluded, notices, no_pmid, no_record = [], [], 0, 0
for p in valid:
    pmid = (H.get(p) or {}).get("pmid")
    if not pmid:
        no_pmid += 1; continue
    t = [x["name"] for x in (PT.get(pmid) or [])]
    if PT.get(pmid) is None:
        no_record += 1
    if "Retracted Publication" in t:
        excluded.append((p, "MEDLINE PublicationType: Retracted Publication", pmid, "|".join(t)))
    if "Retraction of Publication" in t:
        notices.append(p)
ex = {p for p, *_ in excluded}
working = [p for p in valid if p not in ex]
with open(HERE + "corpus_exclusions_post_g3_v2.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pmcid", "reason", "pmid", "pubmed_publication_types"]); w.writerows(excluded)
open(HERE + "working_corpus_v2_pmcids.txt", "w").write("\n".join(working) + "\n")
print(f"G3 valid (RL-186): {len(valid):,}; retracted, excluded: {len(excluded):,}; working corpus v2: {len(working):,}")
print(f"no PMID (status unknown, kept): {no_pmid:,}; PMID without a PubMed type record: {no_record:,}; retraction notices: {len(notices)}")
old = set(l.strip() for l in open(HERE + "corpus_exclusions_post_g3.csv").read().split("\n")[1:] if l.strip() for l in [l.split(",")[0]])
print("of the RL-130 exclusions still valid in RL-186 and excluded again:", len(old & ex), "of", len(old))
