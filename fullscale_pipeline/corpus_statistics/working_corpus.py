# The frozen working corpus for every analysis downstream of G3 (owner decision 2026-10-10):
# the 28,075 articles G3 marks "valid" (PG/g3_v1/g3_articles_v1.jsonl), minus every article
# MEDLINE itself tags "Retracted Publication" (article_types_valid.py, CORPUS_STATISTICS.md
# §8) -- 46 articles, removed because they were retracted, not because of anything G3 found.
#
# G3's own output on RDS is NOT touched or rerun for this: retraction is an article-level fact
# from PubMed, orthogonal to G3's focal-term/parse-level exclusions (§6 item 6), so it is
# applied as a further, separately-recorded filter at this corpus-definition layer, the same
# way the M1 input contract (G3_POSTPROCESSING.md §8 item 3) will have to apply G3's own
# scope_exclusions.csv. Nothing is silently dropped: every excluded pmcid and its reason is
# written to corpus_exclusions_post_g3.csv, same pattern as g3_scope_exclusions/scope_exclusions.csv.
# The 38 valid articles with no PMID (no PubMed record to check) are KEPT -- their retraction
# status is simply unknown, not evidenced, and that gap is a separate, already-documented flag.
import csv, json

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain"
HERE = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics"

valid = sorted(
    json.loads(line)["pmcid"]
    for line in open(f"{PG}/g3_v1/g3_articles_v1.jsonl")
    if json.loads(line)["status"] == "valid"
)
assert len(valid) == 28075

rows = {r["pmcid"]: r for r in csv.DictReader(open(f"{HERE}/article_types_valid.csv"))}
assert set(rows) == set(valid)

excluded = [(p, "MEDLINE PublicationType: Retracted Publication", rows[p]["pmid"], rows[p]["pubmed_publication_types"])
            for p in valid if rows[p]["retracted"] == "True"]
excluded_set = {p for p, *_ in excluded}
working = [p for p in valid if p not in excluded_set]

with open(f"{HERE}/corpus_exclusions_post_g3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pmcid", "reason", "pmid", "pubmed_publication_types"])
    w.writerows(excluded)

with open(f"{HERE}/working_corpus_pmcids.txt", "w") as f:
    f.write("\n".join(working) + "\n")

print(f"G3 valid: {len(valid):,}")
print(f"excluded (retracted): {len(excluded):,}")
print(f"working corpus: {len(working):,}")
no_pmid_in_working = sum(1 for p in working if not rows[p]["pmid"])
print(f"of which, retraction status unknown (no PMID): {no_pmid_in_working:,}")
