# Article types of the 28,075 articles G3 marks "valid" (G3_POSTPROCESSING.md §6 item 6 /
# item 8 step 9; PG/g3_v1/g3_articles_v1.jsonl) -- the corpus M1 will actually build matrices
# from, not the full 34,662-article G2 corpus CORPUS_STATISTICS.md §1-5 describes.
#
# Source of "type" here is deliberately narrow: MEDLINE's own PublicationType tags, fetched
# from NCBI Entrez (corpus_statistics/pubmed_pubtype_fetch.py, cached in
# diagnostics/openalex_authors/pubmed_pubtypes.json) for the PMID printed in the article's own
# PMC header (diagnostics/openalex_authors/headers.json, from the PMC pure_text_corpus header,
# R2 stage -- not OpenAlex, not the publisher's self-declared JATS "Subjects:" line that
# CORPUS_STATISTICS.md §2 row (a) and the older diagnostics/paper_type_table.py used as a
# fallback). Where PMC gives no PMID, or PubMed's own record exists but carries no
# PublicationType, that is reported as a gap, not filled from another source.
import json, csv, collections

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain"
DATA = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors"
OUT = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/article_types_valid.csv"

valid = sorted(
    json.loads(line)["pmcid"]
    for line in open(f"{PG}/g3_v1/g3_articles_v1.jsonl")
    if json.loads(line)["status"] == "valid"
)
assert len(valid) == 28075, f"expected 28,075 valid articles, got {len(valid)}"

H = json.load(open(f"{DATA}/headers.json"))
PT = json.load(open(f"{DATA}/pubmed_pubtypes.json"))

# Priority order over MEDLINE's controlled-vocabulary PublicationType tags only (no free text,
# no regex over self-declared fields). Administrative/boilerplate tags (funding source,
# language note) and co-occurring study-design descriptors are not categories of their own --
# they are reported separately as flags -- so they never pre-empt a real content-type match.
ADMIN_TAGS = {
    "Research Support, Non-U.S. Gov't", "Research Support, N.I.H., Extramural",
    "Research Support, N.I.H., Intramural", "Research Support, U.S. Gov't, Non-P.H.S.",
    "Research Support, U.S. Gov't, P.H.S.", "English Abstract",
}
STUDY_DESIGN_FLAG_TAGS = {
    "Comparative Study", "Validation Study", "Observational Study", "Evaluation Study",
    "Multicenter Study",
}
RETRACTED_TAG = "Retracted Publication"

CATEGORIES = [
    ("Systematic review / meta-analysis / scoping review",
     {"Systematic Review", "Meta-Analysis", "Network Meta-Analysis", "Scoping Review", "Evidence Synthesis"}),
    ("Clinical trial / protocol",
     {"Randomized Controlled Trial", "Clinical Trial", "Clinical Trial Protocol",
      "Clinical Trial, Phase I", "Clinical Trial, Phase II", "Pragmatic Clinical Trial",
      "Equivalence Trial", "Clinical Study"}),
    ("Review (narrative / other)", {"Review"}),
    ("Guideline / consensus statement", {"Guideline", "Consensus Statement"}),
    ("Editorial / letter / comment / news",
     {"Editorial", "Letter", "Comment", "News", "Interview", "Introductory Journal Article"}),
    ("Dataset / data paper", {"Dataset"}),
    ("Conference proceedings / abstract", {"Conference Proceedings"}),
    ("Historical article", {"Historical Article"}),
    ("Research article (Journal Article tag, no more specific type)", {"Journal Article"}),
]
OTHER_NAMED_TAGS = {"Video-Audio Media"}  # seen once; no dedicated bucket


def classify(tags):
    tagset = set(tags)
    for name, members in CATEGORIES:
        if tagset & members:
            return name
    if tagset & OTHER_NAMED_TAGS:
        return "Other (named PubMed type with no dedicated category)"
    if tagset - ADMIN_TAGS - STUDY_DESIGN_FLAG_TAGS - {RETRACTED_TAG}:
        return "Other (named PubMed type with no dedicated category)"
    return "Unclassified (PubMed record has only administrative/boilerplate tags)"


rows = []
no_pmid = []
no_pubmed_record = []
empty_type_list = []
tag_count = collections.Counter()
primary_count = collections.Counter()
retracted = []

for pmcid in valid:
    pmid = (H.get(pmcid) or {}).get("pmid")
    if not pmid:
        no_pmid.append(pmcid)
        rows.append((pmcid, "", "", "No PMID in PMC header", "unknown (no PMID to check)"))
        primary_count["No PMID in PMC header"] += 1
        continue
    types = PT.get(pmid)
    if types is None:
        no_pubmed_record.append(pmcid)
        rows.append((pmcid, pmid, "", "PMID not found by Entrez efetch", "unknown (no PubMed record to check)"))
        primary_count["PMID not found by Entrez efetch"] += 1
        continue
    names = [t["name"] for t in types]
    if not names:
        empty_type_list.append(pmcid)
    for n in names:
        tag_count[n] += 1
    is_retracted = RETRACTED_TAG in names
    if is_retracted:
        retracted.append(pmcid)
    cat = classify(names) if names else "Unclassified (PubMed record has no PublicationType at all)"
    primary_count[cat] += 1
    rows.append((pmcid, pmid, "|".join(names), cat, "True" if is_retracted else "False"))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pmcid", "pmid", "pubmed_publication_types", "primary_type", "retracted"])
    w.writerows(rows)

print(f"valid articles: {len(valid):,}")
print(f"no PMID in PMC header (PMC/Entrez has no linkage at all): {len(no_pmid):,}")
print(f"PMID present but Entrez efetch found no record: {len(no_pubmed_record):,}")
print(f"PubMed record found but PublicationType list empty: {len(empty_type_list):,}")
print(f"tagged 'Retracted Publication' (co-occurs with another type, not its own category): {len(retracted):,}")
print()
print("-- raw MEDLINE PublicationType tags (multi-label; an article can carry several) --")
print("| PublicationType (MEDLINE) | n articles |\n|---|---|")
for n, c in tag_count.most_common():
    print(f"| {n} | {c:,} |")
print()
print("-- single-label primary type (priority order, PubMed tags only) --")
print("| Primary type | n articles |\n|---|---|")
for n, c in primary_count.most_common():
    print(f"| {n} | {c:,} |")
print(f"| **Total** | **{sum(primary_count.values()):,}** |")
