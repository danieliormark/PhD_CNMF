# PMID, DOI and author lines (with ORCID where printed) from the PMC header of every
# article that has at least one focal sentence (focal_sentences_v2.jsonl, 34,662 articles).
import json, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics")
from fullscale_analysis import parse_header

PURE = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus/"
FOCAL = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/focal_sentences_v2.jsonl"
OUT = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/headers.json"
ORCID = re.compile(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")

ids = [json.loads(l)["pmcid"] for l in open(FOCAL)]


def one(pmcid):
    try:
        text = open(PURE + pmcid + ".txt", encoding="utf-8", errors="replace").read(12000)
    except OSError:
        return pmcid, None
    pmid = re.search(r"^PMID:\s*(\d+)", text, re.M)
    doi = re.search(r"^DOI:\s*(\S+)", text, re.M)
    subj, authors = parse_header(text)
    return pmcid, dict(pmid=pmid.group(1) if pmid else None, doi=doi.group(1) if doi else None,
                       n_author_lines=len(authors),
                       orcids=[m.group(1) for a in authors for m in [ORCID.search(a)] if m])


with ThreadPoolExecutor(32) as ex:
    res = dict(ex.map(one, ids))
json.dump(res, open(OUT, "w"))
ok = [v for v in res.values() if v]
print("articles", len(ids), "header found", len(ok), "with PMID", sum(1 for v in ok if v["pmid"]),
      "with DOI", sum(1 for v in ok if v["doi"]))
print("author lines", sum(v["n_author_lines"] for v in ok), "with ORCID", sum(len(v["orcids"]) for v in ok),
      "distinct ORCIDs", len({o for v in ok for o in v["orcids"]}))
