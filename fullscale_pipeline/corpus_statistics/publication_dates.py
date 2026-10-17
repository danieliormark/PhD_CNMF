# Publication-process dates of the working corpus (owner, 2026-10-17; CORPUS_STATISTICS.md §9.12): coverage of each
# date by year, and agreement of the year definitions used so far. Read-only, no network.
# Sources: PMC article header in the pure-text corpus ("Received ...; Revised ...; Accepted ...", "Electronic
# publication date:", "Publication date:"); PubMed (pubmed_authors.json: epub_year = ArticleDate, pub_year =
# JournalIssue PubDate); OpenAlex publication_year (openalex_works.json, same version rule as author_repeat_share.py).
import collections, csv, datetime, json, re, statistics
from concurrent.futures import ThreadPoolExecutor

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
PURE = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus/"
D, OD = FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
W = [l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()]
H, V = json.load(open(D + "headers.json")), json.load(open(D + "openalex_verified.json"))
OW, PM = json.load(open(OD + "openalex_works.json")), json.load(open(OD + "pubmed_authors.json"))
PMID = {r["pmcid"]: r["pmid"] for r in csv.DictReader(open(FP + "corpus_statistics/article_types_valid.csv"))}
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    bd = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return bd if bd and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


def header(p):
    try:
        t = open(PURE + p + ".txt", encoding="utf-8", errors="replace").read(15000)
    except OSError:
        return p, {}
    g = lambda rx: (m.group(1).strip() if (m := re.search(rx, t, re.M)) else None)
    return p, dict(received=g(r"^Received (\d{4}[^;\n]*)"), accepted=g(r"Accepted (\d{4}[^;\n]*)"),
                   pmc_epub=g(r"^Electronic publication date: (.+)$"), pmc_print=g(r"^Publication date: (.+)$"))


def day(s):
    m = re.match(r"(\d{4}) (\w{3}) (\d{1,2})", s or "")
    return datetime.date(int(m.group(1)), MON[m.group(2)], int(m.group(3))) if m and m.group(2) in MON else None


with ThreadPoolExecutor(32) as ex:
    HD = dict(ex.map(header, W))
y4 = lambda s: s[:4] if s and re.match(r"\d{4}", s) else None
Y = {}
for p in W:
    w, rec, h = work(p), PM.get(PMID.get(p) or "") or {}, HD[p]
    Y[p] = dict(openalex=str(w["year"]) if w and w.get("year") else None,
                pubmed_epub=rec.get("epub_year"), pubmed_print=rec.get("pub_year"),
                pmc_epub=y4(h.get("pmc_epub")), pmc_print=y4(h.get("pmc_print")),
                received=y4(h.get("received")), accepted=y4(h.get("accepted")))
K = list(Y[W[0]].keys())

print("Coverage by OpenAlex year (share of that year's articles with each date):\n")
print("| OpenAlex year | articles | " + " | ".join(K[1:]) + " |\n|" + "---|" * (len(K) + 1))
byy = collections.defaultdict(list)
for v in Y.values():
    byy[v["openalex"] or "none"].append(v)
for y in sorted(byy) + ["all"]:
    L = list(Y.values()) if y == "all" else byy[y]
    print(f"| {y} | {len(L):,} | " + " | ".join(f"{100 * sum(1 for v in L if v[k]) / len(L):.0f}%" for k in K[1:]) + " |")

print("\nYear agreement with OpenAlex publication_year (articles where both are known):")
L = list(Y.values())
for k in K[1:]:
    both = [v for v in L if v["openalex"] and v[k]]
    diff = collections.Counter(int(v[k]) - int(v["openalex"]) for v in both if v[k] != v["openalex"])
    print(f"  {k:13s} n={len(both):6,}  same year {sum(v['openalex'] == v[k] for v in both) / len(both):.1%}; "
          f"differences (this - OpenAlex): {dict(sorted(diff.items()))}")
both = [v for v in L if v["pubmed_epub"] and v["pmc_epub"]]
print(f"  PubMed epub vs PMC epub: same year {sum(v['pubmed_epub'] == v['pmc_epub'] for v in both) / len(both):.1%} of {len(both):,}")
pm_rule = [(v["pubmed_epub"] or v["pubmed_print"], v["openalex"]) for v in L if v["openalex"] and (v["pubmed_epub"] or v["pubmed_print"])]
print(f"  PubMed epub-else-print (bylines' year) vs OpenAlex: same year {sum(a == b for a, b in pm_rule) / len(pm_rule):.1%} of {len(pm_rule):,}")

lag = [(e - r).days for p in W if (r := day(HD[p].get("received"))) and (e := day(HD[p].get("pmc_epub")))]
q = statistics.quantiles(lag, n=10)
print(f"\nReceived -> PMC e-publication, days (both precise to the day): n={len(lag):,}, median {statistics.median(lag):.0f}, "
      f"p10 {q[0]:.0f}, p90 {q[8]:.0f}")
json.dump(dict(header_dates=HD, years=Y), open(OD + "publication_dates.json", "w"))
