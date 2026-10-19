# Ticket 107 P1.A check 3 (owner, 2026-10-19): author lists in the PMC full-text header vs PubMed's, per article,
# working corpus (28,029; trial slices reported separately). PMC header author lines (fullscale_analysis.parse_header:
# one line per contributor, name followed by CRediT roles / affiliation markers) vs PubMed's author list
# (pubmed_authors.json). Editors ("Academic Editor", "Guest Editor"...) are not authors and are set aside.
# A PubMed author is "found" if some non-editor header line contains its normalised surname and given-name initial.
# Read-only, no network.
import collections, csv, json, re, sys, unicodedata
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics")
from fullscale_analysis import parse_header

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OD = FP + "diagnostics/author_coverage/"
PURE = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/pure_text_corpus/"
W = [l.strip() for l in open(FP + "corpus_statistics/working_corpus_pmcids.txt") if l.strip()]
PM = json.load(open(OD + "pubmed_authors.json"))
PMID = {r["pmcid"]: r["pmid"] for r in csv.DictReader(open(FP + "corpus_statistics/article_types_valid.csv"))}
TRIAL = json.load(open(OD + "trial_p1_checks.json"))["trial_pmcids"]
TRIAL = set(TRIAL["2020"]) | set(TRIAL["2021"])
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
EDITOR = re.compile(r"\b(editor|reviewer)\b", re.I)


def one(p):
    try:
        t = open(PURE + p + ".txt", encoding="utf-8", errors="replace").read(20000)
    except OSError:
        return p, None
    _, lines = parse_header(t)
    return p, [l for l in lines if not EDITOR.search(l)]


with ThreadPoolExecutor(32) as ex:
    HL = dict(ex.map(one, W))

S = {"all": collections.Counter(), "trial": collections.Counter()}
examples = collections.defaultdict(list)
for p in W:
    rec = PM.get(PMID.get(p) or "")
    lines = HL.get(p)
    scope = ["all"] + (["trial"] if p in TRIAL else [])
    if not rec:
        for s in scope:
            S[s]["no PubMed record"] += 1
        continue
    if lines is None:
        for s in scope:
            S[s]["no PMC text"] += 1
        continue
    auths = [a for a in rec["authors"] if a.get("valid", "Y") == "Y"]
    persons = [a for a in auths if a.get("last")]
    groups = [a for a in auths if not a.get("last")]
    nl = [norm(l) for l in lines]
    found = 0
    for a in persons:
        sur = "".join(re.findall(r"[a-z]+", norm(a["last"])))
        ini = (re.findall(r"[a-z]", norm(a.get("fore") or a.get("initials") or "")) or [""])[0]
        hit = any(sur and sur in "".join(re.findall(r"[a-z]+", l)) and (not ini or re.search(r"\b" + ini, l)) for l in nl)
        found += hit
        if not hit:
            for s in scope:
                S[s]["PubMed authors not found in PMC header"] += 1
            if len(examples["not_found"]) < 15:
                examples["not_found"].append((p, a["last"], a.get("fore")))
    for s in scope:
        S[s]["articles"] += 1
        S[s]["PubMed person authors"] += len(persons)
        S[s]["PubMed collective (group) authors"] += len(groups)
        S[s]["PMC header contributor lines (non-editor)"] += len(lines)
        S[s]["articles: all PubMed authors found"] += found == len(persons)
        S[s]["articles: same count (persons + groups vs header lines)"] += len(lines) == len(auths)
        S[s]["articles: header has more lines"] += len(lines) > len(auths)
        S[s]["articles: header has fewer lines"] += len(lines) < len(auths)
    if len(lines) != len(auths) and len(examples["count_diff"]) < 12:
        examples["count_diff"].append((p, len(auths), len(lines), lines[:3]))
for s in S:
    print(f"\n== {s}")
    for k, v in S[s].items():
        base = S[s]["articles"] if k.startswith("articles:") else S[s]["PubMed person authors"] if k.startswith("PubMed authors not") else None
        print(f"  {k}: {v:,}" + (f" ({100 * v / base:.2f}%)" if base else ""))
print("\nnot found, examples:", examples["not_found"][:10])
print("\ncount differences, examples:")
for e in examples["count_diff"][:8]:
    print("  ", e[0], "PubMed", e[1], "header", e[2], [x[:70] for x in e[3]])
json.dump(dict(stats=S, examples=examples), open(OD + "pmc_vs_pubmed_authors.json", "w"), indent=1, default=dict)
