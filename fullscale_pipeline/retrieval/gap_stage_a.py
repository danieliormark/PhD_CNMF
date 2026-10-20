"""Stage A on the downloaded gap corpus (2026-10-20, ticket 107 P1, RL-157): the preprocessing the original corpus went
through, applied to the new files only, as counts — nothing is removed from any list or corpus. Owner: report the
numbers per step and per year first; removal is decided afterwards.

Each step runs the stage's own code (pmc_preprocessing/*.py, unchanged) with its two path roots P and LCS pointed at a
separate staging tree GR (`LCS/gap_stage/`), which mirrors the original layout:
    GR/llm_corpus_staging/   article_whitelist.txt, masked_corpus_v2/, clean_corpus_v1/, clean_corpus_v2/, P2 outputs
    GR/pmc_preprocessing/    content_regions_v2.csv
and the raw texts read from LCS/pure_text_corpus_gap/. The existing corpus outputs are not touched.
Steps (each refuses to overwrite its output, as the stage scripts do):
    cohort  the files of pure_text_corpus_gap/ with their type (gap eligibility caches) and a preprint flag (R1 rule)
    p0      content regions (build_content_regions.analyse), in parallel
    white   the staging whitelist: P0 include=1, eligible type, not a preprint
    p1      citation masking (citation_masking_v2.main)
    p1b     non-prose removal (nonprose_removal_v1.main)
    p1c     extra non-prose rules (nonprose_extra_rules_v1.main)
    p2      focal blocks with the v3 matcher (focal_extraction_v2.main, focal_terms → focal_terms_v3)
    python3 gap_stage_a.py STEP [--workers N]
"""
import csv, importlib.util, json, os, re, sys, collections
from multiprocessing import Pool

R = "/mnt/hum01-rds/Basov/p91688di/"
LCS = R + "llm_corpus_staging/"
PP = R + "pmc_preprocessing/"
RAW = LCS + "pure_text_corpus_gap/"
GR = LCS + "gap_stage/"
GL, GP = GR + "llm_corpus_staging/", GR + "pmc_preprocessing/"
ALLOWED = {"Research article (Journal Article tag, no more specific type)", "Historical article", "Review (narrative / other)",
           "Systematic review / meta-analysis / scoping review"}
sys.path.insert(0, PP)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m   # worker processes find the module by name
    spec.loader.exec_module(m); return m


def point(m):
    """the stage module m reads and writes in the staging tree, raw texts from the gap corpus"""
    m.P, m.LCS = GR, GL
    for attr in ("RAW", "RAW_DIR"):
        if hasattr(m, attr):
            setattr(m, attr, RAW)
    if hasattr(m, "CLEAN"):
        m.CLEAN = GL + "clean_corpus_v2/"
    return m


PRE_RX = re.compile(r"Article version:\s*preprint", re.I)


def _preprint(p):
    """R1 rule of article_blacklist.py: the text header says the article is a preprint"""
    return bool(PRE_RX.search(open(RAW + p + ".txt", encoding="utf-8", errors="replace").read(4000)))


def cohort():
    out = GR + "cohort.tsv"
    assert not os.path.exists(out), out
    os.makedirs(GL, exist_ok=True); os.makedirs(GP, exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".txt"))
    typ = {}
    for p in (LCS + "gap_eligibility_20261020_body.tsv", LCS + "gap_eligibility_corpus_ext_20261020.tsv",
              LCS + "gap_eligibility_tiab_rest_20261020.tsv"):
        if os.path.exists(p):
            for r in csv.DictReader(open(p), delimiter="\t"):
                typ.setdefault(r["pmcid"], r["primary_type"])
    src = open("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/corpus_statistics/article_types_valid.py").read()
    ns = {}; exec(src[src.index("ADMIN_TAGS ="):src.index("rows = []")], ns)
    S = json.load(open(LCS + "gap_esummary_cache.json")); PT = json.load(open(LCS + "gap_pubtypes_cache.json"))
    with Pool(24) as pool:                                       # the headers are read in parallel (RDS over the network)
        PRE = dict(zip(ids, pool.map(_preprint, ids, chunksize=200)))
    with open(out, "w") as f:
        f.write("pmcid\tyear\tprimary_type\ttype_eligible\tpreprint\n")
        for p in ids:
            t = typ.get(p)
            if t is None:
                pm = (S.get(p) or {}).get("pmid")
                if not pm: t = "type unknown (no PMID)" if p in S else "type unknown (not looked up)"
                elif PT.get(pm) is None: t = "type unknown (no PubMed record)"
                else:
                    names = [x["name"] for x in PT[pm]]; t = ns["classify"](names) if names else "type unknown (no PublicationType)"
            el = t in ALLOWED or t.startswith("type unknown")
            m = re.search(r"\b(19|20)\d\d\b", (S.get(p) or {}).get("pubdate") or (S.get(p) or {}).get("epubdate") or "")
            year = m.group(0) if m else "?"
            f.write(f"{p}\t{year}\t{t}\t{int(el)}\t{int(PRE[p])}\n")
    print("cohort", len(ids))


_p0 = None
def _p0_one(pid):
    try:
        return _p0.analyse(pid)
    except Exception as e:
        return dict(pmcid=pid, include=0, status="p0_error:" + type(e).__name__, regions_json="[]")


def p0(workers):
    global _p0
    out = GP + "content_regions_v2.csv"
    assert not os.path.exists(out), out
    _p0 = point(load("p0", PP + "build_content_regions.py"))
    ids = [r["pmcid"] for r in csv.DictReader(open(GR + "cohort.tsv"), delimiter="\t")]
    with Pool(workers) as pool:
        rows = list(pool.imap(_p0_one, ids, chunksize=50))
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("p0 rows", len(rows), collections.Counter(str(r.get("include")) for r in rows))


def white():
    out = GL + "article_whitelist.txt"
    assert not os.path.exists(out), out
    C = {r["pmcid"]: r for r in csv.DictReader(open(GR + "cohort.tsv"), delimiter="\t")}
    inc = {r["pmcid"] for r in csv.DictReader(open(GP + "content_regions_v2.csv")) if str(r.get("include")) == "1"}
    wl = sorted(p for p, r in C.items() if p in inc and r["type_eligible"] == "1" and r["preprint"] == "0")
    open(out, "w").write("\n".join(wl) + "\n")
    print("whitelist", len(wl), "of", len(C))


def run_main(path, name, argv, swap_focal=False):
    if swap_focal:
        load("focal_terms", PP + "focal_terms.py")                     # v3 imports v2 under this name
        sys.modules["focal_terms"] = load("focal_terms_v3", PP + "focal_terms_v3.py")
    m = point(load(name, path))
    sys.argv = [path] + argv
    m.main()


if __name__ == "__main__":
    step = sys.argv[1]
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    if step == "cohort": cohort()
    elif step == "p0": p0(w)
    elif step == "white": white()
    elif step == "p1": run_main(PP + "citation_masking_v2.py", "p1", ["--outdir", GL, "--workers", str(w)])
    elif step == "p1b": run_main(PP + "nonprose_removal_v1.py", "p1b", ["--outdir", GL, "--workers", str(w)])
    elif step == "p1c": run_main(PP + "nonprose_extra_rules_v1.py", "p1c", ["--outdir", GL, "--workers", str(w)])
    elif step == "p2": run_main(PP + "focal_extraction_v2.py", "p2", ["--outdir", GL, "--workers", str(w)], swap_focal=True)
    else: sys.exit("unknown step " + step)
