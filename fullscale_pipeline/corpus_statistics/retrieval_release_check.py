# Release-date check (owner 2026-10-20, RL-155): for each kept search term, PMC open-access body-text hits 2015-2018 (before
# BERT, released late 2018, and every later model existed) and 2019-2026/09. A term with hits before 2019 collides with
# other words or names (PMC search ignores case). Read-only esearch.
#   python3 retrieval_release_check.py KEPT_TERMS.json OUT.tsv
import json, os, sys, time, urllib.parse, urllib.request
K = os.environ.get("NCBI_API_KEY", "")
def count(t, a, b):
    d = urllib.parse.urlencode(dict(db="pmc", term=f'"{t}"[Body - All Words] AND "open access"[filter]', retmax=0, retmode="json",
                                    mindate=a, maxdate=b, datetype="pdat", **({"api_key": K} if K else {}))).encode()
    for i in range(5):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", data=d, method="POST"), timeout=120).read())
            time.sleep(0.11); return int(r["esearchresult"]["count"])
        except Exception:
            time.sleep(2 * (i + 1))
    return -1
T = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    f.write("term\tbefore_2019\tfrom_2019\n")
    for i, t in enumerate(T):
        f.write(f"{t}\t{count(t, '2015/01/01', '2018/12/31')}\t{count(t, '2019/01/01', '2026/09/30')}\n")
        if i % 200 == 0: f.flush(); print(i, flush=True)
print("done")
