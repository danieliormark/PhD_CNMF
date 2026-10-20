import json, os, time, urllib.parse, urllib.request, sys
K = os.environ.get("NCBI_API_KEY", "")
R = json.load(open("retrieval_gap_body.json"))
out = {}
def count(t, field):
    d = urllib.parse.urlencode(dict(db="pmc", term=f'"{t}"[{field}] AND "open access"[filter]', retmax=0, retmode="json",
                                    api_key=K, mindate="2015/01/01", maxdate="2026/09/30", datetype="pdat")).encode()
    for a in range(4):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", data=d, method="POST"), timeout=60).read())
            time.sleep(0.11); return int(r["esearchresult"]["count"]), r["esearchresult"].get("querytranslation", "")
        except Exception:
            time.sleep(2 ** a)
    return -1, ""
for s, terms in R["terms"].items():
    for t in terms:
        out[f"{s}\t{t}"] = count(t, "Body - All Words")
json.dump(out, open("term_counts_body.json", "w"))
print("done", len(out))
