# Per-term article counts (PMC open access, 2015-01..2026-09, body text and title/abstract) for the ALIAS and G4 terms of a
# retrieval_gap_query.py output: finds the terms that bring in homonyms (owner 2026-10-20, RL-155). Read-only esearch.
#   python3 retrieval_term_counts.py GAP.json OUT.tsv
import json, os, sys, time, urllib.parse, urllib.request
K = os.environ.get("NCBI_API_KEY", "")
def count(q):
    d = urllib.parse.urlencode(dict(db="pmc", term=q + ' AND "open access"[filter]', retmax=0, retmode="json", mindate="2015/01/01",
                                    maxdate="2026/09/30", datetype="pdat", **({"api_key": K} if K else {}))).encode()
    for a in range(5):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", data=d, method="POST"), timeout=120).read())
            time.sleep(0.11); return int(r["esearchresult"]["count"]), r["esearchresult"].get("querytranslation", "")
        except Exception:
            time.sleep(2 * (a + 1))
    return -1, ""
G = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    f.write("set\tterm\tbody\ttiab\tbody_translation\n")
    for s in ("ALIAS", "G4"):
        for i, t in enumerate(G["terms"][s]):
            b, tr = count(f'"{t}"[Body - All Words]'); a, _ = count(f'"{t}"[Title/Abstract]')
            f.write(f"{s}\t{t}\t{b}\t{a}\t{tr[:120]}\n")
            if i % 200 == 0: f.flush(); print(s, i, flush=True)
print("done")
