import json, collections, re, unicodedata
import numpy as np

D = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/openalex_authors/"
W = json.load(open(D + "works_metrics.json"))
A = json.load(open(D + "author_hindex.json"))
print("works", len(W), "authors with h-index", len(A))


def size_bin(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else "11+"


rows = []
for wid, w in W.items():
    n = len(w["authors"]); miss = [a for a in w["authors"] if not a["id"]]
    hs = [A[a["id"]]["h"] for a in w["authors"] if a["id"] in A and A[a["id"]]["h"] is not None]
    rows.append(dict(wid=wid, n=n, bin=size_bin(n), affected=bool(miss), n_miss=len(miss),
                     miss_pos={a["pos"] for a in miss}, pct=w["pct"], fwci=w["fwci"], cited=w["cited"],
                     year=w["year"], field=w["field"], max_h=max(hs) if hs else None,
                     med_h=float(np.median(hs)) if hs else None))

def summ(rs):
    pct = [r["pct"] for r in rs if r["pct"] is not None]; fw = [r["fwci"] for r in rs if r["fwci"] is not None]
    mh = [r["max_h"] for r in rs if r["max_h"] is not None]
    return dict(n=len(rs), pct_median=np.median(pct) if pct else None, top10=np.mean([p >= 0.9 for p in pct]) if pct else None,
                top1=np.mean([p >= 0.99 for p in pct]) if pct else None, fwci_median=np.median(fw) if fw else None,
                cited_median=np.median([r["cited"] for r in rs]), team_max_h_median=np.median(mh) if mh else None)

print("\n=== Paper influence: affected (>=1 author without ID) vs fully identified ===")
for label, rs in [("ALL", rows)] + [(b, [r for r in rows if r["bin"] == b]) for b in ["1", "2-3", "4-6", "7-10", "11+"]]:
    a, u = [r for r in rs if r["affected"]], [r for r in rs if not r["affected"]]
    sa, su = summ(a), summ(u)
    print(f"team size {label:5s} | affected n={sa['n']:5d} pct-med {sa['pct_median']:.3f} top10% {sa['top10']:.3f} top1% {sa['top1']:.4f} "
          f"FWCI-med {sa['fwci_median']:.2f} cites-med {sa['cited_median']:.0f} team-max-h-med {sa['team_max_h_median']}"
          f" || identified n={su['n']:5d} pct-med {su['pct_median']:.3f} top10% {su['top10']:.3f} top1% {su['top1']:.4f} "
          f"FWCI-med {su['fwci_median']:.2f} cites-med {su['cited_median']:.0f} team-max-h-med {su['team_max_h_median']}")

print("\n=== Position of the unidentified bylines ===")
pos_miss = collections.Counter(a["pos"] for w in W.values() for a in w["authors"] if not a["id"])
pos_all = collections.Counter(a["pos"] for w in W.values() for a in w["authors"])
for p in ["first", "middle", "last"]:
    print(f"  {p:6s}: {pos_miss[p]:6d} of unidentified ({pos_miss[p]/sum(pos_miss.values()):.1%})  vs  all bylines {pos_all[p]/sum(pos_all.values()):.1%}")
lead = [r for r in rows if r["affected"] and (r["miss_pos"] & {"first", "last"})]
print(f"  affected papers missing their first or last author: {len(lead)} ({len(lead)/len(rows):.1%} of all papers)")

def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()
key = lambda s: (" ".join(norm(s)[:1] + norm(s)[-1:])) if norm(s) else ""   # first + last token
ided = collections.defaultdict(set)
for w in W.values():
    for a in w["authors"]:
        if a["id"]:
            ided[key(a["raw"])].add(a["id"])
miss = [a for w in W.values() for a in w["authors"] if not a["id"]]
hit = [a for a in miss if key(a["raw"]) in ided]
uniq = [a for a in hit if len(ided[key(a["raw"])]) == 1]
print(f"\n=== Is the person present elsewhere in the corpus with an ID? (same first+last name) ===")
print(f"  unidentified bylines {len(miss)}: name seen with an ID elsewhere {len(hit)} ({len(hit)/len(miss):.1%}); "
      f"of those, the name maps to exactly one ID {len(uniq)} ({len(uniq)/len(miss):.1%})")
hs_present = [A[next(iter(ided[key(a['raw'])]))]["h"] for a in uniq if next(iter(ided[key(a['raw'])])) in A]
print(f"  h-index of those single matching IDs: median {np.median(hs_present):.0f}, share h>=20 {np.mean([h >= 20 for h in hs_present]):.1%}")
all_h = [v["h"] for v in A.values() if v["h"] is not None]
print(f"  h-index of all identified authors: median {np.median(all_h):.0f}, share h>=20 {np.mean([h >= 20 for h in all_h]):.1%}")

print("\n=== Most-cited affected papers ===")
for r in sorted([r for r in rows if r["affected"]], key=lambda r: -(r["cited"] or 0))[:10]:
    m = [a for a in W[r["wid"]]["authors"] if not a["id"]]
    print(f"  {r['wid']} cited {r['cited']:5d} pct {r['pct']} | {r['n_miss']}/{r['n']} unidentified: "
          + ", ".join(f"{a['raw']} ({a['pos']})" for a in m[:3]))
print("\n=== Share affected by field (fields with >=500 papers) ===")
fc = collections.Counter(r["field"] for r in rows); fa = collections.Counter(r["field"] for r in rows if r["affected"])
for f, n in fc.most_common():
    if n >= 500:
        print(f"  {f}: {fa[f]/n:.1%} of {n}")
print("\n=== Share affected by publication year ===")
yc = collections.Counter(r["year"] for r in rows); ya = collections.Counter(r["year"] for r in rows if r["affected"])
print("  " + ", ".join(f"{y}: {ya[y]/yc[y]:.1%} (n={yc[y]})" for y in sorted(k for k in yc if k)))
