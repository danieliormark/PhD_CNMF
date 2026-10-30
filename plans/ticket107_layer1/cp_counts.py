exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"w_stats.py")).read().split('print("articles"')[0])
nest=set(); cous=set(); both=0
for r in recs:
    ps=[p["content_key"] for p in r["parents"]]
    for p in r["parents"]:
        for c in p["children"]: nest.add((c["content_key"],p["content_key"]))
    for c in r["cousins"]:
        for pk in ps: cous.add((c["content_key"],pk))
ov=nest&cous
print("nested ties",len(nest),"cousin ties",len(cous),"pairs that are both",len(ov))
for v in (0.5,0.5**0.5):
    m=len(nest)+v*v*len(cous-nest); print(f"cousin value {v:.2f}: cousin share of squared mass {(v*v*len(cous-nest))/m:.3f}")
