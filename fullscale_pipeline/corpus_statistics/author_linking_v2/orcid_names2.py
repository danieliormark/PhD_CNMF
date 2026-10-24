exec(open("links_v2.py").read().split("U = [b for b in B if not b[\"person\"]]")[0])
import random, itertools
TR = str.maketrans({"æ": "ae", "ø": "o", "ı": "i", "ß": "ss", "đ": "d", "ł": "l", "œ": "oe", "Æ": "ae", "Ø": "o", "Ł": "l"})
def tk(z): return re.findall(r"[a-z]+", norm((z or "").translate(TR)))
def compatible(x, y):
    lx, ly = "".join(tk(x["last"])), "".join(tk(y["last"]))
    fx, fy = tk(x["fore"]), tk(y["fore"])
    allx, ally = "".join(tk(x["last"]) + fx), "".join(tk(y["last"]) + fy)
    sur = lx == ly or lx in ally or ly in allx
    if not sur: return False
    if not fx or not fy: return True
    ix, iy = "".join(t[0] for t in fx), "".join(t[0] for t in fy)
    only_init = lambda f: all(len(t) == 1 for t in f) or (len(f) > 1 and len(f[0]) > 1 and all(len(t) == 1 for t in f[1:]) and False)
    if fx[0] == fy[0] or "".join(fx) == "".join(fy): return True
    if (only_init(fx) or only_init(fy)) and (ix.startswith(iy) or iy.startswith(ix)): return True
    return bool(set(fx) & set(fy + tk(y["last"]))) or bool(set(fy) & set(fx + tk(x["last"])))
o2b = collections.defaultdict(list)
for b in B:
    o = b["pm_orcid"] or b["oa_orcid"]
    if o: o2b[o].append(b)
C = collections.Counter(); conf = []
for o, bs in o2b.items():
    groups = []
    for b in bs:
        for g in groups:
            if compatible(g[0], b): g.append(b); break
        else: groups.append([b])
    if len(groups) == 1: C["one name (variants allowed)"] += 1
    else:
        C["2+ incompatible names"] += 1; same_art = any(x["pmcid"] == y["pmcid"] for g, h in itertools.combinations(groups, 2) for x in g for y in h)
        C["  of which on the same article"] += same_art
        C["  bylines in the minority name(s)"] += sum(len(g) for g in sorted(groups, key=len)[:-1])
        conf.append((o, [(f"{g[0]['last']}, {g[0]['fore']}", len(g), g[0]["pmcid"], g[0]["afftext"][:50]) for g in groups], same_art))
print(dict(C))
random.seed(77)
for x in random.sample(conf, 16): print(" ", x)
