exec(open("links_v2.py").read().split("U = [b for b in B if not b[\"person\"]]")[0])
import random, itertools
TR = str.maketrans({"æ": "ae", "ø": "o", "ı": "i", "ß": "ss", "đ": "d", "ł": "l", "œ": "oe", "Æ": "ae", "Ø": "o", "Ł": "l"})
# German umlauts: a name is compared in two forms, umlauts dropped (ü -> u) and written out (ü -> ue); two names agree if
# they agree in either form. An "ue" already in a name is never shortened (Xue is not Xu, Jae is not Ja).
UMLAUT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "ae", "Ö": "oe", "Ü": "ue"})
TK_MODE = ["drop"]
def tk(z):
    z = (z or "").translate(TR)
    if TK_MODE[0] == "spell": z = z.translate(UMLAUT)
    return re.findall(r"[a-z]+", norm(z))
def _compatible(x, y):
    if sorted(tk(x["last"]) + tk(x["fore"])) == sorted(tk(y["last"]) + tk(y["fore"])): return True   # same parts, other split
    lx, ly = "".join(tk(x["last"])), "".join(tk(y["last"]))
    fx, fy = tk(x["fore"]), tk(y["fore"])
    allx, ally = "".join(tk(x["last"]) + fx), "".join(tk(y["last"]) + fy)
    # a surname may sit inside the other's full name (particles or compound parts written elsewhere: "Momi, Elena De" /
    # "Demomi, Elena"), but only if it has 4+ letters and is at least 60% of the longer surname, so "Li" is never "Liu"
    inside = lambda s, full, other: len(s) >= 4 and len(s) >= 0.6 * max(len(s), len(other)) and s in full
    sur = lx == ly or inside(lx, ally, ly) or inside(ly, allx, lx)
    # a compound surname written in full or in part, if the given names are identical ("Leon, Leticia" /
    # "Leon-Mateos, Leticia"; "Canillas, Fernando" / "Canillas-Del Rey, Fernando")
    px, py = tk(x["last"]), tk(y["last"])
    if not sur and tk(x["fore"]) and tk(x["fore"]) == tk(y["fore"]) and min(len(px), len(py)) >= 1 and (set(px) < set(py) or set(py) < set(px)):
        sur = True
    if not sur: return False
    if not fx or not fy: return True
    if "".join(fx) == "".join(fy): return True                    # Zheng-Hao / Zhenghao, An'an / Anan
    # given names part by part from the left: equal, or one part is the other's initial; one may have fewer parts
    # (Andrew J E / Andrew John Ervine; C Raina / Chandini Raina; A M B / Ana M B); Miji / Minjae and Wei / Wen Wei fail
    return all(p == q or (len(p) == 1 and q.startswith(p)) or (len(q) == 1 and p.startswith(q)) for p, q in zip(fx, fy))
def compatible(x, y):
    for mode in ("drop", "spell"):
        TK_MODE[0] = mode
        if _compatible(x, y): TK_MODE[0] = "drop"; return True
    TK_MODE[0] = "drop"; return False
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
