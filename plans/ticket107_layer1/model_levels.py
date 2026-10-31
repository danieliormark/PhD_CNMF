# Sparsity of the child-level facet under three model-name levels (tier as G3 writes it, version, family),
# with numbers left out of hyperedge identity (decided 2026-10-23). Psychology trial.
import collections, re, os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w_stats.py")).read().split('print("articles"')[0])
TIERS = {"sonnet","opus","haiku","pro","flash","lite","mini","nano","turbo","ultra","max"}
GENERIC = {"llm","language_model","transformer_model"}
NUM = re.compile(r"^[\d.,%]+$")
def level(root, lev):
    if lev == "tier" or root in GENERIC: return root
    seg = root.split("_")
    if lev == "version": return "_".join(s for s in seg if s not in TIERS)
    fam = [s for s in seg if s not in TIERS and not re.search(r"\d", s) and s not in {"v"}]
    f = "_".join(fam) or seg[0]
    return {"chatgpt": "gpt", "gpts": "gpt"}.get(f, f)
def key(atoms, lev):
    out = set()
    for a in atoms:
        root, rest = a["atom"].split("/", 1)
        if NUM.match(root): continue
        out.add((level(root, lev) if rest == "C/focal" else root) + "/" + rest)
    return frozenset(out)
units = []   # (pmcid, [(child atoms, parent id or None)])
for r in recs:
    us = [(c["atoms"], p["id"]) for p in r["parents"] for c in p["children"]]
    us += [(c["atoms"], pid) for c in r["cousins"] for pid in (c.get("parents") or [None])]
    units.append((r["pmcid"], us))
N = len({p for p, _ in units})
print("articles", N)
for lev in ("tier", "version", "family"):
    arts = collections.defaultdict(set); par = set(); named = set(); foc = set()
    for pm, us in units:
        for atoms, pid in us:
            k = key(atoms, lev)
            if not k: continue
            arts[k].add(pm)
            if pid: par.add((k, pid))
            m = {a.split("/")[0] for a in k if a.endswith("/C/focal")}
            foc |= m
            if m - GENERIC: named.add(k)
    H = len(arts); ties = sum(len(v) for v in arts.values()); single = sum(1 for v in arts.values() if len(v) == 1)
    elig = sum(len(v) for v in arts.values() if len(v) >= 2)
    nsingle = sum(1 for k in named if len(arts[k]) == 1)
    print(f"{lev:8s} focal atoms {len(foc):4d} | child hyperedges {H:6d} | single-article {single/H:.3f} | child×article ties {ties} "
          f"density {ties/(H*N):.2e} | ties of hyperedges in 2+ articles {elig} | child×parent ties {len(par)} | "
          f"hyperedges with a named model {len(named)}, single-article {nsingle/len(named):.3f}")
