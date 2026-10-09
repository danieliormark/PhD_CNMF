"""§6 item 4 follow-up: option C (drop in curate) vs option D (keep, then strip the atoms from the finished structures)."""
import json, re, sys, collections
FUSED = re.compile(r"\s?\b(?:not_only|but_also)/M/en")
def strip(edge):
    e = FUSED.sub("", edge)
    e = re.sub(r"\(\s+", "(", e)
    e = re.sub(r"\s\((?:dummy_sibling|focal_he|sibling_he)\)", "", e)
    return e
def post_filter(rec):
    """option D on one unit: the set of parent and cousin edges with the fused atoms removed"""
    par, bad = set(), 0
    for p in rec["parents"]:
        e = strip(p["edge"])
        if "(dummy_sibling" not in e or "(focal_he" not in e:
            bad += 1; continue
        par.add(e)
    cou = set()
    for c in rec["cousins"]:
        e = strip(c["edge"])
        if re.fullmatch(r"\((?:dummy_cousin|cousin_he)\)", e):
            continue
        cou.add(e)
    return par, cou, bad
def main():
    shards = sys.argv[1:]
    c = collections.Counter(); ex = []
    for s in shards:
        K = [json.loads(l) for l in open(f"v0_{s}/g3_test_{int(s):03d}.jsonl")]
        D = [json.loads(l) for l in open(f"v1_{s}/g3_test_{int(s):03d}.jsonl")]
        assert [r["uid"] for r in K] == [r["uid"] for r in D]
        for k, d in zip(K, D):
            c["units"] += 1
            txt = " ".join(p["edge"] for p in k["parents"]) + " " + " ".join(x["edge"] for x in k["cousins"])
            has = bool(re.search(r"\b(?:not_only|but_also)/M/en", txt))
            c["units with a fused atom (keep)"] += has
            par, cou, bad = post_filter(k)
            c["parents invalid after strip"] += bad
            dp = {p["edge"] for p in d["parents"]}; dc = {x["edge"] for x in d["cousins"]}
            if par != dp or cou != dc:
                c["units where C != D"] += 1
                if len(ex) < 12: ex.append((k["uid"], k["text"][:150], sorted(par ^ dp)[:3], sorted(cou ^ dc)[:3]))
            elif has:
                c["units with a fused atom where C == D"] += 1
            if not has and ({p["edge"] for p in k["parents"]} != dp or {x["edge"] for x in k["cousins"]} != dc):
                c["units without a fused atom that still changed"] += 1
    print(dict(c))
    for e in ex:
        print("\n", e[0], "|", e[1]); print("   parents differ:", e[2]); print("   cousins differ:", e[3])


if __name__ == "__main__":
    main()
