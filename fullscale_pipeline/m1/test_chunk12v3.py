# Prerequisite checks of chunk12v3 (ticket 107): synthetic inputs whose cells are computed by hand, and the two W
# algorithms against a naive all-pairs computation. Run: python test_chunk12v3.py  (no real data, a few seconds).
import collections, csv, json, math, os, random, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk12v3 as B
import model_name_map as mnm

ok = 0
def check(name, got, want, tol=1e-9):
    global ok
    if isinstance(want, (int, float)):
        assert abs(got - want) <= tol, f"{name}: got {got}, want {want}"
    else:
        assert got == want, f"{name}: got {got}, want {want}"
    ok += 1

def at(a): return {"atom": a, "src": []}
def rec(pm, uid, parents, cousins):
    """parents: [(id, [[atom, ...], ...])]; cousins: [([atom, ...], [parent ids])]"""
    return dict(pmcid=pm, uid=uid,
                parents=[dict(id=pid, children=[dict(atoms=[at(a) for a in kid]) for kid in kids]) for pid, kids in parents],
                cousins=[dict(atoms=[at(a) for a in ats], parents=ps) for ats, ps in cousins])

# ---------------------------------------------------------------- semantic side
mnm.NOT_MODEL.add(("A2", "grover"))
recs = [
    rec("A1", "A1.u1", [("A1.u1.P1", [["use/P/en"], ["gpt_4/C/focal"], ["model/C/en", "3/C/en"]])], [(["help/P/en"], ["A1.u1.P1"])]),
    rec("A1", "A1.u2", [("A1.u2.P1", [["use/P/en"], ["chatgpt/C/focal"]])], []),
    rec("A2", "A2.u1", [("A2.u1.P1", [["use/P/en"], ["bert28/C/focal"]])], [(["use/P/en"], ["A2.u1.P1"])]),
    rec("A2", "A2.u2", [("A2.u2.P1", [["grover/C/focal"], ["show/P/en"]])], []),   # Grover as an author: unit removed
    rec("A2", "A2.u3", [], [(["show/P/en"], ["A2.u2.P1"])]),     # its sentence's other unit: no parent left -> removed
    rec("A1", "A1.u3", [], [(["help/P/en"], ["A1.u2.P1"])]),     # no parent of its own, cousin of A1.u2's parent: kept
]
log = {}
units = B.semantic_units(recs, log)
check("unit with only a false-positive focal atom is removed", log["units_dropped"],
      [("A2.u2", "no focal atom left"), ("A2.u3", "no parent hyperedge left in its sentence")])
sem = B.build_semantic(units)
K = {frozenset(s): k for k, s in sem["child_sets"].items()}
ci = {k: i for i, k in enumerate(sem["children"])}
col = lambda *atoms: ci[K[frozenset(atoms)]]
check("numbers left out of identity", frozenset({"model/C/en"}) in K, True)
check("glued citation mapped (bert28 -> bert)", frozenset({"bert/C/focal"}) in K, True)
check("show/P/en only in the removed unit", frozenset({"show/P/en"}) in K, False)
check("child hyperedges", len(sem["children"]), 6)
# child×article: A1 tf use 2, help 2 (cousin in u1 and u3), others 1; A2 use 2 (nested + cousin), bert 1; N = 2
idf_use, idf_one = math.log(2 / 3) + 1, 1.0
raw_A1 = {"use": math.log(3) * idf_use, "x": math.log(2) * idf_one, "help": math.log(3) * idf_one}
nA1 = math.sqrt(raw_A1["use"] ** 2 + 3 * raw_A1["x"] ** 2 + raw_A1["help"] ** 2)
CA = sem["CA"].toarray(); a1 = sem["articles"].index("A1"); a2 = sem["articles"].index("A2")
check("child×article A1 use", CA[a1, col("use/P/en")], raw_A1["use"] / math.sqrt(nA1))
check("child×article A1 help (two cousins)", CA[a1, col("help/P/en")], raw_A1["help"] / math.sqrt(nA1))
nA2 = math.sqrt((math.log(3) * idf_use) ** 2 + (math.log(2) * idf_one) ** 2)
check("child×article A2 bert", CA[a2, col("bert/C/focal")], math.log(2) / math.sqrt(nA2))
check("article influence = ||raw row|| after partial normalisation", float((CA[a1] ** 2).sum()), nA1)
check("idf_article of use", sem["idf_article"][col("use/P/en")], idf_use)
# child×parent: 3 parent hyperedges; use in 3 (df 3), others 1
P = 3; idfp_use, idfp_one, idfp_help = math.log(P / 4) + 1, math.log(P / 2) + 1, math.log(P / 3) + 1
PC = sem["PC"].toarray()
p1 = sem["parents"].index(B.content_key(B.content_key(s) for s in [{"use/P/en"}, {"gpt_4/C/focal"}, {"model/C/en"}]))
p3 = sem["parents"].index(B.content_key(B.content_key(s) for s in [{"use/P/en"}, {"bert/C/focal"}]))
r1 = {"use": idfp_use * math.log(2), "gpt": idfp_one * math.log(2), "model": idfp_one * math.log(2),
      "help": idfp_help * math.log(1 + B.COUSIN)}
n1 = math.sqrt(sum(v * v for v in r1.values()))
check("child×parent nested", PC[p1, col("gpt_4/C/focal")], r1["gpt"] / math.sqrt(n1))
check("child×parent cousin", PC[p1, col("help/P/en")], r1["help"] / math.sqrt(n1))
check("a lone cousin tie has half the influence of a lone nested tie", (math.log(1 + B.COUSIN) / math.log(2)) ** 2, 0.5)
p2 = sem["parents"].index(B.content_key(B.content_key(s) for s in [{"use/P/en"}, {"chatgpt/C/focal"}]))
check("cousin of a parentless unit ties to its sentence's parent", bool(PC[p2, col("help/P/en")] > 0), True)
r3 = {"use": idfp_use * math.log(2 + B.COUSIN), "bert": idfp_one * math.log(2)}
n3 = math.sqrt(sum(v * v for v in r3.values()))
check("child×parent nested + cousin in one cell", PC[p3, col("use/P/en")], r3["use"] / math.sqrt(n3))
check("idf_parent of use", sem["idf_parent"][col("use/P/en")], idfp_use)

# ---------------------------------------------------------------- W: soft similarity by hand
sets = [frozenset(x) for x in (
    {"gpt_4/C/focal", "answer/C/en"}, {"gpt_4o/C/focal", "answer/C/en"}, {"answer/C/en", "question/C/en"},
    {"bert/C/focal", "classify/P/en"}, {"mentalbert/C/focal", "classify/P/en"}, {"ref001234/C/en", "classify/P/en"},
    {"question/C/en", "patient/C/en"}, {"llm/C/focal", "answer/C/en"})]
ws = B.WordSpace(sets)
H = len(sets); idf = lambda a: math.log(H / sum(a in s for s in sets))
check("citation token is not a word of W", "ref001234/C/en" in ws.idf, False)
o = idf("answer/C/en") + 0.8 * min(idf("gpt_4/C/focal"), idf("gpt_4o/C/focal"))
w0 = idf("gpt_4/C/focal") + idf("answer/C/en"); w1 = idf("gpt_4o/C/focal") + idf("answer/C/en")
check("soft Jaccard gpt_4 vs gpt_4o (0.8)", ws.sim(0, 1), o / (w0 + w1 - o))
o = idf("classify/P/en") + 0.5 * min(idf("bert/C/focal"), idf("mentalbert/C/focal"))
w3 = idf("bert/C/focal") + idf("classify/P/en"); w4 = idf("mentalbert/C/focal") + idf("classify/P/en")
check("soft Jaccard bert vs mentalbert (0.5)", ws.sim(3, 4), o / (w3 + w4 - o))
check("generic llm gets no soft match", ws.sim(0, 7), idf("answer/C/en") / (w0 + idf("llm/C/focal") + idf("answer/C/en") - idf("answer/C/en")))
check("set with citation token: weight excludes it", ws.weight[5], idf("classify/P/en"))
# assignment: two names each, best one-to-one pairing
s2 = [frozenset({"gpt_4/C/focal", "bert/C/focal"}), frozenset({"gpt_4o/C/focal", "mentalbert/C/focal"}),
      frozenset({"x/C/en"}), frozenset({"x/C/en", "gpt_4/C/focal"})]
ws2 = B.WordSpace(s2)
i2 = lambda a: math.log(4 / sum(a in s for s in s2))
o = 0.8 * min(i2("gpt_4/C/focal"), i2("gpt_4o/C/focal")) + 0.5 * min(i2("bert/C/focal"), i2("mentalbert/C/focal"))
check("one-to-one soft matching of two names each", ws2.overlap(0, 1), o)

# ---------------------------------------------------------------- W: brute force and prefix filter vs naive
def naive(ws):
    c = collections.defaultdict(dict)
    for i in range(ws.H):
        for j in range(ws.H):
            if i != j:
                s = ws.sim(i, j)
                if s >= B.TAU: c[i][j] = s
    return B.select_ties(c)

random.seed(7)
plain = [f"w{n}/C/en" for n in range(120)]
models = ["gpt_4/C/focal", "gpt_4o/C/focal", "gpt_4o_mini/C/focal", "gpt/C/focal", "chatgpt/C/focal", "bert/C/focal",
          "mentalbert/C/focal", "roberta/C/focal", "llama_3/C/focal", "mentallama/C/focal", "llm/C/focal", "claude_3_5/C/focal"]
zipf = [1 / (r + 1) for r in range(len(plain))]
for trial in range(3):
    S = []
    for _ in range(700):
        n = random.choice([1, 1, 2, 2, 2, 3, 4])
        s = set(random.choices(plain, weights=zipf, k=n))
        if random.random() < 0.45: s |= set(random.sample(models, random.choice([1, 1, 1, 2])))
        if random.random() < 0.05: s.add("ref000001/C/en")
        S.append(frozenset(s))
    wsx = B.WordSpace(S)
    Wn = naive(wsx); Wb, _ = B.w_brute_force(wsx, block=64); Wp, _ = B.w_prefix_filter(wsx)
    check(f"W brute force = naive (random set {trial}, {len(Wn)} ties)", Wb, Wn)
    check(f"W prefix filter = naive (random set {trial})", Wp, Wn)
    check(f"similarity symmetric to the last bit (random set {trial})",
          all(wsx.sim(i, j) == wsx.sim(j, i) for i in range(0, 700, 7) for j in range(700)), True)

# ---------------------------------------------------------------- social side by hand
tmp = tempfile.mkdtemp()
rows = [  # article, person, institutions
    ("A1", "A:p", ["I/X"]), ("A1", "A:q", []),
    ("A2", "A:p", ["I/X", "I/Y"]), ("A2", "A:r", ["I/Y"]), ("A2", "U:s", []),
    ("A3", "A:q", [])]
with open(os.path.join(tmp, "auth.jsonl"), "w") as f:
    for n, (a, p, ins) in enumerate(rows): f.write(json.dumps(dict(pmcid=a, pos=n, status="kept", person=p, institutions=ins)) + "\n")
    f.write(json.dumps(dict(pmcid="A3", pos=9, status="removed_R6", person="A:z", institutions=["I/X"])) + "\n")
with open(os.path.join(tmp, "org.csv"), "w") as f:
    f.write("institution,organisation\nX,ORGX\nY,ORGY\n")
with open(os.path.join(tmp, "jn.csv"), "w") as f:
    f.write("pmcid,nlm_id\nA1,J1\nA2,J1\nA3,\n")
soc = B.build_social(["A1", "A2", "A3"], os.path.join(tmp, "auth.jsonl"), os.path.join(tmp, "org.csv"), os.path.join(tmp, "jn.csv"))
pi = {p: i for i, p in enumerate(soc["persons"])}; oi = {o: i for i, o in enumerate(soc["organisations"])}
check("removed bylines are left out", "A:z" in pi, False)
check("P1 1/sqrt(team size)", soc["P1"][pi["A:r"], 1], 1 / math.sqrt(3))
P2 = soc["P2"].toarray()
check("P2 person p: ORGX 1 + 1/2, ORGY 1/2, row norm 1", P2[pi["A:p"], oi["ORGX"]], 1.5 / math.sqrt(2.5))
check("P2 person q (no organisation) is a missing row", pi["A:q"] in soc["P2_missing_rows"], True)
P3 = soc["P3"].toarray()
check("P3 article A2: ORGX 1/2, ORGY 1/2 + 1", P3[1, oi["ORGY"]], 1.5 / math.sqrt(2.5))
check("P3 article A3 (no organisation) is a missing row", list(soc["P3_missing_rows"]), [2])
check("P4 binary", soc["P4"][0, 0], 1.0)
check("P4 article without journal is a missing row", list(soc["P4_missing_rows"]), [2])
w12, w13 = 1 / 4, 1 / 2        # p: 1/(2-1) over |{p,q,r,s}|; q: 1/(2-1) over |{p,q}|
s1, s2_, s3 = math.sqrt(w12 ** 2 + w13 ** 2), w12, w13
P5 = soc["P5"].toarray()
check("P5 option (c), A1-A2", P5[0, 1], w12 / (s1 * s2_) ** 0.25)
check("P5 option (c), A1-A3", P5[0, 2], w13 / (s1 * s3) ** 0.25)
check("P5 symmetric", P5[2, 0], P5[0, 2])
check("P5 diagonal empty (masked in the fit)", float(np.abs(np.diag(P5)).sum()), 0.0)
check("Frobenius over observed cells skips missing rows", B.frobenius_observed(soc["P3"], soc["P3_missing_rows"]),
      math.sqrt(float((P3[:2] ** 2).sum())))
print(f"all {ok} checks passed")
