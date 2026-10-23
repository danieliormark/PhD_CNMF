"""Run the G3 v2 test script on the constructed hard-case sets of G3_POSTPROCESSING.md §6 (parsed with graphbrain as G2 does).

    python run_hard.py SET.json [SET.json ...] [--script PATH] [--show]

Case formats (one set may mix them):
  expect / forbid       strings that must / must not appear in the parents or cousins (items 4 and 1; a case with a "for" list
                        is used only if it names "q4full")
  need / avoid          the same, item 8 participle cases
  together              item 8 cousin cases: every cousin holding one of these words is one cousin holding all of them
  not_only / but_also   item 4 option 2: is the atom expected in the output (true) or not (false)
  (no expectation)      item 5 particle cases: printed, and compared with the item-5 reference copy if --ref is given
  units + d6            D6 / item 11 sentence cases (RL-170): "units" = the texts of one sentence's G2 units (each parsed on its
                        own, as G2 parses the pieces cut at a glued citation); "d6" = parents (count), focal_he (count per parent),
                        embedded ([inner, outer] parent numbers, 1-based over the sentence), cousin_has / cousin_lacks (strings),
                        cousin_units (units holding a cousin), folded (cousins folded into an earlier unit); every record must
                        also pass the script's own check_record against the sentence's parents
Parses (G2's parse_piece, RL-151) are cached in hard/parsed_cache_g2.json (the parser is deterministic for a given text), so reruns need no parser.
"""
import argparse, hashlib, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "hard", "parsed_cache_g2.json")     # RL-151; parsed_cache.json = graphbrain default cuts (before)
G2_SCRIPT = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/graphbrain_parse_v3.py"


def load(path, name="g3v2"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse_all(texts):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [t for t in texts if t not in cache]
    if todo:
        from graphbrain.parsers import create_parser
        parser = create_parser(lang="en", lemmas=True)
        g2 = load(G2_SCRIPT, "g2")
        for t in todo:
            p = g2.parse_piece(parser, t)[0][0]        # RL-151: G2's own sentence-cut rule, as in the pipeline
            cache[t] = dict(main_edge=str(p["main_edge"]), extra_edges=[str(x) for x in p.get("extra_edges", [])],
                            atom2word=sorted(([str(a), w, i] for a, (w, i) in p.get("atom2word", {}).items()), key=lambda x: x[2]))
        json.dump(cache, open(CACHE, "w"), indent=0, ensure_ascii=False)
    return cache


def unit_of(text, parsed):
    uid = "PMCX.r1.L1.S1.U1"
    return dict(uid=uid, hash=hashlib.sha1(f"{uid}|{text}".encode()).hexdigest()[:12], text=text, **parsed)


def run(m, u):
    rec, _ = m.curate_unit("PMCX", dict(sid="PMCX.r1.L1.S1", hash_final="0" * 12), u, m.focal_terms.Matcher(), True, "keep")
    return rec


def run_sentence(m, texts, parsed):
    """D6 / item 11: the units of one sentence, curated and linked as in the script's run()"""
    us, recs = [], []
    for k, t in enumerate(texts, 1):
        uid = f"PMCX.r1.L1.S1.U{k}"
        u = dict(uid=uid, hash=hashlib.sha1(f"{uid}|{t}".encode()).hexdigest()[:12], text=t, **parsed[t])
        rec, _ = m.curate_unit("PMCX", dict(sid="PMCX.r1.L1.S1", hash_final="0" * 12), u, m.focal_terms.Matcher(), True, "keep")
        us.append(u); recs.append(rec)
    folded = m.link_sentence(recs)
    return us, recs, folded


def judge_d6(m, d, us, recs, folded):
    pars = [p for r in recs for p in r["parents"]]
    num = {p["id"]: i for i, p in enumerate(pars, 1)}
    cou = [c["edge"] for r in recs for c in r["cousins"]]
    ok = all(not m.check_record(r, u, list(num)) for u, r in zip(us, recs) if r["parents"] or (num and r["cousins"]))
    ok &= d.get("parents", len(pars)) == len(pars)
    if "focal_he" in d:
        ok &= d["focal_he"] == [sum(c["kind"] == "focal_he" for c in p["children"]) for p in pars]
    if "embedded" in d:
        ok &= sorted(d["embedded"]) == sorted([num[p["id"]], num[o]] for p in pars for o in p.get("embedded_in", []))
    ok &= all(any(s in e for e in cou) for s in d.get("cousin_has", []))
    ok &= not any(s in e for s in d.get("cousin_lacks", []) for e in cou)
    if "cousin_units" in d:
        ok &= d["cousin_units"] == [k for k, r in enumerate(recs, 1) if r["cousins"]]
    ok &= d.get("folded", folded) == folded
    return bool(ok)


def judge(c, rec):
    par = " ".join(x["edge"] for x in rec["parents"]); cou = [x["edge"] for x in rec["cousins"]]
    allout = par + " " + " ".join(cou)
    if "together" in c:
        hold = {e for e in cou if any(f" {w}/" in e for w in c["together"])}
        return len(hold) == 1 and all(f" {w}/" in next(iter(hold)) for w in c["together"])
    if "not_only" in c:
        if not allout.strip():
            return None
        return ("not_only/M/en" in allout) == c["not_only"] and ("but_also/M/en" in allout) == c["but_also"]
    need, avoid = c.get("expect", c.get("need")), c.get("forbid", c.get("avoid", []))
    if need is None:
        return None
    return all(s in allout for s in need) and not any(s in allout for s in avoid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--script", default=os.path.join(HERE, "g3_curation_v2.py"))
    ap.add_argument("--ref", default=None, help="a reference script for sets without expectations")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    m = load(a.script)
    if hasattr(m, "FORMS") and os.path.exists(getattr(m, "COMPOUND_FORMS", "")):    # RL-180: the compound list, as run() loads it
        m.FORMS = {r["key"]: r["form"] for r in __import__("csv").DictReader(open(m.COMPOUND_FORMS, encoding="utf-8"), delimiter="\t")}
    if hasattr(m, "HUB_FAMS") and os.path.exists(getattr(m, "HUB_FAMILIES", "")):   # RL-185: the hub family list, as run() loads it
        m.HUB_FAMS.update(l.strip() for l in open(m.HUB_FAMILIES, encoding="utf-8") if l.strip() and not l.startswith("#"))
    ref = load(a.ref, "ref") if a.ref else None
    total = [0, 0]
    for path in a.sets:
        cases = [c for c in json.load(open(path)) if "for" not in c or "q4full" in c["for"]]
        parsed = parse_all([t for c in cases for t in c.get("units", [c.get("text")]) if t])
        res, same = [], [0, 0]
        for c in cases:
            if "units" in c:                               # D6 / item 11 sentence case
                us, recs, folded = run_sentence(m, c["units"], parsed)
                ok = judge_d6(m, c["d6"], us, recs, folded)
                res.append(ok)
                if a.show and not ok:
                    print(f"  FAIL {c['id']}: {' || '.join(c['units'])}")
                    for r in recs:
                        print("     ", " ".join(p["edge"] for p in r["parents"]), "||", " ".join(x["edge"] for x in r["cousins"]))
                continue
            rec = run(m, unit_of(c["text"], parsed[c["text"]]))
            ok = judge(c, rec)
            res.append(ok)
            out = " ".join(x["edge"] for x in rec["parents"]) + " || " + " ".join(x["edge"] for x in rec["cousins"])
            if ok is None and ref is not None:
                r2 = run(ref, unit_of(c["text"], parsed[c["text"]]))
                o2 = " ".join(x["edge"] for x in r2["parents"]) + " || " + " ".join(x["edge"] for x in r2["cousins"])
                same[0] += out == o2; same[1] += 1
                if out != o2 and a.show:
                    print(f"  DIFF {c['id']}: {c['text']}\n     v2 : {out[:300]}\n     ref: {o2[:300]}")
            if a.show and ok is False:
                print(f"  FAIL {c['id']}: {c['text']}\n     {out[:400]}")
        judged = [r for r in res if r is not None]
        total[0] += sum(judged); total[1] += len(judged)
        line = f"{os.path.basename(path)}: {sum(judged)}/{len(judged)} pass"
        if len(judged) < len(res):
            line += f", {len(res) - len(judged)} without expectation" + (f" ({same[0]}/{same[1]} identical to the reference)" if same[1] else "")
        print(line)
    print(f"TOTAL {total[0]}/{total[1]}")


if __name__ == "__main__":
    main()
