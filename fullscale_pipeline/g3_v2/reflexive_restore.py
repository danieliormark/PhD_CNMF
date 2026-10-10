"""§6 item 8 (owner 2026-10-09): emphatic reflexive pronouns that P6 replaced by their antecedent get the pronoun back.

P6 (coref_resolution_v2.py) replaced every reflexive pronoun by its antecedent. In the emphatic use this only repeats the
noun ("even for the model itself" -> "even for the model the model"); in the object use ("LLMs cannot evaluate
themselves" -> "... evaluate LLMs") the replacement is a statement and stays. The two uses are told apart by the
dependency label of the reflexive in P6's original sentence (spaCy en_core_web_trf):
  emphatic: appos, npadvmod, attr; "in itself"; "by itself/themselves" unless "by" is a passive agent
  object:   everything else (dobj, dative, pobj of other prepositions, nsubj of small clauses, conj)
For each emphatic replacement in a sentence that reached G2 v3, the replacement string is located in the G2 unit text
(the occurrence whose preceding text best matches the resolved sentence) and replaced by the original pronoun; the
unit is then re-parsed alone with graphbrain as G2 parsed its pieces (create_parser(lang="en", lemmas=True)). G3 then
drops the pronoun like any other. A unit the parser splits into several sentences, or whose replacement cannot be
located, is left as it is (counted). Output, one record per restored unit: uid, the G2 unit hash, sid, the G2 text, the
restored text, the restorations, and the new parse (main edge, lemma edges, atom2word).

    python reflexive_restore.py --outdir DIR       (refuses to overwrite)
"""
import argparse, collections, glob, hashlib, json, os, re, sys, time

COREF = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/coref_v2/coref_resolved_v2.jsonl"
SHARDS = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"
REFLEXIVES = {"itself", "themselves", "himself", "herself"}


def emphatic(tok):
    """the reflexive token's use in the original sentence: True emphatic, False object"""
    if tok.dep_ in ("appos", "npadvmod", "attr"):
        return True
    if tok.dep_ == "pobj" and tok.head.text.lower() == "in":
        return True
    if tok.dep_ == "pobj" and tok.head.text.lower() == "by" and tok.head.dep_ != "agent":
        return True
    return False


def common_suffix(a, b):
    a, b = re.sub(r"\s+", " ", a), re.sub(r"\s+", " ", b)
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out_p, sum_p = os.path.join(a.outdir, "reflexive_restored.jsonl"), os.path.join(a.outdir, "reflexive_summary.json")
    for p in (out_p, sum_p):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    t0, st = time.time(), collections.Counter()
    # 1. every reflexive replacement in P6, with its offset in the resolved text
    items = []
    for line in open(COREF, encoding="utf-8"):
        art = json.loads(line)
        for b in art.get("blocks", []):
            for s in b.get("sentences", []):
                reps = sorted(s.get("replacements") or [], key=lambda r: r["start"])
                shift = 0
                for r in reps:
                    rs = r["start"] + shift
                    if r["pronoun"].lower() in REFLEXIVES:
                        items.append(dict(sid=s["sid"], text=s["text"], resolved=s["text_resolved"], start=r["start"],
                                          end=r["end"], pronoun=s["text"][r["start"]:r["end"]], rep=r["replacement"],
                                          rstart=rs, rend=rs + len(r["replacement"])))
                    shift += len(r["replacement"]) - (r["end"] - r["start"])
    st["reflexive_replacements"] = len(items)
    # 2. emphatic or object, from the original sentence
    import spacy
    nlp = spacy.load("en_core_web_trf")
    for doc, it in zip(nlp.pipe([x["text"] for x in items], batch_size=32), items):
        tok = next((t for t in doc if t.idx == it["start"]), None)
        it["dep"] = tok.dep_ if tok is not None else None
        it["emphatic"] = bool(tok is not None and emphatic(tok))
        st["emphatic" if it["emphatic"] else "object_kept"] += 1
    emph = collections.defaultdict(list)
    for it in items:
        if it["emphatic"]:
            emph[it["sid"]].append(it)
    # 3. locate in the G2 units and restore
    restored = []
    for f in sorted(glob.glob(SHARDS)):
        for line in open(f, encoding="utf-8"):
            art = json.loads(line)
            for s in art["sentences"]:
                if s["sid"] not in emph:
                    continue
                st["emphatic_sentences_in_g2"] += 1
                units = {u["uid"]: dict(u, new=u["text"], done=[]) for u in s.get("units", []) if u.get("main_edge")}
                for it in sorted(emph[s["sid"]], key=lambda x: -x["rstart"]):      # right to left: earlier offsets stay valid
                    left = it["resolved"][max(0, it["rstart"] - 30):it["rstart"]]
                    best = None
                    for uid, u in units.items():
                        for m in re.finditer(re.escape(it["rep"]), u["new"]):
                            sc = common_suffix(left, u["new"][:m.start()])
                            if best is None or sc > best[0]:
                                best = (sc, uid, m.start(), m.end())
                    if best is None or best[0] < 8:
                        st["not_located"] += 1
                        continue
                    _, uid, i, j = best
                    u = units[uid]
                    u["new"] = u["new"][:i] + it["pronoun"] + u["new"][j:]
                    u["done"].append(dict(replacement=it["rep"], pronoun=it["pronoun"], dep=it["dep"]))
                    st["located"] += 1
                for u in units.values():
                    if u["done"]:
                        restored.append(dict(pmcid=art["pmcid"], sid=s["sid"], uid=u["uid"], unit_hash=u["hash"],
                                             text_g2=u["text"], text_restored=u["new"], restorations=u["done"]))
    # 4. re-parse each restored unit as G2 parsed its pieces
    from graphbrain.parsers import create_parser
    parser = create_parser(lang="en", lemmas=True)
    with open(out_p, "w", encoding="utf-8") as fo:
        for r in restored:
            parser.atom2token = {}
            parses = [p for p in parser.parse(r["text_restored"])["parses"] if p["main_edge"] is not None]
            if len(parses) != 1:
                st["reparse_not_one_sentence"] += 1
                continue
            p = parses[0]
            r.update(main_edge=p["main_edge"].to_str(), extra_edges=sorted(x.to_str() for x in p["extra_edges"]),
                     atom2word=sorted(([str(at), w, i] for at, (w, i) in p.get("atom2word", {}).items()), key=lambda x: x[2]))
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            st["units_restored"] += 1
    st["seconds"] = round(time.time() - t0)
    json.dump(dict(stats=st, script=os.path.abspath(__file__)), open(sum_p, "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
