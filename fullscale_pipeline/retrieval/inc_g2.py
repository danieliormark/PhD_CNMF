"""Incremental run, step 5 (2026-10-21, RL-167): G2 v3 (phase5_graphbrain/scripts/graphbrain_parse_v3.py, unchanged) with
two approved additions, applied through this wrapper:
  * hyphenated v3 model names stay whole: the hyphen rule keeps a word whose parts contain a focal term; the flat v2 list
    (PP/focal_words.txt) lacks the new names, so every hyphenated surface the v3 matcher accepted
    (LCS/inc_run/focal_surfaces_v3_hyphenated.txt: BioMistral-7b, ProtBERT-BFD, …) is added to it;
  * curly apostrophe (owner, RL-152): a sentence whose parse types a quote mark (’ or ‘) as a predicate is parsed again
    with every U+2019 after a letter or digit written as "'" (transforms.apostrophe_normalised); other sentences are parsed
    as written. sid, hash_final and text_final are never changed.
Caption bodies are read from both v3 P2 outputs (current corpus and new articles).
    python inc_g2.py --selftest-wrapper
    python inc_g2.py <graphbrain_parse_v3.py arguments: --shard I --nshards N --input FILE --outdir DIR | --verify ...>
"""
import importlib.util, json, re, sys
R = "/mnt/hum01-rds/Basov/p91688di/"
PG = R + "phase5_graphbrain/scripts/"
LCS = R + "llm_corpus_staging/"
P2S = [LCS + "p2_v3_current/focal_extractions_v2.jsonl", LCS + "gap_stage/llm_corpus_staging/focal_extractions_v2.jsonl"]
SURF = __import__("os").environ.get("INC_G2_SURF") or LCS + "inc_run/focal_surfaces_v3_hyphenated.txt"   # RL-203: override for A3 part 2
NORM = re.compile(r"(?<=[A-Za-z0-9])’")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


m = load("graphbrain_parse_v3", PG + "graphbrain_parse_v3.py")
_orig_process = m.process_sentence


def quote_as_verb(rec):
    return any(w in ("’", "‘") and a.split("/")[1].startswith("P") for u in rec.get("units", []) for a, w, i in u.get("atom2word", [])
               if "/" in a)


def process_sentence(parser, hedge, s, caption_bodies, url_table):
    rec, notes, seg = _orig_process(parser, hedge, s, caption_bodies, url_table)
    if quote_as_verb(rec) and NORM.search(s["text_final"]):
        s2 = dict(s, text_final=NORM.sub("'", s["text_final"]))
        rec, notes, seg = _orig_process(parser, hedge, s2, caption_bodies, url_table)
        rec["transforms"]["apostrophe_normalised"] = True
    return rec, notes, seg


def caption_bodies():
    bodies = set()
    for p in P2S:
        for line in open(p, encoding="utf-8"):
            for b in json.loads(line)["blocks"]:
                ss = b["sentences"]
                for x, y in zip(ss, ss[1:]):
                    if m.LABEL_END.search(x["text"].rstrip()) and m.CAPTION_BODY.match(y["text"]):
                        bodies.add(y["sid"])
    return bodies


def install():
    seqs = m.focal_sequences()                      # the v2 flat list
    for w in open(SURF, encoding="utf-8"):
        if w.strip():
            seqs.append(tuple(re.split(r"[\s\-]+", w.strip().lower())))
    m.process_sentence = process_sentence
    m.caption_bodies = caption_bodies


if __name__ == "__main__":
    install()
    if "--selftest-wrapper" in sys.argv:
        ok = [m.has_focal_part("BioMistral-7b"), m.has_focal_part("ProtBERT-BFD"), m.has_focal_part("CT-BERT"), not m.has_focal_part("pre-trained")]
        print("hyphen rule:", ok)
        sys.exit(0 if all(ok) else 1)
    sys.argv = ["graphbrain_parse_v3.py"] + sys.argv[1:]
    m.main()
