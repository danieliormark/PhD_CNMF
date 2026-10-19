# Test (owner 2026-10-19) of a G2 change before any re-parse: a curly apostrophe (U+2019) after a letter or digit is written
# as a straight one before parsing. Each sentence is parsed twice with the current G2 code (as written / normalised), both
# parses go through the current G3, and the parents are compared. Inputs: the units whose quote mark G2 v3 parsed as a verb
# (all), a random sample of other units with a curly apostrophe, and constructed hard cases.
import collections, glob, importlib.util, json, random, re, sys

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

g2 = load("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/scripts/graphbrain_parse_v3.py", "g2")
g3 = load("/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/g3_v2/g3_curation_v2.py", "g3")
from graphbrain import hedge
from graphbrain.parsers import create_parser

NORM = re.compile(r"(?<=[A-Za-z0-9])’")
CURLY = re.compile(r"[A-Za-z0-9]’")
HARD = [
    "The LLMs’ responses were differentiated based on the official answers.",
    "ChatGPT’s answers were more accurate than Bard’s answers.",
    "Physicians’ attitudes toward LLMs were mostly positive.",
    "Students described ChatGPT’s output as ‘hallucination’ when it invented references.",
    "The term ‘large language model’ was coined after GPT-3 appeared.",
    "We asked ChatGPT ‘what’s the diagnosis?’ for each case.",
    "It’s clear that GPT-4 doesn’t replace clinicians.",
    "I’m a language model and I can’t browse the internet, said ChatGPT.",
    "O’Connor’s study compared ChatGPT with human tutors.",
    "Unlike the 1990’s expert systems, LLMs learn from text.",
    "The ‘LLMs’ in this study were GPT-4 and Gemini.",
    "GPT-4’s and Claude’s outputs were scored by two raters.",
    "The models’ ‘reasoning’ was assessed with chain-of-thought prompts.",
    "ChatGPT replied: ‘I’m not a doctor, but I can offer general information.’",
    "Parkinson’s disease notes were summarised by BERT.",
]


def parents_of(parser, sid, text):
    rec, notes, seg = g2.process_sentence(parser, hedge, dict(sid=sid, hash_final="0" * 12, text_final=text), set(), {})
    out, qverb = [], 0
    for u in rec["units"]:
        qverb += any(w in ("’", "‘", "'") and a.split("/")[1].startswith("P") for a, w, i in u.get("atom2word", []))
        if not u["main_edge"]:
            continue
        r3, _ = g3.curate_unit("PMCX", rec, u, g3.focal_terms.Matcher(), True, "keep")
        out += [p["edge"] for p in r3["parents"]]
    return out, qverb, len(rec["units"])


def sample(n_other, seed):
    S = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    broken, other = [], []
    for p in sorted(glob.glob("/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            for s in r["sentences"]:
                us = s.get("units", [])
                if not any(CURLY.search(u["text"]) for u in us):
                    continue
                bad = any(w in ("’", "‘") and a.split("/")[1].startswith("P") for u in us for a, w, i in u.get("atom2word", []))
                (broken if bad else other).append(s["sid"])
    random.seed(seed)
    return broken, random.sample(other, n_other), len(other)


if __name__ == "__main__":
    out = sys.argv[1]
    broken, other, n_other = sample(400, 151)
    want = set(broken) | set(other)
    text = {}
    for line in open("/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for s in r.get("sentences", []):
            if s["sid"] in want:
                text[s["sid"]] = s["text_final"]
    parser = create_parser(lang="en", lemmas=True)
    res = dict(n_other_population=n_other, real=[], hard=[])
    for group, sids in (("broken", broken), ("other", other)):
        for sid in sids:
            t = text[sid]; tn = NORM.sub("'", t)
            a = parents_of(parser, sid, t); b = parents_of(parser, sid, tn)
            res["real"].append(dict(group=group, sid=sid, text=t, before=a[0], after=b[0], qverb_before=a[1], qverb_after=b[1],
                                    units_before=a[2], units_after=b[2]))
    for i, t in enumerate(HARD):
        tn = NORM.sub("'", t)
        a = parents_of(parser, f"PMCX.r1.L{i+1}.S1", t); b = parents_of(parser, f"PMCX.r1.L{i+1}.S1", tn)
        res["hard"].append(dict(text=t, before=a[0], after=b[0], qverb_before=a[1], qverb_after=b[1]))
    json.dump(res, open(out, "w"), indent=0, ensure_ascii=False)
    print("done", len(res["real"]), len(res["hard"]))
