"""RL-185 (owner 2026-10-23, B2): G2 v3 through the incremental wrapper inc_g2.py (v3 hyphenated names, curly apostrophes)
plus a dash rewrite before parsing: an aside between em dashes or spaced en dashes (two or more words, no brackets inside,
not a number range) is written in brackets ("models—such as LLMs trained …—to take" -> "models (such as LLMs trained …)
to take"), and an em dash glued between two words is spaced ("AI)—particularly" -> "AI) — particularly"). Applied to the
sentences of inc_dash_prepare.py; transforms.dash_rewritten = True when the text changed. sid, hash_final and text_final
(in P7) are never changed; the units carry the parsed text and their hash, as for the apostrophe re-parses.
    python inc_g2_dash.py --selftest-dash
    python inc_g2_dash.py --shard I --nshards N --input FILE --outdir DIR
"""
import importlib.util, re, sys

HERE = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/retrieval/"
spec = importlib.util.spec_from_file_location("inc_g2", HERE + "inc_g2.py")
w = importlib.util.module_from_spec(spec); sys.modules["inc_g2"] = w; spec.loader.exec_module(w)
m = w.m                                                     # graphbrain_parse_v3
_inner = None
ASIDE = re.compile(r"(?:\s?—\s?|\s–\s)([^—–()\[\]]{2,150}?)(?:\s?—\s?|\s–\s)")
GLUED = re.compile(r"(?<=[A-Za-z0-9)\]])—(?=[A-Za-z(])")


def dash_rewrite(t):
    def br(x):
        inner = x.group(1).strip()
        if len(inner.split()) < 2 or re.fullmatch(r"[\d.,%\s-]+", inner):
            return x.group(0)
        return " (" + inner + ") "
    t2 = ASIDE.sub(br, t)
    t2 = GLUED.sub(" — ", t2)
    return re.sub(r"\s{2,}", " ", t2).replace(" )", ")").replace("( ", "(").replace(" ,", ",").replace(" .", ".")


def process_sentence(parser, hedge, s, caption_bodies, url_table):
    t2 = dash_rewrite(s["text_final"])
    if t2 == s["text_final"]:
        return _inner(parser, hedge, s, caption_bodies, url_table)
    rec, notes, seg = _inner(parser, hedge, dict(s, text_final=t2), caption_bodies, url_table)
    rec["transforms"]["dash_rewritten"] = True
    return rec, notes, seg


if __name__ == "__main__":
    w.install()
    _inner = m.process_sentence                              # inc_g2's (apostrophe rule) process_sentence
    m.process_sentence = process_sentence
    if "--selftest-dash" in sys.argv:
        cases = [("existing models—such as LLMs trained at scale—to take advantage", "existing models (such as LLMs trained at scale) to take advantage"),
                 ("AI)—particularly Large Language Models", "AI) — particularly Large Language Models"),
                 ("in all Fig. 5C–H, while there", "in all Fig. 5C–H, while there"),
                 ("reported 3–6 h of daily use", "reported 3–6 h of daily use"),
                 ("five members – a WIC research expert, an expert in LLM – conducted", "five members (a WIC research expert, an expert in LLM) conducted"),
                 ("traceability—clearly presenting (including principles)—nurses can", "traceability — clearly presenting (including principles) — nurses can")]
        bad = 0
        for a, want in cases:
            got = dash_rewrite(a); ok = got == want; bad += not ok
            print("PASS" if ok else "FAIL", repr(got) if not ok else a[:60])
        sys.exit(1 if bad else 0)
    sys.argv = ["graphbrain_parse_v3.py"] + sys.argv[1:]
    m.main()
