"""
Focal terms v3.1 (2026-10-26, RL-203, A3 part 2; owner 2026-10-26: "I confirm A3 part 2"). The v3 matcher plus G3's rule for
model names written in lower-case Hugging Face style (RL-185: claude-sonnet-4.5, gpt-oss-20b, distilbert-base-uncased,
cardiffnlp/twitter-roberta-base-sentiment), so that P2 and P6 find the sentences that G3 would already read as focal but that
were lost upstream (RL-184's scan: no v3 hit in the sentence, so P2 never selected it). Every v3 hit is kept as it is; the
hub hits are added only on tokens no v3 hit (accepted or rejected) covers. The rule is not copied: G3's own `hub_hits`
(g3_v2/g3_curation_v2.py) and its family list (PG/g3_inputs_v3_1/hub_families.txt) are loaded, so the two stages cannot
diverge. Term names: "hub:<spelling>", as in G3. Hub spellings count as article evidence (like G4 names: a checkpoint name
is unambiguous).
    python3 focal_terms_v3_1.py --selftest
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import focal_terms_v3 as v3

G3 = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/g3_v2/g3_curation_v2.py"
HUB_FAMILIES = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_inputs_v3_1/hub_families.txt"
_src = open(G3, encoding="utf-8").read()
_ns = {"re": re}
exec(_src[_src.index("# RL-185 (owner 2026-10-23): model names"):_src.index("def focal_mentions")], _ns)
_ns["HUB_FAMS"].update(l.strip() for l in open(HUB_FAMILIES, encoding="utf-8") if l.strip() and not l.startswith("#"))
hub_hits = _ns["hub_hits"]


class Matcher(v3.Matcher):
    """v3's matcher plus G3's hub spellings; the interface (anchor_hits, hits) is v2's"""

    def anchor_hits(self, text):
        out = list(super().anchor_hits(text))
        for line in text.split("\n"):
            if line.strip():
                acc, rej = super().hits(line, True)
                out += [h[1] for h in hub_hits(line, [h[1] for h in acc + rej])]
        return out

    def hits(self, line, anchored):
        acc, rej = super().hits(line, anchored)
        return acc + hub_hits(line, [h[1] for h in acc + rej]), rej


SELFTEST = [   # (text, anchored, accepted hub surfaces expected)
    ("We used claude-sonnet-4.5 as the auditor.", False, ["claude-sonnet-4.5"]),
    ("gpt-oss-20b and gpt-oss-120b were run locally.", False, ["gpt-oss-20b", "gpt-oss-120b"]),
    ("We used the cardiffnlp/twitter-roberta-base-sentiment model.", False, ["cardiffnlp/twitter-roberta-base-sentiment"]),
    ("We used bert-base-uncased.", True, []),                         # a v3 hit (A5): not doubled
    ("The plasmid 1-2-3-gpt-6-7-8 was used.", True, []),
    ("The g4em-standard_opt4 physics list was used.", True, []),
    ("Patients aged 18-65 were included.", True, []),
]


def selftest():
    m = Matcher(); bad = 0
    for text, anch, exp in SELFTEST:
        got = sorted(s for t, _, s in m.hits(text, anch)[0] if t.startswith("hub:"))
        ok = got == sorted(exp); bad += not ok
        print("PASS " if ok else "FAIL ", repr(text[:60]), "" if ok else f"got {got} expected {sorted(exp)}")
    for text, anch, exp in v3.SELFTEST:                                # v3 behaviour unchanged on v3's own cases
        got = sorted(s for t, _, s in v3.Matcher.hits(m, text, anch)[0] if t.startswith("v3:"))
        ok = got == sorted(exp); bad += not ok
        if not ok: print("FAIL v3 case", repr(text[:60]), got, exp)
    n = len(SELFTEST) + len(v3.SELFTEST)
    print(f"\n{n - bad} of {n} passed")
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
