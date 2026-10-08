"""Corpus scope exclusions and spelling corrections (G3_POSTPROCESSING.md §6 item 6, owner decisions 2026-10-08).

    python build_scope_exclusions.py [--outdir DIR]       # default: this folder; writes scope_exclusions.csv, spelling_corrections.csv

Inputs: the hand-checked decision lists in decisions/ and the G2 v3 shards (every listed article must be there, every
correction must match its unit text exactly once). Outputs:
  scope_exclusions.csv      pmcid, reason, detail: articles G3 skips and M1 leaves out of every relation
  spelling_corrections.csv  uid, unit_hash, wrong, right: word-for-word fixes applied to a unit's text before G3 matches
                            focal terms (one word for one word, so token positions stay valid)
Ideally this belongs to preprocessing (next to PP/article_blacklist.py); it runs here so that P2 and G2 need no rerun.
"""
import argparse, csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARDS = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards/g2_parsed_*.jsonl"
REASONS = [   # (decision file, reason code)
    ("llm_defined_otherwise.csv", "llm_defined_otherwise"),
    ("llm_no_definition_no_evidence.csv", "llm_no_definition_no_evidence"),
    ("name_homonyms.csv", "name_homonym"),
    ("transformer_only.csv", "transformer_only_not_llm"),
]
CORRECTIONS = [   # (uid, wrong, right) in the 5 kept articles whose LLM definition is misspelt or a misnomer
    ("PMC11929840.r1.L53.S2.U1", "Languge", "Language"),
    ("PMC11849343.r1.L58.S3.U1", "Langue", "Language"),
    ("PMC11601203.r1.L157.S4.U1", "langaue", "language"),
    ("PMC11989310.r1.L54.S1.U1", "learning", "language"),
    ("PMC11881179.r1.L73.S1.U1", "learning", "language"),
    ("PMC11881179.r1.L105.S4.U1", "learning", "language"),
]


def correct_text(text, wrong, right):
    """replace the one occurrence of `wrong` that stands before 'model(s)' (the definition), as a whole word"""
    rx = re.compile(r"(?<![A-Za-z])" + re.escape(wrong) + r"(?=\s+[Mm]odel)")
    hits = list(rx.finditer(text))
    if len(hits) != 1:
        raise ValueError(f"{wrong!r} before 'model' found {len(hits)} times")
    m = hits[0]
    return text[:m.start()] + right + text[m.end():]


def load_corrections(path):
    """{uid: [(wrong, right)]} from spelling_corrections.csv, for G3"""
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["uid"], []).append((r["wrong"], r["right"]))
    return out


def correct_a2w(a2w, wrong, right):
    """the same fix in G2's atom-to-word list: the token `wrong` directly followed by a 'model(s)' token"""
    words = {p: w for _, w, p in a2w}
    pos = [p for p, w in words.items() if w == wrong and words.get(p + 1, "").lower().startswith("model")]
    if len(pos) != 1:
        raise ValueError(f"{wrong!r} before 'model' found at {len(pos)} token positions")
    return [[a, right if p == pos[0] else w, p] for a, w, p in a2w]


def apply_corrections(unit, corrections):
    """G3 hook, called before focal terms are matched: the unit with its text and atom-to-word words corrected
    (returned unchanged if it has no correction). Both must change, or the corrected token is not mapped to its atom
    and the misspelt word survives as an atom of its own."""
    fixes = corrections.get(unit["uid"])
    if not fixes:
        return unit
    u = dict(unit)
    for wrong, right in fixes:
        u["text"] = correct_text(u["text"], wrong, right)
        u["atom2word"] = correct_a2w(u["atom2word"], wrong, right)
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()
    rows, seen = [], {}
    for fn, reason in REASONS:
        for r in csv.DictReader(open(os.path.join(HERE, "decisions", fn))):
            if r["decision"] != "exclude":
                continue
            detail = r.get("note") or r.get("term") or r.get("label") or ""
            if r["pmcid"] in seen:
                sys.exit(f"{r['pmcid']} listed twice ({seen[r['pmcid']]}, {reason})")
            seen[r["pmcid"]] = reason
            rows.append((r["pmcid"], reason, detail))
    units = {u for u, _, _ in CORRECTIONS}
    found_art, found_unit = set(), {}
    for f in sorted(glob.glob(SHARDS)):
        for line in open(f):
            art = json.loads(line)
            if art["pmcid"] in seen:
                found_art.add(art["pmcid"])
            for s in art["sentences"]:
                for u in s.get("units", []):
                    if u["uid"] in units:
                        found_unit[u["uid"]] = u
    missing = set(seen) - found_art
    if missing:
        sys.exit(f"{len(missing)} excluded articles not in G2 v3: {sorted(missing)[:5]}")
    corr = []
    for uid, wrong, right in CORRECTIONS:
        if uid not in found_unit:
            sys.exit(f"unit {uid} not in G2 v3")
        u = found_unit[uid]
        new = correct_text(u["text"], wrong, right)
        assert len(new.split()) == len(u["text"].split())
        correct_a2w(u["atom2word"], wrong, right)
        corr.append((uid, u["hash"], wrong, right))
    with open(os.path.join(a.outdir, "scope_exclusions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pmcid", "reason", "detail"]); w.writerows(sorted(rows))
    with open(os.path.join(a.outdir, "spelling_corrections.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["uid", "unit_hash", "wrong", "right"]); w.writerows(corr)
    by = {}
    for _, reason, _ in rows:
        by[reason] = by.get(reason, 0) + 1
    print(f"{len(rows)} articles excluded {by}; {len(corr)} spelling corrections; all found in G2 v3")


if __name__ == "__main__":
    main()
