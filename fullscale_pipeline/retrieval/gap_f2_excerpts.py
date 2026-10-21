"""Reader excerpts for the manual check of Stage B F2 articles (2026-10-21, RL-161; owner: option B — Haiku readers for
clear cases, hard cases to Sonnet or the assistant). One batch = one plain-text file a reader can judge without the full
article, plus a JSON index that maps every question back to its article, term and sentence ids.
Per article: title, journal, year; abstract (first 150 words) and keywords from the raw PMC header; every P2 block whose
focal (non-statement) sentences hold a surviving ambiguous hit, the hit marked ⟦…⟧, with its section heading; two facts
from the whole text: the number of machine-learning context words (Stage B) and the article's own expansions of
LLM/GPT/BERT. One question per (article, term): does the term mean a language model here?
    python3 gap_f2_excerpts.py --ids FILE --name NAME          (FILE: lines "corpus<TAB>pmcid")
    python3 gap_f2_excerpts.py --all-f2 --batch-size 250       (every F2 article of both corpora)
Outputs in gap_stage/stage_b/manual/: NAME.txt (for the reader), NAME.json (index).
"""
import csv, json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import gap_stage_b as SB
import build_content_regions as P0

L = SB.L
RAW = {"gap": L + "pure_text_corpus_gap/", "current": L + "pure_text_corpus/"}
OUT = SB.OUT + "manual/"
ABS_WORDS, BLOCK_CHARS, MAX_BLOCKS = 150, 900, 4


def header_parts(raw):
    """(abstract, keywords) from the raw PMC text: the abstract is the text after the header block (its last '====' line)
    up to the Introduction or the first body heading; an 'Abstract' heading line, if present, is skipped"""
    lines = raw.split("\n")
    kw = next((m.group(1).strip()[:200] for l in lines[:300] for m in [re.match(r"^\s*(?:keywords?|key words)\s*:\s*(.+)$", l, re.I)] if m), "")
    seps = [i for i, l in enumerate(lines[:400]) if re.match(r"^\W*={5,}", l)]   # a stray control byte may precede it
    buf = []
    for l in lines[(seps[-1] + 1 if seps else 0):]:
        t = l.strip()
        if not t or re.match(r"^abstract\s*:?$", t, re.I):
            continue
        n = P0.norm(t)
        if P0.INTRO.match(n) or P0.BODY.match(n) and n not in ("background", "methods") or P0.RES.match(n) and len(buf) > 3:
            break
        buf.append(t)
        if len(" ".join(buf).split()) > ABS_WORDS:
            break
    return " ".join(" ".join(buf).split()[:ABS_WORDS]), kw


def surface_rx(t):
    """a regex for the hit's surface in its sentence"""
    if t.startswith("v3:"):
        return re.compile(re.escape(t.split(":", 2)[2]))
    return re.compile(r"(?i)(?<![A-Za-z0-9])" + re.escape(t).replace(r"\ ", r"[\s-]") + r"s?(?![A-Za-z0-9])")


def expansions(text):
    out = []
    for m in SB.ABBR.finditer(text):
        ab = m.group(2).upper().rstrip("S")
        words = re.findall(r"[A-Za-z]+", m.group(1)); joined = re.findall(r"[A-Za-z][A-Za-z-]*", m.group(1))
        for w in (words, joined):
            if len(w) >= len(ab) and "".join(x[0] for x in w[-len(ab):]).upper() == ab:
                out.append(f"{m.group(2)} = " + " ".join(m.group(1).split()[-len(ab) - 1:]))
                break
    return sorted(set(out))[:4]


def article(corpus, rec, meta, row):
    raw = open(RAW[corpus] + rec["pmcid"] + ".txt", encoding="utf-8", errors="replace").read()
    rl = raw.split("\n")
    heads = [(i, rl[i].strip()) for i in range(len(rl)) if P0.is_head(rl, i)]
    ab, kw = header_parts(raw)
    year = int(row["year"]) if str(row["year"]).isdigit() else None
    senses = SB.abbr_senses("\n".join(open(f, encoding="utf-8", errors="replace").read()
                                      for f in SB.files_by_article_cached(corpus).get(rec["pmcid"], [])))
    blocks, terms = [], collections.defaultdict(list)
    for b in rec["blocks"]:
        hits = []
        for s in b["sentences"]:
            if not s["focal"] or SB.is_statement(s["text"]):
                continue
            for t in s["terms"]:
                ry = SB.release_year(t)
                if (year and ry and year < ry) or SB.homonym_reason(t, s["text"], senses) or not SB.is_ambig(SB.term_key(t)):
                    continue
                hits.append((s["sid"], t))
        if not hits:
            continue
        txt = []
        for s in b["sentences"]:
            x = s["text"]
            for sid, t in hits:
                if sid == s["sid"]:
                    x = surface_rx(t).sub(lambda m: "⟦" + m.group(0) + "⟧", x, count=3)
            txt.append(re.sub("⟦+", "⟦", re.sub("⟧+", "⟧", x)))
        h = next((h[1] for h in reversed(heads) if h[0] < b["raw_line_first"] - 1), "")
        body = " ".join(txt)
        if len(body) > BLOCK_CHARS:                     # windows of ±250 characters around every marked term, merged
            wins = []
            for m in re.finditer("⟦", body):
                a0, b0 = max(0, m.start() - 250), min(len(body), m.start() + 250)
                if wins and a0 <= wins[-1][1]:
                    wins[-1][1] = b0
                else:
                    wins.append([a0, b0])
            parts, used = [], 0
            for a0, b0 in wins:
                if used > 1500:
                    break
                parts.append(body[a0:b0]); used += b0 - a0
            body = ("…" if wins[0][0] else "") + " … ".join(parts) + ("…" if wins[-1][1] < len(body) else "")
        bterms = set()
        for sid, t in hits:
            k = SB.term_key(t) if not t.startswith("v3:") else t.split(":", 2)[2]
            k = re.sub(r"(?i)^(llm|transformer model|language model)s$", r"\1", k)
            terms[k].append(sid); bterms.add(k)
        blocks.append(dict(heading=h, text=body, terms=bterms))
    return dict(corpus=corpus, pmcid=rec["pmcid"], title=meta.get("title", ""), journal=meta.get("journal", ""),
                year=row["year"], abstract=ab, keywords=kw, n_context=row["n_context"], expansions=expansions(raw),
                blocks=pick_blocks(blocks), n_blocks=len(blocks), terms={k: v for k, v in terms.items()})


def pick_blocks(blocks):
    """at most MAX_BLOCKS passages, chosen so that every asked term appears in at least one (then in text order)"""
    chosen, seen = [], set()
    for i, b in enumerate(blocks):
        if b["terms"] - seen:
            chosen.append(i); seen |= b["terms"]
    for i in range(len(blocks)):
        if len(chosen) >= MAX_BLOCKS:
            break
        if i not in chosen:
            chosen.append(i)
    return [blocks[i] for i in sorted(chosen[:max(MAX_BLOCKS, len(chosen))])]


_FILES = {}
def files_by_article_cached(corpus):
    if corpus not in _FILES:
        _FILES[corpus] = SB.files_by_article(SB.CORPORA[corpus][1])
    return _FILES[corpus]
SB.files_by_article_cached = files_by_article_cached


def render(arts, name):
    q, lines = [], [f"# Batch {name}: {len(arts)} articles", ""]
    for i, a in enumerate(arts, 1):
        aid = f"A{i:03d}"
        lines += [f"## {aid} | {a['journal']} | {a['year']}", f"Title: {a['title']}"]
        if a["keywords"]:
            lines.append(f"Keywords: {a['keywords']}")
        if a["abstract"]:
            lines.append(f"Abstract (start): {a['abstract']}")
        lines.append(f"Facts: machine-learning context words in the whole article: {a['n_context']}; "
                     f"abbreviation expansions in the article: {'; '.join(a['expansions']) or 'none'}")
        for j, b in enumerate(a["blocks"], 1):
            lines.append(f"Passage {j} (section: {b['heading'] or 'unknown'}): {b['text']}")
        if a["n_blocks"] > len(a["blocks"]):
            lines.append(f"(+{a['n_blocks'] - len(a['blocks'])} more passages not shown)")
        for k, term in enumerate(sorted(a["terms"]), 1):
            qid = f"{aid}.{k}"
            what = ("the machine-learning transformer architecture" if term.lower().startswith("transformer model")
                    else "a neural language model (or a model or chatbot built on one)")
            lines.append(f"QUESTION {qid}: does ⟦{term}⟧ refer to {what} here?")
            q.append(dict(qid=qid, corpus=a["corpus"], pmcid=a["pmcid"], term=term, sids=a["terms"][term]))
        lines.append("")
    os.makedirs(OUT, exist_ok=True)
    open(OUT + name + ".txt", "w", encoding="utf-8").write("\n".join(lines))
    json.dump(dict(name=name, articles=[{k: a[k] for k in ("corpus", "pmcid")} for a in arts], questions=q),
              open(OUT + name + ".json", "w"), indent=1)
    return len(q)


def build(pairs, name):
    csv.field_size_limit(10 ** 8)
    rows = {(r["corpus"], r["pmcid"]): r for r in csv.DictReader(open(SB.OUT + "stage_b_articles.tsv"), delimiter="\t")}
    S = json.load(open(L + "gap_esummary_cache.json")); TM = json.load(open(L + "target_metadata.json"))
    want = set(pairs); recs = {}
    for corpus in ("gap", "current"):
        for line in open(SB.CORPORA[corpus][0], encoding="utf-8"):
            pm = line[11:line.index('"', 11)]
            if (corpus, pm) in want:
                recs[(corpus, pm)] = json.loads(line)
    arts = []
    for c, p in pairs:
        m = S.get(p) or {} if c == "gap" else TM.get(p) or {}
        meta = dict(title=m.get("title", ""), journal=m.get("fulljournalname") or m.get("source") or m.get("journal", ""))
        arts.append(article(c, recs[(c, p)], meta, rows[(c, p)]))
    return arts, render(arts, name)


if __name__ == "__main__":
    a = sys.argv
    if "--ids" in a:
        pairs = [tuple(l.split("\t")[:2]) for l in open(a[a.index("--ids") + 1]).read().split("\n") if l.strip()]
        arts, nq = build(pairs, a[a.index("--name") + 1])
        print(len(arts), "articles,", nq, "questions,", sum(not x["terms"] for x in arts), "with no question")
