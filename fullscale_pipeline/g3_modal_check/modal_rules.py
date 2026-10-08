"""Proposed G3 verb-group merge rule for lexical modal verbs (test implementation, not the production G3).

One decision per trigger token. A token may belong to an infinitive group (allow, ask, need, ...) and to a gerund
group (enable_ger, help_in); the infinitive reading is tried first and the gerund reading only if it gives nothing.
Every guard can be switched off, and each decision is also recomputed with each guard left out, so the effect of
every guard (and any overlap between guards) is recorded per candidate.

  python modal_rules.py pool OUT.json GROUPS SHARDS MAX_PER_GROUP SEED
  python modal_rules.py hard CASES.json OUT.json          (parses the sentences with graphbrain, as G2 does)
"""
import json, os, re, sys, random, collections
from graphbrain import hedge

SHDIR = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g2_v3/shards"
ALL_GUARDS = tuple(g for g in ("order", "need", "let", "adj", "fallback", "precedence", "anchor", "passive_ger", "needobj")
                   if g not in os.environ.get("OFF", "").split(","))

GROUPS = {
    "allow": ("inf", "allow allows allowed allowing enable enables enabled enabling permit permits permitted permitting "
                     "help helps helped helping let lets letting"),
    "force": ("inf", "force forces forced forcing"),
    "empower": ("inf", "empower empowers empowered empowering"),
    "ask": ("inf", "ask asks asked asking"),
    "prompt": ("inf", "prompt prompts prompted prompting"),
    "instruct": ("inf", "instruct instructs instructed instructing"),
    "need": ("inf", "need needs needed"),
    "have_to": ("inf", "have has had"),
    "fail": ("inf", "fail fails failed failing"),
    "tend": ("inf", "tend tends tended"),
    "able": ("able", "able unable"),
    "help_in": ("help_in", "help helps helped helping"),
    "enable_ger": ("ger", "allow allows allowed allowing enable enables enabled enabling permit permits permitted permitting"),
}
SEMI = ("need", "have_to", "able", "fail", "tend")
# "let X denote / be / represent ...": a mathematical definition; "let me know": an idiom
LET_DEFINITIONAL = {"denote", "denoted", "represent", "represented", "be", "consist"}
NEXT = {
    "inf": re.compile(r"^(?:\S+\s+){0,6}?to\s+(?!the\b|a\b|an\b|this\b|these\b|their\b|its\b)[a-z_]+", re.I),
    "able": re.compile(r"^to\s+[a-z_]+", re.I),
    "help_in": re.compile(r"^in\s+[a-z_]+ing\b", re.I),
    "ger": re.compile(r"^[a-z_]+ing\b", re.I),
    "have_to": re.compile(r"^to\s+[a-z_]+", re.I),
    "allow": re.compile(r"^(?:\S+\s+){0,6}?(?:to\s+)?[a-z_]+", re.I),
}
FORM2G = collections.defaultdict(list)
for _g, (_k, _forms) in GROUPS.items():
    for _w in _forms.split():
        FORM2G[_w].append(_g)


def phead(x):
    if x.atom:
        return x if x.type()[0] == "P" else None
    if x[0].type()[0] == "M":
        return phead(x[-1])
    return None


def feats(a):
    r = a.role()
    return r[2] if len(r) > 2 else ""


def argroles(a):
    r = a.role()
    return r[1] if len(r) > 1 else ""


def inf_clauses(e, depth=0):
    if e.atom or depth > 4:
        return []
    h = phead(e[0])
    conn = str(e[0])
    if h is not None and len(feats(h)) > 1 and feats(h)[1] == "i":
        return [e]
    if "/Mi/" in conn or "be/Mv.-i" in conn:
        return [e]
    if e[0].type()[0] == "J":
        items = (list(e[1:]) if e[0].atom else list(e[0][1:]) + list(e[1:]))
        return [c for x in items for c in inf_clauses(x, depth + 1)]
    if e[0].type()[0] == "M" and len(e) == 2:
        return inf_clauses(e[1], depth + 1)
    return []


def ger_clauses(e, need_in):
    if e.atom:
        return []
    if e[0].atom and e[0].root() == "in" and e[0].type()[0] == "T":
        return ger_clauses(e[1], False) if len(e) > 1 else []
    if need_in:
        return []
    h = phead(e[0])
    if h is not None and feats(h)[1:3] == "pg":
        return [e]
    if e[0].atom and e[0].type()[0] == "J":
        return [c for x in e[1:] for c in ger_clauses(x, need_in)]
    return []


def verbs_of(c):
    if c.atom:
        return [c] if c.type()[0] == "P" else []
    h = phead(c[0])
    if h is not None:
        return [h]
    if c[0].atom and c[0].type()[0] == "M" and len(c) > 1:
        return verbs_of(c[1])
    if not c[0].atom and c[0].type()[0] == "J":
        return [v for x in c[0][1:] for v in verbs_of(x)]
    if c[0].atom and c[0].type()[0] == "J":
        return [v for x in c[1:] for v in verbs_of(x)]
    return [a for a in c[0].atoms() if a.type()[0] == "P"][:1]


def walk(e):
    yield e
    if not e.atom:
        for s in e:
            yield from walk(s)


def trigger_clause(e, atom):
    for sub in walk(e):
        if not sub.atom:
            h = phead(sub[0])
            if h is not None and str(h) == atom:
                return sub
    return None


def group_decision(e, atom, pos, group, a2pos, tok, guards):
    """verbs merged with the trigger for one group reading, or []"""
    kind = GROUPS[group][0]
    word = tok[pos][1].lower()
    if kind == "able":
        best = None
        for sub in walk(e):
            if sub.atom or not any(str(a) == atom for a in sub.atoms()):
                continue
            cl = [c for x in sub[1:] if not any(str(a) == atom for a in x.atoms()) for c in inf_clauses(x)]
            if cl and (best is None or len(str(sub)) < len(str(best[0]))):
                best = (sub, cl)
        comps, clause = (best[1] if best else []), None
    else:
        clause = trigger_clause(e, atom)
        if clause is None:
            return []
        comps = []
        for a in clause[1:]:
            comps += inf_clauses(a) if kind == "inf" else ger_clauses(a, kind == "help_in")
    verbs = []
    for c in comps:
        for v in verbs_of(c):
            ps = a2pos.get(str(v), [])
            after = [p for p in ps if p > pos]
            verbs.append((str(v), min(after) if after else (min(ps, key=lambda p: abs(p - pos)) if ps else None)))
    if "order" in guards:
        verbs = [(v, p) for v, p in verbs if p is None or p > pos]
    if "adj" in guards and kind == "ger":
        verbs = [(v, p) for v, p in verbs if p == pos + 1]
    if group in SEMI and ("fallback" in guards or "anchor" in guards):
        # the verb right after "to" (or "to be"), if the parser typed it as a predicate
        anchor, q = None, pos + 1
        if q in tok and tok[q][1].lower() == "to":
            q += 1
            if q in tok and tok[q][1].lower() == "be" and q + 1 in tok and tok[q + 1][0].split("/")[1].startswith("P"):
                q += 1
            if q in tok and tok[q][0].split("/")[1].startswith("P"):
                anchor = (tok[q][0], q)
        if anchor and "fallback" in guards and not verbs:
            verbs = [anchor]
        elif anchor and "anchor" in guards and verbs and anchor[1] not in [p for _, p in verbs]:
            verbs = [anchor]          # the parse found an infinitive, but not the one right after the trigger
    if verbs and clause is not None:
        h = phead(clause[0])
        be_before = any(tok.get(r, ("", ""))[1].lower() in ("is", "are", "was", "were", "be", "been", "being")
                        for r in (pos - 1, pos - 2))
        if "passive_ger" in guards and kind == "ger" and ("p" in argroles(h) or (feats(h)[:3] == "<pf" and be_before)):
            return []         # "was enabled using RAG": a gerund after a passive trigger is a means adjunct
        if "needobj" in guards and group == "need" and "o" in argroles(h) and tok.get(pos + 1, ("", ""))[1].lower() != "to":
            return []         # "we need more data to train": object + purpose infinitive
        if "need" in guards and group == "need" and ((len(feats(h)) > 1 and feats(h)[1] == "p") or "p" in argroles(h)):
            return []
    if verbs and "let" in guards and word in ("let", "lets") and (pos == min(tok) or tok.get(pos + 1, ("", ""))[1].lower() in ("us", "'s", "’s")
                                                                   or (verbs[0][0].split("/")[0].lower() in LET_DEFINITIONAL
                                                                       and tok.get((verbs[0][1] or 0) + 1, ("", ""))[1].lower() != "able")
                                                                   or (verbs[0][0].split("/")[0].lower() == "know" and tok.get(pos + 1, ("", ""))[1].lower() in ("me", "us"))):
        return []             # hortative or imperative let: "Let us ...", "let's say", "Let n denote ..."
    seen, out = set(), []
    for v, p in verbs:
        if v not in seen:
            seen.add(v)
            out.append(v.split("/")[0])
    return out


def token_decision(e, atom, pos, groups, a2pos, tok, guards):
    """one decision per token: (group, verbs); infinitive readings before gerund readings"""
    ger_last = os.environ.get("PRECEDENCE", "ger") == "inf"
    inf_first = sorted(groups, key=lambda g: (GROUPS[g][0] in ("ger", "help_in")) == ger_last)
    results = [(g, group_decision(e, atom, pos, g, a2pos, tok, guards)) for g in inf_first]
    if "precedence" in guards:
        for g, v in results:
            if v:
                return g, v
        return None, []
    merged = [(g, v) for g, v in results if v]          # without precedence every reading is applied
    return ("+".join(g for g, _ in merged) or None), [x for _, v in merged for x in v]


def candidates_of_unit(u, groups):
    tok = {q: (a, w) for a, w, q in u["atom2word"]}
    mx = max(tok) if tok else 0
    out = collections.defaultdict(list)
    for a, w, q in u["atom2word"]:
        gs = [g for g in FORM2G.get(w.lower(), []) if g in groups]
        after = " ".join(tok[r][1] for r in range(q + 1, min(mx + 1, q + 10)) if r in tok).strip()
        ok = [g for g in gs if (NEXT.get(g) or NEXT[GROUPS[g][0]]).search(after)]
        if ok:
            out[(a, w, q)] = ok
    return tok, out


def decide_all(u, a, q, gs, tok):
    a2pos = collections.defaultdict(list)
    for at, _, p in u["atom2word"]:
        a2pos[at].append(p)
    e = hedge(u["main_edge"]) if isinstance(u["main_edge"], str) else u["main_edge"]
    full = token_decision(e, a, q, gs, a2pos, tok, set(ALL_GUARDS))
    loo = {}
    for gd in ALL_GUARDS:
        r = token_decision(e, a, q, gs, a2pos, tok, set(ALL_GUARDS) - {gd})
        if r != full:
            loo[gd] = r
    return full, loo


def window(tok, q):
    return " ".join(tok[r][1] for r in range(max(0, q - 12), q) if r in tok) + " [[" + tok[q][1] + "]] " + \
           " ".join(tok[r][1] for r in range(q + 1, q + 16) if r in tok)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pool":
        OUT, groups, shards, maxn, seed = sys.argv[2], sys.argv[3].split(","), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
        sample_groups = groups
        shard_ids = range(50) if shards == "all" else [int(s) for s in shards.split(",")]
        allforms = {w for g in groups for w in GROUPS[g][1].split()}
        quick = re.compile(r"\b(" + "|".join(sorted(allforms, key=len, reverse=True)) + r")\b", re.I)
        pool = collections.defaultdict(list)
        for sid in shard_ids:
            with open(f"{SHDIR}/g2_parsed_{sid:03d}.jsonl") as f:
                for line in f:
                    if not quick.search(line):
                        continue
                    rec = json.loads(line)
                    for s in rec["sentences"]:
                        for u in s.get("units", []):
                            if not u.get("main_edge") or not quick.search(u["text"]):
                                continue
                            tok, cands = candidates_of_unit(u, groups)
                            for (a, w, q), gs in cands.items():
                                if os.environ.get("WORDS") and w.lower() not in os.environ["WORDS"].split(","):
                                    continue
                                for g in gs:           # a token is sampled under each group whose surface pattern it matches
                                    pool[g].append((u, a, w, q, gs, tok))
        rng = random.Random(seed)
        out = []
        for g in sorted(pool):
            if os.environ.get("SAMPLE_POOLS") and g not in os.environ["SAMPLE_POOLS"].split(","):
                continue
            cands = pool[g]
            print(g, "candidates in pool:", len(cands))
            sample = cands if len(cands) <= maxn else rng.sample(cands, maxn)
            for u, a, w, q, gs, tok in sample:
                try:
                    (fg, fv), loo = decide_all(u, a, q, gs, tok)
                except Exception as ex:
                    fg, fv, loo = None, [], {"error": repr(ex)[:80]}
                out.append({"uid": u["uid"], "pool": g, "groups": gs, "word": w, "pos": q, "window": window(tok, q),
                            "group": fg, "verbs": fv, "loo": {k: list(v) for k, v in loo.items()}})
        json.dump(out, open(OUT, "w"), indent=0)
        print("candidates:", len(out), "| merged:", sum(1 for o in out if o["verbs"]))
        print("guards that change the decision:", collections.Counter(k for o in out for k in o["loo"]))
        print("candidates where 2+ guards matter:", sum(1 for o in out if len(o["loo"]) >= 2))
    elif mode == "hard":
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from graphbrain.parsers import create_parser
        parser = create_parser(lang="en", lemmas=True)
        cases = json.load(open(sys.argv[2]))
        out, npass = [], 0
        for c in cases:
            parser.atom2token = {}
            parses = parser.parse(c["text"])["parses"]
            got = []
            for p in parses:
                if p["main_edge"] is None:
                    continue
                words = sorted(([str(at), w_, i_] for at, (w_, i_) in p.get("atom2word", {}).items()), key=lambda x: x[2])
                u = {"main_edge": p["main_edge"], "atom2word": words}
                tok, cands = candidates_of_unit(u, list(GROUPS))
                for (a, w, q), gs in sorted(cands.items(), key=lambda x: x[0][2]):
                    (fg, fv), loo = decide_all(u, a, q, gs, tok)
                    got.append({"trigger": w, "pos": q, "group": fg, "verbs": fv, "loo": {k: list(v) for k, v in loo.items()},
                                "groups": gs})
            got_map = {f"{g['trigger'].lower()}#{i}": g["verbs"] for i, g in enumerate(got)}
            exp = c["expect"]          # list of [trigger word, [verbs]] in sentence order (only for trigger candidates)
            got_list = [[g["trigger"].lower(), g["verbs"]] for g in got]
            ok = got_list == [[t.lower(), v] for t, v in exp]
            npass += ok
            out.append({"id": c["id"], "text": c["text"], "targets": c.get("targets", ""), "expect": exp, "got": got_list,
                        "pass": ok, "detail": got, "edge": str(parses[0]["main_edge"]) if parses else None})
            print(("PASS " if ok else "FAIL ") + c["id"], "| expect", exp, "| got", got_list)
        json.dump(out, open(sys.argv[3], "w"), indent=1)
        print(f"{npass}/{len(cases)} hard cases pass")
