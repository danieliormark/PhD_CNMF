"""Pre-M1 audit of the merged G3 v2 output (RL-176). Independent of check_record (which the merge ran on every unit):
cross-file consistency, sentence-level parent/cousin rules (D6, item 11), the input lists applied, word overlap between
structures, and atom/text quality (lemmas, stop words, junk symbols, non-prose text). Writes a JSON report with counts
and samples for reading.
    python audit_g3_v2.py [--out audit_g3_v2.json]
"""
import argparse, collections, csv, hashlib, json, random, re

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
OUT = PG + "g3_v2/"
INP = PG + "g3_inputs_v3_1/"
STOP = {w.strip() for w in open(PG + "nltk_abridged_stopwords_list.txt", encoding="utf-8") if w.strip()}
STOP |= {'such', 'other', 'many', 'some', 'any', 'own', 'same', 'few', 'several', 'this', 'that', 'these', 'those', 'whilst'}
STOP -= {'not', 'no', 'never', "n't", 'nor'}
random.seed(0)


def h12(i, e):
    return hashlib.sha1(f"{i}|{e}".encode("utf-8")).hexdigest()[:12]


def split_atom(a):
    root, typ, lang = a.rsplit("/", 2) if a.count("/") >= 2 else (a, "", "")
    return root, typ, lang


class Sample:
    """reservoir of k examples per key"""
    def __init__(self, k=12):
        self.k, self.n, self.items = k, collections.Counter(), collections.defaultdict(list)

    def add(self, key, item):
        self.n[key] += 1
        if len(self.items[key]) < self.k:
            self.items[key].append(item)
        elif random.random() < self.k / self.n[key]:
            self.items[key][random.randrange(self.k)] = item


JUNK = [("ref_or_url_token", re.compile(r"^(ref\d{6}|url\d{8})$", re.I)),
        ("digit", re.compile(r"\d")),
        ("uppercase", re.compile(r"[A-Z]")),
        ("non_ascii", re.compile(r"[^\x00-\x7f]")),
        ("symbol", re.compile(r"[^A-Za-z0-9_\-'.À-ɏ]")),
        ("edge_punct", re.compile(r"^[\-_'.]|[\-_'.]$")),
        ("very_long", re.compile(r"^.{31,}$"))]
TEXT = [("table_or_tab", re.compile(r"\t| \| |\|.*\|")),
        ("equation", re.compile(r"(=.*){2,}|\\(frac|sum|mathrm|left|right)|\^\{")),
        ("html_entity", re.compile(r"&(amp|lt|gt|nbsp|#\d+);")),
        ("markup", re.compile(r"</?[a-z]+[ >]|\[\[|\]\]|\{\{")),
        ("url", re.compile(r"https?://|www\.")),
        ("figure_label", re.compile(r"^(Fig\.?|Figure|Table|Supplementary)\s*S?\d", re.I)),
        ("code_like", re.compile(r"[;{}]\s*$|\bdef \w+\(|\breturn\b.*;|==|!=|\+\+"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit_g3_v2.json")
    a = ap.parse_args()
    C = collections.Counter(); S = Sample(); top = collections.defaultdict(collections.Counter)

    # lists
    excl = {r["pmcid"] for r in csv.DictReader(open(INP + "scope_exclusions.csv", encoding="utf-8"))}
    stmt = {r["sid"] for r in csv.DictReader(open(INP + "statement_sentences.csv", encoding="utf-8"))}
    rej_s, rej_a = collections.defaultdict(set), collections.defaultdict(set)
    for r in csv.DictReader(open(INP + "term_rejections.tsv", encoding="utf-8"), delimiter="\t"):
        (rej_s[r["key"]] if r["scope"] == "sentence" else rej_a[r["key"]]).add(r["term"].lower())
    g4 = {}
    for r in csv.DictReader(open(INP + "g4_decisions.csv", encoding="utf-8")):
        g4[(r["spelling"], r["pmcid"])] = r["action"]
    status = {}
    for l in open(OUT + "g3_articles_v2.jsonl"):
        r = json.loads(l); status[r["pmcid"]] = r
    C["articles"] = len(status)
    for p, r in status.items():
        C["status_" + r["status"]] += 1
        if (r["status"] == "excluded") != (p in excl):
            C["PROBLEM_exclusion_mismatch"] += 1

    ids = set(); seen_uid = set(); n_struct = 0
    art_par = collections.Counter(); art_units = collections.Counter()
    cur_sid, sent = None, []

    def close_sentence(recs):
        pids = [p["id"] for r in recs for p in r["parents"]]
        if not pids:
            C["PROBLEM_sentence_without_parent"] += 1; S.add("PROBLEM_sentence_without_parent", recs[0]["uid"])
        keys = collections.Counter(p["content_key"] for r in recs for p in r["parents"])
        if any(v > 1 for v in keys.values()):
            C["sentences_same_parent_content_twice"] += 1
            S.add("same_parent_content_twice", [r["text"] for r in recs][:2])
        if len({r["uid"] for r in recs if r["parents"]}) > 1:
            C["sentences_parents_in_2plus_units"] += 1
        np_ = len(pids); C[f"sentence_parents_{min(np_, 5)}"] += 1

    for l in open(OUT + "g3_test_v2.jsonl"):
        r = json.loads(l); C["units"] += 1
        uid, sid, pm = r["uid"], r["sid"], r["pmcid"]
        if uid in seen_uid: C["PROBLEM_uid_duplicate"] += 1
        seen_uid.add(uid)
        if not (uid.startswith(sid + ".U") and sid.startswith(pm + ".")): C["PROBLEM_id_prefix"] += 1
        st = status.get(pm, {}).get("status")
        if st != "valid": C[f"PROBLEM_unit_of_{st}_article"] += 1
        if sid in stmt: C["PROBLEM_statement_sentence_kept"] += 1
        if sid != cur_sid:
            if sent: close_sentence(sent)
            if sid in {x["sid"] for x in sent[:0]}: pass
            cur_sid, sent = sid, []
        sent.append(r)
        art_units[pm] += 1; art_par[pm] += len(r["parents"])
        if not r["parents"]:
            C["units_without_parent"] += 1
            if not r["cousins"]: C["PROBLEM_unit_without_parent_or_cousin"] += 1
        # focal mentions and the lists
        for m in r["focal_mentions"]:
            C["focal_mentions"] += 1; top["canonical"][m["canonical"]] += 1
            t = m["term"].lower()
            if t in rej_s.get(sid, ()) or t in rej_a.get(pm, ()):
                C["PROBLEM_rejected_term_kept"] += 1; S.add("rejected_term_kept", [uid, m["surface"]])
            d = g4.get((m["surface"], pm)) or g4.get((m["surface"], ""))
            if d == "reject":
                C["PROBLEM_g4_rejected_kept"] += 1; S.add("g4_rejected_kept", [uid, m["surface"]])
            if re.search(r"(base|large|small|uncased|cased|instruct|chat|turbo)$|_\d{3,}", m["canonical"]):
                S.add("canonical_suspicious", [m["surface"], m["canonical"]]); top["canonical_suspicious"][m["canonical"]] += 1
        # text quality
        for name, rx in TEXT:
            if rx.search(r["text"]):
                C["text_" + name] += 1; S.add("text_" + name, r["text"][:300])
        letters = sum(ch.isalpha() for ch in r["text"])
        if len(r["text"]) > 0 and letters / len(r["text"]) < 0.6:
            C["text_low_letter_share"] += 1; S.add("text_low_letter_share", r["text"][:300])
        ntok = len(r["text"].split())
        if ntok < 4: C["text_under_4_tokens"] += 1; S.add("text_under_4_tokens", r["text"])
        if ntok > 120: C["text_over_120_tokens"] += 1
        # structures
        pos_use = collections.Counter()
        par_ids = {p["id"] for p in r["parents"]}
        for p in r["parents"]:
            ids.add(p["id"]); n_struct += 1; C["parents"] += 1
            if p.get("embedded_in"): C["parents_embedded"] += 1
            nf = sum(c["kind"] == "focal_he" for c in p["children"])
            C[f"parent_focal_he_{min(nf, 3)}"] += 1
            if not any(c["kind"] == "sibling_he" for c in p["children"]): C["parents_without_sibling"] += 1
            dummy = p["children"][0]
            if not dummy["atoms"]: C["parents_empty_verb_group"] += 1
            for c in p["children"]:
                ids.add(c["id"]); n_struct += 1; C["children"] += 1
                if not c["atoms"]: C["PROBLEM_empty_child"] += 1
        for c in r["cousins"]:
            ids.add(c["id"]); n_struct += 1; C["cousins"] += 1
            if not c["atoms"]: C["PROBLEM_empty_cousin"] += 1
            if not set(c["parents"]) or any(not q.startswith(sid + ".U") for q in c["parents"]):
                C["PROBLEM_cousin_parent_outside_sentence"] += 1
            C["cousin_parent_links"] += len(c["parents"])
            for f in c.get("also_in", []): C["cousin_also_in"] += 1
        structs = [(c, p["id"]) for p in r["parents"] for c in p["children"]] + [(c, None) for c in r["cousins"]]
        for c, pid in structs:
            for at in c["atoms"]:
                root, typ, _ = split_atom(at["atom"]); C["atoms"] += 1
                for x in at["src"]:
                    if x[2] is not None: pos_use[x[2]] += 1
                if typ == "C" and at["atom"].endswith("/focal"):
                    if pid is None: C["PROBLEM_focal_atom_in_cousin"] += 1
                    continue
                top["root"][root] += 1
                if root.lower() in STOP and root not in ("be", "have", "do"):
                    C["atom_stopword"] += 1; top["atom_stopword"][root] += 1
                for name, rx in JUNK:
                    if rx.search(root):
                        C["atom_" + name] += 1; top["atom_" + name][root] += 1
                        S.add("atom_" + name, [root, [x[1] for x in at["src"]][:3], r["text"][:200]])
                if len(root.replace("_", "")) <= 1: C["atom_single_char"] += 1; top["atom_single_char"][root] += 1
                # lemma checks on the source tokens
                for orig, word, pos in at["src"]:
                    ot = orig.rsplit("/", 2)[1] if orig.count("/") >= 2 else ""
                    w = (word or "").lower()
                    if ot.startswith("Cc.p") and root == w and w.endswith("s") and not w.endswith(("ss", "us", "is")):
                        C["lemma_plural_unchanged"] += 1; top["lemma_plural_unchanged"][root] += 1
                    if ot.startswith(("P", "Mv")) and root == w and re.search(r"(ed|ing)$", w) and len(w) > 5:
                        C["lemma_verb_form_unchanged"] += 1; top["lemma_verb_form_unchanged"][root] += 1
                    if root != w and "_" not in root and w and root[:1] != w[:1]:
                        C["lemma_first_letter_changed"] += 1; top["lemma_first_letter_changed"][f"{w}->{root}"] += 1
                    if random.random() < 0.0005:
                        S.add("lemma_random", [w, ot, root], ) if S.n["lemma_random"] < 10 ** 9 else None
        shared = [q for q, k in pos_use.items() if k > 1]
        if shared:
            C["units_with_token_in_2plus_structures"] += 1; C["tokens_in_2plus_structures"] += len(shared)
            if len(S.items["token_in_2plus_structures"]) < 12 or random.random() < 0.001:
                words = r["text"].split()
                S.add("token_in_2plus_structures", [uid, r["text"][:250], [p["edge"][:250] for p in r["parents"]][:3],
                                                   [c["edge"][:150] for c in r["cousins"]][:3]])
    if sent: close_sentence(sent)
    S.k = 40
    for p, r in status.items():
        if r["status"] == "valid" and art_par[p] == 0: C["PROBLEM_valid_article_without_parent"] += 1
        if r["status"] != "valid" and art_units[p]: C["PROBLEM_nonvalid_article_with_units"] += 1
        if r["status"] == "valid" and (r["units"], r["parents"]) != (art_units[p], art_par[p]):
            C["PROBLEM_article_counts_mismatch"] += 1
    # occurrence file against the structures
    occ_n = 0; roles = collections.Counter()
    for l in open(OUT + "g3_occurrences_v2.jsonl"):
        o = json.loads(l); occ_n += 1; roles[o["role"]] += 1
        if o["occ"] not in ids: C["PROBLEM_occurrence_unknown_id"] += 1
        if not o["occ"].startswith(o["uid"]) or not o["uid"].startswith(o["sid"]): C["PROBLEM_occurrence_ids"] += 1
        for typ, tgt in o["links"]:
            C["link_" + typ] += 1
            if tgt not in ids: C["PROBLEM_link_target_unknown"] += 1
    C["occurrence_rows"] = occ_n; C["structures"] = n_struct
    if occ_n != n_struct: C["PROBLEM_occurrences_vs_structures"] = occ_n - n_struct
    rep = dict(counts=dict(sorted(C.items())), problems={k: v for k, v in C.items() if k.startswith("PROBLEM")},
               occurrence_roles=dict(roles),
               top={k: v.most_common(60) for k, v in top.items() if k != "root"},
               n_distinct_roots=len(top["root"]), samples=dict(S.items))
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep["problems"]), "units", C["units"], "->", a.out)


if __name__ == "__main__":
    main()
