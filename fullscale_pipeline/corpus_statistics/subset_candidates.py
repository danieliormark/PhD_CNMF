# RL-193 (owner 2026-10-26): candidate disciplinary subsets of the working corpus v2 for the first M1 trial. Owner:
# a discipline-based subset, since it may show both model-based and subdiscipline-based communities. Read-only; no
# network. Per article of working_corpus_v2_pmcids.txt:
#   - OpenAlex primary topic (topic -> subfield -> field -> domain, with its score) and the fields of its other topics
#     (openalex_topics.json), for the work chosen as in author_coverage.py --openalex refresh --corpus v2;
#   - NLM journal (PubMed) with its Broad Subject Terms and catalog MeSH (nlm_journals.json, nlm_journals_fetch.py);
#   - year (PubMed electronic year, else print year);
#   - model families named in its G3 units (focal_mentions -> family_map.family_of_term; per-article cache
#     diagnostics/author_coverage/article_families_v2.json, rebuilt with --rebuild-families), G3 units and parents.
#     family_of_term maps the v2 names only and returns the v3 matcher's terms (v3:<class>:<token>) and the hub spellings
#     (hub:<name>) unchanged, so the cache holds those raw; group() folds them for this table only (not a corpus rule):
#     vendor model lines to their family (ChatGPT4, gpt-oss-120b -> GPT (OpenAI); claude-3-opus -> Claude ...), BERT /
#     RoBERTa spellings, plurals, checkpoints and cited forms (Bert, BERTs, bert-base-uncased, BERT21) to bare BERT /
#     RoBERTa, "language-model" to the generic term; every other named model is its own family, case ignored and a
#     version after a hyphen dropped (ProtBert = ProtBERT; ESM-1b, ESM-2 -> ESM; DNABERT-2 -> DNABERT);
#   - bylines: with OpenAlex ID or ORCID (bylines_v2_refresh.jsonl), with an OpenAlex institution ID on the aligned
#     authorship.
# Writes corpus_statistics/subset_candidates_by_article.csv (no names) and subset_candidates.json (one row per
# candidate: OpenAlex field, OpenAlex subfield, NLM Broad Subject Term), printed as tables.
import ast, collections, csv, glob, json, os, re, statistics, sys

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
HERE, D, OD = FP + "corpus_statistics/", FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v2/"
sys.path.insert(0, HERE)
from family_map import FAMILY, family_of_term

wc = [l.strip() for l in open(HERE + "working_corpus_v2_pmcids.txt") if l.strip()]
WC = set(wc)
H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
V = json.load(open(D + "openalex_verified.json")); V.update(json.load(open(D + "openalex_verified_new.json")))
OW = json.load(open(OD + "openalex_works.json"))
TOP = json.load(open(OD + "openalex_topics.json"))
PM = json.load(open(OD + "pubmed_authors.json"))
NLM = json.load(open(OD + "nlm_journals.json"))


def work(p):                                    # same choice as author_coverage.py --openalex refresh
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


GENERIC = lambda f: f.endswith("(generic term)")
VENDOR = [(r"^(chat[ -]?gpt|gpt[ -]?(\d|o\d|oss|4o)|o[134](-|$)|openai)", "GPT (OpenAI)"), (r"^claude", "Claude (Anthropic)"),
          (r"^gemini", "Gemini (Google)"), (r"^gemma", "Gemma (Google)"), (r"^bard", "Bard (Google)"), (r"^palm", "PaLM (Google)"),
          (r"^(meta[ -])?llama", "Llama (Meta)"), (r"^(mistral|mixtral)", "Mistral"), (r"^qwen", "Qwen"),
          (r"^deepseek", "DeepSeek"), (r"^grok", "Grok (xAI)"), (r"^copilot", "Copilot (Microsoft/GitHub)"),
          (r"^gpts$", "GPT (bare/unspecified)"), (r"^(bert|berts|bert\d{2})($|[-_ ](base|large|small|tiny|mini|uncased|cased|multilingual))", "BERT (bare/unspecified)"),
          (r"^(roberta|robertas|roberta\d{2})($|[-_ ](base|large))", "RoBERTa (generic/base)"), (r"^language[-_ ]models?$", "language model (generic term)")]


FAMILY_LABELS = set(FAMILY.values())


def group(f):
    if f in FAMILY_LABELS:
        return f
    tok = f.split(":", 2)[-1] if f.startswith("v3:") else f[4:] if f.startswith("hub:") else f
    low = tok.lower()
    for pat, fam in VENDOR:
        if re.match(pat, low):
            return fam
    low = re.sub(r"[-_ ](base|large|small|tiny|mini|cased|uncased|multilingual)\b.*$", "", low)
    return re.sub(r"[-_ ]v?\d[\w.]*$", "", low).replace("-", "").replace("_", "").replace(" ", "") or low
FAM_CACHE = OD + "article_families_v2.json"
if os.path.exists(FAM_CACHE) and "--rebuild-families" not in sys.argv:
    FAM = json.load(open(FAM_CACHE))
else:
    FAM = {}
    for f in sorted(glob.glob(PG + "shards/g3_test_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["pmcid"] not in WC:
                continue
            a = FAM.setdefault(r["pmcid"], dict(units=0, parents=0, fam={}))
            a["units"] += 1; a["parents"] += len(r["parents"])
            fm = r["focal_mentions"] or []
            for m in ast.literal_eval(fm) if isinstance(fm, str) else fm:
                fam = family_of_term(m["term"])
                a["fam"][fam] = a["fam"].get(fam, 0) + 1
        print("families:", f.rsplit("/", 1)[-1], len(FAM), flush=True)
    json.dump(FAM, open(FAM_CACHE, "w"))

BY = collections.defaultdict(list)
for l in open(OD + "bylines_v2_refresh.jsonl"):
    b = json.loads(l); BY[b["pmcid"]].append(b)

rows = []
for p in wc:
    w, h = work(p), H.get(p) or {}
    t = TOP.get(w["id"].rsplit("/", 1)[-1]) if w else None
    pr = (t or {}).get("primary") or {}
    pm = PM.get(h.get("pmid") or "") or {}
    j = (pm.get("journal") or {}).get("nlm_id")
    nj = NLM.get(j) or {}
    fams = collections.Counter()
    for f, c in ((FAM.get(p) or {}).get("fam") or {}).items():
        fams[group(f)] += c
    named = {f: c for f, c in fams.items() if not GENERIC(f)}
    auth = (w or {}).get("authorships") or []
    bl = BY.get(p, [])
    inst = sum(1 for b in bl if b["aligned"] and b["oa_idx"] is not None and b["oa_idx"] < len(auth)
               and any(i.get("id") for i in auth[b["oa_idx"]].get("institutions") or []))
    rows.append(dict(
        pmcid=p, year=str(pm.get("epub_year") or pm.get("pub_year") or (w or {}).get("year") or ""),
        domain=(pr.get("domain") or {}).get("name") or "", field=(pr.get("field") or {}).get("name") or "",
        subfield=(pr.get("subfield") or {}).get("name") or "", topic=pr.get("name") or "",
        topic_score=pr.get("score") if pr else "",
        other_topic_fields="|".join((x.get("field") or {}).get("name") or "" for x in (t or {}).get("topics", [])[1:]),
        nlm_id=j or "", nlm_broad="|".join(nj.get("broad") or []),
        nlm_mesh_major="|".join(m["term"] for m in nj.get("mesh") or [] if m["major"]),
        units=(FAM.get(p) or {}).get("units", 0), parents=(FAM.get(p) or {}).get("parents", 0),
        families="|".join(f"{f}={c}" for f, c in sorted(fams.items(), key=lambda x: -x[1])),
        main_family=max(named, key=lambda f: (named[f], f)) if named else "(generic terms only)",
        n_bylines=len(bl), n_id_or_orcid=sum(1 for b in bl if b["oa_id"] or b["pm_orcid"] or b["oa_orcid"]),
        n_inst_id=inst,
        oa_author_ids="|".join(sorted({b["oa_id"].rsplit("/", 1)[-1] for b in bl if b["oa_id"]}))))
with open(HERE + "subset_candidates_by_article.csv", "w", newline="") as f:
    cw = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "oa_author_ids"], extrasaction="ignore")
    cw.writeheader(); cw.writerows(rows)


def describe(group):          # 'group' is the list of article rows here
    n = len(group)
    yrs = collections.Counter(r["year"] for r in group)
    fam_art = collections.Counter(f.rsplit("=", 1)[0] for r in group for f in r["families"].split("|") if f)
    main = collections.Counter(r["main_family"] for r in group)
    bl = sum(r["n_bylines"] for r in group)
    persons = collections.Counter(a for r in group for a in r["oa_author_ids"].split("|") if a)
    journals = collections.Counter(r["nlm_id"] for r in group if r["nlm_id"])
    subf = collections.Counter(r["subfield"] for r in group)
    scores = [r["topic_score"] for r in group if r["topic_score"] != ""]
    same = [sum(x == r["field"] for x in r["other_topic_fields"].split("|")) / len(r["other_topic_fields"].split("|"))
            for r in group if r["other_topic_fields"]]
    return dict(
        articles=n, units=sum(r["units"] for r in group), parents=sum(r["parents"] for r in group),
        years={y: yrs[y] for y in sorted(yrs)}, share_2024_on=round(sum(c for y, c in yrs.items() if str(y) >= "2024") / n, 3),
        main_family_top=[(f, c) for f, c in main.most_common(6)],
        families_with_30_articles=sum(1 for f, c in fam_art.items() if c >= 30 and not GENERIC(f)),
        gpt_main_share=round(main["GPT (OpenAI)"] / n, 3), generic_only_share=round(main["(generic terms only)"] / n, 3),
        bylines=bl, byline_id_or_orcid=round(sum(r["n_id_or_orcid"] for r in group) / max(bl, 1), 3),
        byline_inst_id=round(sum(r["n_inst_id"] for r in group) / max(bl, 1), 3),
        articles_all_identified=round(sum(r["n_id_or_orcid"] == r["n_bylines"] > 0 for r in group) / n, 3),
        oa_persons=len(persons), oa_persons_2plus_articles=sum(1 for c in persons.values() if c >= 2),
        journals=len(journals), journals_5plus=sum(1 for c in journals.values() if c >= 5),
        subfields=len(subf), subfield_top=[(s, c) for s, c in subf.most_common(6)],
        median_topic_score=round(statistics.median(scores), 3) if scores else None,
        other_topics_same_field=round(statistics.mean(same), 3) if same else None,
        nlm_broad_any=round(sum(bool(r["nlm_broad"]) for r in group) / n, 3),
        nlm_broad_top=collections.Counter(b for r in group for b in r["nlm_broad"].split("|") if b).most_common(5))


out = dict(corpus=describe(rows), field={}, subfield={}, nlm_broad={})
for key, store in (("field", out["field"]), ("subfield", out["subfield"])):
    g = collections.defaultdict(list)
    for r in rows:
        g[r[key] or "(no OpenAlex topic)"].append(r)
    for k, grp in sorted(g.items(), key=lambda x: -len(x[1])):
        if len(grp) >= 300 or key == "field":
            store[k] = dict(describe(grp), field=collections.Counter(r["field"] for r in grp).most_common(1)[0][0])
g = collections.defaultdict(list)
for r in rows:
    for b in r["nlm_broad"].split("|") if r["nlm_broad"] else ["(no Broad Subject Term)"]:
        g[b].append(r)
for k, grp in sorted(g.items(), key=lambda x: -len(x[1])):
    if len(grp) >= 300:
        out["nlm_broad"][k] = dict(describe(grp), oa_field_top=collections.Counter(r["field"] for r in grp).most_common(4))
json.dump(out, open(HERE + "subset_candidates.json", "w"), indent=1)

c = out["corpus"]
print(f"corpus: {c['articles']} articles, NLM Broad Subject Term on {c['nlm_broad_any']}, median topic score {c['median_topic_score']}")
for level in ("field", "subfield", "nlm_broad"):
    print(f"\n== {level} ==")
    print(f"{'candidate':45s} {'art':>6s} {'units':>7s} {'24+':>5s} {'GPTm':>5s} {'gen':>5s} {'fam30':>5s} {'idOR':>5s} "
          f"{'inst':>5s} {'pers2+':>6s} {'jour5+':>6s} {'subf':>4s} {'tscore':>6s} {'sameF':>5s}")
    for k, d in out[level].items():
        print(f"{k[:45]:45s} {d['articles']:6d} {d['units']:7d} {d['share_2024_on']:5.2f} {d['gpt_main_share']:5.2f} "
              f"{d['generic_only_share']:5.2f} {d['families_with_30_articles']:5d} {d['byline_id_or_orcid']:5.2f} "
              f"{d['byline_inst_id']:5.2f} {d['oa_persons_2plus_articles']:6d} {d['journals_5plus']:6d} {d['subfields']:4d} "
              f"{d['median_topic_score'] or 0:6.3f} {d['other_topics_same_field'] or 0:5.2f}")
