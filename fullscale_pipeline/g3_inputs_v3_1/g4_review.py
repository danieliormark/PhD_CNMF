"""Group-4 model names (focal_terms_v3 'G4': unlisted names accepted by the group-4 rule) — false-hit review for G3 (RL-172;
owner 2026-10-22: false positives must not become focal atoms; a sentence whose only focal term is one is removed).

A random sample of 100 G4 occurrences (60 from spellings seen in one article, 40 from the rest) had 4-5 false hits among
the 60 and 0-1 among the 40; false hits cluster in kinds (gene names, peptide/protein 'Palm' tokens, a physics model
list, words and person names) and real names sometimes carry a glued citation number (SpikeGPT28). This script lists,
for every G4 spelling in the articles kept by g3_inputs_v3_1/scope_exclusions.csv, the candidate rule it falls under
and the spellings of the 700 articles whose only focal terms are G4 names; each with occurrences, articles, an article
title and up to two contexts. The assistant reads the lists and writes decisions/g4_decisions.csv.
    python3 g4_review.py      -> PG/g3_inputs_v3_1/g4/review_rules.txt, review_g4only.txt, g4_types.json (corpus text, RDS only)
"""
import collections, csv, json, os, re

R = "/mnt/hum01-rds/Basov/p91688di/"
PG, L = R + "phase5_graphbrain/", R + "llm_corpus_staging/"
OUT = PG + "g3_inputs_v3_1/g4/"
RULES = {   # candidate kinds (to be checked by reading, not applied blindly)
    "gene_gpt": re.compile(r"^(?:[A-Z][a-z]|[a-z]{1,3})GPT\d{1,2}[a-z]?$"),                 # AtGPT1, OsGPT2, siGPT2 (phosphate translocator genes)
    "palm_chem": re.compile(r"(?:^|[-_])m?Palm\d|^\d+[A-Za-z]+-Palm|[A-Z]FP-Palm|Palm-\d|Palm\d+[A-Z]"),   # Cas10-mPalm2, 177Lu-Palm-3PRGD2
    "physics_list": re.compile(r"_BERT(?:_|$)|^BERT_|Bertini"),                                 # QGSP_BERT_HP
    "caps_word": re.compile(r"^(?:ROBERT|LIBERTY|BERTH\w*|ALBERTA|GILBERT|HERBERT|LAMBERT|NORBERT|HUBERT|BERTRAND|BERTHA)$"),
    "glued_citation": re.compile(r"^(.*(?:GPT|BERT|Bert|BERTa|LLaMA|Llama|Gemini|PaLM|Mistral|Gemma))(\d{2,3})$"),   # SpikeGPT28
}


def main():
    os.makedirs(OUT, exist_ok=True)
    excl = {r["pmcid"] for r in csv.DictReader(open(PG + "g3_inputs_v3_1/scope_exclusions.csv"))}
    E = json.load(open(L + "gap_esummary_cache.json")); T = json.load(open(L + "target_metadata.json"))
    occ, arts, ctx = collections.Counter(), collections.defaultdict(set), collections.defaultdict(list)
    g4only = set()
    for line in open(PG + "g2_v3_1/focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line); pm = r["pmcid"]
        if pm in excl:
            continue
        terms = [t for s in r["sentences"] for t in s["terms"]]
        if any(t.startswith("v3:G4:") for t in terms) and all(t.startswith("v3:G4:") or "transformer" in t for t in terms):
            g4only.add(pm)
        for s in r["sentences"]:
            for t in set(s["terms"]):
                if t.startswith("v3:G4:"):
                    k = t.split(":", 2)[2]; occ[k] += 1; arts[k].add(pm)
                    if len(ctx[k]) < 2:
                        i = s["text"].find(k)
                        ctx[k].append((pm, s["text"][max(0, i - 110):i + len(k) + 90].replace("\n", " ")))
    title = lambda pm: ((E.get(pm) or T.get(pm) or {}).get("title") or "")[:90]
    rule_of = {k: next((n for n, rx in RULES.items() if rx.search(k)), None) for k in occ}

    def block(k):
        out = [f"{k} | {occ[k]} occ, {len(arts[k])} art | rule {rule_of[k]} | T: {title(ctx[k][0][0])}"]
        out += [f"      {pm}: {c}" for pm, c in ctx[k]]
        return "\n".join(out)

    with open(OUT + "review_rules.txt", "w", encoding="utf-8") as f:
        for n in RULES:
            ks = sorted((k for k in occ if rule_of[k] == n), key=lambda k: -occ[k])
            f.write(f"######## {n}: {len(ks)} spellings, {sum(occ[k] for k in ks)} occurrences\n")
            f.write("\n".join(block(k) for k in ks) + "\n")
    only_types = sorted({k for k in occ if arts[k] & g4only and not rule_of[k]}, key=lambda k: (-len(arts[k] & g4only), k))
    with open(OUT + "review_g4only.txt", "w", encoding="utf-8") as f:
        f.write(f"######## spellings in the {len(g4only)} G4-only articles not under a rule: {len(only_types)}\n")
        f.write("\n".join(block(k) for k in only_types) + "\n")
    json.dump(dict(occ=occ, arts={k: sorted(v) for k, v in arts.items()}, rule=rule_of, g4only=sorted(g4only)), open(OUT + "g4_types.json", "w"))
    print("types", len(occ), "occ", sum(occ.values()), "G4-only articles", len(g4only),
          collections.Counter(v for v in rule_of.values() if v), "reading list", len(only_types))


if __name__ == "__main__":
    main()
