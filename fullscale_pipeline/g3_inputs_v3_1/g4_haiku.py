"""Group-4 review by Haiku readers (RL-173; owner 2026-10-22: for rigour, the 4,180 group-4 spellings that neither a
rule nor the assistant's reading covered in RL-172 are each judged by a reader). Rubric G4_READER_RUBRIC.md.
Items: one spelling, its occurrences and articles in the kept corpus, up to three contexts from different articles,
each with the article title. Calibration first, on 72 spellings the assistant already decided in RL-172 (not among the
rubric's examples): 36 model, 36 not_model.
    python3 g4_haiku.py build   -> PG/g3_inputs_v3_1/g4/haiku/: RUBRIC.md, calib.txt, calib.json, batch_NN.txt, batch_NN.json
"""
import collections, csv, json, os, random, sys

R = "/mnt/hum01-rds/Basov/p91688di/"
PG, L = R + "phase5_graphbrain/", R + "llm_corpus_staging/"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = PG + "g3_inputs_v3_1/g4/haiku/"
BATCH = 250
GOLD_NOT = ("StGPT1 OsGPT1 rGPT1 KfGPT1 QgGPT7 pGPTVII SiANGPT2 177Lu-Palm-3PRGD2 Cas10-mPalm2 ND-Palm HF-Palm EK35S-Palm MDDPalm "
            "CKSAAP-Palm NBA-Palm HygroPalm23‐A ePalm BlazePalm VILBERT-LOURMAT Raker-Roberta PEDOT_GPTMS AGPTi FTFP_BERT_HP "
            "order_BERT_HP r-GPTSb CGPT.15 TM_CGPT LIBERTYLUNG BERTIE PUBERTAL PGPT_pred isGPT ROBERTO Roberta-Maria HMW-GS-GPT "
            "GPTV_DLR").split()
GOLD_MODEL = ("KoGPT2 RoGPT2 KnGPT2 ViGPT2 cGPT SafeGPT AMBERT BERTH HUBERT BERTHop MethylBERT MethylGPT polyBERT BERTax CodeBERT "
              "GraphCodeBERT BERT4Rec BERT4Bitter ThyGPT SkinGPT-4 PlantGPT ProteoGPT PBertKla CodonBERT YieldBERT vBERT TweetyBERT "
              "EpiBERT CycleGPT FragGPT DialoGPT UmBERTo BERTurk SetBERT Fed_BERT_MSCNN WE_BERT_CRF").split()


def collect(want):
    excl = {r["pmcid"] for r in csv.DictReader(open(PG + "g3_inputs_v3_1/scope_exclusions.csv"))}
    E = json.load(open(L + "gap_esummary_cache.json")); T = json.load(open(L + "target_metadata.json"))
    occ, arts, ctx = collections.Counter(), collections.defaultdict(set), collections.defaultdict(list)
    for line in open(PG + "g2_v3_1/focal_sentences_v2.jsonl", encoding="utf-8"):
        r = json.loads(line); pm = r["pmcid"]
        if pm in excl:
            continue
        for s in r["sentences"]:
            for t in set(s["terms"]):
                if not t.startswith("v3:G4:"):
                    continue
                k = t.split(":", 2)[2]
                if k not in want:
                    continue
                occ[k] += 1
                if pm not in arts[k] and len(ctx[k]) < 3:
                    i = s["text"].find(k)
                    title = ((E.get(pm) or T.get(pm) or {}).get("title") or "")[:120]
                    ctx[k].append((title, s["text"][max(0, i - 160):i + len(k) + 120].replace("\n", " ")))
                arts[k].add(pm)
    return occ, arts, ctx


def render(items, path, title, occ, arts, ctx):
    lines = [f"# {title}: {len(items)} spellings", ""]
    for qid, k in items:
        lines.append(f"## {qid} | spelling: {k} | {occ[k]} occurrences in {len(arts[k])} articles")
        for j, (t, c) in enumerate(ctx[k], 1):
            lines.append(f"Context {j} (article: {t}): …{c}…")
        lines.append(f"QUESTION {qid}: what does the spelling «{k}» refer to in these contexts — model, not_model, mixed or unsure?")
        lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def build():
    os.makedirs(OUT, exist_ok=True)
    assert not os.path.exists(OUT + "calib.json"), "outputs exist"
    G = json.load(open(PG + "g3_inputs_v3_1/g4/g4_types.json"))
    dec = {r["spelling"] for r in csv.DictReader(open(os.path.join(HERE, "decisions", "g4_decisions.csv")))}
    read = set()
    for l in open(PG + "g3_inputs_v3_1/g4/review_g4only.txt", encoding="utf-8"):
        if l and not l.startswith((" ", "#")) and " | " in l:
            read.add(l.split(" | ")[0])
    rest = sorted(k for k in G["occ"] if k not in dec and k not in read and not G["rule"].get(k))
    gold = {**{k: "not_model" for k in GOLD_NOT}, **{k: "model" for k in GOLD_MODEL}}
    occ, arts, ctx = collect(set(rest) | set(gold))
    missing = [k for k in gold if not ctx[k]]
    assert not missing, f"gold spellings without context: {missing}"
    import shutil
    shutil.copy(os.path.join(HERE, "G4_READER_RUBRIC.md"), OUT + "RUBRIC.md")
    g = sorted(gold); random.seed(5); random.shuffle(g)
    calib = [(f"C{i:03d}", k) for i, k in enumerate(g, 1)]
    render(calib, OUT + "calib.txt", "Calibration", occ, arts, ctx)
    json.dump({q: dict(spelling=k, gold=gold[k]) for q, k in calib}, open(OUT + "calib.json", "w"), indent=1)
    random.seed(17); random.shuffle(rest)
    nb = -(-len(rest) // BATCH)
    for b in range(nb):
        part = rest[b * BATCH:(b + 1) * BATCH]
        items = [(f"G{b + 1:02d}.{i:03d}", k) for i, k in enumerate(part, 1)]
        render(items, OUT + f"batch_{b + 1:02d}.txt", f"Batch {b + 1:02d}", occ, arts, ctx)
        json.dump({q: dict(spelling=k, occ=occ[k], arts=len(arts[k])) for q, k in items}, open(OUT + f"batch_{b + 1:02d}.json", "w"), indent=1)
    print("spellings to judge", len(rest), "occurrences", sum(occ[k] for k in rest), "batches", nb, "| gold", len(gold))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
