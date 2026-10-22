"""§6 item 6 decisions for the articles screened by screen_item6.py (RL-171): the automatic ones by the owner's rules of
2026-10-08, and the assistant's reading of the rest (2026-10-22, contexts in item6/screen.json). Writes
decisions/item6_v3_1.csv (pmcid, corpus, rule, decision, label, note); exclude / keep / restore (an old exclusion lifted).
Old decisions (g3_scope_exclusions/decisions/) stand for every article not listed here.
    python3 write_item6_decisions.py
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_inputs_v3_1/item6/screen.json"
OLD_TRANSFORMER = os.path.join(HERE, "..", "g3_scope_exclusions", "decisions", "transformer_only.csv")

# ---- the assistant's reading (rule 1: "LLM" defined otherwise)
CLASS3 = {
    "PMC8512253": ("exclude", "homonym", "Logic Learning Machine"),
    "PMC11405331": ("exclude", "homonym", "Low-Low-Medium (landscape class)"),
    "PMC11255759": ("exclude", "homonym", "lipid-lowering medications"),
    "PMC11862180": ("exclude", "homonym", "Least Learning Machine"),
    "PMC13351568": ("keep", "misnomer corrected", "'large linguistic model (LLM)' beside ChatGPT: corrected to language"),
}
# ---- rule 3: names (ESM-2 homonyms: climate models and Electronic Supplementary Material; the rest the protein LM)
ESM2_HOMONYM = {
    "PMC9816114": "climate model MRI-ESM-2.0", "PMC11492982": "Electronic Supplementary Material", "PMC11890605": "climate model FIO-ESM-2-0",
    "PMC11638425": "Electronic Supplementary Material", "PMC10468057": "climate model AWI-ESM-2.1", "PMC13415282": "climate model FIO-ESM-2-0",
    "PMC13214739": "climate model MRI-ESM-2.0", "PMC10719290": "climate model FIO-ESM-2-0", "PMC12255103": "Electronic Supplementary Material",
    "PMC8864054": "Electronic Supplementary Material", "PMC10556232": "climate model MRI-ESM-2.0", "PMC11026321": "Electronic Supplementary Material",
    "PMC13575070": "Electronic Supplementary Material", "PMC9341446": "climate model MRI-ESM-2.0", "PMC11514610": "climate model FIO-ESM-2",
    "PMC12602602": "Electronic Supplementary Material", "PMC11772660": "climate model MRI-ESM-2", "PMC8802252": "Electronic Supplementary Material",
    "PMC11554900": "climate model FIO-ESM-2-0", "PMC9361993": "Electronic Supplementary Material", "PMC11050065": "climate model FIO-ESM-2-0",
    "PMC12371055": "climate model MRI-ESM-2-0",
}
PROGEN = {"PMC12946364": ("keep", "model", "the protein language model ProGen (with Evo)"),
          "PMC12148599": ("exclude", "homonym", "ProGen multisite genetics trial"),
          "PMC10341321": ("exclude", "homonym", "DermLite ProGen dermatoscope"),
          "PMC10138877": ("exclude", "homonym", "ProGen antibody supplier"),
          "PMC9174056": ("exclude", "homonym", "ProGen Biotech antibody supplier")}
# BioBridge: all 47 read, all homonyms (collagen scaffold, rib implant, the stem-cell 'biobridge', BioBridge Global, suppliers)
# ---- rule 4: transformer only, LLM-related hits in the analysed text
TRANSFORMER_READ = {
    "PMC13646168": ("keep", "use or compare", "BERTopic used for topic analysis (BERTopic kept in the old screen too)"),
    "PMC13597719": ("keep", "use or compare", "XLNet+SVM model used"),
    "PMC13655001": ("exclude", "passing mention", "foundation / vision-language models listed"),
    "PMC13639930": ("exclude", "vision-language model only", "VLMs in related work"),
    "PMC13603927": ("exclude", "passing mention", "EEGPT listed in a review"),
}
AIUSE = ["PMC12459426", "PMC13064411", "PMC13517445", "PMC13214530", "PMC11784834", "PMC12609805", "PMC12456844", "PMC13406586",
         "PMC13585275", "PMC13071485", "PMC13603191", "PMC13122584", "PMC13549228", "PMC13477592"]   # 14 incl. 'No ... such as ChatGPT or Claude'
for pm in AIUSE:
    TRANSFORMER_READ[pm] = ("exclude", "false hit or peripheral", "only LLM-related hit is an AI-use statement")
# old automatic exclusions ('no LLM-related transformer in analysed text') whose sentences now hold a v3 name
OLD_AUTO_REREAD = {
    "PMC10411891": ("restore", "use or compare", "DeepAIR built on ProtBert"),
    "PMC10610807": ("restore", "use or compare", "BERT-base and BERT-large used as text encoders"),
    "PMC10372225": ("restore", "use or compare", "CheXbert labeler used for the clinical F1 metric"),
    "PMC11511535": ("restore", "use or compare", "MotionBert one of the library's 3D detectors"),
    "PMC11731964": ("restore", "use or compare", "VetBert employed"),
    "PMC11639203": ("exclude", "passing mention", "ASSBert in related work"),
    "PMC11894149": ("exclude", "false hit or peripheral", "BERTHELOT3: an author name with a glued citation (group-4 false hit)"),
    "PMC13115501": ("exclude", "passing mention", "AnomalyBert listed"),
    "PMC11464512": ("exclude", "passing mention", "Bert-based models in background"),
    "PMC12540194": ("exclude", "false hit or peripheral", "DeepPalm, a CNN (group-4 false hit)"),
    "PMC9818166": ("exclude", "passing mention", "CheXbert as an example in a review"),
    "PMC12649439": ("exclude", "false hit or peripheral", "Print-on-Palm (group-4 false hit)"),
    "PMC13371655": ("exclude", "passing mention", "language-model embeddings listed"),
}

if __name__ == "__main__":
    S = json.load(open(SCREEN))
    old_t = {r["pmcid"]: r for r in csv.DictReader(open(OLD_TRANSFORMER))}
    rows = []
    for pm, r in sorted(S.items()):
        od, g = r["old_decision"], r["corpus"]
        cls = r.get("llm", {}).get("c", "")
        if pm in OLD_AUTO_REREAD:
            d, lab, note = OLD_AUTO_REREAD[pm]
            rows.append((pm, g, "4 transformer only (old automatic exclusion, v3 name)", d, lab, note)); continue
        if od:                                              # an old (hand-checked) decision stands
            continue
        if cls.startswith("3"):
            d, lab, note = CLASS3[pm]
            rows.append((pm, g, "1 LLM defined otherwise", d, lab, note))
        elif cls.startswith("6"):
            other = [t for t in r["terms"] if t not in ("llm", "llms")]
            if other:                                       # an accepted model name is evidence (old regex missed these spellings)
                rows.append((pm, g, "2 LLM undefined", "keep", "other accepted focal term", ", ".join(other[:4])))
            else:
                rows.append((pm, g, "2 LLM undefined, no evidence", "exclude", "llm_no_definition_no_evidence", ""))
        if r.get("name_only"):
            n = r["name_only"]
            if n == "biobridge":
                rows.append((pm, g, "3 name only", "exclude", "name_homonym", "BioBridge homonym"))
            elif n == "progen":
                d, lab, note = PROGEN[pm]; rows.append((pm, g, "3 name only", d, "name_homonym" if d == "exclude" else lab, note))
            elif n == "esm-2":
                if pm in ESM2_HOMONYM:
                    rows.append((pm, g, "3 name only", "exclude", "name_homonym", ESM2_HOMONYM[pm]))
                else:
                    rows.append((pm, g, "3 name only", "keep", "model", "ESM-2 protein language model"))
        if r.get("transformer_only"):
            if not r["region_hits"]:
                rows.append((pm, g, "4 transformer only", "exclude", "no LLM-related transformer in analysed text", ""))
            else:
                d, lab, note = TRANSFORMER_READ[pm]
                rows.append((pm, g, "4 transformer only", d, lab, note))
    seen = {}
    for x in rows:
        assert x[0] not in seen, f"{x[0]} decided twice"
        seen[x[0]] = x
    os.makedirs(os.path.join(HERE, "decisions"), exist_ok=True)
    with open(os.path.join(HERE, "decisions", "item6_v3_1.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pmcid", "corpus", "rule", "decision", "label", "note"]); w.writerows(rows)
    import collections
    print(collections.Counter((x[1], x[2], x[3]) for x in rows))
