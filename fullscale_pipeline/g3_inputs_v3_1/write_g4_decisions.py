"""Group-4 false hits and glued citation numbers (RL-172): writes decisions/g4_decisions.csv, read by G3.
Columns: spelling (the v3:G4 surface, exact case), pmcid (empty = every article), action, target, reason, source.
  reject   the mention is not a focal term (G3 skips it; a sentence left without a focal term gets no parent)
  map      the mention is a model; G3 writes canonical(target) instead of canonical(spelling)
Sources: 'read' = the assistant read every spelling of the rule lists and of the 700 articles whose only focal terms
are G4 names (g4_review.py; contexts in PG/g3_inputs_v3_1/g4/); 'citation rule' = a name with 2-3 glued digits: the
digits are a citation number and are dropped; of three glued digits the first is kept when name + that digit is itself a
known name (DNABERT217 -> DNABERT2, ProtGPT215 -> ProtGPT2; but ChatGPT30 -> ChatGPT, DNABERT21 -> DNABERT).
    python3 write_g4_decisions.py
"""
import csv, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
G4 = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_inputs_v3_1/g4/g4_types.json"

REJECT = {   # spelling: reason (read; corpus-wide)
    # genes, enzymes, plasmids and lab reagents
    **{k: "gene or protein (glucose-6-phosphate/phosphate translocator, transaminase)" for k in
       "StGPT1 StGPT2 siGPT1 siGPT2 rGPT1 OsGPT1 OsGPT2 OsGPT2s oeGPT2 AtGPT1 AtGPT2 pGPT1 PoGPT1 KfGPT1 KfGPT2 NbGPT1 proGPT1 proGPT2 "
       "HaGPT1 sgGPT1 sgGPT2 QgGPT1 QgGPT2 QgGPT3 QgGPT4 QgGPT6 QgGPT7 QgGPT9 QgGPT10 QgGPT12 GPT1_ARATH PfGPT gGPT GPTm GPTases "
       "GGPTase-I.250 practitionerGPTGlutamate".split()},
    **{k: "plasmid or vector" for k in
       "pGPT pGPT-1 pGPT-BeauR-ORF7rep pGPT-BeauR-ORF7rep-HIS pGPT-BeauR-S-MM pGPT-BeauR-S1-M41-S2 pGPT-EndoU-deficient-negative "
       "pGPT-IBV-StuI-BamHI pGPT-M41-S-BSM pGPT-M41-S1-BeauR-S2 pGPT-M41-S2-L578F-N617S pGPT-M41-S2-N826S-L857F-I1000V pGPT-M41S "
       "pGPT-NEB193 pGPT-ORF1a-negative pGPT-ORF1a-positive pGPT-ORF1b-negative pGPT-ORF1b-positive pGPT-S pGPT-S-negative "
       "pGPT-S-positive pGPT-based pGPT-nsp15-positive pGPTNEB193 pGPTVII pFX_EGPT".split()},
    **{k: "ANGPT/ANGPTL4 reagent" for k in
       "RNA-SiANGPT2 SiANGPT2 SiANGPT2-treated siANGPT2 rANGPTL4 rANGPTL4-driven rhANGPTL4 shANGPTL4 shANGPTL4+HGPA".split()},
    **{k: "palmitoyl/palmitate/palm, not PaLM" for k in
       "177Lu-Palm-3PRGD2 177Lu-Palm DOTA-Palm-3PRGD2 DOTA-Palm Lu-DOTA-Palm-3PRGD2 Lu-DOTA-Palm YFP-Palm1 mEGFP-Palm1 "
       "pORANGE-mEGFP-Palm1 Cas10-mPalm2 mPalm2 QD-Palm1 Linker-Palm1 N-Palm1-LexA-C pFBL23-Palm1 EK35S-Palm MU-6S-Palm "
       "MyrPalm MyrPalm-ATG8-GFP MyrPalm-GABARAPL1-GFP MyrPalm-GABARAPL2-GFP MyrPalm-LC3B-GFP MyrPalm-LC3C-GFP "
       "ND-Palm ND-Cana-Palm ND-Palm-Cana ND-Palm-No T1D-Palm T1D-Cana-Palm T1D-Diet-Palm T1D-Palm-Cana T1D-Palm-No non-Palm-stressed "
       "HF-Palm OG-SBPalm S-Palm Lime-Palm Corn-SBM-DDGS-Palm Date-Palm Common-Palm PCV-Palm PopcornPalm PhyloPalm CorpPalm CLM-Palm "
       "NDPE-Policies-Cover-83-of-Palm-Oil-Refining-Market.pdf 1-Palm Finger-Palm Finger-To-Palm Fist-Palm".split()},
    **{k: "palmitoylation tool or database" for k in
       "GPS-Palm GPS-Palm48 GPS-Palm60 GPSPalm SwissPalm Swiss-Palm MDDPalm MDD-Palm CKSAAP-Palm NBA-Palm".split()},
    **{k: "device or product, not a language model" for k in
       "HygroPalm23‐A TacPalm ePalm BlazePalm VILBERT-LOURMAT GeminiBio Raker-Roberta ECOBERTURA".split()},
    **{k: "silane, chemistry or materials label" for k in
       "3nGPTMS APTES+GPTMS CTS_GPTMS_5 PEDOT_GPTMS PEO_GPTMS PL_GPTMS-ARS mGPTMS AGPTi GPTi GPTi-coated GPTS.22 Si-GPTS.28".split()},
    **{k: "Geant4 physics list or code identifier" for k in
       "QGSP_BERT_HP FTFP_BERT_HP G4HadronPhysicsFTFP_BERT_HP G4HadronPhysicsQGSP_BERT_HP FTFP_BERT_EMZ QGSP_BERT_EMV "
       "g4h-phy_QGSP_BERT_HP g4h‐phy_FTFP_BERT_HP FTFP_BERT_EMY order_BERT_HP QGSP_BERT39.Figure BERTref "
       "get_BERT_or_XLNET_word_embedding_for_a_dataset.py".split()},
    **{k: "scale, score, regimen, trial or label" for k in
       "R-GPTSa R-GPTSb R-GPTs r-GPTSb r-GPTSb.Figure rGPTSb GPTV5_DLR GPTV_DL GPTV_DLR GPTV_radiomics pGPT4 CGPT.15 TM_CGPT UGPTis "
       "ofGPTrentropy AVGPT+SDPT GPTx+ProAgio GPTx+ProAgio-treated ProAgio+GPTx ProAgio+GPTx-treated BPaLM BPaLM-based BPaLM-related "
       "LIBERTY LIBERTY-CD LIBERTY-CSU LIBERTY-UC LIBERTY-PN LIBERTYLUNG LIBERTO BERTIE PUBERTAL HILBERT2".split()},
    **{k: "database or tool, not a language model" for k in
       "mgPGPT mgPGPT-mapping-db PGPT_BASE_nr_Aug2021n_ul_1 PGPT_pred PGPTs-associated PGPTsof AMBERTOOLS AMBERTOOLS22 "
       "GPTime GPTime26 GPTime30 GPTransformer isGPT MS-GPTNet".split()},
    **{k: "person name" for k in "LAMBERT ALBERTI ALBERTINOI FernandezAlbert LeBert Moreno-Llamas ROBERTO Roberta-Maria Twi-LlamaTag Twi-LlamaTag-mCh".split()},
    **{k: "group-4 false hit in the sample" for k in "BERTHELOT3 DeepPalm Print-on-Palm BDI-FS-GPT HMW-GS-GPT GeminiMol GeminiMol23".split()},
    **{k: "tool, not a model (group-4 rule)" for k in "LLaMA-Factory LLaMA-Factory35 LLaMA-Factory78 LLaMA_Factory LlamaFactory44".split()},
}
ROBERT_KEEP = {"PMC11780922": "RoBERTa (dynamic mask)", "PMC9947453": "Real Option BERT"}   # ROBERT elsewhere: a robot, software, names
MAP = {   # spelling: (target, reason) — read
    "ChapGPT": ("ChatGPT", "typo"), "ChatpGPT": ("ChatGPT", "typo"), "MedPalm-22": ("Med-PaLM 2", "Med-PaLM 2 + citation 2"),
    "Med-Palm41": ("Med-PaLM", "Med-PaLM + citation"), "MED-Palm2": ("Med-PaLM 2", "spelling"), "Med-Palm-2": ("Med-PaLM 2", "spelling"),
    "Med-Palm": ("Med-PaLM", "spelling"), "prot_BERT_bfd": ("ProtBERT-BFD", "checkpoint id"), "Fed_BERT_": ("Fed_BERT_MSCNN", "split token"),
}
FAM = r"(?:GPT|BERT|Bert|BERTa|LLaMA|Llama|Gemini|PaLM|Mistral|Gemma)"
GLUED = re.compile(rf"^(.*{FAM})(\d)?(\d{{1,2}})$")
LISTED = {"gpt", "bert", "roberta", "llama", "gemini", "palm", "mistral", "gemma", "chatgpt", "biobert", "pubmedbert", "scibert", "biogpt",
          "bluebert", "dnabert", "chemberta", "sentence-bert", "med-bert", "instructgpt"}


def main():
    G = json.load(open(G4))
    known = {k.lower() for k in G["occ"]} | LISTED
    rows = []
    for k, why in sorted(REJECT.items()):
        rows.append((k, "", "reject", "", why, "read"))
    for pm, why in ROBERT_KEEP.items():
        rows.append(("ROBERT", pm, "map", "RoBERTa" if "RoBERTa" in why else "ROBERT", why, "read"))
    rows.append(("ROBERT", "*other", "reject", "", "rehabilitation robot ROBERT®, ROBERT software, person names", "read"))
    for k, (t, why) in sorted(MAP.items()):
        rows.append((k, "", "map", t, why, "read"))
    done = {r[0] for r in rows}
    for k in sorted(G["occ"]):
        if k in done or not re.search(r"\d{2,3}$", k):
            continue
        m = GLUED.match(k)
        if not m or not re.search(rf"{FAM}\d{{2,3}}$", k):
            continue
        stem, d1, rest = m.group(1), m.group(2) or "", m.group(3)
        if d1 and (stem + d1).lower() in known and len(d1 + rest) == 3:   # three digits: version + 2-digit citation
            target = stem + d1                                   # DNABERT217 -> DNABERT2 (+ citation 17)
        else:
            target = stem                                        # SpikeGPT28 -> SpikeGPT
        rows.append((k, "", "map", target, "glued citation number", "citation rule"))
    # RL-173: the Haiku review of the remaining spellings (g4_haiku_decisions.py), with the citation rule widened to every
    # kept spelling: a trailing ".NN" or 2-3 glued digits whose removal leaves a spelling of the corpus is a citation
    # (AntiBERTy19 -> AntiBERTy, TCR-BERT.31 -> TCR-BERT); one digit is never removed (DNABERT2, ProtGPT2 are versions)
    done = {r[0] for r in rows}
    spellings = set(G["occ"])
    for r in csv.DictReader(open(os.path.join(HERE, "decisions", "g4_haiku_decisions.csv"))):
        if r["spelling"] in done:
            continue
        if r["action"] == "keep":
            m = re.match(r"^(.*[A-Za-z])(?:\.\d{1,3}|\d{2,3})$", r["spelling"])
            if m and m.group(1) in spellings and len(m.group(1)) >= 4:
                rows.append((r["spelling"], "", "map", m.group(1), "glued citation number", "citation rule (Haiku: model)")); continue
        rows.append((r["spelling"], r["pmcid"], r["action"], r["target"], r["reason"], r["source"]))
    os.makedirs(os.path.join(HERE, "decisions"), exist_ok=True)
    with open(os.path.join(HERE, "decisions", "g4_decisions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["spelling", "pmcid", "action", "target", "reason", "source"]); w.writerows(rows)
    occ = G["occ"]
    rej = [r for r in rows if r[2] == "reject" and r[1] == ""]
    import collections
    print(collections.Counter((r[2], r[5].split(" ")[0]) for r in rows))
    print("rows", len(rows), "| reject spellings", len(rej), "occ", sum(occ.get(r[0], 0) for r in rej),
          "| citation maps", sum(r[5] == "citation rule" for r in rows), "| read maps", sum(r[5] == "read" and r[2] == "map" for r in rows))
    missing = [r[0] for r in rows if r[0] not in occ]
    print("spellings not in the corpus list:", missing[:20])


if __name__ == "__main__":
    main()
