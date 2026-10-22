"""Group-4 boundary review (RL-174; owner 2026-10-22: apply one explicit line to the model/method boundary, and map
"model + algorithm" pipelines to their model). The assistant read every spelling of the boundary set (479: kept spellings
described as framework, system, app, method, tool, agent, pipeline, platform or library; rejected ones described as
method, tool, framework and the like; and every name decided both ways in different spellings) with a context, and
decided by one test:
  KEEP    the name denotes something whose outputs are produced by a language model: a trained or fine-tuned model, or an
          LLM-backed app, agent, assistant, framework or pipeline that answers or generates
  MAP     the name denotes a use or configuration of a specific model -> that model: run, arm and condition labels
          (BERT_FT, GPT4x3), "-based/-driven/-powered/-aligned/-optimized" forms, compounds of a model with other
          components (SBERT-k-means, BiLSTM-CRF-BERT, DeBERTa-GCN), adapters (LLaMA-Adapter), modules of a named system
          (TrialGPT-Ranking), glossary letters glued to the name ("hBERT: bidirectional encoder ..."), misspellings
  REJECT  the outputs are not produced by a language model: procedures that transform models (SparseGPT, QuantGPT, oBERT),
          detectors, metrics, benchmarks, non-generative libraries and tools (LlamaIndex, GPTCache, BERTopic, KeyBERT),
          datasets, files, variables and notation, non-language networks (GPTNeXt)
Spellings that differ from a decided one only in case or a glued citation number get the same decision (a citation
variant of a kept name is mapped to it). Writes decisions/g4_boundary_decisions.csv; write_g4_decisions.py applies it last.
    python3 g4_boundary_decisions.py
"""
import csv, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

MAP = {   # spelling -> underlying model
    "2-RoBERTaLarge-10-traindev": "RoBERTa", "4-RoBERTaLarge_CTD-5-traindev": "RoBERTa", "AIIRLLaMA": "LLaMA", "AIIRMistral": "Mistral",
    "Al-BERT-BiGRU+LDA": "ALBERT", "ALBERT_en_base": "ALBERT", "AMBERT-aligned": "AMBERT", "BatteryBERT-based": "BatteryBERT",
    "BERTbased": "BERT", "BERTcls": "BERT", "BERTCLS": "BERT", "BERTFT": "BERT", "BERTFusionDNN": "BERT", "BERTweet-Large": "BERTweet",
    "BiLSTM-CRF-BERT": "BERT", "BioLinkerBERT": "BioLinkBERT", "BrainGPT-generation": "BrainGPT", "caBERT-based": "caBERT",
    "CAMeLBERT-DNN": "CAMeLBERT", "CASSIA-GPT4.0": "GPT-4", "cGPT.Fig": "cGPT", "ChagtGPT‐4": "ChatGPT-4", "CharBERT-optimized": "CharBERT",
    "ChemBERTaDTA": "ChemBERTa", "ChromBERT-based": "ChromBERT", "ChromBERT-tools": "ChromBERT", "ClinBioBert-Seq2Seq-BIO": "ClinBioBert",
    "ClinBioBert-Seq2Seq-Ttype": "ClinBioBert", "CodeLlama-Instruct": "CodeLlama", "DanceGPT-based": "DanceGPT", "DBERT": "DistilBERT",
    "DeBERTa-Bi-GRU-Multi-Headed": "DeBERTa", "DeBERTa-GCN": "DeBERTa", "DeBERTa-L": "DeBERTa", "DeepARV-ChemBERTa.Fig": "ChemBERTa",
    "DISTILBERT_FT": "DistilBERT", "DistilBERT-MRC": "DistilBERT", "DistilRoBERT": "DistilRoBERTa", "DistilRoBERTa-MT": "DistilRoBERTa",
    "DrBERT-powered": "DrBERT", "Dropout-DeBERTa": "DeBERTa", "DualBERT-Trans-CNN": "BERT", "fBERT": "BERT", "fRoBERTa": "RoBERTa",
    "FST+CRF+KazRoBERTa": "KazRoBERTa", "GCN_ProtBert": "ProtBert", "Gemma3-driven": "Gemma3", "GeneGPT-based": "GeneGPT",
    "GPTCelltype4": "GPTCelltype", "GPTCelltype4o": "GPTCelltype", "GPTs2": "GPT-2", "hBERT": "BERT", "HCCN-ProtBert": "ProtBert",
    "heatmap.MMBERT": "MMBERT", "hRoBERTa": "RoBERTa", "iRoBERTa": "RoBERTa", "irAE-GPT.git.Fig": "irAE-GPT", "KPFBERT-based": "KPFBERT",
    "L-ALBERT-FiT": "ALBERT", "LBERT": "BERT", "LLaMA-adapter": "LLaMA", "LLaMA-Adapter": "LLaMA", "LLaMA-Adapter69": "LLaMA",
    "LLaMA-driven": "LLaMA", "LLaMA‑RAG": "LLaMA", "mBERT-based": "mBERT", "MCBERT-BiLSTM-CRF": "MCBERT", "MedGemma-SP": "MedGemma",
    "Monolingual-SBERT": "SBERT", "MT-ScBERT": "ScBERT", "NSynBioGPT2": "SynBioGPT2", "OBERT": "BERT", "OntoGPT-extracted": "OntoGPT",
    "PhoBERT-CRF": "PhoBERT", "polyBERT-based": "polyBERT", "PONE-D-24-07133BERT4FCA": "BERT4FCA", "ProtBert-based": "ProtBert",
    "ProtBert-like": "ProtBert", "ProtChatGPT5": "ProtChatGPT", "RAG5+Llama3‐fine‐tuned": "Llama3", "reguloGPT-CoT": "reguloGPT",
    "rLlama": "Llama", "ROBERTA_FT": "RoBERTa", "ROBERTA-M": "RoBERTa", "SapBERT-style": "SapBERT", "SBERT-k-means": "SBERT",
    "SBERT-K-Means": "SBERT", "SBERT-LDA": "SBERT", "SBERT-UMAP-HDBSCAN": "SBERT", "SBERT-UMAP-KMeans": "SBERT",
    "Sentence-BERT-KMeans++-TopMine": "Sentence-BERT", "scGPT-Adapter": "scGPT", "ScGPT-Human": "scGPT", "scGPT-Human": "scGPT",
    "scGPT-Scratch": "scGPT", "SecureGPT-enabled": "SecureGPT", "SOAP-BiomedBERT": "BiomedBERT", "SpanBERT-based": "SpanBERT",
    "SportsBERT-based": "SportsBERT", "StructGPT+KGMP": "StructGPT", "Synthcoder-DistilBERT": "DistilBERT", "T4SEpp_ProtBert": "ProtBert",
    "TCMBERT-BiLSTM-Transformer-CRF": "TCMBERT", "TemBERTureCLS": "TemBERTure", "TemBERTureTm": "TemBERTure",
    "TesseractOCR-Roberta": "RoBERTa", "THIRAWAT-SapBERT": "SapBERT", "TigCLaF-AfriBERTa": "AfriBERTa", "TrialGPT-Matching": "TrialGPT",
    "TrialGPT-predicted": "TrialGPT", "TrialGPT-Ranking": "TrialGPT", "TrialGPT-Retrieval": "TrialGPT", "UmBERTo-TPT": "UmBERTo",
    "UTSA-HSC-TrialGPT": "TrialGPT", "variants.DNABERT+Output": "DNABERT", "vGPT-5": "GPT-5", "WoBert-based": "WoBERT",
    "YOLO-SAM-GPT": "GPT", "ZeroShotGPTClassifier": "GPT", "MultiLabelZeroShotGPTClassifier": "GPT", "MICC_GPTs": "MICC-GPTs",
    # run, arm, condition and answer labels of a model (formerly rejected as labels)
    "GPT4x1": "GPT-4", "GPT4x3": "GPT-4", "Tools‐ChatGPT4": "ChatGPT-4", "GPTone": "GPT", "Llama70BCombinedAnswer": "Llama",
    "Llama405BCombinedAnswer": "Llama", "ChatGPT4oCombinedAnswer": "GPT-4o", "PMC_LLAMA_Answer": "PMC-LLaMA", "MChatGPT": "ChatGPT",
    "miGPT": "ChatGPT", "GPTderived": "GPT", "DE-GPT3.5-3": "GPT-3.5", "GPT4b.Fig": "GPT-4",
    # case variants of reviewed names (same meaning)
    "ProtBERT-based": "ProtBert", "protBERT-based": "ProtBert", "monolingual-SBERT": "SBERT", "CharBert-Optimized": "CharBERT",
    "CharBERT-Optimized": "CharBERT", "SOAP-BioMedBERT": "BiomedBERT", "BERTweet-large": "BERTweet", "BERTBased": "BERT",
    "ALBERT_EN_BASE": "ALBERT", "scGPT-human": "scGPT", "RoBERTaFT": "RoBERTa", "LBert": "BERT",
    # glossary letters glued to the name
    "gBERT": "BERT", "uSBERT": "SBERT", "fGPT": "GPT", "lGPT": "GPT", "yGPT": "GPT",
}
KEEP = ("OntoGPT ontoGPT OntoGPT16 ontoGPT21 GPTArticleExtractor GPTArticleExtractor50 TopicGPT TopicGPT59 StructGPT AugGPT AugGPT38 "
        "FrugalGPT FrugalGPT15 FrugalGPT57 ZoteroGPT tpGPT GPT4Designer IoTGemini GPTSwarm147 ViperGPT GPTCelltype AugBERT "
        "BERTprepend MICC-GPTs").split()     # outputs produced by an LLM (formerly rejected as method/tool/framework)
REJECT = ("bert2BERT bert2BERT-like BERTal GPTNeXt GPTNeXt-based GPTZip InfoBERT VisBERT visBERT exBERT38 KnGPT18 QuantGPT17 "
          "SparseGPT19 SparseGPT23").split()   # procedures on models, visual tools, non-language networks


def main():
    rows = [(k, "", "map", v, "boundary review: use or configuration of the model", "read (RL-174)") for k, v in sorted(MAP.items())]
    rows += [(k, "", "keep", "", "boundary review: outputs produced by a language model", "read (RL-174)") for k in KEEP]
    rows += [(k, "", "reject", "", "boundary review: outputs not produced by a language model", "read (RL-174)") for k in REJECT]
    seen = set()
    for r in rows:
        assert r[0] not in seen, r[0]
        seen.add(r[0])
    with open(os.path.join(HERE, "decisions", "g4_boundary_decisions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["spelling", "pmcid", "action", "target", "reason", "source"]); w.writerows(rows)
    print(len(rows), "rows:", len(MAP), "map,", len(KEEP), "keep,", len(REJECT), "reject")


if __name__ == "__main__":
    main()
