# Rubric: is this spelling the name of a model? (group-4 review, RL-173)

Each item is one **spelling** (exact case) that a pattern accepted as a possible model name because it contains GPT,
BERT, RoBERTa, LLaMA/Llama, PaLM/Palm, Gemini, Mistral or Gemma. You see how often it occurs, and up to three contexts
from different articles, each with the article title. Decide what the spelling refers to **in these contexts**.

**model** — the spelling names a neural language model or a transformer model of the BERT / GPT / LLaMA / PaLM /
Gemini / Mistral / Gemma kind, or a direct variant of one:
- general or domain models: DistilBERT, ClinicalBERT, BioGPT, MedGemma, BioMistral, AraBERT, CamemBERT;
- BERT- or GPT-style models of other data (proteins, genes, cells, speech, molecules, time series, images): ProtBERT,
  scGPT, DNABERT2, HuBERT, MolGPT, TimeGPT, SegGPT;
- fine-tunes, checkpoints, sizes and architecture combinations: bert-base-chinese, ProtBERT-BFD, BERT-CRF,
  CC_BERT_BiLSTM, HuBERT-Large, SleepGPT-Dual;
- LLM-based apps and assistants named after the model: a custom GPT, InternalGPT, AtlasGPT, MedGPT app;
- misspellings of such names (ChapGPT, DistiBERT, MotinBert), also with a citation number glued on (SpikeGPT28).

**not_model** — anything else that merely contains those letters:
- genes, enzymes, proteins, reagents, plasmids, peptides (AtGPT1, siGPT2, pGPT-1, rhANGPTL4, YFP-Palm1, MyrPalm-…);
- chemicals and materials (GPTMS silanes, "Palm" = palmitate or palm oil), drug regimens (BPaLM);
- clinical trials and study names (LIBERTY-UC), scales and scores (R-GPTS), labels and abbreviations (GPTV = gross
  peritumoral volume);
- software tools, databases and libraries that are not themselves language models (BERTopic, BERTScore, LLaMA-Factory,
  AMBERTOOLS, GPS-Palm, SwissPalm), physics lists (QGSP_BERT_HP), code identifiers and file names;
- non-language models that only share letters (GeminiMol, a molecular model; GPTime, a Gaussian-process tool;
  GPTransformer for genomic prediction; DeepPalm, a CNN);
- companies, suppliers, devices, products (GeminiBio, TacPalm, the ROBERT® robot), person names (LeBert, Moreno-Llamas).

**mixed** — the contexts disagree (a model in one article, something else in another). Say which in the note.

**unsure** — the contexts do not decide it. Do not force a choice.

Judge from the contexts, not from the letters alone: "GPT" inside a gene name is a gene; a name ending in BERT that is
introduced as "a pre-trained transformer" is a model.

**Output**: one JSON object per line, exactly:
`{"qid": "G01.001", "verdict": "model" | "not_model" | "mixed" | "unsure", "note": "<at most 15 words>"}`
