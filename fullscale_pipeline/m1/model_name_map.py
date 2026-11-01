# Hand-curated model-name map for M1 (chunk12v3), ticket 107. Owner decisions 2026-10-31 and 2026-11-01; UNCHECKED_SIMILARITY pending.
# Applied to the root of every focal atom (".../C/focal") before the model-name levels (tier / version / family).
#   CANON:   same model, other spelling, glued citation number, or adjective form -> canonical root (identity).
#   NOT_MODEL: (article, root) pairs where the focal atom is not a model ("Grover" as an author name).
#   BY_ARTICLE: (article, root) pairs read as another model ("RoBERT" for RoBERTa).
#   LINEAGE: derived model family -> (base family, relation); similarity 0.5 / 0.25 in W by relation.
# Glued citation numbers not listed here are caught by the general rule in glued_citation(): a family name
# followed by digits that are not a known version of that family -> the bare family (owner, 2026-10-31).
import re

KNOWN_VERSIONS = {   # versions written with an underscore by G3 (gpt_4 = "GPT-4"); anything else is a citation
    "gpt": {"1", "2", "3", "3_5", "4", "4o", "4v", "4_1", "4_5", "5", "5_1", "5_2", "5_3", "5_4", "5_5", "o1", "o3", "o4"},
    "chatgpt": set(),
    "llama": {"1", "2", "3", "3_1", "3_2", "3_3", "4"},
    "gemini": {"1", "1_5", "2", "2_5", "3"},
    "gemma": {"1", "2", "3", "4"},
    "claude": {"1", "2", "3", "3_5", "3_7", "4", "4_1", "4_5", "4_6", "4_7", "4_8"},
    "qwen": {"2", "2_5", "3"},
    "deepseek": {"v1", "v2", "v3", "v3_1", "v3_2", "v4", "r1"},
    "grok": {"1", "2", "3", "4"},
    "palm": {"2"}, "med_palm": {"2"}, "paligemma": {"1", "2"},
    "deberta": {"v2", "v3"}, "mdeberta": {"v3"}, "esm": {"2"}, "minigpt": {"4"}, "czegpt": {"2"},
}

CANON = {
    # spelling variants of one model
    "distillbert": "distilbert", "mental_bert": "mentalbert", "mental_roberta": "mentalroberta",
    "mental_roberta_base6": "mentalroberta", "mentalllama": "mentallama", "mental_llama": "mentallama",
    "mllama": "mentallama",                                  # "MentaLLaMA (mLLaMA)"
    "bioclinical_bert": "bioclinicalbert", "bio_clinicalbert": "bioclinicalbert", "bio_plusclinicalbert": "bioclinicalbert",
    "kc_bert": "kcbert", "m_bert": "mbert", "bert_basemultilingual": "mbert",
    "s_bert": "sbert", "sentencebert": "sbert", "sentence_bert": "sbert", "st_bert": "sbert",
    "medpalm": "med_palm", "med_palm2": "med_palm_2", "med_palmmultimodal": "med_palm",
    "debertav3": "deberta_v3", "xlm_reberta": "xlm_roberta", "xlm_roberta_l": "xlm_roberta",
    "ft_roberta_l": "roberta", "pt_roberta_l": "roberta",    # fine-tuned / pre-trained RoBERTa-large
    "robberta": "robbert", "geminipro": "gemini_pro", "berts": "bert", "chatgpts": "chatgpt",
    "chatgpt_v2": "chatgpt", "chatgpt_v4": "chatgpt",        # "ChatGPT v2; OpenAI": not a model version
    "psycbert": "psychbert", "bangla_bert": "banglabert", "bangla_bert_1": "banglabert", "bangla_bert_2": "banglabert",
    "videollama": "video_llama",
    "grok_4_fast": "grok_4", "grok_4_fast_non_reasoning": "grok_4",
    "thegpt_3_5_turbozs_s": "gpt_3_5_turbo", "pgpt3_5": "gpt_3_5", "pgpt_3": "gpt_3",   # p-value subscripts
    "v_llama": "llama", "v_mm_llama": "llama",               # "V-LLaMA": the paper's label for vanilla LLaMA 3.2
    "llama_7bfinetuned": "llama", "llama_7": "llama",        # "LLaMA 7 billion"; "LLaMA7,37" (citation)
    "minigpt": "minigpt_4",                                  # "MiniGPT-471": MiniGPT-4 + citation 71
    "gpt_j_6": "gpt_j", "1_bert": "bert", "2_bert": "bert", "bert_1": "bert", "bert_2": "bert",
    "wbert": "bert",                                         # "WBERT": the weight of the BERT approach
    # adjective forms ("LLaMA-generated", "DialoGPT-enabled")
    "llama_generated": "llama", "llama_derived": "llama", "llama_powered": "llama",
    "dialogpt_enabled": "dialogpt", "deproberta_derived": "deproberta", "deproberta_ensemble": "deproberta",
    "bertweet_sentiment": "bertweet", "hubert_emb": "hubert", "optimized_bert": "bert", "berttiny_pretrained": "bert",
    "graph_roberta_clscreenshot": "graph_roberta_cl",
    # glued citation numbers the general rule cannot see (a known version followed by a citation)
    "llama_338": "llama_3", "deepseek_r171": "deepseek_r1", "deepseek_r128": "deepseek_r1",
    "gemma_32": "gemma", "modernbert_base54": "modernbert", "llamaindex41": "llamaindex",
}

# Per-article readings. GROVER is a real model in the corpus (the fake-news generator, Zellers et al. 2019; the molecular
# graph model, Rong et al. 2020; the DNA model, Sanabria et al. 2024), so "Grover" that is not a model is handled per
# article (owner, 2026-11-01: remove the false positives by hand, keep the model). All 362 corpus mentions were read; the
# 11 articles below are every false positive (authors, acknowledged persons, Grover's quantum search algorithm); the
# 15 sentence units are listed in corpus_statistics/focal_false_positive_units.csv. Rule for the builder: the atom is
# removed; a focal child hyperedge left empty is removed; a sentence unit left with no focal atom is removed (14 of the
# 15 units); an article left with no sentence unit would be removed (none is). "RoBERT" has three readings (Recurrence
# over BERT, Pappagari et al. 2019; the Romanian BERT, Masala et al. 2020; RoBERTa with the final "a" dropped); only
# articles where the text shows which one is meant are mapped. Articles where "GPT" is the Grooved Pegboard Test are
# removed from the trial (RL-204, RL-205; corpus_statistics/psychology_trial_exclusions.csv).
NOT_MODEL = {(p, "grover") for p in (
    "PMC11681292", "PMC12748763", "PMC13433594", "PMC12961534", "PMC12957616", "PMC13318610",
    "PMC12357839", "PMC10917319", "PMC11810639", "PMC12753390", "PMC12011883")}
BY_ARTICLE = {("PMC12370911", "robert"): "roberta",                # "DistilBERT, RoBERT, ALBERT, BERT, and DistilRoBERT"
              ("PMC10308283", "robert_ft1"): "roberta"}            # "RoBERT-FT1 ... RoBERTa-FT2" in one sentence

# Lineage: derived model family -> (base family, relation). Similarity in W by relation (owner, 2026-11-01):
#   weights = trained from the base's own weights (further pre-training on domain or language data, fine-tuning,
#             distillation from it, the base inside a larger network, an application that runs it)       -> 0.5
#   design  = same design as the base, trained anew from scratch (no weights inherited from it)           -> 0.25
#   unchecked = the training was not checked; similarity UNCHECKED_SIMILARITY (owner to choose)
# Only the listed pair gets the similarity; pairs are not chained (mentalroberta-bert is not derived from the table).
SIMILARITY = {"weights": 0.5, "design": 0.25}
UNCHECKED_SIMILARITY = 0.25   # owner, 2026-11-01 ("I agree with your proposal")
W, D, U = "weights", "design", "unchecked"
LINEAGE = {
    # encoder families against BERT
    "distilbert": ("bert", W),       # initialised from BERT's layers and distilled from it (Sanh et al. 2019)
    "roberta": ("bert", D), "albert": ("bert", D), "deberta": ("bert", D), "electra": ("bert", D),
    "modernbert": ("bert", D), "spanbert": ("bert", D), "mobilebert": ("bert", D), "hubert": ("bert", D),
    "xlm_roberta": ("roberta", D), "distilroberta": ("roberta", W),
    # BERT: further pre-training or fine-tuning from BERT's weights
    "mentalbert": ("bert", W), "psychbert": ("bert", W), "biobert": ("bert", W), "clinicalbert": ("bert", W),
    "bioclinicalbert": ("bert", W), "finbert": ("bert", W), "covid_twitter_bert": ("bert", W), "ct_bert": ("bert", W),
    "madrs_bert": ("bert", W), "therapistbert": ("bert", W), "disorbert": ("bert", W), "anxibert": ("bert", W),
    "corebert": ("bert", W), "tod_bert": ("bert", W), "docbert": ("bert", W), "sbert": ("bert", W),
    "mpc_bert": ("bert", W), "sa_bert": ("bert", W), "sr_bert": ("bert", W), "tmbert": ("bert", W),
    "bertagent": ("bert", W), "augbert": ("bert", W), "opinion_bert": ("bert", W),
    "chinese_bert_wwm_ext": ("bert", W), "macbert": ("bert", W),   # initialised from Google's Chinese BERT
    "mm_bert": ("bert", W), "mag_bert": ("bert", W), "videobert": ("bert", W), "visualbert": ("bert", W),
    "vilbert": ("bert", W), "aobert": ("bert", W), "audibert": ("bert", W),
    # BERT design trained from scratch (one language, tweets, structured health records)
    "mbert": ("bert", D), "kobert": ("bert", D), "kcbert": ("bert", D), "kr_bert": ("bert", D), "korbert": ("bert", D),
    "kpfbert": ("bert", D), "klue_bert": ("bert", D), "kobigbird_bert": ("bert", D), "arabert": ("bert", D),
    "marbert": ("bert", D), "banglabert": ("bert", D), "hebert": ("bert", D), "alephbert": ("bert", D),
    "bertje": ("bert", D), "gbert": ("bert", D), "german_bert": ("bert", D), "tohoku_bert": ("bert", D),
    "bertimbau": ("bert", D),        # initialised from multilingual BERT, which is itself trained from scratch
    "bertweet": ("bert", D), "med_bert": ("bert", D),
    "micro_bert": ("bert", D), "micron_bert": ("bert", D),   # vision models pre-trained BERT-style on video frames
    "graph_bert": ("bert", D),       # Graph-BERT: trained on graphs from scratch
    # RoBERTa
    "mentalroberta": ("roberta", W), "emoberta": ("roberta", W), "emoroberta": ("roberta", W),
    "twitter_roberta": ("roberta", W), "deproberta": ("roberta", W), "robertadepressiondetection": ("roberta", W),
    "cd_roberta": ("roberta", W), "ear_roberta": ("roberta", W), "s_roberta": ("roberta", W),
    "multi_roberta": ("roberta", W), "all_roberta": ("roberta", W), "melbert": ("roberta", W),
    "klue_roberta": ("roberta", D), "chinese_roberta_wwm_ext": ("roberta", D), "erlangshen_roberta": ("roberta", D),
    "kosimcse_roberta": ("roberta", D), "kobigbird_roberta": ("roberta", D), "camembert": ("roberta", D),
    "gottbert": ("roberta", D), "umberto": ("roberta", D), "robertuito": ("roberta", D),
    "wangchanberta": ("roberta", D), "robbert": ("roberta", D), "babyberta": ("roberta", D),
    "twitter_xlm_roberta": ("xlm_roberta", W),
    "all_distilroberta": ("distilroberta", W), "nli_distilroberta": ("distilroberta", W),
    "st_distilbert": ("distilbert", W), "multi_qa_distilbert_cos": ("distilbert", W), "distilkobert": ("distilbert", D),
    "mdeberta": ("deberta", D), "nli_deberta": ("deberta", W),
    # GPT
    "instructgpt": ("gpt", W), "dialogpt": ("gpt", W),   # fine-tuned from GPT-3 / initialised from GPT-2
    "biogpt": ("gpt", D), "gatortrongpt": ("gpt", D), "neuro_gpt": ("gpt", D),
    "kogpt2": ("gpt", D), "aragpt2": ("gpt", D), "czegpt": ("gpt", D),
    "gpt_neo": ("gpt", D), "gpt_j": ("gpt", D), "gpt_oss": ("gpt", D),   # GPT design, trained from scratch
    "microsoft_copilot": ("gpt", W),   # a product that runs OpenAI GPT-4-class models
    "bard": ("palm", W),               # Google's chatbot, run on LaMDA, then PaLM 2 (2023), renamed Gemini (2024)
    # applications that run a GPT model (the text says so)
    "gptcoach": ("gpt", W), "navgpt": ("gpt", W), "robotgpt": ("gpt", W), "gptscreenr": ("gpt", W),
    "doximitygpt": ("gpt", W), "gpthistory": ("gpt", W), "creascoregpt": ("gpt", W),
    # LLaMA, PaLM, Gemini, Gemma, Mistral
    "mentallama": ("llama", W), "emotion_llama": ("llama", W), "llamaguard": ("llama", W), "chinese_llama": ("llama", W),
    "cockpit_llama": ("llama", W), "video_llama": ("llama", W), "openllama": ("llama", D),
    "minigpt": ("llama", W), "pandagpt": ("llama", W),   # built on Vicuna, a fine-tuned LLaMA (not on GPT)
    "mixtral": ("mistral", D), "avhubert": ("hubert", D),
    "med_palm": ("palm", W), "flan_palm": ("palm", W), "palm_e": ("palm", W), "med_gemini": ("gemini", W),
    "paligemma": ("gemma", W), "embeddinggemma": ("gemma", W), "mentalmistral": ("mistral", W),
    # the base model combined with another network ("DeBERTa-BiLSTM", "SBERT-CNN"): the base's weights are inside
    "fusion_convbert": ("bert", W), "fusion_bert": ("bert", W), "fusion_robustbert": ("bert", W),
    "concat_bert": ("bert", W), "dual_contextbert": ("bert", W), "lda_bert": ("bert", W), "ensembert": ("bert", W),
    "sbert_cnn": ("sbert", W), "albert_bilstm": ("albert", W), "albert_plusbert_bilstm": ("albert", W),
    "deberta_bilstm": ("deberta", W), "deberta_plusbilstm": ("deberta", W), "hubert_plusernie_health": ("hubert", W),
    "hubert_pluslongformer": ("hubert", W), "hubert_clap": ("hubert", W), "graph_roberta": ("roberta", W),
    "graph_roberta_cl": ("roberta", W), "acs_plusllama": ("llama", W),
    # not checked: the text does not say how the model was trained, and no source was found
    "clinicalbert_zh": ("bert", U), "twhin_bert": ("bert", U), "emobert": ("bert", U), "memobert": ("bert", U),
    "topicbert": ("bert", U), "urdubert": ("bert", U), "bhs_bert": ("bert", U), "mabert": ("bert", U),
    "bubert": ("bert", U), "roberta_beto": ("roberta", U),
    "german_roberta_sentence_transformer": ("roberta", U),
    "medgpt": ("gpt", U), "depgpt": ("gpt", U), "eeg_gpt": ("gpt", U), "bp_gpt": ("gpt", U), "mindgpt": ("gpt", U),
    "safegpt": ("gpt", U), "warmgpt": ("gpt", U), "hivegpt": ("gpt", U), "mdteamgpt": ("gpt", U), "gesgpt": ("gpt", U),
    "antgpt": ("gpt", U),
}
# Names that carry a base's name but are not derived from it: no lineage pair (similarity 0).
#   biomedgpt (a vision-language model of another design, Zhang et al. 2024), affectgpt (built on other models),
#   bggpt (continued pre-training of Mistral and Gemma, not GPT).

def lineage_similarity(fam):
    """(base family, similarity in W) for a derived family, or None."""
    if fam not in LINEAGE: return None
    base, rel = LINEAGE[fam]
    return base, (UNCHECKED_SIMILARITY if rel == U else SIMILARITY[rel])

_DIG = re.compile(r"^(?P<fam>[a-z_]*?[a-z])(?P<num>\d{1,3})$")

def glued_citation(root):
    """'bert28' -> 'bert', 'gpt_9' -> 'gpt' (no GPT-9), 'gpt_4' unchanged, 'gpt2'-style known versions unchanged."""
    m = _DIG.match(root)                      # family letters immediately followed by digits: bert28, roberta34
    if m and not m.group("fam").endswith("_"):
        fam, num = m.group("fam"), m.group("num")
        if num in KNOWN_VERSIONS.get(fam, set()) or (fam.endswith("gpt") and num == "2"): return root   # GPT2, kogpt2
        if fam in KNOWN_VERSIONS or fam in LINEAGE or fam in BASES: return fam
        return root
    head, _, tail = root.rpartition("_")      # family_digits: gpt_9, gemini_8, gpt_111, gpt_44
    if head in KNOWN_VERSIONS and tail.isdigit() and tail not in KNOWN_VERSIONS[head]: return head
    return root

BASES = {"bert", "roberta", "distilbert", "albert", "deberta", "distilroberta", "xlm_roberta", "modernbert",
         "electra", "hubert", "gpt", "llama", "gemini", "gemma", "claude", "qwen", "deepseek", "mistral", "palm", "grok"}

def canonical(root, pmcid=None):
    """Canonical focal-model root, or None if the atom is not a model in this article (the atom is then removed)."""
    if (pmcid, root) in NOT_MODEL: return None
    root = BY_ARTICLE.get((pmcid, root), root)
    root = CANON.get(root, root)
    return CANON.get(glued_citation(root), glued_citation(root))
