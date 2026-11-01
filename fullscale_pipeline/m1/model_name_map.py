# Hand-curated model-name map for M1 (chunk12v3), ticket 107. DRAFT for the owner's check (2026-11-01).
# Applied to the root of every focal atom (".../C/focal") before the model-name levels (tier / version / family).
#   CANON:   same model, other spelling, glued citation number, or adjective form -> canonical root (identity).
#   NOT_MODEL_ARTICLES: articles whose focal "GPT" atoms are not a language model (Grooved Pegboard Test).
#   NOT_MODEL_ROOTS: roots that are not a model in any article (author name, weight symbol).
#   ADAPTED: derived model family -> base family, soft similarity 0.5 in W (owner, 2026-10-31: domain-adapted 0.5).
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

# Per-article readings. GROVER is a real model (Zellers et al. 2019) elsewhere in the corpus, so "Grover" as an author
# is handled per article, not by removing the name. "RoBERT" has three readings (Recurrence over BERT, Pappagari et al.
# 2019; the Romanian BERT, Masala et al. 2020; RoBERTa with the final "a" dropped); only articles where the text shows
# which one is meant are mapped. The two Grooved Pegboard Test articles (PMC10395525, PMC11742816) were removed from the
# trial (owner, 2026-11-01; corpus_statistics/psychology_trial_exclusions.csv).
NOT_MODEL = {("PMC12961534", "grover"), ("PMC13318610", "grover")}   # "Grover et al.", "Grover and colleagues"
BY_ARTICLE = {("PMC12370911", "robert"): "roberta",                # "DistilBERT, RoBERT, ALBERT, BERT, and DistilRoBERT"
              ("PMC10308283", "robert_ft1"): "roberta"}            # "RoBERT-FT1 ... RoBERTa-FT2" in one sentence

# derived family -> base family (0.5 in W). Kinds: domain = further training on domain data; language = trained
# for one language; multimodal = text model extended to other inputs; system = an application built on the model.
ADAPTED = {
    # BERT
    "mentalbert": ("bert", "domain"), "psychbert": ("bert", "domain"), "biobert": ("bert", "domain"),
    "clinicalbert": ("bert", "domain"), "bioclinicalbert": ("bert", "domain"), "clinicalbert_zh": ("bert", "domain"),
    "finbert": ("bert", "domain"), "covid_twitter_bert": ("bert", "domain"), "ct_bert": ("bert", "domain"),
    "twhin_bert": ("bert", "domain"), "bertweet": ("bert", "domain"), "med_bert": ("bert", "domain"),
    "madrs_bert": ("bert", "domain"), "therapistbert": ("bert", "domain"), "disorbert": ("bert", "domain"),
    "anxibert": ("bert", "domain"), "corebert": ("bert", "domain"), "emobert": ("bert", "domain"),
    "memobert": ("bert", "domain"), "tod_bert": ("bert", "domain"), "docbert": ("bert", "domain"),
    "topicbert": ("bert", "domain"), "opinion_bert": ("bert", "domain"), "babyberta": ("roberta", "domain"),
    "mbert": ("bert", "language"), "kobert": ("bert", "language"), "kcbert": ("bert", "language"),
    "kr_bert": ("bert", "language"), "korbert": ("bert", "language"), "kpfbert": ("bert", "language"),
    "klue_bert": ("bert", "language"), "kobigbird_bert": ("bert", "language"), "arabert": ("bert", "language"),
    "marbert": ("bert", "language"), "bertimbau": ("bert", "language"), "banglabert": ("bert", "language"),
    "hebert": ("bert", "language"), "alephbert": ("bert", "language"), "bertje": ("bert", "language"),
    "gbert": ("bert", "language"), "german_bert": ("bert", "language"), "urdubert": ("bert", "language"),
    "tohoku_bert": ("bert", "language"), "chinese_bert_wwm_ext": ("bert", "language"), "macbert": ("bert", "language"),
    "sbert": ("bert", "domain"), "mm_bert": ("bert", "multimodal"), "mag_bert": ("bert", "multimodal"),
    "videobert": ("bert", "multimodal"), "visualbert": ("bert", "multimodal"), "vilbert": ("bert", "multimodal"),
    # RoBERTa
    "mentalroberta": ("roberta", "domain"), "emoberta": ("roberta", "domain"), "emoroberta": ("roberta", "domain"),
    "twitter_roberta": ("roberta", "domain"), "deproberta": ("roberta", "domain"),
    "robertadepressiondetection": ("roberta", "domain"), "cd_roberta": ("roberta", "domain"),
    "ear_roberta": ("roberta", "domain"), "s_roberta": ("roberta", "domain"),
    "klue_roberta": ("roberta", "language"), "chinese_roberta_wwm_ext": ("roberta", "language"),
    "erlangshen_roberta": ("roberta", "language"), "kosimcse_roberta": ("roberta", "language"),
    "kobigbird_roberta": ("roberta", "language"), "roberta_beto": ("roberta", "language"),
    "camembert": ("roberta", "language"), "gottbert": ("roberta", "language"), "umberto": ("roberta", "language"),
    "robertuito": ("roberta", "language"), "wangchanberta": ("roberta", "language"), "robbert": ("roberta", "language"),
    "twitter_xlm_roberta": ("xlm_roberta", "domain"),
    # GPT
    "biogpt": ("gpt", "domain"), "medgpt": ("gpt", "domain"), "biomedgpt": ("gpt", "domain"),
    "gatortrongpt": ("gpt", "domain"), "affectgpt": ("gpt", "domain"), "depgpt": ("gpt", "domain"),
    "eeg_gpt": ("gpt", "domain"), "neuro_gpt": ("gpt", "domain"), "instructgpt": ("gpt", "domain"),
    "kogpt2": ("gpt", "language"), "aragpt2": ("gpt", "language"), "czegpt": ("gpt", "language"), "bggpt": ("gpt", "language"),
    "gptcoach": ("gpt", "system"), "safegpt": ("gpt", "system"), "warmgpt": ("gpt", "system"), "mindgpt": ("gpt", "system"),
    "navgpt": ("gpt", "system"), "robotgpt": ("gpt", "system"), "creascoregpt": ("gpt", "system"),
    "gptscreenr": ("gpt", "system"), "gpthistory": ("gpt", "system"), "doximitygpt": ("gpt", "system"),
    "hivegpt": ("gpt", "system"), "mdteamgpt": ("gpt", "system"), "antgpt": ("gpt", "system"), "gesgpt": ("gpt", "system"),
    "acoustics_gpt": ("gpt", "system"), "luminescence_gpt": ("gpt", "system"), "bp_gpt": ("gpt", "system"),
    # others
    "mentallama": ("llama", "domain"), "emotion_llama": ("llama", "domain"), "llamaguard": ("llama", "domain"),
    "chinese_llama": ("llama", "language"), "cockpit_llama": ("llama", "system"), "video_llama": ("llama", "multimodal"),
    "med_palm": ("palm", "domain"), "flan_palm": ("palm", "domain"), "palm_e": ("palm", "multimodal"),
    "med_gemini": ("gemini", "domain"), "paligemma": ("gemma", "multimodal"), "embeddinggemma": ("gemma", "domain"),
    "mentalmistral": ("mistral", "domain"), "mdeberta": ("deberta", "language"), "nli_deberta": ("deberta", "domain"),
    "nli_distilroberta": ("distilroberta", "domain"), "distilkobert": ("distilbert", "language"),
    "dialogpt": ("gpt", "domain"), "openllama": ("llama", "domain"),
    "st_distilbert": ("distilbert", "domain"), "multi_qa_distilbert_cos": ("distilbert", "domain"),
    "all_distilroberta": ("distilroberta", "domain"), "all_roberta": ("roberta", "domain"),
    "german_roberta_sentence_transformer": ("roberta", "language"), "multi_roberta": ("roberta", "language"),
    "micro_bert": ("bert", "multimodal"), "micron_bert": ("bert", "multimodal"), "aobert": ("bert", "multimodal"),
    "audibert": ("bert", "multimodal"), "bertagent": ("bert", "system"), "augbert": ("bert", "domain"),
    # hybrids: the model combined with another architecture ("DeBERTa-BiLSTM", "SBERT-CNN")
    "fusion_convbert": ("bert", "hybrid"), "fusion_bert": ("bert", "hybrid"), "fusion_robustbert": ("bert", "hybrid"),
    "concat_bert": ("bert", "hybrid"), "dual_contextbert": ("bert", "hybrid"), "graph_bert": ("bert", "hybrid"),
    "lda_bert": ("bert", "hybrid"), "ensembert": ("bert", "hybrid"), "sbert_cnn": ("sbert", "hybrid"),
    "albert_bilstm": ("albert", "hybrid"), "albert_plusbert_bilstm": ("albert", "hybrid"),
    "deberta_bilstm": ("deberta", "hybrid"), "deberta_plusbilstm": ("deberta", "hybrid"),
    "hubert_plusernie_health": ("hubert", "hybrid"), "hubert_pluslongformer": ("hubert", "hybrid"),
    "hubert_clap": ("hubert", "hybrid"), "graph_roberta": ("roberta", "hybrid"), "graph_roberta_cl": ("roberta", "hybrid"),
    "acs_plusllama": ("llama", "hybrid"),
    # single-article BERT variants whose training was not checked
    "mpc_bert": ("bert", "unchecked"), "sa_bert": ("bert", "unchecked"), "sr_bert": ("bert", "unchecked"),
    "tmbert": ("bert", "unchecked"), "bhs_bert": ("bert", "unchecked"), "mabert": ("bert", "unchecked"),
    "melbert": ("bert", "unchecked"), "bubert": ("bert", "unchecked"),
}

_DIG = re.compile(r"^(?P<fam>[a-z_]*?[a-z])(?P<num>\d{1,3})$")

def glued_citation(root):
    """'bert28' -> 'bert', 'gpt_9' -> 'gpt' (no GPT-9), 'gpt_4' unchanged, 'gpt2'-style known versions unchanged."""
    m = _DIG.match(root)                      # family letters immediately followed by digits: bert28, roberta34
    if m and not m.group("fam").endswith("_"):
        fam, num = m.group("fam"), m.group("num")
        if num in KNOWN_VERSIONS.get(fam, set()) or (fam.endswith("gpt") and num == "2"): return root   # GPT2, kogpt2
        if fam in KNOWN_VERSIONS or fam in ADAPTED or fam in BASES: return fam
        return root
    head, _, tail = root.rpartition("_")      # family_digits: gpt_9, gemini_8, gpt_111, gpt_44
    if head in KNOWN_VERSIONS and tail.isdigit() and tail not in KNOWN_VERSIONS[head]: return head
    return root

BASES = {"bert", "roberta", "distilbert", "albert", "deberta", "distilroberta", "xlm_roberta", "modernbert",
         "electra", "hubert", "gpt", "llama", "gemini", "gemma", "claude", "qwen", "deepseek", "mistral", "palm", "grok"}

def canonical(root, pmcid=None):
    """Canonical focal-model root, or None if the atom is not a model (then it is kept as an ordinary concept)."""
    if (pmcid, root) in NOT_MODEL: return None
    root = BY_ARTICLE.get((pmcid, root), root)
    root = CANON.get(root, root)
    return CANON.get(glued_citation(root), glued_citation(root))
