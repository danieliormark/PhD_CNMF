# Model-family grouping: versions of the same model/vendor lineage collapse to one
# family (ChatGPT/GPT-3/GPT-4/... -> GPT; Llama/Llama2/Llama3 -> Llama; Mistral/Mixtral
# -> Mistral), but distinct named models that merely share an architecture name
# (BioBERT, SciBERT, DNABERT, ...) stay separate, per the user's own example
# ("same for mistral, different berts"). Generic/bare architecture terms (BERT,
# RoBERTa, GPT, "language model", "Transformer model") are kept as their own
# "generic" bucket, not merged into any specific model's family.
import sys
sys.path.insert(0, "/mnt/hum01-rds/Basov/p91688di/pmc_preprocessing")
import focal_terms as ft

FAMILY = {
    # --- Major chat/instruct LLM vendors: all versions/products -> one family ---
    "ChatGPT": "GPT (OpenAI)", "GPT-3": "GPT (OpenAI)", "GPT-3.5": "GPT (OpenAI)", "GPT-4": "GPT (OpenAI)",
    "GPT-4o": "GPT (OpenAI)", "GPT-4.5": "GPT (OpenAI)", "GPT-5": "GPT (OpenAI)", "InstructGPT": "GPT (OpenAI)",
    "GPT4": "GPT (OpenAI)", "GPT4o": "GPT (OpenAI)", "GPT-2": "GPT (OpenAI)", "GPT2": "GPT (OpenAI)",
    "GPT-4V": "GPT (OpenAI)", "GPT 4": "GPT (OpenAI)", "OpenAI o1": "GPT (OpenAI)", "OpenAI o3": "GPT (OpenAI)",
    "Chat GPT": "GPT (OpenAI)", "Chat-GPT": "GPT (OpenAI)",
    "Claude 3": "Claude (Anthropic)", "Claude 3.5": "Claude (Anthropic)", "Claude Opus": "Claude (Anthropic)",
    "Claude Sonnet": "Claude (Anthropic)", "Claude Haiku": "Claude (Anthropic)", "Claude": "Claude (Anthropic)",
    "Grok-1": "Grok (xAI)", "Grok-2": "Grok (xAI)", "Grok-3": "Grok (xAI)", "Grok": "Grok (xAI)",
    "Google Gemini": "Gemini (Google)", "Gemini": "Gemini (Google)",
    "Google Bard": "Bard (Google)", "Bard": "Bard (Google)",
    "PaLM 2": "PaLM (Google)", "PaLM-E": "PaLM (Google)", "PaLM": "PaLM (Google)",
    "Llama 2": "Llama (Meta)", "Llama 3": "Llama (Meta)", "Llama-2": "Llama (Meta)", "Llama-3": "Llama (Meta)",
    "Llama2": "Llama (Meta)", "Llama3": "Llama (Meta)", "LLaMA": "Llama (Meta)", "Llama": "Llama (Meta)",
    "Mistral AI": "Mistral", "Mixtral": "Mistral", "Mistral": "Mistral",
    "DeepSeek": "DeepSeek", "Qwen": "Qwen",
    "GitHub Copilot": "Copilot (Microsoft/GitHub)", "Microsoft Copilot": "Copilot (Microsoft/GitHub)",
    "Perplexity AI": "Perplexity AI",
    "RoBERTa": "RoBERTa (generic/base)",
    "Gemma": "Gemma (Google)",
    # --- Generic / bare architecture terms: own bucket, not folded into a vendor family ---
    "LLM": "LLM (generic term)", "LLMs": "LLM (generic term)",
    "large language model": "LLM (generic term)", "large language models": "LLM (generic term)",
    "language model": "language model (generic term)", "language models": "language model (generic term)",
    "GPT": "GPT (bare/unspecified)", "BERT": "BERT (bare/unspecified)",
    "Transformer model": "Transformer (generic term)", "Transformer models": "Transformer (generic term)",
    "Bidirectional encoder representations from transformers": "BERT (bare/unspecified)",
}
# Every other PLAIN/GUARDED/GENERIC entry not listed above is its own, distinct family
# (one model/tool per family) -- this is the "different berts" case: BioBERT, SciBERT,
# PubMedBERT, DNABERT, ChemBERTa, ESM-2, ProtGPT2, MolT5, etc. all stay separate.


def family_of(term):
    return FAMILY.get(term, term)


if __name__ == "__main__":
    all_terms = sorted(set(FAMILY) | set(ft.PLAIN_WORDS) | set(ft.GUARDED_NAMES) | set(ft.GENERIC_NAMES))
    fams = sorted(set(family_of(t) for t in all_terms))
    print(f"{len(all_terms)} distinct canonical terms -> {len(fams)} families")


import re as _re
_FAM_L = {k.lower(): v for k, v in FAMILY.items()}
_CANON_L = {w.lower(): w for w in list(ft.PLAIN_WORDS) + list(ft.GUARDED_NAMES) + list(ft.GENERIC_NAMES)}
_GPT = _re.compile(r"^(chat[ -]?gpt|gpt[ -]?\d)")          # P2's EXTRA_PLAIN spellings: gpt 3.5, gpt3.5, gpt 4.1, gpt4v, ...
_LLAMA = _re.compile(r"^llama[ -]?\d")                      # llama-3.1, llama3.2, llama 4, llama-2.0, ...


def family_of_term(term):
    """Family for a term string as P2 records it (any case, any spelling variant P2's rules accept).
    Added 2026-10-02 after 53 P2 spellings were found unmapped by the case-sensitive lookup."""
    t = term.lower().strip()
    if t in _FAM_L:
        return _FAM_L[t]
    if _GPT.match(t):
        return "GPT (OpenAI)"
    if _LLAMA.match(t):
        return "Llama (Meta)"
    return _CANON_L.get(t, term)
