# RL-196 (owner 2026-10-26): demotion of bare "language model(s)" mentions in the Psychology trial subset (OpenAlex
# primary-topic field Psychology, corpus_statistics/subset_candidates_by_article.csv). The focal matcher accepts
# "language model(s)" without article evidence (focal_terms.py: GENERIC, no NEED_EVIDENCE), so a mention can mean a
# person modelling language for a child, a theory, a regression model, or an older (non-transformer) language model.
# Owner: such mentions are demoted (no longer focal), not their articles removed; an article leaves only if no focal
# unit remains. Scope (owner): a language model counts when it is a pretrained neural language model, transformer or
# comparable (RL-195/196). Per G3 unit whose focal mentions include bare "language model", in this order:
#   1. false positive (not a computational model), hand-read list psychology_lm_false_positive_units.csv -> demote
#   2. another focal mention in the same unit -> keep
#   3. strong transformer-era signal in the unit (a named model, transformer, LLM, generative, chatbot, prompt, PLM,
#      masked language model, vision-language ...) -> keep
#   4. specific older-type signal (n-gram, smoothing, HMM, word2vec, LSTM, ELMo, Flair, spaCy, Stanza, GloVe, LSA,
#      open-vocabulary ...) -> demote, unless the sentence is in HAND_KEEP
#   4b. HAND_DEMOTE_ARTICLES (read by hand) -> demote
#   5. weak transformer-era signal (pretrained, fine-tuned, contextual language model) -> keep
#   6. the article names a model or "LLM" elsewhere (the article evidence GPT already needs) -> keep
#   7. the article-level hand reading of the 190 Psychology articles with bare "language model" as their only focal
#      evidence (psychology_lm_only_reading.csv): A (LLM sense) -> keep; B (older type) or C (not a model) -> demote
# The v3 matcher's hyphenated or underscored form (v3:A6:language-model) is the same term and is treated alike; each
# row carries the mention's own term string (G3 compares without case). A sentence's units never disagree (checked; the script stops if they do), so a row has sentence scope, the form of
# term_rejections.tsv. Writes PG/g3_inputs_v3_1/lm_demotions_psychology.tsv (scope, key, term, reason; read by
# g3_curation_v2.py --extra-rejections) and corpus_statistics/psychology_lm_unit_decisions.csv (one row per unit).
import ast, collections, csv, glob, json, re, sys

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
CS = FP + "corpus_statistics/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
sys.path.insert(0, CS)
from family_map import family_of_term

LM = "language model (generic term)"
ST = re.compile(r"(?i)(transformer|\bBERT|BERT\b|RoBERTa|GPT|\bLLMs?\b|large[_ -]language|large[_ -]scale language|generative|"
                r"chatbot|ChatGPT|Llama|Claude|Gemini|foundation model|vision[_\W]+language|visual language|multimodal language|"
                r"multi_modal language|\bprompt|in_context|few_shot|hallucinat|RLHF|aligned language|\bmpnet|XLNet|\bT5\b|BART|PaLM|"
                r"LaMDA|small language model|\bSLMs?\b|OpenAI|Anthropic|autoregressive|auto_regressive|masked language|Qwen|Gemma|"
                r"Mistral|DeepSeek|ELECTRA|DeBERTa|DistilBERT|Baichuan|HyperClova|SmolLM|PsychAdapter|retrieval augmented|\bPLMs?\b)")
OS = re.compile(r"(?i)(n_?-?gram|\d-?gram|unigram|bigram|trigram|smooth|Witten|Kneser|HMM|hidden Markov|word2vec|\bW2V\b|LSTM|"
                r"Flair|ELMo|spaCy|SciSpacy|en_core|Stanza|Forced Aligner|Cloud Speech|PRLM|phone recogni|NPYLM|Pitman|GloVe|"
                r"fastText|subs2vec|open_vocabulary|\bLSA\b)")
WT = re.compile(r"(?i)(pre_?trained|pretrain|fine[_ -]?tun|contextual(?:ized|ised)? language)")
HAND_KEEP = {"PMC9198501"}            # "unlike predecessor language models such as LSA or word2vec": about contextual models
HAND_DEMOTE_ARTICLES = {"PMC13136401"}  # "pretrained word embeddings": static embeddings, older type

psy = {r["pmcid"]: r for r in csv.DictReader(open(CS + "subset_candidates_by_article.csv")) if r["field"] == "Psychology"}
fp_units = {r["uid"] for r in csv.DictReader(open(CS + "psychology_lm_false_positive_units.csv"))}
hand = {r["pmcid"]: r["class"] for r in csv.DictReader(open(CS + "psychology_lm_only_reading.csv"))}
named = lambda r: any(not f.rsplit("=", 1)[0].endswith("(generic term)") or f.startswith("LLM (generic term)")
                      for f in r["families"].split("|") if f)


def decide(u, p, only, text):
    if u in fp_units:
        return "demote", "false positive: not a computational model"
    if not only:
        return "keep", "other focal mention in the unit"
    if ST.search(text):
        return "keep", "strong transformer-era signal"
    if OS.search(text):
        return ("keep", "hand: about contextual models") if p in HAND_KEEP else ("demote", "older-type language model (sentence signal)")
    if p in HAND_DEMOTE_ARTICLES:
        return "demote", "older-type language model (hand)"
    if WT.search(text):
        return "keep", "weak transformer-era signal"
    if named(psy[p]):
        return "keep", "article names a model or LLM"
    c = hand.get(p)
    if c == "A":
        return "keep", "hand reading: LLM sense"
    if c in ("B", "C"):
        return "demote", "older-type language model (hand)" if c == "B" else "false positive: not a computational model (hand)"
    sys.exit(f"{u}: undecided")


rows, by_sid, terms = [], collections.defaultdict(set), collections.defaultdict(set)
for f in sorted(glob.glob(PG + "g3_v2/shards/g3_test_*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["pmcid"] not in psy:
            continue
        fm = r["focal_mentions"]; fm = ast.literal_eval(fm) if isinstance(fm, str) else (fm or [])
        lm = [m for m in fm if family_of_term(m["term"]) == LM or m["term"].startswith("v3:A6:")]
        if not lm:
            continue
        d, why = decide(r["uid"], r["pmcid"], len(lm) == len(fm), r["text"])
        rows.append(dict(uid=r["uid"], sid=r["sid"], pmcid=r["pmcid"], decision=d, reason=why,
                         terms="|".join(sorted({m["term"] for m in lm}))))
        by_sid[r["sid"]].add((d, why if d == "demote" else ""))
        terms[r["sid"]].update(m["term"] for m in lm)
mixed = [s for s, v in by_sid.items() if len({d for d, _ in v}) > 1]
if mixed:
    sys.exit(f"sentences with both keep and demote: {mixed[:5]}")
with open(CS + "psychology_lm_unit_decisions.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(PG + "g3_inputs_v3_1/lm_demotions_psychology.tsv", "w", encoding="utf-8") as fo:
    fo.write("scope\tkey\tterm\treason\n")
    for s in sorted(by_sid):
        v = by_sid[s]
        if any(d == "demote" for d, _ in v):
            for t in sorted(terms[s]):
                fo.write(f"sentence\t{s}\t{t}\t{sorted(w for d, w in v if d == 'demote')[0]}\n")
c = collections.Counter((r["decision"], r["reason"]) for r in rows)
for (d, why), n in sorted(c.items()):
    print(f"{n:5d} {d:6s} {why}")
print("sentences demoted:", sum(1 for v in by_sid.values() if any(d == "demote" for d, _ in v)),
      "| articles with a demoted unit:", len({r["pmcid"] for r in rows if r["decision"] == "demote"}))
