"""Group-4 Haiku review (RL-173): turns the readers' verdicts on the 4,173 spellings into decisions/g4_haiku_decisions.csv
(spelling, pmcid, action, target, reason, source), merged into g4_decisions.csv by write_g4_decisions.py.
  model       -> keep (source haiku)                       not_model -> reject (source haiku)
  mixed/unsure -> the assistant's decision (source read)   and the assistant's corrections of not_model verdicts below
The assistant read every not_model, mixed and unsure verdict (490) and 50 random model verdicts (50 correct). Lines used:
  - a model's own code class (TFBertModel, GPT2LMHeadModel, DistilBertTokenizer ...) names that model: map to it;
  - "<model>-extracted / -encoded / -assisted / -generated / -guided" refers to the model: map to it (as "-based");
  - a named LLM application, assistant or agent is a model; a named method, library, metric, framework, dataset,
    label, variable or repository is not.
    python3 g4_haiku_decisions.py
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_inputs_v3_1/g4/haiku/"
MAP_CODE = {"TFBertModel": "BERT", "GPT2LMHeadModel": "GPT-2", "TFGPT2LMHeadModel": "GPT-2", "DistilBertModel": "DistilBERT",
            "ModernBertForMaskedLM": "ModernBERT", "DistilBertPreTrainedModel": "DistilBERT", "TFBertForSequenceClassification": "BERT",
            "CamemBertForSequenceClassification": "CamemBERT", "CamemBERTSequenceClassifier": "CamemBERT", "MobileBertTokenizer": "MobileBERT",
            "RoBERTaTokenizer": "RoBERTa", "distilBertTokenizerFast": "DistilBERT", "RoBERTaFastTokenizer": "RoBERTa",
            "AlBertTokenizer": "ALBERT", "DistilBertTokenizer": "DistilBERT", "DistilBertPreprocessor": "DistilBERT",
            "BERTJapaneseTokenizer": "BERT", "BERTCheckpoint": "BERT", "GPT2Blocks": "GPT-2", "LlamaAttention": "LLaMA",
            "LlamaAttention.forward": "LLaMA"}
MAP_REF = {"SBERT-encoded": "SBERT", "caBERT-extracted": "caBERT", "MaTableGPT‐extracted": "MaTableGPT",
           "VisualCheXbert-enhanced": "VisualCheXbert", "TrialGPT-aggregated": "TrialGPT", "CheXbert-based": "CheXbert",
           "BERTMap-generated": "BERTMap", "MedGemma-extracted": "MedGemma", "DistilBERT-extracted": "DistilBERT",
           "EyeGPT-assisted": "EyeGPT", "EyeGPT+ophthalmologist": "EyeGPT", "ChatGPT4‐assisted": "ChatGPT-4",
           "post-ChatGPT-4": "ChatGPT-4", "trust.ChatGPT-generated": "ChatGPT", "pathology.ChatGPT-generated": "ChatGPT",
           "ChatGPTgenerated": "ChatGPT", "GPTwhen": "ChatGPT", "ChatGPTas": "ChatGPT", "ForChatGPT": "ChatGPT",
           "LLaMA-guided": "LLaMA", "AutomataGPT‐inferred": "AutomataGPT", "AMBERT+MEAM": "AMBERT", "SBERT-and": "SBERT",
           "sciences.ALBERT": "ALBERT", "settings.GPT4TS56": "GPT4TS", "F1-score.Wind2vec-BERT": "Wind2vec-BERT",
           "dimensions.GPT-o3-mini": "GPT-o3-mini", "GPTNeox": "GPT-NeoX", "CheXbert22": "CheXbert", "CheXbert40": "CheXbert",
           "CheXbert58": "CheXbert", "GPTCelltype19": "GPTCelltype", "HuggingGPT8": "HuggingGPT", "BERTMap12": "BERTMap"}
MODEL = ("GraphGPT ExBERT AffectGPT SCBERT OpenGPT RobotGPT ZotGPT MistralOCR PeptideBert TreeBERT elEmBERT MaTableGPT "
         "MarkerGeneBERT MedLocalGPT NaviGPT InterviewGPT MemGPT GPTBioInsightor CurateGPT FraudGPT BERTWalk LLaMA-Berry "
         "VisualCheXbert SteelsGPT hackingBuddyGPT").split()        # mixed/unsure decided model (LLM apps, agents, models)


def main():
    ans, meta = {}, {}
    for b in range(1, 18):
        meta.update(json.load(open(D + f"batch_{b:02d}.json")))
        for l in open(D + f"batch_{b:02d}_haiku.jsonl"):
            if l.strip():
                x = json.loads(l); ans[x["qid"]] = x
    rows = []
    for q in sorted(ans):
        sp, v, note = meta[q]["spelling"], ans[q]["verdict"], ans[q].get("note", "")
        if sp in MAP_CODE:
            rows.append((sp, "", "map", MAP_CODE[sp], "the model's own code class", "read (Haiku: " + v + ")"))
        elif sp in MAP_REF:
            rows.append((sp, "", "map", MAP_REF[sp], "refers to the model", "read (Haiku: " + v + ")"))
        elif sp in MODEL:
            rows.append((sp, "", "keep", "", "LLM app, agent or model", "read (Haiku: " + v + ")"))
        elif v == "model":
            rows.append((sp, "", "keep", "", note, "haiku"))
        elif v == "not_model":
            rows.append((sp, "", "reject", "", note, "haiku"))
        else:                                               # remaining mixed/unsure: method, library, label, notation, repository
            rows.append((sp, "", "reject", "", "read: " + note, "read (Haiku: " + v + ")"))
    with open(os.path.join(HERE, "decisions", "g4_haiku_decisions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["spelling", "pmcid", "action", "target", "reason", "source"]); w.writerows(rows)
    import collections
    print(len(rows), collections.Counter((r[2], r[5].split(" ")[0]) for r in rows))


if __name__ == "__main__":
    main()
