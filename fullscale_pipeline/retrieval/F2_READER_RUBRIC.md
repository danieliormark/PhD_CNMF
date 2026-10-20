# Manual check of ambiguous focal terms — reader rubric (v1, 2026-10-21)

You read a batch file. Each article shows its title, journal, year, the start of its abstract, keywords, two facts
computed from the whole article, and one or more passages in which a term is marked ⟦like this⟧. Each QUESTION asks about
one term in one article. Judge only from what is shown; do not guess beyond it.

## The question
Does the marked term refer to the focal concept in this article?
- **language model, LLM(s), BERT, RoBERTa, GPT, Gemini, Llama, Bard, Claude, ProGen, …**: the focal concept is a
  computational *neural* language model or a model/product built on one: large language models and chatbots (ChatGPT,
  Gemini, Claude, Llama, Bard), the BERT/GPT families, and protein, DNA or chemical language models (ESM, ProGen,
  ProtBERT, …). A passing mention counts ("tools such as BERT", "protein language models predict…").
- **transformer model(s)**: the focal concept is the machine-learning *transformer* architecture (any field: text,
  images, time series, proteins). An electrical transformer or a transformer circuit is not.

Answer **no** when the term means something else, for example:
- an abbreviation for another thing: LLM = lipid-lowering medication, low lean mass, linear mixed model, lower limit of
  normal, Logic Learning Machine, lateral longitudinal muscle, a strain or sample code (LLM-320), a protein domain, a
  person's initials, Master of Laws; GPT = the enzyme glutamic pyruvic transaminase or its genes (GPT1/GPT2), Grooved
  Pegboard Test, GP trainee, polyspike train; BERT = bit error rate tester, a treatment regimen, a team name;
- a company, supplier or instrument: Progen/ProGen antibodies or reagents, Gemini Scientific serum, Gemini SEM;
- a person's name: Bard (A. J. Bard), Bert, Roberta;
- "language model" in linguistics, psychology, neuroscience or education (a caregiver as a child's language model; the
  dual-stream language model of the brain; a statistical model named "Language"), in programming-language theory, or a
  classical statistical n-gram / bigram language model.
Answer **statement** when the sentence only says that an AI tool was used to write, edit, translate, draw figures or
write code for the paper itself (an AI-use declaration).
Answer **unsure** when the shown text does not decide it. Do not force a choice.

## Output
One JSON object per question, one per line, nothing else:
{"qid": "A001.1", "verdict": "yes|no|statement|unsure", "reason": "<code>", "note": "<at most 12 words>"}
Reason codes: LM_MODEL (named or generic neural language model), LM_PROTEIN (protein/DNA/chemical LM), TRANSFORMER_ML,
OTHER_ABBREV, OTHER_SUPPLIER, OTHER_PERSON, OTHER_BIO (gene/enzyme/domain/strain), OTHER_LINGUISTIC, OTHER_ELECTRICAL,
OTHER_MISC, AI_USE_STATEMENT, UNCLEAR.
