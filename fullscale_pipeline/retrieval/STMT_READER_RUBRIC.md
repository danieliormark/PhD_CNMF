# AI-use statement or research content? — reader rubric (v1, 2026-10-21)

Each item shows a sentence from a scientific article, marked ⟦like this⟧, with the article title, the section heading
above it, and the sentences before and after it. Decide what the marked sentence is.

**statement** — the sentence is about how *this paper itself* was produced with AI tools, or declares that they were or
were not used for it. Typical places: Acknowledgements, "Use of AI", "Declaration of generative AI", the end of Methods.
Examples:
- "During the preparation of this work the authors used ChatGPT to improve readability and language."
- "ChatGPT (OpenAI) was used only for language editing and grammar correction."
- "We acknowledge the use of GPT-4 for proofreading the manuscript."
- "No large language models were used in writing this paper."
- "Figure 1 was created with the help of ChatGPT." / "Claude was used to assist with drafting the R code for the analysis."
- A sentence that only completes such a declaration ("It was not used to generate or alter any data.").

**research** — anything else, including when the words are similar:
- the study's own methods or results with LLMs ("We used GPT-4 to extract key findings from the reports.",
  "LLMs were used for scientific writing by 48% of respondents.");
- surveys, interviews, participants' quotes about AI-assisted writing;
- other papers' statements or policies, reviews of disclosure practice, recommendations to authors;
- passing mentions of LLMs in the discussion.

**unsure** — the shown text does not decide it. Do not force a choice.

## Output
One JSON object per question, one per line, nothing else:
{"qid": "C001", "verdict": "statement|research|unsure", "note": "<at most 10 words>"}
