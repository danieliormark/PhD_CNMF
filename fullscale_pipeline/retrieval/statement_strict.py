"""Strict sentence-level AI-use statement rule (2026-10-21, RL-169), for single sentences inside kept articles. The broad
rule of gap_stage_b.is_statement stays the article-level F1 rule (validated on whole statement-only articles); inside
genuine LLM papers it was about 10% precise (4 of 40), because research prose uses the same words.
A sentence is a statement about the paper itself if
  * it uses a disclosure formula ("during the preparation of this work", "AI-assisted technologies in the writing
    process", "take(s) full responsibility for the content"), or
  * it negates use for this paper ("No AI tools were used in the preparation of this manuscript", "We did not use
    AI-assisted technologies … in the production of submitted work"), or
  * a use word is followed by an unambiguous disclosure task (language editing, proofreading, grammar, spelling,
    readability, manuscript preparation …), or by this paper as the object of the task ("used to assist with proofreading … of the
    manuscript", "for refining some text of this paper"), or this paper is the passive subject of an editing verb
    ("The manuscript was edited / proofread / translated with …"); every path needs the sentence to carry no research
    marker (participants, respondents, percentages, accuracy, evaluated, prompts, responses, outputs …).
Next to such a statement, a looser negation in the same article ("ChatGPT was not used to generate the patient images")
also counts (companion rule, as in Stage B)."""
import re
import gap_stage_b as SB

PAPER = (r"(?:(?:this|our) (?:manuscript|paper|article|work|review|report|submission|study)|the manuscript|the submitted work"
         r"|the (?:present|current) (?:manuscript|paper|article|study))")
TASK = (r"(?:language (?:editing|polishing|refinement|correction|improvement|check\w*)|English(?: language)? (?:editing|polishing|"
        r"refinement|correction)|proof-?reading|grammar|grammatical|spelling|readability of (?:the|this|our) (?:manuscript|paper|article|text)|copy-?edit\w*|writing process|"
        r"manuscript (?:preparation|writing|drafting|revision|editing)|improv\w* (?:the )?(?:language|wording|clarity|flow|style))")
RESEARCH = re.compile(r"(?i)participants|respondents|surveyed|students|patients|\d+(?:\.\d+)?\s?%|accura|evaluat|compar|"
                      r"\bprompt|responses|answers|questions|outputs?\b|performance|benchmark|defined|eligible|\bstudies\b|included|"
                      r"one stated|reported that|literature\b|materials\b|\bsentences\b|\bwords\b|descriptions|as described|"
                      r"generate responses|to the questions")
FORMULA = re.compile(r"(?i)during the (?:preparation|writing|revision|drafting)[^.]{0,30}of (?:this|the) (?:work|manuscript|paper|article|review|study)"
                     r"|AI[- ]assisted technolog\w* (?:in|during|for) the (?:writing|preparation|scientific writing)"
                     r"|declaration of (?:generative )?AI|take[s]? (?:full )?responsibility for the content")
NEG = re.compile(rf"(?i)\b(?:no|not|never)\b[^.]{{0,150}}\b(?:use|used|utili[sz]ed|employed|involved)\b[^.]{{0,150}}"
                 r"(?:manuscript|submitted work|the production of|the preparation of|the writing of|(?:this|our) (?:paper|article))")
USE_FOR = re.compile(rf"(?i){SB.AIUSE}[^.]{{0,160}}\b(?:of|in|for|to|on) {PAPER}\b")
DISCLOSE = re.compile(rf"(?i){SB.AIUSE}[^.]{{0,160}}{TASK}")
PASSIVE = re.compile(rf"(?i)\b{PAPER}\b[^.]{{0,60}}\b(?:was|were|has been|have been) (?:\w+ ){{0,3}}(?:edited|revised|proof-?read|"
                     rf"polished|translated|refined|checked|improved|corrected|written|drafted|formatted|prepared|assisted)\b")


def is_statement_strict(t):
    return bool((FORMULA.search(t) or NEG.search(t) or PASSIVE.search(t) or DISCLOSE.search(t)) and not RESEARCH.search(t))
