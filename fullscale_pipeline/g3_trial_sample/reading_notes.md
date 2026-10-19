# Reading notes on `sample_30.md` (assistant, 2026-10-19; the owner reads the sample too)

Sample: 30 units drawn at random (seed 107) from the 6,855 units of the 463 trial articles (2020/2021),
`corpus_statistics/g3_trial_sample.py`. Overall: parents, children and cousins are coherent with the G3 rules
(G3_POSTPROCESSING.md §6); focal-term canonicalisation works as designed (e.g. "BERT (Bidirectional Encoder
Representations from Transformers)" → `bert`, #12, #20); no LLM-generated or non-prose text. Points to note:

1. **Provenance is per sentence, not per hyperedge, when a word repeats.** An atom's source words list every
   occurrence of the same parsed atom in the sentence (#4: `stage` ← the 5 occurrences of "stage", in every hyperedge
   that has `stage`), because graphbrain atoms are matched to tokens by their string. In the trial slices 21.1% of
   hyperedges (12.4% of atoms; 50.8% of units) contain such an atom. The model is unaffected (identity comes from the
   atoms), but tracing a community's hyperedges back to the exact original words (owner's interpretation requirement)
   is ambiguous for these: the sentence is known, the exact occurrence is not.
2. **The same focal term gets two atom types.** "fine_tuned BERT" → `bert/C/en` inside a sibling (#23) while other
   BERTs are `bert/C/focal`; "Bioclinical-BERT" → `bioclinical/C/en` + `bert/C/en` next to `bert/C/focal` and
   `pubmedbert/C/focal` in one hyperedge (#25). Atom types are part of the identity, so `bert/C/en` and
   `bert/C/focal` hyperedges are different entities. To check against G3's focal-marking rule (occurrence-level vs
   term-level) and the hyphen rule.
3. **Percent-encoded atoms and numbers:** `lucene%28a` ("Lucene(a"), `83%2e56`, `0%2e1` (#9, #28) — graphbrain's
   escaping of special characters; numbers kept as atoms make one-off hyperedges. Decode for reading; whether numbers
   should be atoms at all is a G3 question.
4. **Participles as the parent's verb group.** When the focal term sits inside a noun phrase with a participle
   ("a transformer model trained in this way", "the classifier based on BERT", "our proposed BERT-based classifiers",
   "datasets used for language model training"), the parent's verb group is that participle (train, base, propose,
   use) and the sentence's main predicate (would yield, used … to label, reports, show) becomes a cousin (#1, #6, #10,
   #29). This follows the rule (verb group of the clause governing the focal term), but it makes many parent
   statements generic: in the trial, 9.7% of parents have the single verb `use`, 1.8% `base`, 1.8% `train`
   (and 11.1% `be`).
5. **Focal term inside a complement clause attached to the higher clause:** "it is imperative that … language models
   reliably produce items …" → parent verb `be` (is imperative); "produce" ends up inside a sibling bag (#16). One case.
6. **Large flattened bags:** coordinated lists and long prepositional chains become one hyperedge (#2, #3, #17; #25
   has 16 atoms). By design, but such hyperedges are almost always unique to one article, so they link nothing;
   relevant to weighting (P2).
7. **Fragment cousins:** a lone modifier as a cousin, `(cousin_he pretrained/M/en)` (#4). Rare.
