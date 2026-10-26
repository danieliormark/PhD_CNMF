# Author identity in the working corpus v2 — methods (draft, 2026-10-26)

Written as a methods / appendix section. It states how bylines become persons for the M1 author facet. Rules marked
**decided** were set by the owner on 2026-10-24; rules marked **proposed** await the owner's decision. Nothing here has
been applied to data yet: the author table is built after the trial subset is chosen. Scripts:
`corpus_statistics/author_linking_v2/` (read-only analyses); log: RUN_LOG RL-188 to RL-190.

## 1. Terms

- **Byline**: one author of one article, as listed by PubMed: surname, given names, position, affiliation text, and an
  ORCID if the publisher supplied one. The working corpus v2 (41,640 articles) has 254,986 personal bylines.
- **OpenAlex author ID**: an identifier OpenAlex assigns by its own algorithm. A byline receives one when it is aligned
  with an OpenAlex authorship of the same article (the bylines are matched in order; a pair counts only when the
  surnames agree). 93.6% of bylines have one.
- **ORCID**: an identifier a researcher registers. We use the one PubMed gives (the publisher's); where OpenAlex shows
  a different ORCID for the same byline, PubMed's is used (decided, 1d).
- **Person**: the unit of the M1 author facet. Bylines are grouped into persons by the rules below.

## 2. Name comparison

Several rules need to decide whether two bylines can be the same person by their names alone. The test is deliberately
conservative: when in doubt it says "different", because a wrong merge joins two people's ties, while a missed merge
only splits one person's ties in two.

1. **Normalisation.** Lower case; accents removed (é → e, ą → a, ü → u); letters without a decomposition transliterated
   (æ → ae, ø → o, ı → i, ß → ss, ł → l, đ → d, œ → oe); each name field split into parts at spaces, hyphens and
   apostrophes ("Zheng-Hao" → zheng, hao).
2. **Same parts, divided differently.** If both names consist of exactly the same parts, they are the same person,
   whatever is in the surname field and what in the given-name field: "Varas Pardo, Pablo" = "Pardo, Pablo Varas";
   "Bhuiyan Masud, Jakir Hossain" = "Masud, Jakir Hossain Bhuiyan".
3. **Surnames.** Otherwise the surnames must agree: identical after joining their parts ("De Momi" = "Demomi"), or one
   surname found inside the other's full name, provided it has at least 4 letters and at least 60% of the length of the
   longer surname ("Momi, Elena De" = "Demomi, Elena"). The length condition prevents "Li" from matching "Liu".
4. **Given names.** If one side has no given name, the surname decides. Otherwise the given names must agree in one of
   two ways: (a) identical after joining their parts ("Zheng-Hao" = "Zhenghao"; "An'an" = "Anan"); or (b) compared part
   by part from the left, every pair of parts is identical or one part is the initial of the other, and one name may
   have fewer parts ("Andrew J E" = "Andrew John Ervine"; "C Raina" = "Chandini Raina"; "A M B" = "Ana M B";
   "F Javier" = "Francisco Javier").

| Pair | Result | Why |
|---|---|---|
| Li, Zhenghao / Li, Zheng-Hao | same | given names identical when joined |
| Seely, Andrew J E / Seely, Andrew John Ervine | same | parts agree, J and E are initials |
| MacIntyre, Chandini Raina / MacIntyre, C Raina | same | C is the initial of Chandini |
| Pardo, Pablo Varas / Varas Pardo, Pablo | same | same parts |
| Bjertnæs, Øyvind / Bjertnaes, Oyvind | same | transliteration |
| Li, Chen / Liu, Chen | different | surnames differ |
| Lee, Miji / Lee, Minjae | different | first given names differ |
| Xu, Shuping / Xu, Shicheng | different | first given names differ |
| Zhang, Wei / Zhang, Wen Wei | different | first given parts differ (wei, wen) |
| Brooksbank, Catherine / Brooksbank, Cath | different | a nickname is not an initial (known conservative error) |
| Kühnel, Lisa / Langnickel, Lisa | different | a changed surname (known conservative error) |

Known conservative errors: nicknames, changed surnames and misspelt surnames count as different people. Their effect
is a missed merge, never a wrong one.

## 3. Rules

**R1 (decided, 1a). One person under several OpenAlex IDs.** Bylines that share an ORCID are one person, even if
OpenAlex gave them different author IDs (OpenAlex split the person). Applied after R3, so that only checked ORCIDs merge.

**R2 (decided, 1b). One OpenAlex ID carrying several ORCIDs.** 102 OpenAlex IDs have bylines with two or more different
PubMed ORCIDs (522 bylines). Reading showed mostly OpenAlex joining different people under one ID (often two co-authors
of the same paper), so the ID is not trusted. Every byline under such an ID is **dropped** from the author facet,
except when both hold:
(a) every pair of names under the ID is the same person by §2; and
(b) for every pair of its ORCIDs, a byline with the one and a byline with the other, on different articles, share an
OpenAlex institution or a co-author whose identity we know.
An ID that passes is one person (its ORCIDs merged). A dropped byline is removed from the author facet only; its article
stays with its other authors. An article left with no author is removed from the working corpus.

| OpenAlex ID | Bylines | Outcome |
|---|---|---|
| A5103076360 | Church, George (ORCID …2076); Church, George M (…6232); both Department of Genetics, Harvard Medical School | kept: one person with two ORCIDs |
| A5100426938 | Luo, Xiao at UCLA Computer Science (…3714) and Luo, Xiao at Toyota Technological Institute at Chicago (…4043), 6 bylines | dropped: names agree, but the two ORCIDs share no institution or co-author (two people) |
| A5064979753 | Liu, Na (Stuttgart) and Yan, Hao (Arizona), co-authors of PMC11499255 | dropped: names differ |

Result: 41 IDs kept, 61 dropped: 323 bylines in 291 articles; 2 articles left with no author (PMC13216025,
PMC13447246).

**R3 (proposed). One ORCID on bylines with different names.** 609 ORCIDs appear on bylines whose names are different
people by §2 (783 bylines in the less frequent name; 129 on one article). Without a check, R1 would merge them: the ORCID
of Jianlin Cheng (12 bylines) also appears on one byline of his co-author Shreya Basnet, so they would become one person.
Many cases, however, are malformed name fields of one person ("E, Klang" for "Klang, Eyal"; "J, Andrew" for
"Jeyabose, Andrew"). Proposed procedure, per ORCID:
1. Group its bylines by name (§2).
2. If every name group also shares an OpenAlex author ID, the two independent sources agree: one person.
   (435 ORCIDs; e.g. "Klang, Eyal" ×47 and "E, Klang" ×1, all under OpenAlex A5039899830.)
3. Otherwise the ORCID stays with the largest name group only (68 ORCIDs; e.g. Cheng, Jianlin ×12 keeps it, Basnet,
   Shreya ×1 loses it). If two groups are equally large, nobody keeps it (106 ORCIDs; e.g. "Moreno, F Javier" and
   "Moreno, Francisco Javier" under two different OpenAlex IDs, one byline each).
4. A byline that loses the ORCID keeps everything else. If it has its own OpenAlex ID, that identifies it (232 bylines;
   Basnet keeps A5071604912). If its OpenAlex ID is one of R2's untrusted IDs, R2 drops it (32). If it has no ID left, it
   is treated like any byline without an identifier: R4 may link it, or it stays its own person (47).
   In all, 311 bylines lose the ORCID; none is dropped by R3 itself.
Alternative (not proposed): drop the bylines that lose the ORCID instead of keeping them without it.

**R4 (decided, 1c, strict option B). Bylines without OpenAlex ID or ORCID** (12,106) are linked to a person whose
identity we know only when all hold: (a) the name key agrees: surname and first given name, written out (a byline
with only an initial is never linked); (b) exactly one identified person with that key has corroborating evidence;
(c) the evidence, on other articles of that person, is at least two shared co-authors whose identity we know, or one
such co-author and a shared OpenAlex institution; (d) if 5 or more identified people share the name key, two
co-authors and a shared institution are required; (e) a co-author under an R2-untrusted ID does not count.
Otherwise the byline stays its own person. Result: 961 linked, 115 with two or more candidates (not linked), 10,848 stay
their own person, 182 with initials only. (Counted with all 102 IDs of R2 treated as untrusted co-authors, before R2's 41 exceptions were
fixed; with only the 61 dropped IDs untrusted the count can rise slightly.) Checks: hide-and-recover precision 0.9905 on 20,000 identified bylines;
false-merge upper bound 2.06% on 20,000 single-article people; 30 links read, 30 plausible (RL-189).

**R5 (decided, 2). Institutions** are climbed to the nearest university parent in OpenAlex's lineage, never above it.
The hand review and the bylines without an institution ID are decided after the trial subset is chosen.

## 4. Names are never sufficient on their own (owner question, 2026-10-26)

The name test of §2 is used only as a necessary condition or to separate people, never alone to join bylines:
R2 keeps an OpenAlex ID only if the names agree **and** the ORCIDs share an institution or a known co-author (the two
Xiao Luos agree by name and are not joined); R3 uses names to detect an ORCID on two people, while joining rests on the
ORCID and, in its step 2, also on OpenAlex's ID; R4 requires the name key **and** exactly one candidate **and** two
known co-authors (or one and an institution; two and an institution for common names). Example: "Liu, Tong" at Duke has
no ID; three identified Tong Lius exist in the corpus; none shares a co-author or institution with this byline, so it
stays its own person. Remaining risk: two people with the same name who also share co-authors or an institution (one
large group); measured as at most 2.06% of single-article people linked to another ID (RL-189), most of them one person
split by OpenAlex.

## 5. Comparison with established practice, and bias (2026-10-26)

**OpenAlex's own rules** (help.openalex.org, Authors → ORCID and → Disambiguation, read 2026-10-26): ORCID is "the
strongest" of six signals (name, co-authors, institutions, topics, citations, ORCID) but not a veto; an ORCID is not
trusted on a work when it is stamped on more than one author of that work; profiles sharing an ORCID are not merged
automatically, "because a shared ORCID is sometimes wrong"; the upstream errors named are swapped co-author iDs, one
author's iD stamped on every authorship, and an iD copied from the wrong form. **Literature:** ORCID-linked author labels
are a standard evaluation source for name disambiguation in MEDLINE (Kim & Owen-Smith 2021, Scientometrics 126,
matching on surname + first initial), and misapplied ORCIDs in publisher and repository metadata are documented as a
minority but real problem (Baglioni et al. 2021, Data Science Journal 20, "five observed misapplications").
**R3 against this:** step 2 (join only where OpenAlex's ID agrees) follows the multi-signal practice; distrusting an
ORCID that sits on two names follows OpenAlex's trust filter; the choice of the largest name group (step 3) is our own
heuristic, not a documented standard. **Refinement for alignment (proposed):** on an article where one ORCID is on two
or more bylines, the ORCID is dropped for every byline of that article (OpenAlex's rule), including the majority
name's; the majority keeps it on its other articles.

**Who is affected** (country of the byline's OpenAlex institution; `author_linking_v2/bias_check.py`):

| | Bylines | China | USA | median team size |
|---|---|---|---|---|
| all bylines | 254,986 | 21.1% | 22.1% | 7 |
| R2, dropped (1b) | 323 | 57.3% | 8.7% | 7 |
| R3, ORCID removed | 311 | 31.5% | 15.8% | 7 |

R2's drop falls on Chinese bylines at 2.7 times their share (about 185 of ~53,800 Chinese bylines, 0.34%, against about
0.05% of US bylines): OpenAlex joins different people most often under frequent Chinese names. R3 removes no byline; it
also affects Chinese bylines more than their share, but those bylines stay in the author facet with their own OpenAlex
ID. Deleting R3's bylines instead would add a second drop with the same skew.
