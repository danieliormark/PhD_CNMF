# Author identity checks on the working corpus v2 (RL-189, 2026-10-24)

Read-only analyses (no network) behind the owner's author-merging decisions of 2026-10-24 (design doc
`plans/ticket107-m1-chunk12v3-design.md`, addenda of 2026-10-24). They read `diagnostics/author_coverage/`
(bylines_v2_refresh.jsonl, openalex_works.json, pubmed_authors.json) and the header and match files of
`diagnostics/openalex_authors/`. Each script `exec`s the one before it (links_v2.py is the base), so run them from this
folder: `python3 <script>.py`.

| Script | What it measures |
|---|---|
| `links_v2.py` | the trial linking rule (author_links_trial.py) on all 12,106 unidentified bylines: step 1 (to an identified person) and step 2 (among unidentified), with examples; the OpenAlex IDs carrying several PubMed ORCIDs. Decision 1d applied (PubMed's ORCID wins where the two differ). |
| `strength.py` | ORCIDs with 2+ name keys; evidence strength of the step-1 links by name commonness |
| `orcid_names2.py` | ORCIDs whose bylines carry incompatible names (transliteration, compound surnames, initials, joined given names allowed) |
| `drop1b.py` | decision 1b: OpenAlex IDs with 2+ ORCIDs kept as one person only if every name is compatible and the bylines of each ORCID share an institution or an identified co-author; the rest dropped, and articles left with no author |
| `hide_recover.py` | strict B rule (owner-approved): hide identified bylines of people on 2+ articles, measure recovery and precision |
| `false_merge.py` | strict B: hide single-article people and remove them as candidates; every link is to another ID (upper bound of false merges) |
| `strict_sample.py` | strict B on the unidentified bylines, with a stratified sample for reading |
| `orcid_conflicts.py` | R3: ORCIDs on bylines with different names, outcome of the decided rule |
| `bias_check.py` | who R2 and R3 affect: country of the byline's institution, team size |
| `oa_name_conflicts.py` | problem C: OpenAlex IDs whose bylines carry different names (and the rejected R6 variant) |
| `c_delete.py` | problem C: deletion variants D1 (decided) and D2 |
| `pipeline_sim.py` | the whole sequence in order: `python3 pipeline_sim.py B_then_A pubmed --C` |
| `author_table.py` | **production (RL-200):** the decided sequence applied, R4 included; one row per byline in `diagnostics/author_coverage/author_table_v2.jsonl` (untracked), counts in `author_table_v2_summary.json`; reproduces `pipeline_sim.py` |

Rules, decisions and results: `fullscale_pipeline/AUTHOR_IDENTITY.md`; log RL-189 to RL-192.
