# Generic child hyperedges — final classification after the owner's review (ticket 107, 2026-11-19; RL-289)

Hyperedges used by at least 50 articles (trial build, hold-out group 1, visible ties), in two tables: **G** genre of scientific writing (candidates for exclusion) and **S** substantive. Decided so far (owner, 2026-11-19): methods and objects of model building / use are substantive (former class 'ambiguous'); focal topic terms are substantive. Rows marked *borderline* are genre words that may denote actions carried out by the authors — please decide them. To correct: write G or S in the last column, or name words to move between lists; the rule is then re-applied to all child hyperedges (word lists: `fullscale_pipeline/m1/fit/generic_hyperedges.py`).

| Class (all child hyperedges) | Hyperedges | Single-word | Share of ties | Share of raw usage | Share of weighted values |
|---|---|---|---|---|---|
| G | 953 | 131 | 12.1% | 16.4% | 8.2% |
| S | 96897 | 6509 | 87.9% | 83.6% | 91.8% |

## Genre (G) — 114 hyperedges

| # | Hyperedge (typed atoms) | Articles using it | Note | Owner's class |
|---|---|---|---|---|
| 1 | `use/P/en` | 1035 |  | |
| 2 | `be/P/en` | 984 |  | |
| 3 | `show/P/en` | 483 |  | |
| 4 | `demonstrate/P/en` | 418 |  | |
| 5 | `include/P/en` | 353 |  | |
| 6 | `have/P/en` | 327 |  | |
| 7 | `be/M/en use/P/en` | 315 |  | |
| 8 | `suggest/P/en` | 307 |  | |
| 9 | `indicate/P/en` | 282 |  | |
| 10 | `example/C/en` | 278 |  | |
| 11 | `make/P/en` | 273 |  | |
| 12 | `base/P/en` | 261 |  | |
| 13 | `result/C/en` | 247 |  | |
| 14 | `find/P/en` | 240 |  | |
| 15 | `remain/P/en` | 231 |  | |
| 16 | `offer/P/en` | 226 |  | |
| 17 | `leverage/P/en` | 222 |  | |
| 18 | `employ/P/en` | 207 |  | |
| 19 | `explore/P/en` | 204 |  | |
| 20 | `report/P/en` | 200 |  | |
| 21 | `follow/P/en` | 197 |  | |
| 22 | `highlight/P/en` | 197 |  | |
| 23 | `present/P/en` | 197 |  | |
| 24 | `represent/P/en` | 195 |  | |
| 25 | `integrate/P/en` | 195 |  | |
| 26 | `introduce/P/en` | 191 |  | |
| 27 | `perform/P/en` | 178 |  | |
| 28 | `finding/C/en` | 176 |  | |
| 29 | `be/P/en not/M/en` | 174 |  | |
| 30 | `produce/P/en` | 163 |  | |
| 31 | `more/M/en` | 159 |  | |
| 32 | `address/P/en` | 157 |  | |
| 33 | `reveal/P/en` | 150 |  | |
| 34 | `involve/P/en` | 147 |  | |
| 35 | `incorporate/P/en` | 136 |  | |
| 36 | `instance/C/en` | 134 |  | |
| 37 | `exhibit/P/en` | 133 |  | |
| 38 | `approach/C/en` | 133 |  | |
| 39 | `addition/C/en` | 132 |  | |
| 40 | `consider/P/en` | 131 |  | |
| 41 | `become/P/en` | 128 |  | |
| 42 | `research/C/en` | 124 |  | |
| 43 | `emerge/P/en` | 123 |  | |
| 44 | `time/C/en` | 123 |  | |
| 45 | `reflect/P/en` | 120 |  | |
| 46 | `obtain/P/en` | 119 |  | |
| 47 | `well/M/en` | 116 |  | |
| 48 | `take/P/en` | 111 |  | |
| 49 | `relate/P/en` | 107 |  | |
| 50 | `result/P/en` | 107 |  | |
| 51 | `be/P/en may/M/en` | 103 |  | |
| 52 | `work/C/en` | 103 |  | |
| 53 | `lead/P/en` | 102 |  | |
| 54 | `note/P/en` | 100 |  | |
| 55 | `datum/C/en` | 98 |  | |
| 56 | `context/C/en` | 98 |  | |
| 57 | `base/P/en be/M/en` | 97 |  | |
| 58 | `give/P/en` | 96 |  | |
| 59 | `contain/P/en` | 93 |  | |
| 60 | `yield/P/en` | 93 |  | |
| 61 | `adopt/P/en` | 91 |  | |
| 62 | `underscore/P/en` | 88 |  | |
| 63 | `way/C/en` | 88 |  | |
| 64 | `serve/P/en` | 88 |  | |
| 65 | `illustrate/P/en` | 87 |  | |
| 66 | `be/M/en show/P/en` | 86 |  | |
| 67 | `refer/P/en` | 85 |  | |
| 68 | `finding/C/en our/M/en` | 84 |  | |
| 69 | `raise/P/en` | 81 |  | |
| 70 | `future/M/en research/C/en` | 81 |  | |
| 71 | `appear/P/en` | 81 |  | |
| 72 | `process/C/en` | 79 |  | |
| 73 | `guide/P/en` | 78 |  | |
| 74 | `process/P/en` | 78 |  | |
| 75 | `adapt/P/en` | 77 |  | |
| 76 | `establish/P/en` | 77 |  | |
| 77 | `analysis/C/en` | 76 | borderline: action by the authors? | |
| 78 | `consist/P/en` | 76 |  | |
| 79 | `paper/C/en` | 76 |  | |
| 80 | `important/C/en` | 75 |  | |
| 81 | `section/C/en` | 72 |  | |
| 82 | `fact/C/en` | 72 |  | |
| 83 | `add/P/en` | 71 |  | |
| 84 | `confirm/P/en` | 70 |  | |
| 85 | `begin/P/en` | 70 |  | |
| 86 | `correspond/P/en` | 69 |  | |
| 87 | `underlie/P/en` | 69 |  | |
| 88 | `determine/P/en` | 68 |  | |
| 89 | `maintain/P/en` | 67 |  | |
| 90 | `be/M/en can/M/en use/P/en` | 67 |  | |
| 91 | `receive/P/en` | 66 |  | |
| 92 | `know/P/en` | 65 |  | |
| 93 | `facilitate/P/en` | 64 |  | |
| 94 | `do/P/en` | 64 |  | |
| 95 | `our/M/en result/C/en` | 64 |  | |
| 96 | `choose/P/en` | 62 |  | |
| 97 | `engage/P/en` | 62 |  | |
| 98 | `most/M/en` | 61 |  | |
| 99 | `emphasize/P/en` | 60 |  | |
| 100 | `be/M/en need/P/en` | 60 |  | |
| 101 | `be/P/en would/M/en` | 60 |  | |
| 102 | `percent/C/en` | 59 |  | |
| 103 | `regard/P/en` | 58 |  | |
| 104 | `write/P/en` | 58 |  | |
| 105 | `hand/C/en` | 58 |  | |
| 106 | `extent/C/en` | 58 |  | |
| 107 | `draw/P/en` | 57 |  | |
| 108 | `be/P/en can/M/en` | 56 |  | |
| 109 | `particular/C/en` | 55 |  | |
| 110 | `reach/P/en` | 54 |  | |
| 111 | `seem/P/en` | 53 |  | |
| 112 | `comprise/P/en` | 51 |  | |
| 113 | `challenge/C/en` | 50 |  | |
| 114 | `hold/P/en` | 50 |  | |

## Substantive (S) — 158 hyperedges

| # | Hyperedge (typed atoms) | Articles using it | Note | Owner's class |
|---|---|---|---|---|
| 1 | `llm/C/focal` | 704 | focal topic term | |
| 2 | `study/C/en` | 457 |  | |
| 3 | `provide/P/en` | 435 |  | |
| 4 | `model/C/en` | 390 |  | |
| 5 | `generate/P/en` | 325 |  | |
| 6 | `evaluate/P/en` | 295 | borderline: action by the authors? | |
| 7 | `achieve/P/en` | 290 |  | |
| 8 | `compare/P/en` | 289 | borderline: action by the authors? | |
| 9 | `chatgpt/C/focal` | 267 | focal topic term | |
| 10 | `train/P/en` | 234 |  | |
| 11 | `develop/P/en` | 219 |  | |
| 12 | `bert/C/focal` | 214 | focal topic term | |
| 13 | `apply/P/en` | 213 |  | |
| 14 | `combine/P/en` | 212 |  | |
| 15 | `improve/P/en` | 199 |  | |
| 16 | `language_model/C/focal` | 198 | focal topic term | |
| 17 | `see/P/en` | 190 |  | |
| 18 | `assess/P/en` | 188 | borderline: action by the authors? | |
| 19 | `utilize/P/en` | 183 |  | |
| 20 | `identify/P/en` | 178 |  | |
| 21 | `outperform/P/en` | 178 | borderline: action by the authors? | |
| 22 | `examine/P/en` | 175 | borderline: action by the authors? | |
| 23 | `focus/P/en` | 174 |  | |
| 24 | `enhance/P/en` | 170 |  | |
| 25 | `fine_tune/P/en` | 169 |  | |
| 26 | `propose/P/en` | 169 |  | |
| 27 | `be/M/en train/P/en` | 169 |  | |
| 28 | `require/P/en` | 168 |  | |
| 29 | `conduct/P/en` | 159 | borderline: action by the authors? | |
| 30 | `support/P/en` | 158 |  | |
| 31 | `participant/C/en` | 157 |  | |
| 32 | `prompt/C/en` | 153 |  | |
| 33 | `align/P/en` | 151 |  | |
| 34 | `gpt_4/C/focal` | 147 | focal topic term | |
| 35 | `user/C/en` | 146 |  | |
| 36 | `capture/P/en` | 146 |  | |
| 37 | `create/P/en` | 143 |  | |
| 38 | `test/P/en` | 141 | borderline: action by the authors? | |
| 39 | `enable/P/en` | 141 |  | |
| 40 | `ensure/P/en` | 138 |  | |
| 41 | `predict/P/en` | 134 |  | |
| 42 | `aim/P/en` | 133 |  | |
| 43 | `response/C/en` | 131 |  | |
| 44 | `text/C/en` | 131 |  | |
| 45 | `investigate/P/en` | 127 | borderline: action by the authors? | |
| 46 | `build/P/en` | 125 |  | |
| 47 | `rely/P/en` | 125 |  | |
| 48 | `increase/P/en` | 120 |  | |
| 49 | `observe/P/en` | 117 |  | |
| 50 | `extract/P/en` | 117 |  | |
| 51 | `our/M/en study/C/en` | 116 |  | |
| 52 | `system/C/en` | 116 |  | |
| 53 | `select/P/en` | 114 |  | |
| 54 | `analyze/P/en` | 109 | borderline: action by the authors? | |
| 55 | `potential/C/en` | 108 |  | |
| 56 | `researcher/C/en` | 108 |  | |
| 57 | `reduce/P/en` | 107 |  | |
| 58 | `bert/C/focal model/C/en` | 107 |  | |
| 59 | `understand/P/en` | 105 |  | |
| 60 | `design/P/en` | 102 | borderline: action by the authors? | |
| 61 | `human/C/en` | 100 |  | |
| 62 | `exist/P/en` | 99 |  | |
| 63 | `lack/P/en` | 98 |  | |
| 64 | `describe/P/en` | 96 |  | |
| 65 | `case/C/en` | 95 |  | |
| 66 | `limitation/C/en` | 94 |  | |
| 67 | `word/C/en` | 93 |  | |
| 68 | `task/C/en` | 92 |  | |
| 69 | `question/C/en` | 91 |  | |
| 70 | `associate/P/en` | 90 |  | |
| 71 | `method/C/en` | 89 | borderline: action by the authors? | |
| 72 | `be/M/en generate/P/en` | 89 |  | |
| 73 | `contribute/P/en` | 89 |  | |
| 74 | `learn/P/en` | 85 |  | |
| 75 | `recent/M/en study/C/en` | 85 |  | |
| 76 | `implement/P/en` | 84 | borderline: action by the authors? | |
| 77 | `allow/P/en` | 84 |  | |
| 78 | `be/M/en conduct/P/en` | 84 | borderline: action by the authors? | |
| 79 | `dataset/C/en` | 82 |  | |
| 80 | `ability/C/en` | 82 |  | |
| 81 | `extend/P/en` | 81 |  | |
| 82 | `gpt_4o/C/focal` | 81 | focal topic term | |
| 83 | `validate/P/en` | 80 | borderline: action by the authors? | |
| 84 | `derive/P/en` | 79 |  | |
| 85 | `be/M/en design/P/en` | 79 | borderline: action by the authors? | |
| 86 | `performance/C/en` | 75 |  | |
| 87 | `automate/P/en` | 74 |  | |
| 88 | `transformer_model/C/focal` | 73 | focal topic term | |
| 89 | `information/C/en` | 71 |  | |
| 90 | `chatbot/C/en` | 71 |  | |
| 91 | `power/P/en` | 71 |  | |
| 92 | `perceive/P/en` | 71 |  | |
| 93 | `author/C/en` | 70 |  | |
| 94 | `simulate/P/en` | 70 |  | |
| 95 | `ai/C/en` | 70 |  | |
| 96 | `present/M/en study/C/en` | 70 |  | |
| 97 | `apply/P/en be/M/en` | 68 |  | |
| 98 | `detect/P/en` | 68 |  | |
| 99 | `language/C/en` | 68 |  | |
| 100 | `ability/C/en llm/C/focal` | 68 |  | |
| 101 | `respond/P/en` | 66 |  | |
| 102 | `possible/C/en` | 66 |  | |
| 103 | `seek/P/en` | 66 |  | |
| 104 | `classify/P/en` | 66 |  | |
| 105 | `gpt_3_5/C/focal` | 65 | focal topic term | |
| 106 | `can/M/en generate/P/en` | 65 |  | |
| 107 | `be/M/en evaluate/P/en` | 65 | borderline: action by the authors? | |
| 108 | `prompt/P/en` | 63 |  | |
| 109 | `llm/C/focal output/C/en` | 62 |  | |
| 110 | `ask/P/en` | 62 |  | |
| 111 | `call/P/en` | 61 |  | |
| 112 | `roberta/C/focal` | 61 | focal topic term | |
| 113 | `deliver/P/en` | 61 |  | |
| 114 | `limit/P/en` | 60 |  | |
| 115 | `define/P/en` | 60 |  | |
| 116 | `calculate/P/en` | 60 | borderline: action by the authors? | |
| 117 | `feature/C/en` | 60 |  | |
| 118 | `grow/P/en` | 59 |  | |
| 119 | `evidence/C/en` | 59 |  | |
| 120 | `interact/P/en` | 59 |  | |
| 121 | `be/M/en select/P/en` | 58 |  | |
| 122 | `range/P/en` | 58 |  | |
| 123 | `can/M/en provide/P/en` | 58 |  | |
| 124 | `experiment/C/en` | 58 | borderline: action by the authors? | |
| 125 | `gpt/C/focal` | 58 | focal topic term | |
| 126 | `differ/P/en` | 57 |  | |
| 127 | `framework/C/en` | 57 |  | |
| 128 | `llm/C/focal performance/C/en` | 57 |  | |
| 129 | `order/C/en` | 57 |  | |
| 130 | `embed/P/en` | 57 |  | |
| 131 | `measure/P/en` | 56 | borderline: action by the authors? | |
| 132 | `be/M/en develop/P/en` | 56 |  | |
| 133 | `be/M/en provide/P/en` | 56 |  | |
| 134 | `output/C/en` | 55 |  | |
| 135 | `excel/P/en` | 55 |  | |
| 136 | `collect/P/en` | 55 | borderline: action by the authors? | |
| 137 | `input/C/en` | 55 |  | |
| 138 | `label/P/en` | 55 |  | |
| 139 | `vary/P/en` | 55 |  | |
| 140 | `rate/P/en` | 54 |  | |
| 141 | `explain/P/en` | 54 |  | |
| 142 | `be/M/en compare/P/en` | 53 | borderline: action by the authors? | |
| 143 | `summarize/P/en` | 53 |  | |
| 144 | `pattern/C/en` | 53 |  | |
| 145 | `people/C/en` | 53 |  | |
| 146 | `mean/P/en` | 52 |  | |
| 147 | `future/M/en study/C/en` | 52 |  | |
| 148 | `pretraine/P/en` | 52 |  | |
| 149 | `operate/P/en` | 51 |  | |
| 150 | `compute/P/en` | 51 | borderline: action by the authors? | |
| 151 | `interpret/P/en` | 50 |  | |
| 152 | `be/M/en fine_tune/P/en` | 50 |  | |
| 153 | `optimize/P/en` | 50 |  | |
| 154 | `patient/C/en` | 50 |  | |
| 155 | `score/P/en` | 50 |  | |
| 156 | `be/M/en observe/P/en` | 50 |  | |
| 157 | `gap/C/en` | 50 |  | |
| 158 | `expand/P/en` | 50 |  | |
