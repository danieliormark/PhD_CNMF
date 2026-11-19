# Generic child hyperedges — classification for the owner's review (ticket 107, updated 2026-11-19)

Hyperedges used by at least 50 articles (trial build, hold-out group 1, visible ties), in two tables: **G** genre of scientific writing (candidates for exclusion) and **S** substantive. Decided so far (owner, 2026-11-19): methods and objects of model building / use are substantive (former class 'ambiguous'); focal topic terms are substantive. Rows marked *borderline* are genre words that may denote actions carried out by the authors — please decide them. To correct: write G or S in the last column, or name words to move between lists; the rule is then re-applied to all child hyperedges (word lists: `fullscale_pipeline/m1/fit/generic_hyperedges.py`).

| Class (all child hyperedges) | Hyperedges | Single-word | Share of ties | Share of raw usage | Share of weighted values |
|---|---|---|---|---|---|
| G | 2220 | 216 | 20.4% | 25.8% | 14.2% |
| S | 95630 | 6424 | 79.6% | 74.2% | 85.8% |

## Genre (G) — 213 hyperedges

| # | Hyperedge (typed atoms) | Articles using it | Note | Owner's class |
|---|---|---|---|---|
| 1 | `use/P/en` | 1035 |  | |
| 2 | `be/P/en` | 984 |  | |
| 3 | `show/P/en` | 483 |  | |
| 4 | `study/C/en` | 457 |  | |
| 5 | `provide/P/en` | 435 |  | |
| 6 | `demonstrate/P/en` | 418 |  | |
| 7 | `include/P/en` | 353 |  | |
| 8 | `have/P/en` | 327 |  | |
| 9 | `be/M/en use/P/en` | 315 |  | |
| 10 | `suggest/P/en` | 307 |  | |
| 11 | `evaluate/P/en` | 295 | borderline: action by the authors? | |S
| 12 | `achieve/P/en` | 290 |  | | S
| 13 | `compare/P/en` | 289 | borderline: action by the authors? | |S
| 14 | `indicate/P/en` | 282 |  | |
| 15 | `example/C/en` | 278 |  | |
| 16 | `make/P/en` | 273 |  | |
| 17 | `base/P/en` | 261 |  | |
| 18 | `result/C/en` | 247 |  | |
| 19 | `find/P/en` | 240 |  | |
| 20 | `remain/P/en` | 231 |  | |
| 21 | `offer/P/en` | 226 |  | |
| 22 | `leverage/P/en` | 222 |  | |
| 23 | `develop/P/en` | 219 |  | |S
| 24 | `apply/P/en` | 213 |  | | S
| 25 | `combine/P/en` | 212 |  | |S
| 26 | `employ/P/en` | 207 |  | |
| 27 | `explore/P/en` | 204 |  | |
| 28 | `report/P/en` | 200 |  | |
| 29 | `improve/P/en` | 199 |  | |S
| 30 | `follow/P/en` | 197 |  | |
| 31 | `highlight/P/en` | 197 |  | |
| 32 | `present/P/en` | 197 |  | |
| 33 | `represent/P/en` | 195 |  | |
| 34 | `integrate/P/en` | 195 |  | |
| 35 | `introduce/P/en` | 191 |  | |
| 36 | `see/P/en` | 190 |  | |S
| 37 | `assess/P/en` | 188 | borderline: action by the authors? | |S
| 38 | `utilize/P/en` | 183 |  | |S
| 39 | `perform/P/en` | 178 |  | |
| 40 | `identify/P/en` | 178 |  | |S
| 41 | `outperform/P/en` | 178 | borderline: action by the authors? | |S
| 42 | `finding/C/en` | 176 |  | |
| 43 | `examine/P/en` | 175 | borderline: action by the authors? | |S
| 44 | `be/P/en not/M/en` | 174 |  | |
| 45 | `focus/P/en` | 174 |  | | S
| 46 | `enhance/P/en` | 170 |  | |S
| 47 | `propose/P/en` | 169 |  | |S
| 48 | `require/P/en` | 168 |  | |S
| 49 | `produce/P/en` | 163 |  | |
| 50 | `more/M/en` | 159 |  | |
| 51 | `conduct/P/en` | 159 | borderline: action by the authors? | |S
| 52 | `support/P/en` | 158 |  | |S
| 53 | `address/P/en` | 157 |  | |
| 54 | `align/P/en` | 151 |  | |S
| 55 | `reveal/P/en` | 150 |  | |
| 56 | `involve/P/en` | 147 |  | |
| 57 | `capture/P/en` | 146 |  | |S
| 58 | `create/P/en` | 143 |  | |S
| 59 | `test/P/en` | 141 | borderline: action by the authors? | |S
| 60 | `enable/P/en` | 141 |  | |S
| 61 | `ensure/P/en` | 138 |  | |S
| 62 | `incorporate/P/en` | 136 |  | |
| 63 | `instance/C/en` | 134 |  | |
| 64 | `exhibit/P/en` | 133 |  | |
| 65 | `aim/P/en` | 133 |  | |S
| 66 | `approach/C/en` | 133 |  | |
| 67 | `addition/C/en` | 132 |  | |
| 68 | `consider/P/en` | 131 |  | |
| 69 | `become/P/en` | 128 |  | |
| 70 | `investigate/P/en` | 127 | borderline: action by the authors? | |S
| 71 | `build/P/en` | 125 |  | |S
| 72 | `rely/P/en` | 125 |  | |S
| 73 | `research/C/en` | 124 |  | |
| 74 | `emerge/P/en` | 123 |  | |
| 75 | `time/C/en` | 123 |  | |
| 76 | `increase/P/en` | 120 |  | | S
| 77 | `reflect/P/en` | 120 |  | |
| 78 | `obtain/P/en` | 119 |  | |
| 79 | `observe/P/en` | 117 |  | |
| 80 | `our/M/en study/C/en` | 116 |  | |
| 81 | `well/M/en` | 116 |  | |
| 82 | `select/P/en` | 114 |  | |
| 83 | `take/P/en` | 111 |  | |
| 84 | `analyze/P/en` | 109 | borderline: action by the authors? | |S
| 85 | `potential/C/en` | 108 |  | |S
| 86 | `researcher/C/en` | 108 |  | |S
| 87 | `relate/P/en` | 107 |  | |
| 88 | `reduce/P/en` | 107 |  | |S
| 89 | `result/P/en` | 107 |  | |
| 90 | `understand/P/en` | 105 |  | |S
| 91 | `be/P/en may/M/en` | 103 |  | |
| 92 | `work/C/en` | 103 |  | |
| 93 | `lead/P/en` | 102 |  | |
| 94 | `design/P/en` | 102 | borderline: action by the authors? | |S
| 95 | `note/P/en` | 100 |  | |
| 96 | `exist/P/en` | 99 |  | |S
| 97 | `datum/C/en` | 98 |  | |
| 98 | `lack/P/en` | 98 |  | |S
| 99 | `context/C/en` | 98 |  | |
| 100 | `base/P/en be/M/en` | 97 |  | |
| 101 | `describe/P/en` | 96 |  | |S
| 102 | `give/P/en` | 96 |  | |
| 103 | `case/C/en` | 95 |  | |S
| 104 | `limitation/C/en` | 94 |  | |S
| 105 | `contain/P/en` | 93 |  | |
| 106 | `yield/P/en` | 93 |  | |
| 107 | `adopt/P/en` | 91 |  | |
| 108 | `question/C/en` | 91 |  | |S
| 109 | `associate/P/en` | 90 |  | |S
| 110 | `method/C/en` | 89 | borderline: action by the authors? | |S
| 111 | `contribute/P/en` | 89 |  | |S
| 112 | `underscore/P/en` | 88 |  | |
| 113 | `way/C/en` | 88 |  | |
| 114 | `serve/P/en` | 88 |  | |
| 115 | `illustrate/P/en` | 87 |  | |
| 116 | `be/M/en show/P/en` | 86 |  | |
| 117 | `refer/P/en` | 85 |  | |
| 118 | `recent/M/en study/C/en` | 85 |  | |
| 119 | `implement/P/en` | 84 | borderline: action by the authors? | |S
| 120 | `allow/P/en` | 84 |  | |S
| 121 | `finding/C/en our/M/en` | 84 |  | |
| 122 | `be/M/en conduct/P/en` | 84 | borderline: action by the authors? | |S
| 123 | `raise/P/en` | 81 |  | |
| 124 | `extend/P/en` | 81 |  | |S
| 125 | `future/M/en research/C/en` | 81 |  | |
| 126 | `appear/P/en` | 81 |  | |
| 127 | `validate/P/en` | 80 | borderline: action by the authors? | |S
| 128 | `derive/P/en` | 79 |  | |S
| 129 | `be/M/en design/P/en` | 79 | borderline: action by the authors? | |S
| 130 | `process/C/en` | 79 |  | |
| 131 | `guide/P/en` | 78 |  | |
| 132 | `process/P/en` | 78 |  | |
| 133 | `adapt/P/en` | 77 |  | |
| 134 | `establish/P/en` | 77 |  | |
| 135 | `analysis/C/en` | 76 | borderline: action by the authors? | |
| 136 | `consist/P/en` | 76 |  | |
| 137 | `paper/C/en` | 76 |  | |
| 138 | `important/C/en` | 75 |  | |
| 139 | `section/C/en` | 72 |  | |
| 140 | `fact/C/en` | 72 |  | |
| 141 | `information/C/en` | 71 |  | |S
| 142 | `power/P/en` | 71 |  | | S
| 143 | `add/P/en` | 71 |  | |
| 144 | `author/C/en` | 70 |  | |S
| 145 | `confirm/P/en` | 70 |  | |
| 146 | `begin/P/en` | 70 |  | |
| 147 | `present/M/en study/C/en` | 70 |  | |
| 148 | `correspond/P/en` | 69 |  | |
| 149 | `underlie/P/en` | 69 |  | |
| 150 | `apply/P/en be/M/en` | 68 |  | |
| 151 | `determine/P/en` | 68 |  | |
| 152 | `maintain/P/en` | 67 |  | |
| 153 | `be/M/en can/M/en use/P/en` | 67 |  | |
| 154 | `possible/C/en` | 66 |  | |S
| 155 | `receive/P/en` | 66 |  | |
| 156 | `seek/P/en` | 66 |  | |S
| 157 | `be/M/en evaluate/P/en` | 65 | borderline: action by the authors? | |S
| 158 | `know/P/en` | 65 |  | |
| 159 | `facilitate/P/en` | 64 |  | |
| 160 | `do/P/en` | 64 |  | |
| 161 | `our/M/en result/C/en` | 64 |  | |
| 162 | `choose/P/en` | 62 |  | |
| 163 | `engage/P/en` | 62 |  | |
| 164 | `ask/P/en` | 62 |  | |S
| 165 | `call/P/en` | 61 |  | |S
| 166 | `deliver/P/en` | 61 |  | |S
| 167 | `most/M/en` | 61 |  | |
| 168 | `emphasize/P/en` | 60 |  | |
| 169 | `be/M/en need/P/en` | 60 |  | |
| 170 | `limit/P/en` | 60 |  | |S
| 171 | `define/P/en` | 60 |  | |S
| 172 | `calculate/P/en` | 60 | borderline: action by the authors? | |S
| 173 | `be/P/en would/M/en` | 60 |  | |
| 174 | `grow/P/en` | 59 |  | |S
| 175 | `percent/C/en` | 59 |  | |
| 176 | `evidence/C/en` | 59 |  | |S
| 177 | `regard/P/en` | 58 |  | |
| 178 | `be/M/en select/P/en` | 58 |  | |S
| 179 | `range/P/en` | 58 |  | |S
| 180 | `write/P/en` | 58 |  | |
| 181 | `can/M/en provide/P/en` | 58 |  | |S
| 182 | `experiment/C/en` | 58 | borderline: action by the authors? | |S
| 183 | `hand/C/en` | 58 |  | |
| 184 | `extent/C/en` | 58 |  | |
| 185 | `differ/P/en` | 57 |  | |S
| 186 | `framework/C/en` | 57 |  | |S
| 187 | `order/C/en` | 57 |  | |S
| 188 | `draw/P/en` | 57 |  | |
| 189 | `be/P/en can/M/en` | 56 |  | |
| 190 | `measure/P/en` | 56 | borderline: action by the authors? | |S
| 191 | `be/M/en develop/P/en` | 56 |  | |S
| 192 | `be/M/en provide/P/en` | 56 |  | |S
| 193 | `particular/C/en` | 55 |  | |
| 194 | `excel/P/en` | 55 |  | |S
| 195 | `collect/P/en` | 55 | borderline: action by the authors? | |S
| 196 | `vary/P/en` | 55 |  | |S
| 197 | `reach/P/en` | 54 |  | |
| 198 | `explain/P/en` | 54 |  | |S
| 199 | `be/M/en compare/P/en` | 53 | borderline: action by the authors? | |S
| 200 | `summarize/P/en` | 53 |  | |S
| 201 | `seem/P/en` | 53 |  | |
| 202 | `pattern/C/en` | 53 |  | |S
| 203 | `mean/P/en` | 52 |  | |S
| 204 | `future/M/en study/C/en` | 52 |  | |S
| 205 | `operate/P/en` | 51 |  | |S
| 206 | `comprise/P/en` | 51 |  | |
| 207 | `compute/P/en` | 51 | borderline: action by the authors? | |S
| 208 | `interpret/P/en` | 50 |  | |S
| 209 | `challenge/C/en` | 50 |  | |
| 210 | `be/M/en observe/P/en` | 50 |  | |S
| 211 | `hold/P/en` | 50 |  | |
| 212 | `gap/C/en` | 50 |  | |S
| 213 | `expand/P/en` | 50 |  | |S

## Substantive (S) — 59 hyperedges

| # | Hyperedge (typed atoms) | Articles using it | Note | Owner's class |
|---|---|---|---|---|
| 1 | `llm/C/focal` | 704 | focal topic term | |
| 2 | `model/C/en` | 390 |  | |
| 3 | `generate/P/en` | 325 |  | |
| 4 | `chatgpt/C/focal` | 267 | focal topic term | |
| 5 | `train/P/en` | 234 |  | |
| 6 | `bert/C/focal` | 214 | focal topic term | |
| 7 | `language_model/C/focal` | 198 | focal topic term | |
| 8 | `fine_tune/P/en` | 169 |  | |
| 9 | `be/M/en train/P/en` | 169 |  | |
| 10 | `participant/C/en` | 157 |  | |
| 11 | `prompt/C/en` | 153 |  | |
| 12 | `gpt_4/C/focal` | 147 | focal topic term | |
| 13 | `user/C/en` | 146 |  | |
| 14 | `predict/P/en` | 134 |  | |
| 15 | `response/C/en` | 131 |  | |
| 16 | `text/C/en` | 131 |  | |
| 17 | `extract/P/en` | 117 |  | |
| 18 | `system/C/en` | 116 |  | |
| 19 | `bert/C/focal model/C/en` | 107 |  | |
| 20 | `human/C/en` | 100 |  | |
| 21 | `word/C/en` | 93 |  | |
| 22 | `task/C/en` | 92 |  | |
| 23 | `be/M/en generate/P/en` | 89 |  | |
| 24 | `learn/P/en` | 85 |  | |
| 25 | `dataset/C/en` | 82 |  | |
| 26 | `ability/C/en` | 82 |  | |
| 27 | `gpt_4o/C/focal` | 81 | focal topic term | |
| 28 | `performance/C/en` | 75 |  | |
| 29 | `automate/P/en` | 74 |  | |
| 30 | `transformer_model/C/focal` | 73 | focal topic term | |
| 31 | `chatbot/C/en` | 71 |  | |
| 32 | `perceive/P/en` | 71 |  | |
| 33 | `simulate/P/en` | 70 |  | |
| 34 | `ai/C/en` | 70 |  | |
| 35 | `detect/P/en` | 68 |  | |
| 36 | `language/C/en` | 68 |  | |
| 37 | `ability/C/en llm/C/focal` | 68 |  | |
| 38 | `respond/P/en` | 66 |  | |
| 39 | `classify/P/en` | 66 |  | |
| 40 | `gpt_3_5/C/focal` | 65 | focal topic term | |
| 41 | `can/M/en generate/P/en` | 65 |  | |
| 42 | `prompt/P/en` | 63 |  | |
| 43 | `llm/C/focal output/C/en` | 62 |  | |
| 44 | `roberta/C/focal` | 61 | focal topic term | |
| 45 | `feature/C/en` | 60 |  | |
| 46 | `interact/P/en` | 59 |  | |
| 47 | `gpt/C/focal` | 58 | focal topic term | |
| 48 | `llm/C/focal performance/C/en` | 57 |  | |
| 49 | `embed/P/en` | 57 |  | |
| 50 | `output/C/en` | 55 |  | |
| 51 | `input/C/en` | 55 |  | |
| 52 | `label/P/en` | 55 |  | |
| 53 | `rate/P/en` | 54 |  | |
| 54 | `people/C/en` | 53 |  | |
| 55 | `pretraine/P/en` | 52 |  | |
| 56 | `be/M/en fine_tune/P/en` | 50 |  | |
| 57 | `optimize/P/en` | 50 |  | |
| 58 | `patient/C/en` | 50 |  | |
| 59 | `score/P/en` | 50 |  | |
