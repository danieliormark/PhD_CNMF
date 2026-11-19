# Generic child hyperedges — draft classification for the owner's review (ticket 107, 2026-11-19)

Hyperedges used by at least 50 articles (trial build, hold-out group 1, visible ties). Draft class: **G** genre of scientific writing, **F** focal topic terms only, **?** ambiguous (generic in form, possibly a method or object of this corpus), **S** substantive. Rule and word lists: `fullscale_pipeline/m1/fit/generic_hyperedges.py`. To correct: change the class in the last column (or name words to move between lists); the rule is then re-run on all hyperedges.

| Class | Hyperedges | Single-word | Share of ties | Share of raw usage | Share of weighted values |
|---|---|---|---|---|---|
| G | 2220 | 216 | 20.4% | 25.8% | 14.2% |
| F | 404 | 176 | 1.7% | 5.7% | 1.8% |
| ? | 0 | 0 | 0.0% | 0.0% | 0.0% |
| S | 95226 | 6248 | 77.9% | 68.4% | 84.0% |

| # | Hyperedge (typed atoms) | Articles using it | Draft class | Owner's class |
|---|---|---|---|---|
| 1 | `use/P/en` | 1035 | G | |
| 2 | `be/P/en` | 984 | G | |
| 3 | `llm/C/focal` | 704 | F | |
| 4 | `show/P/en` | 483 | G | |
| 5 | `study/C/en` | 457 | G | |
| 6 | `provide/P/en` | 435 | G | |
| 7 | `demonstrate/P/en` | 418 | G | |
| 8 | `model/C/en` | 390 | S | |
| 9 | `include/P/en` | 353 | G | |
| 10 | `have/P/en` | 327 | G | |
| 11 | `generate/P/en` | 325 | S | |
| 12 | `be/M/en use/P/en` | 315 | G | |
| 13 | `suggest/P/en` | 307 | G | |
| 14 | `evaluate/P/en` | 295 | G | |
| 15 | `achieve/P/en` | 290 | G | |
| 16 | `compare/P/en` | 289 | G | |
| 17 | `indicate/P/en` | 282 | G | |
| 18 | `example/C/en` | 278 | G | |
| 19 | `make/P/en` | 273 | G | |
| 20 | `chatgpt/C/focal` | 267 | F | |
| 21 | `base/P/en` | 261 | G | |
| 22 | `result/C/en` | 247 | G | |
| 23 | `find/P/en` | 240 | G | |
| 24 | `train/P/en` | 234 | S | |
| 25 | `remain/P/en` | 231 | G | |
| 26 | `offer/P/en` | 226 | G | |
| 27 | `leverage/P/en` | 222 | G | |
| 28 | `develop/P/en` | 219 | G | |
| 29 | `bert/C/focal` | 214 | F | |
| 30 | `apply/P/en` | 213 | G | |
| 31 | `combine/P/en` | 212 | G | |
| 32 | `employ/P/en` | 207 | G | |
| 33 | `explore/P/en` | 204 | G | |
| 34 | `report/P/en` | 200 | G | |
| 35 | `improve/P/en` | 199 | G | |
| 36 | `language_model/C/focal` | 198 | F | |
| 37 | `follow/P/en` | 197 | G | |
| 38 | `highlight/P/en` | 197 | G | |
| 39 | `present/P/en` | 197 | G | |
| 40 | `represent/P/en` | 195 | G | |
| 41 | `integrate/P/en` | 195 | G | |
| 42 | `introduce/P/en` | 191 | G | |
| 43 | `see/P/en` | 190 | G | |
| 44 | `assess/P/en` | 188 | G | |
| 45 | `utilize/P/en` | 183 | G | |
| 46 | `perform/P/en` | 178 | G | |
| 47 | `identify/P/en` | 178 | G | |
| 48 | `outperform/P/en` | 178 | G | |
| 49 | `finding/C/en` | 176 | G | |
| 50 | `examine/P/en` | 175 | G | |
| 51 | `be/P/en not/M/en` | 174 | G | |
| 52 | `focus/P/en` | 174 | G | |
| 53 | `enhance/P/en` | 170 | G | |
| 54 | `fine_tune/P/en` | 169 | S | |
| 55 | `propose/P/en` | 169 | G | |
| 56 | `be/M/en train/P/en` | 169 | S | |
| 57 | `require/P/en` | 168 | G | |
| 58 | `produce/P/en` | 163 | G | |
| 59 | `more/M/en` | 159 | G | |
| 60 | `conduct/P/en` | 159 | G | |
| 61 | `support/P/en` | 158 | G | |
| 62 | `participant/C/en` | 157 | S | |
| 63 | `address/P/en` | 157 | G | |
| 64 | `prompt/C/en` | 153 | S | |
| 65 | `align/P/en` | 151 | G | |
| 66 | `reveal/P/en` | 150 | G | |
| 67 | `involve/P/en` | 147 | G | |
| 68 | `gpt_4/C/focal` | 147 | F | |
| 69 | `user/C/en` | 146 | S | |
| 70 | `capture/P/en` | 146 | G | |
| 71 | `create/P/en` | 143 | G | |
| 72 | `test/P/en` | 141 | G | |
| 73 | `enable/P/en` | 141 | G | |
| 74 | `ensure/P/en` | 138 | G | |
| 75 | `incorporate/P/en` | 136 | G | |
| 76 | `predict/P/en` | 134 | S | |
| 77 | `instance/C/en` | 134 | G | |
| 78 | `exhibit/P/en` | 133 | G | |
| 79 | `aim/P/en` | 133 | G | |
| 80 | `approach/C/en` | 133 | G | |
| 81 | `addition/C/en` | 132 | G | |
| 82 | `consider/P/en` | 131 | G | |
| 83 | `response/C/en` | 131 | S | |
| 84 | `text/C/en` | 131 | S | |
| 85 | `become/P/en` | 128 | G | |
| 86 | `investigate/P/en` | 127 | G | |
| 87 | `build/P/en` | 125 | G | |
| 88 | `rely/P/en` | 125 | G | |
| 89 | `research/C/en` | 124 | G | |
| 90 | `emerge/P/en` | 123 | G | |
| 91 | `time/C/en` | 123 | G | |
| 92 | `increase/P/en` | 120 | G | |
| 93 | `reflect/P/en` | 120 | G | |
| 94 | `obtain/P/en` | 119 | G | |
| 95 | `observe/P/en` | 117 | G | |
| 96 | `extract/P/en` | 117 | S | |
| 97 | `our/M/en study/C/en` | 116 | G | |
| 98 | `well/M/en` | 116 | G | |
| 99 | `system/C/en` | 116 | S | |
| 100 | `select/P/en` | 114 | G | |
| 101 | `take/P/en` | 111 | G | |
| 102 | `analyze/P/en` | 109 | G | |
| 103 | `potential/C/en` | 108 | G | |
| 104 | `researcher/C/en` | 108 | G | |
| 105 | `relate/P/en` | 107 | G | |
| 106 | `reduce/P/en` | 107 | G | |
| 107 | `result/P/en` | 107 | G | |
| 108 | `bert/C/focal model/C/en` | 107 | S | |
| 109 | `understand/P/en` | 105 | G | |
| 110 | `be/P/en may/M/en` | 103 | G | |
| 111 | `work/C/en` | 103 | G | |
| 112 | `lead/P/en` | 102 | G | |
| 113 | `design/P/en` | 102 | G | |
| 114 | `human/C/en` | 100 | S | |
| 115 | `note/P/en` | 100 | G | |
| 116 | `exist/P/en` | 99 | G | |
| 117 | `datum/C/en` | 98 | G | |
| 118 | `lack/P/en` | 98 | G | |
| 119 | `context/C/en` | 98 | G | |
| 120 | `base/P/en be/M/en` | 97 | G | |
| 121 | `describe/P/en` | 96 | G | |
| 122 | `give/P/en` | 96 | G | |
| 123 | `case/C/en` | 95 | G | |
| 124 | `limitation/C/en` | 94 | G | |
| 125 | `contain/P/en` | 93 | G | |
| 126 | `word/C/en` | 93 | S | |
| 127 | `yield/P/en` | 93 | G | |
| 128 | `task/C/en` | 92 | S | |
| 129 | `adopt/P/en` | 91 | G | |
| 130 | `question/C/en` | 91 | G | |
| 131 | `associate/P/en` | 90 | G | |
| 132 | `method/C/en` | 89 | G | |
| 133 | `be/M/en generate/P/en` | 89 | S | |
| 134 | `contribute/P/en` | 89 | G | |
| 135 | `underscore/P/en` | 88 | G | |
| 136 | `way/C/en` | 88 | G | |
| 137 | `serve/P/en` | 88 | G | |
| 138 | `illustrate/P/en` | 87 | G | |
| 139 | `be/M/en show/P/en` | 86 | G | |
| 140 | `learn/P/en` | 85 | S | |
| 141 | `refer/P/en` | 85 | G | |
| 142 | `recent/M/en study/C/en` | 85 | G | |
| 143 | `implement/P/en` | 84 | G | |
| 144 | `allow/P/en` | 84 | G | |
| 145 | `finding/C/en our/M/en` | 84 | G | |
| 146 | `be/M/en conduct/P/en` | 84 | G | |
| 147 | `dataset/C/en` | 82 | S | |
| 148 | `ability/C/en` | 82 | S | |
| 149 | `raise/P/en` | 81 | G | |
| 150 | `extend/P/en` | 81 | G | |
| 151 | `gpt_4o/C/focal` | 81 | F | |
| 152 | `future/M/en research/C/en` | 81 | G | |
| 153 | `appear/P/en` | 81 | G | |
| 154 | `validate/P/en` | 80 | G | |
| 155 | `derive/P/en` | 79 | G | |
| 156 | `be/M/en design/P/en` | 79 | G | |
| 157 | `process/C/en` | 79 | G | |
| 158 | `guide/P/en` | 78 | G | |
| 159 | `process/P/en` | 78 | G | |
| 160 | `adapt/P/en` | 77 | G | |
| 161 | `establish/P/en` | 77 | G | |
| 162 | `analysis/C/en` | 76 | G | |
| 163 | `consist/P/en` | 76 | G | |
| 164 | `paper/C/en` | 76 | G | |
| 165 | `performance/C/en` | 75 | S | |
| 166 | `important/C/en` | 75 | G | |
| 167 | `automate/P/en` | 74 | S | |
| 168 | `transformer_model/C/focal` | 73 | F | |
| 169 | `section/C/en` | 72 | G | |
| 170 | `fact/C/en` | 72 | G | |
| 171 | `information/C/en` | 71 | G | |
| 172 | `chatbot/C/en` | 71 | S | |
| 173 | `power/P/en` | 71 | G | |
| 174 | `perceive/P/en` | 71 | S | |
| 175 | `add/P/en` | 71 | G | |
| 176 | `author/C/en` | 70 | G | |
| 177 | `confirm/P/en` | 70 | G | |
| 178 | `simulate/P/en` | 70 | S | |
| 179 | `ai/C/en` | 70 | S | |
| 180 | `begin/P/en` | 70 | G | |
| 181 | `present/M/en study/C/en` | 70 | G | |
| 182 | `correspond/P/en` | 69 | G | |
| 183 | `underlie/P/en` | 69 | G | |
| 184 | `apply/P/en be/M/en` | 68 | G | |
| 185 | `determine/P/en` | 68 | G | |
| 186 | `detect/P/en` | 68 | S | |
| 187 | `language/C/en` | 68 | S | |
| 188 | `ability/C/en llm/C/focal` | 68 | S | |
| 189 | `maintain/P/en` | 67 | G | |
| 190 | `be/M/en can/M/en use/P/en` | 67 | G | |
| 191 | `respond/P/en` | 66 | S | |
| 192 | `possible/C/en` | 66 | G | |
| 193 | `receive/P/en` | 66 | G | |
| 194 | `seek/P/en` | 66 | G | |
| 195 | `classify/P/en` | 66 | S | |
| 196 | `gpt_3_5/C/focal` | 65 | F | |
| 197 | `can/M/en generate/P/en` | 65 | S | |
| 198 | `be/M/en evaluate/P/en` | 65 | G | |
| 199 | `know/P/en` | 65 | G | |
| 200 | `facilitate/P/en` | 64 | G | |
| 201 | `do/P/en` | 64 | G | |
| 202 | `our/M/en result/C/en` | 64 | G | |
| 203 | `prompt/P/en` | 63 | S | |
| 204 | `choose/P/en` | 62 | G | |
| 205 | `engage/P/en` | 62 | G | |
| 206 | `llm/C/focal output/C/en` | 62 | S | |
| 207 | `ask/P/en` | 62 | G | |
| 208 | `call/P/en` | 61 | G | |
| 209 | `roberta/C/focal` | 61 | F | |
| 210 | `deliver/P/en` | 61 | G | |
| 211 | `most/M/en` | 61 | G | |
| 212 | `emphasize/P/en` | 60 | G | |
| 213 | `be/M/en need/P/en` | 60 | G | |
| 214 | `limit/P/en` | 60 | G | |
| 215 | `define/P/en` | 60 | G | |
| 216 | `calculate/P/en` | 60 | G | |
| 217 | `feature/C/en` | 60 | S | |
| 218 | `be/P/en would/M/en` | 60 | G | |
| 219 | `grow/P/en` | 59 | G | |
| 220 | `percent/C/en` | 59 | G | |
| 221 | `evidence/C/en` | 59 | G | |
| 222 | `interact/P/en` | 59 | S | |
| 223 | `regard/P/en` | 58 | G | |
| 224 | `be/M/en select/P/en` | 58 | G | |
| 225 | `range/P/en` | 58 | G | |
| 226 | `write/P/en` | 58 | G | |
| 227 | `can/M/en provide/P/en` | 58 | G | |
| 228 | `experiment/C/en` | 58 | G | |
| 229 | `hand/C/en` | 58 | G | |
| 230 | `extent/C/en` | 58 | G | |
| 231 | `gpt/C/focal` | 58 | F | |
| 232 | `differ/P/en` | 57 | G | |
| 233 | `framework/C/en` | 57 | G | |
| 234 | `llm/C/focal performance/C/en` | 57 | S | |
| 235 | `order/C/en` | 57 | G | |
| 236 | `draw/P/en` | 57 | G | |
| 237 | `embed/P/en` | 57 | S | |
| 238 | `be/P/en can/M/en` | 56 | G | |
| 239 | `measure/P/en` | 56 | G | |
| 240 | `be/M/en develop/P/en` | 56 | G | |
| 241 | `be/M/en provide/P/en` | 56 | G | |
| 242 | `output/C/en` | 55 | S | |
| 243 | `particular/C/en` | 55 | G | |
| 244 | `excel/P/en` | 55 | G | |
| 245 | `collect/P/en` | 55 | G | |
| 246 | `input/C/en` | 55 | S | |
| 247 | `label/P/en` | 55 | S | |
| 248 | `vary/P/en` | 55 | G | |
| 249 | `rate/P/en` | 54 | S | |
| 250 | `reach/P/en` | 54 | G | |
| 251 | `explain/P/en` | 54 | G | |
| 252 | `be/M/en compare/P/en` | 53 | G | |
| 253 | `summarize/P/en` | 53 | G | |
| 254 | `seem/P/en` | 53 | G | |
| 255 | `pattern/C/en` | 53 | G | |
| 256 | `people/C/en` | 53 | S | |
| 257 | `mean/P/en` | 52 | G | |
| 258 | `future/M/en study/C/en` | 52 | G | |
| 259 | `pretraine/P/en` | 52 | S | |
| 260 | `operate/P/en` | 51 | G | |
| 261 | `comprise/P/en` | 51 | G | |
| 262 | `compute/P/en` | 51 | G | |
| 263 | `interpret/P/en` | 50 | G | |
| 264 | `challenge/C/en` | 50 | G | |
| 265 | `be/M/en fine_tune/P/en` | 50 | S | |
| 266 | `optimize/P/en` | 50 | S | |
| 267 | `patient/C/en` | 50 | S | |
| 268 | `score/P/en` | 50 | S | |
| 269 | `be/M/en observe/P/en` | 50 | G | |
| 270 | `hold/P/en` | 50 | G | |
| 271 | `gap/C/en` | 50 | G | |
| 272 | `expand/P/en` | 50 | G | |
