"""Build the G3 adverb role table (G3_POSTPROCESSING.md §6 item 4, owner decision 2026-10-08).

Input: diagnostics/g3_stoplist/adverb_attachments.json (every single adverb that the tested question-4 rule attaches to a
verb group in G2 v3 shards 0 and 20, --limit 400, with counts and one example). Output: adverb_roles.tsv.
Roles are assigned by the explicit sets below (the assistant's reading, to be reviewed by the owner); words not listed
default to 'manner' if they end in -ly, else 'other' (mostly nouns the parser typed as modifiers). Actions:
keep (attach to the verb group): degree, frequency, manner, focus, likelihood hedge
leave (neither attached nor dropped; the structure stays as the parse gives it): other, parse_leftover
drop: time, stance/attitude, certainty booster, other hedge, discourse, subordinator
A 'flag' marks rows where the role is uncertain or where dropping could change the meaning.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "diagnostics", "g3_stoplist", "adverb_attachments.json")

ROLES = {
    "degree": """significantly substantially fully largely slightly markedly partially greatly entirely highly mostly heavily
        completely thoroughly equally deeply partly dramatically nearly almost considerably exceptionally profoundly
        somewhat modestly marginally comparatively relatively roughly approximately drastically tremendously quite
        excessively sharply much far sufficiently insufficiently adequately strongly severely extensively twice enough
        majorly fairly exactly comparably increasingly progressively perfectly remarkably impressively notably double
        altogether""",
    "frequency": """often frequently sometimes occasionally commonly always usually rarely typically repeatedly continually
        continuously constantly regularly periodically annually daily consistently normally ever seldom""",
    "focus": """even especially particularly specifically mainly primarily exclusively predominantly solely merely simply
        purely alone chiefly only""",
    "hedge_likelihood": "likely unlikely probably possibly perhaps maybe potentially presumably necessarily",
    "hedge_other": "generally apparently arguably seemingly essentially basically sort kind",
    "manner_listed": "well offline online long fine fast verbatim outright worldwide elsewhere priori",
    "time": """recently already currently previously initially subsequently ultimately yet still newly soon later
        historically presently originally formerly thereafter afterward someday sometime temporarily simultaneously
        concurrently traditionally nowadays beforehand afterwards""",
    "stance": """interestingly importantly surprisingly unsurprisingly fortunately unfortunately crucially ideally
        inevitably predictably rightfully strikingly encouragingly curiously intriguingly hopefully regrettably
        understandably""",
    "booster": "clearly certainly undoubtedly definitively definitely obviously evidently truly actually really",
    "discourse": """similarly together rather first finally overall instead otherwise alternatively second next last fifth
        collectively herein hereafter whereby insofar above below course please third anyway therein wherein therefrom
        notwithstanding""",
    "subordinator": "whether unless",
    "particle": "under behind ahead back forward throughout before around away beside despite forwards",
    "parse_leftover": "can is r c ai_based cloud_based base",
}
ACTION = {"degree": "keep", "frequency": "keep", "manner": "keep", "focus": "keep", "hedge_likelihood": "keep",
          "manner_listed": "keep", "other": "leave", "time": "drop", "stance": "drop", "booster": "drop", "hedge_other": "drop",
          "discourse": "drop", "subordinator": "drop", "particle": "open question 5", "parse_leftover": "leave"}
FLAGS = {
    "yet": "in 'not yet validated' a drop leaves 'not validated': consider keeping",
    "still": "persistence ('LLMs still require'), often contrastive: time or keep?",
    "increasingly": "a trend over time, put under degree (kept); could be time",
    "necessarily": "'not necessarily X' would become 'not X' if dropped: kept as a likelihood hedge",
    "remarkably": "in-clause mostly degree ('remarkably high'); as an opener stance",
    "critically": "in-clause mostly manner ('critically appraise'), as an opener stance; default manner",
    "generally": "Hyland hedge; could be frequency (= usually)",
    "consistently": "frequency (= every time) or manner",
    "actually": "booster; sometimes contrastive ('actually decreased')",
    "rather": "mostly 'rather than'",
    "first": "in-clause sequence ('we first investigated'); 'the first model' is an adjective and not affected",
    "even": "focus ('even outperformed'); 'even though/if' already dropped as a subordinator",
    "ever": "frequency or time",
    "far": "degree ('far exceeded'); 'so far' is time",
    "potentially": "likelihood hedge",
    "ultimately": "time or stance (result)",
    "simply": "focus; 'not simply' is fused into not_only",
    "only": "own guarded rule (joins the verb group only when a verb follows)",
    "notably": "in-clause degree ('notably lower'); the opener 'Notably,' is dropped by the opener rule",
    "simultaneously": "time ('at the same time') or manner",
    "traditionally": "time or hedge",
    "course": "'of course' (booster/discourse)",
    "sort": "'sort of' (hedge)",
    "please": "politeness in quoted prompts",
}


def main():
    d = json.load(open(SRC))
    role_of = {w: r for r, ws in ROLES.items() for w in ws.split()}
    out = ["rank\tword\tattachments\trole\taction\tflag\texample"]
    for i, (w, n) in enumerate(d["counts"], 1):
        r = role_of.get(w) or ("manner" if w.endswith("ly") else "other")
        ex = d["examples"].get(w, "").replace("\t", " ").replace("\n", " ")
        out.append(f"{i}\t{w}\t{n}\t{r}\t{ACTION[r]}\t{FLAGS.get(w, '')}\t{ex}")
    path = os.path.join(HERE, "adverb_roles.tsv")
    open(path, "w").write("\n".join(out) + "\n")
    print(path, len(out) - 1, "words")


if __name__ == "__main__":
    main()
