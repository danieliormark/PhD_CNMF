"""Build the G3 adverb role table (G3_POSTPROCESSING.md §6 item 4, owner decisions 2026-10-08).

Input: diagnostics/g3_stoplist/adverb_attachments.json (every single adverb that the tested question-4 rule attaches to a
verb group in G2 v3 shards 0 and 20, --limit 400, with counts and one example). Output: adverb_roles.tsv.
Roles are assigned by the explicit sets below; words not listed default to 'manner' if they end in -ly, else 'other'
(mostly nouns the parser typed as modifiers). Negative and low-frequency forms that the sample did not contain are added
from a count over the text of G2 v3 shards 0-9 (NEGATIVE_FORMS), so that they are not left to the default.

Owner's logic (review of 2026-10-08, commit 03d29d8): keep adverbs that turn or qualify the claim towards negation,
contrast or comparison (instead, similarly to, above average, rarely, incorrectly, inconsistently); drop generic
generalisers and focusers that do not change the claim (often, typically, consistently, even, mainly, closely,
automatically); keep in-clause 'first' (no conflict with spelled-out numbers being kept).

Actions: keep = attach to the verb group; drop; leave = neither attached nor dropped; conditional actions are spelled
out in the action column. The 12 'particle' rows (first left as 'open question 5') carry the actions decided in
G3_POSTPROCESSING.md §6 item 8 (owner 2026-10-09, option 3).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "diagnostics", "g3_stoplist", "adverb_attachments.json")

ROLES = {
    "degree": """significantly substantially fully slightly markedly partially greatly entirely highly heavily
        completely thoroughly equally deeply partly dramatically nearly almost considerably exceptionally profoundly
        somewhat modestly marginally comparatively relatively roughly approximately drastically tremendously quite
        excessively sharply much far sufficiently insufficiently adequately strongly severely extensively enough
        majorly fairly exactly comparably increasingly progressively perfectly remarkably impressively notably double
        altogether minimally negligibly disproportionately""",
    "frequency_low": "sometimes occasionally rarely seldom infrequently twice ever",
    "frequency_high": """often frequently commonly always usually typically repeatedly continually continuously constantly
        regularly normally consistently""",
    "schedule": "periodically annually daily",
    "focus_scalar": "even especially particularly specifically mainly primarily predominantly chiefly largely mostly",
    "focus_exclusive": "only solely exclusively merely purely alone simply",
    "hedge_likelihood": "likely unlikely probably possibly perhaps maybe potentially presumably necessarily",
    "hedge_other": "generally apparently arguably seemingly essentially basically sort kind",
    "manner_listed": "well offline online long fine fast verbatim outright worldwide elsewhere priori",
    "manner_generic": "closely automatically",
    "negative_form": """incorrectly inaccurately inconsistently inappropriately inadequately insignificantly nonsignificantly
        non_significantly unreliably unsuccessfully improperly incompletely unevenly unequally irregularly unpredictably
        uncritically indirectly differently hardly barely scarcely unfavorably unfavourably poorly wrongly erroneously
        mistakenly""",
    "sequence": "first",
    "contrast": "instead rather otherwise",
    "concessive": "yet",
    "comparison_conditional": "similarly",
    "deictic_conditional": "above below",
    "time": """recently already currently previously initially subsequently ultimately still newly soon later
        historically presently originally formerly thereafter afterward someday sometime temporarily simultaneously
        concurrently traditionally nowadays beforehand afterwards""",
    "stance": """interestingly importantly surprisingly unsurprisingly unexpectedly fortunately unfortunately crucially
        ideally inevitably predictably rightfully strikingly encouragingly curiously intriguingly hopefully regrettably
        understandably""",
    "booster": "clearly certainly undoubtedly definitively definitely obviously evidently truly actually really",
    "discourse": """together finally overall alternatively second next last fifth collectively herein hereafter
        whereby insofar course please third anyway therein wherein therefrom notwithstanding""",
    "subordinator": "whether unless",
    "particle": "under behind ahead back forward throughout before around away beside despite forwards",
    "parse_leftover": "can is r c ai_based cloud_based base",
}
ACTION = {"degree": "keep", "frequency_low": "keep", "frequency_high": "drop", "schedule": "keep", "manner": "keep",
          "manner_listed": "keep", "manner_generic": "drop", "focus_scalar": "drop", "focus_exclusive": "keep",
          "hedge_likelihood": "keep", "hedge_other": "drop", "negative_form": "keep", "sequence": "keep",
          "contrast": "keep", "concessive": "keep", "comparison_conditional": "keep if followed by 'to', else drop",
          "deictic_conditional": "drop at the end of a clause (before . , ; : or ')' or the end), else keep",
          "other": "leave", "time": "drop", "stance": "drop", "booster": "drop", "discourse": "drop", "subordinator": "drop",
          "particle": "open question 5", "parse_leftover": "leave"}
# the 12 "particle" rows, decided after item 5 (G3_POSTPROCESSING.md §6 item 8, owner 2026-10-09, option 3)
PARTICLE_ACTION = {**{w: "fuse with its verb like a phrasal particle, else leave in its phrase"
                      for w in ("forward", "forwards", "ahead", "behind", "back", "away")},
                   **{w: "drop" for w in ("before", "throughout", "despite", "beside")},
                   **{w: "keep before a number (numbers rule), else drop" for w in ("around", "under")}}
OWNER_NOTES = {   # verbatim from the owner's review, commit 03d29d8
    "yet": "keep", "still": "drop - does not add new information",
    "otherwise": "not sure about this one - assistant recommended keep (contrastive/conditional, like instead/rather)",
    "periodically": "can be kept similar to manner adverb", "simply": "not sure - assistant recommends keep (exclusive, like merely/purely/solely)",
    "generally": "approve dropping", "actually": "approve dropping",
    "often": "propose to drop", "consistently": "propose to drop", "even": "propose to drop",
    "first": "propose to keep to avoid conflict between rules", "typically": "propose to drop",
    "closely": "propose to drop", "mainly": "propose to drop", "automatically": "propose to drop",
    "similarly": 'propose to keep if we have "similarly to"', "instead": "propose to keep (close to contradiction/negation",
    "above": "propose to keep if not in the end of sentense (e.g. above expectations/average)",
}
FLAGS = {
    "increasingly": "a trend over time, put under degree (kept); could be time",
    "necessarily": "'not necessarily X' would become 'not X' if dropped: kept as a likelihood hedge",
    "remarkably": "in-clause mostly degree ('remarkably high'); as an opener stance",
    "critically": "in-clause mostly manner ('critically appraise'), as an opener stance; default manner",
    "rather": "by analogy with 'instead' (owner): mostly 'rather than' (substitution, close to contrast)",
    "ever": "frequency or time",
    "far": "degree ('far exceeded'); 'so far' is time",
    "ultimately": "time or stance (result)",
    "only": "own guarded rule (joins the verb group only when a verb follows)",
    "notably": "in-clause degree ('notably lower'); the opener 'Notably,' is dropped by the opener rule",
    "simultaneously": "time ('at the same time') or manner",
    "traditionally": "time or hedge",
    "course": "'of course' (booster/discourse)",
    "sort": "'sort of' (hedge)",
    "please": "politeness in quoted prompts",
    "largely": "by analogy with 'mainly' (owner): 'largely due to'; was degree",
    "mostly": "by analogy with 'mainly' (owner); was degree",
    "frequently": "by analogy with 'often' (owner)",
    "usually": "by analogy with 'typically' (owner)",
    "commonly": "by analogy with 'often' (owner)",
    "always": "by analogy with 'often' (owner): a high-frequency generaliser",
    "particularly": "by analogy with 'mainly' / 'even' (owner)",
    "especially": "by analogy with 'mainly' / 'even' (owner)",
    "specifically": "by analogy with 'mainly' (owner); the opener 'Specifically,' is dropped by the opener rule",
    "primarily": "by analogy with 'mainly' (owner)",
    "predominantly": "by analogy with 'mainly' (owner)",
    "below": "by analogy with 'above' (owner)",
    "unexpectedly": "stance, like 'surprisingly'",
    "indirectly": "negative form of 'directly' (kept)",
    "differently": "comparison; kept",
}
NEGATIVE_FORMS = {   # text occurrences in G2 v3 shards 0-9, for words the attachment sample did not contain
    "incorrectly": 228, "differently": 81, "rarely": 75, "indirectly": 51, "minimally": 39, "insufficiently": 33,
    "disproportionately": 30, "inadequately": 13, "inconsistently": 12, "inaccurately": 11, "seldom": 10,
    "inappropriately": 10, "unexpectedly": 9, "uncritically": 8, "barely": 7, "hardly": 7, "improperly": 4, "scarcely": 4,
    "unreliably": 2, "infrequently": 2, "unpredictably": 2, "incompletely": 2, "unevenly": 2, "unsuccessfully": 1,
    "non_significantly": 1, "irregularly": 1, "unequally": 1, "insignificantly": 1, "unfavourably": 1, "negligibly": 1,
}


def main():
    d = json.load(open(SRC))
    role_of = {w: r for r, ws in ROLES.items() for w in ws.split()}
    counts = list(d["counts"])
    seen = {w for w, _ in counts}
    added = [(w, 0) for w, n in sorted(NEGATIVE_FORMS.items(), key=lambda x: -x[1]) if w not in seen]
    out = ["rank\tword\tattachments\trole\taction\tflag\towner_note\texample"]
    for i, (w, n) in enumerate(counts + added, 1):
        r = role_of.get(w) or ("manner" if w.endswith("ly") else "other")
        ex = d["examples"].get(w, "").replace("\t", " ").replace("\n", " ")
        if (w, n) in added:
            ex = f"(not in the attachment sample; {NEGATIVE_FORMS[w]} occurrences in the text of shards 0-9)"
        act, flag = ACTION[r], FLAGS.get(w, '')
        if r == "particle":
            act, flag = PARTICLE_ACTION[w], "decided in G3_POSTPROCESSING.md §6 item 8 (owner 2026-10-09, option 3)"
        out.append(f"{i}\t{w}\t{n}\t{r}\t{act}\t{flag}\t{OWNER_NOTES.get(w, '')}\t{ex}")
    path = os.path.join(HERE, "adverb_roles.tsv")
    open(path, "w").write("\n".join(out) + "\n")
    print(path, len(out) - 1, "words,", len(added), "added negative/low forms")


if __name__ == "__main__":
    main()
