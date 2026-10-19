# Ticket 107 P1.A checks (owner, 2026-10-19), extending name_linking_test.py (CORPUS_STATISTICS.md §9.6).
# A1  Linking bylines without an ID to EACH OTHER: hide-and-recover. A random set of bylines whose identity we know
#     (OpenAlex author ID / ORCID), the size of the real unlinked group, has its identities hidden; the corroborated
#     rule links them among themselves (same name key, different articles, a shared OpenAlex institution or a shared
#     co-author whose ID is known); links are compared with the truth. 3 random draws.
# A2  Name-key variants, in A1 and in the original "link to a person whose ID we know" test (corroborated rule):
#     K1 surname + first given name (current); K2 surname + first initial; K3 surname + all given names.
# Rules and evidence exactly as in name_linking_test.py. Read-only, no network.
import collections, json, random, re, unicodedata

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
OD = FP + "diagnostics/author_coverage/"
H = json.load(open(FP + "diagnostics/openalex_authors/headers.json"))
V = json.load(open(FP + "diagnostics/openalex_authors/openalex_verified.json"))
OW = json.load(open(OD + "openalex_works.json"))
B = [json.loads(l) for l in open(OD + "bylines_refresh.jsonl")]


def work(p):
    m, w = V.get(p) or ["unmatched", None]
    fresh = OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) if w else None
    by_doi = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
    return by_doi if by_doi and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh


Wk = {p: work(p) for p in {b["pmcid"] for b in B}}
for b in B:
    w = Wk[b["pmcid"]]
    a = w["authorships"][b["oa_idx"]] if w and b["oa_idx"] is not None else None
    b["inst"] = {i["id"] for i in (a or {}).get("institutions", []) if i.get("id")}

norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()


def make_key(variant):
    def k(last, fore):
        f, l = re.findall(r"[a-z]+", norm(fore)), "".join(re.findall(r"[a-z]+", norm(last)))
        if not l or not f:
            return None
        if variant == "K1":
            return (l, f[0]) if len(f[0]) > 1 else None
        if variant == "K2":
            return (l, f[0][0])
        if variant == "K3":
            return (l, " ".join(f)) if len(f[0]) > 1 else None
    return k


parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for b in B:
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    for k in ks:
        find(k)
    for k in ks[1:]:
        parent[find(k)] = find(ks[0])
for i, b in enumerate(B):
    ks = [k for k in (("A", b["oa_id"]), ("O", b["pm_orcid"]), ("O", b["oa_orcid"])) if k[1]]
    b["person"] = find(ks[0]) if ks else None
    b["i"] = i
by_art = collections.defaultdict(list)
for b in B:
    by_art[b["pmcid"]].append(b)
n_unlinked = sum(1 for b in B if not b["person"])
known = [b for b in B if b["person"]]
OUT = {"n_real_unlinked": n_unlinked}


def heldout_to_known(keyf):
    """The original test (corroborated rule), with a given name key."""
    for b in B:
        b["key"] = keyf(b["last"], b["fore"])
        b["coauth"] = {c["person"] for c in by_art[b["pmcid"]] if c is not b and c["person"]}
    P_keys, P_arts = collections.defaultdict(set), collections.defaultdict(set)
    P_ev = collections.defaultdict(lambda: collections.defaultdict(lambda: [set(), set()]))
    for b in known:
        p = b["person"]
        if b["key"]:
            P_keys[p].add(b["key"])
        P_arts[p].add(b["pmcid"])
        P_ev[p][b["pmcid"]][0] |= b["inst"]; P_ev[p][b["pmcid"]][1] |= b["coauth"] - {p}
    key2P = collections.defaultdict(set)
    for p, ks in P_keys.items():
        for k in ks:
            key2P[k].add(p)

    def corr(b, c):
        return any((b["inst"] & inst or (b["coauth"] - {c}) & co) for art, (inst, co) in P_ev[c].items() if art != b["pmcid"])

    R = {"multi": collections.Counter(), "single": collections.Counter()}
    for b in known:
        p = b["person"]
        if not b["key"]:
            R["multi" if len(P_arts[p]) > 1 else "single"]["no key"] += 1
            continue
        multi = len(P_arts[p]) > 1
        cands = set(key2P[b["key"]]) - {p}
        if multi and any(x["key"] == b["key"] for art in P_arts[p] - {b["pmcid"]} for x in by_art[art] if x["person"] == p):
            cands.add(p)
        cs = [c for c in cands if corr(b, c)]
        got = cs[0] if len(cs) == 1 else ("ambiguous" if cs else None)
        R["multi" if multi else "single"]["no link" if got is None else "ambiguous" if got == "ambiguous" else
                                         "correct" if got == p else "wrong (false merge)"] += 1
    return R


def heldout_among_unlinked(keyf, seed):
    """A1: hide a random set of known bylines (size of the real unlinked group) and link them among themselves."""
    rng = random.Random(seed)
    hidden = {b["i"] for b in rng.sample(known, n_unlinked)}
    pairs_true, linked = set(), set()
    grp = collections.defaultdict(list)
    for b in known:
        if b["i"] not in hidden:
            continue
        b["key"] = keyf(b["last"], b["fore"])
        # evidence: co-authors whose ID is still known (other hidden bylines count as unknown)
        b["coauth"] = {c["person"] for c in by_art[b["pmcid"]] if c is not b and c["person"] and c["i"] not in hidden}
        if b["key"]:
            grp[b["key"]].append(b)
    same_person = collections.defaultdict(list)
    for b in known:
        if b["i"] in hidden:
            same_person[b["person"]].append(b)
    for bs in same_person.values():
        for x in bs:
            for y in bs:
                if x["i"] < y["i"] and x["pmcid"] != y["pmcid"]:
                    pairs_true.add((x["i"], y["i"]))
    for bs in grp.values():
        for x in bs:
            for y in bs:
                if x["i"] < y["i"] and x["pmcid"] != y["pmcid"] and (x["inst"] & y["inst"] or x["coauth"] & y["coauth"]):
                    linked.add((x["i"], y["i"]))
    correct = linked & pairs_true
    by_i = {b["i"]: b for b in known}
    # bylines merged with a different person (each wrong pair counts its bylines)
    wrong_bylines = {i for pr in linked - pairs_true for i in pr}
    singles = {b["i"] for b in known if b["i"] in hidden and len({x["pmcid"] for x in same_person[b["person"]]}) == 1}
    return dict(hidden=len(hidden), true_pairs=len(pairs_true), linked_pairs=len(linked), correct_pairs=len(correct),
                precision=len(correct) / len(linked) if linked else None,
                recall=len(correct) / len(pairs_true) if pairs_true else None,
                bylines_in_a_wrong_pair=len(wrong_bylines),
                share_of_hidden_in_a_wrong_pair=len(wrong_bylines) / len(hidden),
                hidden_without_any_true_partner=len(singles))


for v in ("K1", "K2", "K3"):
    kf = make_key(v)
    OUT[v] = {"to_known": heldout_to_known(kf), "among_unlinked": [heldout_among_unlinked(kf, s) for s in (1, 2, 3)]}
    print(f"\n=== {v}")
    for pop, c in OUT[v]["to_known"].items():
        n = sum(c.values())
        print(f"  link to a known person, {pop}-article people (n={n:,}): " + ", ".join(f"{k} {100 * x / n:.1f}%" for k, x in c.most_common()))
    for r in OUT[v]["among_unlinked"]:
        print(f"  among unlinked (seed): true pairs {r['true_pairs']}, linked {r['linked_pairs']}, correct {r['correct_pairs']}, "
              f"precision {r['precision']:.3f}, recall {r['recall']:.3f}, bylines in a wrong pair {r['bylines_in_a_wrong_pair']} "
              f"({100 * r['share_of_hidden_in_a_wrong_pair']:.2f}% of hidden)")
json.dump(OUT, open(OD + "name_linking_checks.json", "w"), indent=1, default=dict)
