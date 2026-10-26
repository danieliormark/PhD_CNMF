# RL-200 (owner 2026-10-26): institutions merged at the organisation level for the M1 affiliation facet. Owner: units
# of one university become that university (decided earlier: climb to the nearest university parent, never above it);
# likewise units of a research institute and departments of a corporation. OpenAlex's parents mix the organisation a
# unit belongs to with umbrellas above it (university systems, research societies, ministries), so each parent entity is
# put in a category, and only some categories absorb their children. No network; reads the cache of institutions_fetch.py.
#
# Categories of a parent (first matching rule; hand lists below):
#   SYSTEM        university system / federation / alliance / grouping of establishments          -> absorbs nothing
#   SOCIETY       national research organisation, society, academy, research network (question 1) -> absorbs nothing
#                 unless --merge-societies
#   GOVERNMENT    ministry, government, region, other public body                                 -> absorbs nothing
#   UMBRELLA      status label (IRCCS), funder, owner foundation or holding, company group, data error -> absorbs nothing
#   HEALTHCARE    a hospital or health organisation: absorbs its research institutes and foundations (child type
#                 facility, nonprofit, other) and its sites (a healthcare child whose name contains the parent's name:
#                 "Mayo Clinic in Arizona"); never a university, school or company, nor a separately named hospital
#                 (question 2; --merge-health-systems lets it absorb every child)
# Teaching hospitals: a healthcare institution with both a university parent and a healthcare parent is affiliated, not
# owned (Brigham and Women's: Tufts + Mass General Brigham), and stays itself. KEEP_SELF: read cases of a false parent.
#   ORG           university, company, research institute or other organisation               -> absorbs every child
# An institution's organisation: the organisations of the parents that absorb it (recursively); itself if none; if they
# lead to two or more different organisations it is a joint unit and stays itself (the toy-trial rule: a joint unit of
# one university and research bodies merges into the university, since research bodies absorb nothing here).
#   python3 institution_organisations.py --pmcids psychology_trial_pmcids.txt [--merge-societies] [--merge-health-systems]
# Writes corpus_statistics/institution_organisations_<name>.csv (institution -> organisation, rule) and
# institution_parent_categories.csv (every parent entity met, its category and the rule that set it).
import argparse, collections, csv, json, re

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
HERE, D, OD = FP + "corpus_statistics/", FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
C = json.load(open(OD + "institutions_openalex.json"))
sid = lambda u: u.rsplit("/", 1)[-1]
ty = lambda i: (C.get(i) or {}).get("type")
nm = lambda i: (C.get(i) or {}).get("display_name") or i
parents = lambda i: [sid(a["id"]) for a in (C.get(i) or {}).get("associated_institutions") or [] if a.get("relationship") == "parent"]

SYSTEM_RX = re.compile(r"(?i)\bsystems?\b|federation|federacja|associated universities|institutions of higher|higher education|"
                       r"group of institutions|colleges$|\btrust\b|universities$|^university of london$|^national university of ireland$|"
                       r"^université du québec$|^city university of new york$|^state university of new york$|public universities")
SYSTEM_IDS = {   # hand: groupings of establishments that keep their own identity
    "I4210105918": "Normandie Université (COMUE)", "I203339264": "Université de Lyon (COMUE)", "I2800365227": "Paris-Est Sup (COMUE)",
    "I2746051580": "Université PSL (constituent establishments keep their identity)",
    "I277688954": "Université Paris-Saclay (constituent establishments keep their identity)",
    "I899635006": "Université Grenoble Alpes (constituent establishments keep their identity)",
    "I4210145102": "Institut Polytechnique de Paris (alliance of schools)", "I205703379": "Institut Mines-Télécom (grouping of schools)",
    "I110525433": "Islamic Azad University (branches operate as separate universities; toy review)",
    "I191972202": "Amity University (campuses are separate universities)", "I10245363": "Rajamangala University of Technology (nine universities)",
    "I4210158934": "Research Organization of Information and Systems (inter-university institutes)",
    "I199525922": "National Institutes of Natural Sciences (inter-university institutes)",
    "I2801318690": "Fundación Universitaria San Pablo CEU (foundation running universities)",
    "I2799323385": "Board of the Swiss Federal Institutes of Technology",
}
SOCIETY_RX = re.compile(r"(?i)academy of sciences|^centre national de la recherche scientifique$|^inserm$|^institut des sciences|"
                        r"^national institutes of health$|helmholtz association|leibniz association|fraunhofer-gesellschaft|"
                        r"consejo superior de investigaciones|^department of .* sciences$|^institut national de recherche|"
                        r"^cnrs |^délégation|max planck society|^national research council$|agency for science, technology and research|"
                        r"^instituto de salud carlos iii$|^centro de investigación biomédica en red$|deutsche zentren der gesundheitsforschung|"
                        r"^nwo-i$|^cerca institution$|^academia sinica$|^riken$|national research council of science and technology|"
                        r"^institut de chimie$|^chinese academy of sciences$")
UMBRELLA_IDS = {
    "I4210153126": "IRCCS is a status label, not an organisation", "I87048295": "Wellcome Trust: funder",
    "I4210159963": "Novo Holdings: owner holding", "I2801134892": "Novo Nordisk Foundation: owner foundation",
    "I4391767947": "foundation owning the university", "I4210167194": "Asan Foundation: owner foundation of the hospital",
    "I4210125472": "Toyota Group: group of separate companies", "I2250650973": "Samsung: conglomerate of separate companies",
    "I4210120050": "Triad National Security: operator of Los Alamos, not its parent organisation",
    "I4387155568": "OpenAlex data error (an observatory as parent of a national research council)",
    "I34931013": "NIHR: funder", "I4210087105": "UK Research and Innovation: funder", "I2800991832": "Dutch Research Council: funder",
    "I4210120601": "Saiseikai: foundation running separate hospitals",
}


KEEP_SELF = {
    "I4210087487": "AP-HM is Marseille's own university hospital trust, not a unit of Aix-Marseille Université",
    "I4210132604": "CHRU Brest: OpenAlex lists it under one of its own laboratories (inverted parent)",
}


def category(p, args):
    if p in SYSTEM_IDS:
        return "SYSTEM", SYSTEM_IDS[p]
    if p in UMBRELLA_IDS:
        return "UMBRELLA", UMBRELLA_IDS[p]
    n, t = nm(p), ty(p)
    if SOCIETY_RX.search(n):
        return ("ORG" if args.merge_societies else "SOCIETY"), "research organisation / society / academy / network (name rule)"
    if t == "education" and SYSTEM_RX.search(n):
        return "SYSTEM", "university system or federation (name rule)"
    if t == "government":
        return "GOVERNMENT", "government body (type rule)"
    if t == "funder":
        return "UMBRELLA", "funder (type rule)"
    if t == "healthcare":
        return ("ORG" if args.merge_health_systems else "HEALTHCARE"), "hospital or health organisation (type rule)"
    if t == "other" and re.search(r"(?i)landschaftsverband|office of the director|command", n):
        return "GOVERNMENT", "public body (name rule)"
    return "ORG", f"{t} organisation (default)"


def absorbs(p, child, args):
    cat = category(p, args)[0]
    if cat == "ORG":
        return True
    if cat == "HEALTHCARE":
        core = re.sub(r"(?i)\b(the|nhs|foundation|trust)\b", "", nm(p)).strip(" ,-").lower()
        return ty(child) in ("facility", "nonprofit", "other") or \
            (ty(child) == "healthcare" and len(core) >= 4 and core in nm(child).lower())
    return False


memo = {}


def organisation(i, args, path=()):
    if i in memo:
        return memo[i]
    if i in KEEP_SELF:
        memo[i] = (i, "hand: " + KEEP_SELF[i]); return memo[i]
    ps = parents(i)
    if ty(i) == "healthcare" and any(ty(p) == "education" for p in ps) and any(ty(p) == "healthcare" for p in ps) \
            and not args.merge_health_systems:
        memo[i] = (i, "teaching hospital (university and health-system parents)"); return memo[i]
    ups = [p for p in ps if p not in path and absorbs(p, i, args)]
    if not ups:
        res = (i, "itself")
    else:
        orgs = {organisation(p, args, path + (i,))[0] for p in ups}
        res = (orgs.pop(), "merged") if len(orgs) == 1 else (i, "joint unit (" + "; ".join(sorted(nm(o) for o in orgs)) + ")")
    memo[i] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmcids", default="psychology_trial_pmcids.txt")
    ap.add_argument("--merge-societies", action="store_true")
    ap.add_argument("--merge-health-systems", action="store_true")
    a = ap.parse_args()
    pm = {l.strip() for l in open(HERE + a.pmcids) if l.strip()}
    B = [json.loads(l) for l in open(OD + "bylines_v2_refresh.jsonl")]
    H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
    V = json.load(open(D + "openalex_verified.json")); V.update(json.load(open(D + "openalex_verified_new.json")))
    OW = json.load(open(OD + "openalex_works.json"))

    def work(p):
        m, w = V.get(p) or ["unmatched", None]
        fresh = OW["by_id"].get(sid(w["id"])) if w else None
        bd = OW["by_doi"].get((((H.get(p) or {}).get("doi")) or "").lower())
        return bd if bd and (not fresh or m == "pmid_doi_mismatch_no_doi_hit") else fresh

    occ = collections.Counter()
    for b in B:
        if b["pmcid"] not in pm or not b["aligned"] or b["oa_idx"] is None:
            continue
        au = (work(b["pmcid"]) or {}).get("authorships") or []
        if b["oa_idx"] < len(au):
            for i in au[b["oa_idx"]].get("institutions") or []:
                if i.get("id"):
                    occ[sid(i["id"])] += 1
    rows, seen_par = [], {}
    for i in sorted(occ):
        o, how = organisation(i, a)
        rows.append(dict(institution=i, name=nm(i), type=ty(i), country=(C.get(i) or {}).get("country_code") or "", bylines=occ[i],
                         organisation=o, organisation_name=nm(o), rule=how))
        stack = list(parents(i))
        while stack:
            p = stack.pop()
            if p not in seen_par:
                seen_par[p] = category(p, a); stack += parents(p)
    tag = a.pmcids.replace("_pmcids.txt", "").replace(".txt", "") + ("_societies" if a.merge_societies else "") + \
        ("_healthsystems" if a.merge_health_systems else "")
    with open(HERE + f"institution_organisations_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(HERE + "institution_parent_categories.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["parent", "name", "type", "category", "rule"])
        for p, (c, why) in sorted(seen_par.items(), key=lambda x: (x[1][0], nm(x[0]))):
            w.writerow([p, nm(p), ty(p), c, why])
    orgs = {r["organisation"] for r in rows}
    hows = collections.Counter(r["rule"].split(" (")[0] for r in rows)
    print(f"institutions {len(rows)} (byline occurrences {sum(occ.values())}) -> organisations {len(orgs)}; {dict(hows)}")
    print("parent categories:", dict(collections.Counter(c for c, _ in seen_par.values())))


if __name__ == "__main__":
    main()
