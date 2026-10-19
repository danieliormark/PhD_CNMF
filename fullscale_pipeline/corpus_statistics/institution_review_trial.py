# Ticket 107 P1.B hand review (assistant, 2026-10-19; owner-approved option 4): every trial institution whose
# OpenAlex record names a university ("education") parent, with the decision and its reason. Rule (1): climb to the
# nearest university parent, never above it. Decisions per case below (hand review), written to
# institution_review_trial.csv. Reads the cached OpenAlex institution records; no network.
import csv, json
OD = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/diagnostics/author_coverage/"
c = json.load(open(OD + "trial_institutions_openalex.json"))
ids = list(json.load(open(OD + "trial_institution_lineage.json"))["top"].keys())
sid = lambda u: u.rsplit("/", 1)[-1]

SYSTEM = "keep: parent is a university system / federation / alliance, not the same university"
MULTI = "keep: two or more university parents (joint unit); no single defensible choice"
SUB = "merge: sub-unit (lab, centre, institute, school, campus) of the university"
MED = "merge: medical centre / hospital recorded as part of the university"
JOINT1 = "merge: joint research unit with one university parent (co-parents are research bodies)"
D = {  # institution id -> (decision, target id or None, reason)
 "I4210106115": ("merge", "I15057530", JOINT1), "I4210128565": ("keep", None, "keep: government agency site (CEA), member of the university, not part of it"),
 "I4210160507": ("keep", None, MULTI), "I4210092322": ("keep", None, MULTI), "I4210103698": ("keep", None, MULTI),
 "I4405252802": ("merge", "I76903346", SUB), "I4394709155": ("merge", "I76903346", SUB),
 "I4210116924": ("keep", None, MULTI), "I4210092150": ("merge", "I111277659", MED), "I36672615": ("merge", "I57206974", SUB),
 "I4390039268": ("merge", "I195731000", SUB), "I4210126319": ("keep", None, MULTI),
 "I4210088449": ("merge", "I173439891", "merge: school of HES-SO, the degree-granting university"),
 "I47265099": ("keep", None, SYSTEM), "I2802445891": ("merge", "I131736796", MED), "I4405253306": ("merge", "I88060688", SUB),
 "I4210086369": ("merge", "I204730241", JOINT1), "I4210130152": ("merge", "I204730241", JOINT1),
 "I4210119061": ("keep", None, MULTI), "I4210165077": ("merge", "I204730241", JOINT1),
 "I155419210": ("keep", None, "keep: branch universities of Islamic Azad University operate as separate universities"),
 "I2799980853": ("merge", "I197347611", MED), "I4210085887": ("keep", None, MULTI),
 "I4387152856": ("merge", "I277688954", JOINT1), "I4389425324": ("keep", None, MULTI), "I4210115485": ("merge", "I102197404", JOINT1),
 "I148283060": ("keep", None, "keep: national laboratory (DOE) managed by, not part of, the university system"),
 "I4400008897": ("merge", "I96226040", SUB), "I4405258931": ("merge", "I63966007", SUB), "I4210122954": ("merge", "I63966007", SUB),
 "I4389425352": ("merge", "I241749", JOINT1), "I4210142652": ("merge", "I56590836", JOINT1),
 "I4210150261": ("keep", None, "keep: public health cluster, not a unit of the university"),
 "I4210146690": ("keep", None, "keep: hospital of a public health cluster; its parent is not a university unit"),
 "I74022624": ("keep", None, SYSTEM + " (only parent listed is the system)"),
 "I4387153155": ("merge", "I8087733", SUB), "I135768898": ("keep", None, "keep: parent is the legal trust, not a university"),
 "I4210087136": ("merge", "I2279609970", JOINT1), "I4210137306": ("merge", "I97018004", MED),
 "I4391767685": ("merge", "I170215575", "merge: state key laboratory hosted by the university"),
 "I4391768176": ("merge", "I75689368", "merge: state key laboratory hosted by the university"),
 "I165951966": ("keep", None, "keep: separate institution in 2020-2021 (merged with UTSA only in 2025; ROR shows today's structure)"),
 "I4387153010": ("keep", None, "keep: member school of an alliance (Institut Polytechnique de Paris), legally part of Institut Mines-Télécom"),
 "I4390039331": ("merge", "I33213144", MED + " (via University of Florida Health)"), "I4390039284": ("merge", "I169521973", MED),
 "I4394709116": ("merge", "I31746571", SUB), "I4210108711": ("merge", "I159176309", MED),
 "I2800717037": ("merge", "I33213144", MED), "I4405255059": ("merge", "I169521973", MED),
 "I4405259976": ("merge", "I86767153", "merge: duplicate OpenAlex record of the same university (Université Jean Monnet)"),
 "I204730241": ("keep", None, SYSTEM + " (Université Paris Lumières is an association of universities)"),
 "I4405272987": ("merge", "I205783295", MED + " (via Weill Cornell Medicine)"),
 "I4387153466": ("merge", "I205783295", "merge: medical college of the university"),
 "I2800091995": ("merge", "I24185976", MED),
}
rows = []
for i in sorted(ids, key=lambda i: c[i]["display_name"]):
    ps = [a for a in c[i].get("associated_institutions") or [] if a["relationship"] == "parent"]
    ep = [a for a in ps if a.get("type") == "education"]
    if not ep and i != "I4405259976":
        continue
    dec, tgt, why = D.get(i, ("keep", None, SYSTEM))
    tname = next((a["display_name"] for a in ep if sid(a["id"]) == tgt), None) or (c.get(tgt) or {}).get("display_name") or ""
    rows.append(dict(openalex_id=i, name=c[i]["display_name"], type=c[i]["type"], country=c[i].get("country_code") or "",
                     university_parents="; ".join(f"{a['display_name']} ({sid(a['id'])})" for a in ep),
                     other_parents="; ".join(a["display_name"] for a in ps if a not in ep),
                     decision=dec, target_id=tgt or "", target_name=tname, reason=why))
with open("institution_review_trial.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import collections
print(len(rows), collections.Counter(r["decision"] for r in rows))
print(collections.Counter(r["reason"].split(" (")[0] for r in rows).most_common())
