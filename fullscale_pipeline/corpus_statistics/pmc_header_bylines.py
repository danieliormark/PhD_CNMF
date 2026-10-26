# RL-202 (owner 2026-10-26): bylines for working-corpus articles without a PMID (no PubMed record, so no byline backbone).
# Owner: option (a), authors from OpenAlex, checked against the publisher's record. The PMC article header (JATS front
# matter, contrib-type="author") takes PubMed's place: its personal authors are the bylines, aligned in order with the
# OpenAlex authorships by author_coverage.py's own align() (surname-confirmed pairs), and its ORCIDs are the publisher's
# (used as pm_orcid, so rule 1d prefers them). Same schema as bylines_v2_refresh.jsonl, plus source="pmc_header" and the
# header's first affiliation text (afftext). Collaborations (no personal name) are not bylines, as in PubMed.
# Key and contact from the environment only:  NCBI_API_KEY=... python3 pmc_header_bylines.py [--pmcids a.txt]
# Default list: the trial-list articles without a PMID. Writes (untracked, names) diagnostics/author_coverage/
# bylines_v2_pmc_header.jsonl; PMC XML cached in diagnostics/author_coverage/pmc_header_xml/.
import argparse, collections, json, os, re, time, unicodedata, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
HERE, D, OD = FP + "corpus_statistics/", FP + "diagnostics/openalex_authors/", FP + "diagnostics/author_coverage/"
src = open(HERE + "author_coverage.py").read()
exec(src[src.index("def toks"):src.index("bylines, arts")])               # toks, surname_matches, align, norm_orcid
exec(src[src.index("SAME_VERSION ="):src.index("OTHER_VERSION =")])
XD = OD + "pmc_header_xml/"


def fetch(p):
    f = XD + p + ".xml"
    if not os.path.exists(f):
        q = {"db": "pmc", "id": p[3:], "retmode": "xml"}
        if os.environ.get("NCBI_API_KEY"):
            q["api_key"] = os.environ["NCBI_API_KEY"]
        x = urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(q),
                                   timeout=60).read()
        open(f, "wb").write(x); time.sleep(0.4)
    return ET.parse(f).getroot()


def afftext(x):                                     # the affiliation's text without its label and institution IDs
    parts = []
    def walk(e, top=False):
        if e.tag in ("label", "institution-id", "sup") and not top:
            if e.tail: parts.append(e.tail)
            return
        if e.text: parts.append(e.text)
        for c in e: walk(c)
        if e.tail and not top: parts.append(e.tail)
    walk(x, True)
    return " ".join(" ".join(parts).split()).strip(" ,;")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pmcids", default="psychology_trial_pmcids.txt"); a = ap.parse_args()
    os.makedirs(XD, exist_ok=True)
    H = json.load(open(D + "headers.json")); H.update(json.load(open(D + "headers_new.json")))
    V = json.load(open(D + "openalex_verified.json")); V.update(json.load(open(D + "openalex_verified_new.json")))
    OW = json.load(open(OD + "openalex_works.json"))
    todo = [p for p in (l.strip() for l in open(HERE + a.pmcids)) if p and not (H.get(p) or {}).get("pmid")]
    out, S = [], collections.Counter()
    for p in todo:
        front = fetch(p).find(".//front")
        affs = {x.get("id"): afftext(x) for x in front.iter("aff")}
        au = []
        for c in front.iter("contrib"):
            n = c.find("name")
            if c.get("contrib-type") != "author" or n is None:
                continue
            orc = [x.text for x in c.iter("contrib-id") if x.get("contrib-id-type") == "orcid"]
            ref = [x.get("rid") for x in c.iter("xref") if x.get("ref-type") == "aff"]
            own = [afftext(x) for x in c.iter("aff")]
            txt = own[0] if own else (affs.get(ref[0], "") if ref else (next(iter(affs.values()), "") if len(affs) == 1 else ""))
            au.append(dict(last=(n.findtext("surname") or "").strip(), fore=(n.findtext("given-names") or "").strip(),
                           orcid=norm_orcid(orc[0]) if orc else None, afftext=txt))
        m, w = V.get(p) or ["unmatched", None]
        st = "same_version" if m in SAME_VERSION and w else "other_version" if w else "none"
        oa = (OW["by_id"].get(w["id"].rsplit("/", 1)[-1]) or {}).get("authorships") or [] if w and st == "same_version" else []
        pairs = align([(x["last"], x["fore"]) for x in au], [x["raw"] or x["name"] for x in oa]) if oa else {}
        for i, x in enumerate(au):
            o = oa[pairs[i]] if i in pairs else None
            oo = norm_orcid(o["orcid"]) if o and o.get("orcid") else None
            S["bylines"] += 1; S["aligned"] += o is not None; S["with OpenAlex ID"] += bool(o and o.get("id"))
            S["ORCID publisher"] += bool(x["orcid"]); S["ORCID both, same"] += bool(x["orcid"] and oo == x["orcid"])
            S["ORCID both, differ"] += bool(x["orcid"] and oo and oo != x["orcid"])
            S["with institution"] += bool(o and o.get("institutions"))
            out.append(dict(pmcid=p, pos=i, n=len(au), last=x["last"], fore=x["fore"], pm_orcid=x["orcid"], oa_orcid=oo,
                            oa_id=o.get("id") if o else None, aligned=o is not None, oa_idx=pairs.get(i), oa_status=st,
                            year=(H.get(p) or {}).get("year", ""), n_affs=int(bool(x["afftext"])), source="pmc_header",
                            afftext=x["afftext"]))
        print(p, f"PMC authors {len(au)}, OpenAlex authorships {len(oa)}, aligned {len(pairs)}")
    with open(OD + "bylines_v2_pmc_header.jsonl", "w", encoding="utf-8") as f:
        for b in out:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(dict(S))


if __name__ == "__main__":
    main()
