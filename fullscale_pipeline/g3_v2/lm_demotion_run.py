"""RL-196 (owner 2026-10-26): targeted G3 rerun of the Psychology articles holding a demoted "language model" mention
(PG/g3_inputs_v3_1/lm_demotions_psychology.tsv, built by g3_inputs_v3_1/build_lm_demotions.py). The production output
PG/g3_v2/ is not changed. Per G2 shard holding such articles: their G2 records are copied to input/g2_subset_NNN.jsonl
and curated twice by g3_curation_v2.py with the shard's own positions file, into control/ (no extra rejections) and
demoted/ (--extra-rejections the demotion list). Checks:
  1. control reproduces production for these articles: every unit record, article row and occurrence row identical;
  2. demoted differs from control only in units of demoted sentences (their focal mention gone; a unit left without a
     focal term keeps only cousins, by the sentence-scoped cousin rule) and in article rows (an article with no focal
     unit left becomes invalid); any other difference is listed.
    python lm_demotion_run.py [--python PATH]       (tensor_env, PYTHONPATH=$HOME/np1_for_spacy, as submit_g3_v2.sh)
"""
import argparse, collections, csv, glob, json, os, subprocess, sys

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
G2 = PG + "g2_v3_2/shards/"
PROD = PG + "g3_v2/shards/"
OUT = PG + "g3_v2_lm_demotion/"
DEM = PG + "g3_inputs_v3_1/lm_demotions_psychology.tsv"
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"


def records(path, key):
    return {json.loads(l)[key]: json.loads(l) for l in open(path, encoding="utf-8")} if os.path.exists(path) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(DEM, encoding="utf-8"), delimiter="\t"))
    sids = {r["key"] for r in rows}
    arts = {s.split(".")[0] for s in sids}
    for d in ("input", "control", "demoted"):
        os.makedirs(OUT + d, exist_ok=True)
    shards = []
    for f in sorted(glob.glob(G2 + "g2_parsed_*.jsonl")):
        n = int(f[-9:-6])
        keep = [l for l in open(f, encoding="utf-8") if json.loads(l)["pmcid"] in arts]
        if keep:
            open(OUT + f"input/g2_subset_{n:03d}.jsonl", "w", encoding="utf-8").writelines(keep); shards.append(n)
    print("articles", len(arts), "sentences", len(sids), "shards", len(shards), flush=True)
    for n in shards:
        inp = OUT + f"input/g2_subset_{n:03d}.jsonl"
        for d, extra in (("control", []), ("demoted", ["--extra-rejections", DEM])):
            if os.path.exists(OUT + f"{d}/g3_report_{n:03d}.json"):
                continue
            subprocess.run([a.python, HERE + "g3_curation_v2.py", "--shard", str(n), "--input", inp, "--limit", "0",
                            "--show", "0", "--outdir", OUT + d] + extra, check=True, stdout=subprocess.DEVNULL)
    C, other = collections.Counter(), []
    for n in shards:
        t = f"{n:03d}"
        prod_u = {k: v for k, v in records(PROD + f"g3_test_{t}.jsonl", "uid").items() if v["pmcid"] in arts}
        prod_a = {k: v for k, v in records(PROD + f"g3_articles_{t}.jsonl", "pmcid").items() if k in arts}
        prod_o = {k: v for k, v in records(PROD + f"g3_occurrences_{t}.jsonl", "occ").items() if v["pmcid"] in arts}
        ctl_u, ctl_a = records(OUT + f"control/g3_test_{t}.jsonl", "uid"), records(OUT + f"control/g3_articles_{t}.jsonl", "pmcid")
        ctl_o = records(OUT + f"control/g3_occurrences_{t}.jsonl", "occ")
        dem_u, dem_a = records(OUT + f"demoted/g3_test_{t}.jsonl", "uid"), records(OUT + f"demoted/g3_articles_{t}.jsonl", "pmcid")
        for name, x, y in (("units", prod_u, ctl_u), ("articles", prod_a, ctl_a), ("occurrences", prod_o, ctl_o)):
            C[f"control = production, {name}"] += x == y
            if x != y:
                other.append(f"shard {t}: control differs from production in {name}: "
                             f"{len(set(x) ^ set(y))} keys only in one, {sum(x[k] != y[k] for k in set(x) & set(y))} changed")
        for uid in set(ctl_u) | set(dem_u):
            if ctl_u.get(uid) == dem_u.get(uid):
                continue
            sid = uid.rsplit(".", 1)[0]
            if sid in sids:
                C["units changed in demoted sentences"] += 1
                C["  of which left without a focal term"] += uid in dem_u and dem_u[uid]["focal_mentions"] in ("[]", [], None)
                C["  of which no longer written"] += uid not in dem_u
            else:
                other.append(f"{uid}: changed outside the demoted sentences")
        for p in set(ctl_a) | set(dem_a):
            if (ctl_a.get(p) or {}).get("status") != (dem_a.get(p) or {}).get("status"):
                C[f"article status {(ctl_a.get(p) or {}).get('status')} -> {(dem_a.get(p) or {}).get('status')}"] += 1
    rep = dict(articles=len(arts), sentences=len(sids), shards=len(shards), counts=dict(C), other=other)
    json.dump(rep, open(OUT + "lm_demotion_check.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
