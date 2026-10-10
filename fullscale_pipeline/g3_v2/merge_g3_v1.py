"""
Stage G3, merge step (owner, 2026-10-10): joins the 50 shards of g3_curation_v2.py (submit_g3_v1.sh) into one
corpus-level set of files, re-deriving the completeness check directly from each shard's own g3_articles_NNN.jsonl
against its real G2 v3 input shard file (not from the reports' counts alone, beyond a consistency check on those).

A separate script from g3_curation_v2.py, the same separation G2 v3 kept (merge_g2_v3.py): merging needs neither
graphbrain nor the focal-term matcher, and keeping it separate keeps the two stages independently runnable and
hashable for the run log.

    python merge_g3_v1.py --nshards 50 [--outdir DIR] [--shardsdir DIR]

Checks, all of which stop the merge on failure:
  - every shard's report (g3_report_NNN.json) exists, was run against its own G2 v3 shard file with no --limit, used
    the same script (by sha256) and the same input hashes (INPUT_SHA, read from the script itself, not hardcoded
    here) as every other shard, and shows 0 problems and 0 unit errors;
  - every pmcid of that G2 v3 shard file, read directly, appears exactly once in the shard's g3_articles_NNN.jsonl,
    with no pmcid missing and none extra; article status counts (excluded + valid + invalid_no_parent) equal the
    G2 v3 shard's own article count;
  - no pmcid is duplicated across shards (G2 v3 already partitions articles disjointly; checked here, not assumed).
Refuses to overwrite an existing merged output.

Output (in --outdir, default PG/g3_v1/): g3_test_v1.jsonl, g3_noparent_v1.jsonl, g3_articles_v1.jsonl,
g3_nonprose_v1.jsonl, g3_errors_v1.jsonl, g3_v1_summary.json.
"""
import argparse, collections, hashlib, importlib.util, json, os, sys

PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
G2_SHARDS = PG + "g2_v3/shards/"
HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def g2_pmcids(tag):
    return [json.loads(line)["pmcid"] for line in open(G2_SHARDS + f"g2_parsed_{tag}.jsonl", encoding="utf-8")]


def check_shard(shardsdir, tag, want_script_sha, want_inputs):
    """re-derives one shard's article ledger from its own files against the real G2 v3 shard; exits on any inconsistency"""
    rep_p = shardsdir + f"g3_report_{tag}.json"
    if not os.path.exists(rep_p):
        sys.exit(f"shard {tag} is not done ({rep_p} missing)")
    r = json.load(open(rep_p))
    if r["limit"]:
        sys.exit(f"shard {tag} was run with --limit {r['limit']}, not the whole shard")
    if os.path.basename(r["input"]) != f"g2_parsed_{tag}.jsonl":
        sys.exit(f"shard {tag}: report input is {r['input']}, not g2_parsed_{tag}.jsonl")
    if r["script_sha"] != want_script_sha:
        sys.exit(f"shard {tag}: script_sha {r['script_sha']}, expected {want_script_sha} (every shard must use the same script)")
    if r["inputs"] != want_inputs:
        sys.exit(f"shard {tag}: input hashes {r['inputs']} differ from the script's own INPUT_SHA {want_inputs}")
    if r["n_problems"] != 0:
        sys.exit(f"shard {tag}: {r['n_problems']} problems in the report")
    if r["counts"].get("unit_errors"):
        sys.exit(f"shard {tag}: {r['counts']['unit_errors']} unit errors in the report")

    want = g2_pmcids(tag)
    if len(set(want)) != len(want):
        sys.exit(f"shard {tag}: the G2 v3 shard file itself has a duplicate pmcid")
    arts_p = shardsdir + f"g3_articles_{tag}.jsonl"
    rows = [json.loads(line) for line in open(arts_p, encoding="utf-8")]
    got = [x["pmcid"] for x in rows]
    if len(set(got)) != len(got):
        sys.exit(f"shard {tag}: {arts_p} has a duplicate pmcid")
    if set(got) != set(want):
        missing, extra = set(want) - set(got), set(got) - set(want)
        sys.exit(f"shard {tag}: {arts_p} does not match its G2 v3 shard file (missing {len(missing)}, extra {len(extra)}, "
                  f"e.g. {sorted(missing)[:3] or sorted(extra)[:3]})")
    status = collections.Counter(x["status"] for x in rows)
    if sum(status.values()) != len(want):
        sys.exit(f"shard {tag}: {sum(status.values())} article rows, {len(want)} expected")
    return r["counts"], status, got


def merge(a):
    d = a.shardsdir
    outs = {x: os.path.join(a.outdir, f"g3_{x}_v1.jsonl") for x in ("test", "noparent", "articles", "nonprose", "errors")}
    summ = os.path.join(a.outdir, "g3_v1_summary.json")
    for p in list(outs.values()) + [summ]:
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")
    os.makedirs(a.outdir, exist_ok=True)

    spec = importlib.util.spec_from_file_location("g3_curation_v2", os.path.join(HERE, "g3_curation_v2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    want_inputs = {p: h for p, h in mod.INPUT_SHA.items()}
    want_script_sha = sha(os.path.join(HERE, "g3_curation_v2.py"))

    total, all_status, all_pmcids, jobs = collections.Counter(), collections.Counter(), [], set()
    fps = {x: open(p + ".tmp", "w", encoding="utf-8") for x, p in outs.items()}
    try:
        for i in range(a.nshards):
            tag = f"{i:03d}"
            counts, status, pmcids = check_shard(d, tag, want_script_sha, want_inputs)
            all_pmcids += pmcids
            all_status += status
            for k, v in counts.items():
                if isinstance(v, int):
                    total[k] += v
            r = json.load(open(d + f"g3_report_{tag}.json"))
            if r.get("job"):
                jobs.add(str(r["job"]))
            for x in outs:
                src = d + f"g3_{x}_{tag}.jsonl"
                if os.path.exists(src):
                    with open(src, encoding="utf-8") as f:
                        for line in f:
                            fps[x].write(line)
            print(f"shard {tag}: {status['valid']} valid, {status.get('invalid_no_parent', 0)} invalid, "
                  f"{status.get('excluded', 0)} excluded, {counts['units']} units checked OK", flush=True)
        if len(set(all_pmcids)) != len(all_pmcids):
            dup = [p for p, n in collections.Counter(all_pmcids).items() if n > 1]
            sys.exit(f"{len(dup)} pmcids appear in more than one shard, e.g. {dup[:5]}")
    except SystemExit:
        for fp in fps.values():
            fp.close()
        for p in outs.values():
            if os.path.exists(p + ".tmp"):
                os.remove(p + ".tmp")
        raise
    for fp in fps.values():
        fp.close()
    for p in outs.values():
        os.replace(p + ".tmp", p)
    json.dump(dict(stage="G3 v1", nshards=a.nshards, script_sha=want_script_sha, input_hashes=want_inputs,
                   merge_script_sha=sha(__file__), article_status=dict(all_status), total_articles=sum(all_status.values()),
                   jobs=sorted(jobs), counts=dict(total)),
              open(summ, "w"), indent=1)
    print("merged:", dict(all_status), "| total counts:", dict(total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--outdir", default=PG + "g3_v1/")
    ap.add_argument("--shardsdir", default=None, help="only for tests")
    a = ap.parse_args()
    a.outdir = a.outdir if a.outdir.endswith("/") else a.outdir + "/"
    a.shardsdir = (a.shardsdir if a.shardsdir else a.outdir + "shards/")
    a.shardsdir = a.shardsdir if a.shardsdir.endswith("/") else a.shardsdir + "/"
    merge(a)


if __name__ == "__main__":
    main()
