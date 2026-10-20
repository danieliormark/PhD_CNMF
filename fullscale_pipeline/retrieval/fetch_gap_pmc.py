"""R3b (2026-10-19, ticket 107 P1, RL-154): download the full texts of the retrieval-gap candidates (RL-153) from the
public PMC open-access bucket, by the same method as R3 (fetch_delta_pmc.py: `aws s3 cp --no-sign-request`, the first
version found), with versions 1-9 instead of 1-4 (PIPELINE.md deviation D4). Into a separate folder,
LCS/pure_text_corpus_gap/, so nothing joins the corpus before the keep rule is applied. Resumable: existing files are
skipped. Writes fetch_gap_status.tsv (pmcid, status, version).
    python3 fetch_gap_pmc.py IDS.txt [WORKERS]      (8 parallel downloads by default, as R3)
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

LCS = "/mnt/hum01-rds/Basov/p91688di/llm_corpus_staging/"
OUT = LCS + "pure_text_corpus_gap/"
os.makedirs(OUT, exist_ok=True)
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def fetch(p):
    dst = OUT + p + ".txt"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return p, "present", ""
    for v in range(1, 10):
        r = subprocess.run(["aws", "s3", "cp", f"s3://pmc-oa-opendata/{p}.{v}/{p}.{v}.txt", dst, "--no-sign-request",
                            "--region", "us-east-1", "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return p, "downloaded", str(v)
    return p, "not_found", ""


with ThreadPoolExecutor(max_workers=WORKERS) as ex, open(LCS + "fetch_gap_status.tsv", "a") as st:
    for i, (p, s, v) in enumerate(ex.map(fetch, ids), 1):
        st.write(f"{p}\t{s}\t{v}\n")
        if i % 500 == 0:
            st.flush(); print(f"{i}/{len(ids)}", flush=True)
print("done", len(ids))
