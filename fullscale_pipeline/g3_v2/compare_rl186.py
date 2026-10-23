"""RL-186 trial check: a shard curated by the RL-186 script against the same shard of the RL-185 production run.
Expected differences only: (1) contrast atoms (versus, than, without, unlike, in_contrast) added, and "contrast" gone where
"in contrast" was fused; (2) token positions (atoms[].src) now per occurrence, each a subset of the RL-185 list. Every
other difference is reported. A sample of positions changes is printed for reading.
    python3 compare_rl186.py OLD_DIR NEW_DIR SHARD [SHARD ...]
"""
import collections, json, re, sys

CONTRAST = {"versus/M/en", "than/M/en", "without/M/en", "unlike/M/en", "in_contrast/M/en"}


def strip(edge):
    for a in CONTRAST:
        edge = edge.replace(" " + a, "")
    return edge


def records(path):
    return {r["uid"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def hes(r):
    return [k for p in r["parents"] for k in p["children"]] + r["cousins"]


def main():
    old_dir, new_dir, shards = sys.argv[1], sys.argv[2], sys.argv[3:]
    C = collections.Counter(); other, sample = [], []
    for sh in shards:
        old, new = records(f"{old_dir}/g3_test_{int(sh):03d}.jsonl"), records(f"{new_dir}/g3_test_{int(sh):03d}.jsonl")
        C["units old"] += len(old); C["units new"] += len(new)
        for uid in set(old) | set(new):
            if uid not in old or uid not in new:
                other.append(f"{uid}: in one run only"); continue
            o, n = old[uid], new[uid]
            eo = [x["edge"] for x in o["parents"]] + [x["edge"] for x in o["cousins"]]
            en = [x["edge"] for x in n["parents"]] + [x["edge"] for x in n["cousins"]]
            if eo != en:
                C["units with a contrast atom added"] += any(a in e for e in en for a in CONTRAST)
                if [re.sub(r" contrast/C/en", "", e) for e in eo] != [strip(e) for e in en] and \
                        [e for e in eo] != [strip(e) for e in en]:
                    other.append(f"{uid}: edges differ beyond contrast atoms\n   old {eo}\n   new {en}")
                continue
            for ho, hn in zip(hes(o), hes(n)):
                for ao, an in zip(ho["atoms"], hn["atoms"]):
                    so, sn = {x[2] for x in ao["src"]}, {x[2] for x in an["src"]}
                    C["atom entries"] += 1
                    if so != sn:
                        C["atom entries with fewer positions"] += sn < so
                        if not sn < so:
                            other.append(f"{uid} {hn['id']} {an['atom']}: positions {sorted(so)} -> {sorted(sn)} (not a subset)")
                        elif len(sample) < 12:
                            sample.append(f"{hn['id']} {an['atom']}: {sorted(so)} -> {sorted(sn)}   [{n['text'][:110]}]")
    for k, v in C.items():
        print(f"{k:45s} {v:>10,}")
    print(f"other differences: {len(other)}")
    for x in other[:20]:
        print("  ", x)
    print("sample of position changes:")
    for x in sample:
        print("  ", x)


if __name__ == "__main__":
    main()
