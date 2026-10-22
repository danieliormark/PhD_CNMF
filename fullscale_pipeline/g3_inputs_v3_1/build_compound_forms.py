"""RL-180 (owner 2026-10-22): spellings of one compound written as one word atom. Reads the word atoms (not focal atoms) of a G3
output, puts each root through G3's own clean-up (clean_root, norm_word without forms), groups the roots by their letters with
joiners removed ('pre_trained', 'pretrained', 'pre-trained' -> 'pretrained') and writes, for every group with two or more
spellings, the most frequent spelling as the form all of them take (ties: the one with '_'). The read exceptions
(g3_curation_v2.COMPOUND_EXCEPT: pro_state, co_pilot, ...) keep their own spelling and are left out of their group.
    python3 build_compound_forms.py G3_TEST_JSONL [--out compound_forms.tsv]
"""
import argparse, collections, csv, importlib.util, json, os, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("g3_test")
    ap.add_argument("--out", default=os.path.join(HERE, "compound_forms.tsv"))
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("g3", os.path.join(HERE, "..", "g3_v2", "g3_curation_v2.py"))
    g3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g3)
    cnt = collections.Counter()
    for line in open(a.g3_test, encoding="utf-8"):
        r = json.loads(line)
        for st in [c for p in r["parents"] for c in p["children"]] + r["cousins"]:
            for at in st["atoms"]:
                if at["atom"].endswith("/focal"):
                    continue
                root = urllib.parse.unquote(at["atom"].rsplit("/", 2)[0])
                c = g3.clean_root(root)
                w = g3.norm_word(c) if c else None
                if w and w not in g3.COMPOUND_EXCEPT and not w.replace('.', '').isdigit():
                    cnt[w] += 1
    groups = collections.defaultdict(list)
    for w in cnt:
        groups[w.replace("_", "")].append(w)
    rows = []
    for key, forms in sorted(groups.items()):
        if len(forms) < 2:
            continue
        best = max(forms, key=lambda f: (cnt[f], "_" in f, -len(f)))
        rows.append(dict(key=key, form=best, spellings=" ".join(f"{f}:{cnt[f]}" for f in sorted(forms, key=lambda f: -cnt[f]))))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["key", "form", "spellings"], delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(len(rows), "groups,", sum(len(r["spellings"].split()) for r in rows), "spellings ->", a.out)


if __name__ == "__main__":
    main()
