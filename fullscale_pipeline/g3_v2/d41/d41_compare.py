# D4.1 check: G3 output of the patched script vs the production output (script b3eadb6d385c), one shard.
# Every atom removed or added is classified; anything not explained by a D4.1 rule is printed as UNEXPECTED.
import collections, json, re, sys, urllib.parse

old_p, new_dir, shard = sys.argv[1], sys.argv[2], sys.argv[3]
SUSPECT = re.compile("[ﬀ-ﬆ()\\[\\]{}|=^~<>∑∏√∂∫×÷∈∀∃≠±∗∘⊂⊆∪∩→←↔⇒⊕⊗⋅∣¬∼‘’‚‛“”„‟\"'`´′″Ͱ-Ͽ−\\\\]")


def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        atoms = collections.Counter()
        for s in [c for pp in r["parents"] for c in pp["children"]] + r["cousins"]:
            for a in s.get("atoms", []):
                atoms[(a["atom"], tuple(a["src"][0][:2]))] += 1
        out[r["uid"]] = (r["text"], atoms, len(r["parents"]), len(r["cousins"]))
    return out


O = load(old_p)
N = load(f"{new_dir}/g3_test_{int(shard):03d}.jsonl")
nonprose = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{new_dir}/g3_nonprose_{int(shard):03d}.jsonl")} if True else {}
C = collections.Counter(); kinds = collections.Counter(); unexpected = []; examples = collections.defaultdict(list)
gone = set(O) - set(N); new_only = set(N) - set(O)
C["units old"] = len(O); C["units new"] = len(N); C["units removed"] = len(gone); C["units added"] = len(new_only)
for u in sorted(gone):
    t = O[u][0]
    k = "panel legend" if re.match(r'^[a-h]\s*,\s|^[a-h](?:\s*,\s*[a-h])+\s*[:\s]', t) else "other"
    kinds["unit removed: " + k] += 1
    if k == "other":
        unexpected.append(("unit removed", u, t[:150]))
for u in sorted(set(O) & set(N)):
    to, ao, po, co = O[u]; tn, an, pn, cn = N[u]
    if ao == an and po == pn and co == cn:
        continue
    C["units changed"] += 1
    C["parents changed"] += po != pn; C["cousins changed"] += co != cn
    rem, add = ao - an, an - ao
    for (atom, (srcatom, word)), n in rem.items():
        w = urllib.parse.unquote(atom.split("/")[0])
        if re.fullmatch(r"\d+", w):
            k = "number dropped (citation)"
        elif SUSPECT.search(urllib.parse.unquote(word)) or SUSPECT.search(w):
            match = [a for (a, (s2, w2)) in add if w2 == word]
            k = "word changed" if match else "word dropped"
        else:
            k = "OTHER removal"
            unexpected.append(("atom removed", u, f"{atom} <- {word!r} | {to[:120]}"))
        kinds[k] += n
        if len(examples[k]) < 25:
            new = [a for (a, (s2, w2)) in add if w2 == word]
            examples[k].append(f"{word!r}: {atom} -> {new[0] if new else 'DROP'}")
    for (atom, (srcatom, word)), n in add.items():
        if not any(w2 == word for (a, (s2, w2)) in rem):
            kinds["OTHER addition"] += n
            unexpected.append(("atom added", u, f"{atom} <- {word!r} | {to[:120]}"))
print(json.dumps(C, indent=0))
print(json.dumps(kinds, indent=0))
for k, v in examples.items():
    print(f"\n{k}:\n   " + "\n   ".join(v))
print(f"\nUNEXPECTED ({len(unexpected)}):")
for x in unexpected[:60]:
    print("  ", x)
