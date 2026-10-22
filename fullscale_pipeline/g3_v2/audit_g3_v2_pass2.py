import json, re, collections, random
random.seed(1)
OUT = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/g3_v2/"
SET = r'(base|large|small|tiny|mini|medium|xl|xxl|xxlarge|uncased|cased|multilingual|instruct|it|chat|pro|flash|lite|turbo|\d+(?:\.\d+)?b|\d+m)'
CK = re.compile(r'^(.+?)(?:_' + SET + r'|(?<=bert)' + SET + r'|(?<=roberta)' + SET + r')(_.*)?$')
canon = collections.Counter(); ck = collections.Counter(); ckarts = collections.defaultdict(set)
TX = {"figure_label": re.compile(r"^(Fig\.?|Figure|Table|Supplementary)\s*S?\d", re.I),
      "low_letters": None, "code_like": re.compile(r"[;{}]\s*$|\bdef \w+\(|==|!=|\+\+"),
      "markup": re.compile(r"</?[a-z]+[ >]|\[\[|\]\]|\{\{")}
txs = collections.defaultdict(list); txn = collections.Counter()
shared = collections.Counter(); shared_s = collections.defaultdict(list)
for l in open(OUT + "g3_test_v2.jsonl"):
    r = json.loads(l)
    for m in r["focal_mentions"]:
        c = m["canonical"]; canon[c] += 1
        if CK.match(c) and c not in ("esm_1b",):
            ck[c] += 1; ckarts[c].add(r["pmcid"])
    t = r["text"]
    for k, rx in TX.items():
        hit = (sum(ch.isalpha() for ch in t) / max(1, len(t)) < 0.6) if rx is None else bool(rx.search(t))
        if hit:
            txn[k] += 1
            if len(txs[k]) < 25 or random.random() < 0.01:
                if len(txs[k]) >= 25: txs[k][random.randrange(25)] = t[:220]
                else: txs[k].append(t[:220])
    pos = collections.defaultdict(set)
    for p in r["parents"]:
        for c in p["children"]:
            for a in c["atoms"]:
                for x in a["src"]:
                    if x[2] is not None: pos[x[2]].add(p["id"])
    for c in r["cousins"]:
        for a in c["atoms"]:
            for x in a["src"]:
                if x[2] is not None: pos[x[2]].add(c["id"])
    multi = [q for q, s in pos.items() if len(s) > 1]
    if multi:
        emb = any(p.get("embedded_in") for p in r["parents"])
        kind = "embedded" if emb else ("cousin_and_parent" if any(any(".K" in i for i in pos[q]) for q in multi) else "two_parents_no_embed")
        shared[kind] += 1
        if len(shared_s[kind]) < 8 and random.random() < 0.02:
            words = t.split()
            shared_s[kind].append([t[:200], [p["edge"][:200] for p in r["parents"]], [c["edge"][:120] for c in r["cousins"]][:4]])
json.dump(dict(ck=ck.most_common(), ck_occ=sum(ck.values()), ck_spell=len(ck), ck_art=len(set().union(*ckarts.values())),
               txn=txn, txs=txs, shared=shared, shared_s=shared_s),
          open("/tmp/claude-766719/-mnt-hum01-home01-p91688di/c37b6c8f-0a98-41fc-8ef0-f898a4724f86/scratchpad/pass2.json", "w"), indent=1, ensure_ascii=False)
print(sum(ck.values()), len(ck), dict(txn), dict(shared))
