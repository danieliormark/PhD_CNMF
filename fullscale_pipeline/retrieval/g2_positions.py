"""RL-186 (owner 2026-10-23): exact token position of every atom occurrence of the G2 v3.2 unit edges, for G3's provenance.
G2 stored atom2word as a list of (atom string, word, position), so a word that occurs twice in a unit cannot be tied to its
own place in the edge (G3 listed both positions in both hyperedges: PMC13159497.r1.L191.S1, "current" 5 and 26). This
re-parses every parsed piece of PG/g2_v3_2/ with G2's own parse_piece (same text: the incremental add-ons act on the text
before parsing, and the stored piece text is the text that was parsed), and for every unit whose re-parsed edge is
identical to the stored one writes the positions in the order of hedge(main_edge).all_atoms() (null for builder atoms
without a token, e.g. +/B.am/.). Checks per unit: the edge string is identical; every (atom, position) pair is one of the
stored atom2word entries; the positions cover the stored entries. G2's output is not changed.
Output PG/g2_v3_2_pos/pos_NNN.jsonl ({uid, hash, status, pos}), status "exact" or the failed check; pos_done_NNN.json.
    python g2_positions.py --shard I            (PYTHONPATH=$HOME/np1_for_spacy, tensor_env)
    python g2_positions.py --summary
"""
import argparse, collections, glob, hashlib, importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
INP, OUT = PG + "g2_v3_2/shards/", PG + "g2_v3_2_pos/"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def run_shard(i, limit=0):
    spec = importlib.util.spec_from_file_location("inc_g2", HERE + "inc_g2.py")
    w = importlib.util.module_from_spec(spec); sys.modules["inc_g2"] = w; spec.loader.exec_module(w)
    m = w.m                                               # graphbrain_parse_v3 (parse_piece unchanged by the wrapper)
    from graphbrain.parsers import create_parser
    parser = create_parser(lang="en", lemmas=True)
    src = INP + f"g2_parsed_{i:03d}.jsonl"
    C = collections.Counter(); t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    with open(OUT + f"pos_{i:03d}.jsonl.part", "w", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            if limit and C["articles"] >= limit:
                break
            C["articles"] += 1
            for s in json.loads(line)["sentences"]:
                by_piece = collections.defaultdict(list)
                for u in s.get("units", []):
                    if u["main_edge"] is not None:
                        by_piece[u["piece"]].append(u)
                for pi, units in by_piece.items():
                    C["pieces"] += 1
                    parses, _ = m.parse_piece(parser, s["pieces"][int(pi)]["parsed"])
                    pool = collections.defaultdict(list)
                    for p in parses:
                        if p["main_edge"] is not None:
                            pool[p["main_edge"].to_str()].append(p)
                    for u in units:
                        C["units"] += 1
                        rec = dict(uid=u["uid"], hash=u["hash"], status="exact", pos=None)
                        if not pool.get(u["main_edge"]):
                            rec["status"] = "edge_differs"
                        else:
                            p = pool[u["main_edge"]].pop(0)
                            a2w = {id(k.atom_obj): v for k, v in p["atom2word"].items()}
                            atoms = p["main_edge"].all_atoms()
                            pos = [a2w[id(a)][1] if id(a) in a2w else None for a in atoms]
                            stored = collections.Counter((a, x) for a, _, x in u["atom2word"])
                            got = collections.Counter((str(a), x) for a, x in zip(atoms, pos) if x is not None)
                            if got - stored:
                                rec["status"] = "pair_not_stored"
                            elif set(x for _, x in stored) - set(x for _, x in got):
                                rec["status"] = "stored_position_missing"
                            else:
                                rec["pos"] = pos
                        C[rec["status"]] += 1
                        out.write(json.dumps(rec) + "\n")
    os.replace(OUT + f"pos_{i:03d}.jsonl.part", OUT + f"pos_{i:03d}.jsonl")
    done = dict(shard=i, input=src, input_sha=sha(src), script_sha=sha(os.path.abspath(__file__)), counts=dict(C),
                seconds=round(time.time() - t0), host=os.uname().nodename)
    json.dump(done, open(OUT + f"pos_done_{i:03d}.json", "w"), indent=1)
    print(json.dumps(done))


def summary():
    T = collections.Counter(); shards = sorted(glob.glob(OUT + "pos_done_*.json"))
    for f in shards:
        T.update(json.load(open(f))["counts"])
    rep = dict(shards=len(shards), counts=dict(T))
    json.dump(rep, open(OUT + "positions_summary.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="articles (a quick test; writes to the same place)")
    a = ap.parse_args()
    summary() if a.summary else run_shard(a.shard, a.limit)
