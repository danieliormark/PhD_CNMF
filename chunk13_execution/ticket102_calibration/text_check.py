# Independent raw-text check for the ticket-102 grid.
# For every atom (core_atom, fringe_atom) and every fit: which of the 25 T1 abstracts literally
# contain the word (Porter-stem match, multiword atoms as contiguous stem sequences)?
# Text distribution T_i = sum of those articles' weighted community memberships (the fit's own),
# normalised. This never uses the parsed relation graph, so it is independent of the frozen s_i
# the term trains on. Compares the fit's atom membership P_i to T_i.
import glob, importlib.util, json, os, pickle, re, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from nltk.stem import PorterStemmer
from scipy.spatial.distance import jensenshannon

HERE = os.path.dirname(os.path.abspath(__file__))
TL = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large"
EXE = f"{TL}/chunk13_execution"
DATA = f"{TL}/outputs/Star_extended_matrices_t1_v2.pkl"
ATOMS = ["core_atom", "fringe_atom"]
ps = PorterStemmer()
m = raw = None


def build_occurrence():
    dec = pickle.load(open(f"{TL}/outputs/Star_epistemic_decoders_global_v2.pkl", "rb"))
    ab = pd.read_csv(f"{TL}/corpus_text.csv").set_index("article_id")["abstract"].to_dict()
    art_rev = {v: k for k, v in dec["maps_t1_art"].items()}
    docs = [[ps.stem(t) for t in re.findall(r"[a-z0-9]+", str(ab.get(art_rev[a], "")).lower())] for a in range(len(art_rev))]
    occ = {}
    for f in ATOMS:
        O = np.zeros((len(dec["maps"][f]), len(docs)), bool)
        for key, i in dec["maps"][f].items():
            toks = [ps.stem(t) for t in re.findall(r"[a-z0-9]+", key.split("/")[0].lower())]
            if not toks:
                continue
            n = len(toks)
            for a, d in enumerate(docs):
                O[i, a] = any(d[j:j + n] == toks for j in range(len(d) - n + 1))
        occ[f] = O
    return occ


def init():
    global m, raw, OCC
    os.chdir(os.path.join(HERE, "evalwd"))
    spec = importlib.util.spec_from_file_location("c13", f"{EXE}/chunk13v9.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    raw = m.load_and_validate_data(DATA)
    OCC = pickle.load(open(f"{HERE}/text_occ.pkl", "rb"))


def per_fit(js):
    meta = json.load(open(js)); z = np.load(js[:-5] + ".npz")
    U = {k[3:]: z[k] for k in z.files if k.startswith("U__")}
    Zm = {k[3:]: z[k] for k in z.files if k.startswith("Z__")}
    soc, sem, anc = m.get_active_facets(meta["config"])
    pm = m.build_presence_masks(raw, soc, sem)
    P, eng = m.compute_weighted_membership(U, Zm, pm)
    Pa = np.where((eng["art"] & pm["art"])[:, None], P["art"], 0.0)
    c = dict(n=0, shared=0, shared_excl=0, niche=0, niche_spread=0, unsup_excl=0, js=[])
    for f in ATOMS:
        if f not in P: continue
        O = OCC[f]
        T = O.astype(float) @ Pa
        tot = T.sum(1); nd = O.sum(1)
        ok = pm[f] & eng[f] & (tot > 0)
        for i in np.where(ok)[0]:
            t = T[i] / tot[i]; p = P[f][i]; top = p.argmax()
            c["n"] += 1; c["js"].append(jensenshannon(p, t, base=2))
            if p[top] > 0.9 and t[top] < 0.5: c["unsup_excl"] += 1
            if nd[i] >= 2 and t.max() < 0.6:
                c["shared"] += 1; c["shared_excl"] += int(p[top] > 0.9)
            if t.max() >= 0.9:
                c["niche"] += 1; c["niche_spread"] += int(p[top] < 0.6)
    return dict({k: meta[k] for k in ["config", "K", "lambda_conc", "gamma", "warmup", "seed"]},
                n=c["n"], shared=c["shared"], niche=c["niche"],
                shared_excl=c["shared_excl"] / max(c["shared"], 1), niche_spread=c["niche_spread"] / max(c["niche"], 1),
                unsup_excl=c["unsup_excl"] / max(c["n"], 1), js=float(np.mean(c["js"])) if c["js"] else None)


if __name__ == "__main__":
    occ = build_occurrence()
    pickle.dump(occ, open(f"{HERE}/text_occ.pkl", "wb"))
    for f, O in occ.items():
        print(f, "atoms found in >=1 abstract:", int((O.sum(1) > 0).sum()), "/", len(O),
              "| in >=2:", int((O.sum(1) >= 2).sum()))
    files = sorted(glob.glob(f"{HERE}/main/*.json")) + sorted(glob.glob(f"{HERE}/warm/*.json")) + sorted(glob.glob(f"{HERE}/dual/*.json"))
    with Pool(28, initializer=init) as p:
        rows = p.map(per_fit, files)
    json.dump(rows, open(f"{HERE}/text_rows.json", "w"))
    print("DONE", len(rows))
