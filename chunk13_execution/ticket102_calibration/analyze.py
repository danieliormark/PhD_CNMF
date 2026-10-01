# Per-fit metrics for the ticket-102 grid, evaluated with production chunk13v9.py functions.
import glob, importlib.util, itertools, json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
DATA = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1_v2.pkl"
STAGE = sys.argv[1]; D = os.path.join(HERE, STAGE)
TF = ["core_child_he", "cousin_he", "core_atom", "fringe_atom"]
m = raw = None


def init():
    global m, raw
    os.chdir(os.path.join(HERE, "evalwd")) if os.path.isdir(os.path.join(HERE, "evalwd")) else (os.makedirs(os.path.join(HERE, "evalwd")), os.chdir(os.path.join(HERE, "evalwd")))
    spec = importlib.util.spec_from_file_location("c13", f"{EXE}/chunk13v9.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    raw = m.load_and_validate_data(DATA)


def support(P, act, f):
    """Community distribution an entity inherits from the articles that use it (Penalty_B's propagation)."""
    K = P["art"].shape[1]; E = np.zeros((P[f].shape[0], K))
    for k in range(K):
        W = P["art"][:, k]; pw = {}
        if "M_Parent_Art" in act: pw["parent_he"] = raw["M_Parent_Art"].dot(W)
        cp = []
        if "M_Child_Art" in act: cp.append(raw["M_Child_Art"].dot(W))
        if "M_Child_Parent" in act and "parent_he" in pw: cp.append(raw["M_Child_Parent"].dot(pw["parent_he"]))
        if cp: pw["core_child_he"] = sum(cp)
        cu = []
        if "M_Cousin_Art" in act: cu.append(raw["M_Cousin_Art"].dot(W))
        if "M_Cousin_Parent" in act and "parent_he" in pw: cu.append(raw["M_Cousin_Parent"].dot(pw["parent_he"]))
        if "M_Cousin_Child" in act and "core_child_he" in pw: cu.append(raw["M_Cousin_Child"].dot(pw["core_child_he"]))
        if cu: pw["cousin_he"] = sum(cu)
        if "M_Atom_Child" in act and "core_child_he" in pw: pw["core_atom"] = raw["M_Atom_Child"].dot(pw["core_child_he"])
        if "M_Fringe_Cousin" in act and "cousin_he" in pw: pw["fringe_atom"] = raw["M_Fringe_Cousin"].dot(pw["cousin_he"])
        if f in pw: E[:, k] = pw[f]
    s = E.sum(1, keepdims=True)
    return np.where(s > 0, E / np.where(s > 0, s, 1), np.nan)


def col_weights(U, Z, f):
    w = np.zeros(U[f].shape[1])
    for r, z in Z.items():
        a, b = m.RELATION_MAP[r]
        if a == f: w += np.linalg.norm(z @ U[b].T, axis=1) ** 2
        if b == f: w += np.linalg.norm(U[a] @ z, axis=0) ** 2
    w = np.sqrt(w)
    return np.where(w < 1e-3 * w.max(), 0.0, w) if w.max() > 0 else w


def per_fit(js):
    meta = json.load(open(js))
    z = np.load(js[:-5] + ".npz")
    U = {k[3:]: z[k] for k in z.files if k.startswith("U__")}
    Z = {k[3:]: z[k] for k in z.files if k.startswith("Z__")}
    c, K = meta["config"], meta["K"]
    soc, sem, anc = m.get_active_facets(c)
    act = {k for k in soc + sem if k in raw}
    pm = m.build_presence_masks(raw, soc, sem)
    P, eng = m.compute_weighted_membership(U, Z, pm)
    ent = {"shared": 0, "shared_excl": 0, "n": 0, "unsup_excl": 0}
    Hsh, mass_sh, ids = [], [], []
    for f in TF:
        if f not in P: continue
        S_ = support(P, act, f)
        mask = pm[f] & eng[f] & ~np.isnan(S_).any(1)
        top = P[f].argmax(1); pmax = P[f].max(1)
        X = (U[f] * col_weights(U, Z, f)).sum(1)
        for i in np.where(mask)[0]:
            ent["n"] += 1
            excl = pmax[i] > 0.9
            if excl and S_[i, top[i]] < 0.5: ent["unsup_excl"] += 1
            if S_[i].max() < 0.6:
                ent["shared"] += 1; ent["shared_excl"] += int(excl)
                p = P[f][i]; p = p[p > 0]
                Hsh.append(float(-(p * np.log(p)).sum() / np.log(K)))
        # raw (unweighted) row mass of every live entity, for the draining check
        mass_sh.append(X[pm[f]]); ids.append(np.array([f"{f}:{i}" for i in np.where(pm[f])[0]]))
    desc = m.describe_solution(U, Z, raw, soc, sem, anc, m.MAX_MONOPOLY)
    ev = m.evaluate_complete_solution(U, Z, None, raw, soc, sem, anc, m.MAX_MONOPOLY,
                                      m.ENTROPY_THRESHOLD, m.TARGET_COHERENCE)
    art_top = np.where(eng["art"] & pm["art"], P["art"].argmax(1), -1)
    sem_unengaged = sum(int((pm[f] & ~eng[f]).sum()) for f in TF if f in eng)
    return dict(meta, shared=ent["shared"], shared_excl_frac=ent["shared_excl"] / max(ent["shared"], 1),
                unsup_excl_frac=ent["unsup_excl"] / max(ent["n"], 1), n_engaged_sem=ent["n"],
                shared_H_mean=float(np.mean(Hsh)) if Hsh else None, sem_unengaged=sem_unengaged,
                penA_w=desc["item_attribution_pen_weighted"], penB_w=desc["shared_term_concentration_pen_weighted"],
                penB=desc["shared_term_concentration_pen"], penA=desc["item_attribution_pen"],
                min_share=float(min(desc["community_share"])), soc_penalty=ev["sociological_penalty"],
                collapse_pen=ev["collapse_pen"],
                _art_top=art_top.tolist(), _mass=np.concatenate(mass_sh).tolist(), _ids=np.concatenate(ids).tolist())


def stability(group):
    """Track A/B over all seed pairs of one (cell, setting), with shuffled-row chance level."""
    key, files = group
    fits = []
    for js in sorted(files):
        meta = json.load(open(js))
        if not meta["converged"]: continue
        z = np.load(js[:-5] + ".npz"); fits.append({k[3:]: z[k] for k in z.files if k.startswith("U__")})
    c, K = key[0], key[1]
    if len(fits) < 2: return key, None
    soc, sem, anc = m.get_active_facets(c); facets = m.get_required_facets(soc, sem, anc)
    rng = np.random.default_rng(0); A, B, nA, nB = [], [], [], []
    for i, j in itertools.combinations(range(len(fits)), 2):
        a, b, _, _ = m._pair_tracks(fits[i], fits[j], facets, K); A.append(a); B.append(b)
        sh = {f: fits[j][f][rng.permutation(fits[j][f].shape[0])] for f in facets}
        a0, b0, _, _ = m._pair_tracks(fits[i], sh, facets, K); nA.append(a0); nB.append(b0)
    return key, dict(A=float(np.mean(A)), B=float(np.mean(B)), A_chance_max=float(np.max(nA)),
                     B_chance_max=float(np.max(nB)), n=len(fits))


if __name__ == "__main__":
    files = sorted(glob.glob(f"{D}/*.json"))
    with Pool(28, initializer=init) as p:
        rows = p.map(per_fit, files)
        groups = {}
        for js, r in zip(files, rows):
            groups.setdefault((r["config"], r["K"], r["lambda_conc"], r["gamma"], r["warmup"]), []).append(js)
        stab = dict(p.map(stability, list(groups.items())))
    json.dump(rows, open(f"{HERE}/rows_{STAGE}.json", "w"))
    json.dump([dict(config=k[0], K=k[1], lambda_conc=k[2], gamma=k[3], warmup=k[4], **(v or {})) for k, v in stab.items()],
              open(f"{HERE}/stab_{STAGE}.json", "w"))
    print("DONE", len(rows))
