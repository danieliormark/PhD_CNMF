"""Ticket 105, diagnosis D1 (CLAUDE.md §4.25 update 2026-10-07): soft planted worlds,
profile-based measures, prior arms.

Question: does the temporal prior help where the truth is known, and if not, is the cause the
prior's quality (a poor T1 fit) or the mechanism? Run before any calibration of the weight.

Config C2, VARIANT "atoms" (all C2 relations) or "noatoms" (without M_Atom_Child and
M_Fringe_Cousin: the owner's full-scale plan has no atom facets).

World. Per global facet, a T1-live and a T2-live set with a set overlap (persisting entities);
articles are slice-specific. REGIME "toy": live counts, overlaps and mean degrees of the real
toy T1/T2 matrices; "dense": same live counts, 50% of each T2 facet persisting, degrees x3.
Truth is SOFT: every live entity has a membership profile theta ~ Dirichlet(ALPHA) over the
communities (ALPHA = 0.3: most entities have a leading community, some are mixed). Ties: each
live row entity gets degree max(1, Poisson(mean degree)); each tie picks a community k from
the row entity's profile, then a partner with probability proportional to the partners'
membership in k (mixed-membership generation); duplicates dropped; every live column keeps at
least one tie. Values 1.

T2 truth from T1 truth (K1 = 4), per SCENARIO, for persisting entities:
  growth           profiles unchanged; new entities over the same 4 communities
  migration        25% of persisting entities shift: theta2 = (1-s) theta1 + s e_j, j not their
                   T1 leader, s ~ U(0.5, 0.9) (a profile shift, the soft meaning of migration)
  split            community 0 splits into 0 and 4: each entity's mass on 0 divided by its own
                   fraction q ~ Beta(0.5, 0.5); K2 = 5
  merge            communities 2 and 3 merge (mass on 3 moves to 2); K2 = 3
  dissolve_emerge  community 3 ends (its mass spread over 0-2 in proportion); community 4 is
                   new and held by new entities only; K2 = 4 (labels 0, 1, 2, 4)
"Changed" persisting entities (for the cost measures): migrants; split: theta1[0] >= 0.3;
merge: theta1[3] >= 0.3; dissolve_emerge: theta1[3] >= 0.3. Others are "stable".

Fits (production run_inner_solver, lambda_z_offdiag 0.12, cap 8000): T1 at K = 4, seed 42
(the "knee"), plus 10 T1 refits at seeds 1000-1009 for the seed-based arms; T2 at the true K2,
at T2 seeds 2000 and 2001, for these ARMS:
  none                no prior (weight 0)
  control             knee prior at weight 1e-6 (term on, almost no force: path sensitivity)
  knee                knee prior                     at weights 1e-3, 1e-2
  consensus           consensus prior                at weights 1e-3, 1e-2
  knee_conf           knee prior + confidence        at weights 1e-3, 1e-2
  consensus_conf      consensus + confidence         at weights 1e-3, 1e-2
  oracle              true-membership prior (true T1 profiles of persisting entities)
                                                     at weights 1e-3, 1e-2
  knee_late           knee prior, freeze at epoch 600 instead of 300, weight 1e-2

Measures, all profile-based (no leader classes), fitted profiles mapped to the true
communities by a Hungarian assignment on the cosine between columns:
  - prior quality: mean JS distance between each arm's prior profiles and the true T1 profiles
  - error: mean JS distance between fitted and true T2 profiles, for stable persisting, changed
    persisting and new entities
  - change tracking (growth, migration: same communities in both slices): slope of fitted change
    (JS between the fitted T1 knee profile and the fitted T2 profile) against true change, over
    persisting entities, and the mean fitted change of truly stable entities (invented change)
  - community events: each true T2 community found (best column cosine >= 0.5); fitted
    communities whose membership mass is >= 90% new entities (spurious "new" communities, where
    the truth has none except the emerged one); whether the emerged community was matched
  - matches: accepted T1->T2 pairs and how many are wrong under the scenario's mapping
  - reconstruction (math_loss), convergence, epochs

    python d1_soft_diagnosis.py <atoms|noatoms> <scenario> <toy|dense> [worlds]
"""
import importlib.util, os, sys, json, time, torch
E = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution"
OUT = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("c13v10", f"{E}/chunk13v10.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment as lsa

VARIANT, SCENARIO, REGIME = sys.argv[1], sys.argv[2], sys.argv[3]
WORLDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
SMOKE = os.environ.get("D1_SMOKE") == "1"
CFG, K1, LZ, ALPHA = "C2", 4, 0.12, 0.3
T1_SEEDS = list(range(1000, 1010)) if not SMOKE else [1000, 1001]
T2_SEEDS = [2000, 2001] if not SMOKE else [2000]
ARMS = [("none", 0.0), ("control", 1e-6)] + [(a, w) for a in ("knee", "consensus", "knee_conf",
        "consensus_conf", "oracle") for w in (1e-3, 1e-2)] + [("knee_late", 1e-2)]
if SMOKE:
    ARMS = [("none", 0.0), ("knee", 1e-2), ("consensus_conf", 1e-2), ("oracle", 1e-2), ("knee_late", 1e-2)]
SOC, SEM_ALL, ANC = m.get_active_facets(CFG)
SEM = SEM_ALL if VARIANT == "atoms" else [r for r in SEM_ALL if r not in ("M_Atom_Child", "M_Fringe_Cousin")]

real1 = m.load_and_validate_data(OUT + "Star_extended_matrices_t1_v2.pkl")
real2 = m.load_and_validate_data(OUT + "Star_extended_matrices_t2_v2.pkl")
pm1r, pm2r = m.build_presence_masks(real1, SOC, SEM), m.build_presence_masks(real2, SOC, SEM)
dims = dict(real1["dimensions"])
GLOBAL = [f for f in sorted(pm1r) if f != "art"]
N_ART1, N_ART2 = int(real1["S_Art_Auth"].shape[0]), int(real2["S_Art_Auth"].shape[0])


def mean_deg(raw, rel):
    d = np.diff(sp.csr_matrix(raw[rel]).indptr)
    return float(d[d > 0].mean())


DEG = {r: mean_deg(real2, r) * (3 if REGIME == "dense" else 1) for r in SOC + SEM}
K2T = {"growth": 4, "migration": 4, "split": 5, "merge": 3, "dissolve_emerge": 5}[SCENARIO]
K2FIT = {"growth": 4, "migration": 4, "split": 5, "merge": 3, "dissolve_emerge": 4}[SCENARIO]
NEW_COMMS = {"growth": [0, 1, 2, 3], "migration": [0, 1, 2, 3], "split": [0, 1, 2, 3, 4],
             "merge": [0, 1, 2], "dissolve_emerge": [0, 1, 2, 4]}[SCENARIO]
ALLOWED = {"growth": {0: {0}, 1: {1}, 2: {2}, 3: {3}}, "migration": {0: {0}, 1: {1}, 2: {2}, 3: {3}},
           "split": {0: {0, 4}, 1: {1}, 2: {2}, 3: {3}}, "merge": {0: {0}, 1: {1}, 2: {2}, 3: {2}},
           "dissolve_emerge": {0: {0}, 1: {1}, 2: {2}}}[SCENARIO]


def js_rows(P, Q):
    return m._js_distance_rows(P, Q)


def build_world(rng):
    L1, L2 = {}, {}
    for f in GLOBAL:
        n1, n2 = int(pm1r[f].sum()), int(pm2r[f].sum())
        nb = int((pm1r[f] & pm2r[f]).sum()) if REGIME == "toy" else int(0.5 * n2)
        nb = min(nb, n1, n2)
        perm = rng.permutation(dims[f])
        L1[f] = np.sort(perm[:n1])
        L2[f] = np.sort(np.concatenate([perm[:nb], perm[n1:n1 + n2 - nb]]))
    th1, th2, changed, new = {}, {}, {}, {}
    for f in GLOBAL:
        a = np.zeros((dims[f], K1)); b = np.zeros((dims[f], K2T))
        a[L1[f]] = rng.dirichlet(np.full(K1, ALPHA), len(L1[f]))
        pers = np.intersect1d(L1[f], L2[f]); nw = np.setdiff1d(L2[f], L1[f])
        b[np.ix_(nw, NEW_COMMS)] = rng.dirichlet(np.full(len(NEW_COMMS), ALPHA), len(nw))
        t = a[pers]; ch = np.zeros(len(pers), bool)
        if SCENARIO == "growth":
            b[pers, :4] = t
        elif SCENARIO == "migration":
            b[pers, :4] = t
            mv = np.where(rng.random(len(pers)) < 0.25)[0]
            for i in mv:
                j = rng.choice([x for x in range(4) if x != t[i].argmax()])
                s = rng.uniform(0.5, 0.9)
                b[pers[i], :4] = (1 - s) * t[i] + s * np.eye(4)[j]
            ch[mv] = True
        elif SCENARIO == "split":
            q = rng.beta(0.5, 0.5, len(pers))
            b[pers, 1:4] = t[:, 1:4]; b[pers, 0] = t[:, 0] * q; b[pers, 4] = t[:, 0] * (1 - q)
            ch = t[:, 0] >= 0.3
        elif SCENARIO == "merge":
            b[pers, 0] = t[:, 0]; b[pers, 1] = t[:, 1]; b[pers, 2] = t[:, 2] + t[:, 3]
            ch = t[:, 3] >= 0.3
        elif SCENARIO == "dissolve_emerge":
            rest = t[:, :3].sum(1, keepdims=True)
            b[pers, :3] = t[:, :3] + t[:, 3:4] * t[:, :3] / np.where(rest > 0, rest, 1)
            ch = t[:, 3] >= 0.3
        th1[f], th2[f] = a, b
        c = np.zeros(dims[f], bool); c[pers[ch]] = True; changed[f] = c
        n_ = np.zeros(dims[f], bool); n_[nw] = True; new[f] = n_
    th1["art"] = rng.dirichlet(np.full(K1, ALPHA), N_ART1)
    b = np.zeros((N_ART2, K2T)); b[:, NEW_COMMS] = rng.dirichlet(np.full(len(NEW_COMMS), ALPHA), N_ART2)
    th2["art"] = b; new["art"] = np.ones(N_ART2, bool); changed["art"] = np.zeros(N_ART2, bool)
    L1["art"], L2["art"] = np.arange(N_ART1), np.arange(N_ART2)
    return L1, L2, th1, th2, changed, new


def synth(rng, live, th, n_art):
    out = {}
    for rel in SOC + SEM:
        fa, fb = m.RELATION_MAP[rel]
        na = n_art if fa == "art" else dims[fa]; nb = n_art if fb == "art" else dims[fb]
        rows, cols = live[fa], live[fb]
        colp = th[fb][cols]; colp = colp / np.where(colp.sum(0) > 0, colp.sum(0), 1.0)   # per community
        R, C = [], []
        for i in rows:
            p = th[fa][i]
            if p.sum() <= 0:
                continue
            d = max(1, rng.poisson(DEG[rel]))
            ks = rng.choice(len(p), size=d, p=p / p.sum())
            for k in ks:
                if colp[:, k].sum() > 0:
                    R.append(i); C.append(cols[rng.choice(len(cols), p=colp[:, k])])
        R, C = np.array(R, int), np.array(C, int)
        orph = np.setdiff1d(cols, C)
        if len(orph):
            R = np.concatenate([R, rng.choice(rows, len(orph))]); C = np.concatenate([C, orph])
        X = sp.csr_matrix((np.ones(len(R)), (R, C)), shape=(na, nb)); X.data[:] = 1.0
        out[rel] = X
    d = dict(dims); d["art"] = n_art; out["dimensions"] = d
    return out


def fit(raw, K, seed, prior=None, w=0.0, warm=None):
    m.set_seeds(seed)
    params = {"lambda_z_offdiag": LZ, "lambda_temporal": w}
    if warm is not None:
        params["temporal_warmup"] = warm
    U, Z, d = m.run_inner_solver(raw_data=raw, soc_keys=SOC, sem_keys=SEM, anchor_keys=ANC, K=K,
                                 dimensions=raw["dimensions"], params=params, device=m.DEVICE,
                                 seed_function=lambda s=seed: m.set_seeds(s),
                                 inner_epochs=m.EXTENDED_EPOCHS, temporal_prior=prior if w > 0 else None)
    return ({f: u.cpu().numpy().astype(np.float64) for f, u in U.items()},
            {r: z.cpu().numpy().astype(np.float64) for r, z in Z.items()}, d)


cn = lambda A: A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def to_true(Un, Zn, raw, th, live):
    """Fitted weighted memberships mapped onto the true communities. Returns per facet the
    mapped profiles (rows of live entities), the fitted-column -> true-community map."""
    pm = m.build_presence_masks(raw, SOC, SEM)
    P, eng = m.compute_weighted_membership(Un, Zn, pm)
    facs = [f for f in P if f in th]
    F = np.vstack([P[f][live[f]] for f in facs]); T = np.vstack([th[f][live[f]] for f in facs])
    S = cn(T).T @ cn(F)
    r, c = lsa(-S)
    mapped = {}
    for f in facs:
        A = np.zeros((len(live[f]), T.shape[1])); A[:, r] = P[f][live[f]][:, c]
        s = A.sum(1, keepdims=True); mapped[f] = A / np.where(s > 0, s, 1.0)
    return mapped, {int(cc): int(rr) for rr, cc in zip(r, c)}, S, P, eng


def prior_quality(prior, th1, t1map):
    """Mean JS distance of the prior's profiles from the true T1 profiles, prior columns mapped
    to true communities (t1map: prior column -> true community; identity for the oracle)."""
    ds = []
    for f, v in prior["facets"].items():
        if not len(v["idx"]):
            continue
        A = np.zeros((len(v["idx"]), K1))
        for c, r in t1map.items():
            A[:, r] = v["P"][:, c]
        s = A.sum(1, keepdims=True); A = A / np.where(s > 0, s, 1.0)
        ds.append(js_rows(A, th1[f][v["idx"]]))
    return float(np.mean(np.concatenate(ds))) if ds else None


results = []
t0 = time.time()
for world in range(WORLDS):
    rng = np.random.default_rng(10_000 * world + sum(map(ord, VARIANT + SCENARIO + REGIME)))
    L1, L2, th1, th2, changed, new = build_world(rng)
    raw1, raw2 = synth(rng, L1, th1, N_ART1), synth(rng, L2, th2, N_ART2)
    U1, Z1, d1 = fit(raw1, K1, 42)
    seeds = [fit(raw1, K1, s)[:2] for s in T1_SEEDS]
    k1map_true, k1_to_true, _, P1fit, _ = to_true(U1, Z1, raw1, th1, L1)   # knee column -> true T1 community
    priors = {
        "knee": m.temporal_prior_from_model(U1, Z1, raw1, raw2, SOC, SEM, {"arm": "knee"}),
        "consensus": m.temporal_prior_from_seeds([(U1, Z1)] + seeds, raw1, raw2, SOC, SEM, True, False),
        "knee_conf": m.temporal_prior_from_seeds([(U1, Z1)] + seeds, raw1, raw2, SOC, SEM, False, True),
        "consensus_conf": m.temporal_prior_from_seeds([(U1, Z1)] + seeds, raw1, raw2, SOC, SEM, True, True),
    }
    pm1s, pm2s = m.build_presence_masks(raw1, SOC, SEM), m.build_presence_masks(raw2, SOC, SEM)
    priors["oracle"] = {"facets": {f: {"idx": np.where(pm1s[f] & pm2s[f])[0],
                                       "P": th1[f][np.where(pm1s[f] & pm2s[f])[0]]}
                                   for f in GLOBAL if f in m.TEMPORAL_FACETS},
                        "K1": K1, "source": {"arm": "oracle"}}
    pq = {a: prior_quality(p, th1, {c: c for c in range(K1)} if a == "oracle" else k1_to_true)
          for a, p in priors.items()}
    # fitted T1 knee profiles in true coordinates, for change tracking
    k1_true = {f: np.zeros((dims[f], K1)) for f in GLOBAL}
    for f in GLOBAL:
        k1_true[f][L1[f]] = k1map_true[f]
    for seed in T2_SEEDS:
        for arm, w in ARMS:
            base = "knee" if arm in ("control", "knee_late") else arm
            prior = priors.get(base)
            U2, Z2, d2 = fit(raw2, K2FIT, seed, prior, w, warm=600 if arm == "knee_late" else None)
            mp, f2t, S2, _, _ = to_true(U2, Z2, raw2, th2, L2)
            err = {"stable": [], "changed": [], "new": []}
            fc, tc, stable_fc = [], [], []
            for f in GLOBAL:
                idx = L2[f]
                dj = js_rows(mp[f], th2[f][idx])
                pers = ~new[f][idx]
                err["new"] += list(dj[new[f][idx]])
                err["changed"] += list(dj[pers & changed[f][idx]])
                err["stable"] += list(dj[pers & ~changed[f][idx]])
                if SCENARIO in ("growth", "migration"):
                    pi = idx[pers]
                    keep = k1_true[f][pi].sum(1) > 0
                    fch = js_rows(k1_true[f][pi][keep], mp[f][pers][keep][:, :4])
                    tch = js_rows(th1[f][pi][keep], th2[f][pi][keep][:, :4])
                    fc += list(fch); tc += list(tch)
                    stable_fc += list(fch[~changed[f][pi][keep]])
            slope = float(np.polyfit(tc, fc, 1)[0]) if len(tc) > 2 and np.ptp(tc) > 0 else None
            true_present = [k for k in range(K2T) if any(th2[f][:, k].sum() > 0 for f in th2)]
            found = {k: bool(S2[k].max() >= 0.5) for k in true_present}
            # spurious new-only communities: fitted column c with >= 90% of its membership mass on new entities
            Pw, _ = m.compute_weighted_membership(U2, Z2, m.build_presence_masks(raw2, SOC, SEM))
            mass_new = np.zeros(K2FIT); mass_all = np.zeros(K2FIT)
            for f in GLOBAL:
                mass_all += Pw[f][L2[f]].sum(0); mass_new += Pw[f][L2[f]][new[f][L2[f]]].sum(0)
            share_new = mass_new / np.where(mass_all > 0, mass_all, 1)
            newonly_cols = [int(c) for c in range(K2FIT) if share_new[c] >= 0.9]
            spurious = [c for c in newonly_cols if f2t.get(c) != 4 or SCENARIO != "dissolve_emerge"]
            match = d2.get("temporal_match") or {}
            wrong, emerged_matched = 0, False
            t1map = {c: c for c in range(K1)} if base == "oracle" else k1_to_true
            for p in match.get("pairs", []):
                a, b = t1map.get(p["t1"]), f2t.get(p["t2"])
                wrong += not (a in ALLOWED and b in ALLOWED[a])
                emerged_matched |= (b == 4 and SCENARIO == "dissolve_emerge")
            rec = {"variant": VARIANT, "scenario": SCENARIO, "regime": REGIME, "world": world, "seed": seed,
                   "arm": arm, "weight": w, "prior_quality": pq.get(base), "t1_knee_converged": d1["converged"],
                   "err_stable": float(np.mean(err["stable"])) if err["stable"] else None,
                   "err_changed": float(np.mean(err["changed"])) if err["changed"] else None,
                   "err_new": float(np.mean(err["new"])) if err["new"] else None,
                   "n_stable": len(err["stable"]), "n_changed": len(err["changed"]),
                   "change_slope": slope,
                   "invented_change": float(np.mean(stable_fc)) if stable_fc else None,
                   "true_found": found, "spurious_new_only": len(spurious),
                   "accepted": len(match.get("pairs", [])), "wrong_matches": int(wrong),
                   "emerged_matched": bool(emerged_matched), "math_loss": d2["math_loss"],
                   "converged": d2["converged"], "epochs": len(d2["loss_history"])}
            results.append(rec)
            print(f"{VARIANT}/{SCENARIO}/{REGIME} w{world} s{seed} {arm:14s} {w:<6g} | prior JS "
                  f"{(pq.get(base) or 0):.3f} | err stable {rec['err_stable'] or 0:.3f} changed "
                  f"{rec['err_changed'] or 0:.3f} new {rec['err_new'] or 0:.3f} | slope "
                  f"{(slope if slope is not None else float('nan')):.2f} | found {sum(found.values())}/{len(found)} "
                  f"spurious {len(spurious)} | pairs {rec['accepted']} wrong {wrong} | conv {d2['converged']} "
                  f"{rec['epochs']} | {time.time()-t0:.0f}s", flush=True)
    json.dump(results, open(f"d1_{VARIANT}_{SCENARIO}_{REGIME}{'_smoke' if SMOKE else ''}.json", "w"),
              indent=1, default=float)
