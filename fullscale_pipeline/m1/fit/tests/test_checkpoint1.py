# Checkpoint 1 tests (F0 data, F1 loss, F2 model), ticket 107. Run from fullscale_pipeline/m1:
#   python -m fit.tests.test_checkpoint1            (all checks; real-data checks need the psych_trial_v3 build)
import json, math, sys, time
import numpy as np
import scipy.sparse as sp
import torch

from fit import data as D
from fit.loss import RelationTensors, relation_loss, total_loss, hidden_predictions
from fit.model import Model

torch.set_num_threads(1)
FAILS = []


def check(name, got, want=True, tol=None):
    if tol is None:
        ok = got == want
    else:
        ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(("PASS " if ok else "FAIL ") + f"{name}: {got}" + ("" if tol is None and want is True else f" (want {want})"))
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------- 1. relation weights
w = D.relation_weights(D.APPROACH[1])
check("A0 approach 1: semantic 0.25 each", (w["parent_child"], w["child_article"]) == (0.25, 0.25))
check("A0 approach 1: social 1/6 each", all(abs(w[r] - 1 / 6) < 1e-15 for r in ("author_article", "author_affiliation", "article_journal")))
check("weights sum to 1", sum(w.values()), 1.0, 1e-15)
w = D.relation_weights([r for r in D.APPROACH[2] if r != "article_journal"])
check("J1 (no journal): remaining social 1/4", all(abs(w[r] - 0.25) < 1e-15 for r in ("article_affiliation", "article_article")))
w = D.relation_weights(D.APPROACH[1], journal_multiplier=0.25)
check("J3a: journal 0.5·0.25/2.25", w["article_journal"], 0.5 * 0.25 / 2.25, 1e-15)
w = D.relation_weights(D.APPROACH[1], semantic_split=(0.7, 0.3))
check("A1: child×parent 0.35, child×article 0.15", (round(w["parent_child"], 15), round(w["child_article"], 15)) == (0.35, 0.15))
w = D.relation_weights([r for r in D.APPROACH[2] if r != "article_article"])
check("arm N: two social matrices 1/4", abs(w["article_affiliation"] - 0.25) < 1e-15 and abs(w["article_journal"] - 0.25) < 1e-15)

# ------------------------------------------------------------------------------ 2. loss against a dense brute force
def dense_case(seed, same_facet, n=13, m=11, K=3, hidden=True, missing=(2, 7)):
    rng = np.random.default_rng(seed)
    if same_facet:
        m = n
        X = sp.random(n, n, density=0.25, random_state=seed).toarray(); X = np.triu(X, 1); X = X + X.T
    else:
        X = sp.random(n, m, density=0.3, random_state=seed).toarray()
    obs = np.ones(n, bool)
    if not same_facet:
        obs[list(missing)] = False; X[~obs] = 0
    r, c = np.nonzero(X)
    hid = np.zeros(len(r), bool)
    if hidden:
        hid[rng.choice(len(r), 4, replace=False)] = True
    emp = [(i, j) for i in range(n) for j in range(m) if obs[i] and X[i, j] == 0 and not (same_facet and i == j)]
    e = rng.choice(len(emp), 6 if hidden else 0, replace=False)
    er = np.array([emp[k][0] for k in e], dtype=np.int64); ec = np.array([emp[k][1] for k in e], dtype=np.int64)
    rd = D.RelationData("t", "f", "f" if same_facet else "g", (n, m), 1.0, "social", r[~hid], c[~hid], X[r, c][~hid],
                        obs, same_facet, r[hid], c[hid], X[r, c][hid], er, ec, 1.0)
    mask = np.zeros((n, m), bool); mask[obs] = True
    if same_facet: np.fill_diagonal(mask, False)
    mask[r[hid], c[hid]] = False; mask[er, ec] = False
    A = torch.rand(n, K, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)) + 0.05
    B = A if same_facet else torch.rand(m, K, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 1)) + 0.05
    Z = torch.rand(K, K, dtype=torch.float64, generator=torch.Generator().manual_seed(seed + 2)) + 0.05
    return rd, torch.as_tensor(X), torch.as_tensor(mask), A, Z, B


def dense_loss(X, mask, A, Z, B, beta):
    Xh = A @ Z @ B.T
    if beta == 2:
        return (((X - Xh) ** 2) * mask).sum()
    xl = torch.where(X > 0, X * torch.log(torch.where(X > 0, X, torch.ones_like(X)) / Xh), torch.zeros_like(X))
    return ((xl - X + Xh) * mask).sum()


for same in (False, True):
    for hidden in (False, True):
        for beta in (2, 1):
            rd, X, mask, A, Z, B = dense_case(3 + same + 2 * hidden, same, hidden=hidden)
            rt = RelationTensors(rd)
            for t in (A, Z, B): t.requires_grad_(True)
            l1 = relation_loss(A, Z, B, rt, beta); g1 = torch.autograd.grad(l1, [A, Z] + ([] if same else [B]))
            l2 = dense_loss(X, mask, A, Z, B, beta); g2 = torch.autograd.grad(l2, [A, Z] + ([] if same else [B]))
            tag = f"β={beta}, {'same facet+diagonal' if same else 'two facets+missing rows'}, {'hidden' if hidden else 'no hidden'}"
            check(f"loss = dense brute force ({tag})", l1.item(), l2.item(), 1e-12)
            check(f"gradient = dense brute force ({tag})", max(float((a - b).abs().max()) for a, b in zip(g1, g2)), 0.0, 1e-10)

# hidden predictions are the cell values of the reconstruction
rd, X, mask, A, Z, B = dense_case(9, False)
rt = RelationTensors(rd)
ht, he = hidden_predictions({"f": A, "g": B}, {"t": Z}, rt)
Xh = A @ Z @ B.T
check("hidden predictions = reconstruction at hidden cells",
      float(max((ht - Xh[rd.hid_tie_row, rd.hid_tie_col]).abs().max(), (he - Xh[rd.hid_empty_row, rd.hid_empty_col]).abs().max())), 0.0, 1e-14)

# ------------------------------------------------------------------------------------ 3. model: the scale freedom
rels = {"ab": ("a", "b"), "aa": ("a", "a"), "bc": ("b", "c")}
mdl = Model({"a": 6, "b": 5, "c": 4}, rels, K=3, generator=torch.Generator().manual_seed(0))
s0, Un0, Zs0 = mdl.normalised()
rec0 = {r: (mdl.U[a] @ mdl.Z[r] @ mdl.U[b].T).detach() for r, (a, b) in rels.items()}
cvec = torch.tensor([2.0, 0.5, 7.0], dtype=torch.float64)
with torch.no_grad():                                   # scale column k of U_b by c_k, compensate in every Z touching b
    mdl.U["b"].mul_(cvec)
    mdl.Z["ab"].div_(cvec[None, :]); mdl.Z["bc"].div_(cvec[:, None])
s1, Un1, Zs1 = mdl.normalised()
rec1 = {r: (mdl.U[a] @ mdl.Z[r] @ mdl.U[b].T).detach() for r, (a, b) in rels.items()}
check("scale freedom: reconstruction unchanged", max(float((rec0[r] - rec1[r]).abs().max()) for r in rels), 0.0, 1e-12)
check("scale freedom: U_norm unchanged", max(float((Un0[f] - Un1[f]).abs().max()) for f in Un0), 0.0, 1e-12)
check("scale freedom: Z_scaled unchanged", max(float((Zs0[r] - Zs1[r]).abs().max()) for r in rels), 0.0, 1e-12)
check("scale freedom: U_scales of b move by c", float((s1["b"] / s0["b"] - cvec).abs().max()), 0.0, 1e-12)
check("normalised reconstruction = raw reconstruction",
      max(float(((Un1[a] @ Zs1[r] @ Un1[b].T) - rec1[r]).abs().max()) for r, (a, b) in rels.items()), 0.0, 1e-12)
mdl2 = Model({"a": 3}, {"aa": ("a", "a")}, K=2, init_U={"a": np.full((3, 2), -1.0)}, init_Z={"aa": np.zeros((2, 2))})
check("clamp at construction (no entry below 1e-7)", float(min(mdl2.U["a"].min(), mdl2.Z["aa"].min())), 1e-7, 1e-12)

# ------------------------------------------------------------------------------------------- 4. real build (F0)
try:
    b = D.Build()
except FileNotFoundError:
    print("SKIP real-data checks: build not found"); b = None
if b is not None:
    saved = {"child_article": "child_article", "parent_child": "parent_child",
             "author_affiliation": "P2_author_affiliation", "article_affiliation": "P3_article_affiliation"}
    for rel, f in saved.items():
        r, c, v = b.ties(rel)
        fv = D.fold_values(b, rel, r, c, np.ones(len(r), bool))
        check(f"per-fold weights with nothing hidden = saved matrix ({rel})", float(np.abs(fv - v).max()), 0.0, 1e-12)
    man = json.load(open(b.path + "holdout/manifest.json"))
    for rel in ("child_article", "parent_child", "author_affiliation", "article_affiliation"):
        h = D.load_holdout(b, rel); m_ = man["relations"][rel]
        r, c, g = h["tie_row"], h["tie_col"], h["tie_group"]
        shape = b.load(D.RELATIONS[rel][0]).shape
        nr = np.bincount(r, minlength=shape[0]); nc = np.bincount(c, minlength=shape[1])
        elig = (nr[r] >= 2) & (nc[c] >= 2)
        check(f"{rel}: only eligible ties hidden", bool(np.all(elig[g >= 0])))
        check(f"{rel}: every eligible tie placed in one group", int((g[elig] >= 0).sum()), int(elig.sum()))
        worst_cap = worst_last = 0
        for gg in range(D.G):
            hid = g == gg
            lr = np.bincount(r[hid], minlength=shape[0]); lc = np.bincount(c[hid], minlength=shape[1])
            capr = np.maximum(1, nr // 3); capc = np.maximum(1, nc // 3)
            worst_cap = max(worst_cap, int((lr > capr).sum() + (lc > capc).sum()))
            worst_last = max(worst_last, int(((lr >= nr) & (nr > 0)).sum() + ((lc >= nc) & (nc > 0)).sum()))
        check(f"{rel}: cap max(1, ⌊n/3⌋) never exceeded", worst_cap, 0)
        check(f"{rel}: no entity loses its last tie", worst_last, 0)
        er, ec, eg = h["empty_row"], h["empty_col"], h["empty_group"]
        lin_e = er * shape[1] + ec
        check(f"{rel}: hidden empty cells distinct across groups", len(np.unique(lin_e)), len(lin_e))
        check(f"{rel}: hidden empty cells are not ties", int(np.isin(lin_e, r * shape[1] + c).sum()), 0)
        check(f"{rel}: hidden empty cells avoid missing rows", int(np.isin(er, b.missing_rows(rel)).sum()), 0)
        per = int(min(D.EMPTY_CAP, math.floor(D.EMPTY_SHARE * m_["observed_empty_cells"])))
        check(f"{rel}: empty cells per group = min(1 M, ⌊10%⌋) = {per:,}", bool(np.all(np.bincount(eg) == per)))
        print(f"     {rel}: ties {m_['ties']:,}, eligible {m_['eligible']:,}, per group {min(m_['ties_per_group']):,}–"
              f"{max(m_['ties_per_group']):,}; observed empty {m_['observed_empty_cells']:,}, hidden empty per group {per:,}")
    # one fold, approach 1 and 2
    for appr in (1, 2):
        rds = D.fit_data(b, appr, group=0)
        for rd in rds:
            fro_vis = float(np.sqrt(np.sum(rd.tie_val ** 2)))
            check(f"approach {appr}, group 0, {rd.name}: visible Frobenius norm = 1", fro_vis, 1.0, 1e-12)
        ca = next(rd for rd in rds if rd.name == "child_article")
        # hand check of one hidden child×article tie: idf from visible ties, row norm from visible ties
        tf = b.load("raw_child_article_tf")
        i, j = int(ca.hid_tie_row[0]), int(ca.hid_tie_col[0])
        df_vis = int(np.sum(ca.tie_col == j))
        idf = math.log(2136 / (df_vis + 1)) + 1
        row = ca.tie_row == i
        cols_vis = ca.tie_col[row]
        df_cols = np.bincount(ca.tie_col, minlength=tf.shape[1])
        vals_vis = np.log1p(tf[i, cols_vis].toarray().ravel()) * (np.log(2136 / (df_cols[cols_vis] + 1.0)) + 1)
        want = math.log1p(tf[i, j]) * idf / math.sqrt(math.sqrt(np.sum(vals_vis ** 2))) / ca.fro
        check(f"approach {appr}: hidden tie target at the fold's own weights (hand computation)", float(ca.hid_tie_val[0]), want, 1e-12)
    rds1 = {rd.name: rd for rd in D.fit_data(b, 1, group=0)}; rds2 = {rd.name: rd for rd in D.fit_data(b, 2, group=0)}
    check("semantic hidden cells shared by both approaches",
          all(np.array_equal(getattr(rds1[r], a), getattr(rds2[r], a)) for r in ("child_article", "parent_child")
              for a in ("hid_tie_row", "hid_tie_col", "hid_empty_row", "hid_empty_col")))
    full = {rd.name: rd for rd in D.fit_data(b, 1, group=None)}
    check("final refit (group None): no hidden cells", sum(len(rd.hid_row) for rd in full.values()), 0)
    check("article×article: diagonal masked, no hidden cells",
          next(rd for rd in D.fit_data(b, 2, group=0) if rd.name == "article_article").diag_masked)

    # ------------------------------------------------- 5. real-data loss: finite, and time per evaluation
    for appr in (1, 2):
        rds = D.fit_data(b, appr, group=0)
        rts = [RelationTensors(rd) for rd in rds]
        sizes = {}
        for rd in rds:
            sizes[rd.row_facet] = rd.shape[0]; sizes[rd.col_facet] = rd.shape[1]
        for K in (5, 20):
            mdl = Model(sizes, {rd.name: (rd.row_facet, rd.col_facet) for rd in rds}, K,
                        generator=torch.Generator().manual_seed(1))
            for beta in (2, 1):
                t = time.time()
                for _ in range(3):
                    mdl.zero_grad()
                    tot, parts = total_loss(dict(mdl.U), dict(mdl.Z), rts, beta)
                    tot.backward()
                dt = (time.time() - t) / 3
                ok = bool(torch.isfinite(tot)) and all(bool(torch.isfinite(p.grad).all()) for p in mdl.parameters())
                check(f"approach {appr}, K={K}, β={beta}: loss and gradients finite", ok)
                print(f"     forward+backward {dt:.3f} s per evaluation (one thread, float64, random start)")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
