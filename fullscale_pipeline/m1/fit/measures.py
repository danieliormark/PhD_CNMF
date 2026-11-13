# F6 — measures of finished M1 fits (ticket 107). Each function cites its decision (design doc, dated notes).
# Input: a fit result (solver.fit: raw U per facet, raw Z per relation, relations) and the fit's data.RelationData list.
# All quantities are computed identically for both losses (2026-11-07 (4)) and are unchanged by the scale freedom
# (ticket 79), except where stated. numpy, float64.
#
# Per fit:            weighted memberships and masses; tie-weight parts, side readings, half credit, relative presence;
#                     domain and facet presence, mono-domain / mono-facet flags; off-diagonal share (tie-weight and
#                     squared core-cell forms); N_eff ÷ K; empty columns at five references; entity measures
#                     (reproduced share, on-target share, coverage, accuracy under both metrics, degree-only reference);
#                     per-entity training AUC by tie-count band; held-out error, captured share and hidden-cell AUC by
#                     band; tie alignment with the stratified shuffle reference; spread diagnostics (floor, S3);
#                     membership-side reading of communities; loading/engagement distributions (item (l)).
# Across fits:        placed entities (seed variation); ghosts; Track A/B (weighted memberships; per pair, per seed,
#                     between groups, chance-adjusted); the counting rule.
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

COLUMN_CUTOFF = 1e-3
LOG2 = math.log(2.0)
REFERENCES = {"median": 50.0, "p5": 5.0, "p1": 1.0, "p0.5": 0.5, "p0.1": 0.1}       # 2026-11-07 (empty columns)
TIE_BANDS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10**12, "6+"))


def band_of(n):
    for lo, hi, name in TIE_BANDS:
        if lo <= n <= hi:
            return name
    return "0"


# ---------------------------------------------------------------------------------------------------- normalised view
def normalised(U, Z, relations):
    s = {f: np.linalg.norm(u, axis=0) for f, u in U.items()}
    Un = {f: U[f] / np.maximum(s[f], 1e-300) for f in U}
    Zs = {r: Z[r] * np.outer(s[a], s[b]) for r, (a, b) in relations.items()}
    return Un, Zs


def column_weights(Un, Zs, relations, f):
    """GLOSSARY "Column weights": w_k(f)² = Σ_r Z[k,:](U_gᵀU_g)Z[k,:]ᵀ (or the column form); cut-off 1e-3 × max."""
    w2 = np.zeros(Un[f].shape[1])
    for r, (a, b) in relations.items():
        z = Zs[r]
        if a == f:
            w2 += np.diag(z @ (Un[b].T @ Un[b]) @ z.T)
        if b == f:
            w2 += np.diag(z.T @ (Un[a].T @ Un[a]) @ z)
    w = np.sqrt(np.maximum(w2, 0.0))
    return np.where(w < COLUMN_CUTOFF * w.max(), 0.0, w) if w.max() > 0 else w


def weighted_memberships(U, Z, relations):
    """Per facet: P (rows sum to 1; ticket 100 weighted membership), mass (row sum of U_norm × w), weighted columns Y."""
    Un, Zs = normalised(U, Z, relations)
    out = {}
    for f in Un:
        w = column_weights(Un, Zs, relations, f)
        Y = Un[f] * w
        mass = Y.sum(1)
        P = Y / np.maximum(mass, 1e-300)[:, None]
        out[f] = {"P": P, "mass": mass, "Y": Y, "w": w}
    return out


def entropy_bits(P):
    return -(P * np.log(np.clip(P, 1e-300, None))).sum(-1) / LOG2


# ------------------------------------------------------------------------------------- tie-weight parts (step 6, view C)
def parts(U, Z, rd, floor_zero=1e-7):
    """part(k,l) = Z[k,l] a_k b_l, a = column sums of the row facet over observed rows, b = column sums of the column
    facet; adds up exactly to the reconstructed tie weight T over observed rows × all columns (GLOSSARY).
    floor_zero: if given, raw core cells at or below it (the positivity floor, 1e-7) are read as 0 (option under
    discussion, 2026-11-10: floor-level cells summed over many entities can outweigh one small tie)."""
    a = (U[rd.row_facet] * rd.obs_row_mask[:, None]).sum(0)
    b = U[rd.col_facet].sum(0)
    z = Z[rd.name] if floor_zero is None else np.where(Z[rd.name] <= floor_zero, 0.0, Z[rd.name])
    P = z * np.outer(a, b)
    return P, float(P.sum())


def presence(U, Z, rds, floor_zero=1e-7):
    """Per relation: side readings, half credit h_k, relative presence p_k = K h_k (2026-11-07), within share,
    off-diagonal share (tie-weight form) and the squared core-cell form (step 2 diagnostic), N_eff ÷ K (step 8),
    per-community within part ÷ half credit; and the squared-size reading of half credit (2026-11-07 (4))."""
    out = {}
    for rd in rds:
        P, T = parts(U, Z, rd, floor_zero)
        K = P.shape[0]
        d = np.diag(P)
        row_side = P.sum(1) / T
        col_side = P.sum(0) / T
        h = 0.5 * (row_side + col_side)
        a2 = ((U[rd.row_facet] * rd.obs_row_mask[:, None]) ** 2).sum(0); b2 = (U[rd.col_facet] ** 2).sum(0)
        Psq = Z[rd.name] ** 2 * np.outer(a2, b2)
        hsq = 0.5 * (Psq.sum(1) + Psq.sum(0)) / Psq.sum()
        zs = Z[rd.name] * np.outer(np.sqrt((U[rd.row_facet] ** 2).sum(0)), np.sqrt(b2))
        out[rd.name] = {"T": T, "row_side": row_side, "col_side": col_side, "half_credit": h, "presence": K * h,
                        "within_share": float(d.sum() / T), "off_diagonal_share": float(1 - d.sum() / T),
                        "off_diagonal_core_cells": float(((zs ** 2).sum() - (np.diag(zs) ** 2).sum()) / (zs ** 2).sum()),
                        "within_over_half_credit": d / np.maximum(h * T, 1e-300),
                        "N_eff_over_K": float(1.0 / (h ** 2).sum() / K),
                        "presence_squared_sizes": K * hsq, "domain": rd.domain,
                        "row_facet": rd.row_facet, "col_facet": rd.col_facet}
    return out


def domain_presence(pres):
    """P_k(D) = mean of p_k(r) over D's relation matrices (equal weight); semantic summary also as the minimum over the
    semantic relations and the rank correlation of child×article vs child×parent presence (2026-11-07)."""
    out = {}
    for D in ("semantic", "social"):
        rs = [r for r, v in pres.items() if v["domain"] == D]
        if rs:
            M = np.vstack([pres[r]["presence"] for r in rs])
            out[D] = {"mean": M.mean(0), "min": M.min(0), "relations": rs}
    if "child_article" in pres and "parent_child" in pres:
        from scipy.stats import spearmanr
        out["semantic_rank_correlation"] = float(spearmanr(pres["child_article"]["presence"],
                                                           pres["parent_child"]["presence"]).correlation)
    return out


def facet_presence(pres):
    """P_k(f) = K × mean over relations touching f of k's side reading on f's side (2026-11-08)."""
    acc = {}
    for r, v in pres.items():
        K = len(v["half_credit"])
        acc.setdefault(v["row_facet"], []).append(K * v["row_side"])
        if v["col_facet"] != v["row_facet"]:
            acc.setdefault(v["col_facet"], []).append(K * v["col_side"])
    return {f: np.mean(np.vstack(x), 0) for f, x in acc.items()}


def mono_flags(dom, fac, low=0.1, high=0.5, chance_relations=()):
    """Per fit (the 'most seeds' step is applied across fits): mono-domain if P(D1) < 0.1 and P(D2) ≥ 0.5, unless the
    low domain's relations are fitted only at chance (chance_relations: names with held-out score at chance);
    mono-facet if P_k(f) ≥ 0.5 in exactly one facet and < 0.1 in every other (2026-11-07, 2026-11-08)."""
    K = len(dom["semantic"]["mean"])
    md = []
    for k in range(K):
        s, o = dom["semantic"]["mean"][k], dom["social"]["mean"][k]
        if s < low and o >= high and not set(dom["semantic"]["relations"]) <= set(chance_relations):
            md.append((k, "social only"))
        elif o < low and s >= high and not set(dom["social"]["relations"]) <= set(chance_relations):
            md.append((k, "semantic only"))
    mf = []
    for k in range(K):
        hi = [f for f, p in fac.items() if p[k] >= high]
        lo = [f for f, p in fac.items() if p[k] < low]
        if len(hi) == 1 and len(lo) == len(fac) - 1:
            mf.append((k, hi[0]))
    return {"mono_domain": md, "mono_facet": mf}


# ----------------------------------------------------------------------------------------- empty columns (step 8)
def empty_columns(pres, rds, wm, placed=None):
    """Community k empty in relation r if its tie weight there (half credit × T_r) is below a reference tie weight of
    r (median and 5th/1st/0.5th/0.1th percentiles of the relation's visible tie values; default 0.1th, 2026-11-07);
    empty in facet f if empty in every relation touching f. Also, per reference, how many placed entities would change
    their largest membership if those columns were zeroed."""
    by = {rd.name: rd for rd in rds}
    out = {}
    for ref, q in REFERENCES.items():
        er = {}
        for r, v in pres.items():
            thr = float(np.percentile(by[r].tie_val, q))
            er[r] = np.nonzero(v["half_credit"] * v["T"] < thr)[0].tolist()
        ef = {}
        for f in wm:
            rels = [r for r, v in pres.items() if f in (v["row_facet"], v["col_facet"])]
            K = len(wm[f]["w"])
            ef[f] = [k for k in range(K) if all(k in er[r] for r in rels)]
        changed = {}
        for f, ks in ef.items():
            if not ks:
                changed[f] = 0; continue
            Y = wm[f]["Y"].copy(); Y[:, ks] = 0.0
            m = np.ones(len(Y), bool) if placed is None else placed[f]
            changed[f] = int(((Y.argmax(1) != wm[f]["Y"].argmax(1)) & m & (Y.sum(1) > 0)).sum())
        out[ref] = {"relations": er, "facets": ef, "placed_entities_changing_largest_membership": changed}
    return out


# ------------------------------------------------------------------------------------- entity measures (step 8)
def _row_side(A, Z, B, rd, transpose):
    """Per entity of one side of a relation: Σ_ties x, Σ_ties min(x̂, x), Σ_training x̂, Σ_training (x − x̂)²,
    KL part, degree-only Σ min(x̂₀, x)."""
    if not transpose:
        r, c = rd.tie_row, rd.tie_col; hr, hc = rd.hid_row, rd.hid_col; n = rd.shape[0]
        Am = A * rd.obs_row_mask[:, None]; Bf = B; Zr = Z
    else:
        r, c = rd.tie_col, rd.tie_row; hr, hc = rd.hid_col, rd.hid_row; n = rd.shape[1]
        Am = B; Bf = A * rd.obs_row_mask[:, None]; Zr = Z.T
    x = rd.tie_val
    xh = ((Am[r] @ Zr) * Bf[c]).sum(1)
    sum_x = np.bincount(r, weights=x, minlength=n)
    sum_min = np.bincount(r, weights=np.minimum(xh, x), minlength=n)
    full = Am @ Zr @ Bf.sum(0)                                  # Σ_j x̂_ij over all cells of the entity's line
    hxh = ((Am[hr] @ Zr) * Bf[hc]).sum(1) if len(hr) else np.zeros(0)
    sum_hat = full - np.bincount(hr, weights=hxh, minlength=n)
    G = Bf.T @ Bf
    sq_full = np.einsum("ik,kl,il->i", Am @ Zr, G, Am @ Zr)
    sq = (np.bincount(r, weights=x ** 2, minlength=n) - 2 * np.bincount(r, weights=x * xh, minlength=n)
          + sq_full - np.bincount(hr, weights=hxh ** 2, minlength=n))
    if rd.diag_masked:
        dg = np.nonzero(rd.obs_row_mask)[0]
        dv = ((Am[dg] @ Zr) * Bf[dg]).sum(1)
        sum_hat[dg] -= dv; sq[dg] -= dv ** 2
    pos = x > 0
    kl_t = np.where(pos, x * np.log(np.where(pos, x, 1.0) / np.maximum(xh, 1e-300)) - x, 0.0)
    kl = np.bincount(r, weights=kl_t, minlength=n) + sum_hat
    rt = np.bincount(r, weights=x, minlength=n); ct = np.bincount(c, weights=x, minlength=Bf.shape[0])
    x0 = rt[r] * ct[c] / x.sum()
    sum_min0 = np.bincount(r, weights=np.minimum(x0, x), minlength=n)
    return dict(sum_x=sum_x, sum_min=sum_min, sum_hat=sum_hat, sq=sq, kl=kl, sum_min0=sum_min0,
                ties=np.bincount(r, minlength=n))


def entity_measures(U, Z, rds):
    """Per facet and entity, over the training cells of every relation touching it (2026-11-07): reproduced share
    Σmin(x̂, x)/Σx (ties), on-target share Σmin/Σx̂ (all training cells of its rows/columns), coverage Σx̂/Σx,
    accuracy = error per unit of observed weight under squared error and under generalised KL, and the degree-only
    reference reproduced share (x̂₀ = row total × column total / total, per relation)."""
    acc = {}
    for rd in rds:
        A, B, Zr = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
        sides = [(rd.row_facet, False)] + ([(rd.col_facet, True)] if rd.col_facet != rd.row_facet else [])
        for f, tr in sides:
            s = _row_side(A, Zr, B, rd, tr)
            if f not in acc:
                acc[f] = {k: np.zeros_like(v, dtype=float) for k, v in s.items()}
            for k, v in s.items():
                acc[f][k] += v
    out = {}
    for f, a in acc.items():
        sx = np.maximum(a["sum_x"], 1e-300)
        out[f] = {"reproduced_share": a["sum_min"] / sx, "on_target_share": a["sum_min"] / np.maximum(a["sum_hat"], 1e-300),
                  "coverage": a["sum_hat"] / sx, "accuracy_squared_error": a["sq"] / sx, "accuracy_kl": a["kl"] / sx,
                  "reproduced_share_degree_only": a["sum_min0"] / sx, "ties": a["ties"]}
    return out


def placed_entities(reproduced, reproduced0):
    """Across the seeds of one group (2026-11-07): placed if mean reproduced share − degree-only share exceeds the
    entity's own seed-to-seed SD (lenient, in-sample). reproduced: list of arrays (one per seed); reproduced0: array."""
    R = np.vstack(reproduced)
    return (R.mean(0) - reproduced0) > R.std(0, ddof=1)


# ------------------------------------------------------------------------------------------------ AUC helpers
def _grouped_auc(gid, val, is_pos, n_groups):
    """Mann–Whitney AUC within each group: positions after sorting by (group, value); exactly equal values within a group
    are counted in sorted order (rare for continuous reconstructions)."""
    o = np.lexsort((val, gid))
    g, p = gid[o], is_pos[o]
    first = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(g)]))
    ranks = np.arange(len(g)) - start + 1.0
    npos = np.bincount(g, weights=p.astype(float), minlength=n_groups)
    nneg = np.bincount(g, weights=(~p).astype(float), minlength=n_groups)
    rs = np.bincount(g, weights=ranks * p, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rs - npos * (npos + 1) / 2) / (npos * nneg)
    return auc, npos, nneg


def entity_training_auc(U, Z, rd, side="row", n_empty=200, seed=0):
    """Per entity of one side of a relation (2026-11-06): probability that one of its visible ties is reconstructed above
    one of its visible empty cells; empty cells sampled (up to n_empty per entity, uniformly among its visible empty
    cells, implementation choice: rows of child×parent have 97,850 cells). Returns AUC per entity and its tie count."""
    rng = np.random.default_rng(seed)
    if side == "row":
        A, B, Zr = U[rd.row_facet] * rd.obs_row_mask[:, None], U[rd.col_facet], Z[rd.name]
        r, c, n, m = rd.tie_row, rd.tie_col, rd.shape[0], rd.shape[1]
        hr, hc = rd.hid_row, rd.hid_col; ok_ent = rd.obs_row_mask
    else:
        A, B, Zr = U[rd.col_facet], U[rd.row_facet] * rd.obs_row_mask[:, None], Z[rd.name].T
        r, c, n, m = rd.tie_col, rd.tie_row, rd.shape[1], rd.shape[0]
        hr, hc = rd.hid_col, rd.hid_row; ok_ent = np.ones(n, bool)
    ties = np.bincount(r, minlength=n)
    ents = np.nonzero((ties > 0) & ok_ent)[0]
    er = np.repeat(ents, n_empty); ec = rng.integers(0, m, len(er))
    excl = np.concatenate([r * m + c, hr * m + hc])
    keep = ~np.isin(er * m + ec, excl)
    if side == "col":
        keep &= rd.obs_row_mask[ec]
    if rd.diag_masked:
        keep &= er != ec
    er, ec = er[keep], ec[keep]
    v_t = ((A[r] @ Zr) * B[c]).sum(1); v_e = ((A[er] @ Zr) * B[ec]).sum(1)
    gid = np.concatenate([r, er]); val = np.concatenate([v_t, v_e]); pos = np.r_[np.ones(len(r), bool), np.zeros(len(er), bool)]
    auc, npos, nneg = _grouped_auc(gid, val, pos, n)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc, ties


def banded_mean(values, ties):
    out = {}
    for lo, hi, name in TIE_BANDS:
        m = (ties >= lo) & (ties <= hi) & np.isfinite(values)
        out[name] = {"mean": float(np.mean(values[m])) if m.any() else None, "n": int(m.sum())}
    return out


# --------------------------------------------------------------------------------------- held-out scores (L2.6)
def heldout(U, Z, rd):
    """Hidden ties and hidden empty cells of one relation (2026-11-01, 2026-11-03): squared error with the empty-cell
    part scaled to the group's full share of empty cells (option B), captured share = 1 − error / error of predicting 0;
    generalised KL error the same way (no zero-prediction reference); hidden-cell AUC (ties vs empty cells), overall and
    by the visible tie count of the cell's sparser entity (bands 1, 2, 3–5, 6+)."""
    A, B, Zr = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
    if len(rd.hid_tie_row) == 0:
        return None
    xt = rd.hid_tie_val
    ht = ((A[rd.hid_tie_row] @ Zr) * B[rd.hid_tie_col]).sum(1)
    he = ((A[rd.hid_empty_row] @ Zr) * B[rd.hid_empty_col]).sum(1)
    n_ties_all = len(rd.tie_row) + len(rd.hid_tie_row)
    n_obs = int(rd.obs_row_mask.sum()) * rd.shape[1] - (rd.shape[0] if rd.diag_masked else 0)
    n_empty_obs = n_obs - n_ties_all
    scale = (math.floor(0.10 * n_empty_obs) / len(he)) if len(he) else 0.0
    sq = float(((xt - ht) ** 2).sum() + scale * (he ** 2).sum())
    sq0 = float((xt ** 2).sum())
    kl = float((xt * np.log(xt / np.maximum(ht, 1e-300)) - xt + ht).sum() + scale * he.sum())
    nr = np.bincount(rd.tie_row, minlength=rd.shape[0]); nc = np.bincount(rd.tie_col, minlength=rd.shape[1])
    sparse_t = np.minimum(nr[rd.hid_tie_row], nc[rd.hid_tie_col]); sparse_e = np.minimum(nr[rd.hid_empty_row], nc[rd.hid_empty_col])
    def auc(pv, nv):
        if len(pv) == 0 or len(nv) == 0:
            return None
        rk = rankdata(np.concatenate([pv, nv]))
        return float((rk[:len(pv)].sum() - len(pv) * (len(pv) + 1) / 2) / (len(pv) * len(nv)))
    def ap(pv, nv, w_neg):
        # average precision (S0-M2, 2026-11-05), the sampled empty cells each counted w_neg times so that precision
        # refers to all of the group's empty cells: mean over hidden ties of ties ranked at or above / (ties + w_neg ×
        # empty cells ranked at or above); exact ties of prediction count as ranked above (pessimistic)
        if len(pv) == 0:
            return None
        nv_sorted = np.sort(nv)
        n_neg_ge = len(nv) - np.searchsorted(nv_sorted, pv, side="left")
        pv_sorted = np.sort(pv)
        n_pos_ge = len(pv) - np.searchsorted(pv_sorted, pv, side="left")
        return float(np.mean(n_pos_ge / (n_pos_ge + w_neg * n_neg_ge)))
    from scipy.stats import spearmanr
    rho = float(spearmanr(xt, ht).correlation) if len(xt) > 2 and np.ptp(xt) > 0 else None
    bands = {}
    for lo, hi, name in TIE_BANDS:
        mt = (sparse_t >= lo) & (sparse_t <= hi); me = (sparse_e >= lo) & (sparse_e <= hi)
        bands[name] = {"auc": auc(ht[mt], he[me]), "hidden_ties": int(mt.sum())}
    return {"squared_error": sq, "squared_error_zero_prediction": sq0, "captured_share": 1 - sq / sq0 if sq0 > 0 else None,
            "kl_error": kl, "auc": auc(ht, he), "auc_by_band": bands, "empty_scale": scale,
            "average_precision": ap(ht, he, scale), "spearman_hidden_ties": rho,
            "top01_share_empty_error": (float(np.sort(he ** 2)[::-1][:max(1, len(he) // 1000)].sum() / (he ** 2).sum())
                                        if len(he) and (he ** 2).sum() > 0 else None),
            "max_overprediction_ratio": float(np.max(ht / xt)),
            "hidden_ties": int(len(xt)), "hidden_empty": int(len(he))}


def combined_heldout(hos, weights, key="squared_error"):
    """Σ_r w_r × hidden error of r over the relations with hidden cells (2026-11-03: the training loss's own form)."""
    return float(sum(weights[r] * h[key] for r, h in hos.items() if h is not None))


# -------------------------------------------------------------------------------------------- tie alignment (step 9)
def tie_alignment(rd, P_row, P_col, placed_row=None, placed_col=None, shuffle_side="col", strata_extra=None,
                  n_shuffles=199, seed=0):
    """A_r = Σ X[i,j] Σ_k m_i(k) m_j(k) / Σ X over visible ties with both ends placed (2026-11-07/08); per community the
    k-term. Reference: memberships of one side shuffled within degree band (× W stratum for child hyperedges, passed as
    strata_extra), reported as (observed − shuffled mean)/(1 − shuffled mean). Also per band/stratum of the shuffled
    side and with per-entity weighting (each entity's ties averaged first)."""
    rng = np.random.default_rng(seed)
    r, c, x = rd.tie_row, rd.tie_col, rd.tie_val
    keep = np.ones(len(r), bool)
    if placed_row is not None: keep &= placed_row[r]
    if placed_col is not None: keep &= placed_col[c]
    r, c, x = r[keep], c[keep], x[keep]
    deg_side = np.bincount(c if shuffle_side == "col" else r, minlength=(P_col if shuffle_side == "col" else P_row).shape[0])
    def score(Pr, Pc):
        a = (Pr[r] * Pc[c]).sum(1)
        return a
    a_obs = score(P_row, P_col)
    obs = float((x * a_obs).sum() / x.sum())
    per_k = (x[:, None] * P_row[r] * P_col[c]).sum(0) / x.sum()
    Ps = P_col if shuffle_side == "col" else P_row
    band = np.array([band_of(n) for n in deg_side])
    strat = band if strata_extra is None else np.char.add(band.astype(str), np.char.add("|", strata_extra.astype(str)))
    groups = [np.nonzero(strat == s_)[0] for s_ in np.unique(strat)]
    shuf = []
    for _ in range(n_shuffles):
        perm = np.arange(len(Ps))
        for g in groups:
            perm[g] = g[rng.permutation(len(g))]
        a_s = score(P_row, Ps[perm]) if shuffle_side == "col" else score(Ps[perm], P_col)
        shuf.append(float((x * a_s).sum() / x.sum()))
    m_s = float(np.mean(shuf))
    side_ent = c if shuffle_side == "col" else r
    by = {}
    for s_ in np.unique(strat[side_ent]):
        mk = strat[side_ent] == s_
        by[str(s_)] = {"observed": float((x[mk] * a_obs[mk]).sum() / x[mk].sum()), "ties": int(mk.sum())}
    ent_mean = np.bincount(side_ent, weights=x * a_obs, minlength=len(Ps)) / np.maximum(np.bincount(side_ent, weights=x, minlength=len(Ps)), 1e-300)
    has = np.bincount(side_ent, minlength=len(Ps)) > 0
    return {"observed": obs, "shuffled_mean": m_s, "shuffled_95": float(np.percentile(shuf, 95)),
            "adjusted": (obs - m_s) / (1 - m_s) if m_s < 1 else None, "per_community": per_k,
            "by_stratum": by, "per_entity_weighting": float(ent_mean[has].mean()), "ties_used": int(len(x))}


# ---------------------------------------------------------------------------------- spread diagnostics (steps 4, 5)
def floor_diagnostics(rd_child_article, wm, W=None, beta_w=0.0, cutoffs=(0.5, 0.7, 0.9)):
    """Per child hyperedge with ≥ 2 articles (2026-11-07): own spread H(p_h), the articles' mix spread split into
    within- and between-article parts (bits), shortfall below γ(between + ω within) for γ = 0.8, ω ∈ {0, 1};
    shares with largest membership above 0.5/0.7/0.9 (sensitivity readings, 2026-11-06 (c))."""
    from .terms import article_mixing
    q, n_art = article_mixing(rd_child_article, W, beta_w)
    keep = n_art >= 2
    m = wm[rd_child_article.row_facet]["P"]; p = wm[rd_child_article.col_facet]["P"][keep]
    qk = q[keep]
    within = qk @ entropy_bits(m)
    between = np.maximum(entropy_bits(qk @ m) - within, 0.0)
    own = entropy_bits(p)
    largest = p.max(1)
    return {"n": int(keep.sum()), "own_spread": own, "between": between, "within": within,
            "shortfall_omega0": np.maximum(0, 0.8 * between - own), "shortfall_omega1": np.maximum(0, 0.8 * (between + within) - own),
            "share_largest_above": {str(t): float((largest > t).mean()) for t in cutoffs}}


def excess_spread(rds, wm, facet, rho=1.0):
    """Own spread minus ρ × the largest relation-mix spread over the entity's relations (2026-11-07, step 5)."""
    import scipy.sparse as sp
    best = np.full(wm[facet]["P"].shape[0], -np.inf)
    for rd in rds:
        for side in ("row", "col"):
            f = rd.row_facet if side == "row" else rd.col_facet
            if f != facet or (side == "col" and rd.row_facet == rd.col_facet):
                continue
            M = sp.csr_matrix((rd.tie_val, (rd.tie_row, rd.tie_col)), shape=rd.shape)
            if side == "col": M = M.T.tocsr()
            partner = rd.col_facet if side == "row" else rd.row_facet
            s = np.asarray(M.sum(1)).ravel(); has = s > 0
            mix = (M[has] @ wm[partner]["P"]) / s[has][:, None]
            v = np.full(len(best), -np.inf); v[has] = entropy_bits(mix)
            best = np.maximum(best, v)
    ok = np.isfinite(best)
    exc = np.full(len(best), np.nan); exc[ok] = entropy_bits(wm[facet]["P"][ok]) - rho * best[ok]
    return exc


# ------------------------------------------------------------------- membership side of a community (2026-11-07)
def membership_reading(wm, placed=None, q=0.5):
    """Per facet and community: effective number of members 2^spread of the entities' memberships in k rescaled to sum
    to 1 (no cut-off), number of entities whose largest membership is k, and (sensitivity) number with membership ≥ q."""
    out = {}
    for f, d in wm.items():
        P = d["P"] if placed is None else d["P"][placed[f]]
        if len(P) == 0:
            continue
        col = P / np.maximum(P.sum(0, keepdims=True), 1e-300)
        out[f] = {"effective_members": 2 ** entropy_bits(col.T), "largest_is_k": np.bincount(P.argmax(1), minlength=P.shape[1]),
                  f"members_ge_{q}": (P >= q).sum(0)}
    return out


def loading_distributions(wm, qs=(0.01, 0.1, 0.5, 0.9, 0.99)):
    """Item (l) input (2026-11-06): per facet, quantiles of weighted mass and of the largest membership."""
    return {f: {"mass": np.quantile(d["mass"], qs).tolist(), "largest": np.quantile(d["P"].max(1), qs).tolist()}
            for f, d in wm.items()}


# --------------------------------------------------------------------------- ghosts (step 8; across seeds later)
def ghosts(wm, placed):
    """Communities with no placed entity whose largest weighted membership is k, in any facet (2026-11-07)."""
    K = next(iter(wm.values()))["P"].shape[1]
    hit = np.zeros(K, bool)
    for f, d in wm.items():
        m = placed[f] if placed is not None else np.ones(len(d["P"]), bool)
        if m.any():
            hit[np.unique(d["P"][m].argmax(1))] = True
    return np.nonzero(~hit)[0].tolist()


# ------------------------------------------------------------------------------- Track A / B (CLAUDE.md §4.24, C1)
def _js_distance_e(P, Q):
    M = 0.5 * (P + Q)
    kl = lambda A, B: (A * (np.log(np.clip(A, 1e-300, None)) - np.log(np.clip(B, 1e-300, None)))).sum(1)
    return np.sqrt(np.maximum(0.5 * kl(P, M) + 0.5 * kl(Q, M), 0.0))      # natural log, as the toy (≈0.17 floor)


def match(wm1, wm2, facets):
    """Hungarian matching of fit 2's communities to fit 1's on the weighted columns stacked over facets (cosine)."""
    nz = lambda Y: Y / np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-300)
    A = nz(np.vstack([wm1[f]["Y"] for f in facets])); B = nz(np.vstack([wm2[f]["Y"] for f in facets]))
    r, c = linear_sum_assignment(-(A.T @ B))
    perm = np.empty(len(r), int); perm[r] = c
    return perm                                                # column k of fit 1 ↔ column perm[k] of fit 2


def track_pair(wm1, wm2, placed=None, facets=None):
    """Track A (1 − mean JS distance over placed entities, unweighted) and Track B (cosine of weighted columns' rows,
    weighted by fit 1's mass, all entities) per facet on weighted memberships (owner C1, 2026-11-09); model level = mean
    over facets. The chance reference is the same comparison against fit 2 with its rows shuffled (stability())."""
    facets = facets or sorted(wm1)
    perm = match(wm1, wm2, facets)
    A, B = {}, {}
    for f in facets:
        P1, P2 = wm1[f]["P"], wm2[f]["P"][:, perm]
        m = np.ones(len(P1), bool) if placed is None else placed[f]
        A[f] = float(1 - _js_distance_e(P1[m], P2[m]).mean()) if m.any() else np.nan
        Y1, Y2 = wm1[f]["Y"], wm2[f]["Y"][:, perm]
        cos = (Y1 * Y2).sum(1) / np.maximum(np.linalg.norm(Y1, axis=1) * np.linalg.norm(Y2, axis=1), 1e-300)
        w = wm1[f]["mass"]
        B[f] = float((w * cos).sum() / w.sum()) if w.sum() > 0 else np.nan
    return {"A": A, "B": B, "A_model": float(np.nanmean(list(A.values()))), "B_model": float(np.nanmean(list(B.values())))}


def _shuffled(wm, rng):
    out = {}
    for f, d in wm.items():
        p = rng.permutation(len(d["P"]))
        out[f] = {"P": d["P"][p], "Y": d["Y"][p], "mass": d["mass"][p], "w": d["w"]}
    return out


def stability(wms, placed=None, seed=0):
    """All seed pairs of one group: real and chance Track A/B (chance = fit 2 rows shuffled per facet); per seed: mean of
    its pairs and the maximum chance over its pairs; a seed passes if both mean tracks exceed their chance maxima (toy
    rule, applied per seed, 2026-11-09 (E)); chance-adjusted (real − chance)/(1 − chance)."""
    rng = np.random.default_rng(seed)
    n = len(wms)
    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            real = track_pair(wms[i], wms[j], placed)
            ch = track_pair(wms[i], _shuffled(wms[j], rng), placed)
            pairs[(i, j)] = (real, ch)
    per_seed = []
    for s_ in range(n):
        mine = [v for (i, j), v in pairs.items() if s_ in (i, j)]
        ra = np.mean([v[0]["A_model"] for v in mine]); rb = np.mean([v[0]["B_model"] for v in mine])
        ca = max(v[1]["A_model"] for v in mine); cb = max(v[1]["B_model"] for v in mine)
        per_seed.append({"A": float(ra), "B": float(rb), "chance_max_A": float(ca), "chance_max_B": float(cb),
                         "passes": bool(ra > ca and rb > cb)})
    adj = lambda key: float(np.mean([(v[0][key] - v[1][key]) / (1 - v[1][key]) for v in pairs.values()]))
    return {"per_seed": per_seed, "A_mean": float(np.mean([v[0]["A_model"] for v in pairs.values()])),
            "B_mean": float(np.mean([v[0]["B_model"] for v in pairs.values()])),
            "A_adjusted": adj("A_model"), "B_adjusted": adj("B_model"),
            "per_facet_B": {f: float(np.mean([v[0]["B"][f] for v in pairs.values()])) for f in wms[0]}}


# ------------------------------------------------------------------------------------- counting rule (step 11)
def count_rule(fails, fraction=0.2):
    """fails: groups × seeds boolean array (True = the seed fails the check; NaN-free). A group fails if more than
    `fraction` of its seeds fail; K fails if more than `fraction` of its groups fail or more than `fraction` of all
    fits fail (2026-11-09, A–C)."""
    F = np.asarray(fails, bool)
    g_fail = F.mean(1) > fraction
    return {"group_fails": g_fail.tolist(), "K_fails": bool(g_fail.mean() > fraction or F.mean() > fraction),
            "by_groups": bool(g_fail.mean() > fraction), "pooled": bool(F.mean() > fraction),
            "groups_with_any_failure": [int(i) for i in np.nonzero(F.any(1))[0]], "failures_per_group": F.sum(1).tolist()}


# ------------------------------------------------------------------- two-part held-out criterion (2026-11-13, option D)
def _log2_band(n):
    """Tie-count band: 0, 1, 2–3, 4–7, 8–15, … (floor(log2(n + 1)))."""
    return np.floor(np.log2(np.asarray(n, float) + 1)).astype(np.int64)


def _weighted_ap(pv, nv, w_neg):
    """Average precision with each empty cell weighted (w_neg array or scalar); equal scores count the empty cell above
    the tie (pessimistic, as threshold-based AP)."""
    if len(pv) == 0:
        return None
    w_neg = np.broadcast_to(np.asarray(w_neg, float), nv.shape)
    o = np.argsort(nv); nv_s = nv[o]; cw = np.concatenate([[0.0], np.cumsum(w_neg[o])])
    neg_ge = cw[-1] - cw[np.searchsorted(nv_s, pv, side="left")]
    pv_s = np.sort(pv)
    pos_ge = len(pv) - np.searchsorted(pv_s, pv, side="left")
    return float(np.mean(pos_ge / (pos_ge + neg_ge)))


def _auc(pv, nv):
    if len(pv) == 0 or len(nv) == 0:
        return None
    rk = rankdata(np.concatenate([pv, nv]))
    return float((rk[:len(pv)].sum() - len(pv) * (len(pv) + 1) / 2) / (len(pv) * len(nv)))


def degree_correction_weights(rd, min_empty=20):
    """Weights for the sampled hidden empty cells so that their distribution over joint tie-count bands (row entity's
    band × column entity's band, visible ties, _log2_band) matches the hidden ties' (degree-corrected negatives, after
    Aiyappa et al. 2025); total weight = the group's full share of empty cells (as the uncorrected AP). Joint bands
    with fewer than min_empty sampled empty cells are left out (their hidden ties too; share reported).
    Returns (weights per empty cell, effective sample size, share of hidden ties left out, mask of hidden ties kept)."""
    nr = np.bincount(rd.tie_row, minlength=rd.shape[0]); nc = np.bincount(rd.tie_col, minlength=rd.shape[1])
    br_t, bc_t = _log2_band(nr[rd.hid_tie_row]), _log2_band(nc[rd.hid_tie_col])
    br_e, bc_e = _log2_band(nr[rd.hid_empty_row]), _log2_band(nc[rd.hid_empty_col])
    hc = int(max(bc_t.max(), bc_e.max())) + 1
    lt, le = br_t * hc + bc_t, br_e * hc + bc_e
    L = int(max(lt.max(), le.max())) + 1
    Ec = np.bincount(le, minlength=L).astype(float); Tc = np.bincount(lt, minlength=L).astype(float)
    ok = Ec >= min_empty
    keep_t = ok[lt]
    n_empty_obs = int(rd.obs_row_mask.sum()) * rd.shape[1] - (rd.shape[0] if rd.diag_masked else 0) - len(rd.tie_row) - len(rd.hid_tie_row)
    total = math.floor(0.10 * n_empty_obs)
    Tk = Tc * ok
    wb = np.zeros(L)
    wb[ok] = total * (Tk[ok] / Tk.sum()) / Ec[ok]
    w = wb[le]
    ess = float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
    return w, ess, float(1 - keep_t.mean()), keep_t


def within_class_ap(pt, pe, rd, scale):
    """Existence within tie-count classes (2026-11-13): hidden cells grouped by (row entity's visible ties band × column
    entity's visible ties band), bands 1, 2–3, 4–7, … (_log2_band); within each class the AP of its hidden ties against
    its sampled hidden empty cells, each empty cell counted `scale` times (the class's real share of empty cells); the
    classes' APs averaged with weights = their numbers of hidden ties. Classes without a sampled empty cell are left out.
    Returns (AP, share of hidden ties left out, chance level = the same average of the classes' tie shares)."""
    nr = np.bincount(rd.tie_row, minlength=rd.shape[0]); nc = np.bincount(rd.tie_col, minlength=rd.shape[1])
    hc = 64
    lt = _log2_band(nr[rd.hid_tie_row]) * hc + _log2_band(nc[rd.hid_tie_col])
    le = _log2_band(nr[rd.hid_empty_row]) * hc + _log2_band(nc[rd.hid_empty_col])
    num = den = chance = 0.0; left = 0
    for c in np.unique(lt):
        mt, me = lt == c, le == c
        if not me.any():
            left += int(mt.sum()); continue
        n = int(mt.sum())
        num += n * _weighted_ap(pt[mt], pe[me], scale); den += n
        chance += n * n / (n + scale * me.sum())
    return (num / den if den else None), left / len(lt), (chance / den if den else None)


def partial_spearman(x, p, covariates):
    """Partial Spearman correlation: ranks of x and p each regressed (least squares) on the covariates' ranks; Pearson
    correlation of the residuals; 0 when either residual has no variation."""
    X = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covariates])
    def res(v):
        r = rankdata(v); b = np.linalg.lstsq(X, r, rcond=None)[0]; return r - X @ b
    ex, ep = res(x), res(p)
    if ex.std() < 1e-12 or ep.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ex, ep)[0, 1])


def heldout_two_part(U, Z, rd, value=True):
    """Existence (degree-corrected AP; reported: uncorrected AP, AUC, degree-only baseline) and value (partial Spearman
    among hidden ties, controlling for the two entities' visible tie counts and their product) of one relation."""
    A, B, Zr = U[rd.row_facet], U[rd.col_facet], Z[rd.name]
    if len(rd.hid_tie_row) == 0:
        return None
    xt = rd.hid_tie_val
    ht = ((A[rd.hid_tie_row] @ Zr) * B[rd.hid_tie_col]).sum(1)
    he = ((A[rd.hid_empty_row] @ Zr) * B[rd.hid_empty_col]).sum(1)
    nr = np.bincount(rd.tie_row, minlength=rd.shape[0]); nc = np.bincount(rd.tie_col, minlength=rd.shape[1])
    dt = nr[rd.hid_tie_row].astype(float) * nc[rd.hid_tie_col]; de = nr[rd.hid_empty_row].astype(float) * nc[rd.hid_empty_col]
    n_empty_obs = int(rd.obs_row_mask.sum()) * rd.shape[1] - (rd.shape[0] if rd.diag_masked else 0) - len(rd.tie_row) - len(rd.hid_tie_row)
    scale = math.floor(0.10 * n_empty_obs) / len(he)
    w, ess, unc, cov = degree_correction_weights(rd)
    out = {"ap_corrected": _weighted_ap(ht[cov], he, w), "ap": _weighted_ap(ht, he, scale), "auc": _auc(ht, he),
           "baseline_ap_corrected": _weighted_ap(dt[cov], de, w), "baseline_ap": _weighted_ap(dt, de, scale),
           "baseline_auc": _auc(dt, de), "prevalence": len(xt) / (len(xt) + math.floor(0.10 * n_empty_obs)),
           "correction_ess": ess, "ties_without_matching_empty": unc}
    wc, wc_left, wc_chance = within_class_ap(ht, he, rd, scale)
    out.update({"ap_within_class": wc, "ap_within_class_left_out": wc_left, "ap_within_class_chance": wc_chance,
                "baseline_ap_within_class": within_class_ap(dt, de, rd, scale)[0]})
    if value:
        cr, cc = nr[rd.hid_tie_row], nc[rd.hid_tie_col]
        out["value_partial_spearman"] = partial_spearman(xt, ht, [cr, cc, cr.astype(float) * cc])
        from scipy.stats import spearmanr
        out["value_spearman"] = float(spearmanr(xt, ht).correlation)
        out["baseline_value_partial_spearman"] = partial_spearman(xt, dt, [cr, cc, cr.astype(float) * cc])
    return out


SEMANTIC_HIDDEN = ("child_article", "parent_child")


def combined_two_part(parts, weights):
    """Combined two-part held-out score of one fit (2026-11-13, decided): existence = Σ w̃_r × within-class AP over the
    relations with hidden cells, value = Σ w̃_r × partial Spearman over child×article and child×parent, w̃ = the fit's
    relation weights rescaled to sum to 1 over the relations entering each part (A0: existence 0.375/0.375/0.25, value
    0.5/0.5); the semantic-equal versions (child×article and child×parent at 0.5 each) are the primary score across
    approaches and arms (2026-11-02). Higher is better for both."""
    def avg(key, rels, w):
        rels = [r for r in rels if parts.get(r) is not None and parts[r].get(key) is not None]
        tot = sum(w[r] for r in rels)
        return float(sum(w[r] * parts[r][key] for r in rels) / tot) if tot > 0 else None
    hid = [r for r in parts if parts[r] is not None]
    eq = {r: 1.0 for r in SEMANTIC_HIDDEN}
    return {"existence": avg("ap_within_class", hid, weights),
            "value": avg("value_partial_spearman", SEMANTIC_HIDDEN, weights),
            "existence_semantic_equal": avg("ap_within_class", SEMANTIC_HIDDEN, eq),
            "value_semantic_equal": avg("value_partial_spearman", SEMANTIC_HIDDEN, eq)}
