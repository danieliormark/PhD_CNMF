# F0 — data for the M1 fits (ticket 107). Every rule cites its decision in plans/ticket107-m1-chunk12v3-design.md.
#
# Reads a chunk12v3 build (psych_trial_v3: the weighted matrices plus the raw counts saved for this module), draws and
# saves the hold-out groups once, and assembles, for one fit, each active relation matrix with
#   - the per-fold weights recomputed from the visible cells (2026-11-03): child×article idf_article + partial row
#     normalisation; child×parent idf_parent + partial row normalisation; author×affiliation / article×affiliation row
#     normalisation; nothing for matrices without hidden cells; hidden ties' targets at the fold's own weights;
#   - the whole-matrix (Frobenius) normalisation over the visible observed cells (L2.1, 2026-11-02; 2026-10-31);
#   - the cells left out of the loss: missing rows (unobserved), the article×article diagonal, the hidden ties and the
#     hidden empty cells of the fold;
#   - the relation weight w_r = 0.5 / n_d (L2.2, 2026-11-02), with the journal multiplier and the A1 split as arms.
# Hold-out (2026-10-29, 2026-11-01, 2026-11-05): ties eligible if both entities have >= 2 ties in that matrix; 10 groups;
# in one group an entity loses at most max(1, floor(n/3)) of its n ties and never its last one; empty cells 10% of the
# matrix's observed empty cells per group, at most 1 million, drawn uniformly, disjoint across groups; hidden cells only
# in child×article, child×parent, author×affiliation, article×affiliation (author×article, article×article and
# article×journal: none, 2026-11-01). The semantic groups are drawn once and shared by both approaches (2026-11-05).
import json, math, os
import numpy as np
import scipy.sparse as sp

BUILD = "/mnt/hum01-rds/Basov/p91688di/m1_chunk12v3/psych_trial_v3/"
G = 10
EMPTY_SHARE, EMPTY_CAP = 0.10, 1_000_000
HOLDOUT_SEED = 107
COUSIN = 2 ** (2 ** -0.5) - 1          # as in chunk12v3 (2026-10-31)

FACETS = ("article", "child", "parent", "author", "organisation", "journal")
FACET_INDEX = {"article": "articles", "child": "children", "parent": "parents", "author": "persons",
               "organisation": "organisations", "journal": "journals"}

# name: (file, row facet, column facet, domain, hidden cells?, missing-rows key, diagonal masked?)
RELATIONS = {
    "child_article":       ("child_article",          "article", "child",        "semantic", True,  None, False),
    "parent_child":        ("parent_child",           "parent",  "child",        "semantic", True,  None, False),
    "author_article":      ("P1_author_article",      "author",  "article",      "social",   False, None, False),
    "author_affiliation":  ("P2_author_affiliation",  "author",  "organisation", "social",   True,  "P2_author_affiliation_rows", False),
    "article_affiliation": ("P3_article_affiliation", "article", "organisation", "social",   True,  "P3_article_affiliation_rows", False),
    "article_journal":     ("P4_article_journal",     "article", "journal",      "social",   False, "P4_article_journal_rows", False),
    "article_article":     ("P5_article_article",     "article", "article",      "social",   False, None, True),
}
APPROACH = {1: ["parent_child", "child_article", "author_article", "author_affiliation", "article_journal"],
            2: ["parent_child", "child_article", "article_affiliation", "article_article", "article_journal"]}


# ----------------------------------------------------------------------------------------------------- relation weights
def relation_weights(relations, journal_multiplier=1.0, semantic_split=None):
    """w_r = 0.5 / n_d within each domain (2026-11-02); the journal multiplier acts inside the social half, which is then
    rescaled to 0.5 (option i, 2026-11-02); semantic_split = (child×parent, child×article) shares of the semantic half
    (A0: equal; A1: (0.7, 0.3), 2026-11-02)."""
    sem = [r for r in relations if RELATIONS[r][3] == "semantic"]
    soc = [r for r in relations if RELATIONS[r][3] == "social"]
    w = {}
    if semantic_split is None:
        w.update({r: 0.5 / len(sem) for r in sem})
    else:
        split = dict(zip(("parent_child", "child_article"), semantic_split))
        assert set(sem) == set(split) and abs(sum(semantic_split) - 1) < 1e-12
        w.update({r: 0.5 * split[r] for r in sem})
    raw = {r: (journal_multiplier if r == "article_journal" else 1.0) for r in soc}
    tot = sum(raw.values())
    w.update({r: 0.5 * v / tot for r, v in raw.items()})
    return w


# ----------------------------------------------------------------------------------------------------------- build I/O
class Build:
    def __init__(self, path=BUILD):
        self.path = path if path.endswith("/") else path + "/"
        self.index = json.load(open(self.path + "index.json"))
        self.missing = json.load(open(self.path + "missing.json"))
        self.sizes = {f: len(self.index[FACET_INDEX[f]]) for f in FACETS}
        self._cache = {}

    def load(self, name):
        if name not in self._cache:
            self._cache[name] = sp.load_npz(self.path + name + ".npz").tocsr()
        return self._cache[name]

    def missing_rows(self, rel):
        key = RELATIONS[rel][5]
        return np.asarray(self.missing.get(key, []), dtype=np.int64) if key else np.zeros(0, dtype=np.int64)

    def ties(self, rel):
        """Rows, columns and saved values of the matrix's ties (article×article: both triangles)."""
        M = self.load(RELATIONS[rel][0]).tocoo()
        o = np.lexsort((M.col, M.row))
        return M.row[o].astype(np.int64), M.col[o].astype(np.int64), M.data[o]


# ------------------------------------------------------------------------------------------------------------- hold-out
def _draw_tie_groups(r, c, nr, nc, rng):
    """Assign eligible ties to one of G groups under the cap; returns the group per tie (-1 = never hidden).
    Same procedure as holdout_measure.py (2026-11-05)."""
    cap_r = np.maximum(1, nr // 3); cap_c = np.maximum(1, nc // 3)
    lim_r = np.minimum(cap_r, nr - 1); lim_c = np.minimum(cap_c, nc - 1)
    used_r = np.zeros((G, len(nr)), dtype=np.int32); used_c = np.zeros((G, len(nc)), dtype=np.int32)
    out = np.full(len(r), -1, dtype=np.int64)
    for t in rng.permutation(len(r)):
        i, j = r[t], c[t]
        ok = np.nonzero((used_r[:, i] < lim_r[i]) & (used_c[:, j] < lim_c[j]))[0]
        if len(ok):
            g = rng.choice(ok); out[t] = g; used_r[g, i] += 1; used_c[g, j] += 1
    return out


def _draw_empty_groups(shape, tie_rows, tie_cols, missing_rows, rng):
    """G disjoint groups of empty observed cells, each min(cap, floor(10% of the empty observed cells)), drawn uniformly
    without replacement: the empty cells are ranked 0..n_empty-1 in row-major order over the observed rows, a uniform
    sample of ranks is drawn, and each rank is mapped back to its cell (exact; no rejection)."""
    n, m = shape
    obs_rows = np.setdiff1d(np.arange(n), missing_rows)
    pos_of_row = np.full(n, -1, dtype=np.int64); pos_of_row[obs_rows] = np.arange(len(obs_rows))
    t = np.sort(pos_of_row[tie_rows] * m + tie_cols)          # tie positions among the observed cells
    n_empty = len(obs_rows) * m - len(t)
    per = int(min(EMPTY_CAP, math.floor(EMPTY_SHARE * n_empty)))
    ranks = rng.choice(n_empty, G * per, replace=False)
    a = t - np.arange(len(t))                                 # empty cells before each tie
    q = ranks + np.searchsorted(a, ranks, side="right")       # observed-cell position of each sampled empty rank
    return obs_rows[q // m], q % m, np.repeat(np.arange(G), per), n_empty, per


def draw_holdout(build, seed=HOLDOUT_SEED, out_dir=None):
    """Draw the hold-out groups once and save them under <build>/holdout/ (never overwritten)."""
    out_dir = out_dir or build.path + "holdout/"
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    rng = np.random.default_rng(seed)
    man = {"seed": seed, "groups": G, "empty_share": EMPTY_SHARE, "empty_cap": EMPTY_CAP, "relations": {}}
    for rel, spec in RELATIONS.items():
        if not spec[4]:
            continue
        r, c, _ = build.ties(rel)
        shape = build.load(spec[0]).shape
        nr = np.bincount(r, minlength=shape[0]); nc = np.bincount(c, minlength=shape[1])
        elig = (nr[r] >= 2) & (nc[c] >= 2)
        grp = np.full(len(r), -1, dtype=np.int64)
        grp[elig] = _draw_tie_groups(r[elig], c[elig], nr, nc, rng)
        er, ec, eg, n_empty, per = _draw_empty_groups(shape, r, c, build.missing_rows(rel), rng)
        np.savez_compressed(out_dir + rel + ".npz", tie_row=r, tie_col=c, tie_group=grp,
                            empty_row=er, empty_col=ec, empty_group=eg)
        man["relations"][rel] = {"ties": int(len(r)), "eligible": int(elig.sum()), "placed": int((grp >= 0).sum()),
                                 "ties_per_group": np.bincount(grp[grp >= 0], minlength=G).tolist(),
                                 "observed_empty_cells": int(n_empty), "empty_per_group": per}
    json.dump(man, open(out_dir + "manifest.json", "w"), indent=1)
    return man


def load_holdout(build, rel):
    h = np.load(build.path + "holdout/" + rel + ".npz")
    return {k: h[k] for k in h.files}


# --------------------------------------------------------------------------------------------- per-fold weights (2026-11-03)
def _partial_row_normalise_values(rows, vals, vis, n_rows, power):
    """Divide each tie by f(visible row norm): power 0.5 → row / ||row||^½ (partial), 1 → row / ||row||."""
    norm2 = np.bincount(rows[vis], weights=vals[vis] ** 2, minlength=n_rows)
    f = np.zeros(n_rows)
    pos = norm2 > 0
    f[pos] = norm2[pos] ** (-0.5 * power)
    return vals * f[rows]


def fold_values(build, rel, rows, cols, vis):
    """Weighted values of all ties of `rel` (visible and hidden), with the statistics taken from the visible ties only.
    rows/cols: the ties in Build.ties order; vis: boolean mask of visible ties."""
    shape = build.load(RELATIONS[rel][0]).shape
    if rel == "child_article":
        tf = np.asarray(build.load("raw_child_article_tf")[rows, cols]).ravel()
        df = np.bincount(cols[vis], minlength=shape[1])
        idf = np.log(shape[0] / (df + 1.0)) + 1.0                   # N = all trial articles (as the builder)
        v = np.log1p(tf) * idf[cols]
        return _partial_row_normalise_values(rows, v, vis, shape[0], 0.5)
    if rel == "parent_child":
        nest = np.asarray(build.load("raw_parent_child_nested")[rows, cols]).ravel()
        cous = np.asarray(build.load("raw_parent_child_cousin")[rows, cols]).ravel()
        df = np.bincount(cols[vis], minlength=shape[1])
        idf = np.where(df > 0, np.log(shape[0] / (df + 1.0)) + 1.0, 0.0)
        v = idf[cols] * np.log1p(nest + COUSIN * cous)
        return _partial_row_normalise_values(rows, v, vis, shape[0], 0.5)
    if rel in ("author_affiliation", "article_affiliation"):
        raw = build.load("raw_P2_author_affiliation" if rel == "author_affiliation" else "raw_P3_article_affiliation")
        v = np.asarray(raw[rows, cols]).ravel()
        return _partial_row_normalise_values(rows, v, vis, shape[0], 1.0)
    return np.asarray(build.load(RELATIONS[rel][0])[rows, cols]).ravel()


# ------------------------------------------------------------------------------------------------- one fit's relations
class RelationData:
    """One relation matrix as the loss sees it (numpy; the loss converts to tensors).
    Loss cells = observed rows × all columns, minus the diagonal (same-facet relation), minus the hidden cells."""
    def __init__(self, name, row_facet, col_facet, shape, weight, domain, tie_row, tie_col, tie_val,
                 obs_row_mask, diag_masked, hid_tie_row, hid_tie_col, hid_tie_val, hid_empty_row, hid_empty_col, fro):
        self.name, self.row_facet, self.col_facet, self.shape = name, row_facet, col_facet, shape
        self.weight, self.domain, self.diag_masked, self.fro = weight, domain, diag_masked, fro
        self.tie_row, self.tie_col, self.tie_val = tie_row, tie_col, tie_val
        self.obs_row_mask = obs_row_mask
        self.hid_tie_row, self.hid_tie_col, self.hid_tie_val = hid_tie_row, hid_tie_col, hid_tie_val
        self.hid_empty_row, self.hid_empty_col = hid_empty_row, hid_empty_col

    @property
    def hid_row(self): return np.concatenate([self.hid_tie_row, self.hid_empty_row])
    @property
    def hid_col(self): return np.concatenate([self.hid_tie_col, self.hid_empty_col])


def fit_data(build, approach, group=None, journal_multiplier=1.0, semantic_split=None, drop=(), use_journal=True):
    """The relations of one fit. group: hold-out group 0..9 (None = final refit on all cells, 2026-11-03).
    use_journal=False is arm J1 (no article×journal); journal_multiplier gives J3a/J3b; drop removes relations (arm N)."""
    rels = [r for r in APPROACH[approach] if r not in drop and (use_journal or r != "article_journal")]
    w = relation_weights(rels, journal_multiplier, semantic_split)
    out = []
    for rel in rels:
        f, rf, cf, dom, hidden, _, diag = RELATIONS[rel]
        shape = build.load(f).shape
        r, c, _ = build.ties(rel)
        if hidden and group is not None:
            h = load_holdout(build, rel)
            assert np.array_equal(h["tie_row"], r) and np.array_equal(h["tie_col"], c), "hold-out file does not match build"
            hid = h["tie_group"] == group
            em = h["empty_group"] == group
            er, ec = h["empty_row"][em], h["empty_col"][em]
        else:
            hid = np.zeros(len(r), dtype=bool); er = ec = np.zeros(0, dtype=np.int64)
        vis = ~hid
        v = fold_values(build, rel, r, c, vis)
        obs = np.ones(shape[0], dtype=bool); obs[build.missing_rows(rel)] = False
        assert obs[r].all(), "a tie in a missing row"
        if diag:
            assert not np.any(r == c), "a tie on the masked diagonal"
        fro = float(np.sqrt(np.sum(v[vis] ** 2)))                 # visible observed cells (zeros add nothing)
        v = v / fro
        out.append(RelationData(rel, rf, cf, shape, w[rel], dom, r[vis], c[vis], v[vis], obs, diag,
                                r[hid], c[hid], v[hid], er, ec, fro))
    return out
