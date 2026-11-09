# chunk12v3: the M1 matrix builder for the Psychology trial (ticket 107). Successor of the toy builder chunk12v2.py.
# Every rule below is an owner decision recorded in plans/ticket107-m1-chunk12v3-design.md (dates in brackets).
#
# Semantic side (approaches 1 and 2):
#   child hyperedge  = children and cousins merged by word set, role left out (§1, 2026-10-13); identity built from the
#                      non-number atoms (2026-10-23), focal model atoms mapped by model_name_map.canonical (2026-11-01);
#                      a focal atom that is not a model in its article is removed; a child hyperedge left empty is removed;
#                      a sentence unit left with no focal atom is removed; an article left with no unit is removed.
#   parent hyperedge = the multiset of its children's identities (G3's content key rebuilt from the new child keys).
#   child×article    = log(1 + tf) · idf_article, idf_article = ln(N_articles / (df + 1)) + 1 (2026-10-30); then each
#                      article's row / ||row||^½ (partial normalisation, 2026-10-30).
#   child×parent     = idf_parent · log(1 + n), n = nested + c · cousin occurrences, c = 2^(1/√2) − 1 ≈ 0.633 (a lone
#                      cousin tie has half the influence of a lone nested tie; 2026-10-31); idf_parent = ln(P / (df + 1))
#                      + 1 over parent hyperedges (2026-10-31); then each parent's row / ||row||^½ (2026-11-01).
#   W (child×child)  = idf-weighted Jaccard over words, idf_word = ln(H / df) over child hyperedges (the reading check's
#                      form, 2026-10-27); numbers, citation tokens (REFnnnnnn) and URL tokens (URLnnnnnnnn) are not
#                      words of W; model names are, with the soft ladder of model_name_map.model_similarity
#                      (2026-10-31, 2026-11-01): unmatched model names of the two hyperedges are paired one-to-one to
#                      maximise Σ s · min(idf_a, idf_b) (soft cardinality). Ties: for each hyperedge the 5 most similar
#                      with similarity ≥ τ = 0.4, all candidates equal to the 5th kept; a tie counts when either side
#                      selects it (2026-10-27). Built by brute force and by prefix filtering; both must agree (2026-10-29).
# Social side:
#   P1 author×article     1/√n per tie, n = the article's kept bylines (approach 1; 2026-10-30).
#   P2 author×affiliation a byline naming k organisations gives 1/k to each, summed over the person's bylines, row scaled
#                         to Euclidean norm 1; persons with no organisation: missing row (approach 1; 2026-10-26/30).
#   P3 article×affiliation the same per article; articles with no organisation: missing row (approach 2).
#   P4 article×journal    binary (NLM journal ID); articles without one: missing row (2026-10-30).
#   P5 article×article    w(A,B) = Σ_shared persons 1/(n_p − 1) / |authors(A) ∪ authors(B)| (2026-10-27), then
#                         w / (s_A s_B)^¼, s = Euclidean norm of the article's row of w (option (c), 2026-11-01);
#                         diagonal missing; articles with no tie keep observed zeros (approach 2; 2026-10-30).
# Not done here (done per fold in the fit): the whole-matrix normalisation over observed cells (2026-10-31), the hold-out.
# Also saved (2026-11-09): raw counts (tf, nested, cousin) and the unnormalised affiliation shares, from which the fit
# recomputes the weights per hold-out group from visible cells (2026-11-03).
# Squared-error formulas only; the KL version is built later (2026-10-31).
import argparse, collections, csv, glob, hashlib, json, math, os, re, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_name_map as mnm

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
CS = FP + "corpus_statistics/"
PG = "/mnt/hum01-rds/Basov/p91688di/phase5_graphbrain/"
INPUTS = {
    "trial_list": CS + "psychology_trial_pmcids.txt",
    "g3": PG + "g3_v2/g3_test_v2.jsonl",
    "g3_demoted": PG + "g3_v2_lm_demotion/demoted/g3_test_*.jsonl",
    "author_table": FP + "diagnostics/author_coverage/author_table_v2.jsonl",
    "organisations": CS + "institution_organisations_psychology_trial.csv",
    "journals": CS + "subset_candidates_by_article.csv",
    "model_name_map": os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_name_map.py"),
}
NUM = re.compile(r"^[\d.,%]+$")                      # numbers: members of hyperedges, not of the model (2026-10-23)
MASKED = re.compile(r"^(ref\d{6}|url\d{8})$", re.I)  # citation and URL tokens: not words of W (2026-10-28)
COUSIN = 2 ** (2 ** -0.5) - 1                        # 0.6325: log(1 + c) = log(2) / √2
TAU, K_NEAR = 0.4, 5
EPS = 1e-12


def content_key(items):
    """G3's content key (g3_curation_v2.content_key): sha1 of the sorted items, 16 hex."""
    return hashlib.sha1(" ".join(sorted(items)).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ loading
def load_records(trial, g3=INPUTS["g3"], demoted=INPUTS["g3_demoted"]):
    """Sentence-unit records of the trial: the demoted rerun for its articles, the main G3 output otherwise."""
    recs, dem = [], set()
    for f in sorted(glob.glob(demoted)):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["pmcid"] in trial: dem.add(r["pmcid"]); recs.append(r)
    for l in open(g3, encoding="utf-8"):
        p = l[11:l.index('"', 11)]
        if p in trial and p not in dem: recs.append(json.loads(l))
    return recs


def clean_atoms(atoms, pmcid):
    """Typed atoms of one hyperedge after the identity rules: numbers left out, focal roots mapped, non-models removed."""
    out = set()
    for a in atoms:
        root, rest = a["atom"].split("/", 1)
        if NUM.match(root): continue
        if rest == "C/focal":
            root = mnm.canonical(root, pmcid)
            if root is None: continue
        out.add(root + "/" + rest)
    return frozenset(out)


def semantic_units(recs, log):
    """Per sentence unit: its parents (key, [child sets]) and cousins (child set, [parent keys]), after the rules.
    A cousin lists every parent of its sentence, which may lie in another unit of the sentence; a unit without a parent
    of its own is kept for such cousins (G3_POSTPROCESSING.md §6 item 11; owner, 2026-10-26) while at least one of
    those parents survives."""
    dropped, kept, pid2key = [], [], {}
    for r in recs:                                   # pass 1: parents, and units that lose their last focal atom
        pm = r["pmcid"]
        had_focal = any(a["atom"].endswith("/C/focal") for p in r["parents"] for c in p["children"] for a in c["atoms"])
        pars = []
        for p in r["parents"]:
            kids = [s for s in (clean_atoms(c["atoms"], pm) for c in p["children"]) if s]
            if kids: pars.append((p["id"], content_key(content_key(s) for s in kids), kids))
        if had_focal and not any(a.endswith("/C/focal") for _, _, kids in pars for s in kids for a in s):
            dropped.append((r["uid"], "no focal atom left")); continue
        for pid, pk, _ in pars: pid2key[pid] = pk
        kept.append((r, pars))
    units = []
    for r, pars in kept:                             # pass 2: cousins, tied to the surviving parents of the sentence
        cous = []
        for c in r["cousins"]:
            s = clean_atoms(c["atoms"], r["pmcid"])
            if s: cous.append((s, [pid2key[x] for x in (c.get("parents") or []) if x in pid2key]))
        if not pars and not any(pks for _, pks in cous):
            dropped.append((r["uid"], "no parent hyperedge left in its sentence")); continue
        units.append(dict(pmcid=r["pmcid"], uid=r["uid"], parents=[(pk, kids) for _, pk, kids in pars], cousins=cous))
    log["units_in"] = len(recs); log["units_dropped"] = dropped
    return units


# ------------------------------------------------------------------ semantic matrices
def partial_row_normalise(M):
    """row / ||row||^½ for every non-empty row (CSR)."""
    M = M.tocsr().astype(np.float64)
    n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
    f = np.zeros_like(n); f[n > 0] = n[n > 0] ** -0.5
    return sp.diags(f) @ M


def build_semantic(units):
    tf = collections.Counter()                       # (child, article) occurrences
    nest, cous = collections.Counter(), collections.Counter()   # (child, parent) occurrences
    child_sets = {}
    for u in units:
        a = u["pmcid"]
        for pk, kids in u["parents"]:
            for s in kids:
                k = content_key(s); child_sets[k] = s; tf[(k, a)] += 1; nest[(k, pk)] += 1
        for s, pks in u["cousins"]:
            k = content_key(s); child_sets[k] = s; tf[(k, a)] += 1
            for pk in pks: cous[(k, pk)] += 1
    children = sorted(child_sets); ci = {k: i for i, k in enumerate(children)}
    articles = sorted({a for _, a in tf}); ai = {a: i for i, a in enumerate(articles)}
    pairs = set(nest) | set(cous)
    parents = sorted({p for _, p in pairs}); pi = {p: i for i, p in enumerate(parents)}
    # child×article
    df_a = collections.Counter(k for k, _ in tf)
    idf_article = np.array([math.log(len(articles) / (df_a[k] + 1)) + 1 for k in children])
    r, c, v = zip(*((ai[a], ci[k], math.log1p(t) * idf_article[ci[k]]) for (k, a), t in tf.items()))
    CA = partial_row_normalise(sp.csr_matrix((v, (r, c)), shape=(len(articles), len(children))))
    # child×parent (rows = parent hyperedges)
    df_p = collections.Counter(k for k, _ in pairs)
    idf_parent = np.array([math.log(len(parents) / (df_p[k] + 1)) + 1 if df_p[k] else 0.0 for k in children])
    r, c, v = zip(*((pi[p], ci[k], idf_parent[ci[k]] * math.log1p(nest[(k, p)] + COUSIN * cous[(k, p)])) for k, p in pairs))
    PC = partial_row_normalise(sp.csr_matrix((v, (r, c)), shape=(len(parents), len(children))))
    # raw counts for the per-fold recomputation of the weights in the fit (2026-11-03): tf, nested, cousin occurrences
    TF = sp.csr_matrix(([t for t in tf.values()], ([ai[a] for _, a in tf], [ci[k] for k, _ in tf])),
                       shape=(len(articles), len(children)))
    pl = sorted(pairs)
    NEST = sp.csr_matrix(([nest[q] for q in pl], ([pi[p] for _, p in pl], [ci[k] for k, _ in pl])),
                         shape=(len(parents), len(children)))
    COUS = sp.csr_matrix(([cous[q] for q in pl], ([pi[p] for _, p in pl], [ci[k] for k, _ in pl])),
                         shape=(len(parents), len(children)))
    return dict(children=children, child_sets=child_sets, articles=articles, parents=parents, CA=CA, PC=PC,
                idf_article=idf_article, idf_parent=idf_parent, TF=TF, NEST=NEST, COUS=COUS)


# ------------------------------------------------------------------ W (child×child)
class WordSpace:
    """Words of W per child hyperedge, their idf, and the soft model-name similarity."""
    def __init__(self, sets):
        self.H = len(sets)
        self.words = [frozenset(a for a in s if not MASKED.match(a.split("/")[0])) for s in sets]
        df = collections.Counter(a for s in self.words for a in s)
        self.idf = {a: math.log(self.H / d) for a, d in df.items()}
        self.df = df
        self.weight = np.array([math.fsum(self.idf[a] for a in s) for s in self.words])
        named = sorted({a for s in self.words for a in s if self.is_named(a)})
        self.named = named; self.ni = {a: i for i, a in enumerate(named)}
        roots = [a.split("/")[0] for a in named]
        S = np.zeros((len(named), len(named)))
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                S[i, j] = S[j, i] = mnm.model_similarity(roots[i], roots[j])
        self.S = S
        self.named_of = [sorted(a for a in s if a in self.ni) for s in self.words]

    @staticmethod
    def is_named(a):
        return a.endswith("/C/focal") and a.split("/")[0] not in mnm.GENERIC

    def overlap(self, i, j):
        """Σ idf of shared words + best one-to-one soft matching of the unmatched model names. Symmetric to the last
        bit: the pair is taken in index order and the sums are exact (math.fsum)."""
        if i > j: i, j = j, i
        wi, wj = self.words[i], self.words[j]
        o = math.fsum(self.idf[a] for a in wi & wj)
        ui = [a for a in self.named_of[i] if a not in wj]; uj = [b for b in self.named_of[j] if b not in wi]
        if ui and uj:
            G = np.array([[self.S[self.ni[a], self.ni[b]] * min(self.idf[a], self.idf[b]) for b in uj] for a in ui])
            if G.max() > 0:
                if len(ui) == 1 or len(uj) == 1: o = math.fsum([o, G.max()])
                else:
                    r, c = linear_sum_assignment(-G); o = math.fsum([o] + list(G[r, c]))
        return o

    def sim(self, i, j):
        if i > j: i, j = j, i
        den = self.weight[i] + self.weight[j]
        if den <= 0: return 0.0
        o = self.overlap(i, j)
        return o / (den - o)


def select_ties(cands):
    """cands: {i: {j: s}} with s ≥ τ. Each i keeps its 5 best, plus all equal to the 5th; OR rule."""
    W = {}
    for i, d in cands.items():
        c = sorted(((s, j) for j, s in d.items()), key=lambda x: (-x[0], x[1]))
        if len(c) > K_NEAR:
            kth = c[K_NEAR - 1][0]; c = [x for x in c if x[0] >= kth - EPS]
        for s, j in c: W[(min(i, j), max(i, j))] = s
    return W


def w_brute_force(ws, block=128, log=None):
    """Every pair of child hyperedges sharing a word (or holding related model names) is scored."""
    H = ws.H
    vocab = sorted(ws.df); vi = {a: i for i, a in enumerate(vocab)}
    r = [i for i, s in enumerate(ws.words) for a in s]; c = [vi[a] for s in ws.words for a in s]
    X = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(H, len(vocab)))
    Xw = X @ sp.diags(np.array([ws.idf[a] for a in vocab]))
    XT = X.T.tocsc()
    # model-name part: hyperedges holding named models, their related-name incidence (for candidate generation)
    Mi = [i for i in range(H) if ws.named_of[i]]
    rel = (ws.S > 0)
    Nm = sp.csr_matrix((np.ones(sum(len(ws.named_of[i]) for i in Mi)),
                        ([i for i in Mi for _ in ws.named_of[i]], [ws.ni[a] for i in Mi for a in ws.named_of[i]])),
                       shape=(H, len(ws.named)))
    NmR = (Nm @ sp.csr_matrix(rel.astype(np.float64))).tocsr()   # hyperedge -> names related to its names
    NmT = Nm.T.tocsc()
    nw = np.array([sum(ws.idf[a] for a in ws.named_of[i]) for i in range(H)])   # idf mass of named models
    cands = collections.defaultdict(dict); scored = 0; t0 = time.time()
    for b0 in range(0, H, block):
        rows = np.arange(b0, min(H, b0 + block))
        O = (Xw[rows] @ XT).toarray()                      # Σ idf of the shared words, every pair of the block
        R = (NmR[rows] @ NmT).toarray() > 0                # the two hold related (not identical) model names
        C = (O > 0) | R
        C[np.arange(len(rows)), rows] = False
        scored += int(C.sum())
        # upper bound of the similarity: the soft part is at most the smaller idf mass of named models
        UB = O + R * np.minimum(nw[rows][:, None], nw[None, :])
        den = ws.weight[rows][:, None] + ws.weight[None, :] - UB
        with np.errstate(divide="ignore", invalid="ignore"):
            Jub = np.where(den > 0, UB / den, np.inf)
        bi, js = np.nonzero(C & (Jub >= TAU - 1e-9))
        for b, j in zip(bi, js):
            i = int(rows[b]); s = ws.sim(i, int(j))      # one exact function for every reported value
            if s >= TAU: cands[i][int(j)] = s
        if log is not None and (b0 // block) % 40 == 0:
            print(f"  brute force rows {b0 + len(rows)}/{H}, pairs scored {scored:,}, {time.time() - t0:.0f}s", flush=True)
    return select_ties(cands), scored


def w_prefix_filter(ws, log=None):
    """Prefix filtering (Chaudhuri et al. 2006; Bayardo et al. 2007) with one global token order (rarest first).
    Model names are replaced by the token of their similarity component, weighted by their idf, which bounds the soft
    overlap from above, so every pair reaching τ shares a token in both prefixes."""
    H = ws.H
    # components of the model-name similarity graph
    parent = list(range(len(ws.named)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(*np.nonzero(ws.S > 0)): parent[find(i)] = find(j)
    vt = []                                           # per hyperedge: {virtual token: weight}
    for i in range(H):
        d = collections.Counter()
        for a in ws.words[i]:
            t = ("M", find(ws.ni[a])) if a in ws.ni else ("W", a)
            d[t] += ws.idf[a]
        vt.append(d)
    dfv = collections.Counter(t for d in vt for t in d)
    order = {t: n for n, t in enumerate(sorted(dfv, key=lambda t: (dfv[t], str(t))))}
    index = collections.defaultdict(list); cands = collections.defaultdict(dict); scored = 0
    for i in range(H):
        toks = sorted(vt[i], key=order.get); w = ws.weight[i]
        if w <= 0: continue
        suffix = w; pref = []
        for t in toks:
            if suffix < TAU * w * (1 - 1e-9): break
            pref.append(t); suffix -= vt[i][t]
        seen = set()
        for t in pref:
            for j in index[t]:
                if j in seen: continue
                seen.add(j); scored += 1
                s = ws.sim(i, j)
                if s >= TAU: cands[i][j] = s; cands[j][i] = s
            index[t].append(i)
    return select_ties(cands), scored


# ------------------------------------------------------------------ social matrices
def build_social(articles, author_table=INPUTS["author_table"], organisations=INPUTS["organisations"],
                 journals=INPUTS["journals"]):
    A = set(articles); ai = {a: i for i, a in enumerate(articles)}
    org = {r["institution"]: r["organisation"] for r in csv.DictReader(open(organisations))}
    B = [r for r in map(json.loads, open(author_table)) if r["status"] == "kept" and r["pmcid"] in A]
    orgs_of = lambda r: sorted({org.get(i.rsplit("/", 1)[-1], i.rsplit("/", 1)[-1]) for i in r["institutions"]})
    persons = sorted({r["person"] for r in B}); pi = {p: i for i, p in enumerate(persons)}
    organs = sorted({o for r in B for o in orgs_of(r)}); oi = {o: i for i, o in enumerate(organs)}
    team = collections.Counter(r["pmcid"] for r in B)
    au = collections.defaultdict(set); pa = collections.defaultdict(set)
    for r in B: au[r["pmcid"]].add(r["person"]); pa[r["person"]].add(r["pmcid"])
    # P1
    cells = {(pi[r["person"]], ai[r["pmcid"]]): 1 / math.sqrt(team[r["pmcid"]]) for r in B}
    P1 = sp.csr_matrix((list(cells.values()), tuple(zip(*cells))), shape=(len(persons), len(articles)))
    # P2, P3
    def share(key_of, n_rows):
        acc = collections.Counter()
        for r in B:
            os_ = orgs_of(r)
            for o in os_: acc[(key_of(r), oi[o])] += 1 / len(os_)
        M = sp.csr_matrix((list(acc.values()), tuple(zip(*acc))) if acc else ([], ([], [])), shape=(n_rows, len(organs)))
        n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        f = np.zeros_like(n); f[n > 0] = 1 / n[n > 0]
        return (sp.diags(f) @ M).tocsr(), np.nonzero(n == 0)[0], M.tocsr()
    P2, P2_missing, P2_raw = share(lambda r: pi[r["person"]], len(persons))
    P3, P3_missing, P3_raw = share(lambda r: ai[r["pmcid"]], len(articles))
    # P4
    jn = {r["pmcid"]: r["nlm_id"] for r in csv.DictReader(open(journals)) if r["pmcid"] in A}
    journals_ = sorted({j for j in jn.values() if j}); ji = {j: i for i, j in enumerate(journals_)}
    rr = [ai[a] for a in articles if jn.get(a)]
    P4 = sp.csr_matrix((np.ones(len(rr)), (rr, [ji[jn[a]] for a in articles if jn.get(a)])), shape=(len(articles), len(journals_)))
    P4_missing = np.array([ai[a] for a in articles if not jn.get(a)], dtype=int)
    # P5
    w = collections.Counter()
    for p, arts in pa.items():
        if len(arts) < 2: continue
        arts = sorted(arts)
        for x in range(len(arts)):
            for y in range(x + 1, len(arts)): w[(arts[x], arts[y])] += 1 / (len(arts) - 1)
    for (x, y) in w: w[(x, y)] /= len(au[x] | au[y])
    s = collections.Counter()
    for (x, y), v in w.items(): s[x] += v * v; s[y] += v * v
    s = {a: math.sqrt(v) for a, v in s.items()}
    r5, c5, v5 = [], [], []
    for (x, y), v in w.items():
        vc = v / (s[x] * s[y]) ** 0.25
        r5 += [ai[x], ai[y]]; c5 += [ai[y], ai[x]]; v5 += [vc, vc]
    P5 = sp.csr_matrix((v5, (r5, c5)), shape=(len(articles), len(articles)))
    return dict(persons=persons, organisations=organs, journals=journals_, P1=P1, P2=P2, P2_missing_rows=P2_missing,
                P2_raw=P2_raw, P3_raw=P3_raw,
                P3=P3, P3_missing_rows=P3_missing, P4=P4, P4_missing_rows=P4_missing, P5=P5, bylines=len(B))


def frobenius_observed(M, missing_rows=(), mask_diagonal=False):
    """Euclidean norm of the observed cells (missing rows and, for article×article, the diagonal left out)."""
    M = M.tocsr().copy()
    if len(missing_rows): M[np.asarray(missing_rows), :] = 0
    if mask_diagonal: M.setdiag(0)
    M.eliminate_zeros()
    return float(np.sqrt(M.multiply(M).sum()))


# ------------------------------------------------------------------ main
def sha12(path):
    h = hashlib.sha256()
    for f in sorted(glob.glob(path)):
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory (must not exist)")
    ap.add_argument("--articles", type=int, default=0, help="test: only the first N trial articles (sorted)")
    ap.add_argument("--w", choices=("both", "brute", "prefix", "none"), default="both")
    a = ap.parse_args()
    if os.path.exists(a.out): sys.exit(f"refusing to overwrite {a.out}")
    t0 = time.time(); log = {}
    trial = sorted(l.strip() for l in open(INPUTS["trial_list"]) if l.strip())
    if a.articles: trial = trial[:a.articles]
    recs = load_records(set(trial)); print(f"records {len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    units = semantic_units(recs, log)
    sem = build_semantic(units)
    print(f"articles {len(sem['articles'])}, child hyperedges {len(sem['children'])}, parent hyperedges {len(sem['parents'])}",
          f"units dropped {len(log['units_dropped'])} ({time.time() - t0:.0f}s)", flush=True)
    soc = build_social(sem["articles"])
    print(f"persons {len(soc['persons'])}, organisations {len(soc['organisations'])}, journals {len(soc['journals'])} "
          f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out)
    sets = [sem["child_sets"][k] for k in sem["children"]]
    Wres = {}
    if a.w != "none":
        ws = WordSpace(sets)
        for meth, fn in (("brute", w_brute_force), ("prefix", w_prefix_filter)):
            if a.w in (meth, "both"):
                t1 = time.time(); W, scored = fn(ws, log=log)
                Wres[meth] = dict(W=W, scored=scored, seconds=round(time.time() - t1, 1))
                print(f"W {meth}: ties {len(W)}, pairs scored {scored:,}, {time.time() - t1:.0f}s", flush=True)
        if len(Wres) == 2:
            b, p = Wres["brute"]["W"], Wres["prefix"]["W"]
            log["W_identical"] = (b.keys() == p.keys()) and all(b[k] == p[k] for k in b)
            log["W_only_brute"] = len(b.keys() - p.keys()); log["W_only_prefix"] = len(p.keys() - b.keys())
            print("W identical:", log["W_identical"], flush=True)
        W = next(iter(Wres.values()))["W"]
        if W:
            r, c = zip(*W); v = list(W.values())
            Wm = sp.csr_matrix((v + v, (list(r) + list(c), list(c) + list(r))), shape=(len(sets), len(sets)))
        else:
            Wm = sp.csr_matrix((len(sets), len(sets)))
        sp.save_npz(os.path.join(a.out, "W_child_child.npz"), Wm)
    for name, M in (("child_article", sem["CA"]), ("parent_child", sem["PC"]), ("P1_author_article", soc["P1"]),
                    ("P2_author_affiliation", soc["P2"]), ("P3_article_affiliation", soc["P3"]),
                    ("P4_article_journal", soc["P4"]), ("P5_article_article", soc["P5"])):
        sp.save_npz(os.path.join(a.out, name + ".npz"), M.tocsr())
    # raw ingredients (before idf, log, row normalisation) for the per-fold weights of the fit (2026-11-03)
    for name, M in (("raw_child_article_tf", sem["TF"]), ("raw_parent_child_nested", sem["NEST"]),
                    ("raw_parent_child_cousin", sem["COUS"]), ("raw_P2_author_affiliation", soc["P2_raw"]),
                    ("raw_P3_article_affiliation", soc["P3_raw"])):
        sp.save_npz(os.path.join(a.out, name + ".npz"), M.tocsr())
    np.save(os.path.join(a.out, "idf_article.npy"), sem["idf_article"]); np.save(os.path.join(a.out, "idf_parent.npy"), sem["idf_parent"])
    idx = dict(articles=sem["articles"], children=sem["children"], parents=sem["parents"], persons=soc["persons"],
               organisations=soc["organisations"], journals=soc["journals"],
               child_atoms={k: sorted(sem["child_sets"][k]) for k in sem["children"]})
    json.dump(idx, open(os.path.join(a.out, "index.json"), "w"))
    missing = dict(P2_author_affiliation_rows=soc["P2_missing_rows"].tolist(), P3_article_affiliation_rows=soc["P3_missing_rows"].tolist(),
                   P4_article_journal_rows=soc["P4_missing_rows"].tolist(), P5_article_article="diagonal")
    json.dump(missing, open(os.path.join(a.out, "missing.json"), "w"))
    manifest = dict(builder="chunk12v3", created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1),
                    args=vars(a), inputs={k: sha12(v) for k, v in INPUTS.items()},
                    constants=dict(cousin=COUSIN, tau=TAU, k=K_NEAR, unchecked_similarity=mnm.UNCHECKED_SIMILARITY),
                    counts=dict(articles=len(sem["articles"]), child_hyperedges=len(sem["children"]),
                                parent_hyperedges=len(sem["parents"]), persons=len(soc["persons"]),
                                organisations=len(soc["organisations"]), journals=len(soc["journals"]), bylines=soc["bylines"],
                                units_in=log["units_in"], units_dropped=len(log["units_dropped"]),
                                nonzeros={n: int(M.nnz) for n, M in (("child_article", sem["CA"]), ("parent_child", sem["PC"]),
                                          ("P1", soc["P1"]), ("P2", soc["P2"]), ("P3", soc["P3"]), ("P4", soc["P4"]), ("P5", soc["P5"]))},
                                missing_rows={k: len(v) for k, v in missing.items() if isinstance(v, list)}),
                    frobenius_observed=dict(child_article=frobenius_observed(sem["CA"]), parent_child=frobenius_observed(sem["PC"]),
                                            P1=frobenius_observed(soc["P1"]), P2=frobenius_observed(soc["P2"], soc["P2_missing_rows"]),
                                            P3=frobenius_observed(soc["P3"], soc["P3_missing_rows"]),
                                            P4=frobenius_observed(soc["P4"], soc["P4_missing_rows"]),
                                            P5=frobenius_observed(soc["P5"], mask_diagonal=True)),
                    W={m: dict(ties=len(r["W"]), pairs_scored=r["scored"], seconds=r["seconds"]) for m, r in Wres.items()},
                    W_identical=log.get("W_identical"), W_only_brute=log.get("W_only_brute"), W_only_prefix=log.get("W_only_prefix"), units_dropped=log["units_dropped"])
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
