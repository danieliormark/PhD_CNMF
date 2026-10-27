# 2026-10-27 (owner): before choosing option (b) for approach 2 (article×article as a regulariser in the Psychology
# trial), check what such a regulariser would pull together. Read-only; no network. Owner's questions:
#   1. how much off-subset (but in-corpus) co-authorship would add: two trial articles tied through an article outside
#      the subset that shares an identified person with each (A–C–B); and
#   2. whether the shared-author graph is one giant component (a regulariser would then pull it towards one membership)
#      or splits into groups; community detection, hard and soft.
# Graphs (nodes = trial articles; persons = author_table_v2 persons of kept bylines, OpenAlex ID / ORCID / R4 links):
#   IN    edge A–B if they share a person; weight Σ_p 1/(n_p − 1), n_p = the person's trial articles (the weight proposed
#         for the article×article relation, design doc §2)
#   EXT   IN plus two-hop ties through corpus articles outside the trial: weight Σ_C w(A,C)·w(C,B) with w as above but
#         n_p counted over the whole working corpus (a path through a hub article is damped like a prolific author)
# Methods, each on IN and EXT: connected components; Leiden (modularity, resolution 1, 10 seeds, best kept) and Louvain
# (networkx) as hard partitions; Hy-MMSBM (hypergraphx, Ruggeri et al. 2023) as the soft one, on the hypergraph whose
# hyperedges are the persons (each person's set of trial articles, 2+), K = 4 as in the M1 grid, assortative, 5 seeds.
# Per partition: number and sizes of communities, modularity Q, per-community conductance, share of edge weight inside
# communities; topic purity: share of article pairs in one community with the same OpenAlex primary topic / subfield,
# against all pairs of tied articles and against random pairs. Outputs: corpus_statistics/coauthor_graph/
# summary.json, graph_in.png, graph_in_giant.png, graph_ext_giant.png, communities_in.csv (pmcid, component, leiden, louvain, hymmsbm argmax,
# hymmsbm max share). Run with tensor_env and PYTHONPATH pointing at a folder with hypergraphx, igraph, leidenalg.
import collections, csv, itertools, json, math, os, random
import numpy as np, networkx as nx
import igraph as ig, leidenalg
from hypergraphx import Hypergraph
from hypergraphx.communities.hy_mmsbm.model import HyMMSBM
from hypergraphx.linalg import binary_incidence_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
CS = FP + "corpus_statistics/"; OUT = CS + "coauthor_graph/"
os.makedirs(OUT, exist_ok=True)
trial = {l.strip() for l in open(CS + "psychology_trial_pmcids.txt") if l.strip()}
wc = {l.strip() for l in open(CS + "working_corpus_v2_pmcids.txt") if l.strip()}
meta = {r["pmcid"]: r for r in csv.DictReader(open(CS + "subset_candidates_by_article.csv"))}

P = collections.defaultdict(set)                                     # person -> corpus articles
for l in open(FP + "diagnostics/author_coverage/author_table_v2.jsonl"):
    r = json.loads(l)
    if r["status"] == "kept" and r["pmcid"] in wc and r["how"] != "own_person":
        P[r["person"]].add(r["pmcid"])


def weights(arts_of, keep):
    W = collections.Counter()
    for p, arts in arts_of.items():
        a = sorted(x for x in arts if x in keep)
        if len(a) > 1:
            for x, y in itertools.combinations(a, 2):
                W[(x, y)] += 1 / (len(a) - 1)
    return W


W_in = weights(P, trial)
# two-hop through outside articles: corpus-wide weights, then paths A–C–B with C outside the trial
W_all = collections.defaultdict(dict)
for p, arts in P.items():
    if 1 < len(arts) <= 200 and arts & trial:
        w = 1 / (len(arts) - 1)
        for x, y in itertools.combinations(sorted(arts), 2):
            W_all[x][y] = W_all[x].get(y, 0) + w; W_all[y][x] = W_all[y].get(x, 0) + w
W_ext = collections.Counter(W_in)
for c, nb in W_all.items():
    if c in trial:
        continue
    t = sorted(x for x in nb if x in trial)
    for x, y in itertools.combinations(t, 2):
        W_ext[(x, y)] += nb[x] * nb[y]


def graph(W):
    G = nx.Graph()
    for (x, y), w in W.items():
        G.add_edge(x, y, weight=w)
    return G


def leiden(G, seeds=10):
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in G.edges()])
    g.es["weight"] = [G[a][b]["weight"] for a, b in G.edges()]
    best = max((leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=s) for s in range(seeds)),
               key=lambda p: p.quality())
    return {nodes[i]: c for c, mem in enumerate(best) for i in mem}


def hard_stats(G, part):
    comms = collections.defaultdict(set)
    for n, c in part.items():
        comms[c].add(n)
    Q = nx.algorithms.community.modularity(G, comms.values(), weight="weight")
    tot = G.size(weight="weight"); inside = sum(d["weight"] for a, b, d in G.edges(data=True) if part[a] == part[b])
    cond = [nx.conductance(G, s, weight="weight") for s in comms.values() if 0 < len(s) < G.number_of_nodes()]
    sizes = sorted((len(s) for s in comms.values()), reverse=True)
    return dict(n_communities=len(comms), largest=sizes[:10], singletons_or_pairs=sum(s <= 2 for s in sizes),
                modularity=round(Q, 4), weight_inside=round(inside / tot, 4),
                conductance_median=round(float(np.median(cond)), 4) if cond else None,
                conductance_weighted_by_size=round(sum(c * len(s) for c, s in zip(cond, comms.values())) / G.number_of_nodes(), 4) if cond else None)


def purity(groups, key):
    same = tot = 0
    for g in groups:
        g = [x for x in g if meta.get(x, {}).get(key)]
        for x, y in itertools.combinations(g, 2):
            tot += 1; same += meta[x][key] == meta[y][key]
    return round(same / tot, 4) if tot else None


def random_purity(nodes, key, n=200000, seed=1):
    rng = random.Random(seed); nodes = [x for x in nodes if meta.get(x, {}).get(key)]; s = 0
    for _ in range(n):
        x, y = rng.sample(nodes, 2); s += meta[x][key] == meta[y][key]
    return round(s / n, 4)


def hymmsbm(arts_of, K=4, seeds=5):
    hyes = sorted({tuple(sorted(x for x in a if x in trial)) for a in arts_of.values() if len([x for x in a if x in trial]) > 1})
    nodes = sorted({x for h in hyes for x in h}); idx = {n: i for i, n in enumerate(nodes)}
    H = Hypergraph([tuple(idx[x] for x in h) for h in hyes])
    best, bll = None, -np.inf
    for s in range(seeds):
        m = HyMMSBM(K=K, assortative=True, seed=s); m.fit(H, n_iter=1000)
        ll = m.log_likelihood(H)
        if ll > bll:
            best, bll = m, ll
    _, mp = binary_incidence_matrix(H, return_mapping=True)          # hypergraphx renumbers nodes: row i is node mp[i]
    order = [nodes[mp[i]] for i in range(len(nodes))]
    u = best.u; share = u / np.clip(u.sum(1, keepdims=True), 1e-12, None)
    global best_w; best_w = best.w
    return order, share, len(hyes), float(bll)


S = dict(articles=len(trial))
res = {}
for name, W in (("IN", W_in), ("EXT", W_ext)):
    G = graph(W); comps = sorted(nx.connected_components(G), key=len, reverse=True)
    deg = dict(G.degree())
    d = dict(edges=G.number_of_edges(), articles_with_a_tie=G.number_of_nodes(),
             articles_without=len(trial) - G.number_of_nodes(),
             mean_degree_all_articles=round(2 * G.number_of_edges() / len(trial), 3),
             mean_degree_tied=round(2 * G.number_of_edges() / max(1, G.number_of_nodes()), 3),
             components=len(comps), largest_components=[len(c) for c in comps[:10]],
             giant_share_of_tied=round(len(comps[0]) / G.number_of_nodes(), 4))
    L = leiden(G); LV = {n: i for i, s in enumerate(nx.algorithms.community.louvain_communities(G, weight="weight", seed=1)) for n in s}
    d["leiden"] = hard_stats(G, L); d["louvain"] = hard_stats(G, LV)
    gi = comps[0].copy()
    d["giant_component"] = dict(size=len(gi), leiden_communities_inside=len({L[n] for n in gi}),
                                leiden_sizes_inside=sorted(collections.Counter(L[n] for n in gi).values(), reverse=True)[:10])
    for key in ("topic", "subfield"):
        d[f"purity_{key}"] = dict(random_pairs=random_purity(list(trial), key),
                                  tied_pairs=purity([[a, b] for a, b in G.edges()], key),
                                  components=purity(comps, key), giant_component=purity([comps[0]], key),
                                  leiden=purity([[n for n in L if L[n] == c] for c in set(L.values())], key))
    res[name] = (G, L, LV, comps)
    S[name] = d
nodes, share, nh, ll = hymmsbm(P)
G, L, LV, comps = res["IN"]
amax = dict(zip(nodes, share.argmax(1))); mx = dict(zip(nodes, share.max(1)))
ent = -(share * np.log(np.clip(share, 1e-12, None))).sum(1) / math.log(4)
pairs_same = [amax[a] == amax[b] for a, b in G.edges()]
S["hymmsbm_IN_K4"] = dict(hyperedges_persons=nh, nodes=len(nodes), loglik=round(ll, 2),
                         community_sizes_argmax=sorted(collections.Counter(amax.values()).values(), reverse=True),
                         max_share_median=round(float(np.median(share.max(1))), 3),
                         share_of_articles_max_share_ge_0_9=round(float((share.max(1) >= 0.9).mean()), 3),
                         normalised_entropy_median=round(float(np.median(ent)), 3),
                         giant_component_argmax_sizes=sorted(collections.Counter(amax[n] for n in comps[0] if n in amax).values(), reverse=True),
                         components_split_across_communities=sum(len({amax[n] for n in c if n in amax}) > 1 for c in comps),
                         tied_pairs_same_argmax=round(sum(pairs_same) / len(pairs_same), 4),
                         w_diagonal=[round(float(x), 5) for x in np.diag(best_w)],
                         purity_topic=purity([[n for n in amax if amax[n] == k] for k in range(4)], "topic"),
                         purity_subfield=purity([[n for n in amax if amax[n] == k] for k in range(4)], "subfield"))
comp_of = {n: i for i, c in enumerate(comps) for n in c}
with open(OUT + "communities_in.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pmcid", "component", "leiden", "louvain", "hymmsbm_argmax", "hymmsbm_max_share", "subfield", "topic"])
    for n in sorted(G.nodes()):
        w.writerow([n, comp_of[n], L[n], LV[n], amax.get(n, ""), round(float(mx[n]), 3) if n in mx else "",
                    meta.get(n, {}).get("subfield", ""), meta.get(n, {}).get("topic", "")])
json.dump(S, open(OUT + "summary.json", "w"), indent=1)


def draw(G, part, path, title):
    pos = nx.spring_layout(G, weight="weight", seed=3, k=0.6 / math.sqrt(max(1, G.number_of_nodes())), iterations=200)
    top = [c for c, _ in collections.Counter(part.values()).most_common(19)]
    cmap = plt.get_cmap("tab20"); col = [cmap(top.index(part[n])) if part[n] in top else (0.75, 0.75, 0.75, 1) for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(13, 13))
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.3 + 1.5 * min(1, G[a][b]["weight"]) for a, b in G.edges()], alpha=0.35)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=col, node_size=[10 + 6 * G.degree(n) for n in G.nodes()], linewidths=0)
    ax.set_title(title); ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


draw(G, L, OUT + "graph_in.png", f"Psychology trial: articles sharing an identified author ({G.number_of_nodes()} of "
     f"{len(trial)} articles, {G.number_of_edges()} ties); colour = Leiden community (top 19; grey = others)")
draw(G.subgraph(comps[0]), L, OUT + "graph_in_giant.png", f"Shared-author ties inside the trial: largest component "
     f"({len(comps[0])} articles); colour = Leiden community")
Ge, Le, _, ce = res["EXT"]
draw(Ge.subgraph(ce[0]), Le, OUT + "graph_ext_giant.png", f"With two-hop ties through corpus articles outside the trial: "
     f"largest component ({len(ce[0])} articles); colour = Leiden community")
print(json.dumps(S, indent=1))
