# Ticket 107, D1 step 3 (RL-412): simulated error / power of rules C2, C3, C4. Run from plans/ticket107_layer2.
# C2 / C3 / C4 at one K, G = 10 hold-out groups, δ = 1 SE of a paired difference's mean.
# Scenario "equal": all arms equal → error = some arm not tied. "one better by s SE": success = every other arm out.
import itertools, sys, numpy as np
sys.path.insert(0, "../../fullscale_pipeline/m1"); from fit import d1_analysis as D
rng = np.random.default_rng(11); G = 10; SE = 1 / np.sqrt(G); dl = 1.0 * SE
F = np.array(list(itertools.product([-1, 1], repeat=G)), float)
def tie_sets(X, k, cdun):
    m = X.mean(0); b = int(m.argmax())
    c2 = {i for i in range(k) if m[b] - m[i] <= dl}
    def clear(i, j, c):           # j clearly better than i
        d = X[:, j] - X[:, i]; s = d.std(ddof=1) / np.sqrt(G); return d.mean() - c * s > 0 and d.mean() >= dl
    c3 = {b} | {i for i in range(k) if i != b and not clear(i, b, cdun)}
    pairs = list(itertools.combinations(range(k), 2))
    Dm = np.stack([X[:, j] - X[:, i] for i, j in pairs], 1); C = Dm - Dm.mean(0)
    T = F[:, :, None] * C[None]; cf = np.quantile(np.abs(T.mean(1) / (T.std(1, ddof=1) / np.sqrt(G))).max(1), 0.95)
    out = set()
    for i, j in pairs:
        if clear(i, j, cf): out.add(i)
        if clear(j, i, cf): out.add(j)
    c4 = set(range(k)) - out
    return c2, c3, c4
for k in (3, 5):
    Rd = np.full((k - 1, k - 1), 0.5); np.fill_diagonal(Rd, 1); cdun = D.mvt_crit(Rd, G - 1, 0.95)
    for s in (0, 2, 4):
        n = 2000; hit = np.zeros(3)
        for _ in range(n):
            X = rng.standard_normal((G, k)) / np.sqrt(2); X[:, 0] += s * SE
            for t, ts in enumerate(tie_sets(X, k, cdun)):
                hit[t] += (len(ts) < k) if s == 0 else (ts == {0})
        lab = "P(some arm wrongly out)" if s == 0 else f"arm 0 better by {s} SE: P(only arm 0 left)"
        print(f"k={k} {lab}: C2 {hit[0]/n:.3f}  C3 {hit[1]/n:.3f}  C4 {hit[2]/n:.3f}")
