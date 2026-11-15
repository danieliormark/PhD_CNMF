# Option B test arm (2026-11-15): WordSimilarityContrast. Run: python -m fit.tests.test_w_contrast. P is passed directly (weighted_membership patched to identity).
import numpy as np, scipy.sparse as sp, torch
from fit import terms as T


class _MP:
    def setattr(self, o, k, v): setattr(o, k, v)


def _setup(monkeypatch, n=12, K=4, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.3, random_state=seed); A = A + A.T; A.setdiag(0); A = sp.csr_matrix(A); A.eliminate_zeros()
    monkeypatch.setattr(T, "weighted_membership", lambda Un, Zs, rel, f, rows=None: Un[f])
    P = rng.random((n, K)); P /= P.sum(1, keepdims=True)
    return A, torch.tensor(P, dtype=torch.float64)


def test_value_brute_force(monkeypatch):
    W, P = _setup(monkeypatch)
    t = T.WordSimilarityContrast("child", W)
    Wd = W.toarray(); d = Wd.sum(1); tied = np.nonzero(d > 0)[0]; p = P.numpy()
    R = sum(Wd[i, j] / d[i] * ((p[i] - p[j]) ** 2).sum() for i in tied for j in range(len(p))) / len(tied)
    S = np.mean([((p[i] - p[j]) ** 2).sum() for i in tied for j in tied])
    assert np.isclose(float(t.value({"child": P}, None, None)), R / S, rtol=1e-10)


def test_flattening_gives_no_gain(monkeypatch):
    W, P = _setup(monkeypatch)
    t = T.WordSimilarityContrast("child", W); t0 = T.WordSimilarity("child", W)
    c = torch.full((P.shape[1],), 1.0 / P.shape[1], dtype=torch.float64)
    for a in (0.5, 0.1, 0.01):
        Q = a * P + (1 - a) * c
        assert np.isclose(float(t.value({"child": Q}, None, None)), float(t.value({"child": P}, None, None)), rtol=1e-9)
        assert float(t0.value({"child": Q}, None, None)) < float(t0.value({"child": P}, None, None))   # R itself falls


def test_gradient(monkeypatch):
    W, P = _setup(monkeypatch, n=8, K=3)
    t = T.WordSimilarityContrast("child", W)
    X = P.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: t.value({"child": x}, None, None), (X,))


if __name__ == "__main__":
    for f in (test_value_brute_force, test_flattening_gives_no_gain, test_gradient):
        f(_MP()); print("PASS", f.__name__)
    print("ALL CHECKS PASSED")
