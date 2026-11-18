# Social log-damping arm (2026-11-18): data.damp_values on a small relation.
import numpy as np
from fit import data as D


def test_damp_values():
    rows = np.array([0, 1, 2, 3, 0, 1]); cols = np.array([0, 0, 0, 1, 1, 2])     # column 0: 3 visible ties
    v = np.ones(6); vis = np.array([True, True, True, True, False, False])          # column 2: only a hidden tie
    fro = np.sqrt(v[vis].dot(v[vis]))
    vd, div, info = D.damp_values("article_journal", rows, cols, v, vis, (4, 3), fro)
    f = np.log1p(np.array([3.0, 1.0, 1.0])) / np.array([3.0, 1.0, 1.0])            # d = 3, 1 visible; d = 0 read as 1
    assert np.allclose(vd, v * f[cols])
    data = vd / div
    assert np.isclose(data[vis].sum(), v[vis].sum() / fro)                           # visible mass restored
    weight = vd / info["fro_damped"]
    assert np.isclose(np.sqrt((weight[vis] ** 2).sum()), 1.0)                         # weight arm: Frobenius norm 1
    assert np.allclose(data, info["q"] * weight)                                      # A = q × B, hidden ties included
    assert info["q"] > 1 and info["entities_all_hidden"] == 1
