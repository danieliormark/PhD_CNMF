# child×parent: effect of partial normalisation (row / ||row||^1/2) on every parent hyperedge, by number of ties.
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer1_review.py")).read().split("v = np.array(list(pin.values()))")[0])
import numpy as np
K = sorted(pin); v = np.array([pin[k] for k in K]); t = np.array([pt[k] for k in K])
after = np.sqrt(v)   # influence after = ||row|| = sqrt(sum of squares)
m0, m1 = np.median(v), np.median(after)
print("parents", len(K))
for lo, hi in ((1, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 10**6)):
    s = (t >= lo) & (t <= hi)
    print("ties %d-%s: n=%d share of parents %.1f%% | median influence/median: %.2f -> %.2f | share of matrix mass %.1f%% -> %.1f%%" % (
        lo, hi if hi < 10**6 else "", s.sum(), 100 * s.mean(), np.median(v[s]) / m0, np.median(after[s]) / m1, 100 * v[s].sum() / v.sum(), 100 * after[s].sum() / after.sum()))
