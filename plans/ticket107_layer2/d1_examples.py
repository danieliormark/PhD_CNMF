# Ticket 107, D1 (journal weighting): the worked examples and test-power numbers of the pre-registration
# (D1_journal_preregistration.md). Reads no data and fits no model: journal sizes are taken from journal_variants.txt
# (384 journals; top 10 sizes; 169 with >= 2 articles; 215 singletons) and memberships are invented.
# Run: python d1_examples.py > d1_examples.txt
import numpy as np
from scipy import stats

rng = np.random.default_rng(3)

# ---- (1) shuffle test for journal-shaped communities, with trial-like journal sizes --------------------------------
top = [184, 146, 83, 77, 68, 67, 63, 61, 61, 59]                       # journal_variants.txt
rest = rng.multinomial(1052 - 2 * 159, np.ones(159) / 159) + 2        # the other 159 journals with >= 2 articles
sizes = np.array(top + list(rest) + [1] * 215)
N = sizes.sum()
lab0 = np.repeat(np.arange(len(sizes)), sizes)
multi = sizes >= 2                                                    # owner 2026-11-05: singletons not counted
mid = np.where(sizes == np.sort(rest)[-30])[0][0]                     # a mid-size journal (8 articles)


def j_share(lab, m):
    s = np.bincount(lab, weights=m, minlength=len(sizes)) / m.sum()
    s = np.where(multi, s, 0.0)
    j = s.argmax()
    return s[j], j


base = rng.gamma(1.0, 0.2, N)                                          # a community's article membership masses
cases = {}
cases["A spread over the corpus"] = base.copy()
m = base.copy(); idx = np.where(lab0 == mid)[0]; m[idx] += (0.30 * base.sum() / 0.70) / len(idx)
cases["B 30% of its mass in one mid-size journal"] = m
m = base.copy(); m[lab0 == 0] *= 2.6
cases["C over-represented in the largest journal"] = m
print(f"(1) N = {N} articles, {len(sizes)} journals, largest {sizes.max()} ({sizes.max() / N:.3f} of articles)")
for n_shuf in (200, 999):
    print(f"  shuffles = {n_shuf}")
    for name, m in cases.items():
        s, j = j_share(lab0, m)
        sh = np.array([j_share(rng.permutation(lab0), m)[0] for _ in range(n_shuf)])
        p = (1 + (sh >= s).sum()) / (n_shuf + 1)
        print(f"    {name}: j_k = {s:.3f} (journal of {sizes[j]})  shuffles mean {sh.mean():.3f}  "
              f"95th pct {np.percentile(sh, 95):.3f}  p = {p:.3f}  journal-shaped: {bool(p < 0.05 and s >= 0.25)}")

# ---- (2) Monte Carlo precision of a shuffle-based 95th percentile ----------------------------------------------------
print("(2) the empirical 95th percentile of n shuffles corresponds to a true quantile with SE sqrt(0.05*0.95/n):")
for n in (200, 999, 9999):
    se = np.sqrt(0.05 * 0.95 / n)
    print(f"    n = {n}: SE {se:.4f}; ~95% range of the true quantile {0.95 - 1.96 * se:.3f}-{0.95 + 1.96 * se:.3f}; "
          f"smallest attainable p = {1 / (n + 1):.4f}")

# ---- (3) critical values and power: single test, Dunnett, Bonferroni (5 arms vs J1) ---------------------------------
def dunnett_crit(df, k=5, rho=0.5, n=400_000):
    """Two-sided 95% critical value of max |T_i| for k comparisons with one shared control (correlation 0.5)."""
    C = np.full((k, k), rho); np.fill_diagonal(C, 1.0)
    z = rng.multivariate_normal(np.zeros(k), C, size=n)
    s = np.sqrt(rng.chisquare(df, size=n) / df)
    return np.quantile(np.abs(z / s[:, None]).max(1), 0.95)


def power(c, df, d):            # probability that T exceeds c when the true effect is d standard errors
    return 1 - stats.nct.cdf(c, df, d)


print("(3) critical values and power (true effect in units of the SE of the mean difference)")
for g in (5, 10):
    for label, df in (("own SE", g - 1), ("pooled block error", (g - 1) * 5)):
        c1, cd, cb = stats.t.ppf(0.975, df), dunnett_crit(df), stats.t.ppf(1 - 0.025 / 5, df)
        pw = lambda c: " / ".join(f"{power(c, df, d):.2f}" for d in (2, 3, 4, 5))
        print(f"    {g} groups, {label} (df {df}): critical single {c1:.2f}, Dunnett {cd:.2f}, Bonferroni {cb:.2f}")
        print(f"        power at 2/3/4/5 SE: single {pw(c1)} | Dunnett {pw(cd)} | Bonferroni {pw(cb)}")
print(f"    chance of >= 1 false finding among 5 independent tests at 5%: {1 - 0.95 ** 5:.3f}")
