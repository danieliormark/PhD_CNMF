# WHAT: compare chunk12's anchor weighting log1p(tf)*[ln(N/(df+1))+1] against classic and sklearn-smooth tf-idf.
# OUT: tfidf_comparison.png (illustrative only, not a diagnostic result; numbers annotated on it come from
#      Star_epistemic_decoders_global.pkl / Star_extended_matrices_t{1,2}.pkl, checked 2026-09-24)
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_CLASSIC, C_OURS, C_SK = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3, fixed order

def idf_classic(N, df): return np.log(N / df)
def idf_ours(N, df):    return np.log(N / (df + 1.0)) + 1.0              # chunk12.py line 291
def idf_sk(N, df):      return np.log((N + 1.0) / (df + 1.0)) + 1.0      # sklearn smooth_idf=True

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

fig, axs = plt.subplots(2, 2, figsize=(13, 9))

for ax, N, lab in ((axs[0, 0], 61, "A. idf factor vs df, toy corpus (N = 61)"),
                   (axs[0, 1], 22000, "B. idf factor vs df, full scale (N = 22,000)")):
    df = np.unique(np.round(np.logspace(0, np.log10(N - 1), 300)))
    ax.plot(df, idf_classic(N, df), color=C_CLASSIC, lw=2, ls="-",  label="classic  ln(N/df)")
    ax.plot(df, idf_ours(N, df),    color=C_OURS,    lw=2, ls="--", label="ours  ln(N/(df+1)) + 1")
    ax.plot(df, idf_sk(N, df),      color=C_SK,      lw=2, ls=":",  label="sklearn smooth  ln((N+1)/(df+1)) + 1")
    ax.set_xscale("log"); ax.set_xlabel("document frequency df (articles containing the element)")
    ax.set_ylabel("idf factor"); style(ax, lab)
axs[0, 0].axvline(1, color=INK2, lw=1)
axs[0, 0].annotate("toy corpus: 86-99% of parent / child /\ncousin hyperedges sit here (df = 1),\nso their idf is one constant value",
                   xy=(1, idf_ours(61, 1)), xytext=(1.3, 0.15), color=INK2, fontsize=9,
                   arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
axs[0, 0].legend(frameon=False, loc="upper right", fontsize=9)

ax = axs[1, 0]
tf = np.linspace(1, 10, 200)
ax.plot(tf, tf, color=C_CLASSIC, lw=2, ls="-",  label="raw tf")
ax.plot(tf, np.log1p(tf), color=C_OURS, lw=2, ls="--", label="ours  log(1 + tf)")
ax.plot(tf, 1 + np.log(tf), color=C_SK, lw=2, ls=":",  label="classic sublinear  1 + ln(tf)")
ax.set_xlabel("tf (times the same hyperedge occurs in one article)"); ax.set_ylabel("tf factor")
style(ax, "C. tf factor vs tf"); ax.set_ylim(0, 10.5)
ax.annotate("toy corpus: 92-99% of anchor ties have tf = 1\n(ours = 0.69 there, i.e. effectively a constant)",
            xy=(1, np.log1p(1)), xytext=(2.2, 0.15), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.legend(frameon=False, loc="upper left", fontsize=9)

ax = axs[1, 1]
N = 22000
df = np.unique(np.round(np.logspace(0, np.log10(N * 0.9), 300)))
for f, c, ls, lab in ((idf_classic, C_CLASSIC, "-", "classic"), (idf_ours, C_OURS, "--", "ours")):
    ax.plot(df, (f(N, df) / f(N, 1.0)) ** 2, color=c, lw=2, ls=ls, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("document frequency df"); ax.set_ylabel("(idf(df) / idf(1))^2")
style(ax, "D. Row's weight in the squared-error loss vs a df = 1 row (N = 22,000)")
ax.legend(frameon=False, loc="upper right", fontsize=9)
for d, ytxt in ((1000, 0.03), (10000, 0.008)):
    ax.annotate(f"df={d}: ours {(idf_ours(N, d)/idf_ours(N, 1))**2:.3f}, classic {(idf_classic(N, d)/idf_classic(N, 1))**2:.4f}",
                xy=(d, (idf_ours(N, d) / idf_ours(N, 1)) ** 2), xytext=(1.3, ytxt),
                color=INK2, fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))

fig.suptitle("chunk12 anchor weighting  log1p(tf) x [ln(N/(df+1)) + 1]  vs classic and sklearn tf-idf", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfidf_comparison.png")
fig.savefig(out, dpi=140)
print(out)
