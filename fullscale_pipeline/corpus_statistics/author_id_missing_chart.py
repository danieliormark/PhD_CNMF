# Per-article distribution of the share of authors with neither an OpenAlex ID nor an ORCID known
# (owner, 2026-10-12; CORPUS_STATISTICS.md §9.10). Reads the already-cached
# diagnostics/author_coverage/author_id_missing_per_article.json (28,029 articles; no new fetch).
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FP = "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/"
rows = json.load(open(FP + "diagnostics/author_coverage/author_id_missing_per_article.json"))["per_article"]
pct = [100 * r["missing"] / r["n"] if r["n"] else 0.0 for r in rows]

BLUE = "#2a78d6"
bins = [(0, 0, "0%"), (0, 10, "1-9%"), (10, 20, "10-19%"), (20, 30, "20-29%"), (30, 40, "30-39%"),
        (40, 50, "40-49%"), (50, 60, "50-59%"), (60, 70, "60-69%"), (70, 80, "70-79%"),
        (80, 90, "80-89%"), (90, 101, "90-100%")]
counts = [sum(1 for x in pct if x == 0)]
for i, (lo, hi, _) in enumerate(bins[1:]):
    lower = (lambda x: x > 0) if i == 0 else (lambda x, lo=lo: x >= lo)   # "1-9%" excludes the exact-0 bin
    counts.append(sum(1 for x in pct if lower(x) and x < hi))
n = len(pct)
assert sum(counts) == n, (sum(counts), n)   # every article falls in exactly one bin
labels = [b[2] for b in bins]

fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(labels))
bars = ax.bar(x, counts, width=0.62, color=BLUE, edgecolor="none", zorder=3)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9, color="#3a3a38")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_ylabel("Articles", fontsize=10, color="#52514e")
fig.suptitle("Share of an article's own authors with neither an OpenAlex ID nor an ORCID known",
             fontsize=12.5, color="#0b0b0b", x=0.015, ha="left", y=0.975)
fig.text(0.015, 0.915, f"{n:,} articles (full working corpus)", fontsize=9.5, color="#52514e", ha="left")
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c0")
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", colors="#52514e", labelsize=9)
ax.grid(axis="y", color="#e7e6e0", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for i, c in enumerate(counts):
    ax.text(i, c + max(counts) * 0.015, f"{c:,}\n({100 * c / n:.1f}%)", ha="center", va="bottom",
             fontsize=8, color="#3a3a38")
ax.set_ylim(0, max(counts) * 1.14)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FP + "corpus_statistics/author_id_missing_distribution.png", facecolor=fig.get_facecolor())
print("wrote corpus_statistics/author_id_missing_distribution.png")
