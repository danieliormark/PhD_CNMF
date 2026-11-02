# Ticket 107, layer 2: architecture of the first M1 fits (Psychology trial) as a staged decision tree.
# Proposal for the owner, 2026-11-02; nothing here runs a fit. It enumerates the arms, computes each arm's relation
# weights by the decided rule (w = 0.5 / n_domain; journal multiplier applied inside the social half, which is then
# rescaled to 0.5 — owner, 2026-11-02), counts fits, and prints the tree with the decision taken at each node.
# Run: python3 test_architecture.py  (writes nothing; redirect to keep a copy)

FOLDS, SEEDS = 5, 5                       # hold-out groups used (5 of 10) x seeds per group (S, proposed 5)
K_PROBE = [4, 6, 8]                       # stage 1: a few K values (proposal; owner expects > 4 may be optimal)
K_GRID = list(range(2, 11))               # stage 2: trial K range, decided 2026-10-17 (full grid K = 2-10)
FITS_PER_CELL = FOLDS * SEEDS             # one "cell" = one arm at one K

SEM = ["child×parent", "child×article"]   # child×article counted semantic (L2.2, decided)
SOCIAL = {1: ["author×article", "author×affiliation", "article×journal"],
          2: ["article×affiliation", "article×article", "article×journal"]}

JOURNAL_ARMS = {  # L2.1 addendum: weighting of article×journal, open until this ablation
    "J1 no journal matrix": None,
    "J2 binary (current, P4)": ("binary", 1.0),
    "J3a binary, multiplier 1/4": ("binary", 0.25),
    "J3b binary, multiplier 1/8": ("binary", 0.125),
    "J4 journal column × idf": ("idf", 1.0),
    "J5 columns × 1/√n, singleton journals missing": ("1/sqrt(n)", 1.0),
}


def weights(approach, journal=("binary", 1.0), semantic_split=(0.5, 0.5), drop=()):
    """Relation weights by the decided rule. semantic_split: shares of the semantic half (A0 0.5/0.5, A1 0.7/0.3)."""
    w = {m: 0.5 * s for m, s in zip(SEM, semantic_split)}
    soc = [m for m in SOCIAL[approach] if m not in drop and not (m == "article×journal" and journal is None)]
    raw = {m: (journal[1] if m == "article×journal" else 1.0) for m in soc}
    tot = sum(raw.values())
    w.update({m: 0.5 * v / tot for m, v in raw.items()})
    return w


def fmt(w):
    return ", ".join(f"{m} {v:.3f}" for m, v in w.items())


stages = []
count = {}

# Stage 1 — journal ablation, both approaches, A0, with article×article, at K_PROBE
s1 = []
for a in (1, 2):
    for name, j in JOURNAL_ARMS.items():
        s1.append((f"approach {a} · {name}", weights(a, j), K_PROBE))
count["stage 1"] = sum(len(k) for _, _, k in s1) * FITS_PER_CELL
stages.append(("STAGE 1 — journal ablation (A0; approach 2 with article×article)", s1,
               "DECISION D1 (owner): weighting of article×journal, from: agreement of article memberships between arms "
               "(Track A/B, chance-adjusted; J1 the reference); each matrix's realised share; journal–community "
               "association vs shuffled journals; held-out prediction of the other matrices. Pre-registration of "
               "these outcomes is still TO DO (as done for L2.4/L2.5). → chosen arm J* is used in stages 2–3."))

# Stage 2 — K search at J*, both approaches
s2 = [(f"approach {a} · J* (chosen in D1)", weights(a), K_GRID) for a in (1, 2)]
count["stage 2"] = sum(len(k) for _, _, k in s2) * FITS_PER_CELL
stages.append(("STAGE 2 — choice of K at J* (A0)", s2,
               "DECISION D2: K₁ (approach 1) and K₂ (approach 2) — out-of-sample fit on the combined held-out score of "
               "each approach's matrices with hidden cells (incl. its own affiliation matrix; owner 2026-11-02: non-semantic "
               "matrices stay pertinent within an approach), with Track A/B stability against chance as the gate "
               "(design doc §6). Cells of stage 1 at J* and K in K_PROBE are reused. (weights shown for J2 binary)"))

# Stage 3 — three pre-registered comparisons at J*, at K₁ and K₂
s3 = [("3a approach 1 vs approach 2 — reuses stage-2 fits at K₁ and K₂", None, []),
      ("3b A1 arm, approach 1", weights(1, semantic_split=(0.7, 0.3)), ["K₁", "K₂"]),
      ("3b A1 arm, approach 2", weights(2, semantic_split=(0.7, 0.3)), ["K₁", "K₂"]),
      ("3c approach 2 without article×article (arm N)", weights(2, drop=("article×article",)), ["K₁", "K₂"])]
count["stage 3"] = sum(len(k) for _, _, k in s3) * FITS_PER_CELL
stages.append(("STAGE 3 — pre-registered comparisons at J*, at K₁ and K₂", s3,
               "DECISIONS: 3a L2.5 classes (approach 1 better / approach 2 better / equivalent / inconclusive) × "
               "membership agreement → reading per the pre-registered table, ranking by the owner; 3b A1 criteria "
               "(A1 never replaces A0 automatically); 3c L2.4 classes (adds information / misleads / redundant / "
               "inconclusive) × team-shaped communities → keep or fallback to a regulariser. (weights shown for J2 binary)"))

LATER = ["loss β = 1 (KL) vs β = 2: pre-registered loss-neutral comparison (2026-10-31); if KL is chosen, L2.1/L2.2 are rechecked",
         "final models refitted on all cells after settings are chosen (open)",
         "background component (deferred robustness arm; planted test first)",
         "mechanisms and λ values of §8.2 (regulariser, penalties, Optuna or grid) — must be fixed BEFORE stage 1; "
         "the hold-out scheme comparison of 2026-10-17 to be confirmed as done or still needed"]

print(f"Fits per cell = {FOLDS} hold-out groups × {SEEDS} seeds = {FITS_PER_CELL}\n")
print("PREREQUISITES (before stage 1): M1 fit code; §8.2 mechanisms and λ; " + LATER[3].split(" — ")[1] + "\n")
for title, arms, decision in stages:
    print(title)
    for i, (name, w, ks) in enumerate(arms):
        last = i == len(arms) - 1
        print(("└── " if last else "├── ") + name + (f"   K = {ks}" if ks else ""))
        if w:
            print(("    " if last else "│   ") + "    weights: " + fmt(w))
    print("    ▼ " + decision + "\n")
print("LATER STAGES (not yet designed):")
for x in LATER[:3]:
    print("  • " + x)
print("\nFit counts (before reuse): " + ", ".join(f"{k} {v:,}" for k, v in count.items())
      + f"; total {sum(count.values()):,}")
print(f"For comparison, the unstaged plan (every journal arm × both approaches × full K grid, plus A1 and arm N at J2): "
      f"{(2 * len(JOURNAL_ARMS) * len(K_GRID) + 2 * 2 + 2) * FITS_PER_CELL:,}")
