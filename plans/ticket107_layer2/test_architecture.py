# Ticket 107, layer 2: architecture of the first M1 fits (Psychology trial) as a staged decision tree.
# Proposal for the owner, 2026-11-02; nothing here runs a fit. It enumerates the arms, computes each arm's relation
# weights by the decided rule (w = 0.5 / n_domain; journal multiplier applied inside the social half, which is then
# rescaled to 0.5 — owner, 2026-11-02), counts fits, and prints the tree with the decision taken at each node.
# Run: python3 test_architecture.py  (writes nothing; redirect to keep a copy)

FOLDS = 5                                 # hold-out groups used (5 of 10)
SEEDS = 10                                # per hold-out group; owner 2026-11-03: 10-20, as many as a 1-2 week budget allows
K_PROBE = [3, 5, 8, 13, 20]               # stage 1 = the coarse grid (x~1.6 steps; K_max from Bethe Hessian counts)
K_REFINE = 6                              # stage 2: about this many new K values, every integer inside the winning band(s)
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
s2 = [(f"approach {a} · J* (chosen in D1): refine every integer K inside the band(s) of the coarse curve "
       f"(stage-1 cells at J*); extend to 25, 32 if the best coarse K is 20", weights(a), ["~" + str(K_REFINE) + " new K"])
      for a in (1, 2)]
count["stage 2"] = 2 * K_REFINE * FITS_PER_CELL
stages.append(("STAGE 2 — choice of K at J* (A0), coarse-to-fine", s2,
               "DECISION D2: K₁ (approach 1) and K₂ (approach 2) — out-of-sample fit on the combined held-out score of "
               "each approach's matrices with hidden cells (incl. its own affiliation matrix; owner 2026-11-02: non-semantic "
               "matrices stay pertinent within an approach), with Track A/B stability against chance as the gate "
               "(design doc §6). The band = K values within one standard error of the best held-out score; every interval between "
               "coarse points touching a band is filled in (several intervals if the band is split). (weights shown for J2 binary)"))

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
print(f"For comparison, a full grid K = 2-20 at J* in stage 2 instead of refinement (14 K beyond the coarse 5): {2 * 14 * FITS_PER_CELL:,} fits")
print("Total fits as a function of seeds S: " + ", ".join(f"S={s_}: {sum(count.values()) // SEEDS * s_:,}" for s_ in (10, 15, 20)))
