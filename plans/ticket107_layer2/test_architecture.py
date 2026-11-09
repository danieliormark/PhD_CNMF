# Ticket 107, layer 2: architecture of the first M1 fits (Psychology trial) as a staged decision tree.
# Proposal 2026-11-02; updated 2026-11-09 with §8.2 step 12 (mechanism checks placed by stage). Nothing here runs a fit. It enumerates the arms, computes each arm's relation
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
count = {}      # fits that always run (at SEEDS)
cond = {}       # fits that run only under a condition (label -> fits)
ONE_GROUP = SEEDS  # graded swap: one hold-out group × S seeds

# Build step (§8.2 step 12, owner 2026-11-08/09)
count["build"] = 16 + 1
stages.append(("BUILD STEP — before any real-data fit", [
    ("fit code with both losses, masked loss, new facets, logging every ~50 epochs", None, []),
    ("unit tests (seconds, no fits): autograd vs finite differences (both losses, floor, W, S3 if built); masked-cell "
     "invariance; tie-weight parts and presence unchanged under the scale freedom (ticket 79) — binding: failure = bug", None, []),
    ("code-check world (~16 small fits): K* = 6 planted, noiseless + Poisson + Gaussian, real mask, floor and W off; "
     "pass criteria recorded 2026-11-08 — binding: failure = bug or optimiser settings; never evidence about the method", None, []),
    ("time one fit → set S (owner: 10–20); check the RDS archive quota (≈ 50–70 GB)", None, [])],
    "DECISION: S from the timing; the deferred floor-pass grid (option a/b/c) is decided with it."))

# Stage 0 — loss comparison (owner 2026-11-05/06)
s0 = [(f"pass 1 · approach {a} · J2 binary · loss {b}; every in-loop term off", weights(a), K_PROBE)
      for a in (1, 2) for b in ("β = 2 (squared error)", "β = 1 (KL)")]
s0 += [("pass 1 diagnostics (no extra fits): loss measures S0-M1–M5 and per-entity AUC; off-diagonal share (tie-weight units + squared "
        "core-cell form; → O1/O3); membership stability every ~50 epochs (→ warm-up epoch); floor diagnostics (between/within-article "
        "spread, mosaic vs blend, share of shared hyperedges > 0.9, shortfall; → keep/drop floor, floor pass?); excess spread and S3 gate 1 "
        "(→ S3 slot?); presence, mono-domain and mono-facet communities with weight, size, seed recurrence, full distribution, cause split; "
        "ghosts, N_eff ÷ K, reproduced/on-target share, placed entities, empty columns at 5 references; tie alignment by degree band × W "
        "stratum; would-be vetoes and convergence counted at fractions 0.1/0.2/0.25/0.5, per-seed failure curves over K; loadings and "
        "engagement distributions (→ item (l))", None, []),
       ("graded cross-domain swap, bare (after pass 1): p = 0.25/0.5/0.75/1 (p = 0 = pass 1), each loss's best K, both approaches, "
        "hold-out group 1 × S seeds, own random swap per seed (owner 2026-11-09)", None, ["best K per loss"]),
       ("OPTIONAL slot between passes 1 and 2: floor pass (only if the floor is kept and the losses differ on its diagnostic; "
        "grid deferred: a staged ≈ 3,000 / b approach 1 only ≈ 1,600 / c full 36 settings 7,200); S3 slot (only if gate 1 passes; "
        "calibrated with the floor; grid sized then)", None, []),
       ("pass 2 · λ_child ∈ {0.03, 0.1, 0.3} per loss at its best K", None, ["K*"]),
       ("pass 3 · each loss at its own λ*_child at the candidate K (coarse K inside either loss's band)", None, ["~3 candidate K"])]
count["stage 0"] = 4 * len(K_PROBE) * FITS_PER_CELL + 2 * 2 * 3 * FITS_PER_CELL + 2 * 2 * 3 * FITS_PER_CELL + 4 * 2 * 2 * ONE_GROUP
cond["stage 0: K = 32 if a band reaches 20"] = 2 * 2 * FITS_PER_CELL
cond["stage 0: graded swap at each further candidate K if the curve is ambiguous (per K)"] = 4 * 2 * 2 * ONE_GROUP
cond["stage 0: floor pass (grid deferred; option a)"] = 3000
stages.append(("STAGE 0 — loss comparison (J2, A0, hold-out groups 1–5)", s0,
               "DECISION β (pre-registered 2026-11-06): gate (stability, counted per seed by the 0.2 rule) → membership agreement → "
               "held-out ranking → stability → squared error by default; flags to the owner; re-check at the final K₁, K₂ with terms on. "
               "Binding in stage 0: stability gate and convergence check for the loss choice only; ghost and alignment vetoes reported as "
               "would-be. Decided from stage 0 by the owner: O1/O3; response to mono-domain communities; item (l); stage-1 fraction; "
               "empty-column reference."))

# λ calibration under the chosen loss (§8.2 steps 2–7)
cal = [("floor, if kept: λ × γ with ω and β arms; cost rule (held-out within one SE of λ = 0), hidden-tie match, false-positive check", None, []),
       ("S3, if gate 1 passed: gate 2 (cost rule, hidden-tie match, false-positive check)", None, []),
       ("O3 (redesigned off-diagonal term), if chosen", None, []),
       ("λ_child per coarse K with the floor on (W rule: larger λ within one SE of the best; stability not worse than at λ = 0)", None, K_PROBE),
       ("graded swap per kept term at its chosen λ (W; floor if kept), λ = 0 reference from stage 0: a flatter curve = the term "
        "manufactures heterogeneity → limits λ (W4)", None, ["best K"]),
       ("interaction test: each active term's best value with the others at neighbouring grid values; a move > 1 grid step → "
        "single-objective search with the cost rule as constraint", None, [])]
count["λ calibration"] = 3 * len(K_PROBE) * FITS_PER_CELL * 2 + 4 * 2 * ONE_GROUP
cond["λ calibration: floor recalibration, S3, O3, interaction test (sized after stage 0)"] = 3000
cond["λ calibration: graded swap for the floor, if kept"] = 4 * 2 * ONE_GROUP
stages.append(("λ CALIBRATION — chosen loss, J2, A0, groups 1–5 (order decided)", cal,
               "DECISION: each λ by its pre-registered rule; a term failing W1–W4 at every λ is dropped. NOTE: W is on from here, so "
               "stage-0 pass-1 fits are not reused as stage-1 J2 cells."))

# Stage 1 — journal ablation, both approaches, A0, with article×article, at K_PROBE
s1 = []
for a in (1, 2):
    for name, j in JOURNAL_ARMS.items():
        s1.append((f"approach {a} · {name}", weights(a, j), K_PROBE))
count["stage 1"] = sum(len(k) for _, _, k in s1) * FITS_PER_CELL * 2   # all 10 hold-out groups (owner 2026-11-05)
stages.append(("STAGE 1 — journal ablation (chosen loss and λ's; A0; all 10 hold-out groups)", s1,
               "DECISION D1 (pre-registered, D1_journal_preregistration.md): J* per approach; read at all candidate K, also reported "
               "with vetoed K dropped and the veto outcome per journal arm (auto-accepted). BINDING on the coarse band: stability gate "
               "(Track A/B on weighted memberships; A over placed entities), ghost veto, alignment veto, convergence check — each "
               "counted: a group fails if > 20% of its seeds fail; K fails if > 20% of groups or > 20% of all fits fail (provisional, "
               "fixed from stage 0 before stage-1 results are read); non-convergence alone → owner before exclusion."))

# Stage 2 — K search at J*, both approaches
s2 = [(f"approach {a} · J*: refine every integer K inside the band(s) of the coarse curve + K = 32 always; extend to ~50 if 32 is "
       f"best or in the band", weights(a), ["~" + str(K_REFINE) + " new K", 32]) for a in (1, 2)]
s2 += [("re-check of the loss at K₁, K₂ with final terms on (losing loss refitted, λ transferred by loss share)", None, ["K₁", "K₂"]),
       ("re-check of D1 at K₁, K₂: J1 and runner-up (J* reused)", None, ["K₁", "K₂"])]
count["stage 2"] = 2 * (K_REFINE + 1) * FITS_PER_CELL + 2 * FITS_PER_CELL + 2 * 2 * FITS_PER_CELL
stages.append(("STAGE 2 — choice of K at J* (groups 1–5), coarse-to-fine", s2,
               "DECISION D2 (K protocol §6.8 + step 11): band = all K with score ≤ best + SE of the best (computed from all K) → exclude "
               "K failing the gate, a veto or convergence (0.2 counting) → inspection and text check choose among the rest → if none: "
               "largest smaller K (below the band on the K axis) that passes everything, flagged → if none: 'no K qualifies' with the "
               "status of every K above the band, owner decides. Contradictions in the re-checks go to the owner."))

# Stage 3 — comparisons at J*, at K₁ and K₂; final refit
s3 = [("3a approach 1 vs approach 2 — reuses stage-2 fits at K₁ and K₂", None, []),
      ("3b A1 arm, approach 1", weights(1, semantic_split=(0.7, 0.3)), ["K₁", "K₂"]),
      ("3b A1 arm, approach 2", weights(2, semantic_split=(0.7, 0.3)), ["K₁", "K₂"]),
      ("3c approach 2 without article×article (arm N)", weights(2, drop=("article×article",)), ["K₁", "K₂"]),
      ("final refit on all data (no hidden cells) at the chosen settings, S seeds per approach; reported model = converged seed with "
       "the lowest training loss that passes both vetoes (else flagged, next seed); its Track A/B agreement with the other seeds and "
       "the medoid seed reported (owner 2026-11-09)", None, ["K₁", "K₂"]),
      ("OPTIONAL graded swap at the final settings (confirmation only)", None, ["K₁", "K₂"])]
count["stage 3"] = 3 * 2 * FITS_PER_CELL + 2 * SEEDS
cond["stage 3: optional graded swap"] = 4 * 2 * ONE_GROUP
stages.append(("STAGE 3 — pre-registered comparisons at J*, at K₁ and K₂; final models", s3,
               "DECISIONS: 3a L2.5 classes × membership agreement (an approach failing the stability gate is not ranked); 3b A1 "
               "criteria (A1 never replaces A0 automatically); 3c L2.4 classes × team-shaped communities. (weights shown for J2 binary)"))

LATER = ["background component (deferred robustness arm; planted test first)",
         "ticket 35 (reformulated): how several outer scores would be combined, only if stage 0 shows one is needed"]

print(f"Fits per cell = {FOLDS} hold-out groups × {SEEDS} seeds = {FITS_PER_CELL} (stage 1: 10 groups)\n")
for title, arms, decision in stages:
    print(title)
    for i, (name, w, ks) in enumerate(arms):
        last = i == len(arms) - 1
        print(("└── " if last else "├── ") + name + (f"   K = {ks}" if ks else ""))
        if w:
            print(("    " if last else "│   ") + "    weights: " + fmt(w))
    print("    ▼ " + decision + "\n")
print("NOT YET DESIGNED:")
for x in LATER:
    print("  • " + x)
fixed = sum(count.values())
print("\nFits that always run: " + ", ".join(f"{k} {v:,}" for k, v in count.items()) + f"; total {fixed:,}")
print("Conditional fits: " + "; ".join(f"{k} {v:,}" for k, v in cond.items()) + f"; all conditions {sum(cond.values()):,}")
print(f"Range at S = {SEEDS}: {fixed:,} to {fixed + sum(cond.values()):,} (floor pass option c instead of a: +4,200)")
print("Scaling with S (fits ∝ S): " + ", ".join(f"S={s_}: {fixed * s_ // SEEDS:,}–{(fixed + sum(cond.values())) * s_ // SEEDS:,}" for s_ in (10, 15, 20)))
