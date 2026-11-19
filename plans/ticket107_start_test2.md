# Ticket 107 — start test 2: room for links between different communities (prepared 2026-11-19)

## 1. Why
Under KL every core matrix ends exactly diagonal when the social cores start diagonal, and the semantic ones with
either start (RL-275): a tie between communities can then be rebuilt only through the two ends' shared memberships.
The owner decided that the model class must allow links between different communities (option ii, 2026-11-19: "A
person mostly attributed to community k can coauthor with an author mostly attribtued to community l"). The diagonal
cores are a first-order optimum (no floor cell has a negative gradient), but their off-diagonal cells start at 1e-4,
three orders above the floor. Question: do links between communities survive when the start gives them a real chance,
and what does that cost or gain in fit? Plus the owner's two checks: (1b1) whether relations other than article×article
under-rebuild cross-community ties (closure ratio; RL-270, RL-277) and (1b2) whether a start that helps some relations
costs article×article more than it gains them (RL-276).

## 2. Design (fit/plan.py start_test2; 130 fits)
KL, K = 13, hold-out group 1, seeds 1–10, terms off, fraction 0.03, float32, 8,000-epoch ceiling. All arms start
from the same membership start per seed (hybrid NNDSVD start, social cores diagonal as in the start test), then:
| Arm | Change to the start | Fits |
|---|---|---|
| S1 | article×article keeps the hybrid start's random core | approach 2: 10 |
| S2_5 / S2_10 / S2_20 | every social core: off-diagonal cell (k,l) = 5 / 10 / 20 % of √(d_k d_l), d = its diagonal | 3 × 2 approaches × 10 = 60 |
| S3_5 / S3_10 / S3_20 | as S2, the semantic cores (child×article, child×parent) too | 60 |
Each changed core matrix is rescaled so the relation's starting reconstruction keeps its size (`init.offdiagonal_start`).
References (existing fits, same seeds): diagonal (start test) and random (pass 1).

## 3. Readings (fit/start_test2_analysis.py; aligned numbering)
Per relation: training loss (and w_r × loss, which sums to the total); closure ratio; off-diagonal share of
reconstructed tie weight; share of off-diagonal core cells at the floor; diagonal-rule share; crossing share,
co-membership on crossing ties, spread of the entities at the ends of crossing ties. Per fit: data loss, held-out
existence and value, child×article requirement (communities > 0.5), convergence. Per arm: stability (chance-adjusted
Track A/B). Paired by seed against the diagonal and the random arm; beyond 2 SE = a difference.

## 4. Expected outcomes (stated before reading)
1. Off-diagonal cells return to the floor at every level (off-diagonal share ≈ 0), loss as the diagonal arm → under KL
   the data prefer diagonal cores from these starts; option (ii) cannot be reached through the start. Next: owner
   decides whether to accept diagonal cores as the data's answer or look for other means.
2. Off-diagonal weight survives and training loss is equal (within 2 SE) or lower than the diagonal arm → the diagonal
   start forced its solution. Choose among the levels (proposal: lowest paired training loss; see §5); expect the
   closure ratio of article×article to move toward 1 and the entities at the ends of crossing ties to sharpen (the
   mixing was compensation).
3. Off-diagonal weight survives but training loss is higher beyond 2 SE → two kinds of optima, the diagonal one fits
   better; a trade-off between option (ii) and fit — owner's decision.
4. Off-diagonal share grows with the starting level without the loss settling on one level → the solution depends on
   the start; the start becomes a modelling choice to be fixed by an explicit criterion.
S1 (1b2): adopted for article×article if its closure ratio rises toward the random arm's (0.84) while the other
relations' weighted losses stay within 2 SE of the diagonal arm's; S2/S3 answer the same question for all relations.
S3 vs S2: whether the semantic cores keep off-diagonal weight; if they do, the child×article requirement (> 0.5 per
community) becomes a real test (it is met by construction while those cores are diagonal).

## 5. Choosing the start (proposal, owner to confirm when reading)
Starts are model-design choices, not restarts. Proposal: among arms whose fits converge and meet the child×article
requirement, prefer the lowest paired training loss; where arms are within 2 SE, prefer the one whose closure ratios
are closest to 1 and whose stability is higher. Held-out existence and value are reported beside.

## 6. Results
(to be filled)
