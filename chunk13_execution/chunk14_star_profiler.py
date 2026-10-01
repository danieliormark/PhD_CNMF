"""CHUNK 14 (STAR PROFILER), v9-upgraded replacement for toy_large.ipynb's
"CHUNK 14c (STAR PROFILER)" cell (and the rest of the ad-hoc Chunk 14 family:
14A/14B/14D/14e/14.5).

Why this replaces the notebook version, not just updates it:
- The old cell reads `chunk13_execution/outputs/grid_search_c_class/*.npz`, a
  path written by chunk13v5-v8.1 and never produced by chunk13v9 (zero
  matches in chunk13v9.py); this one reads v9's actual archived output
  (results/<PIPELINE_VERSION>/<config>/K_<K>/pareto_models/<trial>/).
- The old cell's core measure (DCR_sem/Dev) used U_raw's column L2 norms,
  which FINDINGS.md section "A separate, genuinely defective measure exists
  in the notebook" proved is exactly U_scales -- ticket 79's undetermined
  gauge freedom -- and showed it moves (0.1971 -> 0.2052) under a
  transformation that leaves the model provably unchanged. This script uses
  only chunk13v9.py's own current, gauge-invariant functions (Z_scaled-based
  mass, U_prob-based domain balance and hoarding), imported directly from
  the live module rather than reimplemented, so it can never silently drift
  from what the pipeline actually computes.
- The old cell returned one global number per model (Defect 2: a
  constructed all-social-in-community-0 / all-semantic-in-community-1 case
  scored a "perfect" Dev=0.0000 despite zero heterogeneous communities).
  This script is per-community throughout, and -- the part no prior script
  in this project produces -- lists each community's actual top entities
  (article titles, author names, journal names, and the semantic atoms/
  hyperedges themselves, which are self-describing strings like
  'language_model/C/en') so a human can judge whether a community looks
  substantively coherent, not just whether its scores clear a threshold.

Usage:
    python chunk14_star_profiler.py --config C1 --K 3 --trial trial_0025 \
        --out profiles/C1_K3_trial_0025.md
"""
import argparse
import importlib.util
import json
import os
import pickle

import numpy as np
import torch

TOP_N = 10

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--K", type=int, required=True)
ap.add_argument("--trial", required=True, help="e.g. trial_0025")
ap.add_argument("--results", default="results/v9.2.t1_v2")
ap.add_argument("--decoders", default="/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_epistemic_decoders_global_v2.pkl")
ap.add_argument("--data", default="/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1_v2.pkl")
ap.add_argument("--out", default=None)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("c13", "chunk13v9.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

raw_data = m.load_and_validate_data(args.data)
soc_keys, sem_keys, anchor_keys = m.get_active_facets(args.config)
active_matrices = {k for k in (soc_keys + sem_keys) if k in raw_data}
presence_masks = m.build_presence_masks(raw_data, soc_keys, sem_keys)

decoders = pickle.load(open(args.decoders, "rb"))

trial_dir = os.path.join(args.results, args.config, f"K_{args.K}", "pareto_models", args.trial)
meta = json.load(open(os.path.join(trial_dir, "model_metadata.json")))
U = {f: np.array(t) for f, t in torch.load(os.path.join(trial_dir, "U_matrices.pt"), weights_only=False).items()}
Z = {r: np.array(t) for r, t in torch.load(os.path.join(trial_dir, "Z_core.pt"), weights_only=False).items()}
U_prob = m.compute_probability_distributions(U)
K = args.K


def label_for(facet, idx):
    """Human-readable label for one entity. Semantic facets' own map keys
    are already self-describing hyperedge text; auth/art/journ go through
    the name/title decoders; affil has no resolved name available."""
    rev_map = _rev_maps.get(facet)
    if rev_map is None or idx not in rev_map:
        return f"<{facet}#{idx}>"
    key = rev_map[idx]
    if facet == "auth":
        return decoders["author_meta"].get(key, key)
    if facet == "art":
        return decoders["article_meta"].get(key, key)
    if facet == "journ":
        return decoders["journal_meta"].get(key, key)
    if facet == "affil":
        return f"{key} (institution id, no name in decoders)"
    return key  # semantic facets: the map key IS the readable hyperedge text


_rev_maps = {f: {v: k for k, v in decoders["maps"][f].items()} for f in decoders["maps"]}
_rev_maps["art"] = {v: k for k, v in decoders["maps_t1_art"].items()}  # this run uses T1 article indexing


def top_entities(facet, k, n=TOP_N):
    if facet not in U_prob or facet not in presence_masks:
        return []
    live = presence_masks[facet]
    col = U_prob[facet][:, k]
    idxs = np.where(live)[0]
    idxs = idxs[np.argsort(-col[idxs])][:n]
    return [(label_for(facet, int(i)), float(col[i])) for i in idxs]


# --- recompute the four diagnostics' full per-community detail (verified
# exact against chunk13v9.py's own saved scalars in FINDINGS §29) ---
def collapse_shares():
    shares = []
    for rel_key, Zm in Z.items():
        f1, f2 = m.RELATION_MAP[rel_key]
        shares.append(m._relation_community_share(Zm, U[f1], U[f2], m.STRUCTURE_SCORE_THRESHOLD))
    cs = np.mean(shares, axis=0)
    total = cs.sum()
    return cs / total if total > 1e-15 else np.full(K, 1.0 / K)


def coherence_vector():
    active_anchors = [r for r in Z if r in anchor_keys]
    if not active_anchors:
        return None
    scores = []
    for k in range(K):
        ratios = []
        for a in active_anchors:
            Zm = Z[a]
            denom = 0.5 * (Zm[k, :].sum() + Zm[:, k].sum()) + 1e-12
            ratios.append(np.clip(Zm[k, k] / denom, 0.0, 1.0))
        scores.append(np.mean(ratios))
    return np.array(scores)


def domain_balance_vector():
    soc_f = sorted(f for f in presence_masks if m.FACET_DOMAIN.get(f) == "social" and f in U_prob and presence_masks[f].sum() > 0)
    sem_f = sorted(f for f in presence_masks if m.FACET_DOMAIN.get(f) == "semantic" and f in U_prob and presence_masks[f].sum() > 0)
    if not soc_f or not sem_f:
        return None

    def wshare(facets):
        tot = sum(int(presence_masks[f].sum()) for f in facets)
        acc = None
        for f in facets:
            live = U_prob[f][presence_masks[f]]
            term = int(presence_masks[f].sum()) * live.mean(axis=0)
            acc = term if acc is None else acc + term
        return acc / tot

    soc_share, sem_share = wshare(soc_f), wshare(sem_f)
    return soc_share / (soc_share + sem_share + 1e-12)


community_share = collapse_shares()
coherence = coherence_vector()
r_k = domain_balance_vector()
ghost_threshold = 0.5 * (1.0 / K)

lines = []
lines.append(f"# Star profile: {args.config} / K={K} / {args.trial}\n")
lines.append(f"Replaces `toy_large.ipynb` cell 58 (\"CHUNK 14c STAR PROFILER\") -- see "
             f"`chunk14_star_profiler.py`'s module docstring for why the old version's "
             f"core measure was retired, not just updated.\n")
lines.append(f"`math_loss`={meta['optuna_math_loss']:.4f}  `sociological_penalty`={meta['optuna_soc_penalty']:.4f}  "
             f"community_share (relation-level mass) sums to {community_share.sum():.3f}\n")

for k in range(K):
    lines.append(f"\n## Community {k}\n")
    flags = []
    if community_share[k] < ghost_threshold:
        flags.append(f"**GHOST** (share {community_share[k]:.3f} < {ghost_threshold:.3f} = 0.5x fair share)")
    if community_share[k] > 0.60:
        flags.append(f"**MASS-DOMINANT** (share {community_share[k]:.3f} > 0.60 ceiling)")
    lines.append(f"- Relation-level mass share: **{community_share[k]:.3f}**" + (" -- " + "; ".join(flags) if flags else ""))
    if r_k is not None:
        lines.append(f"- Domain mix (social share r_k): **{r_k[k]:.3f}** (0.5 = even; caveat: FINDINGS §25, this measure does not track true domain balance on this corpus)")
    if coherence is not None:
        lines.append(f"- Anchor-relation coherence: **{coherence[k]:.3f}** (weighted 0 in sociological_penalty; shown for reference only, §4.17)")

    for facet, heading in [("art", "Top articles"), ("auth", "Top authors"), ("journ", "Top journals"),
                           ("core_atom", "Top core atoms"), ("core_child_he", "Top child hyperedges"),
                           ("parent_he", "Top parent hyperedges"), ("cousin_he", "Top cousin hyperedges"),
                           ("fringe_atom", "Top fringe atoms")]:
        top = top_entities(facet, k)
        if not top:
            continue
        lines.append(f"\n**{heading} (by U_prob loading on this community):**\n")
        for label, p in top:
            label_str = label if len(label) < 160 else label[:157] + "..."
            lines.append(f"- {p:.3f} -- {label_str}")

report = "\n".join(lines)
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(report)
    print(f"wrote {args.out}")
else:
    print(report)
