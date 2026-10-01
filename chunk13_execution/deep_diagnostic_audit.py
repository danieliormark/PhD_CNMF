"""Recompute the full per-community detail behind sociological_penalty's four
components, for every archived model, directly from the saved U_matrices.pt /
Z_core.pt tensors -- no refitting. Answers five questions the saved JSON
summaries can't, because they only keep scalar aggregates:

1. Domain balance (dev_k) -- full per-community r_k, not just the mean.
2. Item-to-community attribution given the higher-order item (semantic
   Penalty_A) -- currently averaged into semantic_pen and never saved alone.
3. Monopolisation -- two distinct levels: collapse_pen's relation-level mass
   concentration AND semantic Penalty_B's entity-level ubiquity hoarding.
4. Within- vs between-community relations (coherence) -- full per-community
   cohesion vector, not just the weakest one.
5. Ghost communities -- the actual community_share vector (ticket 86's
   K-relative threshold), not just max_share.
"""
import importlib.util
import json
import os

import numpy as np
import torch

RESULTS = "results/v9.2.t1_v2"
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6"]
GHOST_RATIO = 0.5  # ticket 86's threshold: share < GHOST_RATIO * (1/K)

spec = importlib.util.spec_from_file_location("c13", "chunk13v9.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

raw_data = m.load_and_validate_data(
    "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1_v2.pkl"
)
dimensions = raw_data["dimensions"]


def full_collapse_shares(U_final, Z_final):
    shares = []
    for rel_key, Z in Z_final.items():
        f1, f2 = m.RELATION_MAP[rel_key]
        shares.append(m._relation_community_share(Z, U_final[f1], U_final[f2], m.STRUCTURE_SCORE_THRESHOLD))
    community_share = np.mean(shares, axis=0)
    total = community_share.sum()
    return community_share / total if total > 1e-15 else np.full(len(community_share), 1.0 / len(community_share))


def full_coherence(Z_final, anchor_keys):
    active_anchors = [r for r in Z_final if r in anchor_keys]
    if not active_anchors:
        return None
    K = Z_final[active_anchors[0]].shape[0]
    scores = []
    for k in range(K):
        ratios = []
        for a in active_anchors:
            Zm = Z_final[a]
            denom = 0.5 * (Zm[k, :].sum() + Zm[:, k].sum()) + 1e-12
            ratios.append(np.clip(Zm[k, k] / denom, 0.0, 1.0))
        scores.append(np.mean(ratios))
    return np.array(scores)


def full_domain_balance(U_prob, presence_masks):
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


def full_semantic_parts(U_prob, active_matrices, presence_masks, config_id):
    # Faithful copy of evaluate_socio_semantic_reality's body, modified only to
    # return Penalty_A and Penalty_B separately instead of their average.
    target_facets = ["core_child_he", "cousin_he", "core_atom", "fringe_atom"]
    K = U_prob["art"].shape[1]
    part_a_scores = []
    empirical_mass_capture = {f: np.zeros((U_prob[f].shape[0], K)) for f in target_facets if f in U_prob}

    for k in range(K):
        W_art = U_prob["art"][:, k]
        if np.sum(W_art) < 1e-12:
            for f in target_facets:
                if f in U_prob:
                    part_a_scores.append(1.0)
            continue
        pw = {}
        if "M_Parent_Art" in active_matrices:
            pw["parent_he"] = raw_data["M_Parent_Art"].dot(W_art)
        child_paths = []
        if "M_Child_Art" in active_matrices:
            child_paths.append(raw_data["M_Child_Art"].dot(W_art))
        if "M_Child_Parent" in active_matrices and "parent_he" in pw:
            child_paths.append(raw_data["M_Child_Parent"].dot(pw["parent_he"]))
        if child_paths:
            pw["core_child_he"] = sum(child_paths)
        cousin_paths = []
        if "M_Cousin_Art" in active_matrices:
            cousin_paths.append(raw_data["M_Cousin_Art"].dot(W_art))
        if "M_Cousin_Parent" in active_matrices and "parent_he" in pw:
            cousin_paths.append(raw_data["M_Cousin_Parent"].dot(pw["parent_he"]))
        if "M_Cousin_Child" in active_matrices and "core_child_he" in pw:
            cousin_paths.append(raw_data["M_Cousin_Child"].dot(pw["core_child_he"]))
        if cousin_paths:
            pw["cousin_he"] = sum(cousin_paths)
        if "M_Atom_Child" in active_matrices and "core_child_he" in pw:
            pw["core_atom"] = raw_data["M_Atom_Child"].dot(pw["core_child_he"])
        if "M_Fringe_Cousin" in active_matrices and "cousin_he" in pw:
            pw["fringe_atom"] = raw_data["M_Fringe_Cousin"].dot(pw["cousin_he"])

        for f in target_facets:
            if f not in U_prob:
                continue
            if f not in pw:
                part_a_scores.append(1.0)
                continue
            W_f = pw[f]
            empirical_mass_capture[f][:, k] = W_f
            total_w = np.sum(W_f)
            if total_w < 1e-12:
                part_a_scores.append(1.0)
                continue
            model_probs = U_prob[f][:, k]
            if presence_masks is not None and f in presence_masks and presence_masks[f].any():
                baseline = np.mean(model_probs[presence_masks[f]])
            else:
                baseline = np.mean(model_probs)
            weighted_avg = np.sum(W_f * model_probs) / total_w
            score = 1.0 if baseline < 1e-12 else (max(0.0, baseline - weighted_avg) / baseline) ** 2
            part_a_scores.append(score)

    hoarding = []
    for f in target_facets:
        if f not in U_prob or f not in empirical_mass_capture:
            continue
        E = empirical_mass_capture[f]
        row_sums = E.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > 1e-12
        if presence_masks is not None and f in presence_masks:
            valid = valid & presence_masks[f]
        if not valid.any():
            continue
        P = np.zeros_like(E)
        P[valid] = E[valid] / row_sums[valid]
        entropy = -np.sum(P * np.log(P + 1e-12), axis=1)
        norm_entropy = entropy / np.log(K)
        max_loading = U_prob[f].max(axis=1)
        base_pen = (np.maximum(0.0, max_loading - m.MAX_MONOPOLY) / (1.0 - m.MAX_MONOPOLY)) ** 2
        hoarding.append(np.mean((base_pen * norm_entropy)[valid]))

    penalty_a = float(np.mean(part_a_scores)) if part_a_scores else 1.0
    penalty_b = float(np.mean(hoarding)) if hoarding else 0.0
    return penalty_a, penalty_b


rows = []
for c in CONFIGS:
    soc_keys, sem_keys, anchor_keys = m.get_active_facets(c)
    active_matrices = {k for k in (soc_keys + sem_keys) if k in raw_data}
    presence_masks = m.build_presence_masks(raw_data, soc_keys, sem_keys)

    for K in [2, 3, 4, 5, 6]:
        k_dir = os.path.join(RESULTS, c, f"K_{K}", "pareto_models")
        manifest_path = os.path.join(k_dir, "experiment_manifest.json")
        if not os.path.exists(manifest_path):
            continue
        manifest = json.load(open(manifest_path))
        for info in manifest["archived_models"]:
            name = info["folder_name"]
            trial_dir = os.path.join(k_dir, name)
            meta = json.load(open(os.path.join(trial_dir, "model_metadata.json")))
            U = {f: np.array(t) for f, t in torch.load(os.path.join(trial_dir, "U_matrices.pt"), weights_only=False).items()}
            Z = {r: np.array(t) for r, t in torch.load(os.path.join(trial_dir, "Z_core.pt"), weights_only=False).items()}
            U_prob = m.compute_probability_distributions(U)

            share = full_collapse_shares(U, Z)
            coh = full_coherence(Z, anchor_keys)
            dom = full_domain_balance(U_prob, presence_masks)
            pa, pb = full_semantic_parts(U_prob, active_matrices, presence_masks, c)

            n_ghosts = int(np.sum(share < GHOST_RATIO * (1.0 / K)))

            rows.append(dict(
                config=c, K=K, trial=name,
                math_loss=meta["optuna_math_loss"], soc_pen=meta["optuna_soc_penalty"],
                converged_stability=("consensus_track_A_unweighted" in meta),
                max_share=float(share.max()), n_ghosts=n_ghosts,
                weakest_coherence=float(coh.min()) if coh is not None else None,
                coherence_spread=float(coh.max() - coh.min()) if coh is not None else None,
                mean_dev_k=float(np.mean(np.abs(dom - 0.5))) if dom is not None else None,
                max_dev_k=float(np.max(np.abs(dom - 0.5))) if dom is not None else None,
                penalty_a=pa, penalty_b=pb,
            ))

json.dump(rows, open("deep_diagnostic_audit_results.json", "w"), indent=2)
print(f"analysed {len(rows)} archived models")
