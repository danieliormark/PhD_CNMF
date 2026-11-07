# Scratch: toy v9.4 archived models — relative presence (half credit x K) in child×article vs child×parent.
import glob, torch, numpy as np
from scipy.stats import spearmanr
R = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/results/v9.4.t1_v2"
def half(U, Z, f, g, rel):
    z = Z[rel].double().numpy(); a = U[f].double().numpy().sum(0); b = U[g].double().numpy().sum(0)
    P = z * np.outer(a, b); t = P.sum()
    return (P.sum(1) + P.sum(0)) / 2 / t
rows = []
for d in sorted(glob.glob(f"{R}/C*/K_*/pareto_models/trial_*")):
    U = torch.load(f"{d}/U_matrices.pt"); Z = torch.load(f"{d}/Z_core.pt")
    if 'M_Child_Art' not in Z or 'M_Child_Parent' not in Z: continue
    K = Z['M_Child_Art'].shape[0]
    ca = half(U, Z, 'core_child_he', 'art', 'M_Child_Art') * K
    cp = half(U, Z, 'core_child_he', 'parent_he', 'M_Child_Parent') * K
    rows.append((d.split('/')[-4], K, ca, cp))
print("models with both relations:", len(rows))
rho = [spearmanr(ca, cp)[0] for _, K, ca, cp in rows if K > 2]
diff = np.concatenate([np.abs(ca - cp) for *_, ca, cp in rows])
ca_all = np.concatenate([r[2] for r in rows]); cp_all = np.concatenate([r[3] for r in rows]); m = (ca_all + cp_all) / 2
print(f"Spearman CA vs CP across communities (K>2): median {np.median(rho):.2f}, IQR {np.percentile(rho,25):.2f}-{np.percentile(rho,75):.2f}, n {len(rho)}")
print(f"|CA - CP| relative presence: median {np.median(diff):.2f}, 90th pct {np.percentile(diff,90):.2f}")
low = ca_all < 0.25
print(f"communities with CA presence < 0.25: {low.sum()} of {len(ca_all)}; of these, summary >= 0.5: {(m[low] >= 0.5).sum()}")
lowp = cp_all < 0.25
print(f"communities with CP presence < 0.25: {lowp.sum()}; of these, summary >= 0.5: {(m[lowp] >= 0.5).sum()}")
print(f"spread of presence across communities (SD): CA {np.mean([r[2].std() for r in rows]):.2f}, CP {np.mean([r[3].std() for r in rows]):.2f}")
