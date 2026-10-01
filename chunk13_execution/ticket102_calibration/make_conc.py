# Builds chunk13v9_conc.py: production chunk13v9.py + the ticket-102 in-loop
# shared-entity concentration term (scratch experiment only, production untouched).
# lambda_conc = 0 must reproduce production exactly (checked separately).
SRC = "/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/chunk13_execution/chunk13v9.py"
DST = "/tmp/claude-766719/-mnt-hum01-home01-p91688di/c37b6c8f-0a98-41fc-8ef0-f898a4724f86/scratchpad/conc/chunk13v9_conc.py"
src = open(SRC).read()


def sub(old, new):
    global src
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new)


sub("""    lambda_domain_balance = params.get('lambda_domain_balance', 0.0)
""", """    lambda_domain_balance = params.get('lambda_domain_balance', 0.0)
    # Ticket 102 (scratch): shared-entity concentration floor.
    lambda_conc = params.get('lambda_conc', 0.0)
    conc_gamma = params.get('conc_gamma', 0.5)
    conc_warmup = int(params.get('conc_warmup', 300))
    conc_dual_stop = bool(params.get('conc_dual_stop', False))
""")

sub("""    U_raw, Z_raw = initialize_tucker_adapted_nndsvd_and_propagate(""", """    CONC_FACETS = ['core_child_he', 'cousin_he', 'core_atom', 'fringe_atom']
    conc_state = {}   # facet -> (live-entity index tensor, frozen s_i tensor), filled at warm-up
    conc_info = {}

    def _conc_weights(Un, Zs, f):
        # Reconstruction mass carried by each community column of facet f:
        # w_k^2 = sum over relations touching f of ||(slice of U_f Z U_g^T through column k)||_F^2.
        # Same quantity as compute_weighted_membership, via K x K Gram matrices.
        w = torch.zeros(K, device=device, dtype=Un[f].dtype)
        for r, z in Zs.items():
            a, b = RELATION_MAP[r]
            if a == f:
                w = w + torch.diagonal(z @ (Un[b].t() @ Un[b]) @ z.t())
            if b == f:
                w = w + torch.diagonal(z.t() @ (Un[a].t() @ Un[a]) @ z)
        w = torch.sqrt(torch.clamp(w, min=0.0))
        return torch.where(w < 1e-3 * w.max(), torch.zeros_like(w), w)

    def _conc_freeze(Un, Zs):
        # s_i: normalised entropy of the community distribution an entity inherits from the
        # articles that use it (article weighted membership pushed down the raw relations,
        # exactly Penalty_B's empirical_mass_capture). Computed once, then held fixed.
        with torch.no_grad():
            Xa = (Un['art'] * _conc_weights(Un, Zs, 'art')).cpu().numpy().astype(np.float64)
        ma = Xa.sum(1)
        live_a = db_presence_masks.get('art', np.ones(len(ma), bool))
        pos = ma[live_a & (ma > 0)]
        eng = ma >= 1e-3 * np.median(pos) if pos.size else np.zeros(len(ma), bool)
        P_art = np.where(eng[:, None], Xa / np.where(ma > 0, ma, 1.0)[:, None], 0.0)
        act = set(active_matrices)
        E = {f: np.zeros((Un[f].shape[0], K)) for f in CONC_FACETS if f in Un}
        for k in range(K):
            W = P_art[:, k]; pw = {}
            if 'M_Parent_Art' in act: pw['parent_he'] = raw_data['M_Parent_Art'].dot(W)
            cp = []
            if 'M_Child_Art' in act: cp.append(raw_data['M_Child_Art'].dot(W))
            if 'M_Child_Parent' in act and 'parent_he' in pw: cp.append(raw_data['M_Child_Parent'].dot(pw['parent_he']))
            if cp: pw['core_child_he'] = sum(cp)
            cu = []
            if 'M_Cousin_Art' in act: cu.append(raw_data['M_Cousin_Art'].dot(W))
            if 'M_Cousin_Parent' in act and 'parent_he' in pw: cu.append(raw_data['M_Cousin_Parent'].dot(pw['parent_he']))
            if 'M_Cousin_Child' in act and 'core_child_he' in pw: cu.append(raw_data['M_Cousin_Child'].dot(pw['core_child_he']))
            if cu: pw['cousin_he'] = sum(cu)
            if 'M_Atom_Child' in act and 'core_child_he' in pw: pw['core_atom'] = raw_data['M_Atom_Child'].dot(pw['core_child_he'])
            if 'M_Fringe_Cousin' in act and 'cousin_he' in pw: pw['fringe_atom'] = raw_data['M_Fringe_Cousin'].dot(pw['cousin_he'])
            for f in E:
                if f in pw: E[f][:, k] = np.asarray(pw[f]).ravel()
        for f, Ef in E.items():
            tot = Ef.sum(1)
            keep = (tot > 1e-12) & db_presence_masks.get(f, np.ones(len(tot), bool))
            if not keep.any():
                continue
            Pe = Ef[keep] / tot[keep, None]
            s = -(Pe * np.log(Pe + 1e-12)).sum(1) / np.log(K)
            idx = np.where(keep)[0]
            conc_state[f] = (torch.from_numpy(idx).to(device), torch.from_numpy(s).to(device=device, dtype=Un[f].dtype))
            conc_info[f] = {"idx": idx, "s": s}

    U_raw, Z_raw = initialize_tucker_adapted_nndsvd_and_propagate(""")

sub("""        recon_loss = 0.0
        sparsity_loss = 0.0
        z_offdiag_loss = 0.0
""", """        conc_loss = None
        if lambda_conc > 0 and epoch >= conc_warmup:
            if not conc_state:
                _conc_freeze(U_norm, Z_scaled)
            terms = []
            for f, (idx, s) in conc_state.items():
                w = _conc_weights(U_norm, Z_scaled, f).detach()
                X = U_norm[f][idx] * w
                p = X / (X.sum(1, keepdim=True) + 1e-12)
                H = -(p * torch.log(p + 1e-12)).sum(1) / math.log(K)
                terms.append(torch.clamp(conc_gamma * s - H, min=0.0) ** 2)
            conc_loss = torch.cat(terms).mean() if terms else None

        recon_loss = 0.0
        sparsity_loss = 0.0
        z_offdiag_loss = 0.0
""")

sub("""                      + (lambda_domain_balance * domain_balance_loss))
        total_loss.backward()""", """                      + (lambda_domain_balance * domain_balance_loss))
        if conc_loss is not None:
            total_loss = total_loss + lambda_conc * conc_loss
        total_loss.backward()""")

sub("""        loss_value = total_loss.item()
        loss_history.append(loss_value)
""", """        loss_value = total_loss.item()
        loss_history.append(loss_value)
        recon_history.append(pure_recon_loss_val)
""")
sub("""                if rel_change < 1e-4:
                    converged = True
                    break""", """                if conc_dual_stop and lambda_conc > 0 and epoch >= conc_warmup:
                    r_prev = recon_history[-21]
                    rel_change = max(rel_change, abs(recon_history[-1] - r_prev) / r_prev if r_prev > 0 else 0.0)
                if rel_change < 1e-4:
                    converged = True
                    break""")
sub("""        if epoch >= 20:
            prev_loss = loss_history[-21]""", """        if epoch >= 20 and not (lambda_conc > 0 and epoch < conc_warmup + 21):
            prev_loss = loss_history[-21]""")
sub("""    loss_history = []
""", """    loss_history = []
    recon_history = []
""")

sub("""        "loss_history": loss_history,
        "U_scales": U_scales_out,""", """        "loss_history": loss_history,
        "conc_loss": float(conc_loss.item()) if conc_loss is not None else None,
        "conc_info": conc_info,
        "U_scales": U_scales_out,""")

if "\nimport math" not in src:
    src = src.replace("import torch\n", "import torch\nimport math\n", 1)
open(DST, "w").write(src)
print("wrote", DST)
