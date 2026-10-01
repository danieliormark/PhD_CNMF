"""Ticket 101 check: one fixed fit (C1, K=3, lambda_z_offdiag=0.5727904470799616,
seed 42, cap 2000). Run on two machines with the same environment variables;
identical output means results are portable across those machines.
Incline references (incline35, Xeon Gold 6326, ATEN_CPU_CAPABILITY=default,
MKL_CBWR=COMPATIBLE): v9.3 0.8447861671447754 / 1452 epochs; v9.4 (concentration
term on, ticket 102) 0.8629798889160156 / 770 epochs.
torch is imported before numpy on purpose: importing numpy first changes the fit
(2026-10-02: 0.8446382880210876 / 1476 instead of the v9.3 value above)."""
import importlib.util, os, torch
spec = importlib.util.spec_from_file_location("c13", "chunk13v9.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
raw = m.load_and_validate_data("/mnt/hum01-home01/p91688di/tensor_data_staging/toy_large/outputs/Star_extended_matrices_t1_v2.pkl")
soc, sem, anc = m.get_active_facets("C1"); m.set_seeds(42)
U, Z, d = m.run_inner_solver(raw_data=raw, soc_keys=soc, sem_keys=sem, anchor_keys=anc, K=3,
                             dimensions=raw["dimensions"], params={"lambda_z_offdiag": 0.5727904470799616},
                             device=m.DEVICE, seed_function=m.set_seeds)
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
print(f"CROSSCHECK host={os.uname().nodename} cpu={cpu} capability={torch.backends.cpu.get_cpu_capability()} "
      f"ATEN={os.environ.get('ATEN_CPU_CAPABILITY')} MKL_CBWR={os.environ.get('MKL_CBWR')} "
      f"math_loss={d['math_loss']!r} epochs={len(d['loss_history'])}")
