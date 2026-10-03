# CSF3 resources and limits (checked 2026-10-03)

What the University of Manchester CSF3 documentation says about time, CPU and GPU limits, what
can run in parallel, and a rough estimate of what that means for chunk13 (A1) at full scale. Read
from the CSF3 pages listed under Sources on 2026-10-03; the pages change, so re-check before
planning a large run.

## 1. Limits

The documentation does not give separate limits for PGR students. Limits depend on whether the
account is **free at point of use** or belongs to a group that has contributed hardware to the CSF
(a contributor group, which is told its limits when the account is created).

| Resource | Limit | Source |
|---|---|---|
| CPU job wallclock (`multicore`, `multicore_small`, `serial`, `himem`) | 7 days (`-t 7-0`) | Partitions; Time limits |
| Interactive CPU job | 6 hours | Partitions |
| GPU job wallclock (`gpuA`, `gpuL`) | 4 days batch (`-t 4-0`), 1 day interactive | Partitions; GPU jobs |
| `gpuH_short` (H200) | 1 day | Partitions |
| `multicore` node | up to 168 cores, 8 GB RAM per core | Partitions |
| `serial` | 1 core per job, 5-6 GB per core | Partitions |
| `himem` | up to 32 cores, 32 GB per core (64 GB on icelake nodes) | Partitions |
| GPUs for free-at-point-of-use users | up to **4 at once**, A100 80GB (`gpuA`) and/or L40S (`gpuL`), in any mix (since February 2026); "we will NOT accept requests for access to more than 4 GPUs" | GPU jobs |
| GPU access request | **not needed** for A100 80GB and L40S ("you do not need to submit a ticket"); A100 40GB (`gpuA40GB`) and H200 (`gpuH`, `gpuH_short`) are restricted | GPU jobs; Partitions |
| CPU cores per GPU | up to 12 (`gpuA`, `gpuL`), up to 8 (`gpuH`) | Partitions |
| Number of jobs | "submit as many jobs as you wish – the system will run them within the limits" | GPU jobs |
| Job arrays | highest task id 25,000; no per-user array limit documented; `%N` caps how many tasks run at once (`#SBATCH -a 1-100%5`); `--mail-type=ARRAY_TASKS` is rejected for arrays over 20 tasks (since December 2025) | Job arrays |
| Dependencies | `afterok`, `afterany`, `aftercorr` (task i waits for task i of the earlier array) | Job arrays |
| Free-at-point-of-use CPU cores in use at once | **32 — not confirmed.** A web search summary gave this figure, but none of the pages read contains the sentence | — |

There is no default wallclock time: every job must give `-t`. A job still running at its limit gets
SIGTERM, then SIGKILL 30 seconds later.

**To see the real limits of this account**, run on the CSF login node (incline has no Slurm tools):

```
sacctmgr show assoc user=$USER format=account,partition,qos,grptres,maxtres,maxjobs,maxsubmit
```

## 2. What can run in parallel (chunk13)

`submit_chunk13v9.sh` runs one array task per config (6 tasks, 2 cores, 24 h); each task runs the
whole pipeline for its config in sequence (scout, K filter, deep dive, archive, stability). Almost
every fit inside it is independent of the others:

- **Scout.** With one tuned parameter, NSGA-II is random search (CLAUDE.md §4.12), so the λ values
  can be drawn in advance from a fixed seed and split across array tasks (config × K × block of λ
  values). Each task writes its own result file and the files are merged afterwards: many tasks
  writing to one SQLite Optuna study at once is not safe on shared storage.
- **K filter.** Needs every scout of the config; a short job with `--dependency=afterok:<scout array>`.
- **Deep dive.** Another array over λ blocks of the two surviving K.
- **Archive and stability.** Each refit (10 seeds × archived model) is independent: one array task
  per model or per seed. This is the largest stage.
- **Threads inside one fit** shorten one fit but lower throughput: 32 single-threaded fits do more
  work than 8 fits with 4 threads each, and threading risks bit-for-bit reproducibility (ticket 76).

## 3. Rough time at full scale

From the complexity estimate of 2026-10-03 (one epoch is O(K·nnz + K²·N); about 1,100 fits per
config; at full scale roughly 1 CPU-hour per fit, so about 6,600 CPU-hours for 6 configs). These
are order-of-magnitude figures: the M1 matrices are not built, so N, nnz and epochs to convergence
are guesses.

| Layout | Wall time |
|---|---|
| Current: 6 tasks, one per config | about 45 days per task; does not fit within 7 days |
| Arrays over fits, 32 cores at once (if that cap is right) | about 9 days |
| Same, with smaller budgets (50 draws per K in the scout, deep dive to 100, 5 stability seeds) | about 4-5 days |
| 4 GPUs, if a GPU fit is about 10× faster than a CPU fit | about 6 days; 2-3 days with smaller budgets |

Arrays alone save about 5×, about 10× with smaller budgets. The GPU figure needs a timed fit once
M1 exists; GPU results will not match CPU results bit for bit. Whether GPU jobs count towards the
CPU-core cap is not documented.

On the draw budget: on the toy run, 30 random draws give about 95% of the hypervolume of 200 draws
but a different knee model; 50 per K in the scout with the deep dive to 100 kept most of the result
(resampling of the v9.4 studies, 2026-10-03; not yet checked at full scale).

## Sources

- [Slurm partitions](https://ri.itservices.manchester.ac.uk/csf3/batch-slurm/partitions/)
- [Time limits](https://ri.itservices.manchester.ac.uk/csf3/batch/timelimits/)
- [Nvidia GPU jobs (Slurm)](https://ri.itservices.manchester.ac.uk/csf3/batch-slurm/gpu-jobs-slurm/) (page last modified 2026-07-27)
- [Job arrays (Slurm)](https://ri.itservices.manchester.ac.uk/csf3/batch-slurm/job-arrays-slurm/)
- [CSF3 home](https://ri.itservices.manchester.ac.uk/csf3/) ("limited 'free at the point of use' resource ... funded by the University")
