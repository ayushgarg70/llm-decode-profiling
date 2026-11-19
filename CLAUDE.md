# CLAUDE.md

Profiling project: "Where does decode time go?" (Llama-3.1-8B decode on one H100, predicted from first
principles, then measured and profiled). Results live in `notes/REPORT.md` (long form) and
`notes/RESULTS.md` (every number with its run ID, generated).


## Hard rules
- **Never invent numbers.** Every number in the write-up comes from `data/results.csv` / `data/kernels.csv` / `data/ppl.csv` via `notes/RESULTS.md` (generated). If a measurement looks wrong, investigate; don't smooth it over.
- `data/*.csv` are **append-only and written only by scripts**. Never hand-edit. Runs that must not count go in `data/excluded_runs.csv` with a reason.
- Timing runs and profiling runs are **separate**: no nsys/ncu/NVTX during timing.
- **Propose before spending money** (state price × hours, wait for a go). At the end of every GPU session: `scripts/sync_back.sh` → verify files locally → stop/terminate the instance.
- Steady state only: every timed configuration gets a throwaway warm-up process with identical arguments (the first process to compile a shape can run faster; see `notes/LOG.md`).

## Environment
- Local: any machine, no CUDA needed. `uv sync` then `uv run pytest`. Tiny-model CPU paths must keep working so everything can be tested without a GPU.
- GPU: a rented 1×H100 SXM VM reached by SSH. CUDA-only code paths are only exercised there.
- The folder for profiler wrappers is `profiling/`, not `profile/` (stdlib module name).
