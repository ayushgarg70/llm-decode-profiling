# Nsight GUI shot list (for a Claude desktop session with computer use, or by hand)

Timelines are NOT needed from the GUI: `figures/timeline.png` is drawn from the nsys traces by
`analysis/plots.py`. These are the views only the GUIs show well. Save as PNG in `figures/screens/`
with the given names; crop to the panel; light theme; no personal info (paths with usernames) in view.

| # | File | App / report | View | What must be visible |
|---|---|---|---|---|
| 1 | `ncu_lm_head_sol.png` | Nsight Compute, `reports/E2_lm_head.ncu-rep` | Details → GPU Speed Of Light Throughput | DRAM throughput bar near the top (~87%), Compute (SM) bar low |
| 2 | `ncu_lm_head_memory.png` | same | Details → Memory Workload Analysis → Memory Chart | DRAM → L2 → L1/TMA flow with byte counts |
| 3 | `ncu_lm_head_sass.png` | same | Source page, View: SASS, sort by Instructions Executed | `HGMMA` and `UTMALDG` lines highlighted |
| 4 | `ncu_attention_sass.png` | `reports/E4_attn_64k.ncu-rep` | Source page, SASS | `HMMA` and `LDGSTS` lines (the older path), next to #3 |
| 5 | `ncu_B1_vs_B64.png` | `reports/E5_layer5_matmuls_B1` + `_B64`, add B64 as baseline | Details → Compute Workload Analysis | tensor pipe utilization ~3% vs ~22% side by side |
| 6 | `nsys_eager_vs_graph.png` (optional) | Nsight Systems, `traces/E1_own_eager.nsys-rep` and `traces/E3_compiled_graph.nsys-rep` | Timeline, CUDA HW row + CUDA API row, zoomed to ~1 ms | the CPU launch row driving the GPU row (eager) vs one cudaGraphLaunch (graph) |

The `.ncu-rep` / `.nsys-rep` files are in `reports/` and `traces/` (gitignored, local on the Mac).
