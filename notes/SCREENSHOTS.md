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

## Post 2 (B200), branch `blackwell`

Save into `figures/screens/b200/`. **Do not use Speed of Light or any "% of peak" panel from the B200 reports**: ncu locked
clocks to base and its bandwidth percentages on the B200 are far below the timing runs (see `notes/REPORT_B200.md`,
Caveats). These shots are about *which instructions run* and *how the kernel is launched*.

| # | File | Report (Nsight Compute) | Kernel (in the report's kernel list) | View | What must be visible |
|---|---|---|---|---|---|
| 1 | `b200_matmul_sass.png` | `reports/B_layer5_matmuls.ncu-rep` | the first `nvjet_sm100_tst_128x8_64x12_4x1_v_bz_TNT` (MLP gate) | Source page, View: SASS; use the search box | lines with `UTCHMMA`, `UTMALDG` and `LDTM` (the B200 counterpart of post 1's `HGMMA` shot) |
| 2 | `b200_fp4_sass.png` | `reports/B_nvfp4_kernels.ncu-rep` | the first kernel starting `nvjet_sm100_ootst_128x128_256x7_..._Avec16UE4M3` | Source page, SASS | lines with `UTCOMMA` (FP4 tensor-core MMA) and `UTCCP` (block scales copied into tensor memory) |
| 3 | `b200_fp4_launch.png` | same report, same kernel | | Details → Launch Statistics | Grid Size 32, Block Size 256, # SMs 148, Waves Per SM 0.22 |
| 4 | `b200_attention_sass.png` | `reports/B_attn_64k.ncu-rep` | the only kernel (`cudnn_generated_fort_native_sdpa_sm100_flash_fprop...`) | Source page, SASS | `UTCHMMA`, `LDTM`, `STTM` (post 1's H100 attention used `HMMA` + `LDGSTS`) |
| 5 (optional) | `b200_overlap_timeline.png` | Nsight Systems, `traces/B1_compiled_graph.nsys-rep` | | Timeline, CUDA HW row, zoomed to ~30 µs inside one graph replay | consecutive kernels drawn overlapping (each starts ~0.65 µs before the previous ends) |

Taken 2026-11-18 (desktop session). Shots 1 and 4 are stacked crops of the same kernel (instructions ~350 / ~800 lines apart), separated by dashed lines. In shot 5's window the overlaps are 0.3–1.2 µs (the 0.65 µs in the report is the average over all replays). Reports verified unchanged (shasum).
