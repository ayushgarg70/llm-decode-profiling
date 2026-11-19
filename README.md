# Where does decode time go?

Profiling LLM decode token by token: Llama-3.1-8B (BF16) on one H100 SXM. What one generated token
*should* cost from first principles (bytes ÷ bandwidth), what it actually costs, and where every
millisecond of the gap goes: CPU launch gaps, CUDA graphs and fusion, weight and KV-cache reads,
batching, quantization, and a vLLM reference.

Blog post: [Where does decode time go? (Medium)](https://medium.com/@ayushgarg70/where-does-decode-time-go-profiling-llama-3-1-8b-on-an-h100-token-by-token-c3afbe77a459) · Full write-up: [`notes/REPORT.md`](notes/REPORT.md) · Every number with its run ID: [`notes/RESULTS.md`](notes/RESULTS.md)

![GPU timelines of one decode step, launched four ways](figures/timeline.png)

## Headline results (batch 1, one H100 SXM)

| | ms / token | tokens / s | % of bandwidth floor |
|---|---|---|---|
| Floor: 15.0 GB of weights ÷ 3.15 TB/s (measured) | 4.76 | 210 | 100% |
| Hugging Face `generate()` | 18.47 | 54 | 26% |
| This repo, eager PyTorch | 16.39 | 61 | 29% |
| This repo, torch.compile + CUDA graphs | **6.63** | **151** | **72%** |
| vLLM 0.34 (CUDA graphs), 2k context | 6.37 | 157 | 76% |
| This repo, INT4 weights (2k context; perplexity 6.31 → 7.07) | 3.41 | 294 | 40% of INT4's own floor |

Context costs **41.1 ns per token** (the KV cache streams at 3.19 TB/s); batching reaches **6,530 tokens/s**
at batch 128 (2k context). See the report for the sweeps, the Nsight Compute findings and what didn't work.

## Repository layout

| Path | What |
|---|---|
| `decode/` | The decode loop: a ~300-line Llama in plain PyTorch with a static KV cache (`model.py`, `generate.py`), weight loading, the CLI (`run.py`), quantization (`quant.py`), Hugging Face and vLLM baselines |
| `bench/` | Timing with CUDA events, environment capture, GPU telemetry, E0 ceilings, the results CSV writer |
| `analysis/` | First-principles predictions (`predict.py`), nsys kernel grouping, ncu summaries, SASS instruction mix, perplexity, charts (`plots.py`), `RESULTS.md` generator |
| `profiling/` | nsys and ncu wrappers |
| `runs/` | Run lists per GPU session (`s1.sh` E0–E2, `s2.sh` E3–E5, `s3.sh` E6–E7) and an unattended chain runner |
| `data/` | Script-written results: `results.csv` (timing runs, append-only), `kernels.csv` (nsys breakdowns), `ppl.csv`, `ceilings.json`, `excluded_runs.csv` (with reasons), `ncu/` (per-kernel Nsight Compute summaries), `steps/` (every kernel of one traced decode step, extracted from the nsys traces, which are too large for git) |
| `figures/` | Charts generated from `data/` (`analysis/plots.py`); `illustrations/` holds concept art (no data), with text labels added by `analysis/label_illustrations.py` in `illustrations/labeled/` |
| `notes/` | `REPORT.md` (the long write-up), `LOG.md` (raw lab notebook), `RESULTS.md` (generated) |
| `tests/` | CPU tests: predictions vs hand math, parity with Hugging Face on tiny configs, quantization plumbing |

## Reproduce

Locally (any machine, no GPU): `uv sync && uv run pytest` (CPU only: tiny models, the prediction math, the data files).

On a 1×H100 VM with Nsight Systems/Compute installed and access to `meta-llama/Llama-3.1-8B`:

```bash
scripts/push.sh <ssh-host>                     # from your machine: copy the code over
scripts/bootstrap_gpu.sh env                   # on the VM: Python env with CUDA PyTorch
hf auth login && scripts/bootstrap_gpu.sh model
scripts/bootstrap_gpu.sh vllm                  # only for E7 (separate venv)
scripts/check_gpu.sh                           # go / no-go (incl. ncu counter access)
runs/s1.sh e0 ; runs/s1.sh e1 ; ...            # or: SESSION=s2 runs/chain.sh e3 e4 e5
scripts/sync_back.sh <ssh-host>                # from your machine: results, traces, reports
uv run python -m analysis.extract_steps        # only if you have new nsys traces: refresh data/steps/
uv run python -m analysis.plots                # regenerate every chart
uv run python -m analysis.results_table        # regenerate notes/RESULTS.md
```

One step without anything else: `uv run python -m decode.run --mode compiled_graph --batch 1 --context 2048 --fill synthetic`.

## Credits

gpt-fast (static KV cache + torch.compile recipe), torchao, vLLM, Hugging Face transformers, NVIDIA Nsight
documentation. The code, experiment runs and analysis were done together with Claude Code (Anthropic). Concept
illustrations generated with OpenAI Codex; every chart is generated from the measured data.

## License

MIT (see [`LICENSE`](LICENSE)).
