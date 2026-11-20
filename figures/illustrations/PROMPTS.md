# Concept illustrations (Codex CLI 0.169.0, `codex exec`, built-in image generation)

Concept art only: no numbers, no data. Every quantitative figure is generated from the CSVs by
`analysis/plots.py`. Each image was checked by hand for accuracy before use; captions in the post
say "illustration, not to scale".

Shared style suffix for every prompt:
> Style: clean flat vector illustration, isometric or 2.5D, soft shadows, generous whitespace, off-white
> background (#fcfcfb). Palette: blue #2a78d6 for weights, orange #eb6834 for the KV cache, aqua #1baf7a
> accents, greys for structure. NO text, NO letters, NO numbers, NO logos anywhere in the image. Modern,
> elegant, editorial look (like a Stripe or Distill article illustration).

## decode_step_concept.png: one token streams the whole model
Left: GPU memory (HBM) as a big stack full of blue weight tiles plus a smaller group of orange KV-cache
tiles. Middle: a visibly narrow pipe carrying the tiles (tiles queue at its mouth): the bottleneck.
Right: a chip die with ~100+ small compute cores, most dimmed/idle, a few lit. Far right: one small
glowing token pops out.

## launch_overhead_concept.png: eager launches vs one CUDA graph
Two horizontal lanes stacked vertically. Top lane: a small CPU figure/dispatcher hands kernels to the GPU
one at a time: many small blocks on the GPU timeline separated by visible empty gaps (idle). Bottom
lane: the CPU hands over a single bundled package once; the GPU timeline shows the same blocks packed
tightly with no gaps, finishing much earlier.

## batching_concept.png: one weight read shared by many requests
One wide stream of blue weight tiles flows from memory into the compute chip, and fans out to serve
many parallel request lanes at once (8 lanes, each a different soft colour token). Each lane also has
its own small orange KV-cache stack feeding in separately. The message: the blue stream is shared, the
orange stacks are not.

## hopper_paths_concept.png: two ways to feed the tensor cores
Split image, two panels side by side. Left (newer path): a single large block-transfer engine moves
whole tiles of data from memory into on-chip shared memory in big chunks, feeding one large group of
tensor cores working together as a team (4 warps cooperating on one big matrix tile). Right (older
path): many small individual threads each carry small pieces of data, feeding several smaller
independent tensor-core units. Both panels draw from the same memory at the bottom.

### Revision log
- hopper_paths_concept.png v2: v1 showed both weights and KV cache on both paths; regenerated (v1 attached as reference via `codex exec -i v1.png - < prompt`) so the left/new path carries only weights (cuBLAS nvjet: HGMMA + UTMALDG) and the right/old path only the KV cache (cuDNN sdpa_sm80: HMMA + LDGSTS), matching the SASS counts in notes/REPORT.md.
- Note: `codex exec -i FILE "prompt"` fails ("No prompt provided"); `-i` takes several files and swallows the prompt. Pass the prompt on stdin with `-`.

## fixed_cost_concept.png (post 2): faster memory halves the bytes, not the fixed cost
Two horizontal lanes (older GPU on top, twice the memory bandwidth below), each a memory stack and pipe (the bottom
pipe twice as wide) feeding the same row of ~12 kernel blocks. Each block: a short grey hatched segment (fixed setup)
then a solid blue segment (moving data); a few big kernels, many small ones. Bottom lane: every blue segment half as
long, every grey segment the same, so it finishes earlier but not in half the time. (Style suffix as above, with blue
for data being moved.)
- Checked by hand: same 12 kernels in both lanes, grey widths equal (~50 px), blue halved, bottom lane ~68% of the top.
- Post-processed: the memory-stack bands came out orange (top) and green (bottom); recoloured to the post's chart
  colours (H100 blue #2a78d6, B200 orange #eb6834), and 230 px of empty margin cropped from the top, 90 from the bottom.

## tensor_memory_concept.png (post 2): where tensor-core results go, Hopper vs Blackwell
Two panels, a cutaway of one SM each, identical except one thing. Both: GPU memory at the bottom, a block-transfer
engine (TMA) lifting tiles into an on-chip shared-memory tray that feeds a cluster of tensor cores. Left (Hopper): the
tensor cores write results straight into a grid of registers. Right (Blackwell): they write into a separate aqua
slab (tensor memory), and a small arrow carries results from it into the registers.
- Checked by hand against the SASS in notes/REPORT_B200.md (HGMMA results in registers; UTCHMMA results in tensor
  memory, read out with LDTM; UTMALDG on both).

## Labels
The images above contain no text by design. `analysis/label_illustrations.py` adds the labels (what each
shape is, what to notice) and writes the versions the report and post use to `figures/illustrations/labeled/`.
