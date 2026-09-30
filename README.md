# Training Megakernel

Train Qwen3-8B, or a model of Qwen3-8B's width with 2 to 82 layers, on eight
H100s with one cooperative application CUfunction launch per GPU for an entire
resident run.

How it works, and how it was built, is written up in [Writing Training
Megakernels without Writing Training
Megakernels](https://kiddyboots216.github.io/megakernel/).

Trained from the same random initialization on the same 1,000 steps of DCLM
data, with the same AdamW settings, on the same node, the training megakernel
finished in 68.8 minutes and Megatron-LM in 83.3: **1.21x the throughput,
17% less wall time**. After step 100 the two loss curves differ by 0.003 on
average (Pearson correlation 0.9993), and their last 50 steps average within
0.1% of each other. Megatron-LM ran eight-way data parallelism with its
distributed optimizer, full activation recompute and Transformer Engine
attention. Wall time runs from the first logged step to the last; it is one
run per implementation.

Per step, Megatron-LM's median is 4.98 s and the megakernel's 4.08 s, a ratio
of 1.22. A factor of 1.13 of that is work Megatron-LM does and the megakernel
does not (19.0 against 16.8 PFLOP per step, from activation recompute); the
remaining 1.08 is how fast each program executes the work it has. These are
intervals between consecutive log records, not device timings.

A stronger baseline is a staged CUDA graph built from the same FlashAttention 4
and Quack tile code: one captured graph per GPU replaying 3,837 kernel launches
per step. Against it the megakernel is about 1% faster. One Qwen3-8B optimizer
update on eight H100s, median of 50 iterations in both run orders:

| Program | Step latency | Positions per second |
|---|---|---|
| Staged CUDA graph | 4,141 ms | 63,300 |
| Megakernel | 4,092 ms | 64,100 |

Positions count all 32,768 per GPU, summed over eight GPUs.

Each rank enters one cooperative CUfunction that remains resident across
optimizer steps. Inside that function, the GPU executes the complete forward
pass, backward pass, distributed gradient work, global-norm clipping, and
AdamW update. Host services refill token slots, persist checkpoints, and emit
telemetry while the function remains live.

## Supported configuration

- Qwen3-8B's width: hidden size 4,096, 32 query and 8 key-value heads of 128,
  MLP size 12,288 and a 151,936-token vocabulary
- 2 to 82 decoder layers and 1,024 to 32,768 tokens per GPU per step, in
  multiples of 1,024, chosen when the kernel is built; the compiled release is
  Qwen3-8B itself, 36 layers at 32,768 tokens
- one node with eight fully NVLink-connected H100 80 GB GPUs
- 132 cooperative CTAs by 384 threads on every rank
- runtime-selected step count, advancing token slots, and restartable
  checkpoints

Memory bounds the shape. The build estimates the per-GPU training state and
stops if it cannot fit an 80 GB GPU with 3.5 GiB left for the CUDA context and
communication. By that estimate every depth up to 82 fits at 16,384 tokens or
fewer, and at most 41 layers fit at 32,768 tokens. 41 layers at 32,768 tokens
runs with about 3 GiB left per GPU:

| Layers | Tokens per GPU | Estimated state | Measured in use | Step time |
|---|---|---|---|---|
| 41 | 32,768 | 75.1 GiB | 76.5 GiB | 4.73 s |
| 40 | 32,768 | 74.5 GiB | 75.8 GiB | 4.61 s |
| 36 | 32,768 | 71.8 GiB | 73.2 GiB | 4.16 s |
| 40 | 16,384 | 50.0 GiB | 51.4 GiB | 2.13 s |
| 12 | 3,072 | 17.2 GiB | 18.6 GiB | 0.19 s |
| 2 | 1,024 | 10.5 GiB | 11.9 GiB | 0.05 s |

"Measured in use" is what `nvidia-smi` reports per GPU during training. Step
times are means over 100 steps (36 layers), 20 steps (40 and 41 layers at 32,768
tokens) or 200 steps of the DCLM example, leaving out the one step that waited
while a checkpoint was written. A checkpoint stores 12 bytes per optimizer
element, 12.3 GB per GPU at 36 layers. The grid streams it to the host
through two 256 MiB pinned slots and holds the step until every rank's copy is
on disk: 12 to 21 seconds per 36-layer checkpoint in our runs, the time of
three to five steps. The device-side copy takes about a third of a second; the
rest is host-side writing and agreement between the ranks. Pretrained weights exist only for 36 layers; the example can train
other depths from random initialization.

On the same batch, the step-1 gradient norm is within 0.14% of a Hugging Face
Transformers reference at every shape tested: 36 layers at 32,768 tokens, 40 at
16,384, 12 at 3,072 and 2 at 1,024.

## Requirements

The supported environment is based on `nvcr.io/nvidia/pytorch:26.02-py3` with:

- Python 3.12
- CUDA 13.0 runtime, CUDA 13.1 binary tools, and driver 590.48.01 or compatible
- binutils 2.42; source builds also need `cuobjdump` and `nvdisasm`

`pyproject.toml` pins exact versions only of the packages the kernel code is
written against (`nvidia-cutlass-dsl`, and `quack-kernels` for source builds)
and sets minimum versions for everything else. Install PyTorch 2.14 or later
with CUDA 13 wheels, then this package with the `dclm` extra for the example
and `logging` for W&B:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch>=2.14.0'
python -m pip install -e '.[dclm,logging]'
training-megakernel-check --world8
```

`training-megakernel-check` prints a JSON report and exits nonzero if a check
fails. It checks the Python version, the pinned package versions and that
PyTorch was built for CUDA 13; `--world8` adds the GPU count, SM count,
compute capability and NVLink topology, and `--build` the source-build
packages. It does not inspect the driver or binary-tool versions; compare
those directly with the requirements above.

## Run the compiled release

Compiled CUDA code is attached to the GitHub Release rather than stored in
Git. The v0.1.0 release carries one bundle, Qwen3-8B at 36 layers and 32,768
tokens per GPU. Download the bundle from the release, extract it, and check
that it loads:

```bash
training-megakernel-smoke /absolute/path/to/bundle
```

A bundle holds three files: `image.o`, `launch_abi.json` and `manifest.json`.
`launch_abi.json` records the depth and tokens per GPU the bundle was built
for, and the runtime sizes everything from those. The manifest records each
file's size and SHA-256; the loader checks them, and the launch ABI's
checkpoint layout, before it loads CUDA code. The smoke test loads and
resolves the CUfunction without launching it.

## Train on DCLM

The [DCLM example](examples/dclm/) prepares token shards, runs the resident
function on eight GPUs, writes periodic checkpoints, and resumes from the
latest complete one. `prepare.py catalog` indexes local DCLM Parquet files,
`prepare.py shards --bundle BUNDLE` tokenizes them into a shard set for the
bundle's tokens per GPU, and then:

```bash
export TMK_STEPS=1000 TMK_HF_MODEL_SNAPSHOT=/absolute/path/to/Qwen3-8B
export TMK_LEARNING_RATE=1e-5 TMK_WEIGHT_DECAY=0.01
bash examples/dclm/train.sh "$BUNDLE" "$RUN_DIR" "$CHECKPOINT_ROOT" "$SHARDS"
```

The learning rate is constant, without warmup. The defaults, 3e-4 and weight
decay 0.1, are for training from random initialization; continuing the
pretrained Qwen3-8B at 3e-4 makes the loss jump within a few steps, so the
example above uses 1e-5 and 0.01.

The scripts take explicit dataset, tokenizer, model, bundle and output paths
and do not download anything. `random_init.py` writes a randomly initialized
snapshot for bundles of other depths, and `resume.sh` continues from the
newest complete checkpoint. The [example's README](examples/dclm/README.md)
walks through every step.

Avoid starting containers on a node while a run is live: in testing, doing so
stopped the running kernel with an illegal-address error. Querying the GPUs
with `nvidia-smi` did not.

## Build from source

The complete source closure is under `kernel/`. Install the build extra and
FlashAttention 4, then build into a new directory outside the repository:

```bash
python -m pip install -e '.[build]'
python -m pip install --no-deps \
  "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention@890f23878394cbff75a92e415f7b7e99b8fbccba#subdirectory=flash_attn/cute"
python -m pip install einops typing_extensions
training-megakernel-check --world8 --build
CUDA_VISIBLE_DEVICES=0 bash kernel/build.sh /absolute/path/to/new-bundle \
  --depth 36 --sequence 32768
```

`--depth` and `--sequence` choose the shape (the defaults are 36 and 32,768).
A build takes one H100 for 4 to 14 minutes and allocates there the training
state it compiles for; it writes the same three files as the release bundle.
See [kernel/README.md](kernel/README.md).

FlashAttention 4 is pinned to commit `890f238` but is not a declared
dependency: its metadata at that commit requires `nvidia-cutlass-dsl==4.6.0.dev0`,
so letting pip resolve it would downgrade `nvidia-cutlass-dsl` and
`quack-kernels`. Install it with `--no-deps` as above, together with `einops`
and `typing_extensions`; its other requirements come with the build extra. The
preflight's `--build` check confirms version `4.0.0b20.dev8+g890f238`.

## Source checks

```bash
python -m compileall -q src examples kernel
bash -n kernel/build.sh examples/dclm/*.sh
```

## Citation

```bibtex
@misc{panda2026megakernel,
  author       = {Ashwinee Panda},
  title        = {Writing Training Megakernels without Writing Training
                  Megakernels},
  howpublished = {\url{https://kiddyboots216.github.io/megakernel/}},
  year         = {2026},
  month        = {sep},
  note         = {Blog post}
}
```

## License

Project code is licensed under the [MIT License](LICENSE). Build dependencies
remain under their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
