# Qwen3-8B training megakernel

This directory builds the eight-GPU training megakernel for a model of
Qwen3-8B's width: one cooperative CUfunction per GPU that runs forward,
backward, clipped AdamW, token refill and in-launch checkpoints for as many
steps as the host asks, inside one launch. The host fills the next token slot,
persists completed checkpoints and exports telemetry. The decoder depth and
the tokens per GPU are chosen when the kernel is built.

## Layout

- `build.py` compiles the program on one GPU and writes a bundle directory of
  three files. `build.sh` runs it with the build's environment.
- `program/` is the program. `model.py` gives it the build's depth and tokens
  per GPU; every shape-dependent size follows from them through
  `src/training_megakernel/shape.py`, which the host runtime uses too.
  `training_program.py` holds the generator that emits the kernel's source;
  `resident_step.py` supplies the step loop's device code around one training
  step (token-slot refill, per-step records, the in-launch checkpoint, the
  per-step reset and the exit). The other modules are the model, attention,
  projection, normalization, scheduling, optimizer and communication bodies.
- `tools/patch_step_backedge.py` makes the step loop after assembly (see
  below).

## Build

```bash
CUDA_VISIBLE_DEVICES=0 bash kernel/build.sh /absolute/path/to/new-bundle \
  --depth 36 --sequence 32768
```

`build.sh OUTPUT [options]` runs `build.py --out OUTPUT [options]` with
`${TMK_PYTHON:-python3}` on the GPU `CUDA_VISIBLE_DEVICES` selects (GPU 0 by
default). The options are:

- `--depth`: decoder layers, 2 to 82 (default 36)
- `--sequence`: tokens per GPU per step, a multiple of 1,024 from 1,024 to
  32,768 (default 32,768)

`build.py` passes the shape to the program through `TMK_DEPTH` and
`TMK_SEQUENCE`; do not set them yourself.

The build needs one H100 80 GB. The compile allocates the per-GPU training
state of its shape there, because it reads pointer alignment from real device
allocations: about 73 GiB at 36 layers and 32,768 tokens, 50 GiB at 40 layers
and 16,384 tokens, 18 GiB at 12 layers and 3,072 tokens. Before allocating,
the build prints its estimate of the training state and stops if the shape
cannot fit the GPU with 3.5 GiB left for the CUDA context and communication.
It takes about 14 minutes at 36 layers and 32,768 tokens and 4 to 8 minutes at
the smaller shapes.

Attention checkpoints cover the top 36 layers when there are that many, and as
many as fit the optimizer's borrowed gradient buffer otherwise (none at 2
layers); the top 12 layers, or all of them, keep their residual checkpoints.

## The step loop

The program ends each step with a branch that skips the exit sentinel's store
when another step follows, and ptxas assembles that branch as a predicated
`EXIT`. After assembly, `patch_step_backedge.py` rewrites that one instruction
into a branch, with the same predicate, to the kernel's first instruction.
That rewritten instruction is the step loop. It cannot be written in the
source: when the step sits inside a source-level loop, ptxas drops the
register-split (`USETMAXREG`) instructions and the kernel runs far slower.

The patch changes the same 16 bytes in the cubin, in the fatbin that embeds it
and in the object file that embeds the fatbin, and nothing else. If ptxas lays
out the step's tail in a form the patch does not recognize, the build stops
rather than guess. The build also prints a warning if the assembled kernel
contains no `USETMAXREG` instruction, and stops if the kernel's static stack
does not fit the 2,048-byte per-thread CUDA stack the runtime sets.

## Output

The output directory must not exist yet and must be outside the repository.
The build writes it only when it succeeds:

- `image.o`: the exported object holding the patched sm_90a cubin
- `launch_abi.json`: the grid, block, shared memory, stack limit, argument
  layout, checkpoint layout, depth and tokens per GPU the runtime launches with
- `manifest.json`: the function name and each file's size and SHA-256

## Host tools

`nvdisasm` and `cuobjdump` from CUDA 13, and `readelf`, `objcopy` and `nm` from
binutils. The Python environment is described in the top-level README.
