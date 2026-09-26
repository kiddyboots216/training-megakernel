# DCLM pretraining

This example prepares DCLM token shards and streams them through the resident
Qwen3-8B-width megakernel on eight GPUs. The host advances the two token
slots, records telemetry, and writes restartable checkpoints while each rank
remains inside one cooperative CUfunction call.

Everything shape-dependent comes from the bundle: its `launch_abi.json`
records the decoder depth and the tokens per GPU it was built for. Shards,
model snapshots and checkpoints must match those.

Install the repository with the `dclm` extra, and `logging` for W&B (see the
top-level README). The scripts read only local files and download nothing;
supply every path explicitly. Commands run from the repository root.

## Inputs

- DCLM Parquet files in one base directory, `DATA_ROOT`. Each file needs an
  `id` and a `text` column.
- An ordered file list, a JSON file naming the Parquet files by path relative
  to `DATA_ROOT`, in the order training reads them:

  ```json
  {"schema":"training_megakernel_dclm_file_list_v1","files":["shard-00000.parquet","shard-00001.parquet"]}
  ```

- A local Hugging Face snapshot of Qwen3-8B, `MODEL_SNAPSHOT`, for 36-layer
  bundles. The loader checks its `config.json` against the bundle's shape and
  requires every weight in BF16 safetensors. Nothing checks the snapshot's
  revision; note the commit you downloaded.
- The tokenizer, normally `tokenizer.json` from the same snapshot, and its end
  of document token ID (151643, `<|endoftext|>`, for Qwen3).

## Catalog the Parquet files

```bash
python examples/dclm/prepare.py catalog "$ORDERED_LIST" "$DATA_ROOT/catalog.json" \
  --base-directory "$DATA_ROOT" \
  --dataset-repo "$DATASET_REPO" \
  --dataset-revision "$DATASET_REVISION"
```

The catalog records each file's size and row-group row counts in the list's
order. It must be written directly in `DATA_ROOT`, since shard preparation
reads the Parquet files relative to the catalog's directory, and it must not
exist yet. The dataset repository and revision are labels recorded with the
data, not downloaded; shard sets and checkpoints carry them so that a
continuation reads the same data.

## Prepare shards

```bash
python examples/dclm/prepare.py shards "$SHARDS" \
  --bundle "$BUNDLE" \
  --catalog "$DATA_ROOT/catalog.json" \
  --tokenizer-json "$MODEL_SNAPSHOT/tokenizer.json" \
  --tokenizer-repo "$TOKENIZER_REPO" --tokenizer-revision "$TOKENIZER_REVISION" \
  --eod-token-id 151643 \
  --steps "$STEPS"
```

This tokenizes the documents in catalog order, appends the end of document
token to each, and writes the token windows for `--steps` steps. `--bundle`
sets the tokens per GPU and the embedding-route capacity from the bundle's
`launch_abi.json`; every window must fit that capacity, and training accepts
shards prepared for that capacity or less. Every GPU's window holds three
documents of (S - 64) / 3 tokens and one 64-row padding segment, so at 32,768
tokens a step trains on 261,632 real tokens.

`SHARDS` must not exist yet and is immutable once written. It holds
`manifest.json`, one `rank-NNNNN.index.json` and one `rank-NNNNN.tokens.i32`
per GPU, `stream-cursors.jsonl` with the DCLM position at every step
boundary, and `next-cursor.json`. To prepare the next shard set of a longer
run, pass `--cursor "$SHARDS/next-cursor.json"` and set `--stream-start` to
the previous set's `--stream-start` plus its `--steps`.

## Models for other depths

Pretrained Qwen3-8B weights exist only at 36 layers. For a bundle of any other
depth, write a randomly initialized snapshot and train from it:

```bash
python -m pip install transformers
python examples/dclm/random_init.py "$RANDOM_SNAPSHOT" \
  --config "$MODEL_SNAPSHOT" --bundle "$BUNDLE"
```

This takes Qwen3-8B's `config.json` (`--config` accepts the snapshot directory
or the file) with `num_hidden_layers` set to the bundle's depth (`--depth D`
instead of `--bundle` also works), initializes the model with transformers
from a fixed seed (`--seed`, default 0), and saves it as BF16 safetensors.
The run loads it like any other snapshot.

## Train

```bash
export TMK_STEPS="$STEPS"
export TMK_HF_MODEL_SNAPSHOT="$MODEL_SNAPSHOT"
export TMK_CHECKPOINT_STEPS=500,1000

bash examples/dclm/train.sh "$BUNDLE" "$RUN_DIR" "$CHECKPOINT_ROOT" "$SHARDS"
```

`train.sh` starts eight ranks with `torchrun` on the local node. It reads
these variables:

| Variable | Meaning |
|---|---|
| `TMK_STEPS` | Steps to run; required. The shard set must hold that many steps from where the run starts. |
| `TMK_HF_MODEL_SNAPSHOT` | The snapshot to start from, with the bundle's depth; required by `train.sh`. |
| `TMK_CHECKPOINT_STEPS` | Comma-separated, increasing global steps to checkpoint at; by default only the run's last step. |
| `TMK_LEARNING_RATE`, `TMK_WEIGHT_DECAY` | Constant AdamW learning rate and weight decay, set together; by default 3e-4 and 0.1, for training from random initialization. For the pretrained Qwen3-8B use about 1e-5 and 0.01. |
| `TMK_RUN_ID` | The run's name, by default the name of `RUN_DIR`. |
| `TMK_WANDB_MODE` | `disabled` (default), `online` or `offline`. |
| `TMK_WANDB_PROJECT` | The W&B project. |
| `TMK_PYTHON` | The Python interpreter, by default `python3`. |

AdamW otherwise uses betas of 0.9 and 0.95, epsilon 1e-8 and global-norm
clipping at 1.0.

`RUN_DIR` and `CHECKPOINT_ROOT` must not exist yet. Before allocating, every
rank compares the training state's estimated memory with its free device
memory; rank 0 prints the estimate, and the run stops if the state does not
fit.

At each step in `TMK_CHECKPOINT_STEPS` the resident function waits while the
host writes the checkpoint to `CHECKPOINT_ROOT/step-NNNNNNNN`; a `COMPLETE`
file, written last, marks it usable.

Rank 0 prints each step's loss and gradient norm as the device finishes it and
writes the rows to `RUN_DIR/telemetry/telemetry.jsonl`. With `TMK_WANDB_MODE`
set to `online` or `offline`, it also logs them to W&B under `TMK_WANDB_PROJECT`,
as the run named and identified by `TMK_RUN_ID`. At the end,
`RUN_DIR/result.json` records whether every rank's device status was clean,
each step's loss and gradient norm, the checkpoints written and the elapsed
time. `RUN_DIR/run.log` holds the console output and `RUN_DIR/torchrun` each
rank's logs. A run that has not finished after 24 hours is stopped.

## Resume

```bash
unset TMK_HF_MODEL_SNAPSHOT
export TMK_STEPS="$ADDITIONAL_STEPS"
export TMK_CHECKPOINT_STEPS="$NEXT_CHECKPOINT_STEPS"

bash examples/dclm/resume.sh "$BUNDLE" "$NEW_RUN_DIR" "$CHECKPOINT_ROOT" "$SHARDS"
```

`resume.sh` restores the newest complete checkpoint under `CHECKPOINT_ROOT` and
runs `TMK_STEPS` more steps, writing its checkpoints into the same root.
`TMK_CHECKPOINT_STEPS` are global steps after the restored one. It takes the
same variables as `train.sh` except `TMK_HF_MODEL_SNAPSHOT`, which must be
unset: the model comes from the checkpoint. `NEW_RUN_DIR` must not exist yet;
set `TMK_RUN_ID` to the earlier run's ID to continue the same W&B run.

The checkpoint must have been written at the bundle's depth, and the shard set
must continue from the DCLM position the checkpoint stopped at: either the
same shard set, if steps remain in it, or the next one prepared with
`--cursor`. The learning rate and weight decay may differ from the
checkpoint's; if they are not set, the checkpoint's values are kept.
