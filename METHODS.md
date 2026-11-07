# METHODS

How the frozen protocol (PREREG.md) is implemented. Anything not
specified here or in PREREG.md is in DECISIONS.md with a reason.

## Training implementations

- Hand loop: src/train.py `run_training`. One function serves all
  hand arms; arms differ only by the factor flags (init,
  precision, compiled, checkpointing, channels_last, loader
  config). Data order comes from a DataLoader whose generator is
  seeded with the run seed.
- Lightning arm: src/lightning_arm.py. A LightningModule with the
  same model build and configure_optimizers returning the frozen
  AdamW and CosineAnnealingLR. The Trainer gets the same
  explicitly built train DataLoader, logger and checkpointing
  disabled, progress bar and model summary off, sanity validation
  steps 0, and no validation loaders: validation is computed by a
  callback calling the shared evaluate() at train-epoch end, so
  both arms are scored by identical code.
- Config-diff audit: src/config_diff.py compares the hand recipe
  declaration with the Lightning arm's config read off its actual
  optimizer objects, per key, and writes results/config_diff.json.
- Evaluation: src/train.py `evaluate` (no_grad, argmax accuracy
  and summed cross-entropy divided by n). Test correctness is
  stored per run as a boolean array aligned to the unshuffled
  test loader order.

## Timing and memory accounting

- Throughput (frozen): training images over training-phase wall
  time (batch fetch waits included, evaluation excluded), epochs
  2 to 15 pooled. The hand loop and the Lightning callback use
  the same definition. Pure step time (excluding fetch waits) is
  also logged per epoch as train_step_time_s.
- Time-to-accuracy: wall seconds from run start to the first
  epoch end whose validation accuracy reaches 0.96.
- Peak memory: torch.cuda.max_memory_allocated over the run.
  Memory probe cells (A10) reset the counter per cell.
- GPU state: nvidia-smi snapshot (SM and memory clocks,
  temperature, power, driver) at run start and end, stored in
  each run JSON. Consumer GPUs throttle; the snapshots are the
  evidence, and arm order is randomised against drift.

## Benchmarks

- Loader throughput (A7): wall time to iterate the configured
  train loader for 3 epochs, no model. Step decomposition: 100
  training steps timing the fetch wait (time blocked on next())
  separately from the compute body; the first fetch is excluded
  (loader spin-up). Configurations L0 to L3 are frozen in
  PREREG.md; the decomposition also covers L0 with channels_last,
  the second candidate optimization of the A8 rule.
- Profiling (A8): torch.profiler, CPU and CUDA activities, 5
  warmup steps outside the trace, 30 profiled steps, top 15
  operators by self CUDA time (self CPU time on CPU devices),
  chrome trace exported under results/profiles/ (not committed).
- Compile probes: the A5 arm's first training step time is the
  compile cost under default mode (torch.compile is lazy); a
  separate probe attempts a fullgraph=True compile of one step
  and records success or the exact exception.
- FSDP2 check: single-rank gloo group, fully_shard on the four
  ResNet layer blocks and the root, DTensor shape recording, one
  forward and backward step at batch 32. Gloo on one rank is a
  shape and initialisation check only, by design.

## CPU DDP

torch.multiprocessing.spawn, gloo backend, world size 2,
DistributedSampler (seed 42), per-rank batch 64 (effective 128),
2 torch threads per process. Per-epoch train loss is all-reduced
to the global mean for the trajectory comparison. Validation runs
on rank 0 over the full validation set.

## Statistics

src/analyze.py, bootstrap seed 7, 10,000 resamples throughout.
Accuracy differences: paired bootstrap over the 2,700 test
examples, per seed pair, averaged across seed pairs, percentile
95% interval. Throughput ratios: bootstrap over pooled per-epoch
throughputs (epochs 2 to 15). Verdicts follow the frozen rules in
PREREG.md exactly; analyze.py prints them and writes
results/summary.json.

## Reproduction

Same command, same machine: `python -m src.data --verify` (data
and split), `python -m src.run_suite --machine gpu` on BlackBox
or `--machine cpu` here, then `python -m src.analyze`. Run
commands are embedded in every result JSON.
