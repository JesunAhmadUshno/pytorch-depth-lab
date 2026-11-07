# PREREG: PyTorch Depth Lab

Frozen before any training run. This file is committed before any
results exist; the commit hash is recorded in README.md once it
exists. Any change to this file after the first results run is an
amendment: it is appended under Amendments with a date and a reason,
never edited silently.

Study: Batch 4, Track B4, Gap-Closure Programme. Author of record:
Jesun Ahmad Ushno. Research base: Claude verdict pack 10 (CONDITIONAL
GO) with the coordinator verification note of 2026-11-06.

## Dataset (frozen)

- EuroSAT RGB, Version v2, Zenodo record 7711810, file
  EuroSAT_RGB.zip (94.7 MB; record md5 f46e308c4d50d4bf32fedad2d3b).
  The SHA-256 of the downloaded zip is recorded in
  results/dataset.json at fetch time and must match on both
  machines.
- 27,000 images, 64x64, 10 classes, folder-per-class layout.
- Split: stratified by class, seed 20261107, train 21,600 /
  validation 2,700 / test 2,700. The split index lists are saved
  to results/split.json (a derived statistic; no dataset file is
  committed to this repo).
- Licence: MIT, confirmed from the dataset's official repository
  (phelber/EuroSAT, License section) on 2026-11-07 before download.
  See LICENSES.md. The verdict pack's CC BY-NC-SA 4.0 row (E19) was
  not confirmed by the official source and is not used.

## Frozen recipe (every full GPU arm unless an arm varies it)

- Model: torchvision ResNet-18, final layer replaced by
  Linear(512, 10). Parameter count 11,689,512.
- Initialisation: scratch arms use the default initialisation under
  the run seed; the fine-tune arm loads ResNet18 IMAGENET1K_V1
  weights and re-initialises only the final layer under the run
  seed.
- Optimizer: AdamW, lr 1e-3, betas (0.9, 0.999), weight_decay 1e-4.
- Schedule: CosineAnnealingLR, T_max equal to the arm's epoch count.
- Loss: CrossEntropyLoss, no label smoothing.
- Batch size 128. Epochs 15 for full GPU arms.
- Augmentation (train only): RandomHorizontalFlip(0.5),
  RandomVerticalFlip(0.5). No augmentation at evaluation.
- Normalisation (all arms, all splits): ImageNet mean
  (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).
- Evaluation: validation accuracy and NLL at each epoch end; test
  accuracy and NLL once, after the final epoch.
- Seeds: every full GPU arm runs seeds 42 and 43 (n = 2 per arm).
- cudnn.benchmark: False in every arm (deterministic data order is
  worth more here than the autotuner; the Lightning config-diff
  audit must show this flag equal in both arms).

## Measurement definitions (frozen)

- Throughput of a run: total training images divided by total
  training step time, excluding epoch 1 (warmup: allocator,
  autotuner, lazy module load) and excluding evaluation time.
  Per-epoch throughput is also logged.
- Peak memory of a GPU run: torch.cuda.max_memory_allocated over
  the whole run, in MiB.
- Time-to-accuracy (TTA): wall-clock seconds from run start until
  the first epoch end at which validation accuracy is at least
  0.96; "not reached" if no epoch qualifies.
- Accuracy difference: paired bootstrap over the 2,700 test
  examples, 10,000 resamples, bootstrap seed 7, percentile 95%
  interval. Both models score the same test examples, so the
  pairing is exact.
- Throughput ratio interval: bootstrap over per-epoch throughputs
  (epochs 2 to 15, both seeds pooled), 10,000 resamples, bootstrap
  seed 7, percentile 95% interval.
- GPU state per run: nvidia-smi snapshot (SM clock, memory clock,
  temperature, power draw, driver) logged at run start and end.
- Arm order: randomised across the suite with seed 20261107; the
  realised order is written to results/run_order.json by the
  runner before the first arm starts.

## Arms (frozen list)

GPU arms (BlackBox, RTX 5050 Laptop GPU, 8 GB), 15 epochs, seeds
42 and 43 unless stated:

- A1 hand FP32, from scratch. The reference arm.
- A2 hand FP32, fine-tuned from IMAGENET1K_V1. Descriptive arm
  (no hypothesis); reported against A1 on accuracy, NLL, TTA.
- A3 Lightning FP32, from scratch. Baseline for H1. Must use the
  frozen recipe exactly; every differing setting found by the
  config-diff audit (src/config_diff.py output, saved to
  results/config_diff.json) is published, and any difference that
  cannot be closed is named in RESULTS.md next to the H1 verdict.
- A4 hand AMP: torch.autocast FP16 with GradScaler, from scratch.
- A5 hand FP32 with torch.compile (default mode), from scratch.
  Compile wall time is measured on the first training step and
  reported separately; epoch 1 is excluded from throughput as for
  every arm. A separate probe compiles the training step with
  fullgraph=True and records success or the exact failure.
- A6 hand FP32 with gradient checkpointing on the ResNet layer
  blocks, from scratch, batch 128. Plus a memory probe (below).
- A7 data-loading benchmark (no full training): loader-only
  throughput and training step-time decomposition for loader
  configurations L0 (workers 0, pin off), L1 (workers 0, pin on),
  L2 (workers 2, pin on), L3 (workers 4, pin on, persistent
  workers). Loader-only: iterate 3 full epochs of the train
  loader, images per second. Decomposition: 100 training steps at
  batch 128, data-wait time vs compute time per step.
- A8 profiling study: torch.profiler trace of the A1 configuration
  for 30 steps, and of the optimized configuration for 30 steps.
  The optimization is chosen by this frozen rule: if data wait is
  the largest share of step time in the baseline decomposition
  (A7, L0), the optimization is loader configuration L3; if
  compute is the largest share, the optimization is
  channels_last memory format. Top operator table by self CUDA
  time is saved for both traces.
- A9 FSDP2 check (not training): fully_shard the ResNet-18 blocks
  on one GPU, record the sharded parameter shapes, run one
  forward and backward step at batch 32, record success or the
  exact failure. Shape and initialisation claim only.
- A10 memory probe for H6: short runs (60 steps, batch in
  {32, 64, 128, 256}) with and without gradient checkpointing,
  peak allocated memory per cell; plus a max-feasible-batch probe
  at batch 512 with and without checkpointing (success = 20 steps
  without OOM).

CPU arms (Muse VM, 4 cores), reduced sub-protocol (frozen):

- Epochs 3, batch size 128 effective, seed 42, optimizer and
  schedule as the frozen recipe with T_max = 3. Data loader
  workers 0 (RAM limit of the shared VM; recorded here, not
  discovered later). OMP_NUM_THREADS = 2 per process.
- C1 single process.
- C2 DDP with the gloo backend, world size 2, per-rank batch 64
  (effective batch stays 128), DistributedSampler with the run
  seed. World size 2, not the pack's illustrative 4: the VM has
  4 cores shared with other work, and 2 ranks at 2 threads each
  use the machine without oversubscribing it. This is a
  correctness arm; no speed claim is made from it.
- Logged per epoch: mean train loss, validation accuracy, epoch
  wall time. Communication overhead is reported as the C2 to C1
  epoch-time ratio, descriptively.

Custom autograd (unit deliverable, not an arm): a fused
log-softmax plus NLL function with an analytical backward pass in
src/custom_autograd.py, verified by torch.autograd.gradcheck in
double precision in the test suite.

## Hypotheses and frozen verdict rules

- H1 (hand loop vs Lightning): SUPPORTED if the throughput ratio
  (A1 / A3, pooled seeds) is at least 0.90 AND the paired test
  accuracy difference (A1 minus A3) has its 95% bootstrap interval
  inside plus or minus 1.0 percentage point. The band is reported:
  ratio in [0.90, 1.10] is a tie, above 1.10 the hand loop is
  faster, below 0.90 it is slower.
- H2 (AMP): SUPPORTED if the throughput ratio (A4 / A1) is at
  least 1.30 AND the paired accuracy difference (A4 minus A1)
  interval lies inside plus or minus 1.0 pp. Peak memory saving
  is reported either way. Prior note: 1.30x is the pack's prior,
  not a value derived from data.
- H3 (torch.compile): SUPPORTED if the throughput ratio (A5 / A1)
  is at least 1.10 AND the paired accuracy difference interval
  lies inside plus or minus 1.0 pp AND compile time and the
  fullgraph probe outcome are reported. Prior note: 1.10x is the
  pack's prior.
- H4 (profiler): SUPPORTED if the baseline top bottleneck is
  named from the A7 decomposition and the A8 profiler table, one
  optimization is applied under the frozen A8 rule, and that
  bottleneck's share of step time is lower after the optimization.
- H5 (data loading): SUPPORTED if the best loader configuration's
  loader-only throughput exceeds L0's by a ratio above 1.00 (the
  amount is stated in RESULTS.md) AND its data-wait share in the
  decomposition is lower than L0's. If L0's data-wait share is
  below 2% of step time, the verdict is NOT A BOTTLENECK on this
  machine and H5 is reported as such (pack stop rule S7), not as
  supported.
- H6 (gradient checkpointing): SUPPORTED if, at batch 128, peak
  memory falls by at least 30% (A10 probe, checkpointing vs not)
  AND the full-run paired accuracy difference (A6 minus A1)
  interval lies inside plus or minus 1.0 pp. The full-run
  throughput ratio (A6 / A1) is reported against the pack's prior
  band [0.65, 0.90]: inside, above, or below the band is stated
  plainly. Max feasible batch from A10 is reported.
- H7 (CPU DDP): SUPPORTED if the mean absolute difference between
  C1 and C2 per-epoch mean train losses is at most 0.05 nats AND
  the final validation accuracy difference is inside plus or
  minus 1.0 pp. The epoch-time ratio is reported as the
  communication overhead, with no speed claim.

## Stop rules (from the pack, binding)

- S1: if AMP is slower than FP32, report it as a failure; no
  cherry-picking.
- S2: if torch.compile fails or is slower, report the failure
  mode; do not hide it.
- S3: if checkpointing saves less than 30% memory, report it.
- S4: if CPU DDP does not converge to the recipe, report it.
- S5: if an accuracy difference falls outside plus or minus
  1.0 pp, report it next to the verdict it affects.
- S6: if the profiler shows no bottleneck, report it.
- S7: if the data loader is not a bottleneck, report it.

## Claim boundary (frozen)

Results hold for the named machines and pinned versions only.
No claim about multi-GPU scaling, NCCL, CUDA kernels, compilers,
or MLIR. CPU DDP is correctness evidence, not scaling evidence.
FSDP2 is a shape and initialisation check. No number is reported
without its baseline row, n, seeds, interval, dataset version,
and producing command.

## Amendments

- A1 (2026-11-07, before any results): the parameter count in
  the Frozen recipe section reads 11,689,512. That is ResNet-18
  with the 1000-class ImageNet head, the pack's figure. This
  study's model carries the 10-class EuroSAT head and has
  11,181,642 parameters, verified against torchvision at build
  time (both counts reproduced in the build log). The recipe is
  otherwise unchanged; the correction is recorded here rather
  than edited silently, per this file's own rule.
