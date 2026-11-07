# RESULTS: PyTorch Depth Lab

All numbers below are recounted by src/analyze.py from the raw
run artefacts in results/ (per-run JSON, test-correctness arrays,
benchmark JSON). Verdicts follow the frozen rules in PREREG.md
exactly. Six of seven hypotheses are NOT SUPPORTED. They are
reported as measured, under the pack's stop rules, not softened.

## Provenance

- Pre-registration commit: 565dfc0cd8c9bebee94166d53fb9d3756ab62150
  (PREREG.md frozen before any training run; Amendment A1 is a
  parameter-count correction made before results existed).
- Dataset: EuroSAT RGB v2, Zenodo record 7711810, zip SHA-256
  b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90,
  identical file verified on both machines; split seed 20261107,
  21,600 / 2,700 / 2,700, split file byte-identical across
  machines.
- GPU machine: BlackBox, Windows 11, Intel Core Ultra 7 255HX,
  NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB), driver 577.05,
  Python 3.14.7, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
  lightning 2.6.6.
- CPU machine: Muse VM, Ubuntu, 4 cores, 7.7 GB RAM, Python
  3.12.3, torch 2.15.0+cpu, torchvision 0.29.2+cpu,
  lightning 2.6.6.
- Arm order randomised (seed 20261107): results/run_order.json.
- Config-diff audit for H1: results/config_diff.json. Every
  recipe key we control is equal across the two arms; the only
  differing row is the Lightning-only Trainer description, which
  has no hand-loop counterpart.

## Main table (GPU arms, 15 epochs, batch 128, seeds 42 and 43)

| Arm | n (seeds) | Throughput img/s (mean) | Test acc | Test NLL | Peak mem MiB | TTA s |
|---|---|---|---|---|---|---|
| A1 Hand FP32 scratch | 2 | 1655.1 | 0.9580 | 0.1267 | 440 | not reached, not reached |
| A2 Hand FP32 fine-tuned | 2 | 1694.6 | 0.9817 | 0.0734 | 440 | 95, 122 |
| A3 Lightning FP32 scratch | 2 | 2408.4 | 0.9237 | 0.2334 | 441 | not reached, not reached |
| A4 Hand AMP FP16 scratch | 2 | 1865.9 | 0.9583 | 0.1229 | 317 | not reached, not reached |
| A5 Hand FP32 + torch.compile | 2 | 1699.1 | 0.9591 | 0.1297 | 454 | not reached, not reached |
| A6 Hand FP32 + grad ckpt | 2 | 1598.8 | 0.9580 | 0.1270 | 376 | not reached, not reached |

TTA is time to first validation accuracy of 0.96. Only the
fine-tuned arm reached it; the from-scratch arms peak just below
it (A1 validation 0.955 and 0.958 in the final epochs).

## Hypotheses

### H1: hand loop vs Lightning. NOT SUPPORTED.
Throughput ratio A1/A3 = 0.687, 95% CI [0.682, 0.692]: the hand
loop is slower than the Lightning baseline, outside the tie band.
Paired test accuracy difference A1 minus A3 = +3.43 pp, CI
[+2.80, +4.09]: the hand loop is also more accurate, outside the
plus or minus 1.0 pp window. Both criteria fail, in opposite
directions. Two named caveats, both properties of the harness as
frozen: the hand loop synchronises the GPU around every timed
step, which prevents the CPU from running ahead of the GPU and
costs wall-clock throughput; and the accuracy gap is not
explained by any setting in the config-diff audit (epoch-1
validation accuracy is identical at 0.673, the trajectories
diverge from epoch 5), so it is recorded as a framework-internal
difference and an open question, not attributed to a cause we did
not measure.

### H2: AMP. NOT SUPPORTED (prior not met; AMP helped modestly).
Throughput ratio A4/A1 = 1.127, CI [1.121, 1.134], below the
1.30 prior. Accuracy difference +0.04 pp, CI [-0.37, +0.46]:
accuracy is preserved. Peak memory fell 27.8% (439.8 to
317.4 MiB). Reading: with 54% of step time spent waiting on data
(H4), halving compute time cannot move total time much. AMP is
measured here as a real but small win on this workload, not the
1.30x of the prior.

### H3: torch.compile. NOT SUPPORTED (prior not met).
Throughput ratio A5/A1 = 1.027, CI [1.016, 1.039], below the
1.10 prior. Accuracy difference +0.11 pp, CI [-0.30, +0.54].
Compile cost: first training step 2.47 s and 1.98 s in the two
arm runs, 2.46 s in the final probe, all with a warm Inductor
cache; the first compile ever run on this machine, cold cache,
took 35.8 s (environment probe log on BlackBox, not a committed
artefact). Failure modes, as frozen: on the stock Windows stack
torch.compile failed outright with TritonMissing until the
triton-windows package was installed; after that, default mode
and a fullgraph=True probe both compiled and ran (probe first
step 0.025 s warm). Per stop rule S2, the original failure is
part of this result, not edited out.

### H4: profiler. SUPPORTED.
Baseline bottleneck: data loading. At loader configuration L0,
data wait is 53.7% of step time (A7 decomposition, 100 steps).
The frozen rule's optimization, loader configuration L3
(4 workers, pin memory, persistent workers), cuts data wait to
1.0% of step time and mean step time from 81.3 ms to 29.5 ms.
The profiler tables agree on the mechanism: at baseline,
aten::copy_ (host-to-device movement) holds 12.1% of CPU
self-time, second only to convolution backward (20.0%); in the
optimized profile it falls out of the top five and convolution
backward's share rises to 21.6%. Limitation: CUDA activity time
was not capturable on this stack (CUPTI_ERROR_INVALID_DEVICE),
so the operator tables are CPU self-time; the step-time
decomposition is the primary timing evidence.

### H5: data loading. NOT SUPPORTED as frozen.
Loader-only throughput (3 epochs of the train loader, no model):
L0 (workers 0) 2,856 img/s; L1 (pin only) 2,792; L2 (workers 2)
2,243; L3 (workers 4, pin, persistent) 2,812. No configuration
beats workers 0 on raw loader throughput (best/L0 = 1.00), so
the frozen verdict rule is not met, even though L3 nearly
eliminates data wait inside training (H4). The two measurements
answer different questions: raw JPEG decode throughput on this
20-core host is already saturated by one process and extra
workers only add IPC cost, while the training benefit of
workers is prefetch overlap, which a loader-only loop cannot
see. Both facts are the result.

### H6: gradient checkpointing. NOT SUPPORTED (stop rule S3).
Peak memory at batch 128 over 60 steps: 441.2 MiB without
checkpointing, 375.8 MiB with: a 14.8% reduction, below the 30%
criterion. At 64x64 the activations are small next to weights,
gradients, and Adam states, so there is little activation memory
to save. The full-run throughput ratio A6/A1 = 0.966, CI
[0.958, 0.974], sits above the pack's prior band [0.65, 0.90]:
recompute costs almost nothing here because steps are
data-bound, not compute-bound. Accuracy difference 0.00 pp, CI
[-0.17, +0.17]. Batch 512 was feasible with and without
checkpointing on the 8 GB card (the probe ceiling; larger was
not probed). Caveat on the probe: its per-step times were
measured without warmup and in fixed order, so its step_ms
column is order-confounded and is not used for any verdict; the
peak-memory column is the probe's reliable output.

### H7: CPU DDP. NOT SUPPORTED.
Loss trajectories track: per-epoch mean train losses C1
[0.890, 0.585, 0.437] vs C2 [0.899, 0.625, 0.436], mean absolute
difference 0.017 nats, inside the 0.05 rule. Final validation
accuracy does not: C1 0.760 vs C2 0.856, a 9.59 pp difference,
outside the plus or minus 1.0 pp rule (stop rule S4's reporting
duty). At 3 epochs on a cosine schedule the validation accuracy
is still swinging by several points between epochs in both arms
(C1: 0.585, 0.794, 0.760; C2: 0.622, 0.667, 0.856), and the two
arms see different data orders by construction
(DistributedSampler sharding), so the endpoint comparison fails
while the loss trajectories agree. Mean epoch wall time ratio
C2/C1 = 1.19, reported as the communication overhead at this
scale, with no speed claim. Rendezvous history (three attempts,
transport only): DECISIONS.md D8.

## Descriptive results (no hypothesis)

- A2 fine-tuning: test accuracy 0.9817 vs A1 0.9580, paired
  difference +2.37 pp, CI [+1.87, +2.87]; test NLL 0.0734 vs
  0.1267; TTA 95 s and 122 s where A1 never reached the
  threshold. Pretrained weights are worth about 2.4 points and
  the entire time-to-accuracy story on this task.
- A9 FSDP2 check: status ok. fully_shard applied to the four
  ResNet blocks and the root; all 62 parameters are DTensors
  after sharding; one forward and backward step at batch 32 ran
  (loss 1.667, output shape [32, 10]). Shape and initialisation
  claim only.
- Custom autograd: the fused log-softmax plus NLL function in
  src/custom_autograd.py passes torch.autograd.gradcheck in
  double precision and matches F.cross_entropy (test suite).
- GPU state: boundary nvidia-smi snapshots per run show
  temperatures 44 to 55 C and idle clocks at the boundaries (the
  snapshots catch the GPU outside the training loop, so they
  bound thermals loosely rather than prove in-run clocks).

## What this study actually found

One fact organises most of the table: on this machine and task,
training is bound by data loading, not by GPU compute. That is
why AMP gains 1.13x and not 1.30x, why torch.compile gains
1.03x, why gradient checkpointing is nearly free and also saves
little, and why the single effective optimization was a loader
configuration. Framework priors imported from compute-bound
workloads did not survive contact with a 64x64 dataset, and the
pre-registration is what makes that statement evidence instead
of anecdote.

## Reproduce

python -m src.data --verify
python -m src.run_suite --machine gpu   (on BlackBox)
python -m src.run_suite --machine cpu   (on the VM)
python -m src.analyze

Every run JSON carries its own producing command, seed, and
environment block.
