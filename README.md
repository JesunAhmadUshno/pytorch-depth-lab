# PyTorch Depth Lab

A pre-registered measurement study of PyTorch training at the
framework level: a hand-written training loop against a PyTorch
Lightning baseline on EuroSAT land-use classification with
ResNet-18, with AMP, torch.compile, gradient checkpointing,
profiling, and data loading each measured rather than assumed, and
a CPU DDP (gloo) arm that states what it can honestly prove.

Status: COMPLETE. The protocol was frozen in PREREG.md before
any training run, and the study was executed as registered.
Pre-registration commit:
565dfc0cd8c9bebee94166d53fb9d3756ab62150 (its first commit
precedes every results file in this repo's history; one
pre-results amendment, A1, corrects a parameter count).

Headline outcome: on this machine the workload is bound by data
loading, not GPU compute, and most imported priors failed
because of it. Of seven frozen hypotheses, one is supported
(H4, profiling). AMP reached 1.13x against a 1.30 prior,
torch.compile 1.03x against 1.10, gradient checkpointing saved
14.8% memory against a 30% criterion, the hand loop was slower
than Lightning (0.69x) yet more accurate (+3.4 pp), and CPU DDP
tracked the loss trajectory but failed the endpoint accuracy
window. Full numbers, intervals, and the failures as measured:
RESULTS.md. Grade and deductions: AUDIT.md.

## Results (GPU arms, RTX 5050 Laptop GPU, 8 GB)

| Arm | n (seeds) | Throughput img/s (mean) | Test acc | Test NLL | Peak mem MiB | TTA s |
|---|---|---|---|---|---|---|
| A1 Hand FP32 scratch | 2 | 1655.1 | 0.9580 | 0.1267 | 440 | not reached, not reached |
| A2 Hand FP32 fine-tuned | 2 | 1694.6 | 0.9817 | 0.0734 | 440 | 95, 122 |
| A3 Lightning FP32 scratch | 2 | 2408.4 | 0.9237 | 0.2334 | 441 | not reached, not reached |
| A4 Hand AMP FP16 scratch | 2 | 1865.9 | 0.9583 | 0.1229 | 317 | not reached, not reached |
| A5 Hand FP32 + torch.compile | 2 | 1699.1 | 0.9591 | 0.1297 | 454 | not reached, not reached |
| A6 Hand FP32 + grad ckpt | 2 | 1598.8 | 0.9580 | 0.1270 | 376 | not reached, not reached |

Verdicts (frozen rules, PREREG.md): H1 NOT SUPPORTED,
H2 NOT SUPPORTED, H3 NOT SUPPORTED, H4 SUPPORTED,
H5 NOT SUPPORTED, H6 NOT SUPPORTED, H7 NOT SUPPORTED.

## Why this exists

Applied ML work expects an engineer to know why a training step is
slow and to prove a fix with a profiler. This lab is that evidence,
produced under the Gap-Closure Programme standard: hypotheses and
verdict rules frozen before results, baselines beside every number,
failures reported under stop rules instead of hidden.

## How it differs from sibling studies

- P01 (llm-from-scratch) builds a model from primitives; this lab
  measures the training framework on a fixed architecture.
- P03 (vllm-serving-lab) optimises inference serving; this lab is
  training only.
- P04 (cuda-kernels-lab) works below the framework in CUDA; this
  lab stays at the PyTorch framework level and claims nothing
  about kernels, compilers, or MLIR.
- P14 (structured-outputs-lab) concerns output contracts; a
  sibling under the same pre-registration standard only.
- P19 (distributed-training-lab) studies distributed training;
  this lab's CPU DDP arm is a correctness check with no scaling
  claim.

Full design: SPEC.md. Frozen protocol and verdict rules:
PREREG.md. Decisions taken by the builder: DECISIONS.md. Methods:
METHODS.md. Results: RESULTS.md. Audit and grade: AUDIT.md.

## Machines

- GPU arms: BlackBox, a Windows 11 laptop with an NVIDIA GeForce
  RTX 5050 Laptop GPU (8 GB), driver 577.05. A consumer laptop
  GPU, named exactly on every table, never presented as
  datacenter hardware.
- CPU arms: a 4-core Ubuntu VM (7.7 GB RAM), Python 3.12.3.
- Pinned package versions per machine: results/versions.json.

## Claim boundary

Framework-level depth only, on the named hardware and versions.
No multi-GPU scaling, no NCCL, no kernel or compiler claims.
FSDP2 appears only as a shape and initialisation check. CPU DDP
proves correctness of the recipe under data parallelism, not
scaling.

## Licence

Code: Apache-2.0 (LICENSE). Dataset: EuroSAT, MIT, used under its
terms; no dataset files are committed here. See LICENSES.md.
