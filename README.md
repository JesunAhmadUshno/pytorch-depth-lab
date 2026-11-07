# PyTorch Depth Lab

A pre-registered measurement study of PyTorch training at the
framework level: a hand-written training loop against a PyTorch
Lightning baseline on EuroSAT land-use classification with
ResNet-18, with AMP, torch.compile, gradient checkpointing,
profiling, and data loading each measured rather than assumed, and
a CPU DDP (gloo) arm that states what it can honestly prove.

Status: PRE-REGISTERED. The protocol was frozen in PREREG.md
before any training run. No results yet.
Pre-registration commit: recorded in the results-stage commit
(PREREG.md itself is the frozen file; its first commit precedes
every results file in this repo's history).

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
