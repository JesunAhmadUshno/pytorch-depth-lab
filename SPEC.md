# SPEC: PyTorch Depth Lab (Batch 4, Track B4)

Status: pre-registered design. The frozen protocol lives in PREREG.md.
Research base: Claude verdict pack 10, "Track B4 PyTorch Depth Lab"
(AI Platform Research Programme), verdict CONDITIONAL GO, with the
coordinator verification note of 2026-11-06. Adoption decisions in that
note are binding on this build.

## Research question

How much faster, how much lower-memory, and how much accuracy changes
when a hand-written PyTorch training loop is compared with a high-level
baseline (PyTorch Lightning) on the same task, dataset, and model,
under a pre-registered protocol? Sub-questions: AMP measured, not
assumed; torch.compile measured with warmup and failure modes;
profiling that names the top bottleneck before and after one
optimization; gradient checkpointing traded against memory and
throughput; data loading as a measured variable; CPU DDP with gloo as
a correctness statement; FSDP2 as a shape and initialisation check
only.

## Societal impact (intended, not measured)

Training time is money and energy. Knowing why a training step is slow,
and proving a fix with a profiler instead of folklore, is the
transferable skill this lab evidences: framework-level depth that
carries to any model, any framework, any hardware. The impact
statement is an intention, never a measured claim of this study.

## Task and model

- Dataset: EuroSAT RGB (Helber et al.), 27,000 images, 64x64, 10
  classes. Stratified split 21,600 train / 2,700 validation / 2,700
  test. Licence and version: see LICENSES.md and data/README.md.
- Model: torchvision ResNet-18 (11,689,512 parameters with the
  10-class head), trained from scratch and fine-tuned from
  ImageNet weights (IMAGENET1K_V1).

## Design

One frozen recipe (PREREG.md) for every full arm, so the only
differences between arms are the factors under test:

- GPU arms (BlackBox, NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB):
  hand-written FP32 from scratch; hand-written FP32 fine-tuned;
  Lightning FP32 from scratch (baseline, with a published config-diff
  audit); hand-written AMP (FP16 with GradScaler); hand-written FP32
  with torch.compile; hand-written FP32 with gradient checkpointing;
  a data-loading benchmark across loader configurations; a profiling
  study (baseline vs one optimization); an FSDP2 shape and
  initialisation check.
- CPU arms (this VM, 4 cores): single-process training and
  multi-process CPU DDP with the gloo backend, on a reduced
  3-epoch sub-protocol frozen in PREREG.md. These arms are a
  correctness statement, not a scaling claim.
- A custom autograd function with an analytical gradient, verified
  by gradcheck in the test suite (unit-level deliverable, not a
  training arm).

Arm order is randomised; the realised order is logged. GPU clocks,
temperature, and power are logged per GPU run. Time-to-accuracy and
test NLL are reported alongside accuracy. Hypotheses H1 to H7 and
their frozen verdict rules are in PREREG.md. Hypotheses may fail;
failures are reported, never hidden (pack stop rules S1 to S7).

## Machines

- BlackBox: Windows 11 laptop, Intel Core Ultra 7 255HX, 15.4 GB RAM,
  NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB), driver 577.05.
  Every GPU results table names this GPU exactly. It is a consumer
  laptop GPU, never presented as datacenter hardware.
- Muse VM: Ubuntu, 4 cores, 7.7 GB RAM, Python 3.12.3. CPU arms only.
- Exact package versions are pinned per machine in
  results/versions.json after installation. The verdict pack's
  environment rows were the chat's assumptions and are not quoted
  as ours anywhere in this repo.

## Positioning (how this repo differs)

- P01 (llm-from-scratch) builds a language model from primitives;
  this lab builds no model architecture. It measures the training
  framework itself on a fixed, well-known architecture.
- P03 (vllm-serving-lab) is inference serving and optimisation of a
  running model; this lab is training, and it never serves.
- P04 (cuda-kernels-lab) writes CUDA kernels below the framework;
  this lab stays strictly at the PyTorch framework level and makes
  no kernel, compiler, or MLIR claim.
- P14 (structured-outputs-lab) is about model output contracts;
  unrelated except as a sibling study under the same
  pre-registration standard.
- P19 (distributed-training-lab) studies distributed training as
  the subject; this lab's only distributed element is a CPU DDP
  correctness check, and it makes no scaling claim.

## Where the claim stops

Framework-level depth only, on one consumer GPU and one 4-core VM.
No multi-GPU scaling, no NCCL, no CUDA kernel or compiler claims.
FSDP2 is a shape and initialisation check only. Numbers do not
generalise beyond the named GPU, driver, and pinned package
versions; every headline number carries its baseline row, n, seeds,
interval, dataset version, and producing command (see RESULTS.md).

## Deliverables

README.md, SPEC.md, PREREG.md, DECISIONS.md, METHODS.md, RESULTS.md,
ROADMAP.md, AUDIT.md, LICENSES.md, data/README.md, src/ (train.py,
model.py, data.py, benchmark.py, profile.py, lightning_arm.py,
ddp_cpu.py, fsdp_check.py, custom_autograd.py, analyze.py,
run_suite.py), tests/, results/ (raw run JSON, versions, split,
summary). The pack also lists a small workshop-format paper under
paper/; that item is deferred to ROADMAP.md, not silently dropped.
