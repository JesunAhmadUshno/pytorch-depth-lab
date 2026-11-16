# PyTorch Depth Lab

**Question.** On one consumer GPU, how much faster, lighter on memory,
and how different in accuracy is a hand-written PyTorch training loop
against a PyTorch Lightning baseline, and what do AMP, torch.compile,
gradient checkpointing, profiling and data loading each actually buy
when measured rather than assumed?

**Headline.** One of seven pre-registered hypotheses held: ResNet-18
training on EuroSAT was bound by data loading, not GPU compute, so AMP
reached 1.13x throughput (95% CI 1.12 to 1.13) against its 1.30x prior
and torch.compile 1.03x (95% CI 1.02 to 1.04) against 1.10x, on one RTX
5050 Laptop GPU with n = 2 seeds per arm.

**Field.** ML and AI research. **Job family.** ML engineering (training
performance).

**Verdict.** Evaluated. PREREG.md was committed before any code or
result (565dfc0, tag proposal-v1); all twelve GPU runs, the benchmarks
and both CPU arms ran and their raw JSON is in results/; RESULTS.md
scores all seven hypotheses under the frozen rules, six of them NOT
SUPPORTED; tests pass in CI on Python 3.11 and 3.12, and CI regenerates
results/summary.json from the raw runs unchanged. Not v1.0 Released: no
second reader has re-derived the numbers yet.

## Reproduce

Checks, anywhere (CPU only, no dataset, no GPU):

```text
pip install -r requirements.txt
python -m pytest -q
python -m src.analyze && git diff --exit-code results/summary.json results/readme_table.md
```

`src.analyze` recounts every verdict and number in RESULTS.md from the
raw run files in results/runs/, results/benchmarks/ and results/cpu/.

The full study (needs the dataset and the named machines; see
data/README.md and METHODS.md):

```text
python -m src.data --verify
python -m src.run_suite --machine gpu   (BlackBox, RTX 5050 Laptop GPU)
python -m src.run_suite --machine cpu   (4-core VM)
python -m src.analyze
```

Every run JSON carries its own producing command, seed, and
environment block.

## Lifecycle

- Question: DONE 2026-11-07
- Proposal: DONE 2026-11-07
- Baseline: DONE 2026-11-07
- Build: DONE 2026-11-07
- Evaluate: DONE 2026-11-07
- Report: DONE 2026-11-07
- Review: OPEN
- Release: OPEN

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

In plain terms: gradient checkpointing saved 14.8% memory against a
30% criterion, the hand loop was slower than Lightning (0.69x) yet
more accurate (+3.4 pp), and CPU DDP tracked the loss trajectory but
failed the endpoint accuracy window. Full numbers, intervals, and the
failures as measured: RESULTS.md. Limitations and threats to
validity: AUDIT.md.

## Layout

Files named in the locked pre-registration commit (PREREG.md and the
deliverables list in SPEC.md) keep their names; the second column gives
the role each plays in the portfolio standard.

| Path | Role | What it holds |
| --- | --- | --- |
| PREREG.md | proposal | Frozen protocol, hypotheses and verdict rules (tag proposal-v1); amendments append-only; a test fails if the frozen text changes |
| SPEC.md | proposal | Research question, design and positioning; frozen with PREREG.md at 565dfc0; one declared correction (A1) in 8f2ac1b, pinned by a test |
| RESULTS.md | report | Provenance, main table, every hypothesis scored, misses, what the study found |
| AUDIT.md | report | Self-audit: what was verified, limitations and threats to validity |
| METHODS.md | report | How the frozen protocol is implemented |
| DECISIONS.md | decisions | Builder decisions D1 to D11, dated, with reasons |
| ROADMAP.md | future work | Deferred and follow-up items, named so nothing is dropped |
| results/runs/, results/benchmarks/, results/cpu/ | evidence | Raw per-run JSON and test-correctness arrays, as produced on the machines |
| results/dataset.json, split.json, run_order.json, config_diff.json, versions*.json | evidence | Dataset hash, frozen split, realised arm order, Lightning config audit, pinned versions |
| results/summary.json, results/readme_table.md | results | Scored results, regenerated from the evidence by src/analyze.py |
| src/ | code | Hand loop, Lightning arm, benchmarks, profiler, DDP, FSDP2 check, analysis |
| tests/ | tests | Split, model, gradcheck, analysis statistics, layout and lock |
| data/README.md | data | How to fetch EuroSAT; no dataset file is committed |
| LICENSES.md | licences | Dataset, weights and library licences |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | project | Change history, citation, security policy, Apache-2.0 |

Not present, on purpose: experiments/ (the locked protocol names
results/ as the evidence location, so raw evidence stays there), site/
(no dashboard), docs/decisions/ (DECISIONS.md is the locked decisions
file).

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
