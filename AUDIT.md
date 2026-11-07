# AUDIT: PyTorch Depth Lab

Self-audit against the Gap-Closure Programme charter, written
after the results were generated and recounted by src/analyze.py
from the raw artefacts. Grade: 7,200 of 10,000.

## What was verified

- Pre-registration commit 565dfc0cd8c9bebee94166d53fb9d3756ab62150
  precedes every results file. Amendment A1 (the pack's
  parameter count was the ImageNet-head figure; this study's
  10-class model has 11,181,642 parameters) was appended before
  any run and changes no verdict rule.
- EuroSAT licence was confirmed from the dataset's official
  repository before download: MIT. The pack's CC BY-NC-SA 4.0
  row was not confirmed anywhere official and is not used
  (LICENSES.md).
- The same zip (SHA-256 b4f5b234...) was used on both machines;
  the split files are byte-identical across machines; the
  config-diff artefact is identical across machines.
- All 12 GPU runs, 6 benchmark artefacts, and both CPU arms are
  present as raw JSON with producing commands, seeds, and
  environment blocks. Verdicts in RESULTS.md reproduce from
  results/summary.json under the frozen rules.
- Failures were kept, not edited out: torch.compile failed with
  TritonMissing on the stock Windows stack before triton-windows
  was installed; both Lightning runs first failed on a device
  placement bug after fit (fixed, DECISIONS.md D9); the profiler
  could not capture CUDA activity (CUPTI_ERROR_INVALID_DEVICE)
  and fell back to CPU self-time with the limitation stated
  (D10); CPU DDP needed three rendezvous attempts (D8).
- Jev claim_check on the headline claim (data-loading-bound
  finding with the AMP and compile ratios): model jev-1.13.0,
  noul 0.19, route REVIEW (bands: clear 0.85+, block 0.10 or
  below). On the REVIEW route the claim was hand-verified
  against a7_loader.json, summary.json, and PREREG.md at the
  pre-registration commit; every figure matched. Record:
  results/jev-claim-check.json.

## Deductions (total 2,800)

- 500: the A3 accuracy gap (+3.43 pp for the hand loop) is
  unexplained. The config-diff audit proves every setting we
  control is equal, but the study carries no per-step loss
  instrumentation to isolate the framework-internal cause.
- 400: two seeds per GPU arm. The pack allowed two to three;
  the minimum was run. Accuracy intervals are paired bootstraps
  over test examples, not over seeds.
- 350: one dataset, one model, one GPU. External validity is
  narrow by design and stays narrow in every claim.
- 300: the CPU DDP sub-protocol is thin (3 epochs, world size 2,
  one seed). Its H7 endpoint failure is partly an artifact of
  endpoint noise at 3 epochs, which the write-up states but the
  design cannot rule out.
- 250: no in-run GPU clock or thermal series. Boundary
  nvidia-smi snapshots catch idle clocks, so boost behaviour
  during training is unverified.
- 250: profiler evidence is CPU self-time only (CUPTI failed on
  this stack). The H4 verdict rests on the step decomposition,
  with the operator table as support.
- 200: the A10 memory probe ran without warmup in fixed order.
  Its step-time column is order-confounded and unused; only its
  peak-memory column supports H6.
- 150: TTA is uninformative for the from-scratch arms: the 0.96
  threshold sits just above their 15-epoch ceiling, so
  time-to-accuracy compares only through the fine-tuned arm.
- 150: profiler trace files are not committed (gitignored), so
  the profiler evidence in the repo is the JSON summary, not the
  raw traces.
- 150: the A7 loader benchmark runs its configurations in fixed
  sequence inside one process; order and thermal drift across
  its 136 s are not controlled.
- 100: one figure in RESULTS.md (the 35.8 s cold-cache first
  compile) rests on the BlackBox setup log, not a committed
  artefact. It is labelled as such where it appears.

## What the grade is not

It is not a grade of the hypotheses. Six of seven failed against
their frozen rules, and that is the study working as designed:
the pre-registration, the paired statistics, and the failure
records are what the grade rewards. A version of this repo that
reported only the supported hypothesis would grade lower.

## Open items carried to ROADMAP.md

Per-step loss instrumentation for the hand-vs-Lightning gap; a
third seed; in-run clock logging; a compute-bound second task
where the AMP and compile priors get a fair test.
