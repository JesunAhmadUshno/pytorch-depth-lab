# Gap-closure re-execution: pytorch-depth-lab GPU suite, 2026-11-17

Run of record: DECISIONS.md D12. Machine: BlackBox (RTX 5050 Laptop
GPU, driver 577.05), environment of record
C:\Users\Jesun\b4-pytorch-depth\.venv (torch 2.11.0+cu128,
torchvision 0.26.0+cu128, lightning 2.6.6). Dataset zip SHA-256
verified before the run: b4f5b234... (matches results/dataset.json).
Clocks not locked, AC power, protocol exactly as frozen. All suite
steps returned rc=0 (12 training arms, A7 loader benchmark, A8
profiles, A10 memory probe, compile probes, A9 FSDP check), then
src.analyze ran clean. Fresh outputs are in this directory; the
committed results were not touched and no verdict is re-issued.

## Test suite

14 passed, 1 skipped (LF checkout). A first run in a CRLF checkout
(Git for Windows core.autocrlf=true on this machine) failed the
three SHA-256 pin tests in tests/test_layout.py; the pins match the
committed blobs exactly (verified by hashing the blobs), so the
frozen text is unchanged and the failures were a checkout artifact.

## What reproduced exactly

- Test accuracy and NLL per arm (two-seed means): A1 0.9580 /
  0.1267, A2 0.9817 / 0.0734, A3 0.9237 / 0.2334, A4 0.9583 /
  0.1229, A6 0.9580 / 0.1270: identical to the committed record.
  A5 (torch.compile) moved by one test image class: 0.9591 ->
  0.9585 accuracy, NLL 0.1297 -> 0.1298.
- Peak memory per arm: identical (A1 439.8, A2 439.8, A3 441.1,
  A4 317.4, A6 376.4 MiB), except A5 450.8 -> 449.3 MiB.
- H7 config-diff losses: bit-identical; epoch time ratio 1.1930
  both passes.
- FSDP check: ok, 62 DTensors, both passes.
- H6 memory side: reduction 14.83 percent and batch-512
  feasibility identical; H2 memory saving 27.84 percent identical.
- A2 minus A1 accuracy difference: 0.0237 with the same CI.

## What moved: throughput, and two loader-sensitive verdicts

Every arm's throughput fell (the pass ran with the machine under
other load; clocks floated, as in the original protocol). A1
epoch-mean throughput 1,655.2 -> 1,145.9 img/s; A3 2,408.7 ->
2,026.7; A4 1,866.1 -> 1,474.6; A5 1,699.9 -> 1,265.3; A6 1,599.4
-> 1,389.4. The single-process loader (L0) fell hardest: 2,855.8
-> 1,869.5 img/s, while the L3 loader held (2,812.2 -> 2,793.1).
Verdicts recomputed by the frozen rules in src.analyze:

| H | Committed | Fresh | Note |
| --- | --- | --- | --- |
| H1 | NOT SUPPORTED | NOT SUPPORTED | ratio 0.687 -> 0.565 |
| H2 | NOT SUPPORTED | NOT SUPPORTED | ratio 1.127 -> 1.287, CI wider |
| H3 | NOT SUPPORTED | SUPPORTED | ratio 1.027 -> 1.104 (frozen line 1.10); compile step time rose 2.0-2.5 s -> 14.5-59.7 s |
| H4 | SUPPORTED | SUPPORTED | data-wait share 53.7% -> 68.1% before, 1.0% -> 0.9% after |
| H5 | NOT SUPPORTED | SUPPORTED | best config L0 -> L3; L3/L0 1.494 |
| H6 | NOT SUPPORTED | NOT SUPPORTED | throughput ratio 0.966 -> 1.213, above the frozen band on the other side |
| H7 | NOT SUPPORTED | NOT SUPPORTED | all figures identical |

Reading: the study's accuracy, loss, memory, and feasibility
record reproduces exactly. The two verdicts that flipped (H3, H5)
are the ones whose frozen rules compare against the data-bound
baseline arms, and they flipped because the baseline got slower,
not because the optimised arms got faster. Any future version of
the paper should carry this sensitivity note beside H3 and H5.
