| Arm | n (seeds) | Throughput img/s (mean) | Test acc | Test NLL | Peak mem MiB | TTA s |
|---|---|---|---|---|---|---|
| A1 Hand FP32 scratch | 2 | 1655.1 | 0.9580 | 0.1267 | 440 | not reached, not reached |
| A2 Hand FP32 fine-tuned | 2 | 1694.6 | 0.9817 | 0.0734 | 440 | 95, 122 |
| A3 Lightning FP32 scratch | 2 | 2408.4 | 0.9237 | 0.2334 | 441 | not reached, not reached |
| A4 Hand AMP FP16 scratch | 2 | 1865.9 | 0.9583 | 0.1229 | 317 | not reached, not reached |
| A5 Hand FP32 + torch.compile | 2 | 1699.1 | 0.9591 | 0.1297 | 454 | not reached, not reached |
| A6 Hand FP32 + grad ckpt | 2 | 1598.8 | 0.9580 | 0.1270 | 376 | not reached, not reached |
