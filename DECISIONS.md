# DECISIONS: PyTorch Depth Lab

Builder decisions, dated. Protocol-defining choices live in the
frozen PREREG.md; this file records implementation choices and
anything decided after the freeze, with reasons.

- D1 (2026-11-07): Dataset file is EuroSAT_RGB.zip from Zenodo
  record 7711810 (Version v2), the authors' current hosting. The
  older DFKI links are marked deprecated by the official repo.
- D2 (2026-11-07): EuroSAT licence taken as MIT from the official
  repository's License section, checked before download. The
  verdict pack's CC BY-NC-SA 4.0 row was not confirmed by any
  official source. Recorded in LICENSES.md.
- D3 (2026-11-07): CPU DDP world size is 2 (see PREREG.md, CPU
  arms) rather than the pack's illustrative 4 processes: the VM's
  4 cores are shared, and 2 ranks at 2 threads each avoid
  oversubscription. No speed claim rests on this arm.
- D4 (2026-11-07): AMP means torch.autocast FP16 with GradScaler
  (the classic AMP configuration), recorded in PREREG.md. BF16
  was not substituted silently.
- D5 (2026-11-07): Package versions are whatever pip resolves
  from the official indexes on each machine at install time
  (PyTorch CPU wheel index for the VM, cu128 wheel index for
  BlackBox), then pinned in requirements-lock.txt and
  results/versions.json. The pack's versions (torch 2.14.1,
  Python 3.14) were the chat's assumptions and are never quoted
  as ours.
- D6 (2026-11-07): The workshop paper listed in the pack's
  deliverables is deferred to ROADMAP.md. The study, code,
  results, and audit are this build's deliverables.
- D8 (2026-11-07): CPU DDP rendezvous history, three attempts.
  Attempt 1 (env:// store) stalled: TCPStore listening, peer rank
  never joined over the VM's primary interface. Attempt 2
  (tcp://127.0.0.1 store, GLOO_SOCKET_IFNAME=lo) failed at the
  store: "Connection reset by peer" for rank 1. Attempt 3 uses a
  FileStore rendezvous (init_method file://, unique path per
  launch) with GLOO_SOCKET_IFNAME=lo for peer traffic. Transport
  plumbing only; nothing about the correctness question H7 asks.
- D10 (2026-11-07): Profiler metric fallback. On BlackBox,
  Kineto reports CUPTI_ERROR_INVALID_DEVICE and CUDA activity
  time is unavailable; torch 2.11 also renamed the attribute
  (self_device_time_total). src/profile.py now selects the
  strongest self-time metric present and records
  cuda_activity_captured in the payload. The profiler evidence
  for H4 is then the CPU-side operator table beside the A7
  step-time decomposition; the limitation is stated in
  RESULTS.md, not hidden.
- D9 (2026-11-07): Lightning arm fix and re-run. A3 as first
  written failed after fit: Lightning restores the module to its
  pre-fit device (CPU) when fit returns, so the final shared
  evaluate() fed CUDA batches to a CPU model. Fix: move the model
  back to the run device before the final evaluation. Both A3
  seeds were re-run with the fix under the identical frozen
  protocol; the failed attempts are in the suite log.
- D7 (2026-11-07): BlackBox work directory is
  C:\Users\Jesun\b4-pytorch-depth. GPU code runs there from a
  synced copy of src/ plus its own data directory, so the laptop
  is never a second source of truth: results JSON files are
  copied back into this repo's results/ tree.
