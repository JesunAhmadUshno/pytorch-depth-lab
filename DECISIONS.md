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
- D7 (2026-11-07): BlackBox work directory is
  C:\Users\Jesun\b4-pytorch-depth. GPU code runs there from a
  synced copy of src/ plus its own data directory, so the laptop
  is never a second source of truth: results JSON files are
  copied back into this repo's results/ tree.
