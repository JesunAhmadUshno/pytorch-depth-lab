# ROADMAP: deferred and follow-up items

Named openly so nothing is silently dropped.

1. Workshop paper (from the pack's deliverables): one small
   paper, workshop format on ML systems, arguing the joint
   measurement of AMP, torch.compile, and checkpointing under a
   pre-registered protocol on a single consumer GPU. Deferred
   from this build by DECISIONS.md D6.
2. More seeds: n = 2 per arm keeps intervals honest but wide.
   A v1.1 pass at n = 5 for A1, A3, A4 would tighten the H1 and
   H2 intervals.
3. BF16 arm: AMP here is FP16 with GradScaler (the frozen
   definition). A BF16 comparison arm is a legitimate follow-up,
   labelled as its own arm, never substituted retroactively.
4. channels_last as a full arm (it appears in this build only
   inside the A8 profiling rule and the A7 decomposition).
5. Multi-GPU and NCCL: not tested on this hardware (one GPU).
   Any future claim needs a multi-GPU machine and a new
   pre-registration; nothing here pre-approves it.
6. torch.compile on a second stack (a Linux GPU host) to separate
   the Windows stack's failure modes from Inductor's own.
