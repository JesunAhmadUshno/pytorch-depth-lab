# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning; no release is tagged yet.

## [Unreleased]

### Changed
- 2026-11-08: Adopted the portfolio standard v2. README opens with the
  question, headline, field, job family and verdict (Evaluated), then
  reproduce commands, lifecycle and a layout table. AUDIT.md keeps its
  content as limitations and threats to validity; its grade and
  deduction points left the public repo (grades are internal). The
  lock is tagged proposal-v1 on 565dfc0, and a test fails if the
  frozen text of PREREG.md changes (DECISIONS.md D11). CI runs with
  read-only permissions and checks that results/summary.json and
  results/readme_table.md regenerate unchanged from the raw runs.
  CodeQL and Dependabot added; main protected. pyproject version set
  to 0.1.0, since no release has been tagged.

### Added
- 2026-11-07: Results: all GPU arms (two seeds each), the loader,
  memory, compile, profiler and FSDP2 benchmarks, and both CPU arms;
  six of seven hypotheses NOT SUPPORTED (RESULTS.md); self-audit
  (AUDIT.md); follow-ups (ROADMAP.md).
- 2026-11-07: Fixes from the first runs (DECISIONS.md D8, D9, D10).
- 2026-11-07: Build: harness, arms, benchmarks, tests, CI; PREREG
  Amendment A1 (parameter count) before any results.
- 2026-11-07: Pre-registration (565dfc0): SPEC.md and PREREG.md
  frozen before any code or result.
