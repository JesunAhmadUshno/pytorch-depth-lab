"""Portfolio standard checks: the lock holds, the required files exist, no public grade."""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SHA-256 of PREREG.md as committed in 565dfc0 (tag proposal-v1), before any code or result.
LOCKED_SHA256 = "be51b55fe275d764d9a8b1440f1d60a9e87884e8a3f4394a03523c9ed9d7a7d9"
# SHA-256 of PREREG.md with Amendment A1 (8f2ac1b, before any results). A new amendment updates this pin.
AMENDED_SHA256 = "e9b5251c31dcc2369c0b3b7e10fbf26c35d0d131a59ef5e22ba64d94259c32ed"
AMENDMENTS = b"## Amendments\n\n"
# SHA-256 of SPEC.md as of 8f2ac1b: frozen with PREREG.md at 565dfc0, one declared correction (Amendment A1).
SPEC_SHA256 = "a3c2e9e490822fd055bc74385c0e42330dfc78482fb3927fc7f58ca96f3ef713"


def _prereg():
    with open(os.path.join(ROOT, "PREREG.md"), "rb") as f:
        return f.read()


def test_frozen_text_is_unchanged_since_the_lock():
    text = _prereg()
    frozen = text[: text.index(AMENDMENTS) + len(AMENDMENTS)] + b"(none at freeze time)\n"
    assert hashlib.sha256(frozen).hexdigest() == LOCKED_SHA256, \
        "PREREG.md frozen text changed after its lock; append an amendment instead"


def test_amendments_are_pinned():
    assert hashlib.sha256(_prereg()).hexdigest() == AMENDED_SHA256, \
        "PREREG.md amendments changed; record why in DECISIONS.md and update AMENDED_SHA256"


def test_spec_is_pinned():
    with open(os.path.join(ROOT, "SPEC.md"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == SPEC_SHA256, \
            "SPEC.md changed after its A1 correction; declare a PREREG amendment and update SPEC_SHA256"


def test_required_files_exist():
    need = ["README.md", "PREREG.md", "SPEC.md", "RESULTS.md", "AUDIT.md", "METHODS.md", "DECISIONS.md",
            "ROADMAP.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE", "LICENSES.md",
            "src/analyze.py", "results/summary.json", "results/dataset.json", "results/split.json"]
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, "missing: " + ", ".join(missing)
    assert os.listdir(os.path.join(ROOT, "results", "runs")), "no raw runs in results/runs/"


def test_no_public_grade():
    for name in ("README.md", "AUDIT.md", "RESULTS.md", "ROADMAP.md", "METHODS.md", "SPEC.md", "CHANGELOG.md"):
        text = open(os.path.join(ROOT, name), encoding="utf-8").read()
        for bad in ("of 10,000", "/10000", "Grade:", "Deductions"):
            assert bad not in text, name + " carries a grade (" + bad + "); grades are internal"
