"""Source files must be NFC-stable, so test data cannot rot into a passing no-op.

This package is about text that Unicode normalization REWRITES. The characters that
matter here -- the Hebrew presentation forms, NFD sequences -- are exactly the
characters `c` for which `NFC(c) != c`. That is also exactly the set that does not
reliably survive being typed, pasted, or piped through a normalizing editor, terminal
or shell heredoc: the tool applies NFC on the way in and the presentation form silently
becomes its decomposition.

The result is the nastiest kind of broken test. `U+FB2E` in a source file quietly turns
into `U+05D0 U+05B7`; both characters now sit inside the ordinary Hebrew block; nothing
drifts and nothing splits; the test asserts something true about the wrong input and
PASSES. Every regression it was written to catch is now invisible, and there is no
failure anywhere to tell you.

This happened repeatedly while the span-offset work was being done -- to hand-written
checks and to a benchmark whose headline number was meaningless as a result.

So: never write such a character as a literal. Write it as an escape (`"\\uFB2E"`),
which is plain ASCII and survives anything. This test enforces that mechanically, for
every source and test file, so nobody has to remember.
"""
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for d in ("src", "tests") for p in (ROOT / d).rglob("*.py"))


def _offenders(text):
    """(index, char) for every character NFC would rewrite."""
    return [(i, c) for i, c in enumerate(text) if unicodedata.normalize("NFC", c) != c]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_source_file_is_nfc_stable(path):
    """No source file may contain a character that NFC would rewrite."""
    text = path.read_text(encoding="utf-8")
    if unicodedata.normalize("NFC", text) == text:
        return

    lines = text.splitlines()
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1

    report = []
    for idx, ch in _offenders(text):
        lineno = max(i for i, s in enumerate(starts) if s <= idx) + 1
        name = unicodedata.name(ch, "<unnamed>")
        report.append(
            f"    line {lineno}: U+{ord(ch):04X} {name}\n"
            f"      -> write it as \"\\u{ord(ch):04X}\" instead of the character itself")
    pytest.fail(
        f"{path.relative_to(ROOT)} is not NFC-stable.\n"
        + "\n".join(report)
        + "\n\n    A character NFC rewrites cannot survive an editor, terminal or heredoc"
          "\n    intact. If it decomposes in place, tests over it keep PASSING while"
          "\n    silently testing ordinary text. Use the escape; it is ASCII and safe.")


def test_the_guard_actually_detects_a_decomposing_character():
    """Guard the guard: prove the check fires on the exact character that caused this."""
    assert _offenders(chr(0xFB2E)), "U+FB2E must be detected as NFC-unstable"
    assert not _offenders("shalom"), "plain ASCII must not be flagged"
    assert not _offenders(chr(0x05D0)), "an ordinary Hebrew letter must not be flagged"
