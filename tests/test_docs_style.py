"""Markdown style gate: the dash and word rules from AGENTS.md.

Scans every tracked Markdown file outside fenced code blocks and inline code.
The natural-writing skill is exempt because it quotes the words it bans.
Runs without TensorFlow, so `.github/workflows/docs.yml` can run it alone.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXEMPT = {".claude/skills/natural-writing/SKILL.md", "LICENSE"}

# Whole words or phrases from the natural-writing kill list that have no
# legitimate technical meaning in this repo. Words like "enhance" or "robust"
# are kept out of the gate on purpose: they would flag quoted paper titles.
BANNED = [
    "delve",
    "tapestry",
    "pivotal",
    "testament to",
    "showcase",
    "seamless",
    "seamlessly",
    "cutting-edge",
    "groundbreaking",
    "leverage",
    "leverages",
    "leveraging",
    "utilize",
    "utilizes",
    "utilizing",
    "supercharge",
    "in order to",
    "due to the fact",
    "it is important to note",
    "it's worth noting",
    "state-of-the-art",
    "honestly",
]
BANNED_RE = re.compile(r"\b(" + "|".join(map(re.escape, BANNED)) + r")\b", re.I)
EM_DASH_RE = re.compile("[—]")
# A hyphen or en dash with a space on both sides, between words, used as a
# dash. List bullets and table cells start with "-" or "|" and are skipped.
SPACED_DASH_RE = re.compile(r"[A-Za-z0-9).,:;'\"*`] [-–] [A-Za-z(\"'*`]")
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def markdown_files():
    """Tracked Markdown files, relative to the repo root."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.splitlines() if p not in EXEMPT and (ROOT / p).exists()]


def prose_lines(path):
    """(line number, text) for lines outside fenced code, inline code removed."""
    in_fence = False
    for n, line in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield n, INLINE_CODE_RE.sub("``", line)


def violations(path):
    """Every style violation in one file, as 'path:line: reason' strings."""
    found = []
    for n, line in prose_lines(path):
        if EM_DASH_RE.search(line):
            found.append(f"{path}:{n}: em dash")
        if not line.lstrip().startswith(("|", "-", "*")) and SPACED_DASH_RE.search(
            line
        ):
            found.append(f"{path}:{n}: spaced hyphen used as a dash")
        for match in BANNED_RE.finditer(line):
            found.append(f"{path}:{n}: banned word '{match.group(0)}'")
    return found


def test_markdown_follows_the_writing_rules():
    problems = [v for path in markdown_files() for v in violations(path)]
    assert not problems, "See AGENTS.md, Writing:\n" + "\n".join(problems)


def test_the_checker_catches_what_it_should(tmp_path, monkeypatch):
    sample = tmp_path / "sample.md"
    sample.write_text(
        "A result — with an em dash.\n"
        "Accuracy went up - by a lot.\n"
        "We leverage augmentation.\n"
        "- a list item - is fine\n"
        "`a - b` in code is fine\n"
        "```\nx = a - b  # fenced code is fine\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    found = violations("sample.md")
    assert found == [
        "sample.md:1: em dash",
        "sample.md:2: spaced hyphen used as a dash",
        "sample.md:3: banned word 'leverage'",
    ]
