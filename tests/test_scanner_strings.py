"""No tracked file may look like malware to the server's scanner.

The vahinitech.com server runs maldet nightly over home folders and /tmp.
It flagged our own test payloads on 2026-10-16, and on 2026-10-19 found one
in a .pyc: the source split it with +, but Python folds "a" + "b" into one
constant when it compiles. Payloads are joined at run time with a call such
as b"".join(...), and every tracked .py is compiled here and its constants
checked too. The rule:
https://github.com/vahinitech/Umbrella/blob/main/skills/scanner-strings/SKILL.md

Where maldet's hex.dat is readable, every literal signature in it is checked
as well. VAHINI_MALDET_SIGS points at another file; "none" skips it.
"""

import os
import re
import subprocess
import types

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Every pattern is assembled at run time with a call, so neither this file
# nor its compiled form holds a joined signature.
OPEN = b"".join((b"<", b"?php"))
CALLS = b"system|eval|exec|passthru|shell_exec|popen|proc_open|assert|base64_decode"
BUILT_IN = (
    (
        "PHP open tag joined to a shell or eval call",
        re.compile(re.escape(OPEN) + rb"\s*@?(" + CALLS + rb")\s*\(", re.I),
    ),
    (
        "EICAR test string",
        re.compile(b"".join((b"EICAR-", b"STANDARD-ANTIVIRUS-", b"TEST-FILE"))),
    ),
)
SHELL_NAMES = re.compile(
    r"(^|/)(shell|cmd|c99|r57|wso|b374k|webshell|backdoor)\d*\.(php\d?|phtml|asp|aspx|jsp)$",
    re.I,
)
# Media, data and model files are not text anyone writes; checking them
# against 2,000+ signatures is slow, so the maldet half skips them.
MEDIA = re.compile(
    r"\.(pdf|png|jpe?g|webp|gif|ico|woff2?|ttf|otf|onnx|pt|pth|h5|keras|tflite|npz|npy|bin|zip)$",
    re.I,
)


def load_maldet():
    """Literal hex signatures from maldet's hex.dat, if it is readable."""
    path = os.environ.get("VAHINI_MALDET_SIGS", "/usr/local/maldetect/sigs/hex.dat")
    if path == "none":
        return []
    try:
        with open(path, encoding="latin-1") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    sigs = []
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) < 2 or not re.fullmatch(r"([0-9a-fA-F]{2})+", parts[0]):
            continue  # "*" wildcard lines target binaries and archives
        sigs.append((parts[1], bytes.fromhex(parts[0])))
    return sigs


def find_hits(rel, data, maldet):
    """Reasons a scanner would flag this file, or an empty list."""
    hits = []
    if SHELL_NAMES.search(rel):
        hits.append("file name looks like a web shell")
    hits += [name for name, pattern in BUILT_IN if pattern.search(data)]
    if not MEDIA.search(rel):
        hits += ["maldet " + name for name, sig in maldet if sig in data]
    return hits


def constants(code):
    """Every str and bytes constant in a code object, nested ones included."""
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield from constants(value)
        elif isinstance(value, (tuple, frozenset)):
            yield from (item for item in value if isinstance(item, (str, bytes)))
        elif isinstance(value, (str, bytes)):
            yield value


def compiled_hits(rel, source, maldet):
    """Reasons a scanner would flag this file's .pyc, or an empty list."""
    try:
        code = compile(source, rel, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        return []
    hits = []
    for value in constants(code):
        if isinstance(value, str):
            value = value.encode("utf-8", "surrogatepass")
        hits += [h for h in find_hits("a.py", value, maldet) if h not in hits]
    return hits


JOINED = b"x" + OPEN + b" system($_GET[0]); ?>"


def folded_source():
    """A .py whose source splits the payload with + of two literals."""
    quote = b'"'
    return b"".join(
        (
            b"x = b",
            quote,
            OPEN,
            b" ",
            quote,
            b" + b",
            quote,
            b"system($_GET[0]); ?>",
            quote,
            b"\n",
        )
    )


def test_the_checker_catches_the_joined_payload():
    assert find_hits("a.py", JOINED, [])
    split = b'x"' + OPEN + b' " + b"system($_GET[0]); ?>"'
    assert not find_hits("a.py", split, [])
    assert find_hits("web/shell.php", b"ok", [])
    assert not find_hits("scripts/shell_check.py", b"ok", [])


def test_the_checker_catches_a_folded_literal():
    folded = folded_source()
    assert not find_hits("a.py", folded, [])
    assert compiled_hits("a.py", folded, [])
    unfolded = (
        folded.replace(b"x = b", b"x = b''.join((b")
        .replace(b" + b", b", b")
        .replace(b"\n", b"))\n")
    )
    assert not compiled_hits("a.py", unfolded, [])


def test_maldet_signatures_load_when_present():
    maldet = load_maldet()
    if not maldet:
        pytest.skip("maldet signatures not readable here")
    assert any(h.startswith("maldet ") for h in find_hits("a.py", JOINED, maldet))


def test_no_tracked_file_matches():
    maldet = load_maldet()
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout.split(b"\0")
    bad = []
    for raw in filter(None, files):
        rel = raw.decode("utf-8", "surrogateescape")
        path = os.path.join(REPO, rel)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            data = handle.read()
        hits = find_hits(rel, data, maldet)
        if rel.endswith(".py"):
            hits += ["compiled " + h for h in compiled_hits(rel, data, maldet)]
        if hits:
            bad.append(f"{rel}: {', '.join(hits)}")
    assert not bad, (
        "these files would be flagged as malware; build the payload at run time "
        'with b"".join, not + of literals:\n' + "\n".join(bad)
    )
