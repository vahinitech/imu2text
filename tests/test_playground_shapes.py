"""The "Draw it" shape matcher in playground/shapes.js, run under Node.

Each case is a drawing as a person makes it: pixel coordinates on the
page's drawing area, several strokes where the character needs them, and
seeded jitter, slant and stretch, so the drawings differ from the templates
the matcher compares against. Bugs reported on the published page:

* a minus sign was read as U (the matcher had no straight-line rule);
* a 2 was read as Z (the 2 template's arc was 18 degrees long);
* a ÷ could not be drawn: lifting the pen for the dots ended the character;
* a small h was read as V: the matcher knew no h, and accepted any template
  scoring 0.8, which an unknown letter's nearest template usually does;
* a t in two strokes (a hooked stem, then a bar) was "not sure": two-stroke
  shapes were only straight crossings and the digits 4 and 5;
* 11 was read as 4 and 12 as "not sure": everything on the pad was taken as
  one character. Small a, b and n drawn on the numbers task were read as 9,
  5 and 8 by the matcher that had no small letters.
"""

import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

SHAPES_JS = Path(__file__).resolve().parent.parent / "playground" / "shapes.js"
NODE = shutil.which("node")
PAD_H = 170  # the drawing area's height on the page, in CSS pixels

pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")

RUNNER = """
const S = require(process.argv[1]);
let input = "";
process.stdin.on("data", (d) => { input += d; });
process.stdin.on("end", () => {
  const cases = JSON.parse(input);
  const out = cases.map((c) => S.recognise(c.strokes, { size: c.size, task: c.task }));
  process.stdout.write(JSON.stringify(out));
});
"""


def recognise(cases):
    """Run the matcher on [{strokes, task}] and return its answers."""
    payload = [{"size": PAD_H, **c} for c in cases]
    done = subprocess.run(
        [NODE, "-e", RUNNER, str(SHAPES_JS)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


# ---------- drawings ----------
def line(a, b):
    """Points from a to b, one every unit (resampled later)."""
    n = max(2, int(math.dist(a, b) * 40))
    return [
        (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
        for k in range(n + 1)
    ]


def path(*pts):
    """A polyline through the given points."""
    out = []
    for a, b in zip(pts, pts[1:]):
        out += line(a, b)[:-1]
    return out + [pts[-1]]


def curve(cx, cy, rx, ry, start, end):
    """An arc in degrees; 0 points right, 90 points down (screen)."""
    n = max(8, int(abs(end - start) / 4))
    return [
        (
            cx + rx * math.cos(math.radians(start + (end - start) * k / n)),
            cy + ry * math.sin(math.radians(start + (end - start) * k / n)),
        )
        for k in range(n + 1)
    ]


def dot(x, y):
    """A click: one point."""
    return [(x, y)]


# Unit-square sketches of how people write each character, stroke by stroke.
SKETCHES = {
    "-": [path((0.05, 0.5), (0.95, 0.47))],
    "=": [path((0.1, 0.35), (0.9, 0.35)), path((0.1, 0.65), (0.9, 0.66))],
    "+": [path((0.1, 0.5), (0.9, 0.5)), path((0.5, 0.1), (0.5, 0.9))],
    "·": [dot(0.5, 0.5)],
    ":": [dot(0.5, 0.3), dot(0.5, 0.7)],
    "÷": [path((0.1, 0.5), (0.9, 0.5)), dot(0.5, 0.2), dot(0.5, 0.8)],
    "1": [path((0.3, 0.25), (0.55, 0.05), (0.55, 0.95))],
    "2": [
        curve(0.5, 0.3, 0.32, 0.25, 200, 390)
        + path((0.78, 0.4), (0.12, 0.93), (0.9, 0.93))
    ],
    "3": [
        curve(0.5, 0.28, 0.3, 0.22, 205, 450) + curve(0.5, 0.72, 0.32, 0.22, -90, 150)
    ],
    "4": [
        path((0.55, 0.05), (0.12, 0.62), (0.88, 0.62)),
        path((0.65, 0.3), (0.65, 0.95)),
    ],
    "5": [
        path((0.3, 0.08), (0.27, 0.45)) + curve(0.5, 0.66, 0.3, 0.25, -140, 150),
        path((0.3, 0.08), (0.8, 0.08)),
    ],
    "6": [
        path((0.72, 0.05), (0.45, 0.2), (0.28, 0.5))
        + curve(0.52, 0.7, 0.24, 0.23, 180, -180)
    ],
    "7": [path((0.1, 0.1), (0.88, 0.1), (0.4, 0.95))],
    "8": [
        curve(0.5, 0.26, 0.22, 0.2, -40, -270)
        + curve(0.5, 0.71, 0.26, 0.23, -90, 270)
        + curve(0.5, 0.26, 0.22, 0.2, 90, -40)
    ],
    "9": [curve(0.45, 0.28, 0.26, 0.23, 0, -360) + path((0.71, 0.28), (0.67, 0.95))],
    "0": [curve(0.5, 0.5, 0.3, 0.45, -90, 275)],
    "O": [curve(0.5, 0.5, 0.42, 0.42, -100, 262)],
    "C": [curve(0.5, 0.5, 0.4, 0.42, -40, -300)],
    "S": [
        curve(0.5, 0.28, 0.22, 0.22, -30, -270) + curve(0.5, 0.72, 0.22, 0.22, -90, 150)
    ],
    "U": [
        path((0.12, 0.05), (0.12, 0.6))
        + curve(0.5, 0.6, 0.38, 0.33, 180, 0)
        + path((0.88, 0.6), (0.88, 0.05))
    ],
    "V": [path((0.08, 0.05), (0.5, 0.95), (0.92, 0.05))],
    "W": [path((0.03, 0.05), (0.25, 0.95), (0.5, 0.4), (0.75, 0.95), (0.97, 0.05))],
    "M": [path((0.05, 0.95), (0.08, 0.05), (0.5, 0.6), (0.92, 0.05), (0.95, 0.95))],
    "N": [path((0.1, 0.95), (0.1, 0.05), (0.9, 0.95), (0.9, 0.05))],
    "L": [path((0.2, 0.05), (0.2, 0.9), (0.85, 0.9))],
    "Z": [path((0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))],
    "e": [path((0.15, 0.55), (0.82, 0.52)) + curve(0.5, 0.55, 0.34, 0.38, 0, -300)],
    "i": [path((0.5, 0.35), (0.5, 0.95)), dot(0.5, 0.1)],
    "T": [path((0.1, 0.08), (0.9, 0.08)), path((0.5, 0.08), (0.5, 0.95))],
    "X": [path((0.1, 0.1), (0.9, 0.9)), path((0.9, 0.1), (0.1, 0.9))],
    # Small letters in one stroke; the stems of h n m r b p go back up
    # before the arch, as a pen does.
    "h": [
        path((0.22, 0.03), (0.2, 0.97), (0.21, 0.62))
        + curve(0.5, 0.66, 0.29, 0.19, 180, 360)
        + path((0.79, 0.66), (0.8, 0.97))
    ],
    "n": [
        path((0.17, 0.25), (0.15, 0.95), (0.16, 0.52))
        + curve(0.5, 0.52, 0.34, 0.28, 185, 360)
        + path((0.84, 0.52), (0.86, 0.95))
    ],
    "m": [
        path((0.1, 0.3), (0.09, 0.95), (0.1, 0.52))
        + curve(0.3, 0.52, 0.2, 0.22, 180, 360)
        + path((0.5, 0.52), (0.5, 0.93), (0.5, 0.53))
        + curve(0.7, 0.52, 0.2, 0.22, 180, 360)
        + path((0.9, 0.52), (0.91, 0.95))
    ],
    "r": [
        path((0.27, 0.25), (0.25, 0.97), (0.26, 0.6))
        + curve(0.56, 0.58, 0.3, 0.28, 180, 305)
    ],
    "b": [
        path((0.22, 0.02), (0.2, 0.95), (0.21, 0.65))
        + curve(0.5, 0.71, 0.29, 0.24, 180, 540)
    ],
    "p": [
        path((0.22, 0.3), (0.2, 1.0), (0.21, 0.38))
        + curve(0.5, 0.46, 0.29, 0.2, 180, 540)
    ],
    "a": [curve(0.43, 0.62, 0.3, 0.32, -40, -400) + path((0.7, 0.32), (0.73, 0.97))],
    "d": [
        curve(0.43, 0.66, 0.3, 0.28, -40, -400)
        + path((0.7, 0.45), (0.74, 0.02), (0.75, 0.97))
    ],
    "q": [curve(0.43, 0.35, 0.3, 0.28, -40, -400) + path((0.7, 0.15), (0.73, 1.0))],
    "g": [
        curve(0.43, 0.3, 0.3, 0.24, -40, -400)
        + path((0.7, 0.12), (0.73, 0.78))
        + curve(0.45, 0.78, 0.28, 0.19, 0, 170)
    ],
}
# Letters in several strokes, as written: stem first, then bars.
MULTI = {
    "t": [
        path((0.45, 0.02), (0.46, 0.8)) + curve(0.63, 0.8, 0.17, 0.15, 180, 45),
        path((0.12, 0.33), (0.82, 0.3)),
    ],
    "f": [
        curve(0.56, 0.18, 0.2, 0.15, -30, -180) + path((0.36, 0.18), (0.35, 0.98)),
        path((0.1, 0.43), (0.66, 0.41)),
    ],
    "A": [path((0.1, 0.98), (0.5, 0.02), (0.9, 0.98)), path((0.28, 0.62), (0.72, 0.6))],
    "K": [
        path((0.2, 0.02), (0.2, 0.98)),
        path((0.8, 0.02), (0.24, 0.55), (0.82, 0.98)),
    ],
    "Y": [path((0.1, 0.02), (0.5, 0.5), (0.9, 0.02)), path((0.5, 0.5), (0.5, 0.98))],
    "H": [
        path((0.15, 0.02), (0.15, 0.98)),
        path((0.85, 0.02), (0.85, 0.98)),
        path((0.15, 0.5), (0.85, 0.5)),
    ],
    "F": [
        path((0.2, 0.02), (0.2, 0.98)),
        path((0.2, 0.02), (0.85, 0.02)),
        path((0.2, 0.48), (0.7, 0.48)),
    ],
    "E": [
        path((0.2, 0.02), (0.2, 0.98)),
        path((0.2, 0.02), (0.85, 0.02)),
        path((0.2, 0.5), (0.7, 0.5)),
        path((0.2, 0.98), (0.85, 0.98)),
    ],
}
UNKNOWN_MULTI = {
    "#": [
        path((0.35, 0.05), (0.3, 0.95)),
        path((0.7, 0.05), (0.65, 0.95)),
        path((0.1, 0.35), (0.9, 0.35)),
        path((0.1, 0.65), (0.9, 0.65)),
    ],
    "pi": [
        path((0.1, 0.2), (0.9, 0.2)),
        path((0.3, 0.2), (0.3, 0.95)),
        path((0.7, 0.2), (0.72, 0.95)),
    ],
    # A circle crossed by a line (Φ): one shape, not a letter here.
    "phi": [curve(0.5, 0.5, 0.3, 0.3, 0, 360), path((0.5, 0.02), (0.5, 0.98))],
    "two scribbles": [
        path((0.1, 0.1), (0.4, 0.8), (0.2, 0.5), (0.5, 0.2)),
        path((0.6, 0.9), (0.9, 0.3), (0.7, 0.6)),
    ],
}
LETTERS = set("OCSUVWMNLZeiTXhnmrbpadqg")

# Shapes that are not a letter the matcher reads: an f or t without its bar,
# a y as one zigzag, &, a spiral, a zigzag. Each must come back "not sure",
# never as the nearest template.
UNKNOWN = {
    "f": [curve(0.6, 0.2, 0.2, 0.18, -20, -180) + path((0.4, 0.2), (0.4, 1.0))],
    "y": [
        path((0.1, 0.05), (0.45, 0.55)) + path((0.45, 0.55), (0.85, 0.05), (0.3, 1.0))
    ],
    "t": [path((0.5, 0.0), (0.5, 0.85)) + curve(0.65, 0.85, 0.15, 0.12, 180, 90)],
    "&": [
        path((0.85, 0.95), (0.3, 0.3))
        + curve(0.4, 0.25, 0.15, 0.15, 180, 360)
        + path((0.55, 0.25), (0.15, 0.7))
        + curve(0.4, 0.75, 0.25, 0.2, 180, 90)
        + path((0.4, 0.95), (0.85, 0.5))
    ],
    "spiral": [
        [
            (0.5 + 0.4 * t / 40 * math.cos(t / 4), 0.5 + 0.4 * t / 40 * math.sin(t / 4))
            for t in range(40)
        ]
    ],
    "zigzag": [
        path(
            (0.05, 0.5),
            (0.2, 0.2),
            (0.35, 0.8),
            (0.5, 0.2),
            (0.65, 0.8),
            (0.8, 0.2),
            (0.95, 0.8),
        )
    ],
}


def draw(sketch, rng, height=120, jitter=1.2, slant=8, stretch=0.15, left=None):
    """Place a unit sketch on the pad as a person would: sized, slanted,
    stretched, jittered, with mouse events at uneven spacing."""
    sx = height * (1 + rng.uniform(-stretch, stretch))
    shear = math.tan(math.radians(rng.uniform(-slant, slant)))
    ox = 150 + rng.uniform(-20, 20) if left is None else left
    oy = (PAD_H - height) / 2
    strokes = []
    for stroke in sketch:
        pts, k = [], 0
        while k < len(stroke):
            x, y = stroke[k]
            px = ox + x * sx + (0.5 - y) * height * shear + rng.gauss(0, jitter)
            py = oy + y * height + rng.gauss(0, jitter)
            pts.append({"x": round(px, 1), "y": round(py, 1)})
            k += rng.randint(1, 3)
        if len(stroke) > 1 and pts[-1] != stroke[-1]:
            x, y = stroke[-1]
            pts.append(
                {"x": ox + x * sx + (0.5 - y) * height * shear, "y": oy + y * height}
            )
        strokes.append(pts)
    return strokes


def cases_for(shape, n=12, **kw):
    """n seeded drawings of one shape, read in the task that has it."""
    rng = random.Random(shape)
    task = "chars" if shape in LETTERS else "symbols"
    return [
        {"strokes": draw(SKETCHES[shape], rng, **kw), "task": task} for _ in range(n)
    ]


# ---------- the reported bugs ----------
def test_a_minus_sign_is_a_minus_not_u():
    # The stroke from the bug report: a short, slightly rising line.
    reported = [
        [{"x": 208 + k * 3, "y": 400 - k * 0.15 + (k % 3) * 0.4} for k in range(24)]
    ]
    got = recognise(
        [
            {"strokes": reported, "task": "chars"},
            {"strokes": reported, "task": "symbols"},
        ]
        + cases_for("-")
    )
    assert {g["label"] for g in got} == {"-"}
    assert {g["task"] for g in got} == {"symbols"}


def test_a_two_is_a_two_not_z():
    got = recognise(cases_for("2"))
    assert [g["label"] for g in got] == ["2"] * 12
    # Drawn on the Letters task, a round-topped 2 still reads as 2.
    got = recognise([dict(c, task="chars") for c in cases_for("2")])
    assert {g["label"] for g in got} == {"2"}


def test_a_z_with_sharp_corners_stays_z_on_letters():
    assert {g["label"] for g in recognise(cases_for("Z"))} == {"Z"}


def test_division_is_drawn_in_three_parts_and_read_as_the_colon_sign():
    got = recognise(cases_for("÷"))
    assert {(g["shape"], g["label"]) for g in got} == {("÷", ":")}
    # The dots can come first.
    rng = random.Random(1)
    sketch = SKETCHES["÷"]
    got = recognise(
        [{"strokes": draw([sketch[1], sketch[2], sketch[0]], rng), "task": "symbols"}]
    )
    assert got[0]["label"] == ":"


# ---------- every shape the page says it knows ----------
@pytest.mark.parametrize("shape", sorted(SKETCHES))
def test_every_known_shape_is_read(shape):
    got = recognise(cases_for(shape))
    labels = [g["label"] for g in got]
    want = {"÷": ":", "p": "P"}.get(shape, shape)  # a large p is a P
    assert labels.count(want) >= 11, labels


@pytest.mark.parametrize("shape", ["C", "O", "S", "U", "V", "W", "Z"])
def test_small_same_shape_letters_are_read_small(shape):
    got = recognise(cases_for(shape, height=50))
    assert {g["label"] for g in got} == {shape.lower()}


def test_multi_part_symbols_do_not_depend_on_stroke_order():
    rng = random.Random(7)
    for shape in ("+", "=", ":"):
        got = recognise(
            [{"strokes": draw(SKETCHES[shape][::-1], rng), "task": "symbols"}]
        )
        assert got[0]["label"] == shape


def test_the_reported_small_h_is_h_not_v():
    # Pixels of the stroke in the report: a stem leaning about 23 degrees,
    # back up the stem, a narrow arch, and a right leg ending above the foot.
    pts = [(85 + 42 * k / 30, 252 + 98 * k / 30) for k in range(30)]
    pts += [(127 - 6 * k / 12, 350 - 45 * k / 12) for k in range(12)]
    for k in range(16):
        t = k / 16
        pts.append(
            (
                (1 - t) ** 2 * 121 + 2 * t * (1 - t) * 136 + t * t * 152,
                (1 - t) ** 2 * 305 + 2 * t * (1 - t) * 270 + t * t * 300,
            )
        )
    pts += [(152 + 8 * k / 10, 300 + 38 * k / 10) for k in range(11)]
    stroke = [{"x": x - 30, "y": y - 183} for x, y in pts]
    got = recognise([{"strokes": [stroke], "task": "chars", "size": 215}])[0]
    assert got["label"] == "h"


@pytest.mark.parametrize("shape", ["h", "n", "b", "d"])
def test_slanted_handwriting_is_straightened(shape):
    got = recognise(cases_for(shape, slant=24))
    assert [g["label"] for g in got].count(shape) >= 11


@pytest.mark.parametrize("name", sorted(UNKNOWN))
def test_unknown_letters_are_refused_not_guessed(name):
    rng = random.Random(name)
    got = recognise(
        [{"strokes": draw(UNKNOWN[name], rng), "task": "chars"} for _ in range(12)]
    )
    assert {g["label"] for g in got} == {None}, [g.get("label") for g in got]


def test_a_letter_that_shares_a_digit_shape_follows_the_task():
    q = [dict(c, task="chars") for c in cases_for("q")]
    nine = [dict(c, task="symbols") for c in cases_for("9")]
    assert [g["label"] for g in recognise(q)].count("q") >= 11
    assert [g["label"] for g in recognise(nine)].count("9") >= 11


def test_the_reported_two_stroke_t_is_t_in_either_order():
    # Screenshot pixels: a stem with a hook to the right at the foot, and a
    # wide bar about half way down; the pad starts at about (30, 183).
    stem = [(174, 241 + 94 * k / 20) for k in range(21)]
    stem += [(180, 335), (186, 346), (193, 350), (203, 345), (216, 333)]
    bar = [(115 + 145 * k / 20, 298 - 9 * k / 20) for k in range(21)]
    strokes = [[{"x": x - 30, "y": y - 183} for x, y in st] for st in (stem, bar)]
    got = recognise(
        [
            {"strokes": strokes, "task": "chars", "size": 215},
            {"strokes": strokes[::-1], "task": "chars", "size": 215},
        ]
    )
    assert [g["label"] for g in got] == ["t", "t"]


@pytest.mark.parametrize("shape", sorted(MULTI))
def test_multi_stroke_letters_in_any_order_and_direction(shape):
    rng = random.Random(shape)
    sketch = MULTI[shape]
    # As written, in reverse order, and with every other stroke drawn backwards.
    variants = [
        sketch,
        sketch[::-1],
        [st[::-1] if i % 2 else st for i, st in enumerate(sketch)],
    ]
    cases = [
        {"strokes": draw(v, rng), "task": "chars"} for v in variants for _ in range(6)
    ]
    labels = [g["label"] for g in recognise(cases)]
    assert labels.count(shape) >= 17, labels


@pytest.mark.parametrize("name", sorted(UNKNOWN_MULTI))
def test_unknown_multi_stroke_shapes_are_refused(name):
    rng = random.Random(name)
    got = recognise(
        [
            {"strokes": draw(UNKNOWN_MULTI[name], rng), "task": "chars"}
            for _ in range(12)
        ]
    )
    assert {g["label"] for g in got} == {None}


def test_a_plain_cross_is_t_for_letters_and_plus_for_numbers():
    rng = random.Random(5)
    high_bar = [path((0.5, 0.02), (0.5, 0.98)), path((0.15, 0.3), (0.85, 0.3))]
    cases = [
        {"strokes": draw(high_bar, rng), "task": "chars"},
        {"strokes": draw(high_bar, rng), "task": "symbols"},
        {"strokes": draw(SKETCHES["+"], rng), "task": "chars"},
    ]
    assert [g["label"] for g in recognise(cases)] == ["t", "+", "+"]


REPORTED = Path(__file__).resolve().parent / "fixtures" / "playground_reported.json"


def _reported(name):
    """Strokes rebuilt from the report's screenshots, pad pixels."""
    return json.loads(REPORTED.read_text(encoding="utf-8"))[name]


@pytest.mark.parametrize("name", ["n", "a", "b"])
def test_the_reported_small_letters_on_the_numbers_task(name):
    got = recognise(
        [
            {"strokes": _reported(name), "task": "symbols", "size": 215},
            {"strokes": _reported(name), "task": "chars", "size": 215},
        ]
    )
    assert [g["label"] for g in got] == [name, name]
    assert {g["task"] for g in got} == {"chars"}


def test_the_reported_11_and_12_are_two_characters():
    got = recognise(
        [
            {"strokes": _reported("11"), "task": "symbols", "size": 215},
            {"strokes": _reported("12"), "task": "symbols", "size": 215},
        ]
    )
    assert [[c["label"] for c in g["sequence"]] for g in got] == [
        ["1", "1"],
        ["1", "2"],
    ]


def _pair(first, second, rng, gap=18):
    """Two characters side by side, each 90 px high."""
    a = draw(SKETCHES.get(first) or MULTI[first], rng, height=90, left=40)
    right = max(p["x"] for st in a for p in st) + gap
    b = draw(SKETCHES.get(second) or MULTI[second], rng, height=90, left=right)
    return a + b


@pytest.mark.parametrize("pair", ["12", "17", "23", "40", "58", "69", "93", "71"])
def test_two_digits_side_by_side(pair):
    rng = random.Random(pair)
    cases = [
        {"strokes": _pair(pair[0], pair[1], rng), "task": "symbols"} for _ in range(8)
    ]
    got = recognise(cases)
    read = ["".join(c["label"] or "?" for c in g.get("sequence", [g])) for g in got]
    assert read.count(pair) >= 7, read


@pytest.mark.parametrize("pair", ["ab", "hn", "Ce", "tO"])
def test_two_letters_side_by_side(pair):
    rng = random.Random(pair)
    cases = [
        {"strokes": _pair(pair[0], pair[1], rng), "task": "chars"} for _ in range(8)
    ]
    got = recognise(cases)
    read = ["".join(c["label"] or "?" for c in g.get("sequence", [g])) for g in got]
    assert read.count(pair) >= 7, read


def test_the_reported_12_on_the_letters_task_reads_1_2():
    # Reported: drawn on the Letters task, the 1 was read as the letter l.
    got = recognise([{"strokes": _reported("12"), "task": "chars", "size": 215}])
    assert [c["label"] for c in got[0]["sequence"]] == ["1", "2"]


@pytest.mark.parametrize(
    "first,second,task,want",
    [
        # A clear digit makes its neighbour a digit: O is 0 next to a 2.
        ("2", "O", "chars", "20"),
        # A clear letter makes a vertical line the letter l, even on numbers.
        ("b", "l", "symbols", "bl"),
    ],
)
def test_characters_written_together_are_read_together(first, second, task, want):
    rng = random.Random(want + task)
    line = {"l": [path((0.5, 0.02), (0.52, 0.98))]}
    sketches = {**SKETCHES, **MULTI, **line}
    cases = []
    for _ in range(8):
        a = draw(sketches[first], rng, height=90, left=40)
        right = max(p["x"] for st in a for p in st) + 18
        cases.append(
            {
                "strokes": a + draw(sketches[second], rng, height=90, left=right),
                "task": task,
            }
        )
    read = ["".join(c["label"] or "?" for c in g["sequence"]) for g in recognise(cases)]
    assert read.count(want) >= 7, read


def test_one_character_with_a_gap_between_its_strokes_stays_one():
    # An H whose bar stops short of both stems, and a = and a ÷.
    rng = random.Random(9)
    loose_h = [
        path((0.15, 0.02), (0.15, 0.98)),
        path((0.85, 0.02), (0.85, 0.98)),
        path((0.19, 0.5), (0.81, 0.5)),
    ]
    cases = [
        {"strokes": draw(loose_h, rng), "task": "chars"},
        {"strokes": draw(SKETCHES["="], rng), "task": "symbols"},
        {"strokes": draw(SKETCHES["÷"], rng), "task": "symbols"},
    ]
    assert [g.get("label") for g in recognise(cases)] == ["H", "=", ":"]


@pytest.mark.parametrize(
    "shape,sketch",
    [
        # The same bowl; only the stem differs.
        (
            "a",
            [
                curve(0.43, 0.62, 0.3, 0.32, -40, -400)
                + path((0.7, 0.32), (0.73, 0.95), (0.85, 0.88))
            ],
        ),
        (
            "d",
            [
                curve(0.43, 0.66, 0.3, 0.28, -40, -400)
                + path((0.7, 0.45), (0.74, 0.0), (0.75, 0.97))
            ],
        ),
        ("q", [curve(0.43, 0.3, 0.3, 0.26, -40, -400) + path((0.7, 0.1), (0.73, 1.0))]),
        (
            "g",
            [
                curve(0.43, 0.28, 0.3, 0.24, -40, -400)
                + path((0.7, 0.1), (0.73, 0.8))
                + curve(0.45, 0.8, 0.28, 0.2, 0, 170)
            ],
        ),
    ],
)
def test_bowl_letters_are_told_apart_by_the_stem(shape, sketch):
    got = recognise(cases_for_sketch(sketch, shape))
    assert [g["label"] for g in got].count(shape) >= 11


def cases_for_sketch(sketch, seed):
    rng = random.Random(seed)
    return [{"strokes": draw(sketch, rng), "task": "chars"} for _ in range(12)]


# Every one of the 52 letters the model reads, as a person would draw it:
# (sketch, height in pixels). Small and capital letters of the same shape
# are drawn small for the small letter.
ALPHABET = {
    "B": (
        [
            path((0.22, 0.02), (0.2, 0.98), (0.21, 0.03))
            + curve(0.21, 0.26, 0.48, 0.24, -90, 90)
            + curve(0.21, 0.74, 0.58, 0.25, -90, 90)
        ],
        120,
    ),
    "D": (
        [
            path((0.22, 0.02), (0.2, 0.98), (0.21, 0.03))
            + curve(0.21, 0.5, 0.62, 0.48, -90, 90)
        ],
        120,
    ),
    "G": (
        [
            curve(0.5, 0.5, 0.4, 0.44, -45, -320)
            + path((0.9, 0.6), (0.9, 0.53), (0.55, 0.53))
        ],
        120,
    ),
    "I": (
        [
            path((0.12, 0.02), (0.88, 0.02)),
            path((0.5, 0.02), (0.5, 0.98)),
            path((0.12, 0.98), (0.88, 0.98)),
        ],
        120,
    ),
    "J": ([path((0.72, 0.02), (0.7, 0.7)) + curve(0.46, 0.7, 0.24, 0.27, 0, 175)], 120),
    "P": (
        [
            path((0.27, 0.02), (0.25, 0.98), (0.26, 0.03))
            + curve(0.26, 0.28, 0.48, 0.26, -90, 90)
        ],
        120,
    ),
    "Q": (
        [curve(0.47, 0.45, 0.4, 0.42, -90, 270), path((0.55, 0.68), (0.95, 0.98))],
        120,
    ),
    "R": (
        [
            path((0.27, 0.02), (0.25, 0.98), (0.26, 0.03))
            + curve(0.26, 0.28, 0.48, 0.26, -90, 90)
            + path((0.26, 0.54), (0.82, 0.98))
        ],
        120,
    ),
    "j": (
        [
            path((0.72, 0.1), (0.7, 0.7)) + curve(0.46, 0.7, 0.24, 0.27, 0, 175),
            dot(0.71, -0.1),
        ],
        120,
    ),
    "k": (
        [path((0.2, 0.02), (0.2, 0.98)), path((0.7, 0.42), (0.25, 0.7), (0.78, 0.98))],
        120,
    ),
    "y": ([path((0.12, 0.02), (0.5, 0.5)), path((0.88, 0.02), (0.28, 0.98))], 120),
    "l": ([path((0.5, 0.02), (0.52, 0.98))], 120),
    "i": (SKETCHES["i"], 120),
    **{
        c: ((SKETCHES.get(c) or MULTI.get(c)), 120)
        for c in "ACEFHKLMNOSTUVWXYZabdefghmnqrt"
    },
    **{c: (SKETCHES[c.upper()], 50) for c in "cosuvwxz"},
    "p": (SKETCHES["p"], 50),
}


def test_the_alphabet_covers_every_letter_the_model_reads():
    assert sorted(ALPHABET) == sorted(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )


@pytest.mark.parametrize("letter", sorted(ALPHABET))
def test_every_letter_can_be_drawn(letter):
    sketch, height = ALPHABET[letter]
    rng = random.Random("abc" + letter)
    got = recognise(
        [
            {"strokes": draw(sketch, rng, height=height), "task": "chars"}
            for _ in range(12)
        ]
    )
    labels = [g.get("label") for g in got]
    assert labels.count(letter) >= 10, labels


def test_a_scribble_is_refused_with_the_known_shapes():
    rng = random.Random(3)
    scribble = [
        [
            {"x": 150 + rng.uniform(0, 120), "y": 30 + rng.uniform(0, 110)}
            for _ in range(40)
        ]
    ]
    got = recognise([{"strokes": scribble, "task": "symbols"}])[0]
    assert got["label"] is None
    assert "0 to 9" in got["reason"]
