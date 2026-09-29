"""The "Draw it" shape matcher in playground/shapes.js, run under Node.

Each case is a drawing as a person makes it: pixel coordinates on the
page's drawing area, several strokes where the character needs them, and
seeded jitter, slant and stretch, so the drawings differ from the templates
the matcher compares against. Bugs reported on the published page:

* a minus sign was read as U (the matcher had no straight-line rule);
* a 2 was read as Z (the 2 template's arc was 18 degrees long);
* a ÷ could not be drawn: lifting the pen for the dots ended the character.
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
}
LETTERS = set("OCSUVWMNLZeiTX")


def draw(sketch, rng, height=120, jitter=1.2, slant=8, stretch=0.15):
    """Place a unit sketch on the pad as a person would: sized, slanted,
    stretched, jittered, with mouse events at uneven spacing."""
    sx = height * (1 + rng.uniform(-stretch, stretch))
    shear = math.tan(math.radians(rng.uniform(-slant, slant)))
    ox, oy = 150 + rng.uniform(-20, 20), (PAD_H - height) / 2
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
    want = {"÷": ":"}.get(shape, shape)
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
