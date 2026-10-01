"""The movement of a drawing in playground/movement.js, run under Node.

The page shows these signals under the pad and in card 2 as the movement of
the person's own hand. They are worked out from screen positions and times,
so each test draws a shape whose speed, acceleration and turning are known
and checks the numbers that come out.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

MOVEMENT_JS = Path(__file__).resolve().parent.parent / "playground" / "movement.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")

RUNNER = """
const M = require(process.argv[1]);
let input = "";
process.stdin.on("data", (d) => { input += d; });
process.stdin.on("end", () => {
  const m = M.movement(JSON.parse(input));
  if (m) m.scale = Object.fromEntries(M.CHANNELS.map((c) => [c.key, M.scaleOf(m[c.key])]));
  process.stdout.write(JSON.stringify(m));
});
"""


def movement(strokes):
    """Run movement() on strokes of {x, y, t} and return its signals."""
    done = subprocess.run(
        [NODE, "-e", RUNNER, str(MOVEMENT_JS)],
        input=json.dumps(strokes),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


def line(x0, y0, x1, y1, ms, t0=0, step=10):
    """A straight stroke at constant speed, one point every `step` ms."""
    n = ms // step
    return [
        {"x": x0 + (x1 - x0) * i / n, "y": y0 + (y1 - y0) * i / n, "t": t0 + i * step}
        for i in range(n + 1)
    ]


def circle(radius, ms, t0=0, step=10):
    """One turn of a circle at constant speed."""
    n = ms // step
    return [
        {
            "x": radius * math.cos(2 * math.pi * i / n),
            "y": radius * math.sin(2 * math.pi * i / n),
            "t": t0 + i * step,
        }
        for i in range(n + 1)
    ]


def middle(values):
    """The middle half of a signal, away from the ends of a stroke."""
    q = len(values) // 4
    return values[q:-q]


def test_straight_line_has_constant_speed_and_no_turning():
    m = movement([line(0, 0, 100, 0, 1000)])
    assert m["rate"] == 100
    assert m["seconds"] == pytest.approx(1.0)
    assert all(v == pytest.approx(100, abs=0.5) for v in m["vx"])
    assert all(abs(v) < 0.5 for v in m["vy"])
    assert all(abs(v) < 1 for v in m["turn"])
    assert all(a < 1 for a in m["acc"])


def test_screen_down_is_positive():
    m = movement([line(0, 0, 0, 50, 500)])
    assert all(v == pytest.approx(100, abs=0.5) for v in m["vy"])


def test_circle_turns_at_its_rate_and_accelerates_inwards():
    # 50 px radius once round in 2 s: speed 2 pi 50 / 2 = 157 px/s, turning
    # 180 degrees a second, centripetal acceleration v^2 / r = 493 px/s^2.
    m = movement([circle(50, 2000)])
    speed = [math.hypot(a, b) for a, b in zip(m["vx"], m["vy"])]
    assert all(s == pytest.approx(157.1, rel=0.01) for s in middle(speed))
    assert all(t == pytest.approx(180, rel=0.01) for t in middle(m["turn"]))
    assert all(a == pytest.approx(493.5, rel=0.02) for a in middle(m["acc"]))


def test_stroke_ends_are_not_read_as_a_jolt():
    # The smoothing window narrows at a stroke's ends instead of averaging
    # in fewer points on one side, which would pull the end inwards and show
    # up as a burst of acceleration on every stroke.
    m = movement([circle(50, 2000)])
    assert max(m["acc"]) < 500
    assert m["scale"]["acc"] == pytest.approx(493.5, rel=0.02)


def test_pen_up_time_is_marked_and_still():
    first = line(0, 0, 100, 0, 500)
    second = line(0, 40, 100, 40, 500, t0=1000)
    m = movement([first, second])
    up = [i for i, d in enumerate(m["down"]) if d == 0]
    assert up, "the half-second gap between strokes is pen-up time"
    assert up[0] == pytest.approx(51, abs=1) and up[-1] == pytest.approx(99, abs=1)
    for key in ("vx", "vy", "acc", "turn"):
        assert all(m[key][i] == 0 for i in up)
    # The jump back to the left edge between strokes is not a movement of
    # the pen on the page.
    assert max(m["acc"]) < 1
    # 100 px in half a second.
    assert all(
        v == pytest.approx(200, abs=0.5) for i, v in enumerate(m["vx"]) if m["down"][i]
    )


def test_slow_wobble_does_not_count_as_turning():
    # Direction means nothing below 20 px/s: a resting hand's pixel jitter
    # would otherwise read as fast turning.
    still = [{"x": (i % 2) * 0.5, "y": 0, "t": i * 10} for i in range(60)]
    m = movement([still])
    assert all(t == 0 for t in m["turn"])


def test_too_short_or_empty_drawings_give_nothing():
    assert movement([]) is None
    assert movement([[{"x": 0, "y": 0, "t": 0}, {"x": 1, "y": 0, "t": 5}]]) is None
