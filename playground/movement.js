// Vahini playground: the movement of a drawing.
//
// A sensor pen measures how it moves. A mouse or a finger on a screen gives
// only positions with times, but the movement can be worked out from them:
// resample the strokes 100 times a second (the OnHW pen's rate), smooth out
// the screen's pixel steps, and take differences for speed, acceleration and
// turning. Everything here comes from the person's own strokes; nothing is
// made up. What a screen cannot give (tilt, tip force, how the pen turns in
// the hand) is not here at all.
//
// Runs in the page (window.PlaygroundMovement) and under Node for
// tests/test_playground_movement.py.
"use strict";

(function (root) {
  const RATE = 100;          // samples a second
  const STEP = 1000 / RATE;  // ms
  const SMOOTH = 5;          // moving average over 50 ms
  const STILL = 20;          // px/s: slower than this, direction means nothing

  // strokes: arrays of {x, y, t}, t in milliseconds.
  function movement(strokes) {
    const st = strokes.filter((s) => s && s.length);
    if (!st.length) return null;
    const t0 = st[0][0].t, t1 = st[st.length - 1][st[st.length - 1].length - 1].t;
    const n = Math.floor((t1 - t0) / STEP) + 1;
    if (n < 3) return null;
    const x = new Array(n), y = new Array(n), down = new Array(n);
    let k = 0;
    for (let i = 0; i < n; i++) {
      const ts = t0 + i * STEP;
      while (k < st.length - 1 && ts > st[k][st[k].length - 1].t) k++;
      const s = st[k];
      if (ts < s[0].t && k > 0) {
        // Between strokes the pen is up and the page sees nothing: the hand
        // is placed on a straight line from where it lifted to where it lands.
        const a = st[k - 1][st[k - 1].length - 1], b = s[0];
        const f = (ts - a.t) / Math.max(b.t - a.t, 1);
        x[i] = a.x + (b.x - a.x) * f; y[i] = a.y + (b.y - a.y) * f; down[i] = 0;
        continue;
      }
      let j = 1;
      while (j < s.length - 1 && s[j].t < ts) j++;
      const a = s[Math.max(0, j - 1)], b = s[j] || a;
      const f = b.t > a.t ? Math.min(1, Math.max(0, (ts - a.t) / (b.t - a.t))) : 0;
      x[i] = a.x + (b.x - a.x) * f; y[i] = a.y + (b.y - a.y) * f; down[i] = 1;
    }
    // Smoothing and differences run inside each stretch of pen-down or
    // pen-up time, so a lift or a landing is not read as a jolt.
    const vx = new Array(n).fill(0), vy = new Array(n).fill(0);
    const acc = new Array(n).fill(0), turn = new Array(n).fill(0);
    for (let i = 0; i < n;) {
      let j = i;
      while (j + 1 < n && down[j + 1] === down[i]) j++;
      if (down[i]) {
        const sx = smooth(x.slice(i, j + 1)), sy = smooth(y.slice(i, j + 1));
        const dx = diff(sx), dy = diff(sy), ax = diff(dx), ay = diff(dy);
        for (let k = 0; k < dx.length; k++) {
          vx[i + k] = dx[k]; vy[i + k] = dy[k]; acc[i + k] = Math.hypot(ax[k], ay[k]);
          if (!k || Math.hypot(dx[k], dy[k]) < STILL || Math.hypot(dx[k - 1], dy[k - 1]) < STILL) continue;
          let d = Math.atan2(dy[k], dx[k]) - Math.atan2(dy[k - 1], dx[k - 1]);
          if (d > Math.PI) d -= 2 * Math.PI;
          if (d < -Math.PI) d += 2 * Math.PI;
          turn[i + k] = (d * 180) / Math.PI * RATE;
        }
      }
      i = j + 1;
    }
    return { rate: RATE, seconds: (n - 1) / RATE, vx, vy, acc, turn, down };
  }
  function smooth(v) {
    // The window narrows near the ends of a stroke and stays centred, so
    // a stroke's start and end are not pulled inwards.
    return v.map((_, i) => {
      const h = Math.min(Math.floor(SMOOTH / 2), i, v.length - 1 - i);
      let s = 0, c = 0;
      for (let j = i - h; j <= i + h; j++) { s += v[j]; c++; }
      return s / c;
    });
  }
  // Central differences, per second.
  function diff(v) {
    return v.map((_, i) => {
      const a = v[Math.max(0, i - 1)], b = v[Math.min(v.length - 1, i + 1)];
      const span = Math.min(v.length - 1, i + 1) - Math.max(0, i - 1);
      return span ? ((b - a) / span) * RATE : 0;
    });
  }

  // The size a chart scales a channel to: the 98th percentile of its
  // magnitude, so one sharp corner does not flatten the rest.
  function scaleOf(v) {
    const a = v.map(Math.abs).filter((x) => x > 0).sort((p, q) => p - q);
    return a.length ? Math.max(a[Math.min(a.length - 1, Math.floor(a.length * 0.98))], 1e-6) : 1;
  }

  // The channels the page draws, with their units.
  const CHANNELS = [
    { key: "vx", name: "Speed across", unit: "px/s", signed: true, colour: "chart-1" },
    { key: "vy", name: "Speed down", unit: "px/s", signed: true, colour: "chart-2" },
    { key: "acc", name: "Acceleration", unit: "px/s²", signed: false, colour: "chart-3" },
    { key: "turn", name: "Turning", unit: "°/s", signed: true, colour: "chart-1" },
    { key: "down", name: "Pen down", unit: "", signed: false, colour: "warning" },
  ];

  const api = { movement, scaleOf, CHANNELS, RATE };
  root.PlaygroundMovement = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
