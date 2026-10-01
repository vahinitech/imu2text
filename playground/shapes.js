// Vahini playground: the shape matcher behind "Draw it" in card 1.
//
// It turns strokes drawn with a mouse or a finger into one character, so the
// page can show what the model answered for a real recording of that
// character. It is not the AI: the pen model reads motion, and a drawing on a
// screen has none.
//
// Two steps. Symbols built from straight lines and dots (- + = · : ÷, and
// i T X) are read from their parts: how many there are, which way each line
// runs and where the dots sit. Everything else is joined into one path and
// compared with templates using the closed-form best-rotation distance of the
// Protractor recogniser (Y. Li, "Protractor: a fast and accurate gesture
// recognizer", CHI 2010).
//
// Drawing code written by Vahini Technologies for vahinitech.com and released
// here under Apache-2.0 with the owner's approval (2026-09-26). It runs in the
// page (window.PlaygroundShapes) and under Node for
// tests/test_playground_shapes.py.
"use strict";

(function (root) {
  const N = 48;
  const TAU = Math.PI * 2;
  const DEG = Math.PI / 180;

  // ---------- geometry ----------
  const dist = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
  function pathLength(pts) {
    let d = 0;
    for (let j = 1; j < pts.length; j++) d += dist(pts[j - 1], pts[j]);
    return d;
  }
  function box(pts) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of pts) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
    return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
  }
  // n points evenly spaced along the path.
  function resample(points, n = N) {
    const pts = points.map((p) => ({ x: p.x, y: p.y }));
    const step = pathLength(pts) / (n - 1);
    const out = [pts[0]];
    if (step === 0) { while (out.length < n) out.push(pts[0]); return out; }
    let D = 0;
    for (let j = 1; j < pts.length; j++) {
      const d = dist(pts[j - 1], pts[j]);
      if (D + d >= step && d > 0) {
        const t = (step - D) / d;
        const q = { x: pts[j - 1].x + t * (pts[j].x - pts[j - 1].x), y: pts[j - 1].y + t * (pts[j].y - pts[j - 1].y) };
        out.push(q); pts.splice(j, 0, q); D = 0;
      } else D += d;
    }
    while (out.length < n) out.push(pts[pts.length - 1]);
    return out.slice(0, n);
  }
  function toVector(points) {
    const c = points.reduce((a, p) => ({ x: a.x + p.x / points.length, y: a.y + p.y / points.length }), { x: 0, y: 0 });
    const v = [];
    let mag = 0;
    points.forEach((p) => { const x = p.x - c.x, y = p.y - c.y; v.push(x, y); mag += x * x + y * y; });
    mag = Math.sqrt(mag) || 1;
    return v.map((x) => x / mag);
  }

  // ---------- Protractor ----------
  // The best match over every rotation is sqrt(dot^2 + cross^2), so no angle
  // has to be searched.
  function anyRotation(a, b) {
    let dot = 0, cross = 0;
    for (let i = 0; i < a.length; i += 2) { dot += a[i] * b[i] + a[i + 1] * b[i + 1]; cross += a[i] * b[i + 1] - a[i + 1] * b[i]; }
    return Math.min(1, Math.sqrt(dot * dot + cross * cross));
  }
  function reversedVec(v) { const out = []; for (let i = v.length - 2; i >= 0; i -= 2) out.push(v[i], v[i + 1]); return out; }
  function shiftedVec(v, k) {
    const n = v.length / 2, out = [];
    for (let i = 0; i < n; i++) { const j = ((i + k) % n + n) % n; out.push(v[j * 2], v[j * 2 + 1]); }
    return out;
  }
  const dotProduct = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  function rotateVec(v, t) {
    const out = [], cos = Math.cos(t), sin = Math.sin(t);
    for (let i = 0; i < v.length; i += 2) out.push(v[i] * cos - v[i + 1] * sin, v[i] * sin + v[i + 1] * cos);
    return out;
  }
  // ±32° of slack for open shapes: a wobbly V still matches, but an upright
  // V does not turn into a 7, which differs only by orientation.
  const SWEEP = [];
  for (let d = -32; d <= 32; d += 4) SWEEP.push(d * DEG);
  function bestMatch(v, t) {
    const rv = reversedVec(v);
    if (t.closed) {
      // A loop can start anywhere and turn either way.
      let best = Math.max(anyRotation(v, t.v), anyRotation(rv, t.v));
      for (let k = 2; k < v.length / 2; k += 2) best = Math.max(best, anyRotation(shiftedVec(v, k), t.v), anyRotation(shiftedVec(rv, k), t.v));
      return best;
    }
    // Open shapes are compared forwards only when drawn in more than one
    // stroke, because the stroke order is part of the template.
    const dirs = t.strokes > 1 ? [v] : [v, rv];
    let best = -1;
    for (const a of SWEEP) for (const d of dirs) best = Math.max(best, dotProduct(rotateVec(d, a), t.v));
    return best;
  }

  // ---------- templates (unit square, y grows downwards) ----------
  const P = (x, y) => ({ x, y });
  function poly(pts) {
    const out = [];
    for (let j = 0; j < pts.length - 1; j++) for (let k = 0; k < 12; k++) out.push(P(pts[j].x + (pts[j + 1].x - pts[j].x) * k / 12, pts[j].y + (pts[j + 1].y - pts[j].y) * k / 12));
    out.push(pts[pts.length - 1]);
    return out;
  }
  // Screen angles: 0 points right, a quarter turn points down.
  function arc(cx, cy, rx, ry, a0, a1) {
    const out = [];
    for (let k = 0; k <= 28; k++) { const a = a0 + (a1 - a0) * k / 28; out.push(P(cx + rx * Math.cos(a), cy + ry * Math.sin(a))); }
    return out;
  }
  const RIGHT = 0, DOWN = TAU / 4, LEFT = TAU / 2, UP = -TAU / 4;
  // [shape, points, strokes, closed]. A shape name with a slash is resolved
  // by the task being read (O for letters, 0 for numbers).
  const TEMPLATES = [
    ["O/0", arc(0.5, 0.5, 0.42, 0.42, UP, UP - TAU), 1, true],
    ["O/0", arc(0.5, 0.5, 0.3, 0.45, UP, UP - TAU), 1, true],
    ["C", arc(0.5, 0.5, 0.42, 0.42, -TAU / 8, -TAU / 8 - TAU * 0.72), 1],
    ["S", arc(0.5, 0.28, 0.22, 0.22, -TAU / 8, UP - TAU / 2).concat(arc(0.5, 0.72, 0.22, 0.22, UP, DOWN + TAU / 8)), 1],
    ["U", poly([P(0.1, 0.05), P(0.1, 0.6)]).concat(arc(0.5, 0.6, 0.4, 0.35, LEFT, 0)).concat(poly([P(0.9, 0.6), P(0.9, 0.05)])), 1],
    ["V", poly([P(0.05, 0.05), P(0.5, 0.95), P(0.95, 0.05)]), 1],
    ["W", poly([P(0.02, 0.05), P(0.27, 0.95), P(0.5, 0.35), P(0.73, 0.95), P(0.98, 0.05)]), 1],
    ["M", poly([P(0.05, 0.95), P(0.05, 0.05), P(0.5, 0.6), P(0.95, 0.05), P(0.95, 0.95)]), 1],
    ["N", poly([P(0.08, 0.95), P(0.08, 0.05), P(0.92, 0.95), P(0.92, 0.05)]), 1],
    ["L", poly([P(0.15, 0.05), P(0.15, 0.9), P(0.85, 0.9)]), 1],
    ["Z", poly([P(0.08, 0.08), P(0.92, 0.08), P(0.08, 0.92), P(0.92, 0.92)]), 1],
    ["e", poly([P(0.15, 0.55), P(0.85, 0.55)]).concat(arc(0.5, 0.55, 0.35, 0.38, RIGHT, RIGHT - TAU * 0.85)), 1],
    // Small letters written in one stroke. A stem that goes down and back up
    // before an arch (h, n, m, r, b, p) retraces itself, as a pen does.
    ["h", poly([P(0.2, 0.02), P(0.2, 0.98), P(0.2, 0.6)]).concat(arc(0.5, 0.64, 0.3, 0.2, LEFT, TAU)).concat(poly([P(0.8, 0.64), P(0.8, 0.98)])), 1],
    ["h", poly([P(0.3, 0.02), P(0.3, 0.98), P(0.3, 0.58)]).concat(arc(0.5, 0.62, 0.2, 0.2, LEFT, TAU)).concat(poly([P(0.7, 0.62), P(0.72, 0.9)])), 1],
    ["n", poly([P(0.25, 0.3), P(0.25, 0.95), P(0.25, 0.55)]).concat(arc(0.47, 0.55, 0.22, 0.25, LEFT, TAU)).concat(poly([P(0.69, 0.55), P(0.72, 0.9)])), 1],
    ["n", poly([P(0.15, 0.2), P(0.15, 0.95), P(0.15, 0.5)]).concat(arc(0.5, 0.5, 0.35, 0.3, LEFT, TAU)).concat(poly([P(0.85, 0.5), P(0.85, 0.95)])), 1],
    // Pointed arches: many people write n and m as zigzags.
    ["n", poly([P(0.1, 0.4), P(0.2, 1.0), P(0.5, 0.0), P(0.9, 1.0)]), 1],
    ["m", poly([P(0.05, 0.4), P(0.15, 1.0), P(0.35, 0.05), P(0.55, 1.0), P(0.75, 0.05), P(0.95, 1.0)]), 1],
    ["m", poly([P(0.08, 0.25), P(0.08, 0.95), P(0.08, 0.5)]).concat(arc(0.29, 0.5, 0.21, 0.25, LEFT, TAU)).concat(poly([P(0.5, 0.5), P(0.5, 0.95), P(0.5, 0.5)])).concat(arc(0.71, 0.5, 0.21, 0.25, LEFT, TAU)).concat(poly([P(0.92, 0.5), P(0.92, 0.95)])), 1],
    ["r", poly([P(0.25, 0.2), P(0.25, 0.98), P(0.25, 0.55)]).concat(arc(0.55, 0.55, 0.3, 0.3, LEFT, LEFT + TAU * 0.36)), 1],
    ["b", poly([P(0.2, 0.0), P(0.2, 0.95), P(0.2, 0.62)]).concat(arc(0.5, 0.7, 0.3, 0.26, LEFT, LEFT + TAU)), 1],
    // P and p are one shape; the drawing's size decides (SAME_SHAPE).
    ["P", poly([P(0.2, 0.3), P(0.2, 1.0), P(0.2, 0.36)]).concat(arc(0.5, 0.44, 0.3, 0.22, LEFT, LEFT + TAU)), 1],
    ["P", poly([P(0.25, 0.0), P(0.25, 1.0), P(0.25, 0.0)]).concat(arc(0.25, 0.27, 0.5, 0.27, UP, DOWN)), 1],
    // Capitals written in one stroke: down the stem, back up, then the bowls.
    ["B", poly([P(0.2, 0.0), P(0.2, 1.0), P(0.2, 0.0)]).concat(arc(0.2, 0.25, 0.5, 0.25, UP, DOWN)).concat(arc(0.2, 0.75, 0.6, 0.25, UP, DOWN)), 1],
    ["D", poly([P(0.2, 0.0), P(0.2, 1.0), P(0.2, 0.0)]).concat(arc(0.2, 0.5, 0.65, 0.5, UP, DOWN)), 1],
    ["R", poly([P(0.25, 0.0), P(0.25, 1.0), P(0.25, 0.0)]).concat(arc(0.25, 0.27, 0.5, 0.27, UP, DOWN)).concat(poly([P(0.25, 0.54), P(0.8, 1.0)])), 1],
    ["G", arc(0.5, 0.5, 0.42, 0.45, -TAU / 9, -TAU / 9 - TAU * 0.78).concat(poly([P(0.9, 0.62), P(0.9, 0.55), P(0.55, 0.55)])), 1],
    ["J", poly([P(0.7, 0.0), P(0.7, 0.7)]).concat(arc(0.45, 0.7, 0.25, 0.28, RIGHT, TAU / 2)), 1],
    ["J", poly([P(0.2, 0.0), P(0.9, 0.0), P(0.6, 0.0), P(0.6, 0.7)]).concat(arc(0.38, 0.7, 0.22, 0.28, RIGHT, TAU / 2)), 1],
    // y: a u with a long tail that turns back left; k: stem, back up, a
    // loop out to the right, then the leg.
    ["y", poly([P(0.12, 0.0), P(0.15, 0.3)]).concat(arc(0.4, 0.3, 0.25, 0.2, TAU / 2, 0)).concat(poly([P(0.65, 0.3), P(0.65, 0.0), P(0.65, 0.8)])).concat(arc(0.42, 0.8, 0.23, 0.2, 0, TAU / 2)), 1],
    ["k", poly([P(0.2, 0.0), P(0.2, 1.0), P(0.2, 0.7), P(0.7, 0.4), P(0.3, 0.68), P(0.78, 1.0)]), 1],
    // Small letters that start with a bowl drawn anticlockwise from its top right.
    ["a", arc(0.42, 0.6, 0.32, 0.34, -TAU / 8, -TAU / 8 - TAU).concat(poly([P(0.72, 0.3), P(0.74, 0.97)])), 1],
    ["d", arc(0.42, 0.66, 0.32, 0.3, -TAU / 8, -TAU / 8 - TAU).concat(poly([P(0.72, 0.4), P(0.75, 0.0), P(0.75, 0.97)])), 1],
    ["q", arc(0.42, 0.34, 0.32, 0.3, -TAU / 8, -TAU / 8 - TAU).concat(poly([P(0.72, 0.1), P(0.74, 1.0)])), 1],
    ["g", arc(0.42, 0.3, 0.32, 0.26, -TAU / 8, -TAU / 8 - TAU).concat(poly([P(0.72, 0.1), P(0.74, 0.78)])).concat(arc(0.45, 0.78, 0.29, 0.2, RIGHT, TAU / 2)), 1],
    ["1", poly([P(0.25, 0.28), P(0.55, 0.05), P(0.55, 0.95)]), 1],
    ["7", poly([P(0.08, 0.08), P(0.9, 0.08), P(0.42, 0.95)]), 1],
    // 2: over the top, down the diagonal, along the base.
    ["2", arc(0.5, 0.3, 0.3, 0.24, LEFT + 0.35, TAU + 0.5).concat(poly([P(0.73, 0.42), P(0.1, 0.92), P(0.92, 0.92)])), 1],
    ["2", arc(0.5, 0.32, 0.32, 0.28, LEFT, TAU + 0.9).concat(poly([P(0.6, 0.6), P(0.1, 0.92), P(0.92, 0.92)])), 1],
    // A square 2: along the top, straight down, along the base.
    ["2", poly([P(0.05, 0.08), P(0.35, 0.03), P(0.32, 0.95), P(1.0, 0.8)]), 1],
    // 3: two bumps to the right, meeting in the middle.
    ["3", arc(0.5, 0.28, 0.26, 0.22, LEFT + 0.5, TAU + DOWN).concat(arc(0.5, 0.73, 0.3, 0.23, UP, TAU / 2 + 0.4)), 1],
    ["3", poly([P(0.12, 0.06), P(0.85, 0.06), P(0.45, 0.45)]).concat(arc(0.5, 0.7, 0.3, 0.24, UP, TAU / 2 + 0.4)), 1],
    // 4: the open corner and the upright (two strokes).
    ["4", [poly([P(0.6, 0.05), P(0.1, 0.65), P(0.9, 0.65)]), poly([P(0.68, 0.3), P(0.68, 0.95)])], 2],
    ["4", [poly([P(0.2, 0.05), P(0.15, 0.6), P(0.9, 0.6)]), poly([P(0.72, 0.1), P(0.72, 0.95)])], 2],
    // 5: in one stroke, or the body first and the flag second.
    ["5", poly([P(0.82, 0.07), P(0.3, 0.07), P(0.26, 0.45)]).concat(arc(0.48, 0.66, 0.32, 0.26, -0.75 * TAU / 2, 0.85 * TAU / 2)), 1],
    ["5", [poly([P(0.3, 0.07), P(0.26, 0.45)]).concat(arc(0.48, 0.66, 0.32, 0.26, -0.75 * TAU / 2, 0.85 * TAU / 2)), poly([P(0.3, 0.07), P(0.82, 0.07)])], 2],
    // Letters in two or more strokes. Straight crossings (+ T t = X) are read
    // by rule above; these have a curve or more than two lines.
    ["t", [poly([P(0.45, 0.0), P(0.45, 0.78)]).concat(arc(0.62, 0.78, 0.17, 0.17, LEFT, TAU / 8)), poly([P(0.15, 0.32), P(0.8, 0.32)])], 2],
    ["f", [arc(0.55, 0.17, 0.2, 0.15, -TAU / 12, -TAU / 2).concat(poly([P(0.35, 0.17), P(0.35, 1.0)])), poly([P(0.12, 0.42), P(0.65, 0.42)])], 2],
    ["A", [poly([P(0.1, 1.0), P(0.5, 0.0), P(0.9, 1.0)]), poly([P(0.3, 0.62), P(0.7, 0.62)])], 2],
    ["K", [poly([P(0.2, 0.0), P(0.2, 1.0)]), poly([P(0.8, 0.0), P(0.22, 0.55), P(0.8, 1.0)])], 2],
    ["B", [poly([P(0.2, 0.0), P(0.2, 1.0)]), arc(0.2, 0.25, 0.5, 0.25, UP, DOWN).concat(arc(0.2, 0.75, 0.6, 0.25, UP, DOWN))], 2],
    ["D", [poly([P(0.2, 0.0), P(0.2, 1.0)]), arc(0.2, 0.5, 0.65, 0.5, UP, DOWN)], 2],
    ["P", [poly([P(0.25, 0.0), P(0.25, 1.0)]), arc(0.25, 0.27, 0.5, 0.27, UP, DOWN)], 2],
    ["R", [poly([P(0.25, 0.0), P(0.25, 1.0)]), arc(0.25, 0.27, 0.5, 0.27, UP, DOWN).concat(poly([P(0.25, 0.54), P(0.8, 1.0)]))], 2],
    ["Q", [arc(0.47, 0.45, 0.4, 0.43, UP, UP - TAU), poly([P(0.55, 0.65), P(0.95, 1.0)])], 2],
    ["k", [poly([P(0.2, 0.0), P(0.2, 1.0)]), poly([P(0.72, 0.4), P(0.24, 0.7), P(0.78, 1.0)])], 2],
    ["y", [poly([P(0.1, 0.0), P(0.5, 0.5)]), poly([P(0.9, 0.0), P(0.25, 1.0)])], 2],
    // Two crossed diagonals are read by rule; this catches shaky ones.
    ["X", [poly([P(0.1, 0.1), P(0.9, 0.9)]), poly([P(0.9, 0.1), P(0.1, 0.9)])], 2],
    ["I", [poly([P(0.1, 0.0), P(0.9, 0.0)]), poly([P(0.5, 0.0), P(0.5, 1.0)]), poly([P(0.1, 1.0), P(0.9, 1.0)])], 3],
    ["Y", [poly([P(0.1, 0.0), P(0.5, 0.48), P(0.9, 0.0)]), poly([P(0.5, 0.48), P(0.5, 1.0)])], 2],
    ["H", [poly([P(0.15, 0.0), P(0.15, 1.0)]), poly([P(0.85, 0.0), P(0.85, 1.0)]), poly([P(0.15, 0.5), P(0.85, 0.5)])], 3],
    ["F", [poly([P(0.2, 0.0), P(0.2, 1.0)]), poly([P(0.2, 0.0), P(0.85, 0.0)]), poly([P(0.2, 0.48), P(0.7, 0.48)])], 3],
    ["E", [poly([P(0.2, 0.0), P(0.2, 1.0)]), poly([P(0.2, 0.0), P(0.85, 0.0)]), poly([P(0.2, 0.5), P(0.7, 0.5)]), poly([P(0.2, 1.0), P(0.85, 1.0)])], 4],
    // 6: down the left side, then a loop at the bottom.
    ["6", poly([P(0.7, 0.04), P(0.42, 0.22), P(0.25, 0.5)]).concat(arc(0.5, 0.7, 0.25, 0.24, LEFT, LEFT - TAU)), 1],
    // 9: a loop at the top, then the stem.
    ["9", arc(0.45, 0.28, 0.27, 0.24, RIGHT, RIGHT - TAU).concat(poly([P(0.72, 0.28), P(0.68, 0.95)])), 1],
    // 8: an S down, then back up the other side.
    ["8", arc(0.5, 0.26, 0.22, 0.2, -TAU / 8, DOWN - TAU).concat(arc(0.5, 0.71, 0.27, 0.23, UP, UP + TAU)).concat(arc(0.5, 0.26, 0.22, 0.2, DOWN, -TAU / 8)), 1, true],
  ];
  // A stroke drawn up or down, left or right, is the same stroke: each is
  // turned to run top to bottom, or left to right when it is mostly
  // horizontal, before strokes are joined in one path.
  function canonical(stroke) {
    const a = stroke[0], z = stroke[stroke.length - 1];
    const vertical = Math.abs(z.y - a.y) >= Math.abs(z.x - a.x);
    const backwards = vertical ? z.y < a.y : z.x < a.x;
    return backwards ? stroke.slice().reverse() : stroke;
  }
  // Every order the strokes could have been drawn in (at most 4! = 24).
  function orders(items) {
    if (items.length <= 1) return [items];
    return items.flatMap((x, i) => orders(items.filter((_, j) => j !== i)).map((rest) => [x, ...rest]));
  }
  const VEC = TEMPLATES.map(([shape, pts, strokes, closed]) => {
    const path = strokes > 1 ? pts.map(canonical).flat() : pts;
    return { shape, v: toVector(resample(path)), strokes, closed: !!closed };
  });

  // ---------- parts: dots and straight lines ----------
  const SYMBOLS = new Set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "·", ":", "="]);
  // Letters whose small and capital forms are the same shape; the drawing's
  // height decides between them.
  const SAME_SHAPE = new Set(["C", "O", "P", "S", "U", "V", "W", "X", "Z"]);

  // Does the straight segment p (a, z) cross the drawn stroke anywhere?
  function crossesStroke(p, stroke, slack) {
    for (let i = 1; i < stroke.length; i++) {
      if (segmentsCross(p, { a: stroke[i - 1], z: stroke[i] }, slack)) return true;
    }
    return false;
  }
  // Where a stroke that runs top to bottom bends: horizontal spread of its
  // top quarter against its bottom quarter.
  function hookEnd(stroke, b) {
    const spread = (lo, hi) => {
      const xs = stroke.filter((q) => q.y >= lo && q.y <= hi).map((q) => q.x);
      return xs.length ? Math.max(...xs) - Math.min(...xs) : 0;
    };
    const top = spread(b.y0, b.y0 + b.h / 4), bottom = spread(b.y1 - b.h / 4, b.y1);
    if (Math.max(top, bottom) < 0.18 * b.h) return "none";
    return top > bottom ? "top" : "bottom";
  }

  function part(stroke, dotLimit) {
    const b = box(stroke);
    const size = Math.max(b.w, b.h);
    if (size <= dotLimit) return { kind: "dot", b };
    const a = stroke[0], z = stroke[stroke.length - 1];
    const chord = dist(a, z) || 1;
    // Angle from horizontal, 0 to 90 degrees; which end came first does not matter.
    const angle = Math.atan2(Math.abs(z.y - a.y), Math.abs(z.x - a.x)) / DEG;
    // Straight: no point strays far from the line between the ends. Path
    // length would be thrown off by a shaky hand.
    const bulge = Math.max(...stroke.map((p) => Math.abs((z.x - a.x) * (a.y - p.y) - (a.x - p.x) * (z.y - a.y)) / chord));
    const straight = chord >= 0.8 * size && bulge <= 0.12 * chord;
    // A tall stroke, straight or hooked at one end, can be the upright of t.
    const upright = b.h >= 1.5 * b.w;
    const dir = angle < 30 ? "h" : angle > 60 ? "v" : "d";
    const slope = Math.sign((z.x - a.x) * (z.y - a.y));
    return { kind: straight ? "line" : "curve", b, a, z, angle, dir, slope, size, upright, stroke };
  }
  function segmentsCross(p, q, slack) {
    // Do the two segments meet, allowing `slack` pixels of overshoot?
    const grow = (s) => {
      const d = dist(s.a, s.z) || 1, ux = (s.z.x - s.a.x) / d * slack, uy = (s.z.y - s.a.y) / d * slack;
      return [{ x: s.a.x - ux, y: s.a.y - uy }, { x: s.z.x + ux, y: s.z.y + uy }];
    };
    const [a, b] = grow(p), [c, d] = grow(q);
    const cross = (o, e, f) => (e.x - o.x) * (f.y - o.y) - (e.y - o.y) * (f.x - o.x);
    return cross(a, b, c) * cross(a, b, d) <= 0 && cross(c, d, a) * cross(c, d, b) <= 0;
  }

  // Where two segments cross, as a share of each one's length: both must be
  // between 0.2 and 0.8.
  function crossMid(p, q) {
    const r = { x: p.z.x - p.a.x, y: p.z.y - p.a.y }, s = { x: q.z.x - q.a.x, y: q.z.y - q.a.y };
    const den = r.x * s.y - r.y * s.x;
    if (!den) return false;
    const t = ((q.a.x - p.a.x) * s.y - (q.a.y - p.a.y) * s.x) / den;
    const u = ((q.a.x - p.a.x) * r.y - (q.a.y - p.a.y) * r.x) / den;
    return t > 0.2 && t < 0.8 && u > 0.2 && u < 0.8;
  }

  function fromParts(parts, all, task) {
    const dots = parts.filter((p) => p.kind === "dot");
    const rest = parts.filter((p) => p.kind !== "dot");
    const lines = rest.filter((p) => p.kind === "line");
    if (!rest.length) {
      if (dots.length === 1) return { shape: "·" };
      if (dots.length === 2) {
        const [a, b] = dots.map((d) => d.b);
        if (Math.abs(a.cy - b.cy) > Math.abs(a.cx - b.cx)) return { shape: ":" };
      }
      return null;
    }
    if (rest.length === 1 && lines.length === 1) {
      const L = lines[0];
      if (!dots.length) {
        if (L.dir === "h") return { shape: "-" };
        if (L.dir === "v") return { shape: task === "chars" ? "l" : "1" };
        return null;
      }
      const within = (d) => d.b.cx > L.b.x0 - L.size * 0.25 && d.b.cx < L.b.x1 + L.size * 0.25;
      if (dots.length === 2 && L.dir === "h" && dots.every(within)) {
        const above = dots.filter((d) => d.b.cy < L.b.cy).length;
        if (above === 1) return { shape: "÷" };
      }
      if (dots.length === 1 && L.dir === "v" && dots[0].b.cy < L.b.y0 &&
          Math.abs(dots[0].b.cx - L.b.cx) < L.size * 0.4) return { shape: "i" };
      return null;
    }
    // An upright crossed by a bar: T, t, f or +. The upright may be hooked
    // (t, f); the bar is a straight horizontal line.
    if (rest.length === 2 && !dots.length) {
      const bar = rest.find((q) => q.kind === "line" && q.dir === "h");
      const up = rest.find((q) => q !== bar && q.upright && (q.kind === "curve" || q.dir === "v"));
      // The bar must stick out on both sides of the upright where they
      // meet: a 5's flag touches its upright but runs off to one side only.
      const nearest = up && bar && up.stroke.reduce((m, q) => (Math.abs(q.y - bar.b.cy) < Math.abs(m.y - bar.b.cy) ? q : m));
      const overhang = bar && 0.12 * bar.b.w;
      const across = nearest && bar.b.x0 < nearest.x - overhang && bar.b.x1 > nearest.x + overhang;
      if (across && crossesStroke(bar, up.stroke, Math.max(all.w, all.h) * 0.08)) {
        const at = (bar.b.cy - up.b.y0) / (up.b.h || 1);
        const hook = up.kind === "curve" ? hookEnd(up.stroke, up.b) : "none";
        if (hook === "top" && at < 0.7 && task === "chars") return { shape: "f" };
        if (hook === "bottom" && at < 0.7 && task === "chars") return { shape: "t" };
        if (hook === "none") {
          if (at < 0.18) return { shape: "T" };
          return { shape: at < 0.42 && task === "chars" ? "t" : "+" };
        }
      }
    }
    if (rest.length === 2 && lines.length === 2 && !dots.length) {
      const [p, q] = lines;
      const slack = Math.max(all.w, all.h) * 0.12;
      if (p.dir === "h" && q.dir === "h") {
        const overlap = Math.min(p.b.x1, q.b.x1) - Math.max(p.b.x0, q.b.x0);
        const gap = Math.abs(p.b.cy - q.b.cy);
        if (overlap > 0.3 * Math.min(p.size, q.size) && gap > 0.12 * Math.max(p.size, q.size)) return { shape: "=" };
        return null;
      }
      const h = p.dir === "h" ? p : q.dir === "h" ? q : null;
      const v = p.dir === "v" ? p : q.dir === "v" ? q : null;
      if (h && v && segmentsCross(h, v, slack)) {
        // A bar across the very top of the upright is a T. Higher than the
        // middle it is a small t when letters are being read, and a +
        // otherwise.
        const at = (h.b.cy - v.b.y0) / v.size;
        if (at < 0.18) return { shape: "T" };
        return { shape: at < 0.42 && task === "chars" ? "t" : "+" };
      }
      // Two diagonals crossing near their middles: X. A y's arms meet near
      // the end of one of them, so it is left to the templates.
      if (p.dir === "d" && q.dir === "d" && p.slope !== q.slope && segmentsCross(p, q, slack) && crossMid(p, q)) return { shape: "X" };
    }
    return null;
  }

  // ---------- slant ----------
  // Handwriting leans. Rotation is allowed for by the match, but a lean is a
  // shear: the tops of the uprights move sideways and the bottoms do not.
  // Estimate it from the parts of the path that run close to vertical and
  // shear it back to upright.
  function deslant(pts) {
    let sum = 0, weight = 0;
    for (let i = 1; i < pts.length; i++) {
      const dx = pts[i].x - pts[i - 1].x, dy = pts[i].y - pts[i - 1].y;
      const len = Math.hypot(dx, dy);
      if (!len || Math.abs(dx) > Math.abs(dy)) continue;  // within 45 degrees of vertical
      // dx/dy is the same for a stroke drawn up or down.
      sum += (dx / dy) * len;
      weight += len;
    }
    if (!weight) return null;
    const shear = Math.max(-0.7, Math.min(0.7, sum / weight));
    if (Math.abs(shear) < 0.05) return null;
    const cy = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    return pts.map((p) => ({ x: p.x - shear * (p.y - cy), y: p.y }));
  }

  // ---------- corners, to tell a Z from a 2 ----------
  // The largest change of direction over a short stretch of the path, and
  // where along the path (0 to 1) it happens.
  function sharpestTurn(pts, from, to) {
    const k = 3;
    let best = { turn: 0, at: 0 };
    const lo = Math.max(k, Math.floor(from * (pts.length - 1))), hi = Math.min(pts.length - 1 - k, Math.ceil(to * (pts.length - 1)));
    for (let i = lo; i <= hi; i++) {
      const a1 = Math.atan2(pts[i].y - pts[i - k].y, pts[i].x - pts[i - k].x);
      const a2 = Math.atan2(pts[i + k].y - pts[i].y, pts[i + k].x - pts[i].x);
      let d = Math.abs(a2 - a1) % TAU;
      if (d > Math.PI) d = TAU - d;
      if (d > best.turn) best = { turn: d, at: i / (pts.length - 1) };
    }
    return best;
  }

  // ---------- a, d, q and g ----------
  // All four start with the same bowl; the stem decides. The bowl ends where
  // the path comes back closest to where it started. A stem that ends level
  // with the bowl is an a, one that first rises above it a d, one that drops
  // below it a q, or a g when its tail curls back left.
  function bowlLetter(pts) {
    const start = pts[0];
    let close = Math.floor(pts.length * 0.3), best = Infinity;
    for (let i = close; i < pts.length * 0.85; i++) {
      const d = dist(pts[i], start);
      if (d < best) { best = d; close = i; }
    }
    const bowl = box(pts.slice(0, close + 1)), rest = box(pts.slice(close));
    const h = bowl.h || 1;
    if (bowl.y0 - rest.y0 > 0.35 * h) return "d";
    if (rest.y1 - bowl.y1 > 0.35 * h) {
      const tail = pts.slice(Math.floor(pts.length * 0.8));
      const bottom = tail.reduce((m, q) => (q.y > m.y ? q : m));
      return tail[tail.length - 1].x < bottom.x - 0.2 * h ? "g" : "q";
    }
    return "a";
  }

  // ---------- the answer ----------
  // Match needed to answer. Drawings of every shape here score 0.94 or more
  // even when drawn sloppily; letters this matcher does not know (k, f, y,
  // j, t, &) top out around 0.93 against their nearest template, and at the
  // old 0.8 were read as that template (a small h came back as V).
  // tests/test_playground_shapes.py holds both sides of this line.
  const ACCEPT = 0.94;
  const TIE = 0.03;
  const isDigit = (shape) => /^[0-9]$/.test(shape);
  function known(task) {
    return task === "chars"
      ? "every letter A to Z and a to z; small and capital letters of the same shape (c C, o O, p P, s S, u U, v V, w W, x X, z Z) are told apart by size"
      : "0 to 9, and - + = · ÷";
  }
  // strokes: arrays of {x, y}. opts.size: the drawing area's height in
  // pixels. opts.task: "chars" or "symbols", for the shapes that are both.
  // ---------- more than one character ----------
  // Characters written side by side ("12") are split where the strokes stop
  // overlapping left to right. Dots join the character they sit over.
  function characters(strokes, padH) {
    const items = strokes.map((s) => ({ s, b: box(s) }));
    const dotSize = 0.07 * padH;
    const marks = items.filter((it) => Math.max(it.b.w, it.b.h) > dotSize);
    const dots = items.filter((it) => !marks.includes(it));
    if (marks.length < 2) return { groups: [strokes], gap: 0 };
    marks.sort((p, q) => p.b.x0 - q.b.x0);
    const groups = [];
    for (const it of marks) {
      const g = groups[groups.length - 1];
      if (g && it.b.x0 <= g.x1) { g.items.push(it); g.x1 = Math.max(g.x1, it.b.x1); }
      else groups.push({ items: [it], x0: it.b.x0, x1: it.b.x1 });
    }
    for (const d of dots) {
      const home = groups.reduce((m, g) => {
        const off = Math.max(g.x0 - d.b.cx, d.b.cx - g.x1, 0);
        return !m || off < m.off ? { g, off } : m;
      }, null);
      home.g.items.push(d);
    }
    let gap = Infinity;
    for (let i = 1; i < groups.length; i++) gap = Math.min(gap, groups[i].x0 - groups[i - 1].x1);
    const height = box(strokes.flat()).h || 1;
    // Keep drawing order inside each character.
    const order = new Map(strokes.map((s, i) => [s, i]));
    return {
      groups: groups.map((g) => g.items.map((it) => it.s).sort((p, q) => order.get(p) - order.get(q))),
      gap: gap / height,
    };
  }

  // Shapes that are a letter or a digit depending on the task: a vertical
  // line (l, I, 1), a ring (O, o, 0) and a Z (Z, z, 2).
  const TWINS = new Set(["l", "I", "1", "O", "o", "0", "Z", "z"]);
  // Characters written together are read together: next to a digit that is
  // clearly a digit, a twin is read as a digit too, and next to a clear
  // letter as a letter. "12" drawn on the Letters task used to read l, 2.
  // When every character is a twin, they are read as digits whatever the
  // page's task: characters written side by side are nearly always a number
  // ("10", not "lo"), and the letter recordings hold one letter each.
  function agree(groups, parts, opts) {
    const read = parts.filter((p) => p && p.label);
    const clear = read.filter((p) => !TWINS.has(p.label));
    const digits = clear.some((p) => p.task === "symbols");
    const letters = clear.some((p) => p.task === "chars");
    let task;
    if (digits !== letters) task = digits ? "symbols" : "chars";
    else if (!clear.length && read.length > 1) task = "symbols";
    else return parts;
    return retask(groups, parts, opts, task);
  }
  function retask(groups, parts, opts, task) {
    return parts.map((p, i) => (p && p.label && TWINS.has(p.label) && p.task !== task ? recogniseOne(groups[i], { ...opts, task }) : p));
  }
  // When every character is a twin, the same characters read the other way
  // ("10" and "lo"), for the page to offer. Null otherwise: next to a clear
  // 2, "l2" is no reading at all.
  function otherReading(groups, parts, opts) {
    const read = parts.filter((p) => p && p.label);
    const twins = read.filter((p) => TWINS.has(p.label));
    if (!twins.length || twins.length !== read.length) return null;
    const task = twins[0].task === "symbols" ? "chars" : "symbols";
    const other = parts.map((p, i) => (p && p.label && TWINS.has(p.label) ? recogniseOne(groups[i], { ...opts, task }) : p));
    const same = other.every((p, i) => (p && p.label) === (parts[i] && parts[i].label));
    return same ? null : other;
  }

  // strokes: arrays of {x, y}. opts.size: the drawing area's height in
  // pixels. opts.task: "chars" or "symbols". Returns one character, or, for
  // characters written side by side, { sequence: [one per character] }.
  function recognise(strokes, opts = {}) {
    const clean = strokes.filter((s) => s && s.length);
    if (!clean.length) return null;
    const { groups, gap } = characters(clean, opts.size || 170);
    if (groups.length < 2 || opts.ranking) return recogniseOne(clean, opts);
    const whole = recogniseOne(clean, opts);
    const parts = agree(groups, groups.map((g) => recogniseOne(g, opts)), opts);
    const allRead = parts.every((p) => p && p.label);
    // A clear gap means separate characters. A narrow one (a sloppy H whose
    // bar misses a stem) is one character if it reads as one.
    const withOther = () => ({ sequence: parts, other: otherReading(groups, parts, opts) });
    if (allRead && (gap > 0.08 || !whole.label)) return withOther();
    if (whole.label) return whole;
    if (parts.some((p) => p && p.label)) return withOther();
    return whole;
  }

  function recogniseOne(strokes, opts = {}) {
    const task = opts.task === "chars" ? "chars" : "symbols";
    const padH = opts.size || 170;
    const clean = strokes.filter((s) => s && s.length);
    if (!clean.length) return null;
    const all = box(clean.flat());
    const extent = Math.max(all.w, all.h);
    // Dots: tiny marks, or marks much smaller than the rest of the drawing.
    const dotLimit = Math.max(4, Math.min(0.07 * padH, clean.length > 1 ? 0.3 * extent : Infinity));
    const parts = clean.map((s) => part(s, dotLimit));

    let found = fromParts(parts, all, task);
    let score = 1;
    if (!found) {
      const body = clean.filter((s, i) => parts[i].kind !== "dot");
      const flat = body.flat();
      if (flat.length < 2 || pathLength(flat) < dotLimit * 2) return { label: null, reason: `Too small to read. Draw it bigger: ${known(task)}.` };
      const pts = resample(flat);
      const n = body.length;
      // One stroke: its own path, which the match reads either way. More:
      // every stroke order, each stroke turned to its canonical direction.
      // A closed stroke (the ring of a Q) has no top-to-bottom direction,
      // so both directions are tried.
      const choices = body.map((st) => {
        const c = canonical(st);
        const b = box(st);
        const closed = dist(st[0], st[st.length - 1]) < 0.2 * Math.max(b.w, b.h);
        return closed ? [c, c.slice().reverse()] : [c];
      });
      const pick = (i) => (i === choices.length ? [[]] : choices[i].flatMap((c) => pick(i + 1).map((rest) => [c, ...rest])));
      const paths = n === 1
        ? [pts]
        : (n <= 4 ? pick(0).flatMap(orders) : [body]).map((o) => resample(o.flat()));
      const vectors = [];
      for (const path of paths) {
        vectors.push(toVector(path));
        const upright = deslant(path);
        if (upright) vectors.push(toVector(upright));
      }
      // Best score per shape, highest first.
      const byShape = new Map();
      for (const t of VEC) {
        if (t.strokes !== n) continue;
        const sc = Math.max(...vectors.map((v) => bestMatch(v, t)));
        if (!byShape.has(t.shape) || sc > byShape.get(t.shape)) byShape.set(t.shape, sc);
      }
      const ranking = [...byShape].map(([shape, sc]) => ({ shape, sc })).sort((a, b) => b.sc - a.sc);
      if (opts.ranking) return { ranking: ranking.slice(0, 4) };
      let best = ranking[0] || null;
      // A digit and a letter can share a shape (9 and q, 6 and b). When the
      // two score within TIE of each other, the one the current task reads
      // wins.
      if (best) {
        const inTask = (shape) => (task === "symbols") === (isDigit(shape) || shape === "O/0");
        const rival = ranking.find((r) => r !== best && r.sc >= best.sc - TIE && inTask(r.shape));
        if (!inTask(best.shape) && rival && best.shape !== "Z" && best.shape !== "2") best = rival;
      }
      if (!best || best.sc < ACCEPT) {
        return { label: null, reason: `Not sure what that was. Shapes this demo knows: ${known(task)}.` };
      }
      if (best.shape === "Z" || best.shape === "2") {
        // A Z turns sharply at the top right; a 2 goes round.
        const top = sharpestTurn(pts, 0.1, 0.45);
        best.shape = top.turn > 95 * DEG ? "Z" : "2";
      }
      if (["a", "d", "q", "g"].includes(best.shape)) best.shape = bowlLetter(pts);
      // A hook with a dot over it is a small j.
      const dotsOver = parts.filter((q) => q.kind === "dot" && q.b.cy < box(flat).y0);
      if (best.shape === "J" && dotsOver.length === 1) best.shape = "j";
      found = { shape: best.shape };
      score = best.sc;
    }

    let shape = found.shape;
    if (shape === "O/0") shape = task === "chars" ? "O" : "0";
    if (task === "symbols" && shape === "Z") shape = "2";
    if (task === "symbols" && shape === "l") shape = "1";
    let label = shape, note = "";
    if (shape === "÷") { label = ":"; note = "The recordings write division as :, so the AI reads that sign."; }
    if (shape === "·") note = "The multiplication dot.";
    if (SAME_SHAPE.has(shape) && all.h < 0.4 * padH) {
      label = shape.toLowerCase();
      note = "Small, so read as a small letter.";
    }
    return { label, shape: shape === "÷" ? "÷" : label, note, score, task: SYMBOLS.has(label) ? "symbols" : "chars" };
  }

  const api = { recognise, known, resample, toVector };
  root.PlaygroundShapes = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
