// Vahini playground: card 1, "How the pen works".
//
// The mouse (or a finger) stands in for the sensor pen that recorded the
// Fraunhofer OnHW dataset. While you draw, five sensor groups animate (an
// animation, not data). A small shape matcher then guesses which character
// you drew, using the closed-form best-rotation distance of the Protractor
// recogniser (Y. Li, "Protractor: a fast and accurate gesture recognizer",
// CHI 2010). The guess selects a REAL recording of that character from the
// OnHW test set, and the page's own Recognize (app.js) runs the model on it.
//
// Drawing code written by Vahini Technologies for vahinitech.com and released
// here under Apache-2.0 with the owner's approval (2026-09-26). Colours come
// from the Vahini design system (--v-* tokens).
"use strict";

(function () {
  const root = document.getElementById("write");
  if (!root) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--v-${name}`).trim();
  const INK = token("pen-ink");
  const $ = (id) => document.getElementById(id);
  const draw = $("w-draw"), pad = draw.parentNode, pen = $("w-pen"), hint = $("w-hint"), shapeEl = $("w-shape");
  function fit(cv) {
    const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
    const g = cv.getContext("2d"); g.scale(d, d); return g;
  }
  let gD = fit(draw);
  window.addEventListener("resize", () => { gD = fit(draw); repaint(); });

  // ---------- sensor lines: the OnHW pen's five groups (13 channels) ----------
  // Colours go through the CSSOM: the page's CSP blocks inline style attributes.
  const CH = [["Front acc.", "chart-1", "wave"], ["Rear acc.", "chart-2", "wave"], ["Gyro", "chart-3", "wave"],
    ["Magnet", "chart-1", "wave"], ["Force", "warning", "level"]];
  const bus = $("w-bus");
  bus.innerHTML = CH.map((c) => `<div class="w-chan"><em>${c[0]}</em>` +
    '<svg class="w-wave" viewBox="0 0 100 16" preserveAspectRatio="none"><polyline points="0,8 100,8"/></svg></div>').join("");
  [...bus.children].forEach((el, i) => el.style.setProperty("--ch", `var(--v-${CH[i][1]})`));
  const waveEls = [...bus.querySelectorAll("polyline")];
  const levelHistory = CH.map((c) => (c[2] === "level" ? new Array(21).fill(8) : null));
  let tick = 0;
  function pulseBus(speed) {
    tick += 1;
    const amp = Math.min(6.5, 1 + speed * 0.42);
    waveEls.forEach((el, i) => {
      let pts;
      if (CH[i][2] === "level") {
        const h = levelHistory[i]; h.shift(); h.push(8 - amp);
        pts = h.map((y, x) => `${x * 5},${y.toFixed(1)}`);
      } else {
        pts = [];
        for (let x = 0; x <= 100; x += 5) pts.push(`${x},${(8 + Math.sin(x * 0.24 + tick * 0.3 + i * 1.7) * amp).toFixed(1)}`);
      }
      el.setAttribute("points", pts.join(" "));
    });
  }
  function calmBus() {
    levelHistory.forEach((h) => { if (h) h.fill(8); });
    waveEls.forEach((el) => el.setAttribute("points", "0,8 100,8"));
  }

  // ---------- the drawn pen follows a mouse ----------
  if (window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
    pad.addEventListener("pointerenter", () => pen.classList.add("show"));
    pad.addEventListener("pointerleave", () => pen.classList.remove("show"));
    pad.addEventListener("pointermove", (e) => {
      const r = pad.getBoundingClientRect();
      pen.style.transform = `translate(${e.clientX - r.left - 6}px,${e.clientY - r.top - 144}px) rotate(33deg)`;
    });
  }

  // ---------- drawing ----------
  // A finished character is replaced by the next drawing; strokes made
  // within half a second of each other count as one character.
  let strokes = [], cur = null, lastPt = null, drawing = false, timer = null, finished = false;
  function pos(e) { const r = draw.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top, t: performance.now() }; }
  function line(g, a, b, w) {
    g.strokeStyle = INK; g.lineWidth = w; g.lineCap = "round"; g.lineJoin = "round";
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  }
  function repaint() {
    gD.clearRect(0, 0, draw.width, draw.height);
    strokes.forEach((st) => { for (let j = 1; j < st.length; j++) line(gD, st[j - 1], st[j], 3.2); });
  }
  draw.addEventListener("pointerdown", (e) => {
    e.preventDefault(); draw.setPointerCapture(e.pointerId);
    if (finished) { strokes = []; finished = false; gD.clearRect(0, 0, draw.width, draw.height); }
    // Hide the previous answer while a new character is drawn.
    if (state.revealed && !state.busy) { state.revealed = false; renderModel(); }
    drawing = true; cur = [pos(e)]; lastPt = cur[0];
    hint.classList.add("off"); clearTimeout(timer);
    shapeEl.textContent = "Feeling the movement…";
  });
  draw.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = pos(e), dt = Math.max(1, p.t - lastPt.t);
    const speed = Math.hypot(p.x - lastPt.x, p.y - lastPt.y) / dt * 16;
    pulseBus(speed);
    line(gD, lastPt, p, Math.max(2.1, 4.6 - Math.min(2.4, speed * 0.3)));
    cur.push(p); lastPt = p;
  });
  function endStroke() {
    if (!drawing) return;
    drawing = false; calmBus();
    if (cur && cur.length > 2) strokes.push(cur);
    cur = null;
    if (strokes.length) timer = setTimeout(recognise, reduce ? 0 : 500);
  }
  draw.addEventListener("pointerup", endStroke);
  draw.addEventListener("pointercancel", endStroke);
  draw.addEventListener("pointerleave", endStroke);
  $("w-clear").addEventListener("click", () => {
    strokes = []; cur = null; finished = false; clearTimeout(timer);
    gD.clearRect(0, 0, draw.width, draw.height); hint.classList.remove("off"); shapeEl.textContent = ""; calmBus();
  });

  // ---------- the shape matcher ----------
  const N = 48;
  function resample(points) {
    let length = 0;
    for (let j = 1; j < points.length; j++) length += Math.hypot(points[j].x - points[j - 1].x, points[j].y - points[j - 1].y);
    const step = length / (N - 1), out = [{ x: points[0].x, y: points[0].y }];
    let D = 0;
    for (let j = 1; j < points.length; j++) {
      const d = Math.hypot(points[j].x - points[j - 1].x, points[j].y - points[j - 1].y);
      if (D + d >= step && d > 0) {
        const q = { x: points[j - 1].x + ((step - D) / d) * (points[j].x - points[j - 1].x),
          y: points[j - 1].y + ((step - D) / d) * (points[j].y - points[j - 1].y) };
        out.push(q); points.splice(j, 0, q); D = 0;
      } else D += d;
    }
    while (out.length < N) out.push(out[out.length - 1]);
    return out.slice(0, N);
  }
  function toVector(points) {
    const c = points.reduce((a, p) => ({ x: a.x + p.x / points.length, y: a.y + p.y / points.length }), { x: 0, y: 0 });
    const v = [];
    let mag = 0;
    points.forEach((p) => { const x = p.x - c.x, y = p.y - c.y; v.push(x, y); mag += x * x + y * y; });
    mag = Math.sqrt(mag) || 1;
    return v.map((x) => x / mag);
  }
  // Protractor: the best achievable match over every rotation is
  // sqrt(dot^2 + cross^2), so no angle has to be guessed.
  function angleScore(a, b) {
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
  for (let d = -32; d <= 32; d += 4) SWEEP.push(d * Math.PI / 180);
  function bestMatch(v, t) {
    const rv = reversedVec(v);
    if (t.closed) {
      let best = Math.max(angleScore(v, t.v), angleScore(rv, t.v));
      const step = Math.max(1, Math.round(v.length / 2 / 12));
      for (let k = step; k < v.length / 2; k += step) best = Math.max(best, angleScore(shiftedVec(v, k), t.v), angleScore(shiftedVec(rv, k), t.v));
      return best;
    }
    return Math.max(...SWEEP.map((a) => Math.max(dotProduct(rotateVec(v, a), t.v), dotProduct(rotateVec(rv, a), t.v))));
  }
  const P = (x, y) => ({ x, y });
  function poly(pts) {
    const out = [];
    for (let j = 0; j < pts.length - 1; j++) for (let k = 0; k < 12; k++) out.push(P(pts[j].x + (pts[j + 1].x - pts[j].x) * k / 12, pts[j].y + (pts[j + 1].y - pts[j].y) * k / 12));
    out.push(pts[pts.length - 1]);
    return out;
  }
  function arc(cx, cy, r, a0, a1) {
    const out = [];
    for (let k = 0; k <= 28; k++) { const a = a0 + (a1 - a0) * k / 28; out.push(P(cx + r * Math.cos(a), cy + r * Math.sin(a))); }
    return out;
  }
  const TAU = Math.PI * 2;
  const TEMPLATES = [
    ["O", arc(0.5, 0.5, 0.42, -TAU / 4, -TAU / 4 + TAU), true],
    ["C", arc(0.5, 0.5, 0.42, -TAU / 8, -TAU / 8 - TAU * 0.72)],
    ["S", arc(0.5, 0.28, 0.22, -TAU / 4, -TAU / 4 - TAU / 2).concat(arc(0.5, 0.72, 0.22, TAU / 4, TAU * 0.75))],
    ["U", poly([P(0.1, 0.08), P(0.1, 0.6)]).concat(arc(0.5, 0.6, 0.4, Math.PI, TAU / 2 + Math.PI)).concat(poly([P(0.9, 0.6), P(0.9, 0.08)]))],
    ["V", poly([P(0.05, 0.05), P(0.5, 0.95), P(0.95, 0.05)])],
    ["W", poly([P(0.02, 0.05), P(0.27, 0.95), P(0.5, 0.35), P(0.73, 0.95), P(0.98, 0.05)])],
    ["M", poly([P(0.05, 0.95), P(0.05, 0.05), P(0.5, 0.6), P(0.95, 0.05), P(0.95, 0.95)])],
    ["N", poly([P(0.08, 0.95), P(0.08, 0.05), P(0.92, 0.95), P(0.92, 0.05)])],
    ["L", poly([P(0.15, 0.05), P(0.15, 0.9), P(0.85, 0.9)])],
    ["Z", poly([P(0.08, 0.08), P(0.92, 0.08), P(0.08, 0.92), P(0.92, 0.92)])],
    ["1", poly([P(0.5, 0.05), P(0.5, 0.95)])],
    ["7", poly([P(0.08, 0.08), P(0.9, 0.08), P(0.42, 0.95)])],
    ["2", arc(0.5, 0.3, 0.25, Math.PI, TAU * 0.55).concat(poly([P(0.68, 0.45), P(0.12, 0.92), P(0.9, 0.92)]))],
    ["3", arc(0.5, 0.28, 0.2, Math.PI * 0.8, -TAU * 0.3 + Math.PI).concat(arc(0.5, 0.7, 0.24, -TAU / 4, TAU * 0.42))],
  ];
  const VEC = TEMPLATES.map((t) => ({ name: t[0], v: toVector(resample(t[1].slice())), closed: !!t[2] }));
  // ---------- shape → a real recording → the page's Recognize ----------
  const DIGITS = new Set(["1", "2", "3", "7"]);
  const LETTERS = new Set(["O", "C", "S", "U", "V", "W", "M", "N", "L", "Z"]);
  function recognise() {
    finished = true;
    const flat = strokes.flat().map((p) => ({ x: p.x, y: p.y }));
    if (flat.length < 8) { shapeEl.textContent = ""; return; }
    const v = toVector(resample(flat));
    let best = null;
    VEC.forEach((t) => { const sc = bestMatch(v, t); if (!best || sc > best.sc) best = { name: t.name, sc }; });
    if (best.sc < 0.6) {
      shapeEl.textContent = "Not sure what that was. Try one clear shape: O, S, V, Z or 7.";
      return;
    }
    const ch = best.name;
    state.task = DIGITS.has(ch) ? "symbols" : "chars";
    if (state.task === "symbols") state.hand = "right";
    if (state.task === "chars") state.protocol = "indep";
    if (!available(state.task, state.protocol)) state.protocol = "indep";
    const curData = current();
    let i = curData ? curData.samples.findIndex((smp) => smp.label === ch) : -1;
    if (curData && i < 0) i = curData.samples.findIndex((smp) => smp.label.toLowerCase() === ch.toLowerCase());
    if (i < 0) {
      renderAll();
      shapeEl.textContent = `Looks like ${ch}, but there is no recording of it for this hand. Pick one below, or switch hand.`;
      return;
    }
    const other = curData.samples[i].label !== ch;
    state.sample = i;
    renderAll();
    shapeEl.textContent = `Looks like ${ch} · picked a real recording of “${curData.samples[i].label}”${other ? " (its other case)" : ""}.`;
    recognize();
  }
  calmBus();
})();
