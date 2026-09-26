// Vahini playground: "Draw a letter", the first step of the page.
//
// The mouse (or a finger) stands in for the sensor pen that recorded the
// Fraunhofer OnHW dataset. While you draw, five sensor groups animate (an
// animation, not data). A small shape matcher then guesses which character
// you drew, using the closed-form best-rotation distance of the Protractor
// recogniser (Y. Li, "Protractor: a fast and accurate gesture recognizer",
// CHI 2010). That guess picks a REAL recording of the same character from
// the OnHW test set (data/public.js), and the page shows the real model's
// answer and scores for it, with the same hand, writer, model and training
// choices as the rest of the page (app.js state).
//
// Drawing code written by Vahini Technologies for vahinitech.com and released
// here under Apache-2.0 with the owner's approval (2026-09-26). Colours come
// from the Vahini design system (--v-* tokens).
"use strict";

(function () {
  const root = document.getElementById("write");
  if (!root) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = getComputedStyle(document.documentElement);
  const token = (name) => css.getPropertyValue(`--v-${name}`).trim();
  const INK = token("pen-ink");
  const GLOW = token("engine-glow");
  const GRID = token("engine-border");

  const $ = (id) => document.getElementById(id);
  const draw = $("w-draw"), recon = $("w-recon");
  function fit(cv) {
    const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
    const g = cv.getContext("2d"); g.scale(d, d); return g;
  }
  let gD = fit(draw), gR = fit(recon);
  // A resize (or a phone turning) resets both canvases, so redraw what they held.
  window.addEventListener("resize", () => { gD = fit(draw); gR = fit(recon); repaint(); if (rebuilt) drawRebuilt(); });

  // ---------- sensor bus: six channels of the Vahini pen ----------
  // Accelerometer, gyroscope and magnetometer axes oscillate while the hand
  // writes, so they draw as waves; tip force is one analog reading, so it
  // scrolls like a strip chart instead of repeating.
  // The OnHW pen's sensor groups: 13 channels in all.
  const CH = [
    ["Front accelerometer", "chart-1", "wave"], ["Rear accelerometer", "chart-2", "wave"], ["Gyroscope", "chart-3", "wave"],
    ["Magnetometer", "chart-1", "wave"], ["Pen-tip force", "warning", "level"],
  ];
  const bus = $("w-bus");
  // Colours go through the CSSOM: the page's CSP (style-src 'self') blocks
  // inline style attributes.
  bus.innerHTML = CH.map((c) => `<div class="w-chan"><em>${c[0]}</em>` +
    '<svg class="w-wave" viewBox="0 0 100 16" preserveAspectRatio="none"><polyline points="0,8 100,8"/></svg></div>').join("");
  [...bus.children].forEach((el, i) => el.style.setProperty("--ch", `var(--v-${CH[i][1]})`));
  const waveEls = [...bus.querySelectorAll("polyline")];
  const LEVEL_STEPS = 21;
  const levelHistory = CH.map((c) => (c[2] === "level" ? new Array(LEVEL_STEPS).fill(8) : null));
  let samples = 0;
  function wavePoints(speed, idx) {
    const amp = Math.min(6.5, 1 + speed * 0.42), pts = [];
    for (let x = 0; x <= 100; x += 5) pts.push(`${x},${(8 + Math.sin(x * 0.24 + samples * 0.14 + idx * 1.7) * amp).toFixed(1)}`);
    return pts.join(" ");
  }
  function levelPoints(speed, idx) {
    const hist = levelHistory[idx];
    hist.shift(); hist.push(8 - Math.min(6.5, 1 + speed * 0.42));
    return hist.map((y, i) => `${i * 5},${y.toFixed(1)}`).join(" ");
  }
  function pulseBus(speed) {
    waveEls.forEach((el, i) => el.setAttribute("points", CH[i][2] === "level" ? levelPoints(speed, i) : wavePoints(speed, i)));
  }
  function calmBus() {
    levelHistory.forEach((h) => { if (h) h.fill(8); });
    waveEls.forEach((el) => el.setAttribute("points", "0,8 100,8"));
  }

  // ---------- live waveform on the engine page while the pen moves ----------
  const WCH = [["chart-1", 1.0, 0], ["chart-2", 1.4, 1.1], ["chart-3", 0.8, 2.3], ["warning", 1.7, 3.6]].map(([t, f, p]) => [token(t), f, p]);
  let waveT = 0, waveAmp = 0, waveTarget = 0, waveRaf = null;
  function waveFrame() {
    waveRaf = requestAnimationFrame(waveFrame);
    waveT += 0.09; waveAmp += (waveTarget - waveAmp) * 0.12;
    const r = recon.getBoundingClientRect(), w = r.width, h = r.height;
    gR.clearRect(0, 0, recon.width, recon.height);
    gR.save();
    gR.strokeStyle = GRID; gR.lineWidth = 1;
    gR.beginPath(); gR.moveTo(0, h / 2); gR.lineTo(w, h / 2); gR.stroke();
    WCH.forEach((c, i) => {
      const amp = (9 + i * 3.5) * (0.3 + waveAmp);
      const midY = h / 2 + (i - (WCH.length - 1) / 2) * (h * 0.1);
      gR.beginPath(); gR.strokeStyle = c[0]; gR.lineWidth = 2; gR.lineCap = "round";
      gR.shadowColor = c[0]; gR.shadowBlur = 5;
      for (let x = 0; x <= w; x += 4) {
        const y = midY + Math.sin(x * 0.045 * c[1] + waveT * (1 + i * 0.12) + c[2]) * amp;
        if (x === 0) gR.moveTo(x, y); else gR.lineTo(x, y);
      }
      gR.stroke();
    });
    gR.restore();
  }
  function waveStart() { waitLay.classList.add("off"); if (!waveRaf && !reduce) waveFrame(); }
  function waveStop() { cancelAnimationFrame(waveRaf); waveRaf = null; waveT = 0; waveAmp = 0; waveTarget = 0; }

  // ---------- state ----------
  let strokes = [], cur = null, lastPt = null, drawing = false, sendTimer = null, replayRaf = null, rebuilt = false;
  // Every template is one unbroken shape, so after one stroke the demo has
  // what it needs; further strokes wait for Clear.
  let locked = false;
  const hint = $("w-hint"), waitLay = $("w-wait");
  const sampleEl = $("w-samples"), pktEl = $("w-pkts"), eqEl = $("w-eq"), ble = $("w-ble");
  const statusEl = $("w-status"), statusTx = $("w-status-tx");
  const verdict = $("w-verdict"), glyphEl = $("w-glyph"), guessEl = $("w-guess"), guessSub = $("w-guess-sub");
  const steps = [...root.querySelectorAll(".w-step")];
  const pen = $("w-pen"), pad = draw.parentNode;
  const setStep = (n) => steps.forEach((s, i) => s.classList.toggle("on", i === n));
  function setStatus(mode, text) { statusEl.className = `w-status${mode ? ` ${mode}` : ""}`; statusTx.textContent = text; }

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
  function pos(e) { const r = draw.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top, t: performance.now() }; }
  function line(g, a, b, w, col, glow) {
    g.strokeStyle = col; g.lineWidth = w; g.lineCap = "round"; g.lineJoin = "round";
    g.shadowColor = glow ? col : "transparent"; g.shadowBlur = glow ? 7 : 0;
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  }
  draw.addEventListener("pointerdown", (e) => {
    if (locked) return;
    e.preventDefault(); draw.setPointerCapture(e.pointerId);
    drawing = true; cur = [pos(e)]; lastPt = cur[0];
    hint.classList.add("off"); clearTimeout(sendTimer);
    setStep(0); setStatus("busy", "Capturing motion…");
    waveStart();
  });
  draw.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = pos(e), dt = Math.max(1, p.t - lastPt.t);
    const speed = Math.hypot(p.x - lastPt.x, p.y - lastPt.y) / dt * 16;
    samples += Math.max(1, Math.round(dt * 0.1)); // the OnHW pen samples at 100 Hz
    sampleEl.textContent = samples.toLocaleString();
    pulseBus(speed);
    waveTarget = Math.min(1, speed / 42);
    line(gD, lastPt, p, Math.max(2.1, 4.6 - Math.min(2.4, speed * 0.3)), INK, false);
    cur.push(p); lastPt = p;
  });
  function endStroke() {
    if (!drawing) return;
    drawing = false; calmBus(); waveTarget = 0.18;
    if (cur && cur.length > 2) strokes.push(cur);
    cur = null;
    clearTimeout(sendTimer);
    if (strokes.length) {
      locked = true; pad.classList.add("locked");
      setStatus("", "Character captured. Sending…");
      sendTimer = setTimeout(transmit, 1100);
    }
  }
  draw.addEventListener("pointerup", endStroke);
  draw.addEventListener("pointercancel", endStroke);
  draw.addEventListener("pointerleave", endStroke);
  $("w-go").addEventListener("click", () => { clearTimeout(sendTimer); transmit(); });
  function repaint() {
    gD.clearRect(0, 0, draw.width, draw.height);
    strokes.forEach((s) => { for (let j = 1; j < s.length; j++) line(gD, s[j - 1], s[j], 3.2, INK, false); });
  }

  // ---------- send → clean → rebuild → match ----------
  function transmit() {
    if (!strokes.length) return;
    setStep(0); setStatus("busy", "Sending the samples…");
    ble.classList.add("streaming"); verdict.classList.remove("on"); realBox.hidden = true;
    const total = Math.max(1, samples);
    let sent = 0;
    eqEl.textContent = `${samples.toLocaleString()} samples × 13 channels = ${(samples * 13).toLocaleString()} numbers`;
    const tick = setInterval(() => {
      sent = Math.min(total, sent + Math.max(1, Math.round(total / 14)));
      pktEl.textContent = sent;
      if (sent >= total) clearInterval(tick);
    }, reduce ? 8 : 55);
    setTimeout(() => {
      ble.classList.remove("streaming");
      setStep(1); setStatus("busy", "Guessing the shape…");
      setTimeout(reconstruct, reduce ? 40 : 620);
    }, reduce ? 60 : 950);
  }
  // The strokes scaled and centred on the engine screen.
  function rebuiltPoints() {
    const all = strokes.flat();
    const xs = all.map((p) => p.x), ys = all.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    // On wide screens the result box covers the bottom of the screen; keep
    // the path above it. On phones the box sits below the screen.
    const below = getComputedStyle(verdict).position === "absolute" ? 92 : 24;
    const rb = recon.getBoundingClientRect(), padding = 24;
    const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
    const s = Math.min((rb.width - padding * 2) / w, (rb.height - padding - below) / h, 2.2);
    const ox = (rb.width - w * s) / 2 - minX * s, oy = padding + (rb.height - padding - below - h * s) / 2 - minY * s;
    const pts = [];
    strokes.forEach((st, si) => st.forEach((p, pi) => pts.push({ x: p.x * s + ox, y: p.y * s + oy, brk: pi === 0 && si > 0 })));
    return pts;
  }
  function drawRebuilt() {
    const pts = rebuiltPoints();
    gR.clearRect(0, 0, recon.width, recon.height);
    for (let j = 1; j < pts.length; j++) if (!pts[j].brk) line(gR, pts[j - 1], pts[j], 3.2, GLOW, true);
  }
  function reconstruct() {
    waveStop();
    rebuilt = true;
    setStatus("busy", "Guessing the shape…");
    waitLay.classList.add("off");
    gR.clearRect(0, 0, recon.width, recon.height);
    const pts = rebuiltPoints();
    if (reduce) { drawRebuilt(); recognise(); return; }
    let n = 1;
    const per = Math.max(1, Math.round(pts.length / 52));
    cancelAnimationFrame(replayRaf);
    (function step() {
      for (let c = 0; c < per && n < pts.length; c++, n++) if (!pts[n].brk) line(gR, pts[n - 1], pts[n], 3.2, GLOW, true);
      if (n < pts.length) replayRaf = requestAnimationFrame(step); else recognise();
    })();
  }

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
    ["triangle", poly([P(0.5, 0.05), P(0.05, 0.92), P(0.95, 0.92), P(0.5, 0.05)]), true],
    ["star", poly([P(0.5, 0.02), P(0.38, 0.38), P(0.02, 0.38), P(0.31, 0.6), P(0.2, 0.98), P(0.5, 0.75), P(0.8, 0.98), P(0.69, 0.6), P(0.98, 0.38), P(0.62, 0.38), P(0.5, 0.02)]), true],
    ["heart", arc(0.32, 0.3, 0.2, Math.PI, 0).concat(arc(0.68, 0.3, 0.2, Math.PI, 0)).concat(poly([P(0.88, 0.4), P(0.5, 0.95), P(0.12, 0.4)])), true],
    ["check", poly([P(0.08, 0.55), P(0.35, 0.9), P(0.92, 0.1)])],
  ];
  const VEC = TEMPLATES.map((t) => ({ name: t[0], v: toVector(resample(t[1].slice())), closed: !!t[2] }));
  const NICE = { O: "the letter O", C: "the letter C", S: "the letter S", U: "the letter U", V: "the letter V",
    W: "the letter W", M: "the letter M", N: "the letter N", L: "the letter L", Z: "the letter Z",
    1: "the number 1", 2: "the number 2", 3: "the number 3", 7: "the number 7",
    triangle: "a triangle", star: "a star", heart: "a heart", check: "a check mark" };
  const GLYPH = { triangle: "△", star: "☆", heart: "♡", check: "✓" };
  const DIGITS = new Set(["1", "2", "3", "7"]);
  const LETTERS = new Set(["O", "C", "S", "U", "V", "W", "M", "N", "L", "Z"]);
  function recognise() {
    const flat = strokes.flat().map((p) => ({ x: p.x, y: p.y }));
    if (flat.length < 8) { setStatus("", "Waiting for a character…"); return; }
    const v = toVector(resample(flat));
    let best = null;
    VEC.forEach((t) => { const sc = bestMatch(v, t); if (!best || sc > best.sc) best = { name: t.name, sc }; });
    const pct100 = Math.round(Math.max(0, Math.min(0.999, best.sc)) * 100);
    verdict.classList.add("on");
    glyphEl.textContent = GLYPH[best.name] || best.name;
    const name = NICE[best.name] || best.name;
    guessEl.textContent = `${best.sc > 0.72 ? "Looks like" : "Best guess:"} ${name} · ${pct100}% shape match`;
    guessSub.textContent = "A shape guess from your drawing, not the AI.";
    if (DIGITS.has(best.name) || LETTERS.has(best.name)) {
      setStep(2); setStatus("busy", "Picking a real recording…");
      setTimeout(() => pickReal(best.name), reduce ? 0 : 450);
    } else {
      setStatus("done", "Not an OnHW character");
      drawn = null;
      realBox.hidden = false;
      realBox.replaceChildren(el("p", {}, `${name[0].toUpperCase()}${name.slice(1)} is not in the OnHW dataset, which holds letters, digits and maths symbols. Try O, S, Z or 7.`));
    }
  }

  // ---------- a real recording of the drawn character ----------
  // Uses app.js: state, current(), sample(), available(), currentProbs(),
  // classNames(), topK(), modelName(), whyLines(), radioGroup(), renderAll().
  const realBox = $("w-real");
  let drawn = null;
  function writerText(cur) {
    const who = state.protocol === "indep" ? "a new" : "a familiar";
    return `${who} ${cur.left ? "left-handed" : "right-handed"} writer`;
  }
  function pickReal(ch) {
    drawn = ch;
    state.task = DIGITS.has(ch) ? "symbols" : "chars";
    if (state.task === "symbols") state.hand = "right";
    if (state.task === "chars") state.protocol = "indep";
    if (!available(state.task, state.protocol)) state.protocol = "indep";
    renderOpts();
    const cur = current();
    let i = cur ? cur.samples.findIndex((s) => s.label === ch) : -1;
    if (cur && i < 0) i = cur.samples.findIndex((s) => s.label.toLowerCase() === ch.toLowerCase());
    if (i < 0) { showMissing(ch, cur); return; }
    state.sample = i;
    state.revealed = true;
    renderAll();
    showReal(ch);
  }
  function showReal(ch) {
    const cur = current(), s = sample(), names = classNames();
    const [a] = topK(currentProbs(), 1);
    const read = names[a[0]], correct = read === s.label;
    setStep(3); setStatus("done", "Read a real recording");
    realBox.hidden = false;
    const head = el("p", { class: "engine__real-head" });
    head.append(`Real OnHW recording of “${s.label}”, written by ${writerText(cur)}.`);
    if (s.label !== ch) head.append(` There is no “${ch}” in the exported samples, so this is its other case.`);
    const line = el("p", { class: "engine__real-read" });
    line.append(`${modelName()} read `, el("b", {}, read), ` · ${pct(a[1])} `,
      el("span", { class: `v-badge ${correct ? "v-badge--live" : "v-badge--dev"}` }, correct ? "right" : "wrong"));
    const why = el("ul", { class: "engine__why" });
    whyLines(correct).forEach((t) => why.append(el("li", {}, t)));
    const more = el("a", { href: "#workspace" }, "See its sensors and every score ↓");
    realBox.replaceChildren(head, line, why, more);
  }
  function showMissing(ch, cur) {
    setStatus("done", "No recording of that character");
    realBox.hidden = false;
    const p = el("p", {}, cur
      ? `No recording of “${ch}” by ${writerText(cur)} in the exported samples. Pick another hand or writer, or one of these:`
      : "Nothing exported for this choice yet.");
    const row = el("div", { class: "seg seg--small" });
    if (cur) {
      [...new Set(cur.samples.map((s) => s.label))].slice(0, 20).forEach((label) => {
        const b = el("button", { class: "v-chip", type: "button" }, label);
        b.addEventListener("click", () => pickReal(label));
        row.append(b);
      });
    }
    realBox.replaceChildren(p, row);
  }
  function redo() { if (drawn) pickReal(drawn); else renderOpts(); }
  function renderOpts() {
    const cur = current();
    const hand = $("w-opt-hand"), who = $("w-opt-who"), model = $("w-opt-model"), train = $("w-opt-train");
    if (state.task === "chars") {
      radioGroup(hand, [{ value: "right", label: "Right-handed" }, { value: "left", label: "Left-handed" }], state.hand,
        (v) => { state.hand = v; state.sample = 0; redo(); });
      who.replaceChildren(el("span", { class: "hint" }, "New people"));
    } else {
      hand.replaceChildren(el("span", { class: "hint" }, "Right-handed"));
      radioGroup(who, ["indep", "dep"].map((p) => ({ value: p, label: PEOPLE[p], disabled: !available(state.task, p) })),
        state.protocol, (v) => { state.protocol = v; state.sample = 0; redo(); });
    }
    if (cur && cur.mode === "ensemble") {
      radioGroup(model, [...[0, 1, 2, 3, 4].map((i) => ({ value: i, label: `#${i + 1}` })), { value: "mean", label: "×5 vote" }],
        state.model, (v) => { state.model = v; redo(); });
    } else {
      model.replaceChildren(el("span", { class: "hint" }, "One CNN-BiLSTM run"));
    }
    if (cur && cur.mode === "ensemble" && !cur.left) {
      radioGroup(train, [{ value: "right", label: "Right-handed people" }, { value: "with_left", label: "+ left-handed" }],
        state.group, (v) => { state.group = v; redo(); });
    } else {
      train.replaceChildren(el("span", { class: "hint" }, cur && cur.left ? "Right- and left-handed people" : "Right-handed people"));
    }
  }
  renderOpts();

  $("w-clear").addEventListener("click", () => {
    strokes = []; cur = null; samples = 0; locked = false; rebuilt = false; pad.classList.remove("locked");
    sampleEl.textContent = "0"; pktEl.textContent = "0"; eqEl.textContent = "";
    clearTimeout(sendTimer); cancelAnimationFrame(replayRaf); waveStop();
    gD.clearRect(0, 0, draw.width, draw.height); gR.clearRect(0, 0, recon.width, recon.height);
    hint.classList.remove("off"); waitLay.classList.remove("off");
    verdict.classList.remove("on"); ble.classList.remove("streaming");
    realBox.hidden = true; drawn = null;
    setStatus("", "Waiting for a character…"); setStep(-1); calmBus();
  });
  calmBus();
})();
