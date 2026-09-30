// Vahini playground: card 1, "How the pen works", way A: draw it.
//
// The mouse (or a finger) stands in for the sensor pen that recorded the
// Fraunhofer OnHW dataset. While you draw, five sensor groups animate (an
// animation, not data). Nothing is read until you press Recognize, so a
// character in several parts (t, ÷, E) or several characters (12) can take
// as long as they take; then shapes.js reads the strokes. The page then shows the model's
// answer for a REAL recording of that character from the OnHW test set
// (app.js), because the model reads pen motion and a drawing has none.
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
  const draw = $("w-draw"), pad = draw.parentNode, hint = $("w-hint"), shapeEl = $("w-shape"), readBtn = $("w-read");
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

  // ---------- the pen follows a mouse ----------
  // The Vahini pen (/site/design/v1/pen.svg, from the vahini-web component
  // library) when the host serves it, otherwise the small drawn one. Each
  // knows where its tip is, so the tip sits on the pointer.
  const penImg = $("w-pen-img"), penDrawn = $("w-pen");
  let pen = penDrawn, tip = { x: 6, y: 144 };
  function useImage() {
    if (!penImg.naturalWidth) return;
    penImg.hidden = false; penDrawn.remove(); pen = penImg;
    // pen.svg: viewBox -95 -650 200 740, tip at its origin.
    tip = { x: penImg.width * 95 / 200, y: penImg.height * 650 / 740 };
    pen.style.transformOrigin = `${tip.x}px ${tip.y}px`;
  }
  penImg.addEventListener("load", useImage);
  if (penImg.complete) useImage();
  if (window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
    pad.addEventListener("pointerenter", () => pen.classList.add("show"));
    pad.addEventListener("pointerleave", () => pen.classList.remove("show"));
    pad.addEventListener("pointermove", (e) => {
      const r = pad.getBoundingClientRect();
      pen.style.transform = `translate(${e.clientX - r.left - tip.x}px,${e.clientY - r.top - tip.y}px) rotate(${drawing ? 28 : 33}deg)`;
    });
  }

  // ---------- drawing ----------
  // Strokes collect until Recognize is pressed. Reading on a pause guessed
  // when the writer had finished, and guessed wrong between the parts of a
  // t or an E. After a drawing is read, the next stroke starts a new one;
  // after "not sure" it adds to the same drawing.
  let strokes = [], cur = null, lastPt = null, drawing = false, finished = false;
  const unread = () => strokes.length > 0 && !finished;
  function showReadable() { readBtn.disabled = !unread(); }
  function pos(e) { const r = draw.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top, t: performance.now() }; }
  function line(g, a, b, w) {
    g.strokeStyle = INK; g.lineWidth = w; g.lineCap = "round"; g.lineJoin = "round";
    g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
  }
  function dot(g, p) { g.fillStyle = INK; g.beginPath(); g.arc(p.x, p.y, 2.6, 0, Math.PI * 2); g.fill(); }
  function repaint() {
    gD.clearRect(0, 0, draw.width, draw.height);
    strokes.forEach((st) => {
      if (st.length === 1) dot(gD, st[0]);
      for (let j = 1; j < st.length; j++) line(gD, st[j - 1], st[j], 3.2);
    });
  }
  function clear() {
    strokes = []; cur = null; finished = false;
    gD.clearRect(0, 0, draw.width, draw.height); hint.classList.remove("off"); shapeEl.textContent = ""; calmBus();
    showReadable();
  }
  draw.addEventListener("pointerdown", (e) => {
    e.preventDefault(); draw.setPointerCapture(e.pointerId);
    if (finished) clear();
    startDrawing();
    drawing = true; cur = [pos(e)]; lastPt = cur[0];
    dot(gD, cur[0]);
    hint.classList.add("off");
    shapeEl.textContent = strokes.length ? "Next part…" : "Feeling the movement…";
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
    if (cur && cur.length) strokes.push(cur);   // a single point is a dot
    cur = null;
    if (!strokes.length) return;
    shapeEl.textContent = "Add another part, or press Recognize when you have finished.";
    showReadable();
    renderModel();   // the Recognize button in card 2 can read it now
  }
  draw.addEventListener("pointerup", endStroke);
  draw.addEventListener("pointercancel", endStroke);
  $("w-clear").addEventListener("click", () => { clear(); showPick(); });
  readBtn.addEventListener("click", () => { if (unread()) read(); });

  // ---------- strokes → a character → a real recording → Recognize ----------
  function read() {
    finished = true;
    showReadable();
    const task = state.task === "symbols" || state.task === "equations" ? "symbols" : "chars";
    const got = window.PlaygroundShapes.recognise(strokes, { size: draw.getBoundingClientRect().height, task });
    if (!got) { shapeEl.textContent = ""; return; }
    if (got.sequence) { readSequence(got.sequence, got.other); return; }
    if (!got.label) {
      // Keep the strokes: the next one adds to this drawing. Clear starts over.
      finished = false;
      showReadable();
      shapeEl.textContent = `${got.reason} Add a stroke, or press Clear.`;
      drawnNothing();
      return;
    }
    const moved = got.task !== state.task;
    const found = useDrawing(got);
    if (!found) {
      shapeEl.textContent = `Looks like ${got.shape}, but there is no recording of it here. Pick one below.`;
      return;
    }
    shapeEl.textContent = `Looks like ${got.shape}.${got.note ? ` ${got.note}` : ""}` +
      `${moved ? ` Switched to ${TASKS[got.task].label}.` : ""} The AI now reads a real pen recording of “${got.label}”.`;
    recognize();
  }

  // Several characters side by side ("12"). The recordings hold one
  // character each, so the AI reads them one at a time: the first straight
  // away, the others from a button each.
  // other: the same characters read the other way (letters for digits, or
  // digits for letters), offered as a button when some shape could be either.
  function readSequence(seq, other) {
    const shown = seq.map((c) => (c.label ? c.shape : "?")).join("");
    const pick = (i) => {
      const c = seq[i];
      const found = useDrawing(c);
      shapeEl.replaceChildren(
        `Looks like ${shown}: ${seq.length} characters. The AI reads one character at a time. ` +
        (found ? `Now: a real pen recording of “${c.label}”. ` : `There is no recording of “${c.label}” here. `),
      );
      const row = document.createElement("span");
      row.className = "w-seq";
      seq.forEach((ch, j) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "v-chip";
        b.textContent = ch.label ? ch.shape : "?";
        b.setAttribute("aria-pressed", String(j === i));
        b.setAttribute("aria-label", ch.label ? `Read character ${j + 1}, ${ch.shape}` : `Character ${j + 1}: not sure what it is`);
        if (!ch.label) b.disabled = true;
        else b.addEventListener("click", () => pick(j));
        row.append(b);
      });
      shapeEl.append(row);
      if (other) {
        const swap = document.createElement("button");
        swap.type = "button";
        swap.className = "linkbtn w-swap";
        swap.textContent = `Read as ${other.map((ch) => (ch.label ? ch.shape : "?")).join("")} instead`;
        swap.addEventListener("click", () => readSequence(other, seq));
        shapeEl.append(" ", swap);
      }
      if (found) recognize();
    };
    const first = seq.findIndex((c) => c.label);
    if (first < 0) {
      finished = false;
      showReadable();
      shapeEl.textContent = "Not sure about any of those characters. Add a stroke, or press Clear.";
      drawnNothing();
      return;
    }
    pick(first);
  }

  // app.js reads these: the Recognize button in card 2 reads an unread
  // drawing, and picking a recording clears the pad.
  window.PlaygroundDraw = {
    pending: unread,
    readNow: read,
    clear,
  };
  calmBus();
  showReadable();
})();
