// Vahini playground: the chart sections, "Compare the algorithms" and
// "How sure is the AI?". Data: PLAYGROUND_PUBLIC.comparison, .algorithms and
// .uncertainty, exported by scripts/build_playground.py from saved results.
// Uses el, svgEl and widthOf from app.js, which loads first.
"use strict";

(function () {
  const U = PUB.uncertainty;
  if (!U || !PUB.comparison) return;

  // One tooltip for every chart; marks carry their text in data-tip.
  const tip = el("div", { class: "uq-tip", role: "status", hidden: "" });
  document.body.append(tip);
  function bindTips(svg) {
    svg.addEventListener("pointermove", (e) => {
      const t = e.target.closest("[data-tip]");
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.getAttribute("data-tip");
      tip.hidden = false;
      const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
      tip.style.left = `${x}px`;
      tip.style.top = `${e.clientY + 14}px`;
    });
    svg.addEventListener("pointerleave", () => { tip.hidden = true; });
  }
  // Keyboard and touch users get the same text from focusable marks.
  function focusable(node, text) {
    node.setAttribute("data-tip", text);
    node.setAttribute("tabindex", "0");
    node.setAttribute("aria-label", text);
    return node;
  }
  const fmt = (v, d = 1) => Number(v).toFixed(d);

  // Plot frame: returns scale functions and draws the axes and grid.
  function frame(svg, W, H, m, xMax, yMin, yMax, xLabel, yLabel, xTicks, yTicks) {
    const x = (v) => m.l + (v / xMax) * (W - m.l - m.r);
    const y = (v) => H - m.b - ((v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
    for (const t of yTicks) {
      svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)" }));
      svg.append(svgEl("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, `${t}%`));
    }
    for (const t of xTicks) {
      svg.append(svgEl("text", { x: x(t), y: H - m.b + 16, "text-anchor": "middle" }, `${t}%`));
    }
    svg.append(svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, xLabel));
    svg.append(svgEl("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle",
      transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, yLabel));
    return { x, y };
  }
  function legend(box, items) {
    const row = el("div", { class: "uq-legend" });
    for (const [color, label] of items) {
      const k = el("span", { class: "uq-key" });
      k.style.background = color;
      const item = el("span", { class: "uq-item" });
      item.append(k, label);
      row.append(item);
    }
    box.append(row);
  }
  const SERIES = [["single", "var(--series-2)", "One run (seed 0)"], ["vote", "var(--series-1)", "5-run vote"]];

  // 1. Calibration: when it says 80% sure, is it right 80% of the time?
  function calibration() {
    const box = document.getElementById("uq-cal");
    box.replaceChildren();
    const W = widthOf("uq-cal", 1400), H = 280, m = { t: 12, r: 12, b: 40, l: 46 };
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Confidence against accuracy for one run and the 5-run vote" });
    const s = frame(svg, W, H, m, 100, 0, 100, "How sure it said it was", "How often it was right",
      [0, 25, 50, 75, 100], [0, 25, 50, 75, 100]);
    svg.append(svgEl("line", { x1: s.x(0), y1: s.y(0), x2: s.x(100), y2: s.y(100),
      stroke: "var(--muted)", "stroke-dasharray": "4 4" }));
    svg.append(svgEl("text", { x: s.x(62), y: s.y(70) - 6, transform: `rotate(-${Math.atan2(s.y(0) - s.y(100), s.x(100) - s.x(0)) * 180 / Math.PI} ${s.x(62)} ${s.y(70) - 6})` }, "perfectly honest"));
    for (const [key, color, name] of SERIES) {
      const bins = U.calibration[key].bins.filter((b) => b[2] >= 20);
      const d = bins.map((b, i) => `${i ? "L" : "M"}${s.x(b[0] * 100)} ${s.y(b[1] * 100)}`).join("");
      svg.append(svgEl("path", { d, fill: "none", stroke: color, "stroke-width": 2 }));
      for (const [c, a, n] of bins) {
        svg.append(focusable(svgEl("circle", { cx: s.x(c * 100), cy: s.y(a * 100), r: 5, fill: color,
          stroke: "var(--surface)", "stroke-width": 2 }),
          `${name}: said ${fmt(c * 100, 0)}% sure, right ${fmt(a * 100, 0)}% of the time (${n.toLocaleString()} letters)`));
      }
    }
    bindTips(svg);
    box.append(svg);
    legend(box, SERIES.map(([k, c, n]) => [c, `${n}: off by ${fmt(U.calibration[k].ece, 2)} points on average`]));
    const v = U.calibration.vote.bins.find((b) => b[0] > 0.65 && b[0] < 0.75);
    document.getElementById("uq-cal-note").textContent = v
      ? `The vote is too modest: when it says ${fmt(v[0] * 100, 0)}% sure it is right ${fmt(v[1] * 100, 0)}% of the time. Averaging five runs spreads the score over more letters.`
      : "";
  }

  // 2. Refusing the least sure answers.
  function coverage() {
    const box = document.getElementById("uq-cov");
    box.replaceChildren();
    const W = widthOf("uq-cov", 1400), H = 280, m = { t: 12, r: 12, b: 40, l: 46 };
    const all = [...U.coverage.single, ...U.coverage.vote].map((p) => p[1]);
    const yMin = Math.floor(Math.min(...all) / 10) * 10;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Accuracy when the model answers only its surest letters" });
    const ticks = [];
    for (let t = yMin; t <= 100; t += 10) ticks.push(t);
    const s = frame(svg, W, H, m, 100, yMin, 100, "Share of letters it answers (the rest: \"not sure\")",
      "Right, of those it answers", [10, 25, 50, 75, 100], ticks);
    for (const [key, color, name] of SERIES) {
      const pts = U.coverage[key];
      svg.append(svgEl("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${s.x(p[0])} ${s.y(p[1])}`).join(""),
        fill: "none", stroke: color, "stroke-width": 2 }));
      for (const [c, a] of pts) {
        svg.append(focusable(svgEl("circle", { cx: s.x(c), cy: s.y(a), r: 4, fill: color, stroke: "var(--surface)", "stroke-width": 2 }),
          `${name}: answering its surest ${c}%, right ${fmt(a)}%`));
      }
    }
    bindTips(svg);
    box.append(svg);
    legend(box, SERIES.map(([, c, n]) => [c, n]));
    const half = U.coverage.vote.find((p) => p[0] === 50);
    const full = U.coverage.vote.find((p) => p[0] === 100);
    document.getElementById("uq-cov-note").textContent = half && full
      ? `Answering every letter, the vote is right ${fmt(full[1])}% of the time. Answering only the half it is surest about, ${fmt(half[1])}%.`
      : "";
  }

  // 3. Per-letter uncertainty, split into its two kinds.
  function letters() {
    const box = document.getElementById("uq-letters");
    box.replaceChildren();
    const rows = [...U.letters].sort((a, b) => (b.ambiguous + b.disagree) - (a.ambiguous + a.disagree));
    const W = widthOf("uq-letters", 1400), H = 230, m = { t: 10, r: 8, b: 28, l: 40 };
    const max = Math.max(...rows.map((r) => r.ambiguous + r.disagree));
    const yMax = Math.ceil(max * 2) / 2;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Average uncertainty per letter, split into ambiguous and disagreement" });
    const step = (W - m.l - m.r) / rows.length;
    const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    for (let t = 0; t <= yMax + 1e-9; t += 0.5) {
      svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)" }));
      svg.append(svgEl("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, fmt(t, 1)));
    }
    svg.append(svgEl("text", { x: 10, y: (m.t + H - m.b) / 2, "text-anchor": "middle",
      transform: `rotate(-90 10 ${(m.t + H - m.b) / 2})` }, "bits"));
    const bw = Math.max(3, step - 2);
    rows.forEach((r, i) => {
      const x0 = m.l + i * step + (step - bw) / 2;
      const text = `${r.label}: ${fmt(r.accuracy)}% right over ${r.n} letters. Ambiguous ${fmt(r.ambiguous, 2)} bits, runs disagree ${fmt(r.disagree, 2)} bits`;
      const g = focusable(svgEl("g", {}), text);
      g.append(svgEl("rect", { x: x0, y: y(r.ambiguous), width: bw, height: Math.max(0, y(0) - y(r.ambiguous)), fill: "var(--series-1)" }));
      g.append(svgEl("rect", { x: x0, y: y(r.ambiguous + r.disagree), width: bw,
        height: Math.max(0, y(r.ambiguous) - y(r.ambiguous + r.disagree) - 1), fill: "var(--series-2)" }));
      svg.append(g);
      // On a phone the 52 labels do not fit; every other one, and the tooltip has all.
      if (step >= 9 || i % 2 === 0) {
        svg.append(svgEl("text", { x: x0 + bw / 2, y: H - m.b + 14, "text-anchor": "middle",
          class: step >= 16 ? "label-strong" : "" }, r.label));
      }
    });
    bindTips(svg);
    box.append(svg);
    legend(box, [["var(--series-1)", "The movement fits several letters (ambiguous)"],
      ["var(--series-2)", "The 5 runs disagree"]]);
    const top = rows.slice(0, 3).map((r) => `${r.label} (${fmt(r.accuracy, 0)}% right)`).join(", ");
    const last = rows[rows.length - 1];
    const amb = rows.reduce((t, r) => t + r.ambiguous * r.n, 0), dis = rows.reduce((t, r) => t + r.disagree * r.n, 0);
    document.getElementById("uq-letters-note").textContent =
      `Most doubt: ${top}. Least: ${last.label} (${fmt(last.accuracy, 0)}% right). ` +
      `${fmt(100 * amb / (amb + dis), 0)}% of the doubt is the movement fitting several letters; the rest is the runs disagreeing.`;
  }

  // 4. Confusion: which letter is read as which. Off-diagonal cells only.
  function confusion() {
    const note = drawConfusion("uq-conf", U.confusion, "the vote");
    document.getElementById("uq-conf-note").textContent = note;
  }
  function drawConfusion(id, matrix, who) {
    const box = document.getElementById(id);
    box.replaceChildren();
    const names = PUB.classes, C = names.length;
    const W = Math.min(widthOf(id, 620), 620), lab = 18, cell = (W - lab - 4) / C;
    const H = lab + cell * C + 4;
    let max = 0, total = 0;
    matrix.forEach((row, i) => row.forEach((v, j) => { if (i !== j) { max = Math.max(max, v); total += v; } }));
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "How often each letter was read as each other letter" });
    const every = cell >= 10 ? 1 : 2;
    names.forEach((n, i) => {
      if (i % every) return;
      svg.append(svgEl("text", { x: lab + i * cell + cell / 2, y: 12, "text-anchor": "middle", "font-size": 9 }, n));
      svg.append(svgEl("text", { x: lab - 4, y: lab + i * cell + cell / 2 + 3, "text-anchor": "end", "font-size": 9 }, n));
    });
    let casePairs = 0;
    matrix.forEach((row, i) => row.forEach((v, j) => {
      if (i === j || v === 0) return;
      const same = names[i].toLowerCase() === names[j].toLowerCase();
      if (same) casePairs += v;
      const r = svgEl("rect", { x: lab + j * cell, y: lab + i * cell, width: cell, height: cell,
        fill: same ? "var(--series-2)" : "var(--series-1)", opacity: Math.max(0.12, Math.sqrt(v / max)) });
      svg.append(focusable(r, `${names[i]} read as ${names[j]}: ${v} times${same ? " (same letter, other case)" : ""}`));
    }));
    bindTips(svg);
    box.append(svg);
    legend(box, [["var(--series-2)", "Same letter, other case"], ["var(--series-1)", "A different letter"]]);
    return `Rows: the letter written. Columns: what ${who} read. ${fmt(100 * casePairs / total, 1)}% of its ${total.toLocaleString()} mistakes are the same letter in the other case, the two orange lines.`;
  }

  // 5. Disagreement by writing hand.
  function hands() {
    const box = document.getElementById("uq-hands");
    const card = document.getElementById("uq-hands-card");
    if (!U.hands) { card.hidden = true; return; }
    box.replaceChildren();
    const h = U.hands, bins = h.edges.length - 1;
    const W = widthOf("uq-hands", 1400), H = 260, m = { t: 12, r: 12, b: 40, l: 46 };
    const maxShare = Math.max(...h.right.share, ...h.left.share);
    const yMax = Math.ceil(maxShare * 10) * 10;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "How much the 5 runs disagree, for right- and left-handed writers" });
    const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    for (let t = 0; t <= yMax; t += yMax / 4) {
      svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), stroke: "var(--grid)" }));
      svg.append(svgEl("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, `${fmt(t, 0)}%`));
    }
    const step = (W - m.l - m.r) / bins, bw = Math.max(2, step / 2 - 2);
    for (let b = 0; b < bins; b++) {
      [["right", "var(--series-1)", "right-handed"], ["left", "var(--series-2)", "left-handed"]].forEach(([k, color, who], s) => {
        const v = h[k].share[b] * 100;
        const x0 = m.l + b * step + 1 + s * (bw + 2);
        svg.append(focusable(svgEl("rect", { x: x0, y: y(v), width: bw, height: Math.max(0, y(0) - y(v)), fill: color, rx: 2 }),
          `${who}: ${fmt(v)}% of letters have disagreement ${h.edges[b]} to ${h.edges[b + 1]} bits`));
      });
      if (b % 3 === 0) svg.append(svgEl("text", { x: m.l + b * step, y: H - m.b + 16, "text-anchor": "middle" }, fmt(h.edges[b], 1)));
    }
    svg.append(svgEl("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, "How much the 5 runs disagree (bits)"));
    svg.append(svgEl("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle",
      transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, "Share of letters"));
    bindTips(svg);
    box.append(svg);
    legend(box, [["var(--series-1)", `Right-handed (${h.right.n.toLocaleString()} letters)`],
      ["var(--series-2)", `Left-handed (${h.left.n.toLocaleString()} letters)`]]);
    document.getElementById("uq-hands-note").textContent =
      `Typical disagreement: ${fmt(h.right.quartiles[1], 2)} bits for right-handed letters, ${fmt(h.left.quartiles[1], 2)} for left-handed ones. The runs learned mostly from right-handed people, so left-handed writing is less familiar to them.`;
  }

  // ---------- Compare the algorithms ----------
  const CMP = PUB.comparison;
  const TESTS = [
    ["both_indep", "Small + capital, new people"], ["both_dep", "Small + capital, familiar people"],
    ["lower_indep", "Small letters, new people"], ["lower_dep", "Small letters, familiar people"],
    ["upper_indep", "Capitals, new people"], ["upper_dep", "Capitals, familiar people"],
  ];
  let test = "both_indep";
  function hbars(id, rows, max, label) {
    const box = document.getElementById(id);
    box.replaceChildren();
    const W = widthOf(id, 1400), rowH = 30, nameW = Math.min(250, W * 0.42), valW = 60;
    const H = rows.length * rowH + 6;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label });
    const x = (v) => nameW + (v / max) * (W - nameW - valW);
    rows.forEach((r, i) => {
      const yMid = i * rowH + rowH / 2;
      svg.append(svgEl("text", { x: nameW - 8, y: yMid + 4, "text-anchor": "end", class: r.ours ? "label-strong" : "" }, r.name));
      svg.append(focusable(svgEl("rect", { x: nameW, y: yMid - 9, width: Math.max(2, x(r.value) - nameW), height: 18, rx: 4,
        fill: r.ours ? "var(--series-1)" : "var(--raw)" }), r.tip));
      svg.append(svgEl("text", { x: x(r.value) + 6, y: yMid + 4, class: "label-strong" }, `${fmt(r.value, 2)}%`));
    });
    bindTips(svg);
    box.append(svg);
  }
  function official() {
    radioGroup(document.getElementById("algo-tests"), TESTS.map(([v, l]) => ({ value: v, label: l })), test,
      (v) => { test = v; official(); });
    const short = (m) => (!m.ours ? m.name : /attn/.test(m.name) ? "CNN-BiLSTM + attention, tuned (ours)" : "CNN-BiLSTM (ours)");
    const rows = CMP.official.map((m) => ({ name: short(m), value: m.cells[test], ours: m.ours,
      tip: `${m.name}: ${fmt(m.cells[test], 2)}% (${m.source})` })).sort((a, b) => b.value - a.value);
    hbars("algo-official", rows, 100, "Test accuracy of each method on the chosen official test");
    const best = rows[0], pub = rows.filter((r) => !r.ours)[0];
    document.getElementById("algo-official-note").textContent =
      `${best.name} leads at ${fmt(best.value, 2)}%. Best published: ${pub.name}, ${fmt(pub.value, 2)}%. Grey bars are the paper's numbers; blue bars are runs in this repo (one seed, fold 0).`;
  }
  function leftGuessed() {
    const rows = CMP.left_guessed.map((m) => ({ name: m.name.replace("_", "-").toUpperCase(), value: m.test, ours: true,
      tip: `${m.name}: ${fmt(m.test, 1)}%, left-handed set, writers shared between training and test` }));
    hbars("algo-left", rows, 100, "Four designs on the left-handed set");
  }
  // Per-design runs (scripts/run_algorithms.sh): curves, calibration, confusion.
  const RUNS = PUB.algorithms || [];
  let run = 0;
  function runs() {
    const empty = document.getElementById("algo-runs-empty");
    const full = document.getElementById("algo-runs");
    empty.hidden = RUNS.length > 0;
    full.hidden = RUNS.length === 0;
    if (!RUNS.length) return;
    radioGroup(document.getElementById("algo-pick"), RUNS.map((r, i) => ({ value: i, label: r.name })), run,
      (v) => { run = v; runs(); });
    const table = document.getElementById("algo-table");
    table.replaceChildren();
    const head = el("tr");
    ["Design", "Parameters", "Training %", "Test %", "Calibration gap", "Case mix-ups"].forEach((h) => head.append(el("th", { scope: "col" }, h)));
    table.append(el("thead", {}), el("tbody"));
    table.querySelector("thead").append(head);
    [...RUNS].sort((a, b) => b.test - a.test).forEach((r) => {
      const tr = el("tr");
      [r.name, r.params ? r.params.toLocaleString() : "", r.train !== null ? fmt(r.train, 1) : "", fmt(r.test, 2),
        `${fmt(r.ece, 2)} pts`, `${fmt(r.case_share, 0)}% of errors`].forEach((v) => tr.append(el("td", {}, v)));
      table.querySelector("tbody").append(tr);
    });
    const r = RUNS[run];
    curves(r);
    const box = document.getElementById("algo-cal");
    box.replaceChildren();
    calibrationInto("algo-cal", [["x", "var(--series-1)", r.name, r.calibration, r.ece]]);
    document.getElementById("algo-conf-note").textContent = drawConfusion("algo-conf", r.confusion, r.name);
  }
  function curves(r) {
    const box = document.getElementById("algo-curves");
    box.replaceChildren();
    if (!r.history) { box.append(el("p", { class: "note" }, "This run did not save its training history.")); return; }
    const tr = r.history.accuracy.map((v) => v * 100), va = r.history.val_accuracy.map((v) => v * 100);
    const n = tr.length, W = widthOf("algo-curves", 1400), H = 260, m = { t: 12, r: 12, b: 40, l: 46 };
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Training curves for ${r.name}` });
    const s = frame(svg, W, H, m, n, 0, 100, "Training round (epoch)", "Right %", [], [0, 25, 50, 75, 100]);
    for (let e = 0; e <= n; e += Math.max(1, Math.round(n / 6))) svg.append(svgEl("text", { x: s.x(e), y: H - m.b + 16, "text-anchor": "middle" }, String(e)));
    [[tr, "var(--series-2)", "training letters"], [va, "var(--series-1)", "held-back letters (validation)"]].forEach(([vals, color, name]) => {
      svg.append(svgEl("path", { d: vals.map((v, i) => `${i ? "L" : "M"}${s.x(i + 1)} ${s.y(v)}`).join(""), fill: "none", stroke: color, "stroke-width": 2 }));
      vals.forEach((v, i) => svg.append(focusable(svgEl("circle", { cx: s.x(i + 1), cy: s.y(v), r: 3, fill: color }), `Round ${i + 1}: ${fmt(v)}% right on ${name}`)));
    });
    bindTips(svg);
    box.append(svg);
    legend(box, [["var(--series-2)", "Training letters"], ["var(--series-1)", "Held-back letters"]]);
    const gap = tr[n - 1] - va[n - 1];
    document.getElementById("algo-curves-note").textContent =
      `After ${n} rounds: ${fmt(tr[n - 1])}% on training letters, ${fmt(va[n - 1])}% on held-back letters. A gap of ${fmt(gap)} points is how much it memorised.`;
  }
  function calibrationInto(id, series) {
    const box = document.getElementById(id);
    const W = widthOf(id, 1400), H = 260, m = { t: 12, r: 12, b: 40, l: 46 };
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Confidence against accuracy" });
    const s = frame(svg, W, H, m, 100, 0, 100, "How sure it said it was", "How often it was right", [0, 25, 50, 75, 100], [0, 25, 50, 75, 100]);
    svg.append(svgEl("line", { x1: s.x(0), y1: s.y(0), x2: s.x(100), y2: s.y(100), stroke: "var(--muted)", "stroke-dasharray": "4 4" }));
    for (const [, color, name, binsAll, ece] of series) {
      const bins = binsAll.filter((b) => b[2] >= 20);
      svg.append(svgEl("path", { d: bins.map((b, i) => `${i ? "L" : "M"}${s.x(b[0] * 100)} ${s.y(b[1] * 100)}`).join(""), fill: "none", stroke: color, "stroke-width": 2 }));
      bins.forEach(([c, a, n]) => svg.append(focusable(svgEl("circle", { cx: s.x(c * 100), cy: s.y(a * 100), r: 5, fill: color, stroke: "var(--surface)", "stroke-width": 2 }),
        `${name}: said ${fmt(c * 100, 0)}% sure, right ${fmt(a * 100, 0)}% (${n} letters)`)));
      legend(box, [[color, `${name}: off by ${fmt(ece, 2)} points on average`]]);
    }
    bindTips(svg);
    box.prepend(svg);
  }

  function renderAllCharts() { calibration(); coverage(); letters(); confusion(); hands(); official(); leftGuessed(); runs(); }
  renderAllCharts();
  let timer;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(renderAllCharts, 150); });
})();
