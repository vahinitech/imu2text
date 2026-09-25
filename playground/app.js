// Vahini playground (imu2text). Plain JavaScript, no dependencies, no build step.
// Data comes from scripts/build_playground.py; stages from stages.js.
"use strict";

const PUB = window.PLAYGROUND_PUBLIC;
const LOCAL = window.PLAYGROUND_LOCAL || null;
const STAGES = window.PLAYGROUND_STAGES;
const SVG_NS = "http://www.w3.org/2000/svg";
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------- words shown to people (plain English; the glossary explains) ----------
const TASKS = {
  chars: { label: "Letters", note: "One letter per recording: A to Z and a to z." },
  symbols: { label: "Numbers and symbols", note: "One per recording: 0 to 9 and + - · : =" },
  equations: { label: "Maths symbols", note: "Numbers and signs cut out of handwritten sums." },
  words: { label: "Words", note: "Whole German words, written in one go." },
};
const KIND_TITLES = {
  clear: "Easy: the model is sure and right",
  case: "Tricky: small letter or capital?",
  disagree: "Tricky: the 5 models disagree",
  unsure: "Tricky: the model is unsure",
  wrong: "Fooled: sure, but wrong",
  right: "Read correctly",
  fixed: "Fixed by the word list",
  abstain: "No word in the list fits",
};
const WORD_KINDS = ["right", "fixed", "abstain", "wrong"];
const WORD_TITLES = { wrong: "Read wrong" };
const PEOPLE = { indep: "New people", dep: "Familiar people" };

const SENSOR_PANELS = [
  { title: "Front accelerometer", channels: [0, 1, 2] },
  { title: "Rear accelerometer", channels: [3, 4, 5] },
  { title: "Gyroscope", channels: [6, 7, 8] },
  { title: "Magnetometer", channels: [9, 10, 11] },
  { title: "Pen-tip force", channels: [12] },
];
const AXIS_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const AXIS_NAMES = ["x", "y", "z"];

const state = {
  task: "chars", protocol: "indep", hand: "right", sample: 0,
  filter: "none", model: "mean", group: "right", t: null,
  revealed: false, busy: false,
};
const score = { tried: 0, right: 0, seen: new Set() };

// What the current task, people and hand select. `mode` decides how step 3
// draws: an ensemble with its members, a single model, or words.
function current() {
  if (state.task === "words") {
    const w = PUB.words;
    return w ? { mode: "words", samples: w.samples, source: w } : null;
  }
  if (state.task === "chars" && state.protocol === "indep") {
    if (state.hand === "left") {
      return { mode: "ensemble", samples: PUB.left_letters, summary: PUB.left_summary, left: true };
    }
    return { mode: "ensemble", samples: PUB.letters, summary: null, left: false };
  }
  const run = PUB.tasks[`${state.task}_${state.protocol}`];
  return run ? { mode: "single", samples: run.samples, run } : null;
}
function available(task, protocol) {
  if (task === "words") return Boolean(PUB.words);
  if (task === "chars" && protocol === "indep") return true;
  return Boolean(PUB.tasks[`${task}_${protocol}`]);
}
function sample() {
  const cur = current();
  return cur ? cur.samples[Math.min(state.sample, cur.samples.length - 1)] : null;
}

// ---------- small helpers ----------
function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function repoLink(path, label) {
  return el("a", { href: `${STAGES.repo}/blob/main/${path}`, target: "_blank", rel: "noopener" }, label || path);
}
function termLink(id, label) {
  return el("a", { href: `#term-${id}`, class: "term" }, label);
}
function radioGroup(container, options, current, onPick) {
  container.replaceChildren();
  for (const opt of options) {
    const b = el("button", { role: "radio", "aria-checked": String(opt.value === current) }, opt.label);
    if (opt.disabled) { b.disabled = true; b.title = opt.title || ""; }
    b.addEventListener("click", () => onPick(opt.value));
    container.append(b);
  }
}
function pct(x) { return `${(x * 100).toFixed(1)}%`; }
// Charts are drawn at their container's width, so labels keep their size on
// a phone instead of shrinking with a scaled-down drawing.
function widthOf(id, max) {
  const w = document.getElementById(id).clientWidth;
  return Math.max(260, Math.min(max, w || max));
}
function show(id, visible) { document.getElementById(id).hidden = !visible; }

// The selection lives in the URL (#task=chars&sample=3&filter=lowpass...), so
// a view can be shared in an issue or a chat.
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (TASKS[h.get("task")]) state.task = h.get("task");
  if (["indep", "dep"].includes(h.get("protocol"))) state.protocol = h.get("protocol");
  if (["right", "left"].includes(h.get("hand"))) state.hand = h.get("hand");
  const idx = Number(h.get("sample") ?? h.get("letter"));
  if (Number.isInteger(idx) && idx >= 0) state.sample = idx;
  if (STAGES.filters.some((f) => f.id === h.get("filter"))) state.filter = h.get("filter");
  const model = h.get("model");
  if (model === "mean") state.model = "mean";
  else if (/^[0-9]$/.test(model || "")) state.model = Number(model);
  if (PUB.groups[h.get("group")]) state.group = h.get("group");
  if (h.get("show") === "1") state.revealed = true;
  if (!available(state.task, state.protocol)) state.protocol = "indep";
}
function writeHash() {
  const h = new URLSearchParams({
    task: state.task, protocol: state.protocol, hand: state.hand, sample: state.sample,
    filter: state.filter, model: state.model, group: state.group,
  });
  if (state.revealed) h.set("show", "1");
  history.replaceState(null, "", `#${h}`);
}

// A new choice hides the old answer, so every sample is a fresh try.
function choose(update) {
  update();
  state.revealed = false;
  renderAll();
}

// ---------- step 1: task and sample ----------
function renderTask() {
  radioGroup(document.getElementById("task"),
    Object.entries(TASKS).map(([id, t]) => ({ value: id, label: t.label })), state.task,
    (v) => choose(() => {
      state.task = v; state.sample = 0;
      if (!available(v, state.protocol)) state.protocol = "indep";
    }));
  document.getElementById("task-note").textContent = TASKS[state.task].note;

  const protoBox = document.getElementById("protocol");
  if (state.task === "words") {
    protoBox.replaceChildren(el("span", { class: "hint" }, "New people only."));
  } else {
    radioGroup(protoBox, ["indep", "dep"].map((p) => ({
      value: p, label: PEOPLE[p], disabled: !available(state.task, p), title: "not run yet",
    })), state.protocol, (v) => choose(() => { state.protocol = v; state.sample = 0; }));
  }
  const handBox = document.getElementById("hand");
  if (state.task === "chars" && state.protocol === "indep" && PUB.left_letters.length) {
    radioGroup(handBox, [
      { value: "right", label: "Right-handed" },
      { value: "left", label: "Left-handed" },
    ], state.hand, (v) => choose(() => { state.hand = v; state.sample = 0; }));
  } else {
    handBox.replaceChildren();
  }

  const box = document.getElementById("samples");
  box.replaceChildren();
  const cur = current();
  if (!cur) {
    box.append(el("p", { class: "note" }, "Not run yet. This is one of the open tasks below."));
    return;
  }
  const order = cur.mode === "words" ? WORD_KINDS : Object.keys(KIND_TITLES);
  for (const kind of order) {
    const items = cur.samples.map((s, i) => [s, i]).filter(([s]) => s.kind === kind);
    if (!items.length) continue;
    const group = el("div", { class: "letter-group" });
    const title = (cur.mode === "words" && WORD_TITLES[kind]) || KIND_TITLES[kind];
    group.append(el("h3", {}, title));
    const row = el("div", { class: "seg", role: "radiogroup", "aria-label": title });
    for (const [s, i] of items) {
      const text = cur.mode === "words" ? s.ref : s.label;
      const b = el("button", {
        class: cur.mode === "words" ? "word-btn" : "letter-btn", role: "radio",
        "aria-checked": String(i === state.sample),
      }, text);
      b.addEventListener("click", () => choose(() => { state.sample = i; }));
      row.append(b);
    }
    group.append(row);
    box.append(group);
  }
  renderDev("dev-pick");
}

// ---------- step 2: signal and filter ----------
function currentSignal() {
  const s = sample();
  const real = LOCAL && state.task === "chars" && state.hand === "right" && state.protocol === "indep" &&
    s && LOCAL[String(s.test_index)];
  const source = real || PUB.synthetic_signal;
  return { raw: source.none, filtered: source[state.filter], real: Boolean(real) };
}

let crosshairs = [];
function drawPanel(panel, raw, filtered, showRaw) {
  const W = widthOf("signal", 900), H = 84, left = 4, right = 22, top = 16, bottom = 4;
  // Each line is drawn around its own mean: gravity puts one accelerometer
  // axis about 16,000 counts from the others, which would flatten their
  // shapes on a shared scale. Exact values are in the raw-values table.
  const centre = (vals, m) => vals.map((v) => v - m);
  const series = panel.channels.map((c) => {
    const m = raw[c].reduce((acc, v) => acc + v, 0) / raw[c].length;
    return { raw: centre(raw[c], m), filt: centre(filtered[c], m) };
  });
  const all = series.flatMap((s) => (showRaw ? s.raw.concat(s.filt) : s.filt));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { hi += 1; lo -= 1; }
  const n = series[0].filt.length;
  const x = (i) => left + (i / Math.max(n - 1, 1)) * (W - left - right);
  const y = (v) => top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom);
  const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${panel.title} signal` });
  svg.append(svgEl("text", { x: left, y: 11, class: "label-strong" }, panel.title));
  svg.append(svgEl("line", { x1: left, x2: W - right, y1: H - bottom, y2: H - bottom, stroke: "var(--grid)" }));
  series.forEach((s, k) => {
    const color = panel.channels.length === 1 ? "var(--series-1)" : AXIS_COLORS[k];
    if (showRaw) svg.append(svgEl("path", { d: path(s.raw), fill: "none", stroke: "var(--raw)", "stroke-width": 1 }));
    svg.append(svgEl("path", { d: path(s.filt), fill: "none", stroke: color, "stroke-width": 1.6 }));
  });
  // Crosshair for the selected moment; moving over any plot selects a time.
  const cross = svgEl("line", { y1: top - 4, y2: H - bottom, stroke: "var(--text-muted)",
    "stroke-width": 1, "stroke-dasharray": "2 2", visibility: "hidden" });
  svg.append(cross);
  const pick = (clientX) => {
    const box = svg.getBoundingClientRect();
    const vx = ((clientX - box.left) / box.width) * W;
    const i = Math.round(((vx - left) / (W - left - right)) * (n - 1));
    setTime(Math.max(0, Math.min(n - 1, i)));
  };
  svg.addEventListener("pointermove", (e) => pick(e.clientX));
  svg.addEventListener("pointerdown", (e) => pick(e.clientX));
  crosshairs.push((i) => {
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
  });
  if (panel.channels.length > 1) {
    // End labels, pushed apart so they never overlap.
    const ends = series.map((s, k) => ({ k, y: y(s.filt[n - 1]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 11);
    for (const e of ends) svg.append(svgEl("text", { x: W - right + 6, y: e.y + 4 }, AXIS_NAMES[e.k]));
  }
  return svg;
}

// Conversions documented in imu2text/filters.py. The accelerometer one is
// derived from the data (a pen at rest reads 1 g); the gyroscope one is
// inferred from plausible pen rotation, not from a datasheet.
const RAW_ROWS = [
  ["Front accelerometer", [0, 1, 2]], ["Rear accelerometer", [3, 4, 5]],
  ["Gyroscope", [6, 7, 8]], ["Magnetometer", [9, 10, 11]], ["Pen-tip force", [12]],
];
function physical(channel, counts) {
  if (channel <= 5) return `${(counts / 16384).toFixed(3)} g`;
  if (channel <= 8) return `${(counts / 14.3).toFixed(1)} °/s`;
  return "";
}
function channelStats(vals) {
  const sum = vals.reduce((acc, v) => acc + v, 0);
  return { min: Math.min(...vals), max: Math.max(...vals), mean: sum / vals.length };
}
function fmt(v) { return Number(v).toLocaleString(undefined, { maximumFractionDigits: 1 }); }

function renderRawTable() {
  const data = currentSignal().filtered;
  const n = data[0].length;
  const t = Math.min(state.t, n - 1);
  const head = el("thead");
  const hr = el("tr");
  for (const h of ["Sensor", "Axis", "Counts", "Units", "Min", "Max", "Mean"]) hr.append(el("th", { scope: "col" }, h));
  head.append(hr);
  const body = el("tbody");
  for (const [name, chans] of RAW_ROWS) {
    chans.forEach((c, k) => {
      const st = channelStats(data[c]);
      const tr = el("tr");
      tr.append(el("td", {}, k === 0 ? name : ""), el("td", {}, chans.length > 1 ? AXIS_NAMES[k] : ""),
        el("td", { class: "now" }, fmt(data[c][t])), el("td", {}, physical(c, data[c][t])),
        el("td", {}, fmt(st.min)), el("td", {}, fmt(st.max)), el("td", {}, fmt(st.mean)));
      body.append(tr);
    });
  }
  document.getElementById("raw-table").replaceChildren(head, body);
  document.getElementById("time-readout").textContent =
    `${(t / PUB.sample_rate_hz).toFixed(2)} s · sample ${t + 1} of ${n}`;
  document.getElementById("time-readout-short").textContent = `at ${(t / PUB.sample_rate_hz).toFixed(2)} s`;
}
function setTime(i) {
  state.t = i;
  document.getElementById("time").value = String(i);
  for (const draw of crosshairs) draw(i);
  renderRawTable();
}
function downloadCsv() {
  const sig = currentSignal();
  const data = sig.filtered;
  const names = PUB.channels;
  const lines = [["time_s", ...names].join(",")];
  for (let i = 0; i < data[0].length; i++) {
    lines.push([(i / PUB.sample_rate_hz).toFixed(2), ...names.map((_, c) => data[c][i])].join(","));
  }
  const s = sample();
  const kind = sig.real ? `onhw-test${s.test_index}` : "synthetic";
  const a = el("a", {
    href: URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" })),
    download: `vahini-playground-${kind}-${state.filter}.csv`,
  });
  document.body.append(a);
  a.click();
  a.remove();
}

function renderSignal() {
  const sig = currentSignal();
  crosshairs = [];
  const note = document.getElementById("signal-note");
  if (sig.real) note.textContent = "";
  else if (LOCAL) {
    note.textContent = "Practice signal: real recordings are loaded only for right-handed letters, " +
      "so this sample shows a made-up one.";
  } else {
    note.textContent = "Practice signal: the real recordings belong to Fraunhofer IIS and are not " +
      "shared on this page, so you see a made-up pen signal. Developers can load the real ones " +
      "locally (see the developer notes below).";
  }
  radioGroup(document.getElementById("filters"),
    STAGES.filters.map((f) => ({ value: f.id, label: f.name })), state.filter,
    (v) => { state.filter = v; renderSignal(); renderPipeline(); writeHash(); });

  const box = document.getElementById("signal");
  box.replaceChildren();
  const showRaw = state.filter !== "none";
  for (const panel of SENSOR_PANELS) box.append(drawPanel(panel, sig.raw, sig.filtered, showRaw));
  box.append(el("p", { class: "hint" }, (showRaw ? "Grey: before cleaning. Colour: after. " : "") +
    "Lines are centred so you can see their shape; exact numbers are under Raw sensor values."));
  const n = sig.filtered[0].length;
  document.getElementById("time").max = String(n - 1);
  setTime(Math.min(state.t ?? Math.floor(n / 2), n - 1));

  const f = STAGES.filters.find((s) => s.id === state.filter);
  const card = document.getElementById("filter-result");
  card.replaceChildren(
    el("div", {}, "A letter model trained on signals cleaned this way reads"),
    el("div", { class: "big" }, `${f.accuracy.toFixed(2)}% correctly`),
    el("p", {}, f.note),
  );
  const src = el("p", { class: "hint" }, `${f.conditions}. Source: `);
  src.append(repoLink(f.source));
  card.append(src);
  renderDev("dev-signal");
}

// ---------- step 3: the model's answer ----------
function topK(probs, k) {
  return probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, k);
}
function classNames() {
  const cur = current();
  return cur.mode === "single" ? cur.run.classes : PUB.classes;
}
function groupData() {
  const s = sample();
  if (current().left) return s.with_left;
  return s[state.group] || s.right;
}
function currentProbs() {
  const cur = current();
  if (cur.mode === "single") return sample().probs;
  const g = groupData();
  return state.model === "mean" ? g.mean : g.members[state.model];
}
function modelName() {
  const cur = current();
  if (!cur) return "none";
  if (cur.mode === "words") return "Word reader";
  if (cur.mode === "single") return "One model";
  return state.model === "mean" ? "All 5 together" : `Model ${state.model + 1}`;
}
// Bars start at zero width or height and grow when revealed (CSS transition).
function grow(nodes) {
  if (REDUCED_MOTION) { nodes.forEach((n) => n.classList.add("grown")); return; }
  requestAnimationFrame(() => requestAnimationFrame(() => nodes.forEach((n) => n.classList.add("grown"))));
}

function renderBars() {
  const s = sample();
  const names = classNames();
  const top = topK(currentProbs(), 5);
  const W = widthOf("bars", 460), rowH = 30, labelW = 34, valueW = 110;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${rowH * top.length}`, role: "img",
    "aria-label": "The five most likely answers with the model's confidence" });
  const bars = [];
  top.forEach(([cls, p], r) => {
    const name = names[cls];
    const yMid = r * rowH + rowH / 2;
    const barW = Math.max(2, p * (W - labelW - valueW));
    svg.append(svgEl("text", { x: 4, y: yMid + 5, class: "label-strong" }, name));
    const bar = svgEl("rect", { x: labelW, y: yMid - 9, width: barW, height: 18, rx: 4, class: "grow-x",
      fill: name === s.label ? "var(--series-3)" : "var(--series-1)" });
    bars.push(bar);
    svg.append(bar);
    svg.append(svgEl("text", { x: labelW + barW + 6, y: yMid + 4 }, name === s.label ? `${pct(p)}  correct` : pct(p)));
  });
  document.getElementById("bars").replaceChildren(svg);
  grow(bars);
}

function renderMembers() {
  const g = groupData();
  const top = topK(g.mean, 3).map(([cls]) => cls);
  const S = g.members.length, W = widthOf("members", 460), H = 150, base = 120, barW = 10, gap = 3;
  const groupW = S * (barW + gap);
  const groupGap = Math.max(16, Math.min(60, (W - 40 - 3 * groupW) / 2));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "What each of the five models answered for the three most likely letters" });
  svg.append(svgEl("line", { x1: 0, x2: W, y1: base, y2: base, stroke: "var(--grid)" }));
  const bars = [];
  top.forEach((cls, gi) => {
    const x0 = 20 + gi * (groupW + groupGap);
    g.members.forEach((probs, s) => {
      const h = probs[cls] * (base - 10);
      const selected = state.model === s || state.model === "mean";
      const bar = svgEl("rect", { x: x0 + s * (barW + gap), y: base - h, width: barW, height: Math.max(h, 1),
        rx: 2, fill: "var(--series-1)", opacity: selected ? 1 : 0.35, class: "grow-y" });
      bars.push(bar);
      svg.append(bar);
    });
    const meanY = base - g.mean[cls] * (base - 10);
    svg.append(svgEl("line", { x1: x0 - 3, x2: x0 + groupW, y1: meanY, y2: meanY,
      stroke: "var(--text-primary)", "stroke-dasharray": "3 2" }));
    svg.append(svgEl("text", { x: x0 + groupW / 2 - 4, y: base + 18, class: "label-strong" }, PUB.classes[cls]));
  });
  document.getElementById("members").replaceChildren(svg);
  grow(bars);
}

// Uncertainty split used by the reading (and shown in the developer notes):
// total = entropy of the average; disagreement = mutual information.
function uncertainty() {
  const g = groupData();
  const entropy = (p) => -p.reduce((acc, v) => acc + (v > 0 ? v * Math.log2(v) : 0), 0);
  const total = entropy(g.mean);
  const own = g.members.reduce((acc, p) => acc + entropy(p), 0) / g.members.length;
  return { total, own, disagreement: Math.max(0, total - own) };
}

function reading() {
  const cur = current();
  const s = sample();
  const box = document.getElementById("members-read");
  box.replaceChildren();
  const probs = currentProbs();
  const names = classNames();
  const [a, b] = topK(probs, 2);
  const first = names[a[0]], second = names[b[0]];
  const add = (...parts) => parts.forEach((p) => box.append(typeof p === "string" ? document.createTextNode(p) : p));
  if (cur.mode === "ensemble") {
    const u = uncertainty();
    const caseSplit = first.toLowerCase() === second.toLowerCase();
    // Display thresholds for choosing a sentence, not measured results.
    if (u.disagreement >= 0.5) {
      add("The 5 models disagree: each reads something different. This person's writing is unlike what they learned from. ");
    } else if (a[1] >= 0.7) {
      add(`All 5 models agree on ${first}. `);
    } else if (caseSplit) {
      add(`Every model hesitates between ${first} and ${second}: the same shape at a different size, and the pen barely feels size. `);
    } else {
      add(`Every model hesitates between ${first} and ${second}: the pen movement fits both. `);
    }
    add("See ", termLink("uncertainty", "uncertainty"), ".");
  } else if (a[1] < 0.5) {
    add(`The model is unsure: its best guess, ${first}, gets only ${pct(a[1])}. `);
  } else {
    add(`The model is ${pct(a[1])} sure it is ${first}. `);
  }
  return first === s.label;
}

function renderWords() {
  const s = sample();
  const table = el("table", { class: "words-table" });
  const verdict = (hyp) => (hyp === s.ref ? "right" : hyp === "" ? "no answer" : "wrong");
  for (const [name, hyp, cer] of [
    ["Written", s.ref, null],
    ["Read letter by letter", s.greedy, s.cer_greedy],
    ["Read with the word list", s.lexicon, s.cer_lexicon],
  ]) {
    const tr = el("tr");
    tr.append(el("th", { scope: "row" }, name), el("td", { class: "word" }, hyp || "(no answer)"),
      el("td", {}, cer === null ? "" : `${verdict(hyp)} · ${cer.toFixed(0)}% of letters wrong`));
    table.append(tr);
  }
  const box = document.getElementById("words-view");
  box.replaceChildren(table);
  const note = el("p", { class: "hint" });
  note.append("The ", termLink("word-list", "word list"), " holds the 501 words seen in training. ",
    "Letters wrong is the ", termLink("cer", "character error rate"), ".");
  box.append(note);
  return s.lexicon === s.ref;
}

function resultCard() {
  const cur = current();
  const card = document.getElementById("model-result");
  if (cur.mode === "words") {
    const w = PUB.words;
    card.replaceChildren(
      el("div", {}, `Over all ${w.n.toLocaleString()} test words from new people, the word list reads`),
      el("div", { class: "big" }, `${w.exact_lexicon.toFixed(2)}% exactly right`),
      el("div", {}, `Letter by letter: ${w.exact_greedy.toFixed(2)}% exactly right. This model is still ` +
        "under-trained (15 rounds), which is why words are harder than letters here."),
    );
    return;
  }
  if (cur.mode === "single") {
    const run = cur.run;
    card.replaceChildren(
      el("div", {}, `Over all ${run.n_test.toLocaleString()} test samples from ${PEOPLE[state.protocol].toLowerCase()}, one model reads`),
      el("div", { class: "big" }, `${run.accuracy.toFixed(2)}% correctly`),
    );
    return;
  }
  const summary = cur.left ? cur.summary : (PUB.groups[state.group] || PUB.groups.right);
  const acc = state.model === "mean" ? summary.ensemble_accuracy : summary.member_accuracy[state.model];
  const who = cur.left ? "left-handed" : "right-handed";
  card.replaceChildren(
    el("div", {}, `Over all ${summary.n_test.toLocaleString()} letters from new ${who} people, ${modelName().toLowerCase()} reads`),
    el("div", { class: "big" }, `${acc.toFixed(2)}% correctly`),
  );
}

function renderScore() {
  const box = document.getElementById("score");
  box.textContent = score.tried
    ? `This visit: the model got ${score.right} of your ${score.tried} tries right.`
    : "";
}

function renderModel() {
  const cur = current();
  const intro = document.getElementById("model-intro");
  const models = document.getElementById("models");
  const training = document.getElementById("training");
  models.replaceChildren();
  training.replaceChildren();
  document.getElementById("members-read").replaceChildren();
  const verdictBox = document.getElementById("verdict");
  verdictBox.replaceChildren();
  verdictBox.className = "";
  document.getElementById("model-result").replaceChildren();
  show("class-view", false);
  show("words-view", false);
  const button = document.getElementById("recognize");
  button.disabled = !cur || state.busy;
  if (!cur) {
    intro.textContent = "Nothing to recognize yet for this choice.";
    renderDev("dev-model");
    return;
  }
  if (cur.mode === "words") {
    intro.textContent = "A network reads the whole word as a stream of letters. It can read letter by letter, or pick the closest word from a list.";
  } else if (cur.mode === "single") {
    intro.textContent = "One trained model reads the signal and gives a score to every possible answer.";
  } else {
    intro.textContent = "Five copies of the same model were trained separately. Each gives a score to all 52 letters; together they vote.";
    const g = groupData();
    const options = g.seeds.map((s, i) => ({ value: i, label: `Model ${i + 1}` }));
    options.push({ value: "mean", label: "All 5 together" });
    radioGroup(models, options, state.model, (v) => { state.model = v; renderModel(); renderPipeline(); writeHash(); });
    if (!cur.left && Object.keys(PUB.groups).length > 1) {
      radioGroup(training, Object.keys(PUB.groups).map((k) => ({
        value: k, label: k === "right" ? "Learned from right-handed people" : "Also learned from left-handed people",
      })), state.group, (v) => { state.group = v; renderModel(); renderPipeline(); writeHash(); });
    }
  }
  document.getElementById("waiting").hidden = state.revealed;
  if (!state.revealed) {
    renderDev("dev-model");
    return;
  }
  let correct;
  if (cur.mode === "words") {
    show("words-view", true);
    correct = renderWords();
  } else {
    show("class-view", true);
    show("members-col", cur.mode === "ensemble");
    renderBars();
    if (cur.mode === "ensemble") renderMembers();
    correct = reading();
  }
  const s = sample();
  const truth = cur.mode === "words" ? s.ref : s.label;
  const verdict = document.getElementById("verdict");
  verdict.className = `verdict ${correct ? "verdict--right" : "verdict--wrong"}`;
  verdict.textContent = correct ? `Correct: it is ${truth}` : `Not quite: it was ${truth}`;
  resultCard();
  renderDev("dev-model");
}

// ---------- the Recognize run ----------
const RUN_STEPS = [
  { pipe: 1, text: () => "Reading 13 sensor channels" },
  { pipe: 2, text: () => (state.filter === "none" ? "Keeping the raw signal"
    : `Cleaning the signal: ${STAGES.filters.find((f) => f.id === state.filter).name.toLowerCase()}`) },
  { pipe: 3, text: () => (current().mode === "ensemble" && state.model === "mean" ? "5 models are reading it" : "The model is reading it") },
  { pipe: 4, text: () => "Deciding" },
];
function recognize() {
  if (state.busy || !current()) return;
  state.busy = true;
  state.revealed = false;
  renderModel();
  const button = document.getElementById("recognize");
  const status = document.getElementById("run-status");
  const items = document.querySelectorAll(".pipeline li");
  const stepMs = REDUCED_MOTION ? 0 : 520;
  button.disabled = true;
  button.textContent = "Recognizing…";
  RUN_STEPS.forEach((step, i) => {
    setTimeout(() => {
      items.forEach((li) => li.classList.remove("active"));
      items[step.pipe].classList.add("active");
      status.textContent = `${step.text()}…`;
    }, i * stepMs);
  });
  setTimeout(() => {
    items.forEach((li) => li.classList.remove("active"));
    status.textContent = "";
    state.busy = false;
    state.revealed = true;
    button.textContent = "Recognize again";
    renderModel();
    renderPipeline();
    writeHash();
    const key = [state.task, state.protocol, state.hand, state.sample, state.model, state.group].join("|");
    if (!score.seen.has(key)) {
      score.seen.add(key);
      score.tried += 1;
      if (document.querySelector("#verdict.verdict--right")) score.right += 1;
      renderScore();
    }
    document.getElementById("answer-card").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "nearest" });
  }, RUN_STEPS.length * stepMs + (REDUCED_MOTION ? 0 : 200));
}

// ---------- developer notes: the code behind each step ----------
const TUNED = "--models cnn_bilstm_attn --augment 2 --aug-policy extended --label-smoothing 0.1 --lr-schedule --epochs 30 --seed 0 --deterministic";
function devData() {
  const cur = current();
  if (state.task === "words") {
    return {
      pick: ["python -m imu2text.download onhw_words500_indep --out ./data",
        "from imu2text.words import load_onhw_words500",
        'ds = load_onhw_words500("data/Words500_indep_02", fold=0)'],
      model: ["python -m scripts.refit_ctc --data data/Words500_indep_02 \\",
        "    --selection results/ctc/fixed_seed0_run1.json --output results/ctc/refit_seed0.json"],
      files: ["imu2text/words.py", "imu2text/seq2seq.py", "docs/rca_ctc_lengths.md"],
    };
  }
  if (state.task === "symbols" || state.task === "equations") {
    const dir = `data/OnHW-symbols_equations_${state.protocol}`;
    const fn = state.task === "symbols" ? "load_onhw_symbols" : "load_onhw_equations";
    return {
      pick: [`python -m imu2text.download onhw_symbols_${state.protocol} --out ./data`,
        `from imu2text.symbols import ${fn}`, `ds = ${fn}("${dir}")`],
      model: [`python -m imu2text.models --onhw-symbols ${dir} --symbols-kind ${state.task} \\`,
        `    ${TUNED} \\`, `    --save-predictions results/tasks/${state.task}_${state.protocol}.npz`],
      files: ["imu2text/symbols.py", "imu2text/models.py"],
    };
  }
  const left = cur && cur.left || state.group === "with_left";
  return {
    pick: ["python -m imu2text.download onhw_chars" + (left ? " onhw_chars_L" : "") + " --out ./data",
      "from imu2text.models import load_official_split",
      'x, y, classes, (train, val, test) = load_official_split(',
      `    "data/onhw-chars_2021-06-30", "both", "${state.protocol}", 0, 0)`],
    model: ["# one model per seed (0 to 4), then average them",
      `python -m imu2text.models --onhw-chars data/onhw-chars_2021-06-30 --case both --dependency ${state.protocol} \\`,
      ...(left ? ["    --onhw-chars-l data/OnHW-chars_L --both-hands \\"] : []),
      `    ${TUNED} --split-seed 0 \\`,
      `    --save-predictions results/ensemble/${left ? "both" : "right"}_seed0.npz`,
      `python -m scripts.ensemble_chars --out results/ensemble/summary \\`,
      `    --group ${left ? "with_left 'results/ensemble/both_seed*.npz'" : "right 'results/ensemble/right_seed*.npz'"}`],
    files: ["imu2text/models.py", "scripts/ensemble_chars.py", "docs/uncertainty.md"],
  };
}
function codeBlock(lines) {
  const wrap = el("div", { class: "code" });
  const pre = el("pre");
  pre.append(el("code", {}, lines.join("\n")));
  const copy = el("button", { type: "button", class: "copy" }, "Copy");
  copy.addEventListener("click", () => {
    navigator.clipboard?.writeText(lines.join("\n")).then(() => {
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy"; }, 1400);
    });
  });
  wrap.append(copy, pre);
  return wrap;
}
function renderDev(id) {
  const box = document.querySelector(`#${id} .dev__body`);
  if (!box) return;
  box.replaceChildren();
  const d = devData();
  const s = sample();
  if (id === "dev-pick") {
    box.append(el("p", {}, "Download the data and load the same split the page uses:"), codeBlock(d.pick));
    if (s && s.test_index !== undefined) box.append(el("p", { class: "hint" }, `This sample is test item ${s.test_index}.`));
  } else if (id === "dev-signal") {
    box.append(el("p", {}, "Apply a filter to a list of recordings (arrays of shape time × 13):"),
      codeBlock(["from imu2text.filters import apply_filter",
        `cleaned = apply_filter(recordings, "${state.filter}")`,
        `# or when training: python -m imu2text.models ... --filter ${state.filter}`]));
    const p = el("p", { class: "hint" }, "Code: ");
    p.append(repoLink("imu2text/filters.py"), document.createTextNode(" · why filtering cost accuracy: "), repoLink("docs/rca_filters.md"));
    box.append(p);
  } else {
    box.append(el("p", {}, "Train and evaluate what you see here:"), codeBlock(d.model));
    if (state.revealed && current() && current().mode === "ensemble") {
      const u = uncertainty();
      box.append(el("p", { class: "hint" }, `Uncertainty for this letter: ${u.total.toFixed(2)} bits in total, ` +
        `${u.disagreement.toFixed(2)} from the models disagreeing (mutual information), ${u.own.toFixed(2)} within each model.`));
    }
    const p = el("p", { class: "hint" }, "Read: ");
    d.files.forEach((f, i) => { if (i) p.append(document.createTextNode(" · ")); p.append(repoLink(f)); });
    box.append(p);
  }
}

// ---------- step 4: open work ----------
function renderOpen() {
  const box = document.getElementById("open");
  box.replaceChildren();
  for (const item of STAGES.open) {
    const card = el("div", { class: "card" });
    card.append(el("h3", {}, item.name), el("p", {}, item.why));
    const link = item.issue
      ? el("a", { href: `${STAGES.repo}/issues/${item.issue}`, target: "_blank", rel: "noopener" }, `Issue #${item.issue}`)
      : el("a", { href: `${STAGES.repo}/issues/new`, target: "_blank", rel: "noopener" }, "Open an issue to start it");
    card.append(link);
    box.append(card);
  }
}

// ---------- pipeline strip and bound numbers ----------
function renderPipeline() {
  const cur = current();
  const s = sample();
  const set = (id, text) => { document.getElementById(id).textContent = text; };
  const people = state.task === "words" ? "new people" : PEOPLE[state.protocol].toLowerCase();
  set("pipe-task", `${TASKS[state.task].label} · ${people}`);
  set("pipe-filter", STAGES.filters.find((f) => f.id === state.filter).name);
  set("pipe-model", modelName());
  const answer = document.getElementById("pipe-answer");
  answer.className = "";
  if (!cur || !s) { set("pipe-sample", "not run yet"); answer.textContent = "none"; return; }
  set("pipe-sample", cur.mode === "words" ? s.ref : s.label);
  if (!state.revealed) { answer.textContent = "press Recognize"; answer.className = "pending"; return; }
  if (cur.mode === "words") {
    answer.textContent = s.lexicon || "(no answer)";
    answer.className = s.lexicon === s.ref ? "right" : "wrong";
    return;
  }
  const probs = currentProbs();
  const top = classNames()[probs.indexOf(Math.max(...probs))];
  answer.textContent = `${top} · ${pct(Math.max(...probs))}`;
  answer.className = top === s.label ? "right" : "wrong";
}

// Numbers written into the HTML (so crawlers and no-JS readers see them) are
// re-read from the data here, so the page can never show a stale figure.
function bindNumbers() {
  for (const node of document.querySelectorAll("[data-bind]")) {
    const value = node.dataset.bind.split(".").reduce((obj, key) => (obj == null ? obj : obj[key]), PUB);
    if (typeof value === "number") node.textContent = node.dataset.fmt === "pct" ? `${value.toFixed(2)}%` : String(value);
  }
}

// The Vahini logo is supplied by the deployment, not this repository. Show
// it when present; otherwise the drawn mark stays.
function showLogo() {
  const logo = document.getElementById("brand-logo");
  const reveal = () => {
    if (logo.naturalWidth > 0) {
      logo.hidden = false;
      document.querySelector(".brand__mark").hidden = true;
    }
  };
  logo.addEventListener("load", reveal);
  if (logo.complete) reveal();
}

function renderAll() {
  renderTask();
  renderSignal();
  renderModel();
  renderPipeline();
  writeHash();
}

bindNumbers();
showLogo();
document.getElementById("time").addEventListener("input", (e) => setTime(Number(e.target.value)));
document.getElementById("download-csv").addEventListener("click", downloadCsv);
document.getElementById("recognize").addEventListener("click", recognize);
readHash();
renderAll();
renderOpen();
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { renderSignal(); renderModel(); }, 150);
});
