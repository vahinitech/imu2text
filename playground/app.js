// Vahini playground (imu2text). Plain JavaScript, no dependencies, no build step.
// Data comes from scripts/build_playground.py; stages from stages.js.
"use strict";

const PUB = window.PLAYGROUND_PUBLIC;
const LOCAL = window.PLAYGROUND_LOCAL || null;
const STAGES = window.PLAYGROUND_STAGES;
const SVG_NS = "http://www.w3.org/2000/svg";

// ---------- tasks and the samples each one offers ----------
const TASKS = {
  chars: { label: "Characters", note: "52 classes: A to Z and a to z, one letter per recording." },
  symbols: { label: "Symbols", note: "15 classes: the digits 0 to 9 and + - · : =, one per recording." },
  equations: { label: "Equations", note: "15 classes: single symbols cut out of handwritten equations." },
  words: { label: "Words", note: "OnHW-Words500: whole German words, spelled out of 59 characters." },
};
const KIND_TITLES = {
  clear: "The model is sure, and right",
  case: "Split between a letter and its other case",
  disagree: "The five models disagree",
  unsure: "The model is unsure (top choice below 50%)",
  wrong: "Sure, and wrong",
  right: "Plain decoding already spells the word",
  fixed: "The word list fixes it",
  abstain: "No word from the list fits",
};
const WORD_KINDS = ["right", "fixed", "abstain", "wrong"];
const WORD_TITLES = { wrong: "Both decodings wrong" };

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
};

// What the current task, protocol and hand select. `mode` decides how step 3
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
  if (!available(state.task, state.protocol)) state.protocol = "indep";
}
function writeHash() {
  const h = new URLSearchParams({
    task: state.task, protocol: state.protocol, hand: state.hand, sample: state.sample,
    filter: state.filter, model: state.model, group: state.group,
  });
  history.replaceState(null, "", `#${h}`);
}

// ---------- step 1: task and sample ----------
function renderTask() {
  radioGroup(document.getElementById("task"),
    Object.entries(TASKS).map(([id, t]) => ({ value: id, label: t.label })), state.task,
    (v) => {
      state.task = v; state.sample = 0;
      if (!available(v, state.protocol)) state.protocol = "indep";
      renderAll();
    });
  document.getElementById("task-note").textContent = TASKS[state.task].note;

  const protoBox = document.getElementById("protocol");
  if (state.task === "words") {
    protoBox.replaceChildren(el("span", { class: "hint" }, "Unseen writers only."));
  } else {
    radioGroup(protoBox, [
      { value: "indep", label: "Unseen writers", disabled: !available(state.task, "indep") },
      { value: "dep", label: "Seen writers", disabled: !available(state.task, "dep"), title: "not run yet" },
    ], state.protocol, (v) => { state.protocol = v; state.sample = 0; renderAll(); });
  }
  const handBox = document.getElementById("hand");
  if (state.task === "chars" && state.protocol === "indep" && PUB.left_letters.length) {
    radioGroup(handBox, [
      { value: "right", label: "Right-handed writers" },
      { value: "left", label: "Left-handed writers" },
    ], state.hand, (v) => { state.hand = v; state.sample = 0; renderAll(); });
  } else {
    handBox.replaceChildren();
  }

  const box = document.getElementById("samples");
  box.replaceChildren();
  const cur = current();
  if (!cur) {
    box.append(el("p", { class: "note" }, "This combination has not been run yet."));
    return;
  }
  const order = cur.mode === "words" ? WORD_KINDS : Object.keys(KIND_TITLES);
  for (const kind of order) {
    const items = cur.samples.map((s, i) => [s, i]).filter(([s]) => s.kind === kind);
    if (!items.length) continue;
    const group = el("div", { class: "letter-group" });
    const title = (cur.mode === "words" && WORD_TITLES[kind]) || KIND_TITLES[kind];
    group.append(el("h3", {}, title));
    const row = el("div", { class: "controls", role: "radiogroup", "aria-label": title });
    for (const [s, i] of items) {
      const text = cur.mode === "words" ? s.ref : s.label;
      const b = el("button", {
        class: cur.mode === "words" ? "word-btn" : "letter-btn", role: "radio",
        "aria-checked": String(i === state.sample),
      }, text);
      b.addEventListener("click", () => { state.sample = i; renderAll(); });
      row.append(b);
    }
    group.append(row);
    box.append(group);
  }
  renderWhyProtocol();
}

function renderWhyProtocol() {
  const box = document.getElementById("why-protocol");
  box.replaceChildren(el("p", {},
    "Unseen writers means no one in the test set wrote any of the training data. " +
    "That is the test that matters for a pen meant to work for a new writer. Seen writers " +
    "means the same people appear in training and test, so the model has already " +
    "learned their handwriting, and the score is higher."));
  const rows = [];
  for (const task of ["chars", "symbols", "equations"]) {
    const dep = PUB.tasks[`${task}_dep`];
    const indep = task === "chars" ? null : PUB.tasks[`${task}_indep`];
    if (dep && indep) {
      rows.push(`${TASKS[task].label}: ${indep.accuracy.toFixed(2)}% unseen, ${dep.accuracy.toFixed(2)}% seen`);
    }
  }
  if (rows.length) {
    box.append(el("p", {}, `On this page (one model, seed 0): ${rows.join("; ")}.`));
  }
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
  const series = panel.channels.map((c) => ({ raw: raw[c], filt: filtered[c] }));
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
    note.textContent = "Synthetic signal. The local build holds real recordings for the " +
      "right-handed characters only, so this sample shows the made-up signal.";
  } else {
    note.textContent = "Synthetic signal. The OnHW recordings are not redistributed, so the " +
      "published page shows a made-up signal to demonstrate the filters. Build locally with " +
      "your own download to see real recordings (playground/README.md).";
  }
  radioGroup(document.getElementById("filters"),
    STAGES.filters.map((f) => ({ value: f.id, label: f.name })), state.filter,
    (v) => { state.filter = v; renderSignal(); writeHash(); });

  const box = document.getElementById("signal");
  box.replaceChildren();
  const showRaw = state.filter !== "none";
  for (const panel of SENSOR_PANELS) box.append(drawPanel(panel, sig.raw, sig.filtered, showRaw));
  if (showRaw) box.append(el("p", { class: "hint" }, "Grey: before the filter. Colour: after."));
  const n = sig.filtered[0].length;
  document.getElementById("time").max = String(n - 1);
  setTime(Math.min(state.t ?? Math.floor(n / 2), n - 1));

  const f = STAGES.filters.find((s) => s.id === state.filter);
  const card = document.getElementById("filter-result");
  card.replaceChildren(
    el("div", {}, "Test accuracy of a character model trained with this filter"),
    el("div", { class: "big" }, `${f.accuracy.toFixed(2)}%`),
    el("div", {}, f.conditions),
    el("p", {}, f.note),
  );
  const src = el("div", {}, "Source: ");
  src.append(repoLink(f.source), document.createTextNode(" · code: "), repoLink(f.code));
  card.append(src, el("p", { class: "hint" },
    "The models in step 3 were trained without a filter. Per-sample outputs of the filtered models are not exported yet."));
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

function renderBars() {
  const s = sample();
  const names = classNames();
  const top = topK(currentProbs(), 5);
  const W = widthOf("bars", 460), rowH = 30, labelW = 34, valueW = 110;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${rowH * top.length}`, role: "img",
    "aria-label": "Top five classes with probabilities" });
  top.forEach(([cls, p], r) => {
    const name = names[cls];
    const yMid = r * rowH + rowH / 2;
    const barW = Math.max(2, p * (W - labelW - valueW));
    svg.append(svgEl("text", { x: 4, y: yMid + 5, class: "label-strong" }, name));
    svg.append(svgEl("rect", { x: labelW, y: yMid - 9, width: barW, height: 18, rx: 4,
      fill: name === s.label ? "var(--series-3)" : "var(--series-1)" }));
    svg.append(svgEl("text", { x: labelW + barW + 6, y: yMid + 4 }, name === s.label ? `${pct(p)}  correct` : pct(p)));
  });
  document.getElementById("bars").replaceChildren(svg);
}

function renderMembers() {
  const g = groupData();
  const top = topK(g.mean, 3).map(([cls]) => cls);
  const S = g.members.length, W = widthOf("members", 460), H = 150, base = 120, barW = 10, gap = 3;
  const groupW = S * (barW + gap);
  const groupGap = Math.max(16, Math.min(60, (W - 40 - 3 * groupW) / 2));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Each model's probability for the three most likely letters" });
  svg.append(svgEl("line", { x1: 0, x2: W, y1: base, y2: base, stroke: "var(--grid)" }));
  top.forEach((cls, gi) => {
    const x0 = 20 + gi * (groupW + groupGap);
    g.members.forEach((probs, s) => {
      const h = probs[cls] * (base - 10);
      const selected = state.model === s || state.model === "mean";
      svg.append(svgEl("rect", { x: x0 + s * (barW + gap), y: base - h, width: barW, height: Math.max(h, 1),
        rx: 2, fill: "var(--series-1)", opacity: selected ? 1 : 0.35 }));
    });
    const meanY = base - g.mean[cls] * (base - 10);
    svg.append(svgEl("line", { x1: x0 - 3, x2: x0 + groupW, y1: meanY, y2: meanY,
      stroke: "var(--text-primary)", "stroke-dasharray": "3 2" }));
    svg.append(svgEl("text", { x: x0 + groupW / 2 - 4, y: base + 18, class: "label-strong" }, PUB.classes[cls]));
  });
  document.getElementById("members").replaceChildren(svg);

  // Total uncertainty is the entropy of the average. The part caused by the
  // models disagreeing is the mutual information: entropy of the average
  // minus the average entropy. The rest is ambiguity in the signal itself.
  const entropy = (p) => -p.reduce((acc, v) => acc + (v > 0 ? v * Math.log2(v) : 0), 0);
  const total = entropy(g.mean);
  const meanOwn = g.members.reduce((acc, p) => acc + entropy(p), 0) / g.members.length;
  const disagreement = Math.max(0, total - meanOwn);
  const [a, b] = topK(g.mean, 2);
  const caseSplit = PUB.classes[a[0]].toLowerCase() === PUB.classes[b[0]].toLowerCase();
  const first = PUB.classes[a[0]], second = PUB.classes[b[0]];
  // Display thresholds for choosing a sentence, not measured results.
  let read;
  if (disagreement >= 0.5) {
    read = "The models disagree with each other, so the model is unsure: this writer is unlike the training data (epistemic uncertainty)." +
      (meanOwn >= 1 ? " Each model is also unsure on its own." : "");
  } else if (a[1] >= 0.7) {
    read = `The models agree on ${first} (${pct(a[1])} on average).`;
  } else {
    read = `Each model is itself split between ${first} and ${second}, and they agree on that. ${caseSplit ? "The signal barely tells the two cases apart" : "The signal is ambiguous"} (aleatoric uncertainty).`;
  }
  const truth = sample().label;
  read += first === truth ? ` The top choice, ${first}, is right.` : ` The top choice, ${first}, is wrong: the letter is ${truth}.`;
  read += ` Uncertainty ${total.toFixed(2)} bits: ${disagreement.toFixed(2)} from the models disagreeing, ${meanOwn.toFixed(2)} from each model's own doubt.`;
  document.getElementById("members-read").textContent = `${read} Dashed line: the ensemble average.`;
}

function renderWords() {
  const w = PUB.words;
  const s = sample();
  const box = document.getElementById("words-view");
  const table = el("table", { class: "words-table" });
  const verdict = (hyp) => (hyp === s.ref ? "right" : hyp === "" ? "no answer" : "wrong");
  for (const [name, hyp, cer] of [
    ["Written", s.ref, null],
    ["Plain decoding (greedy)", s.greedy, s.cer_greedy],
    ["Decoding restricted to the word list", s.lexicon, s.cer_lexicon],
  ]) {
    const tr = el("tr");
    tr.append(el("th", { scope: "row" }, name), el("td", { class: "word" }, hyp || "(empty)"),
      el("td", {}, cer === null ? "" : `${verdict(hyp)} · CER ${cer.toFixed(1)}%`));
    table.append(tr);
  }
  box.replaceChildren(table, el("p", { class: "hint" },
    "CER is the character error rate: edits needed to turn the output into the written word, " +
    "divided by its length. The word list holds the 501 strings seen in training."));
  const card = document.getElementById("model-result");
  card.replaceChildren(
    el("div", {}, `All ${w.n.toLocaleString()} test words from unseen writers`),
    el("div", { class: "big" }, `CER ${w.cer_greedy.toFixed(2)}% · ${w.exact_greedy.toFixed(2)}% exact`),
    el("div", {}, `With the word list: CER ${w.cer_lexicon.toFixed(2)}%, ${w.exact_lexicon.toFixed(2)}% exact, ` +
      `${w.empty_lexicon.toLocaleString()} words left empty.`),
    el("div", {}, "OnHW-Words500 right-handed, published fold 0, CNN+BiLSTM with a CTC head, " +
      "15 epochs (under-trained), refit on all 42 training writers, seed 0."),
  );
  const src = el("div", {}, "Source: ");
  src.append(repoLink(w.source), document.createTextNode(" · analysis: "), repoLink("docs/rca_ctc_lengths.md"));
  card.append(src);
}

function renderModel() {
  const cur = current();
  const intro = document.getElementById("model-intro");
  show("class-view", cur && cur.mode !== "words");
  show("words-view", cur && cur.mode === "words");
  show("members-col", cur && cur.mode === "ensemble");
  document.getElementById("models").replaceChildren();
  document.getElementById("training").replaceChildren();
  if (!cur) {
    intro.textContent = "";
    document.getElementById("model-result").replaceChildren();
    return;
  }
  if (cur.mode === "words") {
    intro.textContent = "A network with a CTC head reads the whole recording and spells out " +
      "characters. Plain decoding takes the likeliest character at each step; the word list " +
      "only allows words seen in training.";
    renderWords();
    renderWhyModel();
    return;
  }
  if (cur.mode === "single") {
    intro.textContent = "One network (seed 0). Only the right-handed character task has a " +
      "five-seed ensemble so far.";
    renderBars();
    const run = cur.run;
    const card = document.getElementById("model-result");
    card.replaceChildren(
      el("div", {}, `Accuracy on all ${run.n_test.toLocaleString()} test samples`),
      el("div", { class: "big" }, `${run.accuracy.toFixed(2)}%`),
      el("div", {}, `${run.model}, ${run.split}, ${run.classes.length} classes, 30 epochs, seed ${run.seed}, deterministic`),
    );
    const src = el("div", {}, "Source: ");
    src.append(repoLink(run.source), document.createTextNode(" · code: "), repoLink("imu2text/models.py"));
    card.append(src);
    renderWhyModel();
    return;
  }
  // Ensemble of five seeds (characters, unseen writers).
  intro.textContent = "Five copies of the same network, trained with different random seeds. " +
    "Each gives a probability for all 52 letters. The ensemble averages them.";
  const g = groupData();
  const options = g.seeds.map((s, i) => ({ value: i, label: `Seed ${s}` }));
  options.push({ value: "mean", label: `Ensemble of ${g.seeds.length}` });
  radioGroup(document.getElementById("models"), options, state.model,
    (v) => { state.model = v; renderModel(); writeHash(); });
  if (!cur.left && Object.keys(PUB.groups).length > 1) {
    radioGroup(document.getElementById("training"), Object.keys(PUB.groups).map((k) => ({
      value: k, label: k === "right" ? "Trained on right-handed writers" : "Plus left-handed writers",
    })), state.group, (v) => { state.group = v; renderModel(); writeHash(); });
  }
  renderBars();
  renderMembers();
  const summary = cur.left ? cur.summary : (PUB.groups[state.group] || PUB.groups.right);
  const acc = state.model === "mean" ? summary.ensemble_accuracy : summary.member_accuracy[state.model];
  const who = cur.left ? "left-handed" : "right-handed";
  const card = document.getElementById("model-result");
  card.replaceChildren(
    el("div", {}, `Accuracy of this choice on all ${summary.n_test.toLocaleString()} ${who} test letters`),
    el("div", { class: "big" }, `${acc.toFixed(2)}%`),
    el("div", {}, `${summary.model}, ${summary.split}, 52 classes, 30 epochs` +
      (cur.left ? "; left-handed writers held out from a split built here" : "")),
  );
  const src = el("div", {}, "Source: ");
  src.append(repoLink("results/ensemble/summary.md"), document.createTextNode(" · code: "),
    repoLink("imu2text/models.py"), document.createTextNode(", "), repoLink("scripts/ensemble_chars.py"));
  card.append(src);
  renderWhyModel();
}

function renderWhyModel() {
  const title = document.getElementById("why-model-title");
  const box = document.getElementById("why-model");
  box.replaceChildren();
  const p = (text) => box.append(el("p", {}, text));
  if (state.task === "words") {
    title.textContent = "Why a word list helps, and when it hurts";
    p("Plain decoding often lands one or two characters away from a real word. Restricting " +
      "the answer to known words turns many of those into exact matches, which is why exact " +
      "accuracy rises. When no known word fits well, the decoder returns nothing, and every " +
      "character of the word counts as an error, which is why CER goes up.");
    return;
  }
  if (state.task === "chars") {
    title.textContent = "Why some letters are hard";
    p("Most mistakes are a letter read as its own other case: 43% of the errors of the 72.5% " +
      "model (docs/benchmarks.md). A small o and a capital O are the same movement at a " +
      "different size, and the sensors see size poorly: writers form capitals larger and " +
      "faster at once, so the accelerations come out alike. Scored without case, the same " +
      "model reads 84.3%.");
    p("Left-handed writers hold and move the pen differently. With only right-handed and a " +
      "few left-handed writers in training, left-handed letters are read far less reliably.");
    return;
  }
  title.textContent = "Why symbols and equations differ";
  p("Symbols are written one at a time; equation symbols are cut out of whole written " +
    "equations. The equation set is about 17 times larger (39,643 against 2,326 " +
    "samples). Against the published figures, the tuned model is ahead on the two " +
    "larger sets and behind on symbols, the smallest: regularisation needs data " +
    "(docs/benchmarks.md).");
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

function renderAll() {
  renderTask();
  renderSignal();
  renderModel();
  writeHash();
}

document.getElementById("repo-link").href = STAGES.repo;
document.getElementById("time").addEventListener("input", (e) => setTime(Number(e.target.value)));
document.getElementById("download-csv").addEventListener("click", downloadCsv);
readHash();
renderAll();
renderOpen();
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { renderSignal(); renderModel(); }, 150);
});
