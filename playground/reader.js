// Vahini playground: the drawing reader.
//
// A small image classifier that turns a drawing into one of the 62 letters
// and digits, trained by scripts/drawing_reader.py on UJI Pen Characters v2
// (Prat, Castro, Llorens, Marzal and Vilar, 2008, CC BY 4.0,
// doi:10.24432/C5FG8S). The weights are in reader-weights.js. This file
// draws the strokes onto the same 28 x 28 grid the training did and runs the
// network forward: three 3 x 3 convolutions with max pooling, one hidden
// layer, softmax. Plain JavaScript, no library, a few milliseconds a
// drawing.
//
// It reads the drawing, not the pen. The pen models the page explains read
// motion, and their answers come from real recordings.
"use strict";

(function (root) {
  let net = null;

  function decode(t) {
    const bin = typeof atob === "function" ? atob(t.data) : Buffer.from(t.data, "base64").toString("binary");
    const out = new Float32Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      const b = bin.charCodeAt(i);
      out[i] = (b > 127 ? b - 256 : b) * t.scale;
    }
    return { shape: t.shape, v: out };
  }
  function load(weights) {
    if (!weights) return null;
    const layers = weights.layers.map((l) => ({ type: l.type, w: l.w && decode(l.w), b: l.b && decode(l.b) }));
    return { classes: weights.classes, raster: weights.raster, layers };
  }

  // The same drawing as scripts/drawing_reader.py's rasterize(): scaled so
  // the longer side is `box`, centred, each segment sampled every 0.25 px,
  // ink full within 0.5 px of a sample and falling to zero `radius` further.
  function rasterize(strokes, raster) {
    const { size, box, radius } = raster;
    const all = strokes.flat();
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of all) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
    const scale = box / Math.max(x1 - x0, y1 - y0, 1e-6), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const px = [], py = [];
    for (const s of strokes) {
      const q = s.map((p) => [(p.x - cx) * scale + size / 2, (p.y - cy) * scale + size / 2]);
      px.push(q[0][0]); py.push(q[0][1]);
      for (let i = 1; i < q.length; i++) {
        const [ax, ay] = q[i - 1], [bx, by] = q[i];
        const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 0.25));
        for (let k = 1; k <= n; k++) { px.push(ax + (bx - ax) * k / n); py.push(ay + (by - ay) * k / n); }
      }
    }
    const img = new Float32Array(size * size);
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        let best = Infinity;
        const gx = c + 0.5, gy = r + 0.5;
        for (let i = 0; i < px.length; i++) {
          const dx = gx - px[i], dy = gy - py[i];
          const d = dx * dx + dy * dy;
          if (d < best) best = d;
        }
        const d = Math.sqrt(best);
        img[r * size + c] = Math.min(1, Math.max(0, 1 - Math.max(d - 0.5, 0) / radius));
      }
    }
    return img;
  }

  // Keras layouts: conv kernels are [kh][kw][in][out], dense [in][out], and
  // activations run height, width, channel.
  function conv(x, h, w, c, layer) {
    const [kh, kw, , f] = layer.w.shape, K = layer.w.v, B = layer.b.v;
    const out = new Float32Array(h * w * f);
    for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
      for (let o = 0; o < f; o++) {
        let s = B[o];
        for (let a = 0; a < kh; a++) {
          const y = i + a - 1;
          if (y < 0 || y >= h) continue;
          for (let b = 0; b < kw; b++) {
            const xx = j + b - 1;
            if (xx < 0 || xx >= w) continue;
            const base = (y * w + xx) * c, kb = ((a * kw + b) * c) * f + o;
            for (let k = 0; k < c; k++) s += x[base + k] * K[kb + k * f];
          }
        }
        out[(i * w + j) * f + o] = s > 0 ? s : 0;
      }
    }
    return { x: out, h, w, c: f };
  }
  function pool({ x, h, w, c }) {
    const H = Math.floor(h / 2), W = Math.floor(w / 2), out = new Float32Array(H * W * c);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) for (let k = 0; k < c; k++) {
      let m = -Infinity;
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) m = Math.max(m, x[((2 * i + a) * w + 2 * j + b) * c + k]);
      out[(i * W + j) * c + k] = m;
    }
    return { x: out, h: H, w: W, c };
  }
  function dense(x, layer, relu) {
    const [n, m] = layer.w.shape, W = layer.w.v, out = new Float32Array(m);
    for (let o = 0; o < m; o++) {
      let s = layer.b.v[o];
      for (let i = 0; i < n; i++) s += x[i] * W[i * m + o];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }
  function softmax(z) {
    const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  function forward(img) {
    let t = { x: img, h: net.raster.size, w: net.raster.size, c: 1 }, v = null;
    const dl = net.layers.filter((l) => l.type === "Dense");
    for (const l of net.layers) {
      if (l.type === "Conv2D") t = conv(t.x, t.h, t.w, t.c, l);
      else if (l.type === "MaxPooling2D") t = pool(t);
      else if (l.type === "Flatten") v = t.x;
      else if (l.type === "Dense") v = dense(v, l, l !== dl[dl.length - 1]);
    }
    return softmax(Array.from(v));
  }

  // Every class with its probability, most likely first.
  function read(strokes) {
    if (!net) net = load(root.PLAYGROUND_READER_WEIGHTS);
    if (!net) return null;
    const p = forward(rasterize(strokes, net.raster));
    return net.classes.map((label, i) => ({ label, p: p[i] })).sort((a, b) => b.p - a.p);
  }

  const api = { read, rasterize: (s) => (net || (net = load(root.PLAYGROUND_READER_WEIGHTS))) && rasterize(s, net.raster) };
  root.PlaygroundReader = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
