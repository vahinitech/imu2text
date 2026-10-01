"""Train the playground's drawing reader on UJI Pen Characters v2.

The playground's "Draw it" pad has to turn a drawing into one character
before it can show what the pen model reads for a real recording of that
character. Hand-drawn templates (``playground/shapes.js``) missed every
writer whose proportions differed from the template. This trains a small
image classifier instead: each drawing is drawn onto a 28 x 28 grid, so
stroke order, direction and count stop mattering, and the network learns
the shapes from 40 real writers.

Data: UJI Pen Characters (Version 2), Prat, Castro, Llorens, Marzal and
Vilar, 2008, UCI Machine Learning Repository, doi:10.24432/C5FG8S, licensed
CC BY 4.0. Pen trajectories from 60 writers; the published split puts 40 in
``trn`` and 20 in ``tst``. Download ``ujipenchars2.txt`` yourself; the data
is not in this repository. The trained weights are, under the same
attribution (``playground/reader-weights.js``; ``playground/reader.js``
runs them).

Classes: the 62 letters and digits the pen models read (0-9, A-Z, a-z).
Validation comes from 6 of the 40 training writers; the 20 test writers are
scored once, at the end. The current template matcher is scored on the same
test drawings for comparison.

    python -m scripts.drawing_reader data/ujipenchars2/ujipenchars2.txt \\
        --out results/drawing_reader --export playground/reader-weights.js
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess

import numpy as np

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
SIZE = 28  # grid side, pixels
BOX = 20  # the drawing's longer side, pixels, centred in the grid
RADIUS = 1.2  # ink falls off to zero this far outside a 0.5 px core
UNITS_PER_MM = {"UJI": 100.0, "UPV": 152.0}  # from the dataset's readme
VAL_WRITERS = 6


def parse_uji(path: str) -> list:
    """Records of the 62 letters and digits: label, split, writer, strokes.

    Strokes are float arrays of (x, y) in millimetres, y growing downwards.
    """
    records, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("//"):
                continue
            if parts[0] == "WORD":
                split, site, writer = parts[2].split("_", 2)
                cur = {
                    "label": parts[1],
                    "split": split,
                    "writer": f"{site}_{writer.split('-')[0]}",
                    "strokes": [],
                    "mm": UNITS_PER_MM[site],
                }
                if parts[1] in CLASSES:
                    records.append(cur)
            elif parts[0] == "POINTS" and cur is not None:
                xy = np.array(parts[3:], dtype=np.float64).reshape(-1, 2)
                cur["strokes"].append(xy / cur["mm"])
    for r in records:
        del r["mm"]
    return records


def rasterize(strokes, radius: float = RADIUS) -> np.ndarray:
    """Draw strokes onto a SIZE x SIZE grid, scaled so the longer side is BOX.

    Every segment is sampled every 0.25 px; each pixel takes the ink of its
    nearest sample: full within 0.5 px, falling linearly to zero at
    0.5 + radius. playground/reader.js repeats this exactly.
    """
    pts = np.concatenate([np.asarray(s, np.float64) for s in strokes])
    lo, hi = pts.min(0), pts.max(0)
    scale = BOX / max(float((hi - lo).max()), 1e-6)
    centre = (lo + hi) / 2
    samples = []
    for s in strokes:
        p = (np.asarray(s, np.float64) - centre) * scale + SIZE / 2
        samples.append(p[:1])
        for a, b in zip(p[:-1], p[1:]):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / 0.25)))
            t = np.arange(1, n + 1)[:, None] / n
            samples.append(a + (b - a) * t)
    s = np.concatenate(samples)
    grid = np.stack(np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5), -1)
    d = np.sqrt(((grid.reshape(-1, 1, 2) - s[None]) ** 2).sum(-1)).min(1)
    ink = np.clip(1.0 - np.maximum(d - 0.5, 0.0) / radius, 0.0, 1.0)
    return ink.reshape(SIZE, SIZE).astype(np.float32)


def augment(strokes, rng) -> tuple:
    """A plausible other writer: rotated, slanted, stretched, shaky."""
    theta = np.deg2rad(np.clip(rng.normal(0, 7), -15, 15))
    shear = rng.uniform(-0.35, 0.35)
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.9, 1.1)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    m = rot @ np.array([[sx, shear], [0.0, sy]])
    pts = np.concatenate(strokes)
    span = float((pts.max(0) - pts.min(0)).max()) or 1.0
    out = []
    for s in strokes:
        p = (s - pts.mean(0)) @ m.T
        out.append(p + rng.normal(0, 0.012 * span, p.shape))
    return out, rng.uniform(0.9, 1.5)


def build_model(n_classes: int):
    """Three small convolutions and one hidden layer, about 66k weights."""
    from tensorflow import keras  # noqa: PLC0415

    layers = keras.layers
    return keras.Sequential(
        [
            keras.Input((SIZE, SIZE, 1)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(48, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(96, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )


def quantize(model) -> list:
    """Each layer's weights as int8 with one float scale per tensor."""
    out = []
    for layer in model.layers:
        ws = layer.get_weights()
        if not ws:
            out.append({"type": type(layer).__name__})
            continue
        entry = {"type": type(layer).__name__}
        for name, w in zip(("w", "b"), ws):
            scale = float(np.abs(w).max()) / 127 or 1.0
            q = np.round(w / scale).astype(np.int8)
            entry[name] = {
                "shape": list(w.shape),
                "scale": scale,
                "data": base64.b64encode(q.tobytes()).decode("ascii"),
            }
        out.append(entry)
    return out


def dequantized(model, layers: list):
    """A copy of the model running on the int8 weights, to score them."""
    from tensorflow import keras  # noqa: PLC0415

    copy = keras.models.clone_model(model)
    copy.build((None, SIZE, SIZE, 1))
    for layer, entry in zip(copy.layers, layers):
        if "w" not in entry:
            continue
        ws = []
        for name in ("w", "b"):
            e = entry[name]
            q = np.frombuffer(base64.b64decode(e["data"]), np.int8)
            ws.append(q.reshape(e["shape"]).astype(np.float32) * e["scale"])
        layer.set_weights(ws)
    return copy


def scores(proba: np.ndarray, true: np.ndarray) -> dict:
    """Accuracy, top-3, case-insensitive, and digits and letters apart."""
    pred = proba.argmax(1)
    folded = np.char.lower(np.array(CLASSES))
    digit = true < 10
    top3 = (np.argsort(-proba, 1)[:, :3] == true[:, None]).any(1)
    return {
        "n": int(len(true)),
        "accuracy": float((pred == true).mean() * 100),
        "top3": float(top3.mean() * 100),
        "case_insensitive": float((folded[pred] == folded[true]).mean() * 100),
        "digits": float((pred[digit] == true[digit]).mean() * 100),
        "letters": float((pred[~digit] == true[~digit]).mean() * 100),
    }


MATCHER = """
const S = require(process.argv[1]);
let input = "";
process.stdin.on("data", (d) => { input += d; });
process.stdin.on("end", () => {
  const out = JSON.parse(input).map((c) => {
    const r = S.recognise(c.strokes, { size: 170, task: c.task });
    if (!r) return null;
    if (r.sequence) return r.sequence.map((p) => p.label || "?").join("");
    return r.label;
  });
  process.stdout.write(JSON.stringify(out));
});
"""


def matcher_scores(records: list, shapes_js: str) -> dict | None:
    """The template matcher on the same drawings, at a natural size.

    Drawings are placed on the pad at 12 px per mm, the scale at which a
    capital about 10 mm tall fills most of the 170 px pad, and read in the
    task that holds their class, so the matcher gets its best chance.
    """
    node = shutil.which("node")
    if not node:
        return None
    cases = []
    for r in records:
        pts = np.concatenate(r["strokes"])
        origin = pts.min(0) - [(40 / 12), (170 - (pts[:, 1].ptp() * 12)) / 24]
        strokes = [
            [{"x": float(x), "y": float(y)} for x, y in (s - origin) * 12]
            for s in r["strokes"]
        ]
        task = "symbols" if r["label"].isdigit() else "chars"
        cases.append({"strokes": strokes, "task": task})
    done = subprocess.run(
        [node, "-e", MATCHER, os.path.abspath(shapes_js)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(done.stdout)
    true = [r["label"] for r in records]
    right = np.array([g == t for g, t in zip(got, true)])
    folded = np.array([(g or "").lower() == t.lower() for g, t in zip(got, true)])
    refused = np.array([g is None or "?" in (g or "?") for g in got])
    digit = np.array([t.isdigit() for t in true])
    return {
        "n": len(true),
        "accuracy": float(right.mean() * 100),
        "case_insensitive": float(folded.mean() * 100),
        "digits": float(right[digit].mean() * 100),
        "letters": float(right[~digit].mean() * 100),
        "not_sure": float(refused.mean() * 100),
    }


def main() -> None:
    """CLI: train, score on the test writers, export the int8 weights."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("data", help="ujipenchars2.txt")
    ap.add_argument("--out", default="results/drawing_reader")
    ap.add_argument("--export", help="write the weights as a playground script")
    ap.add_argument("--copies", type=int, default=24, help="augmented copies")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--shapes-js", default="playground/shapes.js")
    args = ap.parse_args()

    import tensorflow as tf  # noqa: PLC0415

    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    rng = np.random.default_rng(args.seed)

    records = parse_uji(args.data)
    index = {c: i for i, c in enumerate(CLASSES)}
    train_writers = sorted({r["writer"] for r in records if r["split"] == "trn"})
    val_writers = set(rng.choice(train_writers, VAL_WRITERS, replace=False).tolist())
    fit = [r for r in records if r["split"] == "trn" and r["writer"] not in val_writers]
    val = [r for r in records if r["split"] == "trn" and r["writer"] in val_writers]
    test = [r for r in records if r["split"] == "tst"]
    print(
        f"{len(fit)} fitting drawings ({len(train_writers) - VAL_WRITERS} writers), "
        f"{len(val)} validation ({VAL_WRITERS}), {len(test)} test "
        f"({len({r['writer'] for r in test})} writers, never seen in training)"
    )

    def clean(rs):
        x = np.stack([rasterize(r["strokes"]) for r in rs])[..., None]
        return x, np.array([index[r["label"]] for r in rs])

    xs, ys = [], []
    for r in fit:
        xs.append(rasterize(r["strokes"]))
        ys.append(index[r["label"]])
        for _ in range(args.copies):
            strokes, radius = augment(r["strokes"], rng)
            xs.append(rasterize(strokes, radius))
            ys.append(index[r["label"]])
    x_fit, y_fit = np.stack(xs)[..., None], np.array(ys)
    x_val, y_val = clean(val)
    x_test, y_test = clean(test)

    model = build_model(len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=128,
        shuffle=True,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6, restore_best_weights=True
            )
        ],
    )

    layers = quantize(model)
    int8 = dequantized(model, layers)
    report = {
        "data": "UJI Pen Characters v2, published split: 40 train writers, 20 test",
        "classes": len(CLASSES),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "copies": args.copies,
        "epochs_run": len(history.history["loss"]),
        "weights": int(model.count_params()),
        "validation": scores(model.predict(x_val, verbose=0), y_val),
        "test_float": scores(model.predict(x_test, verbose=0), y_test),
        "test_int8": scores(int8.predict(x_test, verbose=0), y_test),
        "test_template_matcher": matcher_scores(test, args.shapes_js),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.savez_compressed(
        os.path.join(args.out, "test_predictions.npz"),
        true=y_test,
        proba=np.round(int8.predict(x_test, verbose=0), 4),
        classes=np.array(CLASSES),
    )
    print(json.dumps(report, indent=2))

    if args.export:
        payload = {
            "source": "UJI Pen Characters v2 (Prat et al., 2008), CC BY 4.0, "
            "doi:10.24432/C5FG8S; trained by scripts/drawing_reader.py",
            "classes": CLASSES,
            "raster": {"size": SIZE, "box": BOX, "radius": RADIUS},
            "layers": layers,
            "test": report["test_int8"],
        }
        with open(args.export, "w", encoding="utf-8") as f:
            f.write(
                "// Generated by scripts/drawing_reader.py. Do not edit.\n"
                "// Weights trained on UJI Pen Characters v2 (Prat, Castro, "
                "Llorens, Marzal and Vilar,\n// 2008), CC BY 4.0, "
                "doi:10.24432/C5FG8S.\n"
            )
            f.write("window.PLAYGROUND_READER_WEIGHTS = ")
            json.dump(payload, f, separators=(",", ":"))
            f.write(";\n")
        print(f"wrote {args.export} ({os.path.getsize(args.export) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
