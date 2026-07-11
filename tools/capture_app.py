"""Vahini classroom capture app (Tkinter + matplotlib).

Shows one prompt at a time ("Box 12 — write g"), records the 16-channel pen
stream, reconstructs the stroke after pen-lift and shows it to the child,
then advances. Implements the full fieldwork flow of
docs/school_collection_procedure.md:

- calibration at session start (5 s rest -> per-channel bias/noise floor,
  then a spiral) stored as calibration.csv / calibration.json
- pen-down detection from the force channel (threshold derived from the
  measured rest noise, not guessed)
- automatic QC per sample (duration, force activity, signal energy vs. the
  calibrated noise floor, sample-counter continuity)
- Enter/space = accept (auto-accepts after a delay), R = redo
- scheduled child breaks every N accepted samples (default 70) with a
  countdown screen
- resume: re-running the app continues at the next empty box
- storage exactly per procedure §6 (rec_*.csv + labels.csv + meta.json)

Run with real hardware (USB-C serial, one CSV line per sample:
"counter,ch1,...,ch16"):

    python tools/capture_app.py --student S0001 --school school01 \
        --prompts sheets/sheets_S0001_prompts.csv --pen-serial VP-003 \
        --port /dev/ttyACM0

Run without hardware (simulator writes the prompted character itself —
use this to develop, test, and rehearse):

    python tools/capture_app.py --student S0001 --school school01 \
        --prompts sheets/sheets_S0001_prompts.csv --pen-serial SIM --simulate

The reconstruction and QC logic below is import-safe without a display;
tkinter is only imported when the GUI starts (tests import this module
headless).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np

FS = 208                     # pen sample rate (Hz)
N_CHANNELS = 16              # force, front accel(3), front gyro(3), mag(3),
                             # rear accel(3), rear gyro(3)
FORCE, FA = 0, slice(1, 4)   # channel indices used by reconstruction
PRACTICE = list("olA3xT")    # matches tools/make_sheets.py


# --------------------------------------------------------------------------- #
# Reconstruction (accelerometer double integration with drift removal)
# --------------------------------------------------------------------------- #
def _smooth(a: np.ndarray, k: int = 7) -> np.ndarray:
    if len(a) < k:
        return a
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, a)


def pen_down_mask(force: np.ndarray, rest_mean: float, rest_std: float) -> np.ndarray:
    """Boolean mask of pen-on-paper samples, from calibrated rest statistics."""
    th = rest_mean + max(8.0 * rest_std, 1e-3)
    return force > th


def reconstruct_xy(seq: np.ndarray, bias: Optional[np.ndarray] = None,
                   mask: Optional[np.ndarray] = None, fs: int = FS
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,16) raw sample -> (x, y, pen_down) 2-D stroke for display.

    Method: front accelerometer, bias/gravity removed (calibration bias if
    available, else the sample's own mean), smoothed; the pen is at rest at
    the start and end of the recording, so integrate acc -> vel with the
    residual end velocity ramped out (ZUPT), then vel -> position with a
    linear detrend absorbing leftover bias drift. Project to the dominant
    2-D plane (PCA), oriented so writing runs left->right and starts near
    the top. (Empirically the most faithful of the variants tested — see
    tests/test_capture.py, which asserts shape recovery.)
    """
    acc = seq[:, FA].astype(np.float64)
    acc = acc - (bias[1:4] if bias is not None else acc.mean(axis=0))
    acc = _smooth(acc)
    n = len(acc)
    if n < 8:
        return np.zeros(1), np.zeros(1), np.ones(1, dtype=bool)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = mask[:n]
    t = np.arange(n)[:, None]
    vel = np.cumsum(acc, axis=0) / fs
    vel -= vel[-1] * (t / max(n - 1, 1))                       # endpoint ZUPT
    pos = np.cumsum(vel, axis=0) / fs
    pos -= pos[0] + (pos[-1] - pos[0]) * (t / max(n - 1, 1))   # drift detrend
    pos -= pos.mean(axis=0)
    _, _, vt = np.linalg.svd(pos, full_matrices=False)   # dominant writing plane
    xy = pos @ vt[:2].T
    if xy[-1, 0] < xy[0, 0]:                             # write left -> right
        xy[:, 0] *= -1
    if xy[: n // 4, 1].mean() < xy[:, 1].mean():         # start near the top
        xy[:, 1] *= -1
    if mask is None:
        mask = np.ones(n, dtype=bool)
    return xy[:, 0], xy[:, 1], mask[:n]


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def qc_check(seq: np.ndarray, counters: np.ndarray, calib: dict,
             fs: int = FS) -> Tuple[bool, str]:
    """Accept/reject one recording. Returns (ok, child-friendly reason)."""
    dur = len(seq) / fs
    if dur < 0.3:
        return False, "That was very quick — try again, nice and big!"
    if dur > 8.0:
        return False, "That took a little long — one character only."
    if np.any(np.diff(counters) != 1):
        return False, "The pen skipped — let's try that one again."
    down = pen_down_mask(seq[:, FORCE], calib["force_mean"], calib["force_std"])
    if down.mean() < 0.15:
        return False, "I could hardly feel the pen — press a bit harder."
    noise = np.asarray(calib["accel_std"])
    energy = seq[:, FA].std(axis=0)
    if np.all(energy < 3.0 * noise + 1e-9):
        return False, "I didn't see any writing — try again."
    return True, ""


# --------------------------------------------------------------------------- #
# Pen sources
# --------------------------------------------------------------------------- #
def glyph_path(char: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """(path, pen_down) polyline of a character's font outline, n samples.

    Strokes are time-eased (bell-shaped velocity, still at both ends) and
    joined by eased in-air glides — the physical profile of real writing,
    and what makes IMU double integration well-posed.
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    polys = TextPath((0, 0), char, size=1.0,
                     prop=FontProperties(family="DejaVu Sans")).to_polygons()
    if not polys:
        polys = [np.array([[0, 0], [1, 1]])]
    strokes = []
    per = max(n // sum(len(p) for p in polys), 2)
    for p in polys:
        seg = np.asarray(p, dtype=float)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        if d[-1] == 0:
            continue
        n_pts = max(per * len(seg) // 4, 24)
        u = (1 - np.cos(np.linspace(0, np.pi, n_pts))) / 2 * d[-1]
        strokes.append(np.c_[np.interp(u, d, seg[:, 0]),
                             np.interp(u, d, seg[:, 1])])
    pts, down = [], []
    for i, s in enumerate(strokes):
        pts.append(s)
        down.append(np.ones(len(s), dtype=bool))
        if i + 1 < len(strokes):                       # pen lift: smooth glide
            a, b = s[-1], strokes[i + 1][0]
            ease = (1 - np.cos(np.linspace(0, np.pi, 16)))[:, None] / 2
            pts.append(a + (b - a) * ease)
            down.append(np.zeros(16, dtype=bool))
    path = np.vstack(pts)
    mask = np.concatenate(down)
    idx = np.linspace(0, len(path) - 1, n).astype(int)
    return path[idx], mask[idx]


def orient_to_glyph(xy: np.ndarray, char: str) -> np.ndarray:
    """Rotate a reconstruction into the prompted glyph's frame for display.

    PCA leaves two ambiguities: the in-plane rotation and the handedness
    (whether the plane is viewed from above or below). The rotation is fixed
    by Procrustes against the prompted glyph. Reflection is allowed ONLY when
    it is decisively better — the handedness artifact reflects to a
    near-perfect match, whereas a child genuinely mirror-writing a letter
    does not, so real mirror errors stay visible on screen.
    """
    ref, _ = glyph_path(char, len(xy))
    A = xy - xy.mean(axis=0)
    B = ref - ref.mean(axis=0)
    An = A / (np.linalg.norm(A) + 1e-12)
    Bn = B / (np.linalg.norm(B) + 1e-12)
    u, s, vt = np.linalg.svd(An.T @ Bn)
    refl_sim = s.sum()                       # best similarity, reflection allowed
    u_rot = u.copy()
    if np.linalg.det(u @ vt) < 0:
        u_rot[:, -1] *= -1
        rot_sim = (s * [1, -1]).sum()        # rotation-only similarity
    else:
        rot_sim = refl_sim
    use = u @ vt if refl_sim - rot_sim > 0.15 else u_rot @ vt
    return A @ use


class SimulatedPen:
    """Synthesizes the prompted character from its font outline.

    Glyph outline -> resampled 2-D path -> differentiate twice -> front-accel
    channels (+ bias, noise, correlated other channels). This closes the loop:
    the reconstruction on screen should look like the prompted letter, which
    is exactly what tests assert.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.bias = np.concatenate([[0.02],                       # force rest
                                    self.rng.normal(0, 0.3, 3),   # accel bias
                                    self.rng.normal(0, 0.02, 12)])
        self.queue: List[np.ndarray] = []
        self.counter = 0
        self._last = time.monotonic()

    def glyph_path(self, char: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """(path, pen_down) of the character — see module-level glyph_path."""
        return glyph_path(char, n)

    def begin_stroke(self, char: str, duration: float = 1.4) -> None:
        n = int(duration * FS)
        path, down = self.glyph_path(char, n)
        acc = np.gradient(np.gradient(path, axis=0), axis=0) * FS * FS * 0.004
        rows = np.tile(self.bias, (n, 1)) + self.rng.normal(0, 0.01, (n, N_CHANNELS))
        rows[:, 1:3] += acc                                # front accel x, y
        rows[:, FORCE] += np.where(down, 0.9, 0.0) + self.rng.normal(0, .01, n)
        rows[:, 4:7] += _smooth(self.rng.normal(0, 0.05, (n, 3)))   # gyro
        rows[:, 10:13] = rows[:, 1:4] * 0.8                # rear accel echoes front
        self.queue.extend(rows)

    def read(self) -> List[List[float]]:
        """Rows since last call, at real-time rate: [counter, ch1..ch16]."""
        now = time.monotonic()
        n = int((now - self._last) * FS)
        if n <= 0:
            return []
        self._last = now
        out = []
        for _ in range(n):
            row = self.queue.pop(0) if self.queue else \
                self.bias + self.rng.normal(0, 0.01, N_CHANNELS)
            out.append([self.counter] + list(row))
            self.counter += 1
        return out

    def close(self) -> None:
        pass


class SerialPen:
    """Real pen over USB-C serial; one CSV line per sample: counter,ch1..ch16."""

    def __init__(self, port: str, baud: int = 115200):
        import serial                          # pyserial, only needed live
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = b""

    def read(self) -> List[List[float]]:
        self.buf += self.ser.read(65536)
        *lines, self.buf = self.buf.split(b"\n")
        out = []
        for ln in lines:
            parts = ln.decode(errors="ignore").strip().split(",")
            if len(parts) == 1 + N_CHANNELS:
                try:
                    out.append([float(v) for v in parts])
                except ValueError:
                    continue                    # torn line at startup
        return out

    def close(self) -> None:
        self.ser.close()


# --------------------------------------------------------------------------- #
# Session storage (procedure §6)
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, root: str, school: str, student: str, session_id: str,
                 prompts_csv: str, pen_serial: str, hand: str = "R"):
        self.dir = os.path.join(root, school, student, session_id)
        os.makedirs(self.dir, exist_ok=True)
        self.labels_path = os.path.join(self.dir, "labels.csv")
        with open(prompts_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        self.prompts = [("P", c) for c in PRACTICE] + \
                       [(int(r["box"]), r["label"]) for r in rows]
        meta = dict(student=student, school=school, session=session_id,
                    pen_serial=pen_serial, hand=hand, fs=FS,
                    channels=N_CHANNELS, prompts_csv=os.path.basename(prompts_csv),
                    started=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.done = set()
        if os.path.exists(self.labels_path):
            with open(self.labels_path, newline="") as f:
                self.done = {r["box"] for r in csv.DictReader(f)}
        else:
            with open(self.labels_path, "w", newline="") as f:
                csv.writer(f).writerow(["box", "label", "rec", "redo_count", "qc"])
        self.calib: Optional[dict] = None
        cpath = os.path.join(self.dir, "calibration.json")
        if os.path.exists(cpath):
            with open(cpath) as f:
                self.calib = json.load(f)

    def pending(self) -> List[Tuple[object, str]]:
        return [(b, c) for b, c in self.prompts if str(b) not in self.done]

    def save_calibration(self, rest: np.ndarray) -> dict:
        np.savetxt(os.path.join(self.dir, "calibration.csv"),
                   rest, delimiter=",", fmt="%.6f")
        self.calib = dict(force_mean=float(rest[:, 1 + FORCE].mean()),
                          force_std=float(rest[:, 1 + FORCE].std()),
                          bias=[float(v) for v in rest[:, 1:].mean(axis=0)],
                          accel_std=[float(v) for v in rest[:, 2:5].std(axis=0)],
                          rate_hz=float(len(rest) /
                                        max(rest[-1, 0] - rest[0, 0], 1) * 1e6))
        with open(os.path.join(self.dir, "calibration.json"), "w") as f:
            json.dump(self.calib, f, indent=2)
        return self.calib

    def accept(self, box, label: str, rows: np.ndarray, redo_count: int,
               qc: str) -> str:
        rec = f"rec_{box if box != 'P' else 'P' + label}.csv" \
            if box == "P" else f"rec_{box:05d}.csv"
        header = ("host_ts_us,counter,force,fax,fay,faz,fgx,fgy,fgz,"
                  "mx,my,mz,rax,ray,raz,rgx,rgy,rgz")
        np.savetxt(os.path.join(self.dir, rec), rows, delimiter=",",
                   fmt="%.6f", header=header, comments="")
        with open(self.labels_path, "a", newline="") as f:
            csv.writer(f).writerow([box, label, rec, redo_count, qc])
        self.done.add(str(box))
        return rec


# --------------------------------------------------------------------------- #
# GUI (imported lazily so headless tests can import this module)
# --------------------------------------------------------------------------- #
def run_gui(args) -> None:
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    pen = SimulatedPen() if args.simulate else SerialPen(args.port, args.baud)
    sess = Session(args.root, args.school, args.student, args.session,
                   args.prompts, args.pen_serial, args.hand)
    queue = sess.pending()

    root = tk.Tk()
    root.title(f"Vahini — {args.student}")
    if args.fullscreen:
        root.attributes("-fullscreen", True)
    root.configure(bg="white")

    info = tk.Label(root, text="", font=("Helvetica", 20), bg="white", fg="#555")
    info.pack(pady=(18, 0))
    prompt = tk.Label(root, text="", font=("Helvetica", 130, "bold"),
                      bg="white", fg="#111")
    prompt.pack(pady=4)
    msg = tk.Label(root, text="", font=("Helvetica", 18), bg="white", fg="#2a78d6")
    msg.pack()
    fig = Figure(figsize=(4.2, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(pady=8)
    progress = tk.Label(root, text="", font=("Helvetica", 16), bg="white", fg="#555")
    progress.pack(pady=(0, 14))

    state = {"mode": "calib_rest", "buf": [], "rows": [], "idle": 0,
             "redo": 0, "accepted": 0, "t0": time.monotonic(),
             "since_break": 0, "auto_at": None}

    def show_stroke(seq, char):
        ax.clear()
        ax.set_axis_off()
        calib_bias = np.asarray(sess.calib["bias"]) if sess.calib else None
        down = pen_down_mask(seq[:, FORCE],
                             sess.calib["force_mean"], sess.calib["force_std"]) \
            if sess.calib else None
        x, y, m = reconstruct_xy(seq, bias=calib_bias, mask=down)
        xy = orient_to_glyph(np.c_[x, y], str(char))     # display upright
        segs = np.split(xy, np.flatnonzero(np.diff(m.astype(int))) + 1)
        keep = m[0]
        for seg in segs:
            if keep and len(seg) > 1:
                ax.plot(seg[:, 0], seg[:, 1], color="#2a78d6", linewidth=3)
            keep = not keep
        ax.set_aspect("equal", adjustable="datalim")
        canvas.draw()

    def set_prompt():
        if not queue:
            info.config(text="")
            prompt.config(text="\N{WHITE MEDIUM STAR}" * 3)
            msg.config(text="All done — fantastic writing! Please call the teacher.")
            state["mode"] = "end"
            return
        box, char = queue[0]
        info.config(text="PRACTICE — just try it out!" if box == "P"
                    else f"Box {box}")
        prompt.config(text=char)
        kind = ("number" if str(char).isdigit()
                else "CAPITAL letter" if str(char).isupper() else "small letter")
        msg.config(text=f"Write this {kind} in the box, then lift the pen.")
        stars = "\N{WHITE MEDIUM STAR}" * (state["accepted"] * 10 // max(len(sess.prompts), 1))
        progress.config(text=f"{state['accepted']} / {len(sess.prompts)}  {stars}")
        state.update(mode="prompt", rows=[], idle=0)
        if args.simulate:
            root.after(700, lambda: pen.begin_stroke(str(char)))

    def start_break():
        state["mode"] = "break"
        state["since_break"] = 0
        end = time.monotonic() + args.break_secs

        def tick_break():
            left = int(end - time.monotonic())
            if left <= 0:
                set_prompt()
                return
            prompt.config(text=f"{left // 60}:{left % 60:02d}")
            info.config(text="BREAK TIME!")
            msg.config(text="Put the pen in the tray \N{BULLET} stand up \N{BULLET} "
                            "shake your hands \N{BULLET} stretch tall")
            root.after(500, tick_break)
        tick_break()

    def finish_sample():
        seq = np.asarray(state["rows"], dtype=float)     # [counter, ch1..16]
        rows16 = seq[:, 1:]
        ok, reason = qc_check(rows16, seq[:, 0].astype(int), sess.calib)
        if not ok:
            msg.config(text=reason)
            state.update(mode="prompt", rows=[], idle=0, redo=state["redo"] + 1)
            if args.simulate:
                root.after(700, lambda: pen.begin_stroke(str(queue[0][1])))
            return
        show_stroke(rows16, queue[0][1])
        msg.config(text="Look — that's yours!  (Enter = next, R = write it again)")
        state["mode"] = "review"
        state["auto_at"] = time.monotonic() + args.auto_accept / 1000.0

    def accept_current():
        box, char = queue.pop(0)
        seq = np.asarray(state["rows"], dtype=float)
        host0 = time.time() * 1e6
        full = np.c_[host0 + np.arange(len(seq)) / FS * 1e6, seq]
        sess.accept(box, str(char), full, state["redo"],
                    "practice" if box == "P" else "ok")
        state.update(redo=0, accepted=state["accepted"] + 1,
                     since_break=state["since_break"] + (box != "P"))
        if state["since_break"] >= args.break_every and queue:
            start_break()
        else:
            set_prompt()

    def redo_current():
        state.update(mode="prompt", rows=[], idle=0, redo=state["redo"] + 1)
        msg.config(text="No problem — write it once more!")
        if args.simulate:
            root.after(700, lambda: pen.begin_stroke(str(queue[0][1])))

    def on_key(ev):
        if state["mode"] == "review":
            if ev.keysym in ("Return", "space"):
                accept_current()
            elif ev.keysym.lower() == "r":
                redo_current()
        elif state["mode"] == "prompt" and ev.keysym.lower() == "b":
            start_break()                    # unscheduled break, any time
        elif ev.keysym == "Escape":
            root.destroy()                   # safe: session resumes next run

    root.bind("<Key>", on_key)

    def tick():
        rows = pen.read()
        mode = state["mode"]
        if mode == "calib_rest":
            info.config(text="Getting the pen ready")
            prompt.config(text="\N{SLEEPING SYMBOL}")
            msg.config(text="Please don't touch the pen — it is resting.")
            state["buf"].extend(rows)
            if len(state["buf"]) >= 5 * FS:
                rest = np.asarray(state["buf"], dtype=float)
                host0 = time.time() * 1e6
                calib = sess.save_calibration(
                    np.c_[host0 + np.arange(len(rest)) / FS * 1e6, rest])
                rate = calib["rate_hz"]
                state["buf"] = []
                if abs(rate) > 1 and not (FS * 0.95 < rate < FS * 1.05):
                    msg.config(text=f"warning: measured rate {rate:.0f} Hz")
                set_prompt()
        elif mode == "prompt":
            state["rows"].extend(rows)
            if sess.calib and state["rows"]:
                seq = np.asarray(state["rows"], dtype=float)
                down = pen_down_mask(seq[:, 1 + FORCE], sess.calib["force_mean"],
                                     sess.calib["force_std"])
                if down.any():
                    last_down = len(down) - 1 - int(np.argmax(down[::-1]))
                    idle = (len(down) - 1 - last_down) / FS
                    if idle > 0.7:                       # pen lifted long enough
                        first = int(np.argmax(down))
                        state["rows"] = [r for r in
                                         state["rows"][max(first - 10, 0):]]
                        finish_sample()
                elif len(state["rows"]) > 30 * FS:
                    state["rows"] = state["rows"][-10 * FS:]   # drop stale idle
        elif mode == "review":
            if state["auto_at"] and time.monotonic() > state["auto_at"]:
                state["auto_at"] = None
                accept_current()
        root.after(25, tick)

    tick()
    root.mainloop()
    pen.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--school", default="school01")
    ap.add_argument("--session", default="session_01")
    ap.add_argument("--prompts", required=True,
                    help="sheets_<code>_prompts.csv from tools/make_sheets.py")
    ap.add_argument("--pen-serial", required=True)
    ap.add_argument("--hand", choices=["L", "R"], default="R")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--port", help="pen serial port, e.g. /dev/ttyACM0 or COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--simulate", action="store_true",
                    help="no hardware: the app writes the prompts itself")
    ap.add_argument("--break-every", type=int, default=70,
                    help="accepted samples between child breaks")
    ap.add_argument("--break-secs", type=int, default=120)
    ap.add_argument("--auto-accept", type=int, default=1500,
                    help="ms to show the reconstruction before auto-advancing")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()
    if not args.simulate and not args.port:
        ap.error("provide --port for the real pen, or use --simulate")
    run_gui(args)


if __name__ == "__main__":
    main()
