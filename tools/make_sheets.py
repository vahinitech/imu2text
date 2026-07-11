"""Generate the pre-printed A4 prompt sheets (PDF) for a student.

Produces, per student code:
  sheets_<CODE>.pdf          7 printable A4 pages, 48 boxes each
  sheets_<CODE>_prompts.csv  box,label — THE ground-truth prompt list

The capture program MUST play prompts from the CSV, in order. The PDF and the
CSV are generated together from the same seeded shuffle, so paper and screen
can never disagree (docs/school_collection_procedure.md §3).

Layout per box: bold box number (top-left), small reference character
(top-right), four-line copybook ruling in the writing area. Sheet 1 starts
with a shaded PRACTICE row (recordings discarded automatically). Boxes after
the last prompt are marked as finished.

Usage:
  python tools/make_sheets.py --students S0001 S0002 --out sheets/
  python tools/make_sheets.py --students S0001 --reps 3   # shorter session
"""
from __future__ import annotations

import argparse
import csv
import os
import string
import zlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# A4 in inches, portrait
PAGE_W, PAGE_H = 8.27, 11.69
MARGIN = 0.45
HEADER_H = 0.75
COLS, ROWS = 6, 8
BOXES_PER_SHEET = COLS * ROWS
PRACTICE = list("olA3xT")          # fixed practice prompts, row 1 of sheet 1

CELL_W = (PAGE_W - 2 * MARGIN) / COLS
CELL_H = (PAGE_H - 2 * MARGIN - HEADER_H) / ROWS


def build_prompts(student: str, charset: str, reps: int) -> list:
    """Seeded shuffle of charset x reps; no immediate repeats; stable per code."""
    seed = zlib.crc32(student.encode())          # reproducible from the code alone
    rng = np.random.default_rng(seed)
    prompts = list(charset) * reps
    rng.shuffle(prompts)
    for i in range(1, len(prompts)):             # break up accidental AA pairs
        if prompts[i] == prompts[i - 1]:
            j = (i + 1) % len(prompts)
            while prompts[j] == prompts[i - 1] or \
                    (j + 1 < len(prompts) and prompts[j + 1] == prompts[i]):
                j = (j + 1) % len(prompts)
                if j == i:
                    break
            prompts[i], prompts[j] = prompts[j], prompts[i]
    return prompts


def draw_box(ax, x, y, number, char, practice=False, finished=False):
    """One writing box at lower-left (x, y). Coordinates in inches."""
    face = "0.92" if practice else "white"
    ax.add_patch(Rectangle((x, y), CELL_W, CELL_H, fill=True, facecolor=face,
                           edgecolor="0.35", linewidth=0.8))
    if finished:
        ax.text(x + CELL_W / 2, y + CELL_H / 2, "★", ha="center",
                va="center", fontsize=16, color="0.75")
        return
    label_h = 0.22                                   # strip for number + glyph
    top = y + CELL_H
    if practice:
        ax.text(x + 0.05, top - 0.16, "PRACTICE", fontsize=5.5, color="0.4")
    else:
        ax.text(x + 0.05, top - 0.18, str(number), fontsize=8,
                fontweight="bold", color="0.15")
    ax.text(x + CELL_W - 0.07, top - 0.18, char, fontsize=11, ha="right",
            color="0.25", family="DejaVu Sans")
    # four-line copybook ruling: ascender / midline / BASELINE / descender
    area_top, area_bot = top - label_h, y + 0.06
    h = area_top - area_bot
    for frac, lw, col in [(0.92, 0.5, "0.75"), (0.62, 0.5, "0.75"),
                          (0.32, 0.9, "0.55"), (0.06, 0.5, "0.75")]:
        yy = area_bot + frac * h
        ax.plot([x + 0.07, x + CELL_W - 0.07], [yy, yy],
                linestyle=(0, (1.5, 2.5)), linewidth=lw, color=col)


def make_student_sheets(student: str, charset: str, reps: int, out_dir: str) -> str:
    prompts = build_prompts(student, charset, reps)
    n_sheets = int(np.ceil((len(prompts) + len(PRACTICE)) / BOXES_PER_SHEET))

    csv_path = os.path.join(out_dir, f"sheets_{student}_prompts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["box", "label"])
        w.writerows((i + 1, ch) for i, ch in enumerate(prompts))

    pdf_path = os.path.join(out_dir, f"sheets_{student}.pdf")
    slots = [("P", c) for c in PRACTICE] + \
            [(i + 1, c) for i, c in enumerate(prompts)]
    with PdfPages(pdf_path) as pdf:
        for sheet in range(n_sheets):
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, PAGE_W), ax.set_ylim(0, PAGE_H)
            ax.axis("off")
            ax.text(MARGIN, PAGE_H - MARGIN - 0.05,
                    f"VAHINI PEN STUDY  —  Sheet {sheet + 1} of {n_sheets}",
                    fontsize=11, fontweight="bold", va="top")
            ax.text(MARGIN, PAGE_H - MARGIN - 0.32,
                    f"Student: {student}     Pen serial: ________     "
                    f"Date: ________     Hand: L / R",
                    fontsize=8.5, va="top", color="0.25")
            ax.text(PAGE_W / 2, MARGIN * 0.45,
                    "Write ONE character in each box — only when the "
                    "screen shows that box number.",
                    fontsize=7.5, ha="center", color="0.35")
            for k in range(BOXES_PER_SHEET):
                idx = sheet * BOXES_PER_SHEET + k
                row, col = divmod(k, COLS)
                x = MARGIN + col * CELL_W
                y = PAGE_H - MARGIN - HEADER_H - (row + 1) * CELL_H
                if idx < len(slots):
                    number, ch = slots[idx]
                    draw_box(ax, x, y, number, ch, practice=(number == "P"))
                else:
                    draw_box(ax, x, y, None, None, finished=True)
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--students", nargs="+", required=True,
                    help="student codes, e.g. S0001 S0002")
    ap.add_argument("--out", default="sheets", help="output directory")
    ap.add_argument("--charset",
                    default=string.ascii_uppercase + string.ascii_lowercase
                    + string.digits,
                    help="prompt characters (default: A-Z a-z 0-9 = 62)")
    ap.add_argument("--reps", type=int, default=5,
                    help="repetitions per character (default 5 -> 310 boxes)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for student in args.students:
        pdf = make_student_sheets(student, args.charset, args.reps, args.out)
        print(f"{student}: {len(args.charset) * args.reps} prompts -> {pdf}")


if __name__ == "__main__":
    main()
