"""Convert a raw Vahini collection tree into the training pickles.

Expected raw layout (see docs/school_collection_procedure.md §6):

    <raw>/<school>/<student_code>/<session>/
        labels.csv          # box,label,rec,redo_count,qc
        rec_*.csv           # host_ts_us,counter,<16 sensor channels>

Only rows with qc == "ok" are exported. Output (into --out):

    all_x_dat_imu.pkl   list of (T, channels) float32 arrays
    all_gt.pkl          list of label strings (same order)
    writers.pkl         list of student codes  (same order)

These plug straight into the training scripts:

    python onhw_models.py --channels 16 \
        --imu-file  <out>/all_x_dat_imu.pkl \
        --gt-file   <out>/all_gt.pkl \
        --writers-file <out>/writers.pkl --split writer
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np

N_META_COLS = 2  # host_ts_us, counter precede the sensor channels


def read_recording(path: str, channels: int) -> np.ndarray:
    """Read one rec_*.csv -> (T, channels) float32 array (validates counter)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if rows and not rows[0][0].lstrip("-").replace(".", "").isdigit():
        rows = rows[1:]                                    # skip header row
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != N_META_COLS + channels:
        raise ValueError(f"{path}: {data.shape[1]} columns, "
                         f"expected {N_META_COLS + channels}")
    counter = data[:, 1].astype(np.int64)
    dropped = int(np.sum(np.diff(counter) != 1))
    if dropped:
        print(f"  warning: {path}: {dropped} sample-counter gaps (dropped packets)")
    return data[:, N_META_COLS:].astype(np.float32)


def walk_sessions(raw: str):
    """Yield (student_code, session_dir) for every session with a labels.csv."""
    for school in sorted(os.listdir(raw)):
        school_dir = os.path.join(raw, school)
        if not os.path.isdir(school_dir):
            continue
        for student in sorted(os.listdir(school_dir)):
            student_dir = os.path.join(school_dir, student)
            if not os.path.isdir(student_dir):
                continue
            for session in sorted(os.listdir(student_dir)):
                session_dir = os.path.join(student_dir, session)
                if os.path.isfile(os.path.join(session_dir, "labels.csv")):
                    yield student, session_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="root of the raw collection tree")
    ap.add_argument("--out", required=True, help="output directory for the pickles")
    ap.add_argument("--channels", type=int, default=16,
                    help="sensor channels per timestep (16 = Vahini pen)")
    args = ap.parse_args()

    x, gt, writers = [], [], []
    n_sessions = n_skipped = 0
    for student, session_dir in walk_sessions(args.raw):
        n_sessions += 1
        with open(os.path.join(session_dir, "labels.csv"), newline="") as f:
            for row in csv.DictReader(f):
                if row["qc"].strip().lower() != "ok":
                    n_skipped += 1
                    continue
                rec_path = os.path.join(session_dir, row["rec"])
                x.append(read_recording(rec_path, args.channels))
                gt.append(row["label"])
                writers.append(student)

    if not x:
        sys.exit(f"no accepted samples found under {args.raw}")

    os.makedirs(args.out, exist_ok=True)
    for name, obj in [("all_x_dat_imu.pkl", x), ("all_gt.pkl", gt),
                      ("writers.pkl", writers)]:
        with open(os.path.join(args.out, name), "wb") as f:
            pickle.dump(obj, f)

    lengths = [len(s) for s in x]
    print(f"sessions: {n_sessions} | accepted samples: {len(x)} "
          f"| skipped (qc!=ok): {n_skipped}")
    print(f"writers: {len(set(writers))} | classes: {len(set(gt))} "
          f"| seq len min/mean/max: {min(lengths)}/{np.mean(lengths):.0f}/{max(lengths)}")
    print(f"wrote {args.out}/all_x_dat_imu.pkl, all_gt.pkl, writers.pkl")


if __name__ == "__main__":
    main()
