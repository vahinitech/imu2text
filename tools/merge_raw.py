"""Merge ("club") raw trees from multiple capture stations into one master tree.

With 5 pens running simultaneously, each station laptop produces its own
`data/raw/<school>/<student>/<session>/` tree. At the end of the session, copy
each laptop's tree onto the master drive and run:

    python tools/merge_raw.py --stations /media/usb/station1 /media/usb/station2 ... \
        --master data/raw

Because every session directory is uniquely keyed by (school, student_code,
session), merging is a safe copy — UNLESS two stations claim the same student
and session, which means a student-code collision (two children given the
same code, or one child recorded at two stations). The tool detects that and
refuses to guess: identical duplicates are skipped, conflicting ones are
reported for a human to resolve.

Every merge is appended to <master>/merge_log.csv, and per-pen sample counts
are printed so the pen-rotation balance (procedure §11) is visible after
every session.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
import sys
import time
from collections import Counter


def find_sessions(root: str):
    """Yield (school, student, session, abs_path) below a raw tree root."""
    for school in sorted(os.listdir(root)):
        sdir = os.path.join(root, school)
        if not os.path.isdir(sdir):
            continue
        for student in sorted(os.listdir(sdir)):
            stdir = os.path.join(sdir, student)
            if not os.path.isdir(stdir):
                continue
            for session in sorted(os.listdir(stdir)):
                sess = os.path.join(stdir, session)
                if os.path.isfile(os.path.join(sess, "labels.csv")):
                    yield school, student, session, sess


def session_stats(sess: str):
    """(n_ok, pen_serial) for one session directory."""
    with open(os.path.join(sess, "labels.csv"), newline="") as f:
        n_ok = sum(1 for r in csv.DictReader(f)
                   if (r.get("qc") or "").strip().lower() == "ok")
    pen = ""
    meta = os.path.join(sess, "meta.json")
    if os.path.isfile(meta):
        with open(meta) as f:
            pen = json.load(f).get("pen_serial", "")
    return n_ok, pen


def trees_identical(a: str, b: str) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(trees_identical(os.path.join(a, d), os.path.join(b, d))
               for d in cmp.common_dirs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stations", nargs="+", required=True,
                    help="raw tree roots copied from each station laptop")
    ap.add_argument("--master", required=True, help="master raw tree")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen; copy nothing")
    args = ap.parse_args()

    os.makedirs(args.master, exist_ok=True)
    merged, skipped, conflicts = [], [], []
    pen_counts: Counter = Counter()

    for station in args.stations:
        if not os.path.isdir(station):
            sys.exit(f"station tree not found: {station}")
        for school, student, session, src in find_sessions(station):
            dest = os.path.join(args.master, school, student, session)
            n_ok, pen = session_stats(src)
            if os.path.exists(dest):
                if trees_identical(src, dest):
                    skipped.append(dest)          # already merged earlier
                    continue
                conflicts.append((station, school, student, session))
                continue
            if not args.dry_run:
                shutil.copytree(src, dest)
                # verify the copy before trusting it
                if not trees_identical(src, dest):
                    sys.exit(f"copy verification FAILED for {dest} — "
                             f"master may be on a failing drive")
            merged.append((station, school, student, session, n_ok, pen))
            pen_counts[pen or "unknown"] += n_ok

    if not args.dry_run and merged:
        log = os.path.join(args.master, "merge_log.csv")
        new = not os.path.exists(log)
        with open(log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "station", "school", "student", "session",
                            "ok_samples", "pen_serial"])
            for st, sc, stu, se, n, pen in merged:
                w.writerow([time.strftime("%Y-%m-%d %H:%M"), st, sc, stu,
                            se, n, pen])

    print(f"merged: {len(merged)} sessions "
          f"({sum(m[4] for m in merged)} ok samples) | "
          f"already present: {len(skipped)} | conflicts: {len(conflicts)}")
    if pen_counts:
        print("samples per pen this merge:",
              dict(sorted(pen_counts.items())))
    if conflicts:
        print("\nCONFLICTS — same student+session with DIFFERENT content; "
              "resolve by hand (was a student code reused?):")
        for st, sc, stu, se in conflicts:
            print(f"  {sc}/{stu}/{se}   (from {st})")
        sys.exit(1)


if __name__ == "__main__":
    main()
