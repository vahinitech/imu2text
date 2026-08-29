"""Download and unpack the Fraunhofer IIS OnHW dataset family.

The OnHW (Online HandWriting) datasets are recorded with a STABILO DigiPen
that contains two 3-axis accelerometers, a 3-axis gyroscope, a 3-axis
magnetometer, and a force sensor (13 channels at 100 Hz). They are
released by Fraunhofer IIS as direct-download ZIP archives at:

    https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html

This script lists the available archives, downloads one or more of them
into a directory of your choice, and extracts them. It does not convert
between pickle/npy formats - the loaders in ``imu2text/chars.py`` and
``imu2text/seq2seq.py`` handle the format-specific reading.

Usage
-----
    # list every available archive with its size and contents
    python -m imu2text.download --list

    # download just the small left-handed chars dataset (3.5 MB) for a smoke test
    python -m imu2text.download onhw_chars_L --out ./data

    # download the full right-handed chars dataset (896 MB) with official 5-fold splits
    python -m imu2text.download onhw_chars --out ./data

    # download every archive (several GB)
    python -m imu2text.download all --out ./data

Datasets
--------
The ``DATASETS`` dictionary below lists every published archive. The keys
are stable identifiers used by ``--dataset``; the values carry the URL, the
approximate size, and a short description. The names match the table in
``docs/onhw_enhancement_guide.md``.

| Key                       | Size    | Description                                       |
|---------------------------|---------|---------------------------------------------------|
| onhw_chars                | 896 MB  | OnHW-chars right-handed (.npy, 30 official splits)|
| onhw_chars_L              | 3.5 MB  | OnHW-chars left-handed (.pkl, no splits)          |
| onhw_symbols_dep          | 95 MB   | OnHW-symbols writer-dependent (.pkl, 5 folds)     |
| onhw_symbols_indep        | 95 MB   | OnHW-symbols writer-independent (.pkl, 5 folds)   |
| onhw_symbols_L            | 7.5 MB  | OnHW-symbols left-handed (.pkl)                   |
| onhw_equations_dep        | 1.1 GB  | OnHW-equations WD (.pkl, 5 folds)                 |
| onhw_equations_indep      | 1.1 GB  | OnHW-equations WI (.pkl, 5 folds)                 |
| onhw_equations_dep_ctc    | 1.0 GB  | OnHW-equations WD, per-symbol CTC split           |
| onhw_equations_indep_ctc  | 1.0 GB  | OnHW-equations WI, per-symbol CTC split           |
| onhw_words500_dep         | 849 MB  | OnHW-words500 WD (.pkl, 5 folds)                  |
| onhw_words500_indep       | 849 MB  | OnHW-words500 WI (.pkl, 5 folds)                  |
| onhw_words500_dep_L       | 36 MB   | OnHW-words500 WD left-handed                      |
| onhw_words500_indep_L     | 14 MB   | OnHW-words500 WI left-handed                      |
| onhw_wordsTraj_p1         | 1.0 GB  | OnHW-wordsTraj person 1 (trajectory regression)   |
| onhw_wordsTraj_p2         | 942 MB  | OnHW-wordsTraj person 2 (trajectory regression)   |
| icrow_dep                 | 103 MB  | ICROW comparison dataset, WD                      |
| icrow_indep               | 103 MB  | ICROW comparison dataset, WI                      |
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from typing import Dict, NamedTuple

ONHW_BASE = "https://www2.iis.fraunhofer.de/LV-OnHW"


class DatasetSpec(NamedTuple):
    url: str
    size: str
    description: str


DATASETS: Dict[str, DatasetSpec] = {
    # ---- OnHW-chars (character classification) ----
    "onhw_chars": DatasetSpec(
        url=f"{ONHW_BASE}/onhw-chars_2021-06-30.zip",
        size="896 MB",
        description="OnHW-chars right-handed: 31,275 samples, 119 writers, 52 "
        "classes. .npy format with 30 official splits "
        "(lower/upper/both x dep/indep x 5 folds).",
    ),
    "onhw_chars_L": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-chars_L.zip",
        size="3.5 MB",
        description="OnHW-chars left-handed: 2,270 samples, 9 writers, 52 "
        "classes. .pkl format, no splits (loaders infer writers).",
    ),
    # ---- OnHW-symbols (single symbol classification, 15 classes) ----
    "onhw_symbols_dep": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-symbols_equations_dep.zip",
        size="95 MB",
        description="OnHW-symbols writer-dependent: 2,326 single-symbol "
        "samples, 27 writers, 15 classes (digits 0-9 + +-:.:=).",
    ),
    "onhw_symbols_indep": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-symbols_equations_indep.zip",
        size="95 MB",
        description="OnHW-symbols writer-independent: same data, WI 5-fold.",
    ),
    "onhw_symbols_L": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-symbols_equations_L.zip",
        size="7.5 MB",
        description="OnHW-symbols + equations left-handed (small smoke-test set).",
    ),
    # ---- OnHW-equations (sequence-to-sequence, 15-class charset) ----
    "onhw_equations_dep": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-equations_dep.zip",
        size="1.1 GB",
        description="OnHW-equations writer-dependent: 10,713 equation samples, "
        "55 writers, 15-symbol charset. .pkl format, 5-fold CV.",
    ),
    "onhw_equations_indep": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-equations_indep.zip",
        size="1.1 GB",
        description="OnHW-equations writer-independent: same data, WI 5-fold.",
    ),
    "onhw_equations_dep_ctc": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-equations_dep_split_ctc.zip",
        size="1.0 GB",
        description="OnHW-equations WD, per-symbol CTC split: 39,643 "
        "single-symbol slices for CTC training.",
    ),
    "onhw_equations_indep_ctc": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-equations_indep_split_ctc.zip",
        size="1.0 GB",
        description="OnHW-equations WI, per-symbol CTC split.",
    ),
    # ---- OnHW-words500 (sequence-to-sequence, 57-char German vocab) ----
    "onhw_words500_dep": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-Words500_dep.zip",
        size="849 MB",
        description="OnHW-words500 writer-dependent: 25,218 samples, ~50 "
        "writers, 500-word closed vocabulary, 57-char charset "
        "(A-Za-z + German umlauts).",
    ),
    "onhw_words500_indep": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-Words500_indep.zip",
        size="849 MB",
        description="OnHW-words500 writer-independent.",
    ),
    "onhw_words500_dep_L": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-Words500_dep_L.zip",
        size="36 MB",
        description="OnHW-words500 WD left-handed.",
    ),
    "onhw_words500_indep_L": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-Words500_indep_L.zip",
        size="14 MB",
        description="OnHW-words500 WI left-handed.",
    ),
    # ---- OnHW-wordsTraj (trajectory regression, 2 writers) ----
    "onhw_wordsTraj_p1": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-wordsTraj_person1.zip",
        size="1.0 GB",
        description="OnHW-wordsTraj person 1: 4 sources (Wacom 30 Hz, 4 cameras "
        "60 Hz, IMU 100 Hz, pixel labels). Trajectory regression.",
    ),
    "onhw_wordsTraj_p2": DatasetSpec(
        url=f"{ONHW_BASE}/OnHW-wordsTraj_person2.zip",
        size="942 MB",
        description="OnHW-wordsTraj person 2.",
    ),
    # ---- ICROW (comparison dataset, adapted from IRONOFF) ----
    "icrow_dep": DatasetSpec(
        url=f"{ONHW_BASE}/ICROW_dep.zip",
        size="103 MB",
        description="ICROW writer-dependent: comparison dataset adapted from IRONOFF.",
    ),
    "icrow_indep": DatasetSpec(
        url=f"{ONHW_BASE}/ICROW_indep.zip",
        size="103 MB",
        description="ICROW writer-independent.",
    ),
}


def list_datasets() -> None:
    """Print the table of available datasets."""
    print(f"{'Key':<28} {'Size':<10} Description")
    print("-" * 100)
    for key, spec in DATASETS.items():
        print(f"{key:<28} {spec.size:<10} {spec.description}")


def _safe_extract(zf: zipfile.ZipFile, out_dir: str) -> None:
    """Extract every member, refusing any that would escape ``out_dir``.

    ``ZipFile.extractall`` sanitises absolute paths but a member named
    ``../../x`` still lands outside the target directory ("zip slip"). These
    archives come off the network, so each destination is resolved and
    checked against the output root before anything is written.
    """
    root = os.path.realpath(out_dir)
    for member in zf.infolist():
        dest = os.path.realpath(os.path.join(root, member.filename))
        if dest != root and not dest.startswith(root + os.sep):
            raise ValueError(
                f"refusing to extract {member.filename!r}: it would write to "
                f"{dest}, outside {root}"
            )
    zf.extractall(out_dir)


def _archive_root(zf: zipfile.ZipFile) -> str:
    """The single top-level directory of an archive, or "" if there isn't one.

    The Fraunhofer archives each unpack into one folder, but reading
    ``namelist()[0]`` assumes the first entry is inside it. Take the common
    first path segment across all members instead, so an archive with a
    stray root-level file (or a flat archive) is reported honestly.
    """
    tops = {name.split("/")[0] for name in zf.namelist() if name.strip("/")}
    return tops.pop() if len(tops) == 1 else ""


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """urllib callback: print a simple progress bar to stderr."""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100.0)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "#" * filled + "." * (bar_len - filled)
    sys.stderr.write(f"\r  [{bar}] {pct:5.1f}%  ")
    sys.stderr.flush()
    if pct >= 100.0:
        sys.stderr.write("\n")


def download_one(
    key: str, out_dir: str, extract: bool = True, skip_existing: bool = True
) -> str:
    """Download one dataset archive and (optionally) extract it.

    Returns the path to the extracted top-level folder.
    """
    if key not in DATASETS:
        raise KeyError(f"unknown dataset: {key!r}. Use --list to see options.")
    spec = DATASETS[key]
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"{key}.zip")

    if skip_existing and os.path.exists(zip_path):
        print(f"[skip] {zip_path} already exists ({os.path.getsize(zip_path)} bytes)")
    else:
        print(f"[get ] {spec.url} -> {zip_path}")
        urllib.request.urlretrieve(spec.url, zip_path, reporthook=_download_progress)

    if not extract:
        return zip_path

    print(f"[xtr ] {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract(zf, out_dir)
        top = _archive_root(zf)
    extracted_path = os.path.join(out_dir, top) if top else out_dir
    print(f"[done] extracted to {extracted_path}")
    return extracted_path


def main() -> None:
    """CLI: list the dataset catalog, or download and extract archives."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "datasets",
        nargs="*",
        default=[],
        help="dataset keys to download (use 'all' for every dataset, "
        "or omit and pass --list to just print the catalog)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the catalog of available datasets and exit",
    )
    ap.add_argument(
        "--out", default="./data", help="output directory (default: ./data)"
    )
    ap.add_argument(
        "--no-extract",
        action="store_true",
        help="download only, do not extract the ZIP archive",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the ZIP already exists locally",
    )
    args = ap.parse_args()

    if args.list or not args.datasets:
        list_datasets()
        if not args.datasets:
            print("\nPass dataset keys (or 'all') to download. Examples:")
            print(
                "  python -m imu2text.download onhw_chars_L --out ./data  # 3.5 MB smoke test"
            )
            print(
                "  python -m imu2text.download onhw_chars onhw_symbols_dep --out ./data"
            )
        return

    keys = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    for key in keys:
        try:
            download_one(
                key, args.out, extract=not args.no_extract, skip_existing=not args.force
            )
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {key}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
