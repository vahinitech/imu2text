"""Acquire and validate the published SIP feature tables, never raw pen signals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import pandas as pd

RECORD = "https://zenodo.org/records/19485897"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
FILES = {
    "UNO_B.xlsx": (1503433, "c64ceff325457e8f9344cacb25a63b40"),
    "UNO_C.xlsx": (1460919, "a37e1f22c724b24b3985315c39ee7581"),
    "NUM_B.xlsx": (1510764, "9b57d71b75043ac08458a2b775af6efd"),
    "NUM_C.xlsx": (1453946, "33ebf96a5534d1a611b2a25b5e63f5cd"),
    "indicators_description.docx": (32646, "fe3663462a3176aebc3977cebd288aa9"),
}
COLUMNS = json.loads(
    Path(__file__).with_name("sip_schema.json").read_text(encoding="utf-8")
)
FEATURES = COLUMNS[11:]  # Exclude ten context/target fields and ExecutionTime.


def verify_file(path: Path) -> dict:
    """Match an allowlisted v1 file to its publisher size and checksum."""
    if path.name not in FILES:
        raise ValueError(f"Not a SIP v1 file: {path.name}")
    size, checksum = FILES[path.name]
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected size: {path.name}")
    content = path.read_bytes()
    if hashlib.md5(content, usedforsecurity=False).hexdigest() != checksum:
        raise ValueError(f"Publisher checksum mismatch: {path.name}")
    return {
        "file": path.name,
        "bytes": size,
        "publisher_md5": checksum,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def download(directory: Path) -> dict:
    """Fetch only pinned files; never replace an existing mismatched input."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, (size, _) in FILES.items():
        destination = directory / name
        if not destination.exists():
            url = f"{RECORD}/files/{name}?download=1"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read(size + 1)
            # A temporary directory allows checksum verification before publication.
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                candidate = Path(temporary) / name
                candidate.write_bytes(content)
                verify_file(candidate)
                candidate.replace(destination)
        entries.append(verify_file(destination))
    manifest = {
        "record": RECORD,
        "doi": "10.5281/zenodo.19485897",
        "version": "v1",
        "license": LICENSE,
        "license_metadata_checked": "2026-11-18",
        "metadata_source": "https://zenodo.org/api/records/19485897",
        "creators": [
            "Linda Greta Dui",
            "Simone Toffoli",
            "Stefania Fontolan",
            "Chiara Piazzalunga",
            "Cristiano Termine",
            "Simona Ferrante",
        ],
        "article": "https://doi.org/10.1371/journal.pdig.0001503",
        "changes": "Original files unchanged; analysis reports are derived outputs.",
        "files": entries,
    }
    (directory / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail on schema drift, bad targets/grades, infinities or text features."""
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError("Expected the ordered 189-column SIP v1 schema and data rows")
    frame = frame.copy()
    for column in ["year", "Z", "ExecutionTime", *FEATURES]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {column}")
    if frame["year"].isna().any():
        raise ValueError("Grade must be present on every row")
    if not frame["year"].isin(range(1, 9)).all():
        raise ValueError("Grade must be an integer from 1 to 8")
    if (frame["ExecutionTime"].dropna() <= 0).any():
        raise ValueError("ExecutionTime must be positive when present")
    return frame


def load_table(path: Path) -> pd.DataFrame:
    """Read one verified table, without pooling tasks or inventing subject IDs."""
    if path.suffix != ".xlsx":
        raise ValueError("Expected a task .xlsx file")
    verify_file(path)
    return validate_frame(pd.read_excel(path, engine="openpyxl", sheet_name="Sheet1"))


def describe(frame: pd.DataFrame) -> dict:
    """Return aggregates only; no participant rows or identifiers are exported."""
    frame = validate_frame(frame)
    features = frame[FEATURES]
    duplicate_rows = features.duplicated(keep=False)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "feature_count": len(FEATURES),
        "subject_ids_available": False,
        "duplicate_feature_rows": int(duplicate_rows.sum()),
        "duplicate_feature_groups": int(
            features.loc[duplicate_rows].drop_duplicates().shape[0]
        ),
        "label": "Z <= -1.5: task-specific low-speed group, not a diagnosis",
        "labelled_rows": int(frame["Z"].notna().sum()),
        "missing_target_rows": int(frame["Z"].isna().sum()),
        "low_speed_rows": int((frame["Z"] <= -1.5).sum()),
        "grade_counts": {
            str(k): int(v) for k, v in frame["year"].value_counts().sort_index().items()
        },
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items() if v},
        "feature_summary": {
            column: {
                "observed": int(features[column].notna().sum()),
                "median": (
                    None
                    if features[column].isna().all()
                    else float(features[column].median())
                ),
            }
            for column in FEATURES
        },
    }
