"""Research CLI for SIP feature tables, not a paper reproduction or clinical tool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imu2text import sip_data

WARNINGS = [
    "Exploratory feature-table baseline, not exact paper reproduction.",
    "Z-based task labels are not diagnoses or proof of screening validity.",
    "No raw signals, trace reconstruction or Vahini hardware validation.",
    "Single held-out split; no external cohort or uncertainty estimate.",
]


def read_groups(path: Path, rows: int) -> np.ndarray:
    """Read complete zero-based row mapping; reject blanks and ambiguous joins."""
    mapping = pd.read_csv(path, dtype={"subject_id": str}, keep_default_na=False)
    if list(mapping.columns) != ["row_index", "subject_id"]:
        raise ValueError("Groups CSV must have row_index,subject_id columns")
    if len(mapping) != rows or mapping["row_index"].duplicated().any():
        raise ValueError("Groups CSV must cover each table row exactly once")
    mapping = mapping.sort_values("row_index")
    if mapping["row_index"].tolist() != list(range(rows)):
        raise ValueError("Groups CSV must cover all zero-based row indices")
    groups = mapping["subject_id"].str.strip().to_numpy()
    if any(not value for value in groups):
        raise ValueError("Subject codes cannot be blank")
    return groups


def partition(labels, groups, seed, exploratory):
    """Choose one split without searching test performance or target balance."""
    indices = np.arange(len(labels))
    if groups is not None:
        if (
            len(groups) != len(labels)
            or pd.isna(groups).any()
            or any(not str(value).strip() for value in groups)
        ):
            raise ValueError("Each row needs a nonmissing subject code")
        train, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                indices, labels, groups
            )
        )
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Subject overlap between train and test")
    elif exploratory:
        train, test = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels
        )
    else:
        raise ValueError(
            "Subject IDs are absent. Supply --groups or explicitly choose "
            "--exploratory-row-split; row splits are not writer-independent."
        )
    if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
        raise ValueError("Both classes are required in each split; revise the protocol")
    return train, test


def evaluate(model, features, labels):
    """Report positive-class metrics and counts without individual predictions."""
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probability)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, prediction, labels=[0, 1]
        ).tolist(),
    }


def baseline(frame, task, seed=0, groups=None, exploratory=False):
    """Fit train-only imputation/scaling and fixed logistic/dummy baselines."""
    if task not in ("UNO_B", "UNO_C", "NUM_B", "NUM_C"):
        raise ValueError("Unknown task")
    frame = sip_data.validate_frame(frame)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(frame):
            raise ValueError("Groups must map the original, unfiltered table")
    # First graders have limited cursive exposure; document this fixed exclusion.
    grade_eligible = frame["year"].to_numpy() >= (2 if task.endswith("_C") else 1)
    labelled = frame["Z"].notna().to_numpy()
    eligible = grade_eligible & labelled
    source_rows = np.flatnonzero(eligible)
    selected = frame.loc[eligible]
    if groups is not None:
        groups = groups[eligible]
    labels = (selected["Z"].to_numpy() <= -1.5).astype(int)
    train, test = partition(labels, groups, seed, exploratory)
    features = selected[sip_data.FEATURES].to_numpy(dtype=float)
    # Identical feature records across the split warrant investigation, not a score.
    hashes = pd.util.hash_pandas_object(
        selected[sip_data.FEATURES], index=False
    ).to_numpy()
    if set(hashes[train]) & set(hashes[test]):
        raise ValueError(
            "Identical feature rows cross the split; investigate duplicate records"
        )
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", random_state=seed, max_iter=2000),
    )
    with catch_warnings():
        simplefilter("error", ConvergenceWarning)
        model.fit(features[train], labels[train])
    dummy = DummyClassifier(strategy="most_frequent").fit(
        features[train], labels[train]
    )
    warnings = list(WARNINGS)
    if groups is None:
        warnings.append(
            "Row split only: unknown subject overlap; not writer-independent."
        )
    else:
        warnings.append(
            "Subject grouping relies on the caller's verified pseudonymous mapping."
        )
    split_payload = json.dumps(
        {"train": source_rows[train].tolist(), "test": source_rows[test].tolist()}
    )
    report = {
        "task": task,
        "seed": seed,
        "label_threshold": -1.5,
        "classes": 2,
        "positive_class": "Z <= -1.5",
        "feature_count": len(sip_data.FEATURES),
        "excluded_predictors": sip_data.COLUMNS[:11],
        "excluded_first_grade_cursive_rows": int((~grade_eligible).sum()),
        "excluded_missing_target_rows": int((~labelled).sum()),
        "total_excluded_rows": int((~eligible).sum()),
        "split": (
            "subject-grouped 80/20"
            if groups is not None
            else "exploratory stratified row 80/20"
        ),
        "split_sha256": hashlib.sha256(split_payload.encode()).hexdigest(),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive": int(labels[train].sum()),
        "test_positive": int(labels[test].sum()),
        "logistic_hyperparameters": model.named_steps[
            "logisticregression"
        ].get_params(),
        "logistic_iterations": model.named_steps["logisticregression"].n_iter_.tolist(),
        "logistic_regression": evaluate(model, features[test], labels[test]),
        "most_frequent_dummy": evaluate(dummy, features[test], labels[test]),
        "protocol": (
            "No tuning; global training-set median and scale, "
            "not paper within-grade scaling."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "warnings": warnings,
    }
    return report, model


def main(argv=None):
    """Run download, aggregate audit or an explicitly scoped research baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Fetch five pinned CC BY 4.0 files")
    fetch.add_argument("--directory", type=Path, default=Path("data/sip-19485897"))
    for name in ("inspect", "baseline"):
        command = commands.add_parser(name)
        command.add_argument("table", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "baseline":
            identity = command.add_mutually_exclusive_group()
            identity.add_argument("--groups", type=Path)
            identity.add_argument("--exploratory-row-split", action="store_true")
            command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "download":
        print(json.dumps(sip_data.download(args.directory), indent=2))
        return
    if args.output.suffix != ".json":
        parser.error("--output must be a .json report, not an input workbook")
    inputs = [args.table]
    if args.command == "baseline" and args.groups:
        inputs.append(args.groups)
    if args.output.resolve() in [path.resolve() for path in inputs]:
        parser.error("--output cannot overwrite an input file")
    frame = sip_data.load_table(args.table)
    if args.command == "inspect":
        report = sip_data.describe(frame)
    else:
        groups = read_groups(args.groups, len(frame)) if args.groups else None
        report, _ = baseline(
            frame, args.table.stem, args.seed, groups, args.exploratory_row_split
        )
    report["input"] = sip_data.verify_file(args.table)
    report["source"] = sip_data.RECORD
    report["license"] = sip_data.LICENSE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote aggregate research report: {args.output}")


if __name__ == "__main__":
    main()
