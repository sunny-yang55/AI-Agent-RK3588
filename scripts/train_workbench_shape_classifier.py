#!/usr/bin/env python3
"""Train and evaluate a local three-shape classifier from labelled ROI images.

The dataset must be arranged as:
  datasets/workbench_shapes/<colour>_<shape>/<train|val|test>/*.jpg

The saved model is intentionally not activated by the voice runtime.  Review
the generated report first; only a model with strong independent test results
may be promoted to mechanical-arm target selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/workbench_shapes")
    parser.add_argument("--model", default="models/vision/workbench_shape_svm.xml")
    parser.add_argument("--report", default="reports/vision-workbench/shape-model-report.json")
    return parser.parse_args()


def load_split(dataset: Path, split: str):
    import cv2

    from tools.vision.shape_classifier import SHAPE_LABELS, extract_shape_features

    features, labels, sample_paths, errors, counts = [], [], [], [], Counter()
    for colour_shape_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        shape = next(
            (
                candidate
                for candidate in SHAPE_LABELS
                if colour_shape_dir.name.endswith("_" + candidate)
            ),
            None,
        )
        if shape is None:
            continue
        for path in sorted((colour_shape_dir / split).glob("*.jpg")):
            image = cv2.imread(str(path))
            try:
                features.append(extract_shape_features(image).values)
                labels.append(SHAPE_LABELS.index(shape))
                sample_paths.append(path.relative_to(ROOT).as_posix())
                counts[colour_shape_dir.name] += 1
            except ValueError as exc:
                errors.append(f"{path}: {exc}")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        sample_paths,
        counts,
        errors,
    )


def evaluate(model, features: np.ndarray, labels: np.ndarray, sample_paths: list[str]) -> dict:
    from tools.vision.shape_classifier import SHAPE_LABELS

    if not len(labels):
        return {
            "samples": 0,
            "accuracy": None,
            "per_shape": {},
            "confusion_matrix": {},
            "misclassified": [],
        }
    _ok, predicted = model.predict(features)
    predicted = predicted.reshape(-1).astype(np.int32)
    per_shape = {}
    confusion_matrix = {}
    for index, label in enumerate(SHAPE_LABELS):
        mask = labels == index
        per_shape[label] = {
            "samples": int(mask.sum()),
            "accuracy": round(float((predicted[mask] == labels[mask]).mean()), 4) if mask.any() else None,
        }
        confusion_matrix[label] = {
            predicted_label: int(((labels == index) & (predicted == predicted_index)).sum())
            for predicted_index, predicted_label in enumerate(SHAPE_LABELS)
        }
    misclassified = [
        {
            "actual": SHAPE_LABELS[int(actual)],
            "predicted": SHAPE_LABELS[int(prediction)],
            "image": path,
        }
        for actual, prediction, path in zip(labels, predicted, sample_paths)
        if actual != prediction
    ]
    return {
        "samples": int(len(labels)),
        "accuracy": round(float((predicted == labels).mean()), 4),
        "per_shape": per_shape,
        "confusion_matrix": confusion_matrix,
        "misclassified": misclassified,
    }


def main() -> int:
    import cv2

    from tools.vision.shape_classifier import SHAPE_LABELS

    args = parse_args()
    dataset = ROOT / args.dataset
    if not dataset.is_dir():
        print(f"[ShapeTrain] dataset not found: {dataset}")
        return 2
    train_x, train_y, _train_paths, train_counts, train_errors = load_split(dataset, "train")
    required = {label for label in SHAPE_LABELS if not (train_y == SHAPE_LABELS.index(label)).any()}
    if required:
        print(f"[ShapeTrain] missing train samples for: {', '.join(sorted(required))}")
        return 2
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    train_x = (train_x - mean) / scale
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setC(2.0)
    model.setGamma(0.5)
    model.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    output_model = ROOT / args.model
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model))
    metadata_path = output_model.with_suffix(".json")
    metadata_path.write_text(
        json.dumps({"labels": SHAPE_LABELS, "mean": mean.tolist(), "scale": scale.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {"train_counts": dict(train_counts), "discarded": train_errors}
    for split in ("train", "val", "test"):
        features, labels, sample_paths, _counts, errors = load_split(dataset, split)
        report[split] = evaluate(model, (features - mean) / scale, labels, sample_paths)
        report["discarded"].extend(errors)
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ShapeTrain] model={output_model.relative_to(ROOT)}")
    print(f"[ShapeTrain] report={report_path.relative_to(ROOT)}")
    print(f"[ShapeTrain] test_accuracy={report['test']['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
