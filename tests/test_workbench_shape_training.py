"""Guardrails for the labelled workbench shape training pipeline."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkbenchShapeTrainingTests(unittest.TestCase):
    def test_training_uses_independent_dataset_splits_and_persists_report(self):
        source = (ROOT / "scripts/train_workbench_shape_classifier.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('for split in ("train", "val", "test")', source)
        self.assertIn('cv2.ml.SVM_create()', source)
        self.assertIn('shape-model-report.json', source)
        self.assertIn('"confusion_matrix"', source)
        self.assertIn('"misclassified"', source)
        self.assertIn('"per_class"', source)
        self.assertIn('choices=("contour", "hog", "hybrid")', source)
        self.assertIn('augmented_images', source)

    def test_feature_module_is_colour_agnostic_and_has_three_labels(self):
        source = (ROOT / "tools/vision/shape_classifier.py").read_text(encoding="utf-8")
        self.assertIn('SHAPE_LABELS = ("cube", "cylinder", "triangular_pyramid")', source)
        self.assertIn('cv2.HuMoments', source)
        self.assertIn('extract_hog_shape_features', source)
        self.assertIn('cv2.HOGDescriptor', source)
        self.assertIn('largest saturated object contour', source)


if __name__ == "__main__":
    unittest.main()
