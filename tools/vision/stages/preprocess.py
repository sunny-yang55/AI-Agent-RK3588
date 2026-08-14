"""
Vision Preprocess Stage

Sprint B1.7.2
支持两种图片输入：图片路径、直接传图片数据
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class PreprocessStage:
    """
    Vision input preprocessing.

    Responsibilities:
    - validate input
    - load image
    - normalize format
    """

    name = "preprocess"

    def run(
        self,
        image: Any,
    ) -> np.ndarray:
        """
        Execute preprocessing.

        Parameters
        ----------
        image:
            image path or numpy image.

        Returns
        -------
        np.ndarray
            BGR image array.
        """

        # Case 1:
        # numpy image input
        if isinstance(image, np.ndarray):
            return image

        # Case 2:
        # image path input
        if isinstance(
            image,
            (str, Path),
        ):

            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            img = cv2.imread(str(image_path))

            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            return img

        raise TypeError(f"Unsupported image type: {type(image)}")
