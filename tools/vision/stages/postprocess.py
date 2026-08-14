"""
Vision Postprocess Stage

Sprint B1.7.2
统一检测结果格式
后端解耦
为 RKNN YOLO 做准备
"""

from __future__ import annotations

from typing import Any


class PostprocessStage:
    """
    Vision result postprocessing.

    Responsibilities:
    - normalize backend output
    - attach metadata
    """

    name = "postprocess"

    def run(
        self,
        result: Any,
    ):
        """
        Process backend result.

        Parameters
        ----------
        result:
            backend detection result

        Returns
        -------
        Any
            processed result
        """

        if result is None:
            raise ValueError("Detection result is None")

        return result
