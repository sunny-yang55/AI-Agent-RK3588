"""
Vision Pipeline

Unified vision processing pipeline.

Sprint B1.7.1把当前的 YOLOVision 单后端调用，升级为统一视觉处理流水线。
"""

from __future__ import annotations

from typing import Any

from .result import DetectionResult
from .vision_base import VisionBase


class VisionPipeline:

    def __init__(
        self,
        backend,
        preprocess=None,
        postprocess=None,
    ):

        self.backend = backend

        self.preprocess = preprocess
        self.postprocess = postprocess

    async def initialize(self):
        """
        Initialize backend.
        """

        await self.backend.initialize()

    async def process(
        self,
        image: Any,
        **kwargs,
    ) -> DetectionResult:
        """
        Process image through pipeline stages.
        Flow:
        image→preprocess→backend.detect→postprocess→DetectionResult
        """

        # 1. preprocess

        if self.preprocess:

            image = self.preprocess.run(image)

        # 2. backend inference

        result = await self.backend.detect(
            image,
            **kwargs,
        )

        # 3. postprocess

        if self.postprocess:

            result = self.postprocess.run(result)

        return result

    async def cleanup(self):
        """
        Cleanup backend resources.
        """

        await self.backend.cleanup()
