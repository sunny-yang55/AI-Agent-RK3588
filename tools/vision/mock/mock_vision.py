"""
功能：输入任意图片：
返回固定检测结果：
实现 MockVision。
注册 MockVision 到 ToolRegistry。
"""

"""
模拟视觉实现

用于框架测试的假视觉检测器。

目的：
- 验证 VisionBase 接口
- 验证 Executor 集成
- 验证 DetectionResult 流程

没有真实模型推理。
"""
import time
from typing import Any

from tools.vision.result import (
    BoundingBox,
    Detection,
    DetectionResult,
)
from tools.vision.vision_base import VisionBase


class MockVision(VisionBase):
    """
    Mock Vision Tool.

    Simulates object detection result.
    """

    name = "mock_vision"

    description = "Mock vision detector for testing"

    version = "0.1.0"

    capabilities = frozenset(
        {
            "object_detection",
            "mock",
        }
    )

    async def detect(self, image: Any, **kwargs) -> DetectionResult:
        """
        Simulate object detection.

        Parameters
        ----------
        image:
            Any input image placeholder.

        Returns
        -------
        DetectionResult
        """

        return DetectionResult(
            image_id="mock_image",
            timestamp=time.time(),
            detections=[
                Detection(
                    id=0,
                    label="person",
                    confidence=0.95,
                    bbox=BoundingBox(
                        xmin=100,
                        ymin=100,
                        xmax=200,
                        ymax=300,
                    ),
                )
            ],
            metadata={
                "model": "MockVision",
                "source": "simulation",
            },
        )
