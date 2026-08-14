"""
Vision Base Interface

Abstract interface for all vision implementations.

Examples:
- YOLOVision
- RKNNVision
- OpenCVVision

Vision Platform only depends on this interface.
"""

from abc import abstractmethod
from typing import Any

from tools.common.base_tool import BaseTool

from .result import DetectionResult


class VisionBase(BaseTool):
    async def _execute_impl(self, **kwargs) -> DetectionResult:
        """
        BaseTool bridge.

        Forward framework execution
        to vision detection.
        """

        return await self.detect(**kwargs)

    """
    Abstract Vision Tool.

    All vision implementations must inherit this class.
    """

    name = "vision_base"

    description = "Abstract vision capability interface"

    async def initialize(self) -> None:
        """
        Initialize vision resources.

        Examples:
        - load model
        - create RKNN context
        - allocate GPU resource

        Default:
        no operation.
        """

        return None

    @abstractmethod
    async def detect(self, image: Any, **kwargs) -> DetectionResult:
        """
        Run vision inference.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        DetectionResult
            Unified vision result.
        """

        pass

    async def cleanup(self) -> None:
        """
        Release resources.

        Default:
        no operation.
        """

        return None
