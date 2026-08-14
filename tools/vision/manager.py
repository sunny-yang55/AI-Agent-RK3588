"""
Vision Pipeline Manager

Build and manage vision pipeline.

Sprint B1.8.1
"""

from __future__ import annotations

from .config import VisionConfig
from .factory import VisionFactory
from .pipeline import VisionPipeline
from .stages.postprocess import PostprocessStage
from .stages.preprocess import PreprocessStage


class VisionPipelineManager:
    """
    Vision pipeline lifecycle manager.
    """

    def __init__(
        self,
        config_path: str,
    ):

        self.config = VisionConfig(config_path)

        self.pipeline: VisionPipeline | None = None

    def create(self) -> VisionPipeline:
        """
        Create pipeline instance.
        """

        backend_name = self.config.backend

        if backend_name == "yolo":

            backend = VisionFactory.create_from_config(str(self.config.path))

        else:

            raise ValueError(f"Unsupported backend: {backend_name}")

        self.pipeline = VisionPipeline(
            backend=backend,
            preprocess=PreprocessStage(),
            postprocess=PostprocessStage(),
        )

        return self.pipeline

    async def initialize(self):
        """
        Initialize pipeline.
        """

        if self.pipeline is None:

            self.create()

        await self.pipeline.initialize()

    async def process(
        self,
        image,
    ):

        if self.pipeline is None:

            await self.initialize()

        image = self.pipeline.preprocess.run(image)

        result = await self.pipeline.process(image)

        result = self.pipeline.postprocess.run(result)

        return result

    async def cleanup(self):

        if self.pipeline:

            await self.pipeline.cleanup()
