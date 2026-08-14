"""
Vision Factory

Create vision backend instances.

Sprint B1.6.2
"""

from __future__ import annotations

from typing import Type

import tools.vision
from tools.vision.config import VisionConfig
from tools.vision.registry import VisionRegistry
from tools.vision.vision_base import VisionBase


class VisionFactory:
    """
    Vision backend factory.

    Example
    -------
    vision = VisionFactory.create("yolo")
    """

    @classmethod
    def create_from_config(
        cls,
        config_path: str,
    ):

        config = VisionConfig(config_path)

        backend = config.backend

        if backend == "yolo":

            yolo_cfg = config.yolo_config

            return cls.create(
                "yolo",
                **yolo_cfg,
            )

        raise ValueError(f"Unsupported vision backend: {backend}")

    @staticmethod
    def create(
        name: str,
        **kwargs,
    ) -> VisionBase:
        """
        Create vision backend instance.

        Parameters
        ----------
        name:
            Backend name registered in VisionRegistry.

        kwargs:
            Backend initialization parameters.

        Returns
        -------
        VisionBase
            Vision backend instance.
        """

        backend_cls: Type[VisionBase] | None = VisionRegistry.get(name)

        if backend_cls is None:
            raise ValueError(f"Vision backend '{name}' is not registered.")

        return backend_cls(**kwargs)
