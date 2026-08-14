"""
Vision Configuration Loader

Sprint B1.6.3
"""

from pathlib import Path

import yaml


class VisionConfig:
    """
    Vision configuration loader.
    """

    def __init__(self, config_path: str):

        self.path = Path(config_path)

        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        with open(
            self.path,
            "r",
            encoding="utf-8",
        ) as f:

            self.data = yaml.safe_load(f)

    @property
    def backend(self):

        return self.data.get("vision", {}).get("backend")

    @property
    def yolo_config(self):

        return self.data.get("vision", {}).get("yolo", {})
