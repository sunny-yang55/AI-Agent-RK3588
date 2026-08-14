from typing import Dict

from .base import Capability


class CapabilityRegistry:
    """
    Capability注册中心
    """

    def __init__(self):

        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability):

        name = capability.name

        self._capabilities[name] = capability

    def get(self, name: str):

        return self._capabilities.get(name)

    def list(self):

        return list(self._capabilities.keys())

    def describe(self):

        return [c.info() for c in self._capabilities.values()]
