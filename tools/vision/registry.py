"""
Vision Backend Registry

Sprint B1.6.1
"""


class VisionRegistry:

    _registry = {}

    @classmethod
    def register(cls, name, backend):

        cls._registry[name] = backend

    @classmethod
    def get(cls, name):

        return cls._registry.get(name)
