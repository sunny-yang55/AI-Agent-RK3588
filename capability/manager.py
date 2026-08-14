class CapabilityManager:

    def __init__(self, registry):

        self.registry = registry

    def execute(self, name: str, **kwargs):

        capability = self.registry.get(name)

        if capability is None:
            raise ValueError(f"Capability not found: {name}")

        return capability.execute(**kwargs)

    def list(self):

        return self.registry.list()

    def get(self, name):

        return self.registry.get(name)
