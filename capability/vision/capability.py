from capability.base import Capability


class VisionCapability(Capability):

    name = "vision"

    description = "视觉感知能力，包括目标检测、图像理解"

    def initialize(self):

        super().initialize()

        print("Vision capability initialized")

        return True

    def execute(self, **kwargs):

        image = kwargs.get("image")

        self.status = self.status.RUNNING

        return {"status": "success", "module": "vision", "image": image}

    def shutdown(self):

        print("Vision capability shutdown")

        super().shutdown()
