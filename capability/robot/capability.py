from capability.base import Capability


class RobotCapability(Capability):

    name = "robot"

    description = "机器人运动控制能力"

    def execute(self, **kwargs):

        command = kwargs.get("command")

        return {"status": "success", "module": "robot", "command": command}
