from capability.base import Capability


class SpeechCapability(Capability):

    name = "speech"

    description = "语音识别和语音合成能力"

    def execute(self, **kwargs):

        return {"status": "success", "module": "speech"}
