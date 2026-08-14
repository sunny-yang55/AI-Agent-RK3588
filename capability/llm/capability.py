from capability.base import Capability


class LLMCapability(Capability):

    name = "llm"

    description = "大语言模型推理能力"

    def execute(self, **kwargs):

        prompt = kwargs.get("prompt")

        return {"status": "success", "module": "llm", "prompt": prompt}
