from core.input_event import (
    AgentInput,
    AudioData,
    InputType,
)


class InputAdapter:

    @staticmethod
    def from_text(text):

        return AgentInput(
            type=InputType.TEXT,
            data=text,
        )

    @staticmethod
    def from_speech(result):

        text = result.text.strip()

        if not text:
            return None

        return AgentInput(type=InputType.TEXT, data=text, metadata={"source": "speech"})
