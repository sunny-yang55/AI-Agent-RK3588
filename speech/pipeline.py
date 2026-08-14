"""
Speech Pipeline

Audio
 |
ASR
 |
Agent
 |
Response Adapter
 |
TTS
"""

import logging

logger = logging.getLogger(__name__)


class SpeechPipeline:

    def __init__(self, agent, recognizer, synthesizer, audio_input=None):

        self.agent = agent
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.audio_input = audio_input

    def _extract_response(self, result):
        """
        Agent结果标准化

        保证TTS得到字符串
        """

        if isinstance(result, str):

            return result

        if isinstance(result, dict):

            if "response" in result:

                return result["response"]

            if "content" in result:

                return result["content"]

        return str(result)

    def process_text(self, text):

        print()
        print("[Agent INPUT]")
        print(text)
        print()

        logger.info("User text: %s", text)

        result = self.agent.run_sync(text)

        response = self._extract_response(result)

        logger.info("Agent response: %s", response)

        self.synthesizer.speak(response)

        return response

    def run(self):

        if self.audio_input is None:

            raise RuntimeError("Audio input is not configured")

        audio = self.audio_input.record()

        return self.process_audio(audio)

    def process_audio(self, audio):

        raise RuntimeError("Deprecated: ASR should be handled by SpeechTool")
