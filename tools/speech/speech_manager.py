from .speech_tool import SpeechTool


class SpeechManager:
    def __init__(self):
        self.tool = SpeechTool()

    def listen(self):
        return self.tool.listen()

    def speak(self, text: str, display: bool = True, *, allow_interrupt: bool = True):
        return self.tool.speak(text, display=display, allow_interrupt=allow_interrupt)

    def open_conversation_window(self):
        self.tool.open_conversation_window()
