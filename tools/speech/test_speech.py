from tools.speech.speech_manager import SpeechManager

manager = SpeechManager()

result = manager.listen()

manager.speak(result.text)
