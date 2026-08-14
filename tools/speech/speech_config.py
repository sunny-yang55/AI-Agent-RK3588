from dataclasses import dataclass


@dataclass
class SpeechConfig:
    wake_word: str = "AI-Agent"
    language: str = "zh-CN"
    sample_rate: int = 16000
