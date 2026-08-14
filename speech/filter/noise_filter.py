import re


class NoiseFilter:

    def __init__(self):

        self.noise_words = [
            "嗯",
            "嗯嗯",
            "啊",
            "啊啊",
            "那个",
            "这个",
            "好的",
            "哦",
            "哦哦",
            "哎",
        ]

    def clean(self, text):

        if not text:
            return ""

        text = text.strip()

        # 去SenseVoice标签

        text = re.sub(r"<\|.*?\|>", "", text)

        return text.strip()

    def is_noise(self, text):

        if not text:

            return True

        if len(text) <= 1:

            return True

        if text in self.noise_words:

            return True

        return False
