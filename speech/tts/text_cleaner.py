import re


def clean_tts_text(text):
    """
    TTS语音文本清洗

    目标:
    LLM文本
        ↓
    适合语音播报的纯文本
    """

    if not text:
        return ""

    # =====================
    # 1. 去emoji
    # =====================

    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    # 去emoji变体符号
    text = re.sub(r"[\ufe0e\ufe0f]", "", text)

    # =====================
    # 2. 去Markdown
    # =====================

    # 粗体
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # 标题
    text = re.sub(r"#+", "", text)

    # 代码符号
    text = text.replace("`", "")

    # =====================
    # 3. 去列表符号
    # =====================

    text = re.sub(r"^[\s]*[-•·]\s*", "", text, flags=re.MULTILINE)

    # =====================
    # 4. 去特殊装饰符号
    # =====================

    text = re.sub(r"[【】\[\]<>]", "", text)

    # 常见AI符号

    text = re.sub(r"[✅☑️✔️🔹🔸]", "", text)

    # =====================
    # 5. 去括号说明
    # =====================

    text = re.sub(r"[\(\（].*?[\)\）]", "", text)

    # =====================
    # 6. 合并空格
    # =====================

    text = re.sub(r"\s+", " ", text)

    return text.strip()
