"""
SenseVoice ASR Backend
Speech v0.7.0
Backend: SenseVoice-small
用途: Windows开发、RK3588迁移
"""

import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

logging.getLogger("modelscope").setLevel(logging.ERROR)
logging.getLogger("funasr").setLevel(logging.ERROR)

from funasr import AutoModel

logger = logging.getLogger(__name__)

_model_instance = None


class HideOutput:

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        self.devnull = open(os.devnull, "w")

        sys.stdout = self.devnull
        sys.stderr = self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

        self.devnull.close()


class SenseVoiceASR:
    _instance_count = 0

    def __init__(self):
        SenseVoiceASR._instance_count += 1

        # print(f"[DEBUG] SenseVoiceASR instance count = {SenseVoiceASR._instance_count}")
        global _model_instance

        logger.info("Loading SenseVoice model...")

        # 如果模型已经存在，直接复用
        if _model_instance is not None:

            print("[SenseVoice] Reuse existing model")

            self.model = _model_instance

            return

        try:

            # print("[SenseVoice] 下载模型中...")
            with HideOutput():

                self.model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    trust_remote_code=True,
                )

            # 保存模型实例
            _model_instance = self.model

            # print("[SenseVoice] Model loaded successfully")

        except Exception as e:

            print("====================")
            print("SenseVoice Load Error")
            print(type(e))
            print(e)

            import traceback

            traceback.print_exc()

            print("====================")

            raise

        logger.info("SenseVoice loaded")

    def transcribe(self, audio):
        """
        audio:
        {
            type,
            data,
            sample_rate,
            channels
        }
        """

        waveform = audio["data"]

        with open(os.devnull, "w") as f:
            old_stdout = sys.stdout
            sys.stdout = f

            result = self.model.generate(
                input=waveform,
                cache={},
                language="zh",
                use_itn=True,
            )

            sys.stdout = old_stdout

        if isinstance(result, list):

            text = result[0]["text"]

            tokens = [
                "<|zh|>",
                "<|en|>",
                "<|BGM|>",
                "<|neutral|>",
                "<|NEUTRAL|>",
                "<|happy|>",
                "<|HAPPY|>",
                "<|sad|>",
                "<|SAD|>",
                "<|angry|>",
                "<|ANGRY|>",
                "<|fearful|>",
                "<|FEARFUL|>",
                "<|disgusted|>",
                "<|DISGUSTED|>",
                "<|Speech|>",
                "<|withitn|>",
            ]

            for t in tokens:
                text = text.replace(t, "")

            return text.strip()

        return str(result)
