"""Speech tool with wake-word and continuous-conversation state."""

from __future__ import annotations

import importlib.util
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import voice_ui as ui

WAKE_TOKEN = "小安"
WAKE_FILLERS = ("你好", "您好", "在吗")
EXIT_COMMANDS = ("关闭语音对话", "小安再见", "再见小安", "停止对话", "退出程序", "退出", "再见", "拜拜")
SLEEP_COMMANDS = ("结束对话", "休眠", "待机")
STOP_COMMANDS = ("停止", "停一下", "听一下", "别说了", "不要说了", "安静", "取消", "停")
NOISE_WORDS = {"嗯", "嗯嗯", "啊", "哦", "哦哦", "额", "那个", "这个", "然后", "就是", "没"}


def normalize(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text)
    return re.sub(r"[。！？!?，,；;：:\s]", "", text).strip()


def strip_wake_word(text: str) -> tuple[bool, str]:
    clean = normalize(text)
    if WAKE_TOKEN not in clean:
        return False, clean

    # Product rule: only the literal characters “小安” may wake the assistant.
    # Remove repeated mentions (小安小安) and a greeting immediately surrounding
    # the wake word, but never accept phonetic/homophone aliases such as 小韩.
    command = clean.replace(WAKE_TOKEN, "")
    if command in {"你好", "您好", "在吗", "请问", "请问在吗"}:
        return True, ""
    for filler in WAKE_FILLERS:
        if command == filler:
            command = ""
            break
        if command.startswith(filler):
            command = command[len(filler):]
            break
    return True, command


def classify_local_command(text: str, *, require_wake: bool = False) -> str | None:
    """Return a local control command without sending it to the LLM."""
    clean = normalize(text)
    woke, command = strip_wake_word(text)
    target = command if woke else clean
    if require_wake and not woke:
        return None
    if any(item in target for item in EXIT_COMMANDS):
        return "exit"
    if any(item in target for item in STOP_COMMANDS):
        return "stop"
    if any(item in target for item in SLEEP_COMMANDS):
        return "sleep"
    return None


def classify_playback_command(text: str) -> str | None:
    """Make stopping maximally responsive while playback is active."""
    clean = normalize(text)
    if "停" in clean:
        return "stop"
    # Exit/sleep remain wake-prefixed during loudspeaker playback to reduce
    # accidental commands caused by the assistant's own voice.
    return classify_local_command(text, require_wake=True)


@dataclass
class SpeechResult:
    text: str
    audio: dict | None = None
    success: bool = True
    exit: bool = False
    error: str | None = None
    woke: bool = False
    sleeping: bool = False


class SpeechTool:
    def __init__(self) -> None:
        self.audio = self.asr = self.tts = None
        self.initialized = False
        self._tts_attempted = False
        # Kept only for backwards-compatible configuration parsing. The active
        # conversation no longer expires after a fixed number of seconds.
        self.continuous_seconds = float(os.getenv("AI_AGENT_CONTINUOUS_DIALOG_SECONDS", "0"))
        self.always_listen = os.getenv("AI_AGENT_ALWAYS_LISTEN", "0").lower() not in {"0", "false", "no"}
        self.wake_required = (not self.always_listen) and os.getenv(
            "AI_AGENT_WAKE_WORD_ENABLED", "1"
        ).lower() not in {"0", "false", "no"}
        self._awake_until = 0.0 if self.wake_required else float("inf")
        self._pending_control: str | None = None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def is_awake(self) -> bool:
        return time.monotonic() < self._awake_until

    def open_conversation_window(self) -> None:
        # Once explicitly woken, remain active until a local sleep/exit command
        # or process shutdown. Per-utterance VAD timeouts still apply.
        self._awake_until = float("inf")

    def sleep(self) -> None:
        self._awake_until = 0.0

    def _select_asr_backend(self) -> str:
        requested = os.getenv("AI_AGENT_ASR_BACKEND", "auto").strip().lower()
        if requested != "auto":
            return requested
        model = Path(os.getenv("SENSEVOICE_RKNN_MODEL", str(
            self._project_root() / "models/speech/sensevoice/sensevoice_time100_fp.rknn"
        )))
        return "rknn" if model.is_file() and importlib.util.find_spec("rknnlite") else "funasr"

    def _lazy_init(self) -> None:
        if self.initialized:
            return
        ui.debug("[系统] 加载语音功能...")
        backend = self._select_asr_backend()
        if backend == "hybrid":
            from speech.asr.chunked_rknn_sensevoice_asr import ChunkedRKNNSenseVoiceASR
            from speech.asr.zipformer_rknn_asr import ZipformerRKNNASR
            from speech.audio.hybrid_streaming_microphone import HybridStreamingMicrophoneInput
            model_dir = self._project_root() / "models/speech/zipformer"
            ui.state("正在加载实时识别模型…", "◌")
            partial_asr = ZipformerRKNNASR(model_dir)
            ui.state("Zipformer 实时识别已就绪", "✓")
            self.asr = ChunkedRKNNSenseVoiceASR()
            ui.state("SenseVoice 精准校正已就绪", "✓")
            self.audio = HybridStreamingMicrophoneInput(partial_asr)
            ui.debug("[系统] 语音后端: Zipformer动态识别 + SenseVoice最终校正")
        elif backend == "rknn":
            from speech.asr.chunked_rknn_sensevoice_asr import ChunkedRKNNSenseVoiceASR
            from speech.audio.alsa_microphone import ALSAMicrophoneInput
            self.audio, self.asr = ALSAMicrophoneInput(), ChunkedRKNNSenseVoiceASR()
            ui.debug("[系统] 语音后端: RK3588 ALSA VAD + SenseVoice RKNN")
        else:
            from speech.asr.sensevoice_asr import SenseVoiceASR
            from speech.audio.microphone import MicrophoneInput
            self.audio, self.asr = MicrophoneInput(), SenseVoiceASR()
            print("[系统] 语音后端: sounddevice + FunASR")
        if self.always_listen:
            ui.debug("[对话] 持续监听；普通对话无需唤醒词，播报中打断需说“小安 + 停止词”")
        else:
            state = "等待唤醒词“小安”" if self.wake_required else "持续对话"
            ui.state(f"{state}；唤醒后无需重复唤醒")
        self.initialized = True

    def listen(self) -> SpeechResult:
        self._lazy_init()
        was_awake = self.is_awake
        if ui.USER_MODE:
            ui.state("正在聆听…" if self.always_listen or was_awake else "等待唤醒…")
        else:
            print("==========请直接说话==========" if self.always_listen or was_awake else "==========等待唤醒：小安你好==========")
        try:
            audio = self.audio.record()
            if audio is None:
                return SpeechResult(text="", success=False, sleeping=not self.is_awake)
            if not ui.USER_MODE:
                print("[Agent] 精准识别中...")
            else:
                ui.recognition_finalize()
            text = self.asr.transcribe(audio)
        except Exception as exc:
            ui.debug(f"[语音] 录音或识别失败: {exc}")
            return SpeechResult(text="", success=False, error=str(exc))

        ui.debug(f"[ASR] {text}")
        if ui.USER_MODE and normalize(text):
            ui.recognition_result(normalize(text))
        clean = normalize(text)
        woke, command = strip_wake_word(text)
        # Exit and sleep are global safety controls. They must work even after
        # the normal follow-up window has expired.
        local = classify_local_command(text)
        if local == "exit":
            return SpeechResult(text=clean, audio=audio, success=True, exit=True, woke=woke)
        if local == "sleep":
            self.sleep()
            return SpeechResult(text="", audio=audio, success=False, sleeping=True, woke=woke)
        if local == "stop":
            # A stop phrase can arrive immediately after playback has ended.
            # Never forward it to the LLM as an ordinary user request.
            if ui.USER_MODE:
                ui.state("当前播报已停止", "■")
            return SpeechResult(text="", audio=audio, success=False, woke=woke)
        if not self.always_listen and not was_awake:
            if not woke:
                ui.debug("[唤醒] 未检测到唤醒词，忽略")
                return SpeechResult(text="", audio=audio, success=False, sleeping=True)
            self.open_conversation_window()
            ui.state("已唤醒，进入持续对话", "✓")
            if not command:
                return SpeechResult(text="", audio=audio, success=False, woke=True)
            clean = command
        elif woke:
            if not self.always_listen:
                self.open_conversation_window()
            # A user may still say the assistant name out of habit. Strip it
            # from the LLM request even though it is not required in this mode.
            clean = command

        if not clean or clean in NOISE_WORDS or len(clean) <= 1:
            return SpeechResult(text="", audio=audio, success=False, woke=woke)
        # An utterance that began inside the window remains valid even if ASR
        # completes just after the deadline. Playback will renew it again.
        if not self.always_listen:
            self.open_conversation_window()
        return SpeechResult(text=clean, audio=audio, success=True, exit=False, woke=woke)

    def _lazy_init_tts(self) -> bool:
        if self.tts is not None:
            return True
        if self._tts_attempted:
            return False
        self._tts_attempted = True
        try:
            from speech.tts.tts_engine import TTSEngine
            self.tts = TTSEngine()
            return True
        except Exception as exc:
            print(f"[TTS] 当前环境未启用语音播放，仅显示文字: {exc}")
            return False

    def _listen_for_playback_control(self, stop_event: threading.Event) -> None:
        """Listen only for wake-prefixed stop/exit commands during playback."""
        try:
            # Do not run ASR while Edge is still synthesizing the first chunk.
            # Barge-in becomes useful only after real loudspeaker playback.
            if self.tts is not None and not self.tts.wait_for_playback_start(stop_event):
                return
            ui.debug("[打断] 播放已开始，启用语音打断监听")
            from speech.audio.alsa_microphone import ALSAMicrophoneInput
            monitor = ALSAMicrophoneInput(
                device=getattr(self.audio, "device", None),
                start_timeout=0.5,
                max_record_time=3.5,
                end_silence=0.35,
            )
            while not stop_event.is_set():
                audio = monitor.record(stop_event, quiet=True)
                if audio is None or stop_event.is_set():
                    continue
                text = self.asr.transcribe(audio)
                command = classify_playback_command(text)
                if command in {"stop", "exit", "sleep"}:
                    ui.debug(f"[打断] 识别到: {text}")
                    self._pending_control = command
                    if self.tts is not None:
                        self.tts.stop()
                    stop_event.set()
                    return
        except Exception as exc:
            if not stop_event.is_set():
                ui.debug(f"[打断] 监听不可用，本次继续播报: {exc}")

    def speak(self, text: str, display: bool = True,
              *, allow_interrupt: bool = True) -> str | None:
        if not text:
            return None
        if display:
            if ui.USER_MODE:
                ui.assistant_start()
                print(text)
            else:
                print("\nAI-Agent:")
                print(text)
        if self._lazy_init_tts():
            from speech.tts.text_cleaner import clean_tts_text
            spoken = clean_tts_text(text)
            if spoken:
                if ui.USER_MODE:
                    message = (
                        "正在播报…（可说“小安停止”）"
                        if allow_interrupt else "正在播报…"
                    )
                    ui.state(message, "▶")
                else:
                    print("[TTS] 语音播报...")
                stop_event = threading.Event()
                monitor = None
                self.tts.prepare_speak()
                if allow_interrupt and os.getenv(
                    "AI_AGENT_BARGE_IN_ENABLED", "1"
                ).lower() not in {"0", "false", "no"}:
                    monitor = threading.Thread(
                        target=self._listen_for_playback_control,
                        args=(stop_event,), daemon=True, name="barge-in-listener"
                    )
                    monitor.start()
                try:
                    self.tts.speak(spoken)
                finally:
                    stop_event.set()
                    if monitor is not None:
                        monitor.join(timeout=2.0)
                        time.sleep(float(os.getenv("AI_AGENT_AUDIO_RECOVERY_DELAY", "0.25")))
        # An explicitly activated conversation never expires by wall clock.
        if self.always_listen:
            self._awake_until = float("inf")
        elif self.wake_required and self.is_awake:
            self.open_conversation_window()
        control, self._pending_control = self._pending_control, None
        if control == "sleep":
            self.sleep()
        return control
