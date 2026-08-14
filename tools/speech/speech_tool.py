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

WAKE_WORDS = tuple(
    word.strip() for word in os.getenv(
        "AI_AGENT_WAKE_WORDS", "小安小安,小安,你好小安,AI助手"
    ).split(",") if word.strip()
)
# SenseVoice frequently renders “小安” as these homophones on the target USB mic.
WAKE_ALIASES = ("向安", "晓安", "小岸", "小按")
EXIT_COMMANDS = ("关闭语音对话", "再见小安", "停止对话", "退出程序", "退出", "再见", "拜拜")
SLEEP_COMMANDS = ("结束对话", "休眠", "待机")
STOP_COMMANDS = ("停止", "停一下", "听一下", "别说了", "不要说了", "安静", "取消", "停")
NOISE_WORDS = {"嗯", "嗯嗯", "啊", "哦", "哦哦", "额", "那个", "这个", "然后", "就是", "没"}


def normalize(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text)
    return re.sub(r"[。！？!?，,；;：:\s]", "", text).strip()


def strip_wake_word(text: str) -> tuple[bool, str]:
    clean = normalize(text)
    for word in sorted(WAKE_WORDS + WAKE_ALIASES, key=len, reverse=True):
        position = clean.find(normalize(word))
        if position >= 0:
            target = normalize(word)
            return True, clean[:position] + clean[position + len(target):]
    return False, clean


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
        self.continuous_seconds = float(os.getenv("AI_AGENT_CONTINUOUS_DIALOG_SECONDS", "8"))
        # Normal turns are open by default. Wake words are reserved for the
        # playback barge-in channel, where they prevent loudspeaker feedback.
        self.always_listen = os.getenv("AI_AGENT_ALWAYS_LISTEN", "1").lower() not in {"0", "false", "no"}
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
        if self.always_listen:
            self._awake_until = float("inf")
        else:
            self._awake_until = time.monotonic() + self.continuous_seconds

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
        if self._select_asr_backend() == "rknn":
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
            state = "等待唤醒词" if self.wake_required else "连续对话"
            print(f"[唤醒] {state}；唤醒后连续对话 {self.continuous_seconds:g} 秒")
        self.initialized = True

    def listen(self) -> SpeechResult:
        self._lazy_init()
        was_awake = self.is_awake
        ui.state("正在聆听…") if ui.USER_MODE else print("==========请直接说话==========" if self.always_listen or was_awake else "==========等待唤醒：小安小安==========")
        try:
            audio = self.audio.record()
            if audio is None:
                return SpeechResult(text="", success=False, sleeping=not self.is_awake)
            # In user mode, recognition is intentionally silent. Short/noise
            # recordings are common and showing this state on every attempt
            # makes the terminal flicker. Debug mode still exposes the phase.
            if not ui.USER_MODE:
                print("[Agent] 识别中...")
            text = self.asr.transcribe(audio)
        except Exception as exc:
            ui.debug(f"[语音] 录音或识别失败: {exc}")
            return SpeechResult(text="", success=False, error=str(exc))

        ui.debug(f"[ASR] {text}")
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
        if not self.always_listen and not was_awake:
            if not woke:
                print("[唤醒] 未检测到唤醒词，忽略")
                return SpeechResult(text="", audio=audio, success=False, sleeping=True)
            self.open_conversation_window()
            print(f"[唤醒] 已唤醒，进入 {self.continuous_seconds:g} 秒连续对话")
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
                start_timeout=0.8,
                max_record_time=2.8,
                end_silence=0.4,
            )
            while not stop_event.is_set():
                audio = monitor.record(stop_event, quiet=True)
                if audio is None or stop_event.is_set():
                    continue
                text = self.asr.transcribe(audio)
                command = classify_local_command(text, require_wake=True)
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
        # The 15-second follow-up window begins after playback returns, not
        # when a possibly long LLM/TTS response first started.
        if self.always_listen:
            self._awake_until = float("inf")
        elif self.wake_required and self.is_awake:
            self.open_conversation_window()
        control, self._pending_control = self._pending_control, None
        if control == "sleep":
            self.sleep()
        return control
