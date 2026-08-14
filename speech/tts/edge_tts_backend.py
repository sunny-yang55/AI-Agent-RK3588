"""Interruptible Edge-TTS playback for RK3588."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import tempfile
import threading
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import edge_tts
import pygame

from .synthesizer import SpeechSynthesizer
from .piper_tts_backend import piper_available
import voice_ui as ui

logger = logging.getLogger(__name__)


class EdgeTTS(SpeechSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        pygame.mixer.init()
        self._stopped = threading.Event()
        self._channel = pygame.mixer.Channel(0)
        self._piper = None
        self.partial_output = False
        self._playback_started = None
        logger.info("EdgeTTS initialized")

    def stop(self) -> None:
        self._stopped.set()
        try:
            self._channel.stop()
            pygame.mixer.music.stop()
        except pygame.error:
            pass

    def set_playback_started_callback(self, callback) -> None:
        self._playback_started = callback

    def speak(self, text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return False
        self._stopped.clear()
        self.partial_output = False
        try:
            self._synthesize_and_play(text)
            return not self._stopped.is_set()
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)
            return False

    @staticmethod
    def _split_text(text: str) -> list[str]:
        max_chars = max(40, int(os.getenv("AI_AGENT_EDGE_CHUNK_CHARS", "110")))
        sentences = [part.strip() for part in re.findall(r".*?[。！？!?；;]|.+$", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > max_chars:
                chunks.append(current)
                current = ""
            while len(sentence) > max_chars:
                room = max_chars - len(current)
                current += sentence[:room]
                chunks.append(current)
                current, sentence = "", sentence[room:]
            current += sentence
        if current:
            chunks.append(current)
        return chunks

    async def _save_chunk(self, text: str, filename: str) -> None:
        voice = os.getenv("AI_AGENT_EDGE_VOICE", "zh-CN-YunxiNeural")
        timeout = max(1.0, float(os.getenv("AI_AGENT_EDGE_TIMEOUT", "8")))
        try:
            await asyncio.wait_for(
                edge_tts.Communicate(text=text, voice=voice).save(filename),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Edge-TTS单段超过 {timeout:g} 秒") from exc

    def _synthesize_one(self, index: int, text: str) -> tuple[int, str]:
        retries = max(0, int(os.getenv("AI_AGENT_EDGE_RETRIES", "1")))
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            handle = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            filename = handle.name
            handle.close()
            try:
                asyncio.run(self._save_chunk(text, filename))
                if os.path.getsize(filename) < 512:
                    raise RuntimeError("Edge-TTS returned an empty audio chunk")
                return index, filename
            except Exception as exc:
                last_error = exc
                ui.debug(
                    f"[TTS] Edge第 {index + 1} 段尝试 {attempt + 1}/{retries + 1} 失败: {exc}"
                )
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
                if attempt < retries:
                    time.sleep(0.2)
        raise RuntimeError(f"chunk {index + 1} synthesis failed: {last_error}")

    def _piper_chunk(self, index: int, text: str) -> str:
        if not piper_available():
            raise RuntimeError("Piper fallback is not available")
        if self._piper is None:
            from .piper_tts_backend import PiperTTS
            self._piper = PiperTTS()
        ui.debug(f"[TTS] Edge第 {index + 1} 段失败，仅由Piper补播这一段")
        filename = self._piper.synthesize_to_file(text)
        if filename is None:
            raise RuntimeError(f"Piper fallback failed for chunk {index + 1}")
        return filename

    def _synthesize_and_play(self, text: str) -> None:
        """Play chunk 0 as soon as it is ready while later chunks synthesize."""
        chunks = self._split_text(text)
        workers = min(len(chunks), max(1, int(os.getenv("AI_AGENT_EDGE_PARALLEL", "3"))))
        started = time.monotonic()
        files: list[str] = []
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="edge-tts")
        futures = [pool.submit(self._synthesize_one, i, chunk) for i, chunk in enumerate(chunks)]
        try:
            for expected, future in enumerate(futures):
                    if self._stopped.is_set():
                        break
                    try:
                        index, filename = future.result()
                        source = "Edge"
                    except Exception as exc:
                        logger.warning("Edge chunk %d failed: %s", expected + 1, exc)
                        index = expected
                        filename = self._piper_chunk(index, chunks[index])
                        source = "Piper兜底"
                    if self._stopped.is_set():
                        try:
                            os.remove(filename)
                        except FileNotFoundError:
                            pass
                        break
                    if index != expected:
                        raise RuntimeError(f"TTS chunk order mismatch: {index} != {expected}")
                    files.append(filename)
                    sound = pygame.mixer.Sound(filename)
                    duration = float(sound.get_length())
                    if duration < 0.15:
                        raise RuntimeError(f"chunk {index + 1} audio duration is invalid: {duration:.2f}s")
                    if index == 0:
                        ui.debug(f"[TTS] 首段就绪: {time.monotonic() - started:.2f}s，立即开始播放")
                    ui.debug(f"[TTS] 播放 {index + 1}/{len(chunks)}（{source}），音频 {duration:.2f}s")
                    self._channel.play(sound)
                    if not self.partial_output and self._playback_started is not None:
                        self._playback_started()
                    self.partial_output = True
                    # Do not trust a transient false from get_busy() on the
                    # RK3588 MP3 decoder. Keep this Sound alive for its decoded
                    # duration unless an explicit barge-in stops it.
                    deadline = time.monotonic() + duration + 0.05
                    while time.monotonic() < deadline and not self._stopped.is_set():
                        time.sleep(0.02)
                    self._channel.stop()
        finally:
            for future in futures:
                if future.cancel():
                    continue
                if future.done():
                    try:
                        _, unused_file = future.result()
                    except Exception:
                        continue
                    if unused_file not in files:
                        try:
                            os.remove(unused_file)
                        except FileNotFoundError:
                            pass
                else:
                    # A running request is bounded by AI_AGENT_EDGE_TIMEOUT.
                    # Remove its result when it eventually leaves the worker.
                    def cleanup_late(done_future):
                        try:
                            _, unused_file = done_future.result()
                            os.remove(unused_file)
                        except (Exception, FileNotFoundError):
                            pass
                    future.add_done_callback(cleanup_late)
            # Every running Edge request has its own hard timeout. Do not make
            # stop()/Ctrl+C wait for unrelated queued chunks.
            pool.shutdown(wait=False, cancel_futures=True)
            for filename in files:
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
        if not self._stopped.is_set():
            ui.debug(f"[TTS] 全部 {len(chunks)} 段播放完成")
