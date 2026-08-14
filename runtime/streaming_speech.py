"""Sentence-buffered background TTS for streamed LLM responses."""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from collections.abc import Callable


_SENTENCE_END = re.compile(r"^(.+?[。！？!?；;\n]+)(.*)$", re.DOTALL)
class StreamingSpeechPlayer:
    """Turn token chunks into ordered sentence-level TTS jobs.

    LLM generation stays on the caller thread while a single worker performs
    synthesis and playback.  The worker is deliberately serial so audio never
    overlaps and Piper is only accessed by one thread.
    """

    def __init__(
        self,
        speak: Callable[[str], object],
        *,
        min_chars: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        self._speak = speak
        # Retained as accepted arguments for compatibility with older callers.
        # Streaming speech now waits for sentence punctuation and never slices
        # Chinese text at an arbitrary character count.
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._buffer = ""
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="streaming-tts",
            daemon=True,
        )
        self._started = False
        self._submitted = 0
        self._closed = False
        self.first_audio_s: float | None = None
        self._response_started = time.perf_counter()

    @property
    def submitted(self) -> int:
        return self._submitted

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self.start()
        self._buffer += chunk
        self._drain_complete_sentences()

    def finish(self, *, wait: bool = True) -> None:
        """Flush the final text and close input.

        ``wait=False`` is the non-blocking path used when LLM generation has
        completed.  Playback can then drain in the background while generation
        timing is reported.  Call ``wait()`` before opening the microphone.
        """
        if self._closed:
            if wait:
                self.wait()
            return
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._submit(tail)
        self._closed = True
        if self._started:
            self._queue.put_nowait(None)
        if wait:
            self.wait()

    def wait(self) -> None:
        if self._started and self._thread.is_alive():
            self._thread.join()

    def _drain_complete_sentences(self) -> None:
        while self._buffer:
            hard = _SENTENCE_END.match(self._buffer)
            if hard:
                sentence, self._buffer = hard.groups()
                if sentence.strip():
                    self._submit(sentence)
                continue

            break

    def _submit(self, text: str) -> None:
        text = text.strip()
        if text:
            self._submitted += 1
            self._queue.put_nowait(text)

    def _worker(self) -> None:
        while True:
            sentence = self._queue.get()
            try:
                if sentence is None:
                    return
                if self.first_audio_s is None:
                    self.first_audio_s = time.perf_counter() - self._response_started
                    print(f"\n[TTS] 首句开始播报: {self.first_audio_s:.2f}s")
                self._speak(sentence)
            except Exception as exc:
                print(f"[TTS] 流式播报失败，跳过当前句: {exc}")
            finally:
                self._queue.task_done()
