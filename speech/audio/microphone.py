import queue
import time

import numpy as np
import sounddevice as sd

from .input import AudioSource
from .vad import VoiceActivityDetector


class MicrophoneInput(AudioSource):

    def __init__(
        self,
        sample_rate=16000,
        chunk_duration=0.5,
        silence_duration=0.7,
        max_record_time=15,
    ):

        self.sample_rate = sample_rate

        self.chunk_size = int(sample_rate * chunk_duration)

        self.silence_duration = silence_duration

        self.max_record_time = max_record_time

        self.vad = VoiceActivityDetector()

    def record(self):

        print("[音频] 等待讲话...")

        audio_queue = queue.Queue()

        def callback(indata, frames, time_info, status):

            audio_queue.put(indata.copy())

        frames = []

        speech_started = False

        silence_time = 0

        start_time = time.time()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=callback,
        ):

            while True:

                chunk = audio_queue.get()

                mono = chunk.reshape(-1)

                has_voice = self.vad.has_speech(
                    mono,
                    self.sample_rate,
                )

                if has_voice:

                    speech_started = True

                    silence_time = 0

                    frames.append(chunk)

                    print("[VAD] speech")

                else:

                    if speech_started:

                        frames.append(chunk)

                        silence_time += len(chunk) / self.sample_rate

                        print("[VAD] silence", round(silence_time, 2))

                # 结束条件

                if speech_started and silence_time >= self.silence_duration:

                    break

                if time.time() - start_time > self.max_record_time:

                    break

        print("[音频 ] 录音结束")

        if len(frames) == 0:

            audio = np.zeros(self.sample_rate, dtype="float32")

        else:

            audio = np.concatenate(frames, axis=0)

            audio = audio.reshape(-1)

        if audio.shape[0] < 1600:

            print("[Audio] Too short")

            audio = np.pad(audio, (0, 1600 - audio.shape[0]))
        return {
            "type": "audio",
            "data": audio,
            "sample_rate": self.sample_rate,
            "channels": 1,
        }
