import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "build-v1.3.4"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))

if "webrtcvad" not in sys.modules:
    fake_webrtcvad = types.ModuleType("webrtcvad")
    fake_webrtcvad.Vad = lambda mode=2: None
    sys.modules["webrtcvad"] = fake_webrtcvad

from speech.audio.alsa_microphone import ALSAMicrophoneInput


class _PipeProcess:
    def __init__(self, fd):
        self.stdout = os.fdopen(fd, "rb", buffering=0)

    def poll(self):
        return None


class ALSAVADTimeoutTests(unittest.TestCase):
    def test_no_pcm_data_obeys_wall_clock_deadline(self):
        read_fd, write_fd = os.pipe()
        process = _PipeProcess(read_fd)
        started = time.monotonic()
        try:
            result = ALSAMicrophoneInput._read_frame(
                process, 960, time.monotonic() + 0.05
            )
        finally:
            os.close(write_fd)
            process.stdout.close()
        self.assertIsNone(result)
        self.assertLess(time.monotonic() - started, 0.25)

    def test_fragmented_pcm_is_reassembled(self):
        read_fd, write_fd = os.pipe()
        process = _PipeProcess(read_fd)

        def writer():
            os.write(write_fd, b"a" * 300)
            time.sleep(0.01)
            os.write(write_fd, b"b" * 660)
            os.close(write_fd)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            result = ALSAMicrophoneInput._read_frame(
                process, 960, time.monotonic() + 0.5
            )
        finally:
            thread.join()
            process.stdout.close()
        self.assertEqual(result, b"a" * 300 + b"b" * 660)


if __name__ == "__main__":
    unittest.main()
