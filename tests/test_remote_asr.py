import io
import json
import threading
import unittest
import urllib.request
import wave

import numpy as np

from remote_asr.server import create_server, decode_wav


def make_wav() -> bytes:
    samples = np.zeros(1600, dtype="<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


class DummyRecognizer:
    def transcribe(self, audio):
        assert audio["sample_rate"] == 16000
        assert audio["channels"] == 1
        return "开放时间早上9点至下午5点。"


class RemoteASRTest(unittest.TestCase):
    def test_decode_wav(self):
        audio = decode_wav(make_wav())
        self.assertEqual(audio["data"].shape, (1600,))
        self.assertEqual(audio["data"].dtype, np.float32)

    def test_health_and_recognize(self):
        server = create_server("127.0.0.1", 0, DummyRecognizer())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(f"{base_url}/health") as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertTrue(health["ok"])

            request = urllib.request.Request(
                f"{base_url}/recognize",
                data=make_wav(),
                headers={"Content-Type": "audio/wav"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
            self.assertEqual(result["text"], "开放时间早上9点至下午5点。")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
