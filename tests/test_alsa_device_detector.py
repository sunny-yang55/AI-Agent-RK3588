from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from speech.audio import alsa_device_detector as detector


ARECORD_LIST = """**** CAPTURE hardware devices ****
card 0: Device [USB Audio Device], device 0: USB Audio [USB Audio]
card 1: Device_1 [USB Composite Device], device 0: USB Audio [USB Audio]
card 2: rockchiphdmiin [rockchip,hdmiin], device 0: rockchip,hdmiin i2s-hifi-0 [rockchip,hdmiin i2s-hifi-0]
"""


class ALSADeviceDetectorTests(unittest.TestCase):
    def test_list_capture_devices_uses_stable_card_ids(self):
        completed = subprocess.CompletedProcess([], 0, stdout=ARECORD_LIST, stderr="")
        with patch.object(detector.subprocess, "run", return_value=completed):
            devices = detector.list_capture_devices()
        self.assertEqual(
            [item.alsa_name for item in devices],
            [
                "plughw:CARD=Device,DEV=0",
                "plughw:CARD=Device_1,DEV=0",
                "plughw:CARD=rockchiphdmiin,DEV=0",
            ],
        )

    def test_detect_prefers_composite_and_skips_failed_candidate(self):
        candidates = [
            detector.ALSACaptureDevice(0, "Device", "USB Audio Device", 0, "USB Audio"),
            detector.ALSACaptureDevice(1, "Device_1", "USB Composite Device", 0, "USB Audio"),
        ]
        calls = []

        def fake_probe(device, **_kwargs):
            calls.append(device)
            return (False, "busy") if "Device_1" in device else (True, "OK")

        with patch.object(detector, "list_capture_devices", return_value=candidates), patch.object(
            detector, "probe_capture_device", side_effect=fake_probe
        ):
            device, label = detector.detect_capture_device()
        self.assertEqual(calls[0], "plughw:CARD=Device_1,DEV=0")
        self.assertEqual(device, "plughw:CARD=Device,DEV=0")
        self.assertIn("USB Audio Device", label)


if __name__ == "__main__":
    unittest.main()
