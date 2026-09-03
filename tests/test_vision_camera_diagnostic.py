"""Tests for pure camera diagnostic helpers."""

import unittest

from scripts.test_vision_camera import DiagnosticStats, duration_reached


class CameraDiagnosticTests(unittest.TestCase):
    def test_average_fps_uses_elapsed_monotonic_time(self):
        stats = DiagnosticStats(started_at=10.0, frames=50)
        self.assertEqual(stats.average_fps(12.0), 25.0)

    def test_average_fps_is_zero_when_no_time_elapsed(self):
        stats = DiagnosticStats(started_at=10.0, frames=50)
        self.assertEqual(stats.average_fps(10.0), 0.0)

    def test_zero_duration_is_unlimited(self):
        self.assertFalse(duration_reached(0.0, 10.0, 1000.0))

    def test_positive_duration_stops_at_deadline(self):
        self.assertFalse(duration_reached(30.0, 10.0, 39.9))
        self.assertTrue(duration_reached(30.0, 10.0, 40.0))


if __name__ == "__main__":
    unittest.main()
