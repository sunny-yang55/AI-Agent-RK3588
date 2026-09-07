"""Safety-first bridge between workbench vision and the robot arm.

This package deliberately contains no import-time serial access and never
executes a movement by itself.  It turns stable full-camera visual facts into
a validated pick plan; a later hardware adapter may execute an explicitly
confirmed plan.
"""

from .calibration import CameraRobotCalibration, CalibrationError
from .coordinator import PickPlan, RobotArmCoordinator

__all__ = [
    "CalibrationError",
    "CameraRobotCalibration",
    "PickPlan",
    "RobotArmCoordinator",
]
