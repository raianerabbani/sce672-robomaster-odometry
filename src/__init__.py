"""RoboMaster EP Odometry & Kalman Estimation Package.
SCE-672: Advanced Robotics, KFUPM
"""

from src.tag_detector import TagDetector, TagDetection
from src.kalman_filter import OdometryVisionKalmanFilter, FilterState
from src.motion_controller import StraightLineMotionController
from src.robot_interface import RoboMasterRobot, MockRobot, BaseRobot
from src.trial_manager import TrialManager, TrialRecord, ExperimentStatistics
from src.visualizer import VisualizerHUD

__all__ = [
    "TagDetector",
    "TagDetection",
    "OdometryVisionKalmanFilter",
    "FilterState",
    "StraightLineMotionController",
    "RoboMasterRobot",
    "MockRobot",
    "BaseRobot",
    "TrialManager",
    "TrialRecord",
    "ExperimentStatistics",
    "VisualizerHUD",
]
