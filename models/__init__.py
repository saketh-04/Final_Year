"""
HumanMM — Models Package.

Exports the core model interfaces and factory for convenient import.
"""

from models.base_model import BaseModel
from models.model_factory import ModelFactory
from models.yolo_detector import YOLODetector, Detection
from models.mediapipe_pose import MediaPipePoseBackend, PersonPose, Keypoint
from models.vitpose_pose import ViTPosePoseBackend
from models.bytetrack_tracker import ByteTrackTracker, Track, TrackState
from models.deepsort_tracker import DeepSORTTracker
from models.gvhmr_wrapper import MotionRecoveryModel, SMPLOutput

__all__ = [
    "BaseModel",
    "ModelFactory",
    "YOLODetector",
    "Detection",
    "MediaPipePoseBackend",
    "PersonPose",
    "Keypoint",
    "ViTPosePoseBackend",
    "ByteTrackTracker",
    "Track",
    "TrackState",
    "DeepSORTTracker",
    "MotionRecoveryModel",
    "SMPLOutput",
]
