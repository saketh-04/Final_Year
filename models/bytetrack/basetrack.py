from __future__ import annotations

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


class TrackState(Enum):
    """
    Represent the possible states of a tracked object in the pipeline.
    
    Attributes:
        New: Initial state when a detection is first seen but not yet confirmed.
        Tracked: The object is actively being tracked.
        Lost: The object was previously tracked but is currently missing.
        Removed: The object is no longer tracked and the ID is retired.
    """
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class BaseTrack(ABC):
    """
    Abstract base class for all tracking objects in the HumanMM pipeline.
    
    This class implements the core identity management and state machine logic
    required for ByteTrack V2. It ensures unique ID assignment across the 
    tracking lifecycle and provides a standardized interface for motion updates.

    Attributes:
        _count (int): Global class-level counter to ensure unique IDs.
        track_id (int): Unique identifier for the specific track instance.
        is_activated (bool): Flag indicating if the track has been initialized.
        state (TrackState): Current lifecycle state of the track.
        history (List[Any]): Historical data/states of the track.
        features (List[np.ndarray]): Appearance features for Re-ID (if used).
        start_frame (int): The frame index where the track was first created.
        frame_id (int): The last frame index where the track was updated.
        time_since_update (int): Number of frames since the last detection update.
    """
    
    _count: int = 0

    track_id: int = 0
    is_activated: bool = False
    state: TrackState = TrackState.New
    
    history: List[np.ndarray] = field(default_factory=list, repr=False)
    features: List[np.ndarray] = field(default_factory=list, repr=False)
    
    start_frame: int = 0
    frame_id: int = 0
    time_since_update: int = 0

    @property
    def end_frame(self) -> int:
        """Returns the most recent frame ID associated with this track."""
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """
        Increments and returns the global track counter.
        
        Returns:
            int: The next available unique track ID.
        """
        BaseTrack._count += 1
        return BaseTrack._count

    @abstractmethod
    def activate(self, *args: Any, **kwargs: Any) -> None:
        """
        Activates a new track or re-activates a lost track.
        
        This must be implemented by subclasses to handle specific motion 
        model initialization (e.g., Kalman Filter state).
        """
        pass

    @abstractmethod
    def predict(self) -> None:
        """
        Advances the track state using a motion model prediction.
        """
        pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Updates the track state with a new detection.
        
        Args:
            *args: Implementation specific detection data.
            **kwargs: Implementation specific metadata.
        """
        pass

    def mark_lost(self) -> None:
        """Transitions the track state to Lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Transitions the track state to Removed."""
        self.state = TrackState.Removed

    @classmethod
    def clear_count(cls) -> None:
        """
        Resets the global track ID counter.
        Useful for processing independent video batches or unit tests.
        """
        cls._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """
        Get the current bounding box in (top left x, top left y, width, height) format.
        Implemented by child classes based on internal state.
        """
        raise NotImplementedError("Subclasses must implement 'tlwh' property access.")

    @property
    def tlbr(self) -> np.ndarray:
        """
        Get the current bounding box in (top left x, top left y, bottom right x, bottom right y).
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """
        Convert bounding box format to (center x, center y, aspect ratio, height).
        Aspect ratio is width / height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes basic track information for logging or downstream HumanMM tasks.
        """
        return {
            "track_id": self.track_id,
            "state": self.state.name,
            "start_frame": self.start_frame,
            "frame_id": self.frame_id,
            "is_activated": self.is_activated
        }

    def __repr__(self) -> str:
        """String representation of the track instance."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"