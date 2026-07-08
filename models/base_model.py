"""
HumanMM — Abstract Base Model Interface.

Defines the contract that every model backend must fulfil.  By programming to
this interface (the Dependency Inversion Principle), the pipeline is fully
decoupled from any specific framework (YOLO, MediaPipe, ViTPose, etc.).

Design Pattern: Template Method + Strategy
- ``BaseModel`` defines lifecycle (``load`` → ``preprocess`` → ``run`` → ``postprocess``)
- Concrete subclasses override the abstract methods
- The ``PipelineRunner`` works only with ``BaseModel`` references
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all HumanMM model backends.

    All model wrappers (detector, tracker, pose estimator, motion recovery)
    must inherit from this class and implement the abstract methods.

    Args:
        name: Human-readable name for this model (used in logging).
        device: Compute device string (``"cuda"`` or ``"cpu"``).
        config: Optional backend-specific configuration dictionary.

    Attributes:
        name: Model name.
        device: Compute device.
        is_loaded: Whether ``load()`` has been called successfully.
    """

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.device = device
        self.config: Dict[str, Any] = config or {}
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle (Template Method Pattern)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load model weights and prepare the model for inference.

        This is the public entry point.  It calls :meth:`load` and sets
        ``is_loaded = True`` on success.

        Raises:
            RuntimeError: If ``load()`` raises an exception.

        Example:
            >>> model = YOLODetector(device="cuda")
            >>> model.initialize()
            >>> assert model.is_loaded
        """
        log.info("Initializing model: {}", self.name)
        try:
            self.load()
            self.is_loaded = True
            log.info("Model loaded successfully: {}", self.name)
        except Exception as exc:
            log.error("Failed to load model {}: {}", self.name, exc)
            raise RuntimeError(f"Model initialization failed: {self.name}") from exc

    @abstractmethod
    def load(self) -> None:
        """Load model weights from disk or remote hub.

        Concrete implementations must download/cache weights, build the model
        architecture, and move it to ``self.device``.

        Raises:
            FileNotFoundError: If required weight files are missing.
            RuntimeError: If model construction fails.
        """

    @abstractmethod
    def run(self, input_data: Any, **kwargs: Any) -> Any:
        """Run inference on a single input.

        Args:
            input_data: Model-specific input (e.g. a BGR frame, a list of
                bounding boxes, or a dict of tensors).
            **kwargs: Additional runtime arguments.

        Returns:
            Model-specific output.  See concrete class docstrings for the
            exact return type and structure.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """

    def preprocess(self, raw_input: Any) -> Any:
        """Preprocess raw input before passing it to :meth:`run`.

        Override in subclasses if preprocessing is non-trivial.  The default
        implementation is an identity pass-through.

        Args:
            raw_input: Raw input (e.g. a BGR ``numpy.ndarray``).

        Returns:
            Preprocessed input ready for ``run()``.
        """
        return raw_input

    def postprocess(self, raw_output: Any) -> Any:
        """Postprocess raw model output into a structured result.

        Override in subclasses if postprocessing is non-trivial.  The default
        implementation is an identity pass-through.

        Args:
            raw_output: Raw output from the model forward pass.

        Returns:
            Structured, application-ready result.
        """
        return raw_output

    def release(self) -> None:
        """Release GPU memory and other resources held by the model.

        Override in subclasses that hold CUDA tensors or device handles.
        The default implementation is a no-op.
        """
        log.debug("Model released: {}", self.name)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Assert that :meth:`initialize` has been called before inference.

        Raises:
            RuntimeError: If the model has not been initialised.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"Model '{self.name}' has not been initialized. "
                "Call model.initialize() before model.run()."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, device={self.device!r}, loaded={self.is_loaded})"
