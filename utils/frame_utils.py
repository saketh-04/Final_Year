"""
HumanMM — Frame Utility Module.

Provides frame-level I/O, conversion, and preprocessing helpers used
throughout the pipeline.  All frame operations are centralized here so that
downstream modules never call ``cv2`` or ``PIL`` directly for basic tasks.

Example:
    >>> from utils.frame_utils import load_frame, resize_frame, bgr_to_rgb
    >>> frame = load_frame("data/frames/0001.jpg")
    >>> frame_rgb = bgr_to_rgb(frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

log = get_logger(__name__)

# Type alias for a single video frame (H × W × C, uint8, BGR)
Frame = np.ndarray


def load_frame(path: Union[str, Path]) -> Frame:
    """Load a single image frame from disk as a BGR NumPy array.

    Args:
        path: Path to the image file.

    Returns:
        Image as a ``numpy.ndarray`` of shape ``(H, W, 3)`` with dtype
        ``uint8`` in BGR colour order.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If OpenCV fails to decode the image.

    Example:
        >>> frame = load_frame("outputs/frames/0001.jpg")
        >>> print(frame.shape)  # (720, 1280, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frame file not found: {path}")

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"OpenCV failed to decode image: {path}")

    return frame


def save_frame(frame: Frame, path: Union[str, Path], quality: int = 95) -> None:
    """Save a BGR NumPy frame to disk as JPEG or PNG.

    Args:
        frame: Image array of shape ``(H, W, 3)`` or ``(H, W)``.
        path: Destination file path.  Extension determines format.
        quality: JPEG quality [1–100].  Ignored for PNG.

    Raises:
        OSError: If the parent directory cannot be created.
        ValueError: If OpenCV fails to encode the frame.

    Example:
        >>> save_frame(frame, "outputs/frames/0001.jpg", quality=90)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    else:
        params = []

    success = cv2.imwrite(str(path), frame, params)
    if not success:
        raise ValueError(f"OpenCV failed to write frame to: {path}")


def resize_frame(
    frame: Frame,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> Frame:
    """Resize a frame to the target width and height.

    Args:
        frame: Input image array.
        width: Target width in pixels.
        height: Target height in pixels.
        interpolation: OpenCV interpolation flag.  Defaults to bilinear.

    Returns:
        Resized frame as a NumPy array.

    Example:
        >>> resized = resize_frame(frame, 640, 360)
    """
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def resize_keeping_aspect(
    frame: Frame,
    max_width: int,
    max_height: int,
) -> Frame:
    """Resize a frame so it fits within a bounding box, preserving aspect ratio.

    Args:
        frame: Input image array.
        max_width: Maximum allowed width.
        max_height: Maximum allowed height.

    Returns:
        Resized frame (may be smaller than the bounding box).
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale >= 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame: Frame) -> Frame:
    """Convert a BGR frame to RGB.

    Required before passing frames to libraries that expect RGB (MediaPipe,
    PyTorch torchvision, Matplotlib, etc.).

    Args:
        frame: BGR image array of shape ``(H, W, 3)``.

    Returns:
        RGB image array of shape ``(H, W, 3)``.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: Frame) -> Frame:
    """Convert an RGB frame to BGR (for saving with OpenCV).

    Args:
        frame: RGB image array of shape ``(H, W, 3)``.

    Returns:
        BGR image array of shape ``(H, W, 3)``.
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def frame_to_pil(frame: Frame) -> Image.Image:
    """Convert a BGR OpenCV frame to a PIL Image (RGB).

    Args:
        frame: BGR NumPy array.

    Returns:
        PIL ``Image`` in RGB mode.
    """
    return Image.fromarray(bgr_to_rgb(frame))


def pil_to_frame(image: Image.Image) -> Frame:
    """Convert a PIL Image (RGB) to a BGR OpenCV frame.

    Args:
        image: PIL ``Image`` object.

    Returns:
        BGR NumPy array.
    """
    return rgb_to_bgr(np.array(image))


def crop_bbox(
    frame: Frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad: int = 0,
) -> Frame:
    """Crop a bounding box region from a frame with optional padding.

    Coordinates are clamped to valid frame boundaries automatically.

    Args:
        frame: Source image array.
        x1: Left boundary (pixel).
        y1: Top boundary (pixel).
        x2: Right boundary (pixel).
        y2: Bottom boundary (pixel).
        pad: Extra pixels to add on all sides.

    Returns:
        Cropped image array.

    Raises:
        ValueError: If the resulting crop is empty.
    """
    h, w = frame.shape[:2]
    x1c = max(0, x1 - pad)
    y1c = max(0, y1 - pad)
    x2c = min(w, x2 + pad)
    y2c = min(h, y2 + pad)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        raise ValueError(
            f"Crop is empty after clamping — bbox ({x1},{y1},{x2},{y2}) frame ({w}×{h})"
        )
    return crop


def draw_text_with_background(
    frame: Frame,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 4,
) -> Frame:
    """Draw a text label with a filled background rectangle.

    Args:
        frame: Image to draw on (modified in-place).
        text: String to render.
        position: ``(x, y)`` top-left of the text.
        font_scale: OpenCV font scale.
        thickness: Text stroke thickness.
        text_color: BGR text colour.
        bg_color: BGR background rectangle colour.
        padding: Background padding in pixels.

    Returns:
        Modified frame (same array, edited in-place).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        thickness=cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return frame


def normalize_frame(frame: Frame) -> np.ndarray:
    """Normalize a uint8 frame to float32 in the range [0.0, 1.0].

    Args:
        frame: Input uint8 image array.

    Returns:
        Float32 array in [0.0, 1.0].
    """
    return frame.astype(np.float32) / 255.0


def denormalize_frame(frame: np.ndarray) -> Frame:
    """Convert a float32 [0.0, 1.0] frame back to uint8 [0, 255].

    Args:
        frame: Float32 image array.

    Returns:
        uint8 image array.
    """
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def get_frame_shape(frame: Frame) -> Tuple[int, int]:
    """Return ``(width, height)`` of a frame.

    Args:
        frame: Image array.

    Returns:
        Tuple of ``(width, height)`` in pixels.
    """
    h, w = frame.shape[:2]
    return w, h


def stack_frames_side_by_side(frames: list[Frame], gap: int = 2, gap_color: int = 50) -> Frame:
    """Horizontally concatenate multiple frames of the same height.

    Frames that differ in size are resized to match the first frame's height
    before stacking.

    Args:
        frames: List of BGR image arrays.
        gap: Pixel gap between panels.
        gap_color: Greyscale value (0–255) for the gap fill colour.

    Returns:
        Horizontally concatenated frame.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames list must not be empty")

    target_h = frames[0].shape[0]
    panels = []
    for f in frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = int(f.shape[1] * scale)
            f = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        panels.append(f)
        if gap > 0:
            divider = np.full((target_h, gap, 3), gap_color, dtype=np.uint8)
            panels.append(divider)

    # Remove the trailing divider
    if gap > 0:
        panels = panels[:-1]

    return np.concatenate(panels, axis=1)


def stack_frames_2x2(
    top_left: Frame,
    top_right: Frame,
    bottom_left: Frame,
    bottom_right: Frame,
    target_w: int = 640,
    target_h: int = 360,
    gap: int = 2,
    gap_color: int = 40,
) -> Frame:
    """Arrange four frames in a 2×2 grid layout.

    Args:
        top_left: Frame for top-left quadrant.
        top_right: Frame for top-right quadrant.
        bottom_left: Frame for bottom-left quadrant.
        bottom_right: Frame for bottom-right quadrant.
        target_w: Width of each individual panel.
        target_h: Height of each individual panel.
        gap: Pixel gap between panels.
        gap_color: Greyscale fill for gaps.

    Returns:
        Combined 2×2 grid frame.
    """
    def _resize(f: Frame) -> Frame:
        return cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    tl, tr, bl, br = _resize(top_left), _resize(top_right), _resize(bottom_left), _resize(bottom_right)

    top_row = stack_frames_side_by_side([tl, tr], gap=gap, gap_color=gap_color)
    bot_row = stack_frames_side_by_side([bl, br], gap=gap, gap_color=gap_color)

    if gap > 0:
        h_gap = np.full((gap, top_row.shape[1], 3), gap_color, dtype=np.uint8)
        return np.concatenate([top_row, h_gap, bot_row], axis=0)
    return np.concatenate([top_row, bot_row], axis=0)
