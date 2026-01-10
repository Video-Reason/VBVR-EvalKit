"""
Frame sampling module for multi-frame evaluation.

Provides hybrid sampling strategy combining:
- Time-based uniform sampling from video tail
- Keyframe detection based on scene changes
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """Container for a sampled frame with metadata."""
    image: np.ndarray          # RGB image array
    frame_index: int           # Original frame index in video
    timestamp: float           # Timestamp in seconds
    is_keyframe: bool = False  # Whether detected as keyframe


class FrameSampler:
    """
    Hybrid frame sampler for video evaluation.

    Combines time-based uniform sampling with keyframe detection to capture
    both temporal coverage and significant scene changes.

    Attributes:
        n_frames: Default number of frames to sample
        last_seconds: Duration from video end to sample from
        keyframe_threshold: Threshold for keyframe detection (higher = fewer keyframes)
        min_keyframe_interval: Minimum frames between keyframes
    """

    def __init__(
        self,
        n_frames: int = 5,
        last_seconds: float = 3.0,
        keyframe_threshold: float = 30.0,
        min_keyframe_interval: int = 5
    ):
        self.n_frames = n_frames
        self.last_seconds = last_seconds
        self.keyframe_threshold = keyframe_threshold
        self.min_keyframe_interval = min_keyframe_interval

    def sample(
        self,
        video_path: Union[str, Path],
        n_frames: Optional[int] = None,
        last_seconds: Optional[float] = None,
        strategy: str = "hybrid"
    ) -> List[SampledFrame]:
        """
        Sample frames from video using specified strategy.

        Args:
            video_path: Path to video file
            n_frames: Number of frames to sample (overrides default)
            last_seconds: Duration from end to sample from (overrides default)
            strategy: Sampling strategy - "uniform", "keyframe", or "hybrid"

        Returns:
            List of SampledFrame objects sorted by timestamp

        Raises:
            ValueError: If video cannot be opened or has no frames
        """
        video_path = Path(video_path)
        n_frames = n_frames or self.n_frames
        last_seconds = last_seconds or self.last_seconds

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0 or fps == 0:
                raise ValueError(f"Video has no frames or invalid FPS: {video_path}")

            duration = total_frames / fps

            # Calculate frame range for sampling
            sample_duration = min(last_seconds, duration)
            start_time = max(0, duration - sample_duration)
            start_frame = int(start_time * fps)

            logger.debug(
                f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.2f}s. "
                f"Sampling from frame {start_frame} ({start_time:.2f}s)"
            )

            if strategy == "uniform":
                return self._sample_uniform(cap, start_frame, total_frames, n_frames, fps)
            elif strategy == "keyframe":
                return self._sample_keyframes(cap, start_frame, total_frames, n_frames, fps)
            elif strategy == "hybrid":
                return self._sample_hybrid(cap, start_frame, total_frames, n_frames, fps)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        finally:
            cap.release()

    def _sample_uniform(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        total_frames: int,
        n_frames: int,
        fps: float
    ) -> List[SampledFrame]:
        """
        Sample frames uniformly distributed in time range.

        Divides the sampling range into n_frames equal intervals and takes
        one frame from each interval.
        """
        frame_range = total_frames - start_frame

        if frame_range <= n_frames:
            # Fewer frames available than requested, take all
            indices = list(range(start_frame, total_frames))
        else:
            # Uniform distribution
            step = frame_range / n_frames
            indices = [int(start_frame + i * step) for i in range(n_frames)]
            # Always include the last frame
            if indices[-1] != total_frames - 1:
                indices[-1] = total_frames - 1

        return self._read_frames(cap, indices, fps, is_keyframe=False)

    def _sample_keyframes(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        total_frames: int,
        n_frames: int,
        fps: float
    ) -> List[SampledFrame]:
        """
        Sample frames based on scene change detection.

        Uses frame difference to detect significant changes, then selects
        the top n_frames keyframes plus the final frame.
        """
        keyframe_indices = self._detect_keyframes(cap, start_frame, total_frames)

        # Always include last frame
        if total_frames - 1 not in keyframe_indices:
            keyframe_indices.append(total_frames - 1)

        # Limit to n_frames, prioritizing later keyframes (closer to end)
        if len(keyframe_indices) > n_frames:
            keyframe_indices = keyframe_indices[-n_frames:]

        return self._read_frames(cap, keyframe_indices, fps, is_keyframe=True)

    def _sample_hybrid(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        total_frames: int,
        n_frames: int,
        fps: float
    ) -> List[SampledFrame]:
        """
        Hybrid sampling combining uniform and keyframe strategies.

        Strategy:
        1. Detect keyframes in the sampling range
        2. Take uniform samples as baseline
        3. Replace uniform samples with nearby keyframes if available
        4. Ensure temporal coverage while capturing important changes
        """
        # Get uniform samples as baseline
        frame_range = total_frames - start_frame
        step = max(1, frame_range // n_frames)
        uniform_indices = [int(start_frame + i * step) for i in range(n_frames)]

        # Ensure last frame is included
        uniform_indices[-1] = total_frames - 1

        # Detect keyframes
        keyframe_indices = set(self._detect_keyframes(cap, start_frame, total_frames))

        # Hybrid selection: prefer keyframes near uniform sample points
        selected_indices = []
        keyframe_flags = []

        for uniform_idx in uniform_indices:
            # Look for keyframes within half-step distance
            search_range = step // 2
            nearby_keyframes = [
                kf for kf in keyframe_indices
                if abs(kf - uniform_idx) <= search_range and kf not in selected_indices
            ]

            if nearby_keyframes:
                # Choose closest keyframe
                chosen = min(nearby_keyframes, key=lambda x: abs(x - uniform_idx))
                selected_indices.append(chosen)
                keyframe_flags.append(True)
                keyframe_indices.discard(chosen)
            else:
                selected_indices.append(uniform_idx)
                keyframe_flags.append(False)

        # Sort by frame index
        sorted_pairs = sorted(zip(selected_indices, keyframe_flags), key=lambda x: x[0])
        selected_indices = [p[0] for p in sorted_pairs]
        keyframe_flags = [p[1] for p in sorted_pairs]

        return self._read_frames_with_flags(cap, selected_indices, keyframe_flags, fps)

    def _detect_keyframes(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        total_frames: int
    ) -> List[int]:
        """
        Detect keyframes using frame difference.

        Computes absolute difference between consecutive frames and marks
        frames with difference above threshold as keyframes.

        Returns:
            List of frame indices identified as keyframes
        """
        keyframes = []
        prev_gray = None
        last_keyframe = -self.min_keyframe_interval  # Allow first frame to be keyframe

        for frame_idx in range(start_frame, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # Compute frame difference
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = np.mean(diff)

                # Check if significant change and enough interval since last keyframe
                if (mean_diff > self.keyframe_threshold and
                    frame_idx - last_keyframe >= self.min_keyframe_interval):
                    keyframes.append(frame_idx)
                    last_keyframe = frame_idx

            prev_gray = gray

        logger.debug(f"Detected {len(keyframes)} keyframes in range [{start_frame}, {total_frames})")
        return keyframes

    def _read_frames(
        self,
        cap: cv2.VideoCapture,
        indices: List[int],
        fps: float,
        is_keyframe: bool
    ) -> List[SampledFrame]:
        """Read frames at specified indices."""
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(SampledFrame(
                    image=rgb_frame,
                    frame_index=idx,
                    timestamp=idx / fps,
                    is_keyframe=is_keyframe
                ))
            else:
                logger.warning(f"Failed to read frame at index {idx}")

        return frames

    def _read_frames_with_flags(
        self,
        cap: cv2.VideoCapture,
        indices: List[int],
        keyframe_flags: List[bool],
        fps: float
    ) -> List[SampledFrame]:
        """Read frames with individual keyframe flags."""
        frames = []

        for idx, is_kf in zip(indices, keyframe_flags):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(SampledFrame(
                    image=rgb_frame,
                    frame_index=idx,
                    timestamp=idx / fps,
                    is_keyframe=is_kf
                ))
            else:
                logger.warning(f"Failed to read frame at index {idx}")

        return frames

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get basic video information.

        Returns:
            Dictionary with keys: total_frames, fps, duration, width, height
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            info = {
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
            return info
        finally:
            cap.release()
