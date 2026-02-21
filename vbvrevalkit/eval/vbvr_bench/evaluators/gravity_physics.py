"""Evaluator for O-62_gravity_physics_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-62_gravity_physics_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class GravityPhysicsEvaluator(BaseEvaluator):
    """
    O-62: Gravity physics simulation evaluator.
    
    Rule-based evaluation:
    - Physics accuracy (50%): Correct trajectory following gravity equations
      - v(t) = v₀ - g·t
      - h(t) = h₀ + v₀·t - ½g·t²
    - Final position accuracy (30%): Ball at correct location
    - Motion quality (15%): Smooth acceleration
    - Visual preservation (5%): Scene elements unchanged
    """
    
    TASK_WEIGHTS = {
        'physics_accuracy': 0.50,
        'final_position': 0.30,
        'motion_quality': 0.15,
        'visual_preservation': 0.05
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        final_frame = video_frames[-1]
        first_frame = video_frames[0]
        
        scores['physics_accuracy'] = self._evaluate_physics(video_frames)
        scores['final_position'] = self._evaluate_position(
            final_frame, gt_final_frame
        )
        scores['motion_quality'] = self._evaluate_motion(video_frames)
        scores['visual_preservation'] = self._evaluate_visual(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red ball position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        return None
    
    def _evaluate_physics(self, video_frames: List[np.ndarray]) -> float:
        """Check if physics simulation is accurate."""
        # Track ball positions through video
        positions = []
        for frame in video_frames[::max(1, len(video_frames)//20)]:
            pos = self._detect_ball(frame)
            if pos is not None:
                positions.append(pos[1])  # y position (height)
        
        if len(positions) < 5:
            return 0.5
        
        # For gravity, y position should follow parabolic curve
        # Second derivative (acceleration) should be approximately constant
        velocities = np.diff(positions)
        accelerations = np.diff(velocities)
        
        if len(accelerations) < 2:
            return 0.5
        
        # Check if acceleration is roughly constant (gravity)
        accel_variance = np.var(accelerations)
        mean_accel = np.mean(np.abs(accelerations))
        
        if mean_accel > 0:
            cv = np.sqrt(accel_variance) / mean_accel
            return max(0, 1.0 - cv)
        
        return 0.5
    
    def _evaluate_position(self, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if ball is at correct final position."""
        final_pos = self._detect_ball(final)
        
        if gt_final is not None:
            gt_pos = self._detect_ball(gt_final)
            
            # If ball not in generated final but in GT final, check if frames match
            if final_pos is None and gt_pos is not None:
                # Check if the frames are nearly identical (GT vs GT case)
                diff = cv2.absdiff(final, gt_final)
                if np.mean(diff) < 10:
                    return 1.0  # Frames are essentially identical
                return 0.0  # Ball missing from generated
            
            if final_pos is not None and gt_pos is not None:
                distance = np.sqrt((final_pos[0] - gt_pos[0])**2 + (final_pos[1] - gt_pos[1])**2)
                if distance < 20:
                    return 1.0
                elif distance < 50:
                    return 0.7
                elif distance < 100:
                    return 0.4
                return 0.1
        
        if final_pos is None:
            return 0.0
        
        # Check if ball is near ground (bottom of frame)
        h = final.shape[0]
        if final_pos[1] > h * 0.7:  # Ball should be in lower part
            return 0.8
        
        return 0.5
    
    def _evaluate_motion(self, video_frames: List[np.ndarray]) -> float:
        """Check if motion is smooth with acceleration."""
        if len(video_frames) < 5:
            return 0.5
        
        # Track ball positions
        positions = []
        for frame in video_frames[::max(1, len(video_frames)//20)]:
            pos = self._detect_ball(frame)
            if pos is not None:
                positions.append(pos[1])
        
        if len(positions) < 3:
            return 0.5
        
        # Check for acceleration (increasing velocity)
        velocities = np.diff(positions)
        if len(velocities) < 2:
            return 0.5
        
        # For falling objects, velocity should generally increase (positive acceleration)
        increasing = np.sum(np.diff(velocities) >= -5)  # Allow some noise
        total = len(velocities) - 1
        
        if total == 0:
            return 0.5
        return min(1.0, increasing / total + 0.3)
    
    def _evaluate_visual(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if scene elements (ground, markers) are preserved."""
        # Compare bottom portion (ground)
        h = first.shape[0]
        ground_first = first[h-60:, :]
        ground_final = final[h-60:, :]
        
        # Compare histograms
        hist_first = cv2.calcHist([ground_first], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_final = cv2.calcHist([ground_final], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        correlation = cv2.compareHist(hist_first, hist_final, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)
