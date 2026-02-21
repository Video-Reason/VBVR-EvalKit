"""Evaluator for O-2_pigment_color_mixing_subtractive_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-2_pigment_color_mixing_subtractive_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class PigmentColorMixingEvaluator(BaseEvaluator):
    """
    O-2: Pigment color mixing (subtractive) evaluator.
    
    Rule-based evaluation:
    - Color mixing rule correctness (60%): CMY subtractive mixing result color
    - Mixing area fill accuracy (25%): Complete fill in marked zone
    - Scene preservation (10%): Original circles unchanged
    - Visual quality (5%): Clean edges, no artifacts
    
    CMY Subtractive Mixing Rules:
    - Cyan + Magenta = Blue
    - Cyan + Yellow = Green
    - Magenta + Yellow = Red
    - Cyan + Magenta + Yellow = Black
    """
    
    # CMY subtractive mixing expected results (in BGR format)
    CMY_MIXING_RULES = {
        ('cyan', 'magenta'): (255, 0, 0),      # Blue
        ('cyan', 'yellow'): (0, 255, 0),        # Green
        ('magenta', 'yellow'): (0, 0, 255),     # Red
        ('cyan', 'magenta', 'yellow'): (0, 0, 0),  # Black
    }
    
    TASK_WEIGHTS = {
        'mixing_correctness': 0.60,
        'fill_accuracy': 0.25,
        'scene_preservation': 0.10,
        'visual_quality': 0.05
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
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        scores['mixing_correctness'] = self._evaluate_color_mixing(
            first_frame, final_frame, gt_final_frame
        )
        scores['fill_accuracy'] = self._evaluate_fill_region(
            first_frame, final_frame
        )
        scores['scene_preservation'] = self._evaluate_preservation(
            first_frame, final_frame
        )
        scores['visual_quality'] = self._evaluate_visual_quality(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_color_mixing(
        self, 
        first_frame: np.ndarray, 
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if mixed color matches expected result."""
        # Get mixed color from final frame (center region)
        mixed_color = self._get_mixed_region_color(final_frame)
        
        if mixed_color is None:
            return 0.0
        
        # If GT final frame available, compare with GT center color
        if gt_final_frame is not None:
            gt_mixed_color = self._get_mixed_region_color(gt_final_frame)
            if gt_mixed_color is not None:
                color_diff = np.sqrt(np.sum((np.array(mixed_color) - np.array(gt_mixed_color)) ** 2))
                
                if color_diff < 30:
                    return 1.0
                elif color_diff < 60:
                    return 0.8
                elif color_diff < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Check if mixed color follows CMY rules
        input_colors = self._detect_input_colors(first_frame)
        
        if len(input_colors) < 2:
            return 0.2
        
        expected_color = self._calculate_expected_mix(input_colors)
        
        if expected_color is None:
            return 0.0
        
        color_diff = np.sqrt(np.sum((np.array(mixed_color) - np.array(expected_color)) ** 2))
        
        if color_diff < 50:
            return 1.0
        elif color_diff < 100:
            return 0.7
        elif color_diff < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_fill_region(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if mixing region is properly filled."""
        h, w = first_frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 60
        
        # Get center region
        region = final_frame[max(0, cy-size):min(h, cy+size), max(0, cx-size):min(w, cx+size)]
        
        if region.size == 0:
            return 0.5
        
        # Check if region has uniform color (properly filled)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Low saturation variance indicates uniform fill
        sat_var = np.var(hsv[:, :, 1])
        
        if sat_var < 500:
            return 1.0
        elif sat_var < 1000:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if original pigment circles are preserved."""
        # Count colored regions in left/right thirds
        h, w = first_frame.shape[:2]
        
        first_left = self._count_colored_pixels(first_frame[:, :w//3])
        first_right = self._count_colored_pixels(first_frame[:, 2*w//3:])
        final_left = self._count_colored_pixels(final_frame[:, :w//3])
        final_right = self._count_colored_pixels(final_frame[:, 2*w//3:])
        
        # Circles should be preserved (similar pixel counts)
        left_ratio = min(first_left, final_left) / max(first_left, final_left, 1)
        right_ratio = min(first_right, final_right) / max(first_right, final_right, 1)
        
        avg_ratio = (left_ratio + right_ratio) / 2
        
        if avg_ratio > 0.8:
            return 1.0
        elif avg_ratio > 0.6:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_visual_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate visual quality."""
        gray = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if 0.01 < edge_ratio < 0.15:
            return 1.0
        return 0.5
    
    def _detect_input_colors(self, frame: np.ndarray) -> List[str]:
        """Detect CMY colors present in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        colors_present = []
        
        # Cyan detection (hue ~90) - lower saturation threshold
        cyan_mask = cv2.inRange(hsv, np.array([80, 50, 50]), np.array([100, 255, 255]))
        if np.sum(cyan_mask > 0) > 500:
            colors_present.append('cyan')
        
        # Magenta detection (hue ~140-170) - lower saturation threshold
        magenta_mask = cv2.inRange(hsv, np.array([130, 50, 50]), np.array([170, 255, 255]))
        if np.sum(magenta_mask > 0) > 500:
            colors_present.append('magenta')
        
        # Yellow detection (hue ~20-40) - lower saturation threshold
        yellow_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([45, 255, 255]))
        if np.sum(yellow_mask > 0) > 500:
            colors_present.append('yellow')
        
        return colors_present
    
    def _get_mixed_region_color(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Get average color of the center mixing region."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 40
        
        region = frame[max(0, cy-size):min(h, cy+size), max(0, cx-size):min(w, cx+size)]
        
        if region.size == 0:
            return None
        
        avg_color = np.mean(region, axis=(0, 1))
        return tuple(int(c) for c in avg_color)
    
    def _calculate_expected_mix(self, input_colors: List[str]) -> Optional[Tuple[int, int, int]]:
        """Calculate expected mixed color based on CMY rules."""
        key = tuple(sorted(input_colors))
        
        for rule_key, result in self.CMY_MIXING_RULES.items():
            if set(key) == set(rule_key):
                return result
        
        return None
    
    def _count_colored_pixels(self, region: np.ndarray) -> int:
        """Count non-white colored pixels."""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        return np.sum(hsv[:, :, 1] > 50)
