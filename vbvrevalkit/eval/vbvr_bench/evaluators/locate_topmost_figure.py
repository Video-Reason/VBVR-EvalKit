"""Evaluator for G-140_locate_topmost_unobscured_figure_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-140_locate_topmost_unobscured_figure_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class LocateTopmostFigureEvaluator(BaseEvaluator):
    """
    G-140: Locate topmost (unobscured) figure in overlapping shapes.
    
    Rule-based evaluation:
    - Z-order identification (45%): Outline the topmost (least occluded) shape
    - Outline accuracy (30%): Red outline follows shape boundary
    - Marking uniqueness (15%): Only one shape marked
    - Visual clarity (10%): Clear visible outline
    """
    
    TASK_WEIGHTS = {
        'z_order_identification': 0.45,
        'outline_accuracy': 0.30,
        'marking_uniqueness': 0.15,
        'visual_clarity': 0.10
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
        
        # 1. Z-order identification (45%)
        scores['z_order_identification'] = self._evaluate_z_order(
            first_frame, final_frame
        )
        
        # 2. Outline accuracy (30%)
        scores['outline_accuracy'] = self._evaluate_outline(final_frame)
        
        # 3. Marking uniqueness (15%)
        scores['marking_uniqueness'] = self._evaluate_uniqueness(final_frame)
        
        # 4. Visual clarity (10%)
        scores['visual_clarity'] = self._evaluate_clarity(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_z_order(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the topmost (least occluded) shape is outlined."""
        # Detect the topmost shape from first frame
        topmost_center = self._find_topmost_shape_center(first_frame)
        
        # Detect outline position in final frame
        outline = self._detect_outline(final_frame)
        
        if outline is None:
            return 0.0
        
        outline_center = self._get_contour_center(outline)
        
        if topmost_center is None or outline_center is None:
            return 0.3
        
        # Check if outline is around topmost shape
        dist = np.sqrt((outline_center[0] - topmost_center[0])**2 + 
                      (outline_center[1] - topmost_center[1])**2)
        
        if dist < 50:
            return 1.0
        elif dist < 100:
            return 0.7
        elif dist < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_outline(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate outline drawing quality."""
        outline = self._detect_outline(final_frame)
        
        if outline is None:
            return 0.0
        
        # Check if outline forms a closed shape
        perimeter = cv2.arcLength(outline, True)
        area = cv2.contourArea(outline)
        
        if perimeter > 100 and area > 500:
            # Check circularity (how well-formed the outline is)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            if circularity > 0.1:  # Reasonable shape
                return 1.0
            return 0.7
        elif perimeter > 50:
            return 0.5
        return 0.3
    
    def _evaluate_uniqueness(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if only one shape is outlined."""
        outlines = self._detect_all_outlines(final_frame)
        
        if len(outlines) == 0:
            return 0.0
        elif len(outlines) == 1:
            return 1.0
        else:
            return max(0.3, 1.0 - 0.3 * (len(outlines) - 1))
    
    def _evaluate_clarity(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate outline visual clarity."""
        outline = self._detect_outline(final_frame)
        if outline is None:
            return 0.0
        
        perimeter = cv2.arcLength(outline, True)
        if perimeter > 150:
            return 1.0
        elif perimeter > 100:
            return 0.8
        elif perimeter > 50:
            return 0.5
        return perimeter / 100
    
    def _find_topmost_shape_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the center of the topmost (least occluded) shape."""
        # Convert to grayscale and find shapes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find shape boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # The topmost shape typically has the most visible edge pixels
        # Find contour with most edge density
        best_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _detect_outline(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the main red outline in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _detect_all_outlines(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect all red outlines in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return [c for c in contours if cv2.contourArea(c) > 100]
    
    def _get_contour_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
