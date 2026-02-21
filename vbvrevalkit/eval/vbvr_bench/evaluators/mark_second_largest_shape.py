"""Evaluator for G-161_mark_second_largest_shape_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-161_mark_second_largest_shape_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class MarkSecondLargestShapeEvaluator(BaseEvaluator):
    """
    G-161: Mark the second largest shape.
    
    Rule-based evaluation:
    - Size recognition (40%): Identify second largest correctly
    - Marking precision (35%): Border accurately marks target
    - Marking quality (15%): Border style and color
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'size_recognition': 0.40,
        'marking_precision': 0.35,
        'marking_quality': 0.15,
        'scene_preservation': 0.10
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
        
        # CRITICAL CHECK: Only ONE marking should exist
        marking_count = self._count_red_markings(final_frame)
        if marking_count == 0:
            # No marking at all - task failed
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': 'no_marking_found'
            }
            return 0.0
        elif marking_count > 1:
            # Multiple markings - violation of "only one mark" rule
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': f'multiple_markings_found: {marking_count}'
            }
            return 0.0
        
        # CRITICAL CHECK: No new objects should be generated
        first_shapes = self._detect_shapes_with_area(first_frame)
        final_shapes_no_red = self._detect_shapes_without_red(final_frame)
        
        if len(final_shapes_no_red) > len(first_shapes):
            # New objects generated - violation
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': f'new_objects_generated: first={len(first_shapes)}, final={len(final_shapes_no_red)}'
            }
            return 0.0
        
        # 1. Size recognition (40%) - CRITICAL: Is the CORRECT shape marked?
        size_recognition_score = self._evaluate_size_recognition(
            first_frame, final_frame
        )
        scores['size_recognition'] = size_recognition_score
        
        # CRITICAL: If wrong shape is marked, other scores should be penalized
        correct_shape_marked = size_recognition_score > 0.5
        
        # 2. Marking precision (35%) - Only counts if correct shape is marked
        if correct_shape_marked:
            scores['marking_precision'] = self._evaluate_marking_precision(final_frame)
        else:
            scores['marking_precision'] = 0.0  # Wrong shape - no credit for marking
        
        # 3. Marking quality (15%) - Only counts if correct shape is marked
        if correct_shape_marked:
            scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        else:
            scores['marking_quality'] = 0.0  # Wrong shape - no credit
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _count_red_markings(self, frame: np.ndarray) -> int:
        """Count number of red markings in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Count significant red contours (filter noise)
        return len([c for c in contours if cv2.contourArea(c) > 100])
    
    def _detect_shapes_without_red(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes excluding red marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        frame_no_red = frame.copy()
        frame_no_red[red_mask > 0] = [255, 255, 255]
        return self._detect_shapes_with_area(frame_no_red)
    
    def _evaluate_size_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if second largest shape is correctly identified."""
        # Find shapes sorted by area
        shapes = self._detect_shapes_with_area(first_frame)
        
        if len(shapes) < 2:
            return 0.0  # STRICT: Not enough shapes to have "second largest"
        
        # Sort by area (descending)
        sorted_shapes = sorted(shapes, key=lambda s: s[2], reverse=True)
        second_largest_center = (sorted_shapes[1][0], sorted_shapes[1][1])
        
        # Detect red border/marking in final frame
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        marking_center = self._get_contour_center(marking)
        
        if marking_center is None:
            return 0.0  # STRICT: Can't determine marking position
        
        # Check if marking is on second largest
        dist = np.sqrt((marking_center[0] - second_largest_center[0])**2 + 
                      (marking_center[1] - second_largest_center[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.5  # STRICT: Reduced from 0.7
        else:
            return 0.0  # STRICT: Wrong shape marked
    
    def _evaluate_marking_precision(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border marking precision."""
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        # Check if marking forms a proper border
        perimeter = cv2.arcLength(marking, True)
        area = cv2.contourArea(marking)
        
        if perimeter > 100 and area > 500:
            return 1.0
        elif perimeter > 50:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border marking quality."""
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        perimeter = cv2.arcLength(marking, True)
        if perimeter > 100:
            return 1.0
        else:
            return perimeter / 100
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original shapes are preserved - NO NEW OBJECTS."""
        first_count = len(self._detect_shapes_with_area(first_frame))
        
        # Count shapes excluding red marking
        final_count = len(self._detect_shapes_without_red(final_frame))
        
        # STRICT: No new objects allowed
        if final_count > first_count:
            return 0.0  # New objects generated - violation
        
        if final_count == first_count:
            return 1.0  # Perfect preservation
        elif final_count == first_count - 1:
            return 0.7  # One shape lost
        else:
            return 0.0  # STRICT: Too many shapes changed
    
    def _detect_shapes_with_area(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes with (x, y, area)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, area))
        
        return shapes
    
    def _detect_red_border(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect red border/outline in the frame."""
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
    
    def _get_contour_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
