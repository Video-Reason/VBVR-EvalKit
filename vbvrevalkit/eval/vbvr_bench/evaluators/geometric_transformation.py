"""Evaluator for O-6_2d_geometric_transformation_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-6_2d_geometric_transformation_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class GeometricTransformationEvaluator(BaseEvaluator):
    """
    O-6: 2D geometric transformation (rotation) evaluator.
    
    Rule-based evaluation:
    - Rotation center correctness (30%): Rotate around marked point
    - Rotation angle accuracy (35%): Shape aligns with target outline
    - Position alignment precision (25%): Overlap with target
    - Shape fidelity (10%): Size and shape preserved
    """
    
    TASK_WEIGHTS = {
        'rotation_center': 0.30,
        'rotation_angle': 0.35,
        'position_alignment': 0.25,
        'shape_fidelity': 0.10
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
        
        scores['rotation_center'] = self._evaluate_rotation_center(video_frames)
        scores['rotation_angle'] = self._evaluate_rotation_angle(first_frame, final_frame)
        scores['position_alignment'] = self._evaluate_position(first_frame, final_frame)
        scores['shape_fidelity'] = self._evaluate_shape_fidelity(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_rotation_center(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Check if rotation is around the correct center point."""
        # Track shape center across frames
        centers = []
        for frame in video_frames[::max(1, len(video_frames)//10)]:
            center = self._find_shape_center(frame)
            if center:
                centers.append(center)
        
        if len(centers) < 3:
            return 0.0  # STRICT: Not enough centers detected
        
        # Check if centers follow circular path
        # Calculate variance of distances from a potential center
        all_x = [c[0] for c in centers]
        all_y = [c[1] for c in centers]
        
        avg_x = np.mean(all_x)
        avg_y = np.mean(all_y)
        
        distances = [np.sqrt((c[0] - avg_x)**2 + (c[1] - avg_y)**2) for c in centers]
        dist_var = np.var(distances)
        
        # Low variance means consistent rotation around a center
        if dist_var < 100:
            return 1.0
        elif dist_var < 500:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_rotation_angle(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if rotation angle is correct."""
        # Detect target outline in first frame
        target = self._detect_target_outline(first_frame)
        
        # Detect shape in final frame
        shape = self._detect_main_shape(final_frame)
        
        if target is None or shape is None:
            return 0.5
        
        # Compare shape orientation with target
        target_angle = self._get_contour_angle(target)
        shape_angle = self._get_contour_angle(shape)
        
        angle_diff = abs(target_angle - shape_angle)
        angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wrap-around
        
        if angle_diff < 10:
            return 1.0
        elif angle_diff < 20:
            return 0.8
        elif angle_diff < 45:
            return 0.5
        else:
            return 0.2
    
    def _evaluate_position(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if shape aligns with target outline."""
        # Get target and shape positions
        target_center = self._find_target_center(first_frame)
        shape_center = self._find_shape_center(final_frame)
        
        if target_center is None or shape_center is None:
            return 0.5
        
        dist = np.sqrt((target_center[0] - shape_center[0])**2 + 
                      (target_center[1] - shape_center[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_shape_fidelity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if shape is preserved."""
        first_shape = self._detect_main_shape(first_frame)
        final_shape = self._detect_main_shape(final_frame)
        
        if first_shape is None or final_shape is None:
            return 0.5
        
        # Compare areas
        first_area = cv2.contourArea(first_shape)
        final_area = cv2.contourArea(final_shape)
        
        if first_area == 0:
            return 0.5
        
        area_ratio = final_area / first_area
        
        if 0.8 < area_ratio < 1.2:
            return 1.0
        elif 0.6 < area_ratio < 1.4:
            return 0.7
        else:
            return 0.4
    
    def _find_shape_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find center of main shape."""
        shape = self._detect_main_shape(frame)
        if shape is None:
            return None
        
        M = cv2.moments(shape)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _detect_main_shape(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the main (colored) shape."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect colored (not gray) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _detect_target_outline(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect target outline (dashed or dotted)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _find_target_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find center of target outline."""
        target = self._detect_target_outline(frame)
        if target is None:
            return None
        
        M = cv2.moments(target)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _get_contour_angle(self, contour: np.ndarray) -> float:
        """Get orientation angle of contour."""
        if len(contour) < 5:
            return 0.0
        
        ellipse = cv2.fitEllipse(contour)
        return ellipse[2]
