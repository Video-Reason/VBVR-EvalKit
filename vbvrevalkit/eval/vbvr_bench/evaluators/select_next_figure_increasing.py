"""Evaluator for G-131_select_next_figure_increasing_size_sequence_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-131_select_next_figure_increasing_size_sequence_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class SelectNextFigureIncreasingEvaluator(BaseEvaluator):
    """
    G-131: Select next figure increasing size evaluator.
    
    Evaluates:
    - Increasing pattern recognition (40%): Recognize small to large pattern
    - Shape type matching (30%): Correct shape type selected
    - Candidate selection accuracy (20%): Correct candidate marked
    - Visual annotation quality (10%): Red circle proper
    """
    
    TASK_WEIGHTS = {
        'pattern': 0.40,
        'shape_type': 0.30,
        'selection': 0.20,
        'annotation': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate select next figure increasing task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect red circle marking
        gen_red = self._detect_red_circle(last_frame)
        gt_red = self._detect_red_circle(gt_last)
        
        # 1. Pattern recognition (40%): Check if correct answer selected
        pattern_score = self._evaluate_pattern_recognition(gen_red, gt_red, last_frame, gt_last)
        scores['pattern'] = pattern_score
        
        # 2. Shape type matching (30%): Correct shape type
        shape_score = self._evaluate_shape_type(gen_red, gt_red, last_frame, gt_last)
        scores['shape_type'] = shape_score
        
        # 3. Selection accuracy (20%): Red circle position
        selection_score = self._evaluate_selection_accuracy(gen_red, gt_red)
        scores['selection'] = selection_score
        
        # 4. Annotation quality (10%): Red circle proper
        annotation_score = self._evaluate_annotation_quality(gen_red)
        scores['annotation'] = annotation_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red circle annotation."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 100:
            return None
        
        # Check circularity
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
        
        return {
            'center': (cx, cy),
            'area': area,
            'circularity': circularity,
            'contour': largest
        }
    
    def _evaluate_pattern_recognition(self, gen_red: Optional[Dict], gt_red: Optional[Dict],
                                       gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if the increasing pattern was recognized."""
        if gen_red is None:
            return 0.0
        if gt_red is None:
            return 0.5
        
        # Compare marked positions
        gen_center = gen_red['center']
        gt_center = gt_red['center']
        
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.8
        elif normalized_dist < 0.20:
            return 0.5
        else:
            return max(0.2, 1.0 - normalized_dist)
    
    def _evaluate_shape_type(self, gen_red: Optional[Dict], gt_red: Optional[Dict],
                             gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if correct shape type was selected."""
        if gen_red is None or gt_red is None:
            return 0.0 if gen_red is None else 0.5
        
        # Extract region inside red circle and compare
        gen_center = gen_red['center']
        gt_center = gt_red['center']
        
        # Get regions around centers
        radius = 40
        
        gen_roi = self._extract_roi(gen_frame, gen_center, radius)
        gt_roi = self._extract_roi(gt_frame, gt_center, radius)
        
        if gen_roi is None or gt_roi is None:
            return 0.5
        
        # Compare using histogram
        gen_hist = cv2.calcHist([gen_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        gt_hist = cv2.calcHist([gt_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        cv2.normalize(gen_hist, gen_hist)
        cv2.normalize(gt_hist, gt_hist)
        
        similarity = cv2.compareHist(gen_hist, gt_hist, cv2.HISTCMP_CORREL)
        
        return max(0, similarity)
    
    def _extract_roi(self, frame: np.ndarray, center: Tuple[int, int], radius: int) -> Optional[np.ndarray]:
        """Extract region of interest around center."""
        h, w = frame.shape[:2]
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(w, center[0] + radius)
        y2 = min(h, center[1] + radius)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2]
    
    def _evaluate_selection_accuracy(self, gen_red: Optional[Dict], gt_red: Optional[Dict]) -> float:
        """Evaluate red circle position accuracy."""
        if gen_red is None:
            return 0.0
        if gt_red is None:
            return 0.5
        
        gen_center = gen_red['center']
        gt_center = gt_red['center']
        
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        
        if distance < 20:
            return 1.0
        elif distance < 50:
            return 0.7
        elif distance < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_annotation_quality(self, gen_red: Optional[Dict]) -> float:
        """Evaluate quality of red circle annotation."""
        if gen_red is None:
            return 0.0
        
        # Check circularity (should be close to 1 for a circle)
        circularity = gen_red['circularity']
        
        if circularity > 0.7:
            return 1.0
        elif circularity > 0.5:
            return 0.7
        elif circularity > 0.3:
            return 0.4
        else:
            return 0.2


# Mapping of task names to evaluators
OPEN60_EVALUATORS_PART2 = {
    'G-25_seperate_object_spinning_data-generator': SeparateObjectsSpinningEvaluator,
    'G-29_chart_extreme_with_data_data-generator': ChartExtremeEvaluator,
    'G-31_directed_graph_navigation_data-generator': DirectedGraphNavigationEvaluator,
    'G-39_attention_shift_different_data-generator': AttentionShiftEvaluator,
    'G-41_grid_highest_cost_data-generator': GridHighestCostEvaluator,
    'G-43_understand_scene_structure_data-generator': UnderstandSceneStructureEvaluator,
    'G-45_key_door_matching_data-generator': KeyDoorMatchingEvaluator,
    'G-51_predict_next_color_data-generator': PredictNextColorEvaluator,
    'G-54_connecting_color_data-generator': ConnectingColorEvaluator,
    'G-131_select_next_figure_increasing_size_sequence_data-generator': SelectNextFigureIncreasingEvaluator,
