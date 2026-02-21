"""Evaluator for G-134_select_next_figure_large_small_alternating_sequence_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-134_select_next_figure_large_small_alternating_sequence_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class SelectNextFigureLargeSmallEvaluator(BaseEvaluator):
    """
    G-134: Select next figure large-small alternating evaluator.
    
    Rule-based evaluation:
    - Alternating pattern recognition (40%): Recognize big-small-big pattern
    - Shape type matching (30%): Correct shape type selected (same as sequence)
    - Size judgment (20%): Correct size (small) selected based on pattern
    - Visual annotation quality (10%): Red circle properly marks the selection
    """
    
    TASK_WEIGHTS = {
        'pattern': 0.40,
        'shape_type': 0.30,
        'size': 0.20,
        'annotation': 0.10
    }
    
    def _detect_shapes_with_size(self, frame: np.ndarray) -> List[Dict]:
        """Detect shapes and their sizes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find colored areas (non-white, non-black)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Determine shape type
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)
            
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                shape_type = 'square'
            elif vertices == 5:
                shape_type = 'pentagon'
            else:
                shape_type = 'circle'
            
            shapes.append({
                'type': shape_type,
                'center': (cx, cy),
                'area': area,
                'vertices': vertices
            })
        
        return shapes
    
    def _detect_red_circle_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:  # Reasonably circular
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _detect_marking_by_diff(self, first_frame: np.ndarray, final_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect marking by comparing first and final frames (for cases where shapes are red)."""
        # Compute difference
        diff = cv2.absdiff(first_frame, final_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours in difference
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Marking should be reasonably large
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate select next figure large-small alternating task."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_first = gt_first_frame
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect shapes and red marking
        gen_shapes = self._detect_shapes_with_size(last_frame)
        gt_shapes = self._detect_shapes_with_size(gt_last)
        
        # Try standard red marking detection first
        gen_marking = self._detect_red_circle_marking(last_frame)
        gt_marking = self._detect_red_circle_marking(gt_last)
        
        # If shapes are red, use frame difference to detect marking
        if first_frame is not None and gt_first is not None:
            gen_marking_diff = self._detect_marking_by_diff(first_frame, last_frame)
            gt_marking_diff = self._detect_marking_by_diff(gt_first, gt_last)
            
            # Use diff-based marking if available (more reliable when shapes are red)
            if gen_marking_diff is not None:
                gen_marking = gen_marking_diff
            if gt_marking_diff is not None:
                gt_marking = gt_marking_diff
        
        # 1. Pattern recognition: Check if marking is at correct position
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['pattern'] = max(0, 1.0 - dist / 100.0)
        elif gen_marking is not None:
            scores['pattern'] = 0.2  # Detection failed
        else:
            scores['pattern'] = 0.0
        
        # 2. Shape type matching: Check if marked shape has correct type
        if gen_shapes and gt_shapes:
            # Find shape nearest to marking
            gen_marked_shape = None
            if gen_marking is not None:
                min_dist = float('inf')
                for shape in gen_shapes:
                    dist = np.sqrt((shape['center'][0] - gen_marking[0])**2 + 
                                  (shape['center'][1] - gen_marking[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        gen_marked_shape = shape
            
            gt_marked_shape = None
            if gt_marking is not None:
                min_dist = float('inf')
                for shape in gt_shapes:
                    dist = np.sqrt((shape['center'][0] - gt_marking[0])**2 + 
                                  (shape['center'][1] - gt_marking[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        gt_marked_shape = shape
            
            if gen_marked_shape is not None and gt_marked_shape is not None:
                if gen_marked_shape['type'] == gt_marked_shape['type']:
                    scores['shape_type'] = 1.0
                else:
                    scores['shape_type'] = 0.3
            else:
                scores['shape_type'] = 0.2  # Detection failed
        else:
            scores['shape_type'] = 0.2  # Detection failed
        
        # 3. Size judgment: Check if marked shape has correct size category
        if gen_shapes and gt_shapes and gen_marking is not None and gt_marking is not None:
            # Get size of marked shapes
            gen_marked_area = 0
            for shape in gen_shapes:
                dist = np.sqrt((shape['center'][0] - gen_marking[0])**2 + 
                              (shape['center'][1] - gen_marking[1])**2)
                if dist < 100:
                    gen_marked_area = shape['area']
                    break
            
            gt_marked_area = 0
            for shape in gt_shapes:
                dist = np.sqrt((shape['center'][0] - gt_marking[0])**2 + 
                              (shape['center'][1] - gt_marking[1])**2)
                if dist < 100:
                    gt_marked_area = shape['area']
                    break
            
            if gen_marked_area > 0 and gt_marked_area > 0:
                # Compare relative sizes
                area_ratio = min(gen_marked_area, gt_marked_area) / max(gen_marked_area, gt_marked_area)
                scores['size'] = area_ratio
            else:
                scores['size'] = 0.2  # Detection failed
        else:
            scores['size'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Check red circle presence and quality
        if gen_marking is not None:
            hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
            hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
            red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
            
            red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
            red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
            
            scores['annotation'] = red_overlap / red_union if red_union > 0 else 0.5
        else:
            scores['annotation'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
