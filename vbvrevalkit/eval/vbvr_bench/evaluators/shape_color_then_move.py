"""Evaluator for O-11_shape_color_then_move_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-11_shape_color_then_move_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeColorThenMoveEvaluator(BaseEvaluator):
    """
    O-11: Shape color then move evaluator.
    
    Rule-based evaluation:
    - First row preservation (40%): A, B, C in top row MUST remain unchanged
    - Second row completion (35%): D stays, E and F are added with B's color
    - Color accuracy (20%): E and F have B's color
    - Shape count (5%): Final should have 6 shapes (3 top + 3 bottom)
    """
    
    TASK_WEIGHTS = {
        'first_row_preservation': 0.40,
        'second_row_completion': 0.35,
        'color_accuracy': 0.20,
        'shape_count': 0.05
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
        
        h, w = first_frame.shape[:2]
        
        # Detect shapes in first and final frames
        first_shapes = self._detect_shapes_with_info(first_frame)
        final_shapes = self._detect_shapes_with_info(final_frame)
        
        # 1. CRITICAL: First row (top half) must be preserved
        scores['first_row_preservation'] = self._evaluate_first_row_preservation(
            first_shapes, final_shapes, h
        )
        
        # If first row is not preserved, heavily penalize
        if scores['first_row_preservation'] < 0.5:
            self._last_task_details = {
                'first_row_preservation': scores['first_row_preservation'],
                'second_row_completion': 0.0,
                'color_accuracy': 0.0,
                'shape_count': 0.0,
                'first_row_destroyed': True
            }
            return scores['first_row_preservation'] * self.TASK_WEIGHTS['first_row_preservation']
        
        # 2. Second row should have 3 shapes (D, E, F)
        scores['second_row_completion'] = self._evaluate_second_row(
            first_shapes, final_shapes, h
        )
        
        # 3. Color accuracy - E and F should have B's color
        scores['color_accuracy'] = self._evaluate_color_accuracy(
            first_shapes, final_shapes, h, w
        )
        
        # 4. Shape count
        scores['shape_count'] = self._evaluate_shape_count(final_shapes)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_shapes_with_info(self, frame: np.ndarray) -> List[Dict]:
        """Detect shapes with position and color info."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 30
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area - shapes should be reasonably sized
            if 1000 < area < 20000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    mask_cnt = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(frame, mask=mask_cnt)[:3]
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    shapes.append({
                        'center': (cx, cy),
                        'hue': int(hsv_c[0]),
                        'area': area
                    })
        return shapes
    
    def _evaluate_first_row_preservation(self, first_shapes: List[Dict], 
                                         final_shapes: List[Dict], h: int) -> float:
        """Check if first row (A, B, C) is preserved."""
        # Get shapes in top half
        first_top = [s for s in first_shapes if s['center'][1] < h // 2]
        final_top = [s for s in final_shapes if s['center'][1] < h // 2]
        
        if len(first_top) == 0:
            return 0.5
        
        # Should have same number of shapes in top row
        if len(final_top) < len(first_top):
            return 0.1  # Shapes removed from top row
        
        # Check if hues are preserved
        first_hues = sorted([s['hue'] for s in first_top])
        final_hues = sorted([s['hue'] for s in final_top[:len(first_top)]])
        
        matched = 0
        for fh in first_hues:
            for i, fnlh in enumerate(final_hues):
                hue_diff = abs(fh - fnlh)
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    matched += 1
                    break
        
        return matched / len(first_hues) if first_hues else 0.0
    
    def _evaluate_second_row(self, first_shapes: List[Dict], 
                             final_shapes: List[Dict], h: int) -> float:
        """Check if second row has D, E, F."""
        # Get shapes in bottom half
        first_bottom = [s for s in first_shapes if s['center'][1] >= h // 2]
        final_bottom = [s for s in final_shapes if s['center'][1] >= h // 2]
        
        # First frame should have 1 shape (D), final should have 3 (D, E, F)
        if len(first_bottom) == 0:
            return 0.5
        
        if len(final_bottom) >= 3:
            return 1.0
        elif len(final_bottom) == 2:
            return 0.6
        elif len(final_bottom) == 1:
            return 0.3
        else:
            return 0.0
    
    def _evaluate_color_accuracy(self, first_shapes: List[Dict], 
                                 final_shapes: List[Dict], h: int, w: int) -> float:
        """Check if E and F have B's color."""
        # Find B's color (top-middle shape in first frame)
        first_top = [s for s in first_shapes if s['center'][1] < h // 2]
        b_shapes = [s for s in first_top if w // 3 < s['center'][0] < 2 * w // 3]
        
        if len(b_shapes) == 0:
            return 0.5
        
        b_hue = b_shapes[0]['hue']
        
        # Get E and F (middle and right shapes in bottom row of final frame)
        final_bottom = [s for s in final_shapes if s['center'][1] >= h // 2]
        ef_shapes = [s for s in final_bottom if s['center'][0] > w // 3]
        
        if len(ef_shapes) == 0:
            return 0.0
        
        # Check if E and F have B's color
        correct = 0
        for s in ef_shapes:
            hue_diff = abs(s['hue'] - b_hue)
            hue_diff = min(hue_diff, 180 - hue_diff)
            if hue_diff < 20:
                correct += 1
        
        return correct / len(ef_shapes) if ef_shapes else 0.0
    
    def _evaluate_shape_count(self, final_shapes: List[Dict]) -> float:
        """Check if final frame has correct number of shapes (6)."""
        if len(final_shapes) == 6:
            return 1.0
        elif len(final_shapes) >= 5:
            return 0.7
        elif len(final_shapes) >= 4:
            return 0.4
        else:
            return 0.2
