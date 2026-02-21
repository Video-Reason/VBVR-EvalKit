"""Evaluator for O-9_shape_scaling_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-9_shape_scaling_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeScalingAnalogyEvaluator(BaseEvaluator):
    """
    O-9: Shape scaling analogy evaluator.
    
    Rule-based evaluation:
    - Element preservation (40%): A, B, C (left-up, right-up, left-bottom) remain UNCHANGED
    - Scaling ratio correctness (35%): D size follows A→B trend (larger/smaller)
    - Shape type matching (20%): D has same shape type as C
    - Position correctness (5%): D is in bottom-right quadrant
    """
    
    TASK_WEIGHTS = {
        'element_preservation': 0.40,
        'scaling_ratio': 0.35,
        'shape_type_matching': 0.20,
        'position_correctness': 0.05
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
        
        # Get shapes in each quadrant
        first_shapes = self._get_quadrant_shapes_detailed(first_frame)
        final_shapes = self._get_quadrant_shapes_detailed(final_frame)
        
        # 1. CRITICAL: Check if A, B, C are preserved (unchanged)
        scores['element_preservation'] = self._evaluate_element_preservation(
            first_shapes, final_shapes
        )
        
        # If elements are not preserved, heavily penalize
        if scores['element_preservation'] < 0.5:
            self._last_task_details = {
                'element_preservation': scores['element_preservation'],
                'scaling_ratio': 0.0,
                'shape_type_matching': 0.0,
                'position_correctness': 0.0,
                'elements_changed': True
            }
            return scores['element_preservation'] * self.TASK_WEIGHTS['element_preservation']
        
        # 2. Check scaling ratio
        scores['scaling_ratio'] = self._evaluate_scaling_ratio(first_shapes, final_shapes)
        
        # 3. Check shape type matching
        scores['shape_type_matching'] = self._evaluate_shape_type(first_shapes, final_shapes)
        
        # 4. Check position
        scores['position_correctness'] = self._evaluate_position(final_shapes)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _get_quadrant_shapes_detailed(self, frame: np.ndarray) -> Dict:
        """Get shapes in each quadrant with detailed info."""
        h, w = frame.shape[:2]
        
        quadrants = {
            'A': frame[:h//2, :w//2],      # Top-left
            'B': frame[:h//2, w//2:],      # Top-right
            'C': frame[h//2:, :w//2],      # Bottom-left
            'D': frame[h//2:, w//2:]       # Bottom-right
        }
        
        result = {}
        for name, region in quadrants.items():
            shapes = self._detect_shapes_detailed(region)
            if shapes:
                # Take the largest shape
                largest = max(shapes, key=lambda s: s['area'])
                result[name] = largest
        
        return result
    
    def _detect_shapes_detailed(self, region: np.ndarray) -> List[Dict]:
        """Detect shapes with detailed info (area, vertices, color)."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Detect colored shapes
        mask = hsv[:, :, 1] > 30
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Get shape type by vertices
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    vertices = len(approx)
                    # Get color
                    mask_cnt = np.zeros(region.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_color = cv2.mean(region, mask=mask_cnt)[:3]
                    color_arr = np.array(mean_color, dtype=np.uint8).reshape(1, 1, 3)
                    hsv_c = cv2.cvtColor(color_arr, cv2.COLOR_BGR2HSV)[0, 0]
                    
                    shapes.append({
                        'center': (cx, cy),
                        'area': area,
                        'vertices': vertices,
                        'hue': int(hsv_c[0]),
                        'color': mean_color
                    })
        
        return shapes
    
    def _evaluate_element_preservation(self, first_shapes: Dict, final_shapes: Dict) -> float:
        """Check if A, B, C remain unchanged."""
        preserved = 0
        total = 0
        
        for quadrant in ['A', 'B', 'C']:
            if quadrant not in first_shapes:
                continue
            total += 1
            
            if quadrant not in final_shapes:
                continue  # Shape disappeared - bad
            
            first_s = first_shapes[quadrant]
            final_s = final_shapes[quadrant]
            
            # Check area similarity
            area_ratio = final_s['area'] / first_s['area'] if first_s['area'] > 0 else 0
            if 0.7 < area_ratio < 1.3:
                # Check color similarity
                hue_diff = abs(first_s['hue'] - final_s['hue'])
                hue_diff = min(hue_diff, 180 - hue_diff)
                if hue_diff < 20:
                    preserved += 1
        
        return preserved / total if total > 0 else 0.0
    
    def _evaluate_scaling_ratio(self, first_shapes: Dict, final_shapes: Dict) -> float:
        """Check if D follows the A→B scaling trend."""
        # Get A, B sizes from first frame (or final for A, B since they should be unchanged)
        a_size = first_shapes.get('A', {}).get('area', 0)
        b_size = first_shapes.get('B', {}).get('area', 0)
        c_size = final_shapes.get('C', {}).get('area', 0)
        d_size = final_shapes.get('D', {}).get('area', 0)
        
        if a_size == 0 or c_size == 0:
            return 0.5
        
        if d_size == 0:
            return 0.0  # No D shape generated
        
        # Determine trend: is B larger or smaller than A?
        ab_ratio = b_size / a_size if a_size > 0 else 1
        
        # D should follow the same trend relative to C
        cd_ratio = d_size / c_size if c_size > 0 else 1
        
        # Check if trend direction matches
        ab_trend = "larger" if ab_ratio > 1.1 else ("smaller" if ab_ratio < 0.9 else "same")
        cd_trend = "larger" if cd_ratio > 1.1 else ("smaller" if cd_ratio < 0.9 else "same")
        
        if ab_trend != cd_trend:
            return 0.2  # Wrong trend direction
        
        # Check if ratio is similar
        ratio_diff = abs(ab_ratio - cd_ratio) / max(ab_ratio, cd_ratio, 1)
        
        if ratio_diff < 0.15:
            return 1.0
        elif ratio_diff < 0.3:
            return 0.7
        elif ratio_diff < 0.5:
            return 0.5
        else:
            return 0.3
    
    def _evaluate_shape_type(self, first_shapes: Dict, final_shapes: Dict) -> float:
        """Check if D has the same shape type as C."""
        c_shape = first_shapes.get('C', {})
        d_shape = final_shapes.get('D', {})
        
        if not c_shape or not d_shape:
            return 0.5
        
        c_vertices = c_shape.get('vertices', 0)
        d_vertices = d_shape.get('vertices', 0)
        
        # Same number of vertices = same shape type
        if c_vertices == d_vertices:
            return 1.0
        elif abs(c_vertices - d_vertices) <= 1:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_position(self, final_shapes: Dict) -> float:
        """Check if D exists in bottom-right quadrant."""
        if 'D' in final_shapes and final_shapes['D'].get('area', 0) > 0:
            return 1.0
        return 0.0
