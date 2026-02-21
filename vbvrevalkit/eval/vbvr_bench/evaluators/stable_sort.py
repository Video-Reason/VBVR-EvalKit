"""Evaluator for G-3_stable_sort_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-3_stable_sort_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class StableSortEvaluator(BaseEvaluator):
    """
    G-3: Stable sort objects evaluator.
    
    Rule-based evaluation:
    - Classification correctness (30%): Shapes correctly grouped by type (same type adjacent)
    - Order correctness (30%): Each group sorted small to large (left to right)
    - Object fidelity (30%): Shape types, sizes, colors preserved from initial frame
    - Layout accuracy (10%): Horizontal alignment (same y-coordinate)
    """
    
    TASK_WEIGHTS = {
        'classification': 0.30,
        'order': 0.30,
        'fidelity': 0.30,
        'layout': 0.10
    }
    
    def _detect_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored shapes and return their properties."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find non-white/non-black areas (colored shapes)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip noise
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Determine shape type by vertex count
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            vertices = len(approx)
            
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                shape_type = 'square'
            else:
                shape_type = 'circle'
            
            # Get dominant color at centroid region
            color = frame[max(0, cy-5):cy+5, max(0, cx-5):cx+5].mean(axis=(0, 1))
            
            shapes.append({
                'type': shape_type,
                'center': (cx, cy),
                'area': area,
                'color': tuple(color.astype(int).tolist()),
            })
        
        return shapes
    
    def _color_distance(self, c1: Tuple, c2: Tuple) -> float:
        """Calculate color distance."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    def _group_by_color(self, shapes: List[Dict], threshold: float = 50) -> Dict[str, List[Dict]]:
        """Group shapes by similar color."""
        if not shapes:
            return {}
        
        groups = {}
        for shape in shapes:
            color = shape['color']
            matched = False
            for group_color, group_shapes in groups.items():
                if self._color_distance(color, eval(group_color)) < threshold:
                    group_shapes.append(shape)
                    matched = True
                    break
            if not matched:
                groups[str(color)] = [shape]
        
        return groups
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate stable sort task with rule-based logic."""
        if len(video_frames) < 2:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Detect shapes in initial and final frames
        initial_shapes = self._detect_shapes(first_frame)
        final_shapes = self._detect_shapes(last_frame)
        gt_final_shapes = self._detect_shapes(gt_final_frame) if gt_final_frame is not None else []
        
        scores = {}
        
        # 1. Classification (30%): Check if shapes are grouped by type/color
        final_groups = self._group_by_color(final_shapes)
        gt_groups = self._group_by_color(gt_final_shapes) if gt_final_shapes else final_groups
        
        if len(final_shapes) >= 6 and len(final_groups) >= 2:
            # Check if shapes of same color are adjacent (grouped)
            final_sorted = sorted(final_shapes, key=lambda s: s['center'][0])
            
            # Count color transitions (fewer = better grouping)
            transitions = 0
            for i in range(1, len(final_sorted)):
                if self._color_distance(final_sorted[i]['color'], final_sorted[i-1]['color']) > 50:
                    transitions += 1
            
            # Ideal: 1 transition for 2 groups
            expected_transitions = len(final_groups) - 1
            if transitions <= expected_transitions:
                scores['classification'] = 1.0
            else:
                scores['classification'] = max(0, 1.0 - (transitions - expected_transitions) * 0.3)
        else:
            # Wrong number of shapes
            scores['classification'] = max(0, len(final_shapes) / 6.0) * 0.5
        
        # 2. Order (30%): Check if each group is sorted small to large (left to right)
        order_score = 0.0
        if final_groups:
            group_scores = []
            for group_shapes in final_groups.values():
                if len(group_shapes) >= 2:
                    # Sort by x-position
                    sorted_by_x = sorted(group_shapes, key=lambda s: s['center'][0])
                    # Check if sizes increase left to right
                    sizes = [s['area'] for s in sorted_by_x]
                    
                    # Count correctly ordered pairs
                    correct_pairs = sum(1 for i in range(len(sizes) - 1) if sizes[i] < sizes[i + 1])
                    total_pairs = len(sizes) - 1
                    group_scores.append(correct_pairs / total_pairs if total_pairs > 0 else 1.0)
            
            order_score = np.mean(group_scores) if group_scores else 0.5
        scores['order'] = order_score
        
        # 3. Fidelity (30%): Check if shapes are preserved from initial frame
        fidelity_score = 0.0
        if initial_shapes and final_shapes:
            # Check object count preservation
            count_match = max(0, 1.0 - abs(len(initial_shapes) - len(final_shapes)) / max(len(initial_shapes), 1))
            
            # Check size preservation (total area should be similar)
            initial_total_area = sum(s['area'] for s in initial_shapes)
            final_total_area = sum(s['area'] for s in final_shapes)
            area_ratio = min(initial_total_area, final_total_area) / max(initial_total_area, final_total_area) if max(initial_total_area, final_total_area) > 0 else 0
            
            # Check shape type preservation
            initial_types = sorted([s['type'] for s in initial_shapes])
            final_types = sorted([s['type'] for s in final_shapes])
            type_match = sum(1 for a, b in zip(initial_types, final_types) if a == b) / max(len(initial_types), len(final_types), 1)
            
            fidelity_score = 0.4 * count_match + 0.3 * area_ratio + 0.3 * type_match
        scores['fidelity'] = fidelity_score
        
        # 4. Layout (10%): Check horizontal alignment
        layout_score = 0.0
        if final_shapes:
            y_coords = [s['center'][1] for s in final_shapes]
            y_variance = np.var(y_coords)
            # Good alignment: variance < 100 pixels
            layout_score = max(0, 1.0 - y_variance / 5000.0)
        scores['layout'] = layout_score
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
