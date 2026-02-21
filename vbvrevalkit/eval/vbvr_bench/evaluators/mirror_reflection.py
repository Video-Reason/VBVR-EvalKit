"""Evaluator for O-19_mirror_reflection_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-19_mirror_reflection_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class MirrorReflectionEvaluator(BaseEvaluator):
    """
    O-19: Mirror Reflection
    
    Task: Given incident ray angle and mirror reflectivity, draw the 
    reflected ray according to reflection law (angle in = angle out).
    
    Rule-based evaluation:
    1. Reflection angle accuracy (40%) - Incident angle = reflected angle
    2. Symmetry correctness (30%) - Ray symmetric about normal
    3. Ray extension (20%) - Extends to image boundary
    4. Starting point accuracy (10%) - Starts from incident point
    """
    
    TASK_WEIGHTS = {
        'reflection_angle': 0.40,
        'symmetry': 0.30,
        'ray_extension': 0.20,
        'starting_point': 0.10
    }
    
    def _detect_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect all lines in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                detected.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'angle': angle,
                    'length': length
                })
        
        return detected
    
    def _detect_colored_line(self, frame: np.ndarray, color: str) -> Optional[Dict]:
        """Detect line of specific color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        color_ranges = {
            'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            'blue': [([100, 100, 100], [130, 255, 255])],
            'green': [([35, 100, 100], [85, 255, 255])],
        }
        
        if color not in color_ranges:
            return None
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges[color]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=30,
                                minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            longest = max(lines, key=lambda l: np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2))
            x1, y1, x2, y2 = longest[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return {
                'start': (x1, y1),
                'end': (x2, y2),
                'angle': angle,
                'length': length
            }
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate mirror reflection ray drawing accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Detect lines
        gen_lines = self._detect_lines(gen_final)
        gt_lines = self._detect_lines(gt_final)
        
        # Try to detect reflected ray (often red or blue)
        gen_reflected = self._detect_colored_line(gen_final, 'red')
        if gen_reflected is None:
            gen_reflected = self._detect_colored_line(gen_final, 'blue')
        
        gt_reflected = self._detect_colored_line(gt_final, 'red')
        if gt_reflected is None:
            gt_reflected = self._detect_colored_line(gt_final, 'blue')
        
        # 1. Reflection angle: Compare ray angles
        if gen_reflected is not None and gt_reflected is not None:
            angle_diff = abs(gen_reflected['angle'] - gt_reflected['angle'])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            scores['reflection_angle'] = max(0, 1.0 - angle_diff / 30.0)
        else:
            scores['reflection_angle'] = 0.2  # Detection failed
        
        # 2. Symmetry: Check if angles are symmetric about normal
        if gen_lines and len(gen_lines) >= 2:
            # Find incident and reflected rays
            angles = [l['angle'] for l in gen_lines if l['length'] > 50]
            if len(angles) >= 2:
                # Check for angle symmetry
                angles_sorted = sorted(angles)
                angle_spread = angles_sorted[-1] - angles_sorted[0]
                if 60 < angle_spread < 120:
                    scores['symmetry'] = 0.8
                else:
                    scores['symmetry'] = 0.2  # Detection failed
            else:
                scores['symmetry'] = 0.2  # Detection failed
        else:
            scores['symmetry'] = 0.2  # Detection failed
        
        # 3. Ray extension: Check line length
        if gen_reflected is not None and gt_reflected is not None:
            length_ratio = min(gen_reflected['length'], gt_reflected['length']) / max(gen_reflected['length'], gt_reflected['length'], 1)
            scores['ray_extension'] = length_ratio
        elif gen_reflected is not None:
            scores['ray_extension'] = min(1.0, gen_reflected['length'] / 100.0)
        else:
            scores['ray_extension'] = 0.2  # Detection failed
        
        # 4. Starting point: Compare overall structure
        if gen_final.shape == gt_final.shape:
            gray_gen = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY) if len(gen_final.shape) == 3 else gen_final
            gray_gt = cv2.cvtColor(gt_final, cv2.COLOR_BGR2GRAY) if len(gt_final.shape) == 3 else gt_final
            
            gen_edges = cv2.Canny(gray_gen, 50, 150)
            gt_edges = cv2.Canny(gray_gt, 50, 150)
            
            edge_overlap = np.sum(gen_edges & gt_edges)
            edge_union = np.sum(gen_edges | gt_edges)
            
            scores['starting_point'] = edge_overlap / max(edge_union, 1)
        else:
            scores['starting_point'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


# Export mapping for this batch
OPEN60_EVALUATORS_PART4 = {
    'G-250_color_triple_intersection_red_data-generator': ColorTripleIntersectionEvaluator,
    'O-10_shape_outline_fill_data-generator': ShapeOutlineFillEvaluator,
    'O-12_shape_color_then_scale_data-generator': ShapeColorThenScaleEvaluator,
    'O-13_shape_outline_then_move_data-generator': ShapeOutlineThenMoveEvaluator,
    'O-14_shape_scale_then_outline_data-generator': ShapeScaleThenOutlineEvaluator,
    'O-15_ball_bounces_given_time_data-generator': BallBounceEvaluator,
    'O-16_color_addition_data-generator': ColorAdditionEvaluator,
    'O-21_construction_blueprint_data-generator': ConstructionBlueprintEvaluator,
    'O-18_glass_refraction_data-generator': GlassRefractionEvaluator,
    'O-19_mirror_reflection_data-generator': MirrorReflectionEvaluator,
