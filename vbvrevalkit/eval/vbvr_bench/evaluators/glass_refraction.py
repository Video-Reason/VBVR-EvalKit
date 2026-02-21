"""Evaluator for O-18_glass_refraction_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-18_glass_refraction_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class GlassRefractionEvaluator(BaseEvaluator):
    """
    O-18: Glass Refraction (Snell's Law)
    
    Task: Given incident angle and refractive index, calculate and draw 
    the refracted ray according to Snell's law.
    
    Rule-based evaluation:
    1. Snell's law application (50%) - Correct angle calculation
    2. Refracted ray direction (30%) - Accurate angle
    3. Ray completeness (15%) - Extends to edge
    4. Scene fidelity (5%) - Original elements preserved
    """
    
    TASK_WEIGHTS = {
        'snells_law': 0.50,
        'ray_direction': 0.30,
        'ray_completeness': 0.15,
        'scene_fidelity': 0.05
    }
    
    def _detect_lines(self, frame: np.ndarray) -> List[Dict]:
        """Detect lines in the frame."""
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
    
    def _detect_red_line(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red line (refracted ray)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        lines = cv2.HoughLinesP(red_mask, 1, np.pi / 180, threshold=30,
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
        """Evaluate refraction ray drawing accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Detect red lines (refracted rays)
        gen_ray = self._detect_red_line(gen_final)
        gt_ray = self._detect_red_line(gt_final)
        
        # 1. Snell's law: Compare ray angles
        if gen_ray is not None and gt_ray is not None:
            angle_diff = abs(gen_ray['angle'] - gt_ray['angle'])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            scores['snells_law'] = max(0, 1.0 - angle_diff / 30.0)
        else:
            scores['snells_law'] = 0.5 if gen_ray is None and gt_ray is None else 0.0
        
        # 2. Ray direction: Same as Snell's law with looser tolerance
        if gen_ray is not None and gt_ray is not None:
            angle_diff = abs(gen_ray['angle'] - gt_ray['angle'])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            scores['ray_direction'] = max(0, 1.0 - angle_diff / 45.0)
        else:
            scores['ray_direction'] = 0.2  # Detection failed
        
        # 3. Ray completeness: Check line length
        if gen_ray is not None and gt_ray is not None:
            length_ratio = min(gen_ray['length'], gt_ray['length']) / max(gen_ray['length'], gt_ray['length'], 1)
            scores['ray_completeness'] = length_ratio
        elif gen_ray is not None:
            scores['ray_completeness'] = min(1.0, gen_ray['length'] / 100.0)
        else:
            scores['ray_completeness'] = 0.0
        
        # 4. Scene fidelity: Compare non-ray elements
        if len(video_frames) >= 1:
            first_frame = video_frames[0]
            gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) if len(first_frame.shape) == 3 else first_frame
            gray_final = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY) if len(gen_final.shape) == 3 else gen_final
            
            if gray_first.shape == gray_final.shape:
                diff = np.abs(gray_first.astype(float) - gray_final.astype(float)).mean()
                # Small diff means scene preserved (only ray added)
                scores['scene_fidelity'] = max(0, 1.0 - diff / 50.0)
            else:
                scores['scene_fidelity'] = 0.2  # Detection failed
        else:
            scores['scene_fidelity'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
