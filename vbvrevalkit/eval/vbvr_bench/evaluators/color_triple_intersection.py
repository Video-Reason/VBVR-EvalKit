"""Evaluator for G-250_color_triple_intersection_red_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-250_color_triple_intersection_red_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ColorTripleIntersectionEvaluator(BaseEvaluator):
    """
    G-250: Color Triple Intersection Red
    
    Task: In a Venn diagram with 3 circles, identify and fill the triple 
    intersection region (where all 3 circles overlap) with red color.
    
    Rule-based evaluation:
    1. Triple intersection identification (40%) - Correct region identified
    2. Fill coverage (30%) - >=95% of triple intersection filled
    3. Fill precision (20%) - No overflow to other regions (>=95%)
    4. Visual quality (10%) - Pure red color, uniform fill
    """
    
    TASK_WEIGHTS = {
        'triple_intersection_identification': 0.40,
        'fill_coverage': 0.30,
        'fill_precision': 0.20,
        'visual_quality': 0.10
    }
    
    def _detect_red_region(self, frame: np.ndarray) -> np.ndarray:
        """Detect red-filled regions in frame."""
        if len(frame.shape) == 2:
            return np.zeros_like(frame, dtype=bool)

        # Red detection in HSV (frames are BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        return mask > 0
    
    def _check_red_purity(self, frame: np.ndarray, red_mask: np.ndarray) -> float:
        """Check if red color is pure (close to RGB 255,0,0)."""
        if red_mask.sum() < 10:
            return 0.0
        
        red_pixels = frame[red_mask]
        if len(red_pixels) == 0:
            return 0.0
        
        # BGR format - check blue channel is high (red in BGR)
        mean_b = np.mean(red_pixels[:, 0])
        mean_g = np.mean(red_pixels[:, 1])
        mean_r = np.mean(red_pixels[:, 2])
        
        # Check closeness to pure red (0, 0, 255) in BGR
        b_score = max(0, 1 - mean_b / 100)
        g_score = max(0, 1 - mean_g / 100)
        r_score = max(0, 1 - abs(mean_r - 255) / 100)
        
        return (r_score + g_score + b_score) / 3
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the Venn diagram."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                    param1=50, param2=30, minRadius=50, maxRadius=300)
        
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected.append({
                    'center': (i[0], i[1]),
                    'radius': i[2]
                })
        
        return detected
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate triple intersection filling accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        scores = {}
        
        # Detect red pixels in generated and GT
        red_mask_gen = self._detect_red_region(gen_final)
        red_mask_gt = self._detect_red_region(gt_final)
        
        # Calculate intersection and union for precision/recall
        intersection = np.logical_and(red_mask_gen, red_mask_gt).sum()
        gt_area = red_mask_gt.sum()
        gen_area = red_mask_gen.sum()
        
        # Fill coverage (recall) - how much of GT region is covered
        coverage = intersection / max(gt_area, 1)
        
        # Fill precision - how much of generated is correct
        precision = intersection / max(gen_area, 1)
        
        # Check if any red region was identified
        identification_score = 0.0
        if gen_area > 100:
            identification_score = 0.5
            if coverage > 0.5:
                identification_score = min(1.0, coverage + 0.3)
        
        # Visual quality - check red color purity
        visual_quality = self._check_red_purity(gen_final, red_mask_gen)
        
        scores['triple_intersection_identification'] = identification_score
        scores['fill_coverage'] = coverage
        scores['fill_precision'] = precision
        scores['visual_quality'] = visual_quality
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
