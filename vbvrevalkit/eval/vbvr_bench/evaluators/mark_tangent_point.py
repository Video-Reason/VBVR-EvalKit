"""Evaluator for G-222_mark_tangent_point_of_circles_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-222_mark_tangent_point_of_circles_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class MarkTangentPointEvaluator(BaseEvaluator):
    """
    G-222: Mark tangent point of circles evaluator.
    
    Rule-based evaluation:
    - External tangent circle pair identification (40%): Correct pair found
    - Tangent point calculation accuracy (40%): Precise point location
    - Marking position accuracy (15%): Mark centered on tangent point
    - Visual annotation quality (5%): Black circle proper
    """
    
    TASK_WEIGHTS = {
        'pair_id': 0.40,
        'calculation': 0.40,
        'position': 0.15,
        'annotation': 0.05
    }
    
    def _detect_circles(self, frame: np.ndarray) -> List[Dict]:
        """Detect circles in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                                    param1=50, param2=30, minRadius=20, maxRadius=200)
        
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                detected.append({
                    'center': (i[0], i[1]),
                    'radius': i[2]
                })
        
        return detected
    
    def _find_tangent_pairs(self, circles: List[Dict]) -> List[Tuple[int, int, Tuple[float, float]]]:
        """Find externally tangent circle pairs and their tangent points."""
        tangent_pairs = []
        
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                c1, c2 = circles[i], circles[j]
                
                # Convert to float to avoid overflow
                c1x, c1y = float(c1['center'][0]), float(c1['center'][1])
                c2x, c2y = float(c2['center'][0]), float(c2['center'][1])
                r1, r2 = float(c1['radius']), float(c2['radius'])
                
                dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                
                # Check if externally tangent (distance ≈ r1 + r2)
                expected_dist = r1 + r2
                if abs(dist - expected_dist) < 10:  # Tolerance
                    # Calculate tangent point
                    t = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                    tx = c1x + t * (c2x - c1x)
                    ty = c1y + t * (c2y - c1y)
                    
                    tangent_pairs.append((i, j, (tx, ty)))
        
        return tangent_pairs
    
    def _detect_black_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect black circle marking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Black marking
        black_mask = (gray < 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30 or area > 1000:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:
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
        """Evaluate mark tangent point task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect markings
        gen_marking = self._detect_black_marking(last_frame)
        gt_marking = self._detect_black_marking(gt_last)
        
        # Detect circles and find tangent pairs
        gen_circles = self._detect_circles(last_frame)
        tangent_pairs = self._find_tangent_pairs(gen_circles)
        
        # 1. Pair identification: Check if marking is near a tangent point
        if gen_marking is not None and tangent_pairs:
            near_tangent = False
            for _, _, tangent_point in tangent_pairs:
                dist = np.sqrt((gen_marking[0] - tangent_point[0])**2 + 
                              (gen_marking[1] - tangent_point[1])**2)
                if dist < 30:
                    near_tangent = True
                    break
            scores['pair_id'] = 1.0 if near_tangent else 0.3
        else:
            scores['pair_id'] = 0.2  # Detection failed
        
        # 2. Calculation accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['calculation'] = max(0, 1.0 - dist / 30.0)
        else:
            scores['calculation'] = 0.2  # Detection failed
        
        # 3. Position accuracy: Same with looser tolerance
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Black pixel IoU
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        
        black_mask_gen = (gray_gen < 50).astype(np.uint8)
        black_mask_gt = (gray_gt < 50).astype(np.uint8)
        
        black_overlap = np.sum((black_mask_gen > 0) & (black_mask_gt > 0))
        black_union = np.sum((black_mask_gen > 0) | (black_mask_gt > 0))
        
        scores['annotation'] = black_overlap / black_union if black_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
