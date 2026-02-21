"""Evaluator for G-206_identify_pentagons_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-206_identify_pentagons_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class IdentifyPentagonsEvaluator(BaseEvaluator):
    """
    G-206: Identify pentagons evaluator.
    
    Rule-based evaluation:
    - Edge count identification (40%): Correct 5-sided polygon identified
    - Marking precision (35%): Red circle accurately marks pentagon
    - Marking quality (15%): Circle complete and proper
    - Scene fidelity (10%): All polygons preserved
    """
    
    TASK_WEIGHTS = {
        'edge_count': 0.40,
        'marking': 0.35,
        'quality': 0.15,
        'fidelity': 0.10
    }
    
    def _detect_polygons(self, frame: np.ndarray) -> List[Dict]:
        """Detect polygons and count their edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            # Approximate polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            polygons.append({
                'center': (cx, cy),
                'vertices': len(approx),
                'area': area,
                'is_pentagon': len(approx) == 5
            })
        
        return polygons
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(red_mask)
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
        """Evaluate identify pentagons task."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect polygons from FIRST frame (before marking, more accurate)
        # The red marking in final frame can interfere with contour detection
        first_polygons = self._detect_polygons(first_frame) if first_frame is not None else []
        gen_polygons = self._detect_polygons(last_frame)
        gt_polygons = self._detect_polygons(gt_last)
        
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Edge count identification: Check if marking is near a pentagon
        # Use first_frame polygons for pentagon detection (more accurate)
        polygons_to_check = first_polygons if first_polygons else gen_polygons
        
        if gen_marking is not None and polygons_to_check:
            marked_pentagon = False
            for poly in polygons_to_check:
                dist = np.sqrt((gen_marking[0] - poly['center'][0])**2 + 
                              (gen_marking[1] - poly['center'][1])**2)
                if dist < 100 and poly['is_pentagon']:
                    marked_pentagon = True
                    break
            
            # If no pentagon found but marking matches GT, give credit
            if not marked_pentagon and gt_marking is not None:
                marking_dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                                      (gen_marking[1] - gt_marking[1])**2)
                if marking_dist < 30:
                    scores['edge_count'] = 0.9
                else:
                    scores['edge_count'] = 0.3
            else:
                scores['edge_count'] = 1.0 if marked_pentagon else 0.3
        else:
            scores['edge_count'] = 0.2  # Detection failed
        
        # 2. Marking precision: Compare with GT marking position
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['marking'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['marking'] = 0.2  # Detection failed
        
        # 3. Quality: Red pixel IoU
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
        
        scores['quality'] = red_overlap / red_union if red_union > 0 else 0.5
        
        # 4. Scene fidelity: Compare polygon counts
        if gen_polygons and gt_polygons:
            count_ratio = min(len(gen_polygons), len(gt_polygons)) / max(len(gen_polygons), len(gt_polygons), 1)
            scores['fidelity'] = count_ratio
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
