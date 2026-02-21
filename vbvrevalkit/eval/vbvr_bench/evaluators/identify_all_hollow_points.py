"""Evaluator for G-158_identify_all_hollow_points_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-158_identify_all_hollow_points_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class IdentifyAllHollowPointsEvaluator(BaseEvaluator):
    """
    G-158: Identify all hollow points evaluator.
    
    Rule-based evaluation:
    - Hollow point identification accuracy (40%): Distinguish hollow from solid
    - Marking completeness (30%): All hollow points marked
    - Marking position accuracy (20%): Circles centered on hollow points
    - Visual annotation quality (10%): Red circles proper
    """
    
    TASK_WEIGHTS = {
        'identification': 0.40,
        'completeness': 0.30,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_hollow_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect hollow (unfilled) circular points."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find circles using Hough transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30, minRadius=5, maxRadius=50)
        
        hollow_points = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, r = i[0], i[1], i[2]
                
                # Check if hollow (center is similar to background)
                # Sample center and edge
                if cy >= r and cy + r < frame.shape[0] and cx >= r and cx + r < frame.shape[1]:
                    center_val = gray[cy, cx]
                    edge_vals = []
                    for angle in range(0, 360, 45):
                        ex = int(cx + r * np.cos(np.radians(angle)))
                        ey = int(cy + r * np.sin(np.radians(angle)))
                        if 0 <= ex < frame.shape[1] and 0 <= ey < frame.shape[0]:
                            edge_vals.append(gray[ey, ex])
                    
                    if edge_vals:
                        edge_avg = np.mean(edge_vals)
                        # Hollow if center is brighter than edge (outline only)
                        if center_val > edge_avg + 30:
                            hollow_points.append((cx, cy))
        
        return hollow_points
    
    def _detect_red_markings(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect red circle markings."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
        
        return centers
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate identify all hollow points task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect red markings
        gen_markings = self._detect_red_markings(last_frame)
        gt_markings = self._detect_red_markings(gt_last)
        
        # 1. Identification: Compare number of markings
        if gt_markings:
            count_diff = abs(len(gen_markings) - len(gt_markings))
            scores['identification'] = max(0, 1.0 - count_diff * 0.2)
        else:
            scores['identification'] = 0.2  # Detection failed
        
        # 2. Completeness: Recall - how many GT markings are matched
        if gen_markings and gt_markings:
            matched = 0
            for gtm in gt_markings:
                for gm in gen_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    if dist < 40:
                        matched += 1
                        break
            scores['completeness'] = matched / len(gt_markings)
        else:
            scores['completeness'] = 0.5 if not gt_markings else 0.0
        
        # 3. Position accuracy: Average distance between matched markings
        if gen_markings and gt_markings:
            total_dist = 0
            matched_count = 0
            for gm in gen_markings:
                min_dist = float('inf')
                for gtm in gt_markings:
                    dist = np.sqrt((gm[0] - gtm[0])**2 + (gm[1] - gtm[1])**2)
                    min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    total_dist += min_dist
                    matched_count += 1
            
            if matched_count > 0:
                avg_dist = total_dist / matched_count
                scores['position'] = max(0, 1.0 - avg_dist / 50.0)
            else:
                scores['position'] = 0.2  # Detection failed
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Red pixel IoU
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
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
