"""Evaluator for G-9_identify_objects_in_region_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-9_identify_objects_in_region_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class IdentifyObjectsInRegionEvaluator(BaseEvaluator):
    """
    G-9: Identify objects in region evaluator.
    
    Rule-based evaluation:
    - Region identification (30%): Correct target region identified
    - Shape identification (30%): Correct target shape type identified
    - Marking completeness (25%): All target objects marked, no extras
    - Border quality (15%): Green border complete and proper
    """
    
    TASK_WEIGHTS = {
        'region': 0.30,
        'shape': 0.30,
        'completeness': 0.25,
        'border_quality': 0.15
    }
    
    def _detect_green_borders(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect green border markings and return their centers."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
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
        """Evaluate identify objects in region task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        last_frame = video_frames[-1]
        
        # Detect green borders in generated and GT frames
        gen_borders = self._detect_green_borders(last_frame)
        gt_borders = self._detect_green_borders(gt_final_frame)
        
        # 1. Region identification: Check if borders are in correct regions
        # Compare border positions with GT
        if gen_borders and gt_borders:
            matched = 0
            for gb in gen_borders:
                for gtb in gt_borders:
                    dist = np.sqrt((gb[0] - gtb[0])**2 + (gb[1] - gtb[1])**2)
                    if dist < 50:
                        matched += 1
                        break
            scores['region'] = matched / max(len(gt_borders), 1)
        else:
            scores['region'] = 0.5 if not gt_borders else 0.0
        
        # 2. Shape identification: Compare number of marked objects
        if gt_borders:
            count_diff = abs(len(gen_borders) - len(gt_borders))
            scores['shape'] = max(0, 1.0 - count_diff * 0.3)
        else:
            scores['shape'] = 0.2  # Detection failed
        
        # 3. Completeness: Check precision and recall
        if gen_borders and gt_borders:
            # Precision: how many generated borders match GT
            precision_matches = 0
            for gb in gen_borders:
                for gtb in gt_borders:
                    dist = np.sqrt((gb[0] - gtb[0])**2 + (gb[1] - gtb[1])**2)
                    if dist < 50:
                        precision_matches += 1
                        break
            precision = precision_matches / len(gen_borders) if gen_borders else 0
            
            # Recall: how many GT borders are matched
            recall_matches = 0
            for gtb in gt_borders:
                for gb in gen_borders:
                    dist = np.sqrt((gb[0] - gtb[0])**2 + (gb[1] - gtb[1])**2)
                    if dist < 50:
                        recall_matches += 1
                        break
            recall = recall_matches / len(gt_borders) if gt_borders else 0
            
            scores['completeness'] = 0.5 * precision + 0.5 * recall
        else:
            scores['completeness'] = 0.5 if not gt_borders else 0.0
        
        # 4. Border quality: Check green pixel coverage
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_final_frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        green_mask_gen = cv2.inRange(hsv_gen, lower_green, upper_green)
        green_mask_gt = cv2.inRange(hsv_gt, lower_green, upper_green)
        
        green_overlap = np.sum((green_mask_gen > 0) & (green_mask_gt > 0))
        green_union = np.sum((green_mask_gen > 0) | (green_mask_gt > 0))
        
        green_iou = green_overlap / green_union if green_union > 0 else 0.5
        scores['border_quality'] = green_iou
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
