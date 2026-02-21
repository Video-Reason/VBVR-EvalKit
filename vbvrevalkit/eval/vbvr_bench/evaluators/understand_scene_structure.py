"""Evaluator for G-43_understand_scene_structure_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-43_understand_scene_structure_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class UnderstandSceneStructureEvaluator(BaseEvaluator):
    """
    G-43: Understand scene structure evaluator.
    
    Evaluates:
    - Room identification correctness (50%): Correct room type identified
    - Marking accuracy (30%): Green box accurately marks target room
    - Visual normality (15%): Border complete and accurate
    - Scene fidelity (5%): Floorplan preserved
    """
    
    TASK_WEIGHTS = {
        'room_id': 0.50,
        'marking': 0.30,
        'visual': 0.15,
        'fidelity': 0.05
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate scene structure task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect green markings
        gen_green = self._detect_green_marking(last_frame)
        gt_green = self._detect_green_marking(gt_last)
        
        # 1. Room identification (50%): Check if correct room is marked
        room_score = self._evaluate_room_identification(gen_green, gt_green, last_frame, gt_last)
        scores['room_id'] = room_score
        
        # 2. Marking accuracy (30%): Green box position
        marking_score = self._evaluate_marking_accuracy(gen_green, gt_green)
        scores['marking'] = marking_score
        
        # 3. Visual normality (15%): Border completeness
        visual_score = self._evaluate_visual_normality(gen_green)
        scores['visual'] = visual_score
        
        # 4. Scene fidelity (5%): Floorplan preserved
        fidelity_score = self._evaluate_scene_fidelity(last_frame, gt_last, gen_green, gt_green)
        scores['fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_green_marking(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect green rectangular marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(largest)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        return {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'area': area,
            'contour': largest
        }
    
    def _evaluate_room_identification(self, gen_green: Optional[Dict], gt_green: Optional[Dict],
                                      gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if correct room is identified."""
        if gen_green is None:
            return 0.0
        if gt_green is None:
            return 0.0  # STRICT: No GT to compare
        
        # Compare marked regions
        gen_center = gen_green['center']
        gt_center = gt_green['center']
        
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        # STRICTER: Score based on distance
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.7
        elif normalized_dist < 0.20:
            return 0.3
        else:
            return 0.0  # STRICT: Wrong room marked
    
    def _evaluate_marking_accuracy(self, gen_green: Optional[Dict], gt_green: Optional[Dict]) -> float:
        """Evaluate green marking accuracy using IoU."""
        if gen_green is None:
            return 0.0
        if gt_green is None:
            return 0.0  # STRICT: No GT to compare
        
        gen_bbox = gen_green['bbox']
        gt_bbox = gt_green['bbox']
        
        # Calculate IoU
        x1 = max(gen_bbox[0], gt_bbox[0])
        y1 = max(gen_bbox[1], gt_bbox[1])
        x2 = min(gen_bbox[0] + gen_bbox[2], gt_bbox[0] + gt_bbox[2])
        y2 = min(gen_bbox[1] + gen_bbox[3], gt_bbox[1] + gt_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        gen_area = gen_bbox[2] * gen_bbox[3]
        gt_area = gt_bbox[2] * gt_bbox[3]
        union = gen_area + gt_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _evaluate_visual_normality(self, gen_green: Optional[Dict]) -> float:
        """Evaluate if marking is a proper rectangle."""
        if gen_green is None:
            return 0.0
        
        contour = gen_green['contour']
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            return 1.0
        elif len(approx) >= 3 and len(approx) <= 6:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_scene_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray,
                                  gen_green: Optional[Dict], gt_green: Optional[Dict]) -> float:
        """Evaluate if floorplan is preserved."""
        # Mask out green regions and compare
        gen_mask = np.ones(gen_frame.shape[:2], dtype=np.uint8) * 255
        gt_mask = np.ones(gt_frame.shape[:2], dtype=np.uint8) * 255
        
        if gen_green:
            cv2.drawContours(gen_mask, [gen_green['contour']], -1, 0, -1)
        if gt_green:
            cv2.drawContours(gt_mask, [gt_green['contour']], -1, 0, -1)
        
        combined_mask = gen_mask & gt_mask
        
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        if np.sum(combined_mask > 0) == 0:
            return 0.5
        
        diff = np.abs(gen_gray.astype(float) - gt_gray.astype(float))
        masked_diff = diff[combined_mask > 0]
        
        avg_diff = np.mean(masked_diff) / 255.0
        return max(0, 1.0 - avg_diff * 2)
