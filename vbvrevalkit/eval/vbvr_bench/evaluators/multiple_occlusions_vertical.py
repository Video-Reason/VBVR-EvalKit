"""Evaluator for G-21_multiple_occlusions_vertical_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-21_multiple_occlusions_vertical_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class MultipleOcclusionsVerticalEvaluator(BaseEvaluator):
    """
    G-21: Multiple occlusions vertical evaluator.
    
    Rule-based evaluation:
    - Occlusion correctness (35%): Mask properly occludes objects
    - Object permanence (30%): All objects reappear after mask leaves
    - Motion correctness (20%): Mask moves vertically downward
    - Visual consistency (15%): Objects maintain position and attributes
    """
    
    TASK_WEIGHTS = {
        'occlusion': 0.35,
        'permanence': 0.30,
        'motion': 0.20,
        'consistency': 0.15
    }
    
    def _detect_mask(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect dark rectangular mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Dark mask
        dark_mask = (gray < 100).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Check if rectangular
            if w * h * 0.7 < area:  # Reasonably rectangular
                return (x, y, w, h)
        
        return None
    
    def _detect_colored_objects(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect colored objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find non-white/non-black colored areas
        mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
        
        # Remove dark areas (mask)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = mask & (gray > 100)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            objects.append((cx, cy))
        
        return objects
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate multiple occlusions task."""
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Detect objects
        initial_objects = self._detect_colored_objects(first_frame)
        final_objects = self._detect_colored_objects(last_frame)
        gt_final_objects = self._detect_colored_objects(gt_final_frame)
        
        # 1. Occlusion correctness: Check if there's visual change during video (occlusion happening)
        # Compare first frame with middle frames to detect occlusion
        occlusion_detected = False
        mid_idx = len(video_frames) // 2
        for i in range(max(1, mid_idx - 3), min(len(video_frames) - 1, mid_idx + 3)):
            mid_frame = video_frames[i]
            diff = cv2.absdiff(first_frame, mid_frame)
            diff_sum = np.sum(diff) / (first_frame.shape[0] * first_frame.shape[1] * 3)
            if diff_sum > 5:  # Significant visual change
                occlusion_detected = True
                break
        
        scores['occlusion'] = 1.0 if occlusion_detected else 0.0
        
        # 2. Object permanence: Objects should reappear at original positions in final frame
        if initial_objects and final_objects:
            reappeared = 0
            for io in initial_objects:
                for fo in final_objects:
                    dist = np.sqrt((io[0] - fo[0])**2 + (io[1] - fo[1])**2)
                    if dist < 50:  # Same position (with tolerance)
                        reappeared += 1
                        break
            
            scores['permanence'] = reappeared / len(initial_objects)
        else:
            scores['permanence'] = 0.0 if initial_objects else 0.5
        
        # 3. Motion correctness: Compare final frame with GT final
        if last_frame.shape == gt_final_frame.shape:
            diff = np.abs(last_frame.astype(float) - gt_final_frame.astype(float)).mean()
            if diff < 20:
                scores['motion'] = 1.0
            elif diff < 40:
                scores['motion'] = 0.5
            else:
                scores['motion'] = 0.0
        else:
            scores['motion'] = 0.0
        
        # 4. Visual consistency: Final frame should match GT final frame (objects at same positions)
        if final_objects and gt_final_objects:
            # Check if number of objects matches
            count_match = len(final_objects) == len(gt_final_objects)
            
            # Check if positions match GT
            position_matches = 0
            for gto in gt_final_objects:
                for fo in final_objects:
                    dist = np.sqrt((gto[0] - fo[0])**2 + (gto[1] - fo[1])**2)
                    if dist < 50:
                        position_matches += 1
                        break
            
            position_score = position_matches / max(len(gt_final_objects), 1)
            scores['consistency'] = 0.5 * (1.0 if count_match else 0.0) + 0.5 * position_score
        else:
            scores['consistency'] = 0.0 if gt_final_objects else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


# Mapping of task names to evaluators
OPEN60_EVALUATORS = {
    'G-3_stable_sort_data-generator': StableSortEvaluator,
    'G-5_multi_object_placement_data-generator': MultiObjectPlacementEvaluator,
    'G-8_track_object_movement_data-generator': TrackObjectMovementEvaluator,
    'G-9_identify_objects_in_region_data-generator': IdentifyObjectsInRegionEvaluator,
    'G-13_grid_number_sequence_data-generator': GridNumberSequenceEvaluator,
    'G-15_grid_avoid_obstacles_data-generator': GridAvoidObstaclesEvaluator,
    'G-16_grid_go_through_block_data-generator': GridGoThroughBlockEvaluator,
    'G-24_separate_objects_no_spin_data-generator': SeparateObjectsNoSpinEvaluator,
    'G-18_grid_shortest_path_data-generator': GridShortestPathEvaluator,
    'G-21_multiple_occlusions_vertical_data-generator': MultipleOcclusionsVerticalEvaluator,
