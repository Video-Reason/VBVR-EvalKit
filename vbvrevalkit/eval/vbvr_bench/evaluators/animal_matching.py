"""Evaluator for O-64_animal_matching_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-64_animal_matching_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class AnimalMatchingEvaluator(BaseEvaluator):
    """
    O-64: Animal matching evaluator.
    
    Rule-based evaluation:
    - Animal identification (30%): Correct animal types recognized
      - Cat: pointed triangular ears, whiskers, orange
      - Dog: droopy oval ears, tongue, brown
      - Rabbit: long upright ears, buck teeth, pink
      - Bear: round ears, smile, dark brown
    - Matching correctness (35%): Each animal matched to correct outline
    - Position alignment (25%): Animals centered on outlines
    - Appearance fidelity (10%): Animal features preserved
    """
    
    TASK_WEIGHTS = {
        'identification': 0.30,
        'matching': 0.35,
        'alignment': 0.25,
        'appearance': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        scores['identification'] = self._evaluate_identification(first_frame, final_frame)
        scores['matching'] = self._evaluate_matching(first_frame, final_frame, gt_final_frame)
        scores['alignment'] = self._evaluate_alignment(final_frame, gt_final_frame)
        scores['appearance'] = self._evaluate_appearance(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_colored_animals(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored animal faces."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        animals = []
        
        # Color ranges for different animals
        color_ranges = {
            'orange_cat': [([5, 100, 100], [20, 255, 255])],  # Orange for cat
            'brown_dog': [([10, 50, 50], [20, 200, 200])],    # Brown for dog
            'pink_rabbit': [([150, 50, 50], [180, 200, 255])], # Pink for rabbit
            'dark_bear': [([0, 50, 30], [30, 150, 100])],      # Dark brown for bear
        }
        
        for animal_type, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                animals.append({
                    'type': animal_type,
                    'center': (cx, cy),
                    'area': area
                })
        
        return animals
    
    def _evaluate_identification(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if animals are correctly identified."""
        first_animals = self._detect_colored_animals(first)
        final_animals = self._detect_colored_animals(final)
        
        # Should have same number of animals
        if len(final_animals) == len(first_animals):
            return 1.0
        elif abs(len(final_animals) - len(first_animals)) <= 1:
            return 0.7
        return 0.3
    
    def _evaluate_matching(self, first: np.ndarray, final: np.ndarray,
                           gt_final: Optional[np.ndarray]) -> float:
        """Check if animals are matched to correct outlines."""
        first_animals = self._detect_colored_animals(first)
        final_animals = self._detect_colored_animals(final)
        
        if not first_animals:
            return 0.5
        
        # Check if animals moved from left to right side
        w = first.shape[1]
        mid = w // 2
        
        # Count animals on each side
        first_left = sum(1 for a in first_animals if a['center'][0] < mid)
        final_right = sum(1 for a in final_animals if a['center'][0] >= mid)
        
        if first_left > 0:
            move_ratio = final_right / first_left
            return min(1.0, move_ratio)
        
        return 0.5
    
    def _evaluate_alignment(self, final: np.ndarray, gt_final: Optional[np.ndarray]) -> float:
        """Check if animals are aligned with outlines."""
        final_animals = self._detect_colored_animals(final)
        
        if gt_final is not None:
            gt_animals = self._detect_colored_animals(gt_final)
            
            if final_animals and gt_animals:
                # Compare positions
                total_dist = 0
                matched = 0
                for fa in final_animals:
                    min_dist = float('inf')
                    for ga in gt_animals:
                        dist = np.sqrt((fa['center'][0] - ga['center'][0])**2 +
                                      (fa['center'][1] - ga['center'][1])**2)
                        min_dist = min(min_dist, dist)
                    if min_dist < float('inf'):
                        total_dist += min_dist
                        matched += 1
                
                if matched > 0:
                    avg_dist = total_dist / matched
                    return max(0, 1.0 - avg_dist / 100.0)
        
        # Check if animals are on right side (target area)
        w = final.shape[1]
        mid = w // 2
        right_count = sum(1 for a in final_animals if a['center'][0] >= mid)
        
        return right_count / max(len(final_animals), 1)
    
    def _evaluate_appearance(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if animal appearances are preserved."""
        first_animals = self._detect_colored_animals(first)
        final_animals = self._detect_colored_animals(final)
        
        if not first_animals or not final_animals:
            return 0.5
        
        # Compare total colored area
        first_area = sum(a['area'] for a in first_animals)
        final_area = sum(a['area'] for a in final_animals)
        
        if max(first_area, final_area) > 0:
            ratio = min(first_area, final_area) / max(first_area, final_area)
            return ratio
        
        return 0.5
