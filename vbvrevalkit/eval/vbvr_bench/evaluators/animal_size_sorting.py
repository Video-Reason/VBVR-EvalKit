"""Evaluator for O-65_animal_size_sorting_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-65_animal_size_sorting_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class AnimalSizeSortingEvaluator(BaseEvaluator):
    """
    O-65: Animal size sorting evaluator.
    
    Rule-based evaluation:
    - Sorting correctness (40%): Correct small-to-large order (left to right)
    - Baseline alignment (30%): All animals on baseline
    - Animal fidelity (20%): Size, appearance preserved
    - Completeness (10%): All animals included
    """
    
    TASK_WEIGHTS = {
        'sorting': 0.40,
        'alignment': 0.30,
        'fidelity': 0.20,
        'completeness': 0.10
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
        
        scores['sorting'] = self._evaluate_sorting(final_frame)
        scores['alignment'] = self._evaluate_alignment(final_frame)
        scores['fidelity'] = self._evaluate_fidelity(first_frame, final_frame)
        scores['completeness'] = self._evaluate_completeness(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_animals(self, frame: np.ndarray) -> List[Dict]:
        """Detect animal figures by color and size."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find colored (non-white, non-black) regions
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        animals = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:  # Filter noise
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            x, y, w, h = cv2.boundingRect(cnt)
            size = max(w, h)
            
            animals.append({
                'center': (cx, cy),
                'area': area,
                'size': size,
                'bbox': (x, y, w, h),
                'bottom_y': y + h
            })
        
        # Sort by x position
        animals.sort(key=lambda a: a['center'][0])
        return animals
    
    def _evaluate_sorting(self, final: np.ndarray) -> float:
        """Check if animals are sorted correctly (small to large, left to right)."""
        animals = self._detect_animals(final)
        
        if len(animals) < 2:
            return 0.0  # STRICT: Not enough animals detected
        
        # Check if sizes increase left to right
        sizes = [a['size'] for a in animals]
        
        correct_pairs = 0
        total_pairs = len(sizes) - 1
        
        for i in range(total_pairs):
            if sizes[i] <= sizes[i + 1]:
                correct_pairs += 1
        
        return correct_pairs / total_pairs if total_pairs > 0 else 0.0  # STRICT
    
    def _evaluate_alignment(self, final: np.ndarray) -> float:
        """Check if animals are aligned on baseline."""
        animals = self._detect_animals(final)
        
        if len(animals) < 2:
            return 0.0  # STRICT: Not enough animals
        
        # Check if bottom y-coordinates are similar (aligned on baseline)
        bottom_ys = [a['bottom_y'] for a in animals]
        
        variance = np.var(bottom_ys)
        mean_y = np.mean(bottom_ys)
        
        # Low variance = good alignment
        if mean_y > 0:
            cv = np.sqrt(variance) / mean_y
            return max(0, 1.0 - cv * 5)
        
        return 0.0  # STRICT: Mean Y is invalid
    
    def _evaluate_fidelity(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if animal appearances are preserved."""
        first_animals = self._detect_animals(first)
        final_animals = self._detect_animals(final)
        
        if not first_animals or not final_animals:
            return 0.0  # STRICT: No animals detected
        
        # Compare sizes (should be preserved)
        first_sizes = sorted([a['size'] for a in first_animals])
        final_sizes = sorted([a['size'] for a in final_animals])
        
        if len(first_sizes) != len(final_sizes):
            return 0.0  # STRICT: Different number of animals
        
        # Check size preservation
        size_diffs = [abs(f - l) / max(f, 1) for f, l in zip(first_sizes, final_sizes)]
        return 1.0 - min(1.0, np.mean(size_diffs))
    
    def _evaluate_completeness(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if all animals are included."""
        first_animals = self._detect_animals(first)
        final_animals = self._detect_animals(final)
        
        first_count = len(first_animals)
        final_count = len(final_animals)
        
        if first_count == 0:
            return 0.0  # STRICT: No animals to compare
        
        # STRICT: Must have same count
        if final_count != first_count:
            return max(0, 1.0 - abs(final_count - first_count) / first_count)
        return 1.0
