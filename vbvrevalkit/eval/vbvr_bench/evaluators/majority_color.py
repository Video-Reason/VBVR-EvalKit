"""Evaluator for O-38_majority_color_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-38_majority_color_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class MajorityColorEvaluator(BaseEvaluator):
    """
    O-38: Majority Color Identification
    
    CRITICAL RULES:
    1. All objects size and shape must NOT change
    2. All objects should change to the SAME color (majority color from first frame)
    3. Final frame should have only ONE color (the majority color)
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'shapes_preserved': 0.30,      # Same number of shapes
            'single_color': 0.55,          # Only one color in final - MOST IMPORTANT
            'correct_majority': 0.15       # Correct majority color
        }
    
    def _count_total_shapes(self, frame: np.ndarray) -> int:
        """Count total number of colored shapes."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        sat_mask = (sat > 50).astype(np.uint8) * 255
        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 200])
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate majority color identification.
        
        CRITICAL RULES:
        1. Shape count must be preserved
        2. All shapes should become ONE color (majority)
        """
        
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        scores = {}
        
        # 1. CRITICAL: Shape count must be preserved
        first_shape_count = self._count_total_shapes(first_frame)
        final_shape_count = self._count_total_shapes(gen_final)
        
        if first_shape_count > 0:
            count_change = abs(final_shape_count - first_shape_count) / first_shape_count
            if count_change > 0.5:
                # Shapes changed significantly
                scores['shapes_preserved'] = 0.0
            else:
                scores['shapes_preserved'] = max(0, 1.0 - count_change)
        else:
            scores['shapes_preserved'] = 0.0
        
        # 2. Final frame should have only ONE color
        gen_final_colors = self._count_shapes_by_color(gen_final)
        
        if len(gen_final_colors) == 0:
            scores['single_color'] = 0.0
        elif len(gen_final_colors) == 1:
            scores['single_color'] = 1.0
        else:
            # Multiple colors - penalize
            total_shapes = sum(gen_final_colors.values())
            max_color_count = max(gen_final_colors.values())
            scores['single_color'] = max_color_count / total_shapes * 0.5  # Max 0.5 if not single
        
        # 3. Check if correct majority color
        initial_colors = self._count_shapes_by_color(first_frame)
        gt_final_colors = self._count_shapes_by_color(gt_final)
        
        if initial_colors and gt_final_colors:
            majority_color = max(initial_colors.items(), key=lambda x: x[1])[0]
            
            if majority_color in gen_final_colors:
                scores['correct_majority'] = 1.0
            else:
                scores['correct_majority'] = 0.0
        else:
            scores['correct_majority'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _count_shapes_by_color(self, frame: np.ndarray) -> Dict[str, int]:
        """Count shapes by color."""
        color_counts = {}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'pink': ([140, 50, 100], [170, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Filter noise
                    count += 1
            
            if count > 0:
                color_counts[color_name] = count
        
        return color_counts
    
    def _evaluate_majority_identification(self, initial_colors: Dict[str, int],
                                          gen_colors: Dict[str, int],
                                          gt_colors: Dict[str, int]) -> float:
        """Evaluate if correct majority color was identified."""
        if not gt_colors:
            return 0.0  # STRICT: No GT to compare
        
        # Find majority color in GT (should be only one color remaining)
        gt_majority = max(gt_colors.items(), key=lambda x: x[1], default=('none', 0))
        
        # Find dominant color in generated
        if not gen_colors:
            return 0.0
        
        gen_dominant = max(gen_colors.items(), key=lambda x: x[1], default=('none', 0))
        
        # Check if same color
        if gt_majority[0] == gen_dominant[0]:
            return 1.0
        
        # Check if it's close (could be color detection variance)
        if gen_dominant[1] > 0:
            return 0.3
        
        return 0.0
    
    def _evaluate_non_majority_removal(self, gen_colors: Dict[str, int],
                                        gt_colors: Dict[str, int]) -> float:
        """Evaluate if non-majority colors were removed."""
        if not gt_colors:
            return 0.5
        
        # GT should have only one color (majority)
        gt_color_count = len([c for c, n in gt_colors.items() if n > 0])
        gen_color_count = len([c for c, n in gen_colors.items() if n > 0])
        
        if gt_color_count == 1:
            if gen_color_count == 1:
                return 1.0
            elif gen_color_count == 2:
                return 0.6
            else:
                return max(0.2, 1.0 - gen_color_count * 0.2)
        
        return 0.5
    
    def _evaluate_majority_preservation(self, gen_colors: Dict[str, int],
                                         gt_colors: Dict[str, int]) -> float:
        """Evaluate if all majority color shapes were preserved."""
        if not gt_colors:
            return 0.5
        
        gt_majority = max(gt_colors.items(), key=lambda x: x[1], default=('none', 0))
        
        if gt_majority[0] in gen_colors:
            gen_count = gen_colors[gt_majority[0]]
            gt_count = gt_majority[1]
            
            if gt_count == 0:
                return 0.5
            
            ratio = gen_count / gt_count
            
            if 0.9 <= ratio <= 1.1:
                return 1.0
            elif 0.7 <= ratio <= 1.3:
                return 0.7
            else:
                return max(0.2, ratio if ratio < 1 else 2 - ratio)
        
        return 0.0
    
    def _evaluate_visual_consistency(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate visual cleanliness."""
        # Check if background is mostly white
        gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        white_ratio = np.sum(gray > 240) / gray.size
        
        if white_ratio > 0.7:
            return 1.0
        elif white_ratio > 0.5:
            return 0.7
        else:
            return 0.4
