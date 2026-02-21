"""Evaluator for G-51_predict_next_color_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-51_predict_next_color_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class PredictNextColorEvaluator(BaseEvaluator):
    """
    G-51: Predict next color evaluator.
    
    Evaluates:
    - Pattern identification accuracy (50%): Correct pattern recognized
    - Answer color accuracy (30%): Correct color predicted
    - Visual presentation quality (15%): Proper block style
    - Task understanding (5%): Only fill position 5
    """
    
    TASK_WEIGHTS = {
        'pattern': 0.50,
        'color': 0.30,
        'visual': 0.15,
        'understanding': 0.05
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate predict next color task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect color blocks
        gen_blocks = self._detect_color_blocks(last_frame)
        gt_blocks = self._detect_color_blocks(gt_last)
        
        # 1. Pattern identification (50%): Check if answer matches GT pattern
        pattern_score = self._evaluate_pattern(gen_blocks, gt_blocks)
        scores['pattern'] = pattern_score
        
        # 2. Color accuracy (30%): Check 5th block color
        color_score = self._evaluate_color_accuracy(gen_blocks, gt_blocks)
        scores['color'] = color_score
        
        # 3. Visual presentation (15%): Block style consistency
        visual_score = self._evaluate_visual_presentation(gen_blocks)
        scores['visual'] = visual_score
        
        # 4. Task understanding (5%): Only 5th position filled
        understanding_score = self._evaluate_task_understanding(gen_blocks, gt_blocks)
        scores['understanding'] = understanding_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_color_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored blocks in the frame."""
        blocks = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Filter noise
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if roughly square (block shape)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 <= aspect_ratio <= 1.4:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        blocks.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area
                        })
        
        # Sort by x position (left to right)
        blocks.sort(key=lambda b: b['center'][0])
        
        return blocks
    
    def _evaluate_pattern(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if the pattern was correctly identified."""
        if len(gen_blocks) < 5 or len(gt_blocks) < 5:
            return 0.0
        
        # Compare the 5th block (answer)
        gen_5th = gen_blocks[4] if len(gen_blocks) > 4 else None
        gt_5th = gt_blocks[4] if len(gt_blocks) > 4 else None
        
        if gen_5th is None or gt_5th is None:
            return 0.0
        
        if gen_5th['color'] == gt_5th['color']:
            return 1.0
        
        return 0.0
    
    def _evaluate_color_accuracy(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if the predicted color is correct."""
        if len(gen_blocks) < 5 or len(gt_blocks) < 5:
            return 0.0
        
        gen_5th = gen_blocks[4] if len(gen_blocks) > 4 else None
        gt_5th = gt_blocks[4] if len(gt_blocks) > 4 else None
        
        if gen_5th is None or gt_5th is None:
            return 0.0
        
        if gen_5th['color'] == gt_5th['color']:
            return 1.0
        
        # Partial credit for similar colors
        similar_colors = {
            ('red', 'orange'): 0.3,
            ('orange', 'yellow'): 0.3,
            ('blue', 'purple'): 0.3,
            ('green', 'blue'): 0.2,
        }
        
        color_pair = tuple(sorted([gen_5th['color'], gt_5th['color']]))
        return similar_colors.get(color_pair, 0.0)
    
    def _evaluate_visual_presentation(self, gen_blocks: List[Dict]) -> float:
        """Evaluate visual consistency of blocks."""
        if len(gen_blocks) < 5:
            return 0.0
        
        # Check if all blocks have similar size
        areas = [b['area'] for b in gen_blocks[:5]]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        if mean_area > 0:
            cv = std_area / mean_area
            return max(0.3, 1.0 - cv)
        
        return 0.5
    
    def _evaluate_task_understanding(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if only position 5 was filled."""
        # Check if first 4 blocks match GT (unchanged)
        if len(gen_blocks) < 4 or len(gt_blocks) < 4:
            return 0.0
        
        matches = 0
        for i in range(4):
            if gen_blocks[i]['color'] == gt_blocks[i]['color']:
                matches += 1
        
        return matches / 4
