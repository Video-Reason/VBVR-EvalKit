"""Evaluator for O-16_color_addition_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-16_color_addition_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ColorAdditionEvaluator(BaseEvaluator):
    """
    O-16: Color Addition (Additive Color Mixing)
    
    Task: Two colored balls move toward each other and merge, showing 
    additive color mixing in the overlap region.
    
    Rule-based evaluation:
    1. Additive mixing accuracy (40%) - Correct RGB addition
    2. Movement trajectory (30%) - Equal speed, meet at midpoint
    3. Overlap handling (20%) - Proper blend in overlap region
    4. Visual fidelity (10%) - Ball size, shape preserved
    """
    
    TASK_WEIGHTS = {
        'mixing': 0.40,
        'movement': 0.30,
        'overlap': 0.20,
        'fidelity': 0.10
    }
    
    def _detect_colored_regions(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored regions and their properties."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return []
        
        regions = []
        
        # Define color ranges (lower saturation threshold for blended colors)
        color_ranges = {
            'red': [([0, 50, 100], [10, 255, 255]), ([160, 50, 100], [180, 255, 255])],
            'green': [([35, 50, 100], [85, 255, 255])],
            'blue': [([100, 50, 100], [140, 255, 255])],  # Extended to include violet/purple
            'yellow': [([20, 50, 100], [35, 255, 255])],
            'cyan': [([85, 50, 100], [100, 255, 255])],
            'magenta': [([140, 50, 100], [160, 255, 255])],
        }
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                regions.append({
                    'color': color_name,
                    'center': (cx, cy),
                    'area': area
                })
        
        return regions
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate additive color mixing animation."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Detect colored regions
        gen_regions = self._detect_colored_regions(gen_final)
        gt_regions = self._detect_colored_regions(gt_final)
        
        # 1. Mixing accuracy: Compare color distribution (non-white regions only)
        if gen_final.shape == gt_final.shape:
            # Only compare non-white regions
            gt_gray = cv2.cvtColor(gt_final, cv2.COLOR_BGR2GRAY)
            non_white_mask = gt_gray < 240
            
            if np.sum(non_white_mask) > 100:
                gen_masked = gen_final[non_white_mask]
                gt_masked = gt_final[non_white_mask]
                color_diff = np.abs(gen_masked.astype(float) - gt_masked.astype(float)).mean()
                
                # Stricter threshold for non-background comparison
                if color_diff < 20:
                    scores['mixing'] = 1.0
                elif color_diff < 40:
                    scores['mixing'] = 0.5
                else:
                    scores['mixing'] = 0.0
            else:
                scores['mixing'] = 0.0  # No colored content
        else:
            scores['mixing'] = 0.0  # Detection failed
        
        # 2. Movement: Check if balls moved toward center
        if len(video_frames) >= 2:
            first_regions = self._detect_colored_regions(video_frames[0])
            
            if first_regions and gen_regions:
                # Calculate center of mass movement
                first_com = np.mean([r['center'] for r in first_regions], axis=0) if first_regions else (0, 0)
                final_com = np.mean([r['center'] for r in gen_regions], axis=0) if gen_regions else (0, 0)
                
                frame_center = (gen_final.shape[1] // 2, gen_final.shape[0] // 2)
                
                # Check if regions moved toward center
                first_dist = np.sqrt((first_com[0] - frame_center[0])**2 + (first_com[1] - frame_center[1])**2)
                final_dist = np.sqrt((final_com[0] - frame_center[0])**2 + (final_com[1] - frame_center[1])**2)
                
                if first_dist > final_dist:
                    scores['movement'] = min(1.0, (first_dist - final_dist) / 50.0 + 0.5)
                else:
                    scores['movement'] = 0.0
            else:
                scores['movement'] = 0.0  # Detection failed
        else:
            scores['movement'] = 0.0  # Detection failed
        
        # 3. Overlap handling: Check for blended region
        if gen_regions:
            # Multiple colors or blended = good overlap handling
            unique_colors = set(r['color'] for r in gen_regions)
            scores['overlap'] = min(1.0, len(unique_colors) / 2.0)
        else:
            scores['overlap'] = 0.0  # Detection failed
        
        # 4. Fidelity: Compare region counts
        if gen_regions and gt_regions:
            count_ratio = min(len(gen_regions), len(gt_regions)) / max(len(gen_regions), len(gt_regions), 1)
            scores['fidelity'] = count_ratio
        else:
            scores['fidelity'] = 0.0  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
