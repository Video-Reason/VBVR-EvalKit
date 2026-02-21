"""Evaluator for O-24_domino_chain_gap_analysis_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-24_domino_chain_gap_analysis_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class DominoChainGapEvaluator(BaseEvaluator):
    """
    O-24: Domino Chain Gap Analysis
    
    Task: Find where domino chain stops due to gap too large. Identify 
    the last domino that falls before the gap.
    
    Rule-based evaluation:
    1. Gap identification (40%) - Correct gap location
    2. Last fallen domino identification (35%) - Correct domino number
    3. Domino state accuracy (15%) - Fallen vs standing correct
    4. Chain animation quality (10%) - Sequential falling
    """
    
    TASK_WEIGHTS = {
        'gap_identification': 0.40,
        'last_fallen': 0.35,
        'domino_state': 0.15,
        'animation_quality': 0.10
    }
    
    def _analyze_domino_colors(self, frame: np.ndarray) -> Dict[str, int]:
        """Analyze domino color states."""
        if len(frame.shape) != 3:
            return {'red': 0, 'blue': 0}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection (fallen)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue detection (standing)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        return {
            'red': int(np.sum(red_mask > 0)),
            'blue': int(np.sum(blue_mask > 0))
        }
    
    def _find_gap_position(self, frame: np.ndarray) -> Optional[int]:
        """Find x-position of gap in domino chain."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Find vertical structures (dominoes)
        edges = cv2.Canny(gray, 50, 150)
        
        # Project to x-axis
        x_projection = np.sum(edges, axis=0)
        
        # Find gaps (low values)
        threshold = np.max(x_projection) * 0.2
        gap_positions = np.where(x_projection < threshold)[0]
        
        if len(gap_positions) > 0:
            # Find largest continuous gap
            gaps = []
            start = gap_positions[0]
            for i in range(1, len(gap_positions)):
                if gap_positions[i] - gap_positions[i-1] > 5:
                    gaps.append((start, gap_positions[i-1]))
                    start = gap_positions[i]
            gaps.append((start, gap_positions[-1]))
            
            if gaps:
                largest_gap = max(gaps, key=lambda g: g[1] - g[0])
                return (largest_gap[0] + largest_gap[1]) // 2
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate domino chain gap analysis."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Gap identification: Compare gap positions
        gen_gap = self._find_gap_position(gen_final)
        gt_gap = self._find_gap_position(gt_final)
        
        if gen_gap is not None and gt_gap is not None:
            gap_diff = abs(gen_gap - gt_gap)
            scores['gap_identification'] = max(0, 1.0 - gap_diff / 100.0)
        else:
            scores['gap_identification'] = 0.5 if gen_gap == gt_gap else 0.3
        
        # 2. Last fallen: Compare color distribution
        gen_colors = self._analyze_domino_colors(gen_final)
        gt_colors = self._analyze_domino_colors(gt_final)
        
        gen_ratio = gen_colors['red'] / max(gen_colors['red'] + gen_colors['blue'], 1)
        gt_ratio = gt_colors['red'] / max(gt_colors['red'] + gt_colors['blue'], 1)
        
        ratio_diff = abs(gen_ratio - gt_ratio)
        scores['last_fallen'] = max(0, 1.0 - ratio_diff * 2)
        
        # 3. Domino state accuracy
        if gen_colors['red'] + gen_colors['blue'] > 0 and gt_colors['red'] + gt_colors['blue'] > 0:
            red_match = min(gen_colors['red'], gt_colors['red']) / max(gen_colors['red'], gt_colors['red'], 1)
            blue_match = min(gen_colors['blue'], gt_colors['blue']) / max(gen_colors['blue'], gt_colors['blue'], 1)
            scores['domino_state'] = 0.5 * red_match + 0.5 * blue_match
        else:
            scores['domino_state'] = 0.2  # Detection failed
        
        # 4. Animation quality: Compare frame-by-frame
        if len(video_frames) >= 2 and len(gt_frames) >= 2:
            motion_scores = []
            for i in range(1, min(len(video_frames), 5)):
                diff = cv2.absdiff(video_frames[i], video_frames[i-1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            if motion_scores:
                variance = np.var(motion_scores)
                scores['animation_quality'] = max(0, 1.0 - variance / 500.0)
            else:
                scores['animation_quality'] = 0.2  # Detection failed
        else:
            scores['animation_quality'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
