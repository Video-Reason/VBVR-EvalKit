"""Evaluator for O-23_domino_chain_branch_path_prediction_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-23_domino_chain_branch_path_prediction_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class DominoChainBranchEvaluator(BaseEvaluator):
    """
    O-23: Domino Chain Branch Path Prediction
    
    Task: Y-shaped domino structure - push START, dominoes fall through 
    trunk to fork, then both branches fall (unless blocked by gap).
    
    Rule-based evaluation:
    1. Fallen domino accuracy (40%) - Correct dominoes fall
    2. Gap rule application (30%) - Gap blocks chain correctly
    3. Chain reaction sequence (20%) - Correct order
    4. Fall direction accuracy (10%) - Correct tilt directions
    """
    
    TASK_WEIGHTS = {
        'fallen_domino': 0.40,
        'gap_rule': 0.30,
        'chain_sequence': 0.20,
        'fall_direction': 0.10
    }
    
    def _count_standing_dominoes(self, frame: np.ndarray) -> int:
        """Count number of vertical (standing) domino shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect vertical lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)
        
        standing_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 20 and dx < 10:  # Roughly vertical
                    standing_count += 1
        
        return standing_count
    
    def _count_fallen_dominoes(self, frame: np.ndarray) -> int:
        """Count number of tilted/fallen domino shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)
        
        fallen_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                # Tilted: significant both dx and dy
                if dx > 10 and dy > 10 and abs(dy - dx) < max(dx, dy) * 0.5:
                    fallen_count += 1
        
        return fallen_count
    
    def _detect_dominoes_by_color(self, frame: np.ndarray) -> Dict[str, int]:
        """Detect dominoes by their colors (red=fallen, blue=standing typically)."""
        if len(frame.shape) != 3:
            return {'red': 0, 'blue': 0}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue detection
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'red': len([c for c in red_contours if cv2.contourArea(c) > 100]),
            'blue': len([c for c in blue_contours if cv2.contourArea(c) > 100])
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate domino chain branch prediction."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Fallen domino accuracy: Compare domino states
        gen_standing = self._count_standing_dominoes(gen_final)
        gt_standing = self._count_standing_dominoes(gt_final)
        gen_fallen = self._count_fallen_dominoes(gen_final)
        gt_fallen = self._count_fallen_dominoes(gt_final)
        
        if gt_standing + gt_fallen > 0:
            standing_match = max(0, 1.0 - abs(gen_standing - gt_standing) / max(gt_standing + gt_fallen, 1))
            fallen_match = max(0, 1.0 - abs(gen_fallen - gt_fallen) / max(gt_standing + gt_fallen, 1))
            scores['fallen_domino'] = 0.5 * standing_match + 0.5 * fallen_match
        else:
            scores['fallen_domino'] = 0.2  # Detection failed
        
        # 2. Gap rule: Compare color distribution (red=fallen, blue=standing)
        gen_colors = self._detect_dominoes_by_color(gen_final)
        gt_colors = self._detect_dominoes_by_color(gt_final)
        
        gen_ratio = gen_colors['red'] / max(gen_colors['red'] + gen_colors['blue'], 1)
        gt_ratio = gt_colors['red'] / max(gt_colors['red'] + gt_colors['blue'], 1)
        
        ratio_diff = abs(gen_ratio - gt_ratio)
        scores['gap_rule'] = max(0, 1.0 - ratio_diff * 2)
        
        # 3. Chain sequence: Analyze progression through video
        if len(video_frames) >= 3:
            # Check if fallen count increases over time
            early_fallen = self._count_fallen_dominoes(video_frames[len(video_frames)//4])
            mid_fallen = self._count_fallen_dominoes(video_frames[len(video_frames)//2])
            late_fallen = self._count_fallen_dominoes(video_frames[-1])
            
            if early_fallen <= mid_fallen <= late_fallen:
                scores['chain_sequence'] = 1.0
            elif early_fallen <= late_fallen:
                scores['chain_sequence'] = 0.7
            else:
                scores['chain_sequence'] = 0.3
        else:
            scores['chain_sequence'] = 0.2  # Detection failed
        
        # 4. Fall direction: Compare overall structure
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['fall_direction'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['fall_direction'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
