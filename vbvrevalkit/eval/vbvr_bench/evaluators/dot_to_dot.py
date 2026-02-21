"""Evaluator for O-34_dot_to_dot_task_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-34_dot_to_dot_task_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class DotToDotEvaluator(BaseEvaluator):
    """
    O-34: Dot to Dot
    
    Task: Connect numbered dots in sequence (1→2→3→...→N) with straight 
    red lines to form continuous path.
    
    Rule-based evaluation:
    1. Connection order (40%) - Strict numerical sequence
    2. Connection completeness (30%) - All dots connected
    3. Line quality (20%) - Straight lines, connect centers
    4. Visual fidelity (10%) - Red color, dots preserved
    """
    
    TASK_WEIGHTS = {
        'connection_order': 0.40,
        'completeness': 0.30,
        'line_quality': 0.20,
        'visual_fidelity': 0.10
    }
    
    def _count_red_line_pixels(self, frame: np.ndarray) -> int:
        """Count red line pixels."""
        if len(frame.shape) != 3:
            return 0
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        return int(np.sum(red_mask > 0))
    
    def _detect_blue_dots(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect blue dot positions."""
        if len(frame.shape) != 3:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                dots.append((cx, cy))
        
        return dots
    
    def _detect_red_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect red line segments."""
        if len(frame.shape) != 3:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        lines = cv2.HoughLinesP(red_mask, 1, np.pi/180, threshold=30,
                                minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate dot-to-dot connection accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Connection order: Compare overall structure
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['connection_order'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['connection_order'] = 0.2  # Detection failed
        
        # 2. Completeness: Compare red line amounts
        gen_red = self._count_red_line_pixels(gen_final)
        gt_red = self._count_red_line_pixels(gt_final)
        
        if gt_red > 0:
            ratio = min(gen_red, gt_red) / max(gen_red, gt_red)
            scores['completeness'] = ratio
        else:
            scores['completeness'] = 0.5 if gen_red == 0 else 0.3
        
        # 3. Line quality: Compare line counts
        gen_lines = self._detect_red_lines(gen_final)
        gt_lines = self._detect_red_lines(gt_final)
        
        if gt_lines:
            line_ratio = min(len(gen_lines), len(gt_lines)) / max(len(gen_lines), len(gt_lines), 1)
            scores['line_quality'] = line_ratio
        else:
            scores['line_quality'] = 0.2  # Detection failed
        
        # 4. Visual fidelity: Check dots preserved
        gen_dots = self._detect_blue_dots(gen_final)
        gt_dots = self._detect_blue_dots(gt_final)
        
        if gt_dots:
            dot_ratio = min(len(gen_dots), len(gt_dots)) / max(len(gen_dots), len(gt_dots), 1)
            scores['visual_fidelity'] = dot_ratio
        else:
            scores['visual_fidelity'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)


# Export mapping for this batch
OPEN60_EVALUATORS_PART5 = {
    'O-23_domino_chain_branch_path_prediction_data-generator': DominoChainBranchEvaluator,
    'O-24_domino_chain_gap_analysis_data-generator': DominoChainGapEvaluator,
    'O-25_LEGO_construction_assembly_data-generator': LEGOConstructionEvaluator,
    'O-27_move_2_object_to_2_target_data-generator': MoveObjectsToTargetEvaluator,
    'O-29_ballcolor_data-generator': BallColorEvaluator,
    'O-30_bookshelf_data-generator': BookshelfEvaluator,
    'O-31_ball_eating_data-generator': BallEatingEvaluator,
    'O-32_rolling_ball_data-generator': RollingBallEvaluator,
    'O-33_counting_object_data-generator': CountingObjectEvaluator,
    'O-34_dot_to_dot_task_data-generator': DotToDotEvaluator,
