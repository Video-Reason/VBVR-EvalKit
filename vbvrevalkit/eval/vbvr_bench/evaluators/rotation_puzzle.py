"""Evaluator for O-44_rotation_puzzle_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-44_rotation_puzzle_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class RotationPuzzleEvaluator(BaseEvaluator):
    """
    O-44: Rotation Puzzle (Pipe Connection)
    
    Task: Rotate L-shaped pipe tiles in 2x2 grid to connect all pipes 
    into continuous path.
    
    Key evaluation criteria:
    1. Path connection (40%) - All pipes connected
    2. Rotation accuracy (30%) - Correct 90° rotations
    3. Position preservation (20%) - Tiles stay in place
    4. Alignment precision (10%) - Pipe openings aligned
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'path_connection': 0.40,
            'rotation_accuracy': 0.30,
            'position_preservation': 0.20,
            'alignment_precision': 0.10
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate rotation puzzle solution."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        scores = {}
        
        # 1. Path connection (40%): Check if pipes form connected path
        connection_score = self._evaluate_path_connection(gen_final, gt_final)
        scores['path_connection'] = connection_score
        
        # 2. Rotation accuracy (30%): Check pipe orientations match GT
        rotation_score = self._evaluate_rotation_accuracy(gen_final, gt_final)
        scores['rotation_accuracy'] = rotation_score
        
        # 3. Position preservation (20%): Check tiles are in correct positions
        position_score = self._evaluate_position_preservation(first_frame, gen_final)
        scores['position_preservation'] = position_score
        
        # 4. Alignment precision (10%): Check pipe openings align
        alignment_score = self._evaluate_alignment_precision(gen_final, gt_final)
        scores['alignment_precision'] = alignment_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_blue_pipes(self, frame: np.ndarray) -> np.ndarray:
        """Detect blue pipe regions."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue pipe color
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return blue_mask
    
    def _evaluate_path_connection(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if pipes form a connected path."""
        gen_blue = self._detect_blue_pipes(gen_frame)
        gt_blue = self._detect_blue_pipes(gt_frame)
        
        if np.sum(gt_blue > 0) == 0:
            return 0.5
        
        # IoU of blue regions
        intersection = np.sum((gen_blue > 0) & (gt_blue > 0))
        union = np.sum((gen_blue > 0) | (gt_blue > 0))
        
        if union > 0:
            iou = intersection / union
            return iou
        
        return 0.5
    
    def _evaluate_rotation_accuracy(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if pipe orientations match GT."""
        # Divide frame into 2x2 quadrants and compare pipe orientations
        h, w = gen_frame.shape[:2]
        
        quadrant_scores = []
        for row in range(2):
            for col in range(2):
                y1, y2 = row * h // 2, (row + 1) * h // 2
                x1, x2 = col * w // 2, (col + 1) * w // 2
                
                gen_quad = gen_frame[y1:y2, x1:x2]
                gt_quad = gt_frame[y1:y2, x1:x2]
                
                gen_blue = self._detect_blue_pipes(gen_quad)
                gt_blue = self._detect_blue_pipes(gt_quad)
                
                if np.sum(gt_blue > 0) > 0:
                    intersection = np.sum((gen_blue > 0) & (gt_blue > 0))
                    gt_area = np.sum(gt_blue > 0)
                    
                    quadrant_scores.append(intersection / gt_area)
        
        return np.mean(quadrant_scores) if quadrant_scores else 0.5
    
    def _evaluate_position_preservation(self, first_frame: np.ndarray, gen_final: np.ndarray) -> float:
        """Evaluate if tiles stayed in their grid positions."""
        # Detect tile boundaries in both frames
        h, w = first_frame.shape[:2]
        
        # Check if 2x2 grid structure is preserved
        # Look for vertical and horizontal dividing lines
        
        gen_gray = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY)
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        # Check center regions for grid lines
        center_y = h // 2
        center_x = w // 2
        
        # Sample around center for grid line detection
        gen_center_h = gen_gray[center_y-5:center_y+5, :]
        gen_center_v = gen_gray[:, center_x-5:center_x+5]
        
        first_center_h = first_gray[center_y-5:center_y+5, :]
        first_center_v = first_gray[:, center_x-5:center_x+5]
        
        # Compare patterns
        h_diff = np.mean(np.abs(gen_center_h.astype(float) - first_center_h.astype(float)))
        v_diff = np.mean(np.abs(gen_center_v.astype(float) - first_center_v.astype(float)))
        
        avg_diff = (h_diff + v_diff) / 2
        
        if avg_diff < 20:
            return 1.0
        elif avg_diff < 50:
            return 0.7
        else:
            return max(0.3, 1.0 - avg_diff / 100)
    
    def _evaluate_alignment_precision(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate pipe opening alignment."""
        gen_blue = self._detect_blue_pipes(gen_frame)
        gt_blue = self._detect_blue_pipes(gt_frame)
        
        # Check alignment at quadrant boundaries
        h, w = gen_frame.shape[:2]
        
        # Horizontal boundary (middle row)
        gen_h_boundary = gen_blue[h//2-10:h//2+10, :]
        gt_h_boundary = gt_blue[h//2-10:h//2+10, :]
        
        # Vertical boundary (middle column)
        gen_v_boundary = gen_blue[:, w//2-10:w//2+10]
        gt_v_boundary = gt_blue[:, w//2-10:w//2+10]
        
        h_match = np.sum((gen_h_boundary > 0) & (gt_h_boundary > 0)) / max(1, np.sum(gt_h_boundary > 0))
        v_match = np.sum((gen_v_boundary > 0) & (gt_v_boundary > 0)) / max(1, np.sum(gt_v_boundary > 0))
        
        return (h_match + v_match) / 2
