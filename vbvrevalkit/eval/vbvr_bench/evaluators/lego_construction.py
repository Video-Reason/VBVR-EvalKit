"""Evaluator for O-25_LEGO_construction_assembly_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-25_LEGO_construction_assembly_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class LEGOConstructionEvaluator(BaseEvaluator):
    """
    O-25: LEGO Construction Assembly
    
    Task: Follow LEGO assembly instructions - move highlighted brick to 
    arrow-indicated position on partial model.
    
    Rule-based evaluation:
    1. Position accuracy (35%) - Brick at arrow position
    2. Stud alignment (30%) - Studs properly aligned
    3. Rotation correctness (20%) - Brick orientation correct
    4. Connection stability (15%) - Brick properly connected
    """
    
    TASK_WEIGHTS = {
        'position': 0.35,
        'stud_alignment': 0.30,
        'rotation': 0.20,
        'connection': 0.15
    }
    
    def _detect_highlighted_brick(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect highlighted (usually yellow/bright) brick position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        # Yellow/highlighted detection
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _analyze_structure(self, frame: np.ndarray) -> Dict:
        """Analyze LEGO structure properties."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'edge_count': np.sum(edges > 0),
            'contour_count': len(contours),
            'total_area': sum(cv2.contourArea(c) for c in contours)
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate LEGO brick placement accuracy."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Position accuracy: Compare brick positions
        gen_brick = self._detect_highlighted_brick(gen_final)
        gt_brick = self._detect_highlighted_brick(gt_final)
        
        if gen_brick is not None and gt_brick is not None:
            dist = np.sqrt((gen_brick[0] - gt_brick[0])**2 + (gen_brick[1] - gt_brick[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 2. Stud alignment: Compare edge structures
        gen_struct = self._analyze_structure(gen_final)
        gt_struct = self._analyze_structure(gt_final)
        
        if gt_struct['edge_count'] > 0:
            edge_ratio = min(gen_struct['edge_count'], gt_struct['edge_count']) / max(gen_struct['edge_count'], gt_struct['edge_count'])
            scores['stud_alignment'] = edge_ratio
        else:
            scores['stud_alignment'] = 0.2  # Detection failed
        
        # 3. Rotation: Compare overall frame similarity
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['rotation'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['rotation'] = 0.2  # Detection failed
        
        # 4. Connection: Check structure completeness
        if gt_struct['contour_count'] > 0:
            contour_ratio = min(gen_struct['contour_count'], gt_struct['contour_count']) / max(gen_struct['contour_count'], gt_struct['contour_count'])
            scores['connection'] = contour_ratio
        else:
            scores['connection'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
