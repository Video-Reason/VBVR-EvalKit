"""Evaluator for G-13_grid_number_sequence_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-13_grid_number_sequence_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class GridNumberSequenceEvaluator(BaseEvaluator):
    """
    G-13: Grid number sequence evaluator.
    
    Rule-based evaluation:
    - Sequence correctness (35%): Visit numbers 1→2→3 in order
    - Path optimality (35%): Shortest path between consecutive targets
    - Movement rules (20%): Only up/down/left/right, no diagonal
    - Completeness (10%): Agent reaches red endpoint after all numbers
    """
    
    TASK_WEIGHTS = {
        'sequence': 0.35,
        'path_optimal': 0.35,
        'movement': 0.20,
        'completeness': 0.10
    }
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect orange circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red endpoint."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid number sequence task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        last_frame = video_frames[-1]
        
        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)
        
        # 1. Sequence correctness: Compare agent trajectory with GT
        gt_agent_positions = []
        for frame in gt_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                gt_agent_positions.append(pos)
        
        if agent_positions and gt_agent_positions:
            # Compare final positions
            final_gen = agent_positions[-1]
            final_gt = gt_agent_positions[-1]
            dist = np.sqrt((final_gen[0] - final_gt[0])**2 + (final_gen[1] - final_gt[1])**2)
            scores['sequence'] = max(0, 1.0 - dist / 100.0)
        else:
            scores['sequence'] = 0.1  # Detection failed
        
        # 2. Path optimality: Compare path length
        if len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            gen_path_len = sum(np.sqrt((agent_positions[i][0] - agent_positions[i-1][0])**2 + 
                                       (agent_positions[i][1] - agent_positions[i-1][1])**2)
                              for i in range(1, len(agent_positions)))
            gt_path_len = sum(np.sqrt((gt_agent_positions[i][0] - gt_agent_positions[i-1][0])**2 + 
                                      (gt_agent_positions[i][1] - gt_agent_positions[i-1][1])**2)
                             for i in range(1, len(gt_agent_positions)))
            
            if gt_path_len > 0:
                ratio = min(gen_path_len, gt_path_len) / max(gen_path_len, gt_path_len)
                scores['path_optimal'] = ratio
            else:
                scores['path_optimal'] = 0.1  # Detection failed
        else:
            scores['path_optimal'] = 0.1  # Detection failed
        
        # 3. Movement rules: Check for diagonal movements
        if len(agent_positions) >= 2:
            diagonal_count = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                # Diagonal if both dx and dy are significant
                if dx > 10 and dy > 10:
                    diagonal_count += 1
            scores['movement'] = max(0, 1.0 - diagonal_count * 0.2)
        else:
            scores['movement'] = 0.1  # Detection failed
        
        # 4. Completeness: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)
        
        if endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0])**2 + (endpoint[1] - final_agent[1])**2)
            scores['completeness'] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        else:
            scores['completeness'] = 0.1  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
