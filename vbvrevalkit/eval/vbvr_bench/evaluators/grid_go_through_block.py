"""Evaluator for G-16_grid_go_through_block_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-16_grid_go_through_block_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class GridGoThroughBlockEvaluator(BaseEvaluator):
    """
    G-16: Grid go through block evaluator.
    
    Rule-based evaluation:
    - Block visit completeness (40%): All blue blocks visited
    - Path optimality (30%): TSP-optimal path through all blocks
    - Completion (20%): Agent reaches red endpoint after visiting all
    - Movement rules (10%): Only up/down/left/right movement
    """
    
    TASK_WEIGHTS = {
        'block_visit': 0.40,
        'path_optimal': 0.30,
        'completion': 0.20,
        'movement': 0.10
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
    
    def _detect_blue_blocks(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect blue blocks to visit."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            blocks.append((cx, cy))
        
        return blocks
    
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
        """Evaluate grid go through block task."""
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Detect blue blocks in first frame
        blue_blocks = self._detect_blue_blocks(first_frame)
        
        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)
        
        # 1. Block visit completeness: Check if agent visits all blue blocks
        if agent_positions and blue_blocks:
            visited_blocks = set()
            for pos in agent_positions:
                for i, block in enumerate(blue_blocks):
                    dist = np.sqrt((pos[0] - block[0])**2 + (pos[1] - block[1])**2)
                    if dist < 40:
                        visited_blocks.add(i)
            scores['block_visit'] = len(visited_blocks) / max(len(blue_blocks), 1)
        else:
            scores['block_visit'] = 0.2  # Detection failed
        
        # 2. Path optimality: Compare path length with GT
        gt_agent_positions = []
        for frame in gt_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                gt_agent_positions.append(pos)
        
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
                scores['path_optimal'] = 0.2  # Detection failed
        else:
            scores['path_optimal'] = 0.2  # Detection failed
        
        # 3. Completion: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)
        
        if endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0])**2 + (endpoint[1] - final_agent[1])**2)
            scores['completion'] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        else:
            scores['completion'] = 0.2  # Detection failed
        
        # 4. Movement rules: Check for diagonal movements
        if len(agent_positions) >= 2:
            diagonal_count = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 10 and dy > 10:
                    diagonal_count += 1
            scores['movement'] = max(0, 1.0 - diagonal_count * 0.2)
        else:
            scores['movement'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
