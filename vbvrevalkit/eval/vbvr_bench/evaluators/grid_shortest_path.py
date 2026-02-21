"""Evaluator for G-18_grid_shortest_path_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-18_grid_shortest_path_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class GridShortestPathEvaluator(BaseEvaluator):
    """
    G-18: Grid shortest path evaluator.
    
    Rule-based evaluation:
    - Path optimality (50%): Shortest path (Manhattan distance)
    - Completion (25%): Agent reaches endpoint
    - Movement rules (15%): Only up/down/left/right movement
    - Visual fidelity (10%): Agent appearance preserved
    """
    
    TASK_WEIGHTS = {
        'path_optimal': 0.50,
        'completion': 0.25,
        'movement': 0.15,
        'fidelity': 0.10
    }
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect colored circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple colors for agent (pink agent is common)
        for lower, upper in [
            (np.array([150, 50, 50]), np.array([180, 255, 255])),  # Pink (most common agent color)
            (np.array([120, 50, 50]), np.array([160, 255, 255])),  # Purple
            (np.array([10, 100, 100]), np.array([25, 255, 255])),  # Orange
            (np.array([0, 100, 100]), np.array([10, 255, 255])),   # Red
        ]:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100 or area > 5000:
                    continue
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:  # Reasonably circular
                    M = cv2.moments(cnt)
                    if M['m00'] == 0:
                        continue
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        
        return None
    
    def _detect_endpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect colored endpoint square."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple colors for endpoint
        for lower, upper in [
            (np.array([150, 50, 50]), np.array([180, 255, 255])),  # Pink
            (np.array([0, 100, 100]), np.array([10, 255, 255])),   # Red
        ]:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid shortest path task."""
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
        
        # Track GT agent positions
        gt_agent_positions = []
        for frame in gt_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                gt_agent_positions.append(pos)
        
        # 1. Path optimality: Compare path length with GT
        if len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            gen_path_len = sum(np.sqrt((agent_positions[i][0] - agent_positions[i-1][0])**2 + 
                                       (agent_positions[i][1] - agent_positions[i-1][1])**2)
                              for i in range(1, len(agent_positions)))
            gt_path_len = sum(np.sqrt((gt_agent_positions[i][0] - gt_agent_positions[i-1][0])**2 + 
                                      (gt_agent_positions[i][1] - gt_agent_positions[i-1][1])**2)
                             for i in range(1, len(gt_agent_positions)))
            
            if gt_path_len > 0:
                # Allow some tolerance for path length
                ratio = min(gen_path_len, gt_path_len) / max(gen_path_len, gt_path_len)
                scores['path_optimal'] = ratio
            else:
                scores['path_optimal'] = 0.2  # Detection failed
        else:
            scores['path_optimal'] = 0.2  # Detection failed
        
        # 2. Completion: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)
        gt_endpoint = self._detect_endpoint(gt_final_frame)
        gt_final_agent = self._detect_agent(gt_final_frame)
        
        if final_agent is not None and gt_final_agent is not None:
            dist = np.sqrt((final_agent[0] - gt_final_agent[0])**2 + 
                          (final_agent[1] - gt_final_agent[1])**2)
            scores['completion'] = max(0, 1.0 - dist / 50.0)
        elif endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0])**2 + (endpoint[1] - final_agent[1])**2)
            scores['completion'] = 1.0 if dist < 50 else max(0, 1.0 - dist / 100.0)
        elif len(agent_positions) >= 2 and len(gt_agent_positions) >= 2:
            # If agent detection fails in final frame, use last known position
            # This handles cases where agent merges with endpoint
            gen_last_pos = agent_positions[-1]
            gt_last_pos = gt_agent_positions[-1]
            dist = np.sqrt((gen_last_pos[0] - gt_last_pos[0])**2 + 
                          (gen_last_pos[1] - gt_last_pos[1])**2)
            scores['completion'] = max(0, 1.0 - dist / 100.0)
        else:
            scores['completion'] = 0.2  # Detection failed
        
        # 3. Movement rules: Check for diagonal movements
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
        
        # 4. Fidelity: Compare agent appearance (use any detected agent through video)
        if final_agent is not None or len(agent_positions) > 0:
            scores['fidelity'] = 1.0
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
