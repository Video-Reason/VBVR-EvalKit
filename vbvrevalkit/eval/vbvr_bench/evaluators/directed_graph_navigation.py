"""Evaluator for G-31_directed_graph_navigation_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-31_directed_graph_navigation_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class DirectedGraphNavigationEvaluator(BaseEvaluator):
    """
    G-31: Directed graph navigation evaluator.
    
    CRITICAL RULES:
    1. Blue triangle (agent) must move from green circle to red circle
    2. Agent must reach the red circle (endpoint)
    3. All circle colors (green, red) must NOT change
    """
    
    TASK_WEIGHTS = {
        'completion': 0.35,           # Agent reaches red endpoint
        'circles_preserved': 0.50,    # Circle colors unchanged - MORE IMPORTANT
        'path_quality': 0.15          # Follows graph structure
    }
    
    def _count_circle_colors(self, frame: np.ndarray) -> Tuple[int, int]:
        """Count green and red pixels in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_count = np.sum(green_mask > 0)
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_count = np.sum(red_mask > 0)
        
        return green_count, red_count
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate directed graph navigation task.
        
        CRITICAL RULES:
        1. Agent must reach red endpoint
        2. Circle colors (green, red) must NOT change significantly
        """
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # CRITICAL: Check if circle colors are preserved
        first_green, first_red = self._count_circle_colors(first_frame)
        final_green, final_red = self._count_circle_colors(last_frame)
        
        first_total = first_green + first_red
        final_total = final_green + final_red
        
        # Circle colors should not change dramatically
        total_change = abs(final_total - first_total) / max(first_total, 1)
        
        if total_change > 1.0:  # More than 100% increase
            # Circle colors changed significantly - task failed
            scores['circles_preserved'] = 0.0
            scores['completion'] = 0.0
            scores['path_quality'] = 0.0
            self._last_task_details = scores
            self._last_task_details['circles_changed'] = True
            return 0.0
        else:
            scores['circles_preserved'] = max(0, 1.0 - total_change)
        
        # Detect nodes and agent
        nodes_first = self._detect_nodes(first_frame)
        gen_agent_final = self._detect_agent(last_frame)
        
        # 1. Completion: Check if agent reached red endpoint
        if gen_agent_final is not None and nodes_first.get('end') is not None:
            end_pos = nodes_first['end']
            dist = np.sqrt((gen_agent_final[0] - end_pos[0])**2 + 
                          (gen_agent_final[1] - end_pos[1])**2)
            if dist < 50:
                scores['completion'] = 1.0
            elif dist < 100:
                scores['completion'] = 0.3  # STRICT: Not at endpoint
            else:
                scores['completion'] = 0.0  # STRICT: Failed to reach endpoint
        else:
            scores['completion'] = 0.0
        
        # 2. Path quality: Check if agent followed graph structure
        agent_positions = self._track_agent(video_frames)
        if len(agent_positions) >= 2:
            # Check for smooth movement (no teleporting)
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 150 or dy > 150:
                    large_jumps += 1
            scores['path_quality'] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores['path_quality'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect blue triangular agent position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest blue region (the agent)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _track_agent(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track agent position across all frames."""
        positions = []
        for frame in frames:
            pos = self._detect_agent(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_nodes(self, frame: np.ndarray) -> Dict:
        """Detect graph nodes (green=start, red=end, white=intermediate)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        nodes = {'start': None, 'end': None, 'intermediate': []}
        
        # Green (start node)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    nodes['start'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                break
        
        # Red (end node)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    nodes['end'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                break
        
        return nodes
    
    def _evaluate_path_length(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent took shortest path."""
        if not agent_positions or not nodes.get('end'):
            return 0.0
        
        # Check if agent reached end node
        final_pos = agent_positions[-1]
        end_node = nodes['end']
        
        dist_to_end = np.sqrt((final_pos[0] - end_node[0])**2 + (final_pos[1] - end_node[1])**2)
        
        # If agent reached end (within threshold)
        if dist_to_end < 50:
            # Count number of significant position changes (steps)
            steps = 0
            prev_pos = agent_positions[0]
            for pos in agent_positions[1:]:
                dist = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                if dist > 20:  # Significant movement
                    steps += 1
                    prev_pos = pos
            
            # Fewer steps is better (assuming optimal is 3-5 steps for typical graph)
            if steps <= 5:
                return 1.0
            elif steps <= 7:
                return 0.8
            elif steps <= 10:
                return 0.6
            else:
                return 0.4
        else:
            # Didn't reach end
            return max(0.2, 1.0 - dist_to_end / 500)
    
    def _evaluate_direction_compliance(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent follows arrow directions."""
        if len(agent_positions) < 2:
            return 0.5
        
        # For now, check if movement is generally forward (left to right or top to bottom)
        # This is a simplification; full implementation would need edge detection
        forward_moves = 0
        total_moves = 0
        
        for i in range(1, len(agent_positions)):
            dx = agent_positions[i][0] - agent_positions[i-1][0]
            dy = agent_positions[i][1] - agent_positions[i-1][1]
            
            if abs(dx) > 10 or abs(dy) > 10:  # Significant movement
                total_moves += 1
                # Generally forward progress (towards end)
                if nodes.get('end') and nodes.get('start'):
                    # Check if moving towards end
                    prev_dist = np.sqrt((agent_positions[i-1][0] - nodes['end'][0])**2 + 
                                       (agent_positions[i-1][1] - nodes['end'][1])**2)
                    curr_dist = np.sqrt((agent_positions[i][0] - nodes['end'][0])**2 + 
                                       (agent_positions[i][1] - nodes['end'][1])**2)
                    if curr_dist < prev_dist:
                        forward_moves += 1
        
        return forward_moves / max(1, total_moves)
    
    def _evaluate_movement_legality(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent moves along edges (not jumping)."""
        if len(agent_positions) < 2:
            return 0.5
        
        # Check for smooth movement (no large jumps)
        smooth_moves = 0
        total_moves = 0
        
        for i in range(1, len(agent_positions)):
            dist = np.sqrt((agent_positions[i][0] - agent_positions[i-1][0])**2 + 
                          (agent_positions[i][1] - agent_positions[i-1][1])**2)
            
            if dist > 5:  # Significant movement
                total_moves += 1
                if dist < 100:  # Reasonable step size
                    smooth_moves += 1
        
        return smooth_moves / max(1, total_moves)
    
    def _evaluate_graph_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if graph structure is preserved."""
        # Compare edge structures using edge detection
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)
        
        # Compare edge maps
        intersection = np.sum((gen_edges > 0) & (gt_edges > 0))
        union = np.sum((gen_edges > 0) | (gt_edges > 0))
        
        if union > 0:
            return intersection / union
        return 0.5
