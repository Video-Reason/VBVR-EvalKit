"""Evaluator for G-41_grid_highest_cost_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-41_grid_highest_cost_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class GridHighestCostEvaluator(BaseEvaluator):
    """
    G-41: Grid highest cost path evaluator.
    
    CRITICAL RULES:
    1. Yellow dot (Pacman) must move from green grid to red grid
    2. Pacman must reach the red goal cell
    3. Grid structure (colors) must NOT change significantly
    4. Should follow high-cost path (follow GT route)
    """
    
    TASK_WEIGHTS = {
        'completion': 0.45,       # Pacman reaches red goal
        'grid_preserved': 0.35,   # Grid colors unchanged
        'movement': 0.20          # Step by step movement
    }
    
    def _count_grid_colors(self, frame: np.ndarray) -> Tuple[int, int]:
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
        """Evaluate grid highest cost task.
        
        CRITICAL RULES:
        1. Pacman must reach red goal
        2. Grid colors must be preserved
        """
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # CRITICAL: Check if grid colors are preserved
        first_green, first_red = self._count_grid_colors(first_frame)
        final_green, final_red = self._count_grid_colors(last_frame)
        
        first_total = first_green + first_red
        final_total = final_green + final_red
        
        total_change = abs(final_total - first_total) / max(first_total, 1)
        
        if total_change > 1.0:  # More than 100% increase
            scores['grid_preserved'] = 0.0
            scores['completion'] = 0.0
            scores['movement'] = 0.0
            self._last_task_details = scores
            self._last_task_details['grid_changed'] = True
            return 0.0
        else:
            scores['grid_preserved'] = max(0, 1.0 - total_change)
        
        # Detect Pacman and red goal
        agent = self._detect_pacman(last_frame)
        goal = self._detect_red_goal(first_frame)
        
        # 1. Completion: Check if Pacman reached red goal
        if agent is not None and goal is not None:
            dist = np.sqrt((agent[0] - goal[0])**2 + (agent[1] - goal[1])**2)
            if dist < 50:
                scores['completion'] = 1.0
            elif dist < 100:
                scores['completion'] = 0.5
            else:
                scores['completion'] = 0.1
        else:
            scores['completion'] = 0.0
        
        # 2. Movement: Check for step-by-step movement
        agent_positions = self._track_pacman(video_frames)
        if len(agent_positions) >= 2:
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 150 or dy > 150:
                    large_jumps += 1
            scores['movement'] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores['movement'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_red_goal(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red goal cell."""
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
        return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    def _detect_pacman(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect yellow Pac-Man agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow color range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _track_pacman(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track Pac-Man position across frames."""
        positions = []
        for frame in frames:
            pos = self._detect_pacman(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_grid(self, frame: np.ndarray) -> Dict:
        """Detect 4x4 grid structure."""
        h, w = frame.shape[:2]
        
        # Assume 4x4 grid
        cell_w = w // 4
        cell_h = h // 4
        
        return {
            'rows': 4,
            'cols': 4,
            'cell_width': cell_w,
            'cell_height': cell_h
        }
    
    def _pos_to_cell(self, pos: Tuple[int, int], grid_info: Dict) -> Tuple[int, int]:
        """Convert pixel position to grid cell."""
        col = pos[0] // grid_info['cell_width']
        row = pos[1] // grid_info['cell_height']
        return (min(row, grid_info['rows'] - 1), min(col, grid_info['cols'] - 1))
    
    def _evaluate_path_cost(self, positions: List[Tuple[int, int]], grid_info: Dict, 
                            gt_frame: np.ndarray) -> float:
        """Evaluate if path has high cost."""
        if not positions:
            return 0.0
        
        # Get unique cells visited
        cells_visited = set()
        for pos in positions:
            cell = self._pos_to_cell(pos, grid_info)
            cells_visited.add(cell)
        
        # More cells visited generally means higher cost path (for max cost problem)
        # Optimal path should visit many high-value cells
        max_possible_cells = grid_info['rows'] * grid_info['cols']
        coverage = len(cells_visited) / max_possible_cells
        
        # For highest cost path, we expect more cells to be visited
        if coverage > 0.7:
            return 1.0
        elif coverage > 0.5:
            return 0.8
        elif coverage > 0.3:
            return 0.6
        else:
            return max(0.3, coverage * 2)
    
    def _evaluate_movement_legality(self, positions: List[Tuple[int, int]]) -> float:
        """Evaluate if movements are orthogonal (no diagonal)."""
        if len(positions) < 2:
            return 0.5
        
        legal_moves = 0
        total_moves = 0
        
        for i in range(1, len(positions)):
            dx = abs(positions[i][0] - positions[i-1][0])
            dy = abs(positions[i][1] - positions[i-1][1])
            
            if dx > 5 or dy > 5:  # Significant movement
                total_moves += 1
                # Orthogonal: one direction dominates
                if dx > 3 * dy or dy > 3 * dx:
                    legal_moves += 1
        
        return legal_moves / max(1, total_moves)
    
    def _evaluate_completeness(self, gen_frame: np.ndarray, gt_frame: np.ndarray, 
                               positions: List[Tuple[int, int]]) -> float:
        """Evaluate if agent reached destination (bottom-right)."""
        if not positions:
            return 0.0
        
        final_pos = positions[-1]
        h, w = gen_frame.shape[:2]
        
        # Destination is bottom-right corner
        dest = (w * 0.75, h * 0.75)  # Approximate center of bottom-right cell
        
        dist = np.sqrt((final_pos[0] - dest[0])**2 + (final_pos[1] - dest[1])**2)
        frame_diag = np.sqrt(h**2 + w**2)
        normalized_dist = dist / frame_diag
        
        if normalized_dist < 0.15:
            return 1.0
        elif normalized_dist < 0.25:
            return 0.7
        elif normalized_dist < 0.40:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_grid_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if grid structure is preserved."""
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect grid lines using edge detection
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)
        
        # Compare edge density
        gen_edge_density = np.sum(gen_edges > 0) / gen_edges.size
        gt_edge_density = np.sum(gt_edges > 0) / gt_edges.size
        
        if gt_edge_density > 0:
            ratio = gen_edge_density / gt_edge_density
            return min(1.0, max(0.3, 1.0 - abs(1.0 - ratio)))
        
        return 0.5
