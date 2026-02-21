"""Evaluator for G-15_grid_avoid_obstacles_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-15_grid_avoid_obstacles_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class GridAvoidObstaclesEvaluator(BaseEvaluator):
    """
    G-15: Grid avoid obstacles evaluator.
    
    CRITICAL RULES:
    1. Yellow dot (agent) must move step by step from blue grid to red grid
    2. Agent must finally reach the red grid (endpoint)
    3. Grid colors (blue start, red end) must NOT change
    4. Avoid black X obstacles
    """
    
    TASK_WEIGHTS = {
        'completion': 0.45,       # Agent reaches red endpoint
        'grid_preserved': 0.30,   # Grid colors unchanged
        'avoidance': 0.15,        # No collision with obstacles
        'movement': 0.10          # Step by step movement
    }
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect yellow circular agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    def _detect_obstacles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black X obstacles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Black obstacles
        black_mask = (gray < 50).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 5000:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            obstacles.append((cx, cy))
        
        return obstacles
    
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
    
    def _get_grid_cell_colors(self, frame: np.ndarray, grid_size: int = 10) -> Dict[Tuple[int, int], str]:
        """Get the dominant color for each grid cell."""
        h, w = frame.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        cell_colors = {}
        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * cell_h + 10
                y2 = (row + 1) * cell_h - 10
                x1 = col * cell_w + 10
                x2 = (col + 1) * cell_w - 10
                
                if y2 <= y1 or x2 <= x1:
                    cell_colors[(row, col)] = 'white'
                    continue
                
                cell_hsv = hsv[y1:y2, x1:x2]
                sat = cell_hsv[:, :, 1].flatten()
                hue = cell_hsv[:, :, 0].flatten()
                
                # Check if cell is colored (saturated)
                sat_mask = sat > 80
                if np.sum(sat_mask) > 100:
                    dom_hue = np.median(hue[sat_mask])
                    # Classify color
                    if 100 <= dom_hue <= 130:
                        color = 'blue'
                    elif dom_hue <= 10 or dom_hue >= 160:
                        color = 'red'
                    elif 20 <= dom_hue <= 35:
                        color = 'yellow'
                    elif 35 <= dom_hue <= 85:
                        color = 'green'
                    else:
                        color = 'other'
                    cell_colors[(row, col)] = color
                else:
                    cell_colors[(row, col)] = 'white'
        
        return cell_colors
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid obstacle avoidance task.
        
        CRITICAL RULES:
        1. Agent must reach red endpoint
        2. Grid cell colors must NOT change (check each cell)
        3. Agent must avoid obstacles
        """
        if len(video_frames) < 1 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # CRITICAL: Check if grid cell colors are preserved
        first_colors = self._get_grid_cell_colors(first_frame)
        final_colors = self._get_grid_cell_colors(last_frame)
        
        # Count cells that changed color
        # Key rule: blue and red cells should not change to other colors
        # Yellow (agent) can move around
        changed_cells = 0
        total_grid_cells = 0
        
        for key in first_colors:
            first_color = first_colors[key]
            final_color = final_colors.get(key, 'white')
            
            # Count all colored cells
            if first_color in ['blue', 'red']:
                total_grid_cells += 1
                # Blue/red cells should remain blue/red (or be covered by yellow agent)
                if first_color != final_color and final_color != 'yellow':
                    changed_cells += 1
            
            # Check if a cell that was white/yellow now has a different grid color
            # (This could happen if grid structure changed)
            if first_color in ['white', 'yellow']:
                # If final is blue/red, check if it's a valid reveal (agent moved away from start)
                # For simplicity, count any new blue/red as suspicious
                if final_color in ['blue', 'red']:
                    # This is OK if it's the start cell being revealed
                    pass  # Allow this
        
        # Also check total number of blue+red cells
        first_br_count = sum(1 for c in first_colors.values() if c in ['blue', 'red'])
        final_br_count = sum(1 for c in final_colors.values() if c in ['blue', 'red'])
        
        # If significantly more blue/red cells appeared, grid changed
        if final_br_count > first_br_count + 2:  # Allow up to 2 new cells (start revealed + tolerance)
            changed_cells += (final_br_count - first_br_count - 2)
        
        # Grid preservation score - STRICTER
        if changed_cells > 0:
            scores['grid_preserved'] = 0.0
            scores['completion'] = 0.0
            scores['avoidance'] = 0.0
            scores['movement'] = 0.0
            self._last_task_details = scores
            self._last_task_details['cells_changed'] = changed_cells
            return 0.0
        else:
            scores['grid_preserved'] = 1.0
        
        # Detect obstacles in first frame
        obstacles = self._detect_obstacles(first_frame)
        
        # Track agent positions through video
        agent_positions = []
        for frame in video_frames:
            pos = self._detect_agent(frame)
            if pos is not None:
                agent_positions.append(pos)
        
        # 1. Completion: Check if agent reaches endpoint
        endpoint = self._detect_endpoint(last_frame)
        final_agent = self._detect_agent(last_frame)
        
        if endpoint is not None and final_agent is not None:
            dist = np.sqrt((endpoint[0] - final_agent[0])**2 + (endpoint[1] - final_agent[1])**2)
            # Stricter threshold - must be within 40 pixels
            if dist < 40:
                scores['completion'] = 1.0
            elif dist < 80:
                scores['completion'] = 0.3  # STRICT: Close but not at endpoint
            else:
                scores['completion'] = 0.0  # STRICT: Failed to reach endpoint
        else:
            scores['completion'] = 0.0
        
        # 2. Obstacle avoidance
        if agent_positions and obstacles:
            collision_count = 0
            for pos in agent_positions:
                for obs in obstacles:
                    dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
                    if dist < 30:
                        collision_count += 1
                        break
            scores['avoidance'] = max(0, 1.0 - collision_count / len(agent_positions))
        else:
            scores['avoidance'] = 0.0
        
        # 3. Movement: Check for step-by-step movement
        if len(agent_positions) >= 2:
            # Check if agent moves step by step (not teleporting)
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 100 or dy > 100:  # Too large a jump
                    large_jumps += 1
            scores['movement'] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores['movement'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
