"""Evaluator for G-47_multiple_keys_for_one_door_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-47_multiple_keys_for_one_door_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class MultipleKeysForOneDoorEvaluator(BaseEvaluator):
    """
    G-47: Multi-key collection maze evaluator.
    
    Rule-based evaluation:
    - Key collection completeness (30%): Agent must collect 2 keys (colored objects disappear)
    - Visit order optimization (25%): Keys collected before reaching door
    - Path efficiency (25%): Minimal backtracking
    - Movement legality (20%): No wall crossing, valid grid moves
    """
    
    TASK_WEIGHTS = {
        'key_collection': 0.30,
        'order_optimization': 0.25,
        'path_efficiency': 0.25,
        'movement_legality': 0.20
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Key collection completeness (30%)
        # Rule: Count colored key objects in first vs final frame
        scores['key_collection'] = self._evaluate_key_collection(
            first_frame, final_frame
        )
        
        # 2. Visit order optimization (25%)
        # Rule: Track agent reaching key positions before door
        scores['order_optimization'] = self._evaluate_visit_order(video_frames)
        
        # 3. Path efficiency (25%)
        # Rule: Check for minimal backtracking
        scores['path_efficiency'] = self._evaluate_path_efficiency(video_frames)
        
        # 4. Movement legality (20%)
        # Rule: Check for wall crossing and valid moves
        scores['movement_legality'] = self._evaluate_movement_legality(
            video_frames, first_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_key_collection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Count keys collected (colored objects that disappear)."""
        # Detect colored objects (keys) - typically yellow, blue, red
        initial_keys = self._count_key_objects(first_frame)
        final_keys = self._count_key_objects(final_frame)
        
        # Expected: 2 keys should be collected (disappear)
        keys_collected = initial_keys - final_keys
        
        # Check if agent reached the door (green agent at door position)
        agent_at_door = self._check_agent_at_door(final_frame)
        
        if keys_collected >= 2 and agent_at_door:
            return 1.0
        elif keys_collected >= 2:
            return 0.8  # Keys collected but didn't reach door
        elif keys_collected == 1:
            return 0.5  # Partial collection
        else:
            return 0.0
    
    def _evaluate_visit_order(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Check if keys are collected before reaching door."""
        door_reached_frame = None
        key_collection_frames = []
        
        for i, frame in enumerate(video_frames):
            # Track when agent reaches door area
            if self._check_agent_at_door(frame) and door_reached_frame is None:
                door_reached_frame = i
            
            # Track key collection events (reduction in key count)
            if i > 0:
                prev_keys = self._count_key_objects(video_frames[i-1])
                curr_keys = self._count_key_objects(frame)
                if curr_keys < prev_keys:
                    key_collection_frames.append(i)
        
        # All keys should be collected before or at door reach
        if door_reached_frame is None:
            return 0.3  # Never reached door
        
        if len(key_collection_frames) >= 2:
            if all(f < door_reached_frame or f == door_reached_frame for f in key_collection_frames):
                return 1.0
            return 0.6
        elif len(key_collection_frames) == 1:
            return 0.4
        return 0.2
    
    def _evaluate_path_efficiency(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Check for minimal backtracking."""
        positions = []
        for frame in video_frames[::max(1, len(video_frames)//30)]:
            pos = self._detect_agent_position(frame)
            if pos:
                positions.append(pos)
        
        if len(positions) < 3:
            return 0.5
        
        # Count backtracking (returning to same cell)
        visited = set()
        revisits = 0
        for pos in positions:
            cell = (pos[0] // 30, pos[1] // 30)  # Grid cell approximation
            if cell in visited:
                revisits += 1
            visited.add(cell)
        
        revisit_ratio = revisits / len(positions)
        
        if revisit_ratio < 0.1:
            return 1.0
        elif revisit_ratio < 0.2:
            return 0.8
        elif revisit_ratio < 0.3:
            return 0.6
        else:
            return max(0.2, 1.0 - revisit_ratio)
    
    def _evaluate_movement_legality(
        self, 
        video_frames: List[np.ndarray],
        first_frame: np.ndarray
    ) -> float:
        """Rule-based: Check for wall crossing (illegal moves)."""
        # Detect walls from first frame (black pixels)
        wall_mask = self._detect_walls(first_frame)
        
        if wall_mask is None:
            return 0.5
        
        # Track agent movement and check for wall collision
        violations = 0
        total_moves = 0
        
        for i in range(1, min(len(video_frames), 50)):
            agent_pos = self._detect_agent_position(video_frames[i])
            if agent_pos is not None:
                x, y = agent_pos
                h, w = wall_mask.shape
                if 0 <= y < h and 0 <= x < w:
                    if wall_mask[y, x] > 0:
                        violations += 1
                total_moves += 1
        
        if total_moves == 0:
            return 0.5
        
        violation_rate = violations / total_moves
        
        if violation_rate == 0:
            return 1.0
        elif violation_rate <= 0.05:
            return 0.7
        elif violation_rate <= 0.1:
            return 0.4
        else:
            return 0.1
    
    def _count_key_objects(self, frame: np.ndarray) -> int:
        """Count colored key objects (orange, cyan, yellow, red, blue markers)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Orange keys (HSV hue ~15)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Yellow keys (HSV hue ~30)
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Cyan keys (HSV hue ~90)
        lower_cyan = np.array([80, 100, 100])
        upper_cyan = np.array([100, 255, 255])
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Red keys
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Blue keys
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        combined = orange_mask | yellow_mask | cyan_mask | red_mask | blue_mask
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count significant contours (keys)
        keys = sum(1 for cnt in contours if 200 < cv2.contourArea(cnt) < 5000)
        return keys
    
    def _check_agent_at_door(self, frame: np.ndarray) -> bool:
        """Check if green agent is at door position."""
        agent_pos = self._detect_agent_position(frame)
        if agent_pos is None:
            return False
        
        # Door is typically at a specific location (bottom right or marked differently)
        # Check for brown/orange door color near agent
        x, y = agent_pos
        h, w = frame.shape[:2]
        
        # Sample region around agent
        region = frame[max(0, y-40):min(h, y+40), max(0, x-40):min(w, x+40)]
        if region.size == 0:
            return False
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Door color (brown/orange)
        lower_door = np.array([10, 50, 50])
        upper_door = np.array([25, 255, 255])
        door_mask = cv2.inRange(hsv, lower_door, upper_door)
        
        door_pixels = np.sum(door_mask > 0)
        return door_pixels > 100
    
    def _detect_walls(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect wall regions (black/dark pixels) in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, wall_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        return wall_mask
    
    def _detect_agent_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect green agent position in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Green color range
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None
