"""Evaluator for O-39_maze_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-39_maze_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class MazePathfindingEvaluator(BaseEvaluator):
    """
    O-39: Maze pathfinding evaluator.
    
    Rule-based evaluation:
    - Path validity (45%): No wall crossing, continuous path
    - Path completeness (30%): Start to end, all marked
    - Navigation accuracy (20%): Adjacent moves only
    - Element preservation (5%): Maze structure unchanged
    """
    
    TASK_WEIGHTS = {
        'path_validity': 0.45,
        'path_completeness': 0.30,
        'navigation_accuracy': 0.20,
        'element_preservation': 0.05
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
        
        scores['path_validity'] = self._evaluate_path_validity(first_frame, final_frame)
        scores['path_completeness'] = self._evaluate_path_completeness(first_frame, final_frame)
        scores['navigation_accuracy'] = self._evaluate_navigation(first_frame, final_frame)
        scores['element_preservation'] = self._evaluate_preservation(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_path_validity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if path is valid (no wall crossing)."""
        # Detect path markers (orange/yellow)
        path_mask = self._detect_path_markers(final_frame)
        
        # Detect walls (black)
        wall_mask = self._detect_walls(first_frame)
        
        if path_mask is None or wall_mask is None:
            return 0.5
        
        # Check path doesn't cross walls
        overlap = cv2.bitwise_and(path_mask, wall_mask)
        overlap_pixels = np.sum(overlap > 0)
        path_pixels = np.sum(path_mask > 0)
        
        if path_pixels == 0:
            return 0.3
        
        violation_ratio = overlap_pixels / path_pixels
        
        if violation_ratio < 0.05:
            return 1.0
        elif violation_ratio < 0.1:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_path_completeness(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if path is complete from start to end."""
        # Detect start (green) and end (red flag)
        start_pos = self._find_start_position(first_frame)
        end_pos = self._find_end_position(first_frame)
        
        # Detect path markers
        path_mask = self._detect_path_markers(final_frame)
        
        if start_pos is None or end_pos is None or path_mask is None:
            return 0.5
        
        # Check if path reaches start and end
        path_near_start = self._check_path_near_position(path_mask, start_pos)
        path_near_end = self._check_path_near_position(path_mask, end_pos)
        
        if path_near_start and path_near_end:
            return 1.0
        elif path_near_start or path_near_end:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_navigation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if moves are valid (adjacent only)."""
        # Check path continuity
        path_mask = self._detect_path_markers(final_frame)
        
        if path_mask is None:
            return 0.5
        
        # Count connected components (should be 1 for continuous path)
        num_labels, _ = cv2.connectedComponents(path_mask)
        
        if num_labels == 2:  # 1 background + 1 path
            return 1.0
        elif num_labels <= 4:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if maze structure is preserved."""
        # Compare wall regions
        first_walls = self._detect_walls(first_frame)
        final_walls = self._detect_walls(final_frame)
        
        if first_walls is None or final_walls is None:
            return 0.5
        
        # Compare wall preservation
        intersection = np.sum((first_walls > 0) & (final_walls > 0))
        first_total = np.sum(first_walls > 0)
        
        if first_total == 0:
            return 0.5
        
        preservation_ratio = intersection / first_total
        
        if preservation_ratio > 0.9:
            return 1.0
        elif preservation_ratio > 0.7:
            return 0.7
        else:
            return 0.4
    
    def _detect_path_markers(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect orange/yellow path markers."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        return mask
    
    def _detect_walls(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect maze walls (black/dark)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, walls = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        return walls
    
    def _find_start_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find green start marker."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _find_end_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find red end marker (flag)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _check_path_near_position(self, path_mask: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if path reaches near a position."""
        x, y = pos
        h, w = path_mask.shape
        
        # Check in a radius around position
        radius = 30
        y_min = max(0, y - radius)
        y_max = min(h, y + radius)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius)
        
        region = path_mask[y_min:y_max, x_min:x_max]
        return np.sum(region > 0) > 50
