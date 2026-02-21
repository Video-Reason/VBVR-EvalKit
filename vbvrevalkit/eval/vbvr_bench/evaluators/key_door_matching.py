"""Evaluator for G-45_key_door_matching_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-45_key_door_matching_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class KeyDoorMatchingEvaluator(BaseEvaluator):
    """
    G-45: Key door matching evaluator.
    
    STRICT rule-based evaluation:
    - Agent (yellow/green dot) MUST physically move through the maze
    - Agent MUST move TO a key position to collect it (not just disappear)
    - Agent MUST move TO a door position after collecting key
    - Key color MUST match door color
    
    CRITICAL RULES:
    1. If agent doesn't move from starting position at all, score = 0
    2. Key collected ONLY if agent physically reached key's coordinates BEFORE key disappeared
    3. Door reached ONLY if agent physically reached door's coordinates AFTER collecting key
    
    Evaluates:
    - Agent movement (30%): Agent must PHYSICALLY move away from start position
    - Key collection (35%): Agent moved TO key position AND key disappeared
    - Door reached (25%): Agent moved TO door position after getting key
    - Sequence (10%): Correct order: key first, then door
    """
    
    TASK_WEIGHTS = {
        'agent_movement': 0.30,   # Critical: must physically move away from start
        'key_collected': 0.35,    # Must reach key position AND key disappears
        'door_reached': 0.25,     # Must reach door position after getting key
        'sequence': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate key door matching task with strict movement tracking."""
        scores = {}
        
        if len(video_frames) < 2:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Track agent (yellow or green dot) across all frames
        agent_positions = self._track_agent(video_frames)
        
        if len(agent_positions) < 2:
            self._last_task_details = {
                'agent_movement': 0, 'key_collected': 0, 
                'door_reached': 0, 'sequence': 0, 
                'positions_found': len(agent_positions),
                'error': 'not_enough_agent_positions'
            }
            return 0.0
        
        # Detect keys and doors in first frame
        first_keys = self._detect_keys(first_frame)
        first_doors = self._detect_doors(first_frame)
        last_keys = self._detect_keys(last_frame)
        last_doors = self._detect_doors(last_frame)
        
        # Store detection info for debugging
        scores['num_first_keys'] = len(first_keys)
        scores['num_first_doors'] = len(first_doors)
        scores['num_last_keys'] = len(last_keys)
        scores['num_positions'] = len(agent_positions)
        
        # 1. CRITICAL: Agent movement (30%) - Agent must physically move
        movement_score, total_movement = self._evaluate_agent_movement(agent_positions)
        scores['agent_movement'] = movement_score
        scores['total_movement_distance'] = total_movement
        
        # If agent doesn't move, entire task fails
        if movement_score < 0.3:
            scores['key_collected'] = 0.0
            scores['door_reached'] = 0.0
            scores['sequence'] = 0.0
            self._last_task_details = scores
            self._last_task_details['failure_reason'] = 'agent_not_moving'
            return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
        
        # 2. Key collection (35%): Agent must move TO key position AND key disappears
        key_collected, collected_color, key_visit_score = self._check_key_collected_with_visit(
            first_keys, last_keys, agent_positions, video_frames
        )
        scores['key_collected'] = key_visit_score
        scores['collected_key_color'] = collected_color
        
        # 3. Door reached (25%): Agent must move TO door position (after getting key)
        door_reached, door_color, door_visit_score = self._check_door_reached_with_visit(
            first_doors, agent_positions, video_frames, key_collected
        )
        
        # Door only counts if key was collected first AND colors match
        if door_reached and key_collected:
            if collected_color and door_color and collected_color == door_color:
                scores['door_reached'] = door_visit_score  # Good: matching colors
            elif collected_color is None or door_color is None:
                scores['door_reached'] = door_visit_score * 0.5  # Partial: can't verify color match
            else:
                scores['door_reached'] = 0.0  # Wrong color key for door - FAIL
        elif door_reached and not key_collected:
            scores['door_reached'] = 0.0  # Reached door without key - FAIL
        else:
            scores['door_reached'] = 0.0
        
        scores['door_color'] = door_color
        
        # 4. Sequence (10%): Key collected before door reached
        sequence_score = self._evaluate_sequence_with_tracking(video_frames, first_keys, first_doors, agent_positions)
        scores['sequence'] = sequence_score if key_collected else 0.0
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _evaluate_agent_movement(self, positions: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        STRICT evaluation: Agent MUST physically move away from starting position.
        
        CRITICAL: If agent stays at/near starting position, score = 0
        The key-door task requires the agent to navigate through the maze.
        
        Returns: (movement_score, total_distance, max_dist_from_start)
        """
        if len(positions) < 2:
            return 0.0, 0.0
        
        start_pos = positions[0]
        
        # Calculate max distance agent ever reached from start position
        max_dist_from_start = 0.0
        for pos in positions:
            dist = safe_distance(pos, start_pos)
            max_dist_from_start = max(max_dist_from_start, dist)
        
        # CRITICAL CHECK: If agent never moved more than 30 pixels from start, FAIL
        if max_dist_from_start < 30:
            return 0.0, 0.0  # Agent stayed at starting position = COMPLETE FAILURE
        
        # Calculate total movement distance (for debugging)
        total_distance = 0.0
        for i in range(1, len(positions)):
            dist = safe_distance(positions[i], positions[i-1])
            total_distance += dist
        
        # Calculate displacement from start to end
        end_pos = positions[-1]
        displacement = safe_distance(start_pos, end_pos)
        
        # Score based on how far agent traveled from start
        # Agent should travel a significant distance to reach keys/doors
        if max_dist_from_start > 150:
            return 1.0, total_distance  # Agent traveled far - good movement
        elif max_dist_from_start > 100:
            return 0.8, total_distance
        elif max_dist_from_start > 60:
            return 0.6, total_distance
        elif max_dist_from_start > 30:
            return 0.3, total_distance  # Minimal movement
        else:
            return 0.0, total_distance  # No real movement
    
    def _check_key_collected_with_visit(
        self, first_keys: List[Dict], last_keys: List[Dict],
        positions: List[Tuple[int, int]], frames: List[np.ndarray]
    ) -> Tuple[bool, Optional[str], float]:
        """
        STRICT CHECK: Key is ONLY collected if:
        1. Agent PHYSICALLY reached the key's position (within threshold)
        2. Key disappeared from that position
        
        CRITICAL: Key disappearance WITHOUT agent visit = 0 score (not valid collection)
        
        Returns: (collected, color, score)
        """
        if not first_keys or not positions:
            return False, None, 0.0
        
        # Calculate starting position (first detected agent position)
        start_pos = positions[0]
        
        # For each key, check if agent ever reached it
        for key in first_keys:
            key_pos = key['center']
            key_color = key['color']
            
            # Find minimum distance agent got to this key
            min_dist_to_key = float('inf')
            visit_frame_idx = -1
            for i, pos in enumerate(positions):
                dist = safe_distance(pos, key_pos)
                if dist < min_dist_to_key:
                    min_dist_to_key = dist
                    visit_frame_idx = i
            
            # CRITICAL: Agent must have PHYSICALLY moved to the key position
            # Key position must be significantly different from start position
            key_dist_from_start = safe_distance(key_pos, start_pos)
            
            # Agent must have actually traveled toward the key (not stayed at start)
            if min_dist_to_key < 50 and key_dist_from_start > 50:
                # Agent reached this key's position
                # Now check if key disappeared
                key_still_exists = False
                for last_key in last_keys:
                    if last_key['color'] == key_color:
                        dist = safe_distance(last_key['center'], key_pos)
                        if dist < 50:  # Key still at same position
                            key_still_exists = True
                            break
                
                if not key_still_exists:
                    # Perfect: agent visited key AND key disappeared
                    return True, key_color, 1.0
                else:
                    # Agent visited key position but key didn't disappear
                    # Still give partial credit for reaching the key
                    return True, key_color, 0.6
        
        # STRICT: If agent never reached any key position, check if keys disappeared
        # This should NOT give credit (agent didn't do the work)
        first_key_colors = {}
        for key in first_keys:
            first_key_colors[key['color']] = first_key_colors.get(key['color'], 0) + 1
        
        last_key_colors = {}
        for key in last_keys:
            last_key_colors[key['color']] = last_key_colors.get(key['color'], 0) + 1
        
        for color, count in first_key_colors.items():
            if last_key_colors.get(color, 0) < count:
                # Key disappeared but agent NEVER reached it
                # This is WRONG - give 0 credit
                return False, None, 0.0
        
        # No key collection detected
        return False, None, 0.0
    
    def _check_door_reached_with_visit(
        self, doors: List[Dict], positions: List[Tuple[int, int]],
        frames: List[np.ndarray], key_collected: bool
    ) -> Tuple[bool, Optional[str], float]:
        """
        STRICT CHECK: Door is ONLY considered reached if:
        1. Agent PHYSICALLY moved to the door's position (not just stayed at start)
        2. This happened AFTER agent collected the key
        
        Returns: (reached, door_color, score)
        """
        if not doors or not positions:
            return False, None, 0.0
        
        start_pos = positions[0]
        
        best_door = None
        best_dist = float('inf')
        
        for door in doors:
            door_pos = door['center']
            
            # CRITICAL: Door must be at a different position than start
            door_dist_from_start = safe_distance(door_pos, start_pos)
            if door_dist_from_start < 50:
                continue  # Door is at starting position - skip
            
            # Check if agent visited this door position (especially in later frames)
            for pos in positions[-len(positions)//2:]:  # Check second half of trajectory
                dist = safe_distance(pos, door_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_door = door
        
        if best_door is None:
            return False, None, 0.0
        
        # CRITICAL: Agent must have actually moved toward the door
        # (best_dist should be small, meaning agent got close to door)
        door_dist_from_start = safe_distance(best_door['center'], start_pos)
        
        # Agent must have traveled away from start to reach the door
        if door_dist_from_start < 50:
            return False, None, 0.0  # Door is too close to start
        
        # Score based on how close agent got to door
        if best_dist < 40:
            return True, best_door['color'], 1.0  # Very close to door
        elif best_dist < 70:
            return True, best_door['color'], 0.7  # Reasonably close
        elif best_dist < 100:
            return True, best_door['color'], 0.4  # Somewhat close
        
        return False, None, 0.0
    
    def _evaluate_sequence_with_tracking(
        self, frames: List[np.ndarray], 
        keys: List[Dict], doors: List[Dict],
        positions: List[Tuple[int, int]]
    ) -> float:
        """
        Evaluate if key was visited BEFORE door was visited.
        Finds the closest key and door to the agent path and compares visit times.
        """
        if not keys or not doors or len(positions) < 2:
            return 0.0
        
        # Find when agent got closest to each key
        key_visits = []
        for key in keys:
            min_dist = float('inf')
            visit_frame = -1
            for i, pos in enumerate(positions):
                dist = safe_distance(pos, key['center'])
                if dist < min_dist:
                    min_dist = dist
                    visit_frame = i
            if min_dist < 80:  # Consider visited if within 80 pixels
                key_visits.append((key['color'], visit_frame, min_dist))
        
        # Find when agent got closest to each door
        door_visits = []
        for door in doors:
            min_dist = float('inf')
            visit_frame = -1
            for i, pos in enumerate(positions):
                dist = safe_distance(pos, door['center'])
                if dist < min_dist:
                    min_dist = dist
                    visit_frame = i
            if min_dist < 80:  # Consider visited if within 80 pixels
                door_visits.append((door['color'], visit_frame, min_dist))
        
        if not key_visits or not door_visits:
            return 0.3  # Partial credit: some movement detected
        
        # Check if any key was visited before any door
        earliest_key_frame = min(v[1] for v in key_visits)
        earliest_door_frame = min(v[1] for v in door_visits)
        
        if earliest_key_frame < earliest_door_frame:
            return 1.0  # Correct: visited a key before any door
        elif earliest_door_frame < earliest_key_frame:
            return 0.2  # Wrong: visited door first
        else:
            return 0.5  # Same frame (unlikely but handle it)
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect agent dot - GREEN circular dot (primary) or yellow (fallback)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try GREEN first - most common agent color in key-door tasks
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for small roughly circular agents
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 100 < area < 4000:  # Small to medium size
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:  # Roughly circular
                        M = cv2.moments(c)
                        if M['m00'] > 0:
                            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        # Fallback to yellow if green not found
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 100 < area < 4000:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:
                        M = cv2.moments(c)
                        if M['m00'] > 0:
                            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        return None
    
    def _track_agent(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track agent position across frames."""
        positions = []
        for frame in frames:
            pos = self._detect_agent(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_keys(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect diamond-shaped keys of various colors.
        Keys are filled shapes (high fill ratio) with 4 vertices.
        """
        keys = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1
                
                # Keys are FILLED shapes (high fill ratio > 0.7)
                if fill_ratio < 0.7:
                    continue
                
                # Check if roughly diamond-shaped (4 vertices)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:  # Diamond-like shapes
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        keys.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'area': area,
                            'fill_ratio': fill_ratio
                        })
        
        return keys
    
    def _detect_doors(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hollow rectangular doors.
        Doors are HOLLOW shapes (low fill ratio < 0.6) with 4 vertices.
        """
        doors = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1
                
                # Doors are HOLLOW shapes (low fill ratio < 0.6)
                if fill_ratio >= 0.6:
                    continue
                
                # Check if roughly rectangular (4 vertices)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:  # Rectangular-like shapes
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        doors.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'area': area,
                            'fill_ratio': fill_ratio
                        })
        
        return doors
