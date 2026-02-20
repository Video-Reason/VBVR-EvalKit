"""
Specific evaluators for Hidden_40 tasks.
These evaluators implement PURE RULE-BASED scoring for 10 specific tasks.
NO SSIM or image comparison with ground truth is used.
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


class SelectNextFigureAlternatingEvaluator(BaseEvaluator):
    """
    G-135: Select next figure in small-big alternating sequence.
    
    Rule-based evaluation:
    - Pattern recognition (40%): Identify "small-big-small" pattern in existing sequence
    - Selection correctness (35%): Next should be "big" - largest candidate selected
    - Marking accuracy (15%): Red circle marks exactly one figure
    - Animation quality (10%): Circle appears with smooth expansion
    """
    
    TASK_WEIGHTS = {
        'pattern_recognition': 0.40,
        'selection_correctness': 0.35,
        'marking_accuracy': 0.15,
        'animation_quality': 0.10
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
        
        # 1. Pattern recognition (40%)
        # Rule: Check if pattern analysis shows alternating sizes
        scores['pattern_recognition'] = self._evaluate_pattern_recognition(
            first_frame, final_frame
        )
        
        # 2. Selection correctness (35%)
        # Rule: Red circle should mark the largest candidate figure
        scores['selection_correctness'] = self._evaluate_selection(
            first_frame, final_frame
        )
        
        # 3. Marking accuracy (15%)
        # Rule: Exactly one red circle marking
        scores['marking_accuracy'] = self._evaluate_marking(final_frame, first_frame)
        
        # 4. Animation quality (10%)
        # Rule: Circle should expand smoothly
        scores['animation_quality'] = self._evaluate_animation(video_frames)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_pattern_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if alternating pattern is understood."""
        # Detect existing shapes
        shapes = self._detect_shapes_with_sizes(first_frame)
        
        if len(shapes) < 3:
            return 0.5
        
        # Only consider sequence shapes (top half) for pattern recognition
        h = first_frame.shape[0]
        sequence_shapes = [s for s in shapes if s[1] < h // 2]
        
        if len(sequence_shapes) < 3:
            return 0.5
        
        # Sort by x-position (left to right sequence)
        shapes_sorted = sorted(sequence_shapes, key=lambda s: s[0])
        sizes = [s[2] for s in shapes_sorted]
        
        if len(sizes) < 3:
            return 0.5
        
        # Check for alternating pattern: small-big-small or big-small-big
        is_alternating = True
        for i in range(len(sizes) - 2):
            if sizes[i] < sizes[i+1] > sizes[i+2] or sizes[i] > sizes[i+1] < sizes[i+2]:
                continue
            else:
                is_alternating = False
                break
        
        if is_alternating:
            return 1.0
        return 0.5
    
    def _evaluate_selection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the correct candidate is marked based on pattern."""
        # Detect red circle marking (new markings only)
        circles = self._detect_red_circles(final_frame, first_frame)
        
        if len(circles) == 0:
            return 0.0
        
        marked_pos = circles[0][:2]  # Get position of marked item
        
        # Detect all shapes
        all_shapes = self._detect_shapes_with_sizes(first_frame)
        
        if len(all_shapes) == 0:
            return 0.5
        
        # Separate into sequence (top half) and candidates (bottom half)
        h = first_frame.shape[0]
        sequence_shapes = sorted([s for s in all_shapes if s[1] < h // 2], key=lambda s: s[0])
        candidate_shapes = [s for s in all_shapes if s[1] >= h // 2]
        
        if len(candidate_shapes) == 0:
            return 0.5
        
        # Find which candidate is marked
        marked_candidate = None
        min_dist = float('inf')
        for cand in candidate_shapes:
            dist = np.sqrt((cand[0] - marked_pos[0])**2 + (cand[1] - marked_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                marked_candidate = cand
        
        if marked_candidate is None or min_dist > 100:
            return 0.3
        
        # Determine expected size based on alternating pattern
        if len(sequence_shapes) >= 2:
            sizes = [s[2] for s in sequence_shapes]
            
            # Calculate threshold as mean of min and max sizes
            min_size = min(sizes)
            max_size = max(sizes)
            threshold = (min_size + max_size) / 2
            
            # Classify each shape as small or big
            pattern = ['small' if s <= threshold else 'big' for s in sizes]
            last_type = pattern[-1]
            
            # Determine expected next type
            expected_type = 'big' if last_type == 'small' else 'small'
            
            # Check if marked candidate matches expected type
            marked_type = 'small' if marked_candidate[2] <= threshold else 'big'
            
            if marked_type == expected_type:
                return 1.0
            else:
                return 0.5
        
        # Fallback: check if marked candidate is among the larger ones
        candidate_sizes = [c[2] for c in candidate_shapes]
        if marked_candidate[2] >= np.median(candidate_sizes):
            return 0.8
        return 0.5
    
    def _evaluate_marking(
        self, 
        final_frame: np.ndarray, 
        first_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Evaluate red circle marking quality."""
        circles = self._detect_red_circles(final_frame, first_frame)
        
        if len(circles) == 0:
            return 0.0
        elif len(circles) == 1:
            return 1.0  # Correct number of markings
        else:
            return max(0.3, 1.0 - 0.2 * (len(circles) - 1))  # Penalty for multiple
    
    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 3:
            return 0.5
        
        # Check for smooth circle expansion
        circle_sizes = []
        for frame in video_frames[len(video_frames)//2:]:
            circles = self._detect_red_circles(frame)
            if circles:
                circle_sizes.append(circles[0][2] if len(circles[0]) > 2 else 30)
        
        if len(circle_sizes) < 2:
            return 0.5
        
        # Check if sizes increase smoothly
        increases = sum(1 for i in range(1, len(circle_sizes)) 
                       if circle_sizes[i] >= circle_sizes[i-1] * 0.95)
        smoothness = increases / (len(circle_sizes) - 1)
        
        return smoothness
    
    def _detect_shapes_with_sizes(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes with their (x, y, area)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Lower threshold to detect smaller shapes
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, area))
        
        return shapes
    
    def _detect_candidate_shapes(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect candidate shapes (typically in bottom portion)."""
        h, w = frame.shape[:2]
        
        # Focus on bottom half or right portion where candidates usually are
        bottom_region = frame[h//2:, :]
        
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"]) + h//2  # Adjust for cropped region
                    candidates.append((cx, cy, area))
        
        return candidates
    
    def _detect_red_circles(
        self, 
        frame: np.ndarray, 
        first_frame: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int, int]]:
        """Detect red circles in the frame (new markings only if first_frame provided)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # If first_frame provided, only detect NEW red regions (markings)
        if first_frame is not None:
            hsv_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
            mask_first = cv2.inRange(hsv_first, lower_red1, upper_red1) | cv2.inRange(hsv_first, lower_red2, upper_red2)
            # Only keep red regions that are new (not in first frame)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_first))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circles.append((int(x), int(y), int(radius)))
        
        return circles


class LocatePointInOverlappingAreaEvaluator(BaseEvaluator):
    """
    G-136: Locate points in overlapping region of two shapes.
    
    Rule-based evaluation:
    - Overlap identification (30%): Detect intersection region correctly
    - Point containment (35%): Mark points actually in overlap region
    - Marking completeness (20%): All overlap points marked (recall)
    - Marking accuracy (15%): No false positives (precision)
    """
    
    TASK_WEIGHTS = {
        'overlap_identification': 0.30,
        'point_containment': 0.35,
        'marking_completeness': 0.20,
        'marking_accuracy': 0.15
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
        
        # Detect overlap region from first frame
        overlap_region = self._detect_overlap_region(first_frame)
        
        # Detect all points from first frame
        all_points = self._detect_points(first_frame)
        
        # Detect marked circles in final frame
        marked_circles = self._detect_circles(final_frame)
        
        # 1. Overlap identification (30%)
        scores['overlap_identification'] = self._evaluate_overlap_understanding(
            overlap_region
        )
        
        # 2. Point containment (35%)
        scores['point_containment'] = self._evaluate_point_marking(
            marked_circles, overlap_region, all_points
        )
        
        # 3. Marking completeness (20%) - compare with GT marking count
        gt_marked_circles = self._detect_circles(gt_final_frame) if gt_final_frame is not None else []
        scores['marking_completeness'] = self._evaluate_completeness(
            marked_circles, overlap_region, all_points, gt_marked_circles
        )
        
        # 4. Marking accuracy (15%)
        scores['marking_accuracy'] = self._evaluate_marking_accuracy(
            marked_circles, overlap_region
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_overlap_understanding(self, overlap_region: Optional[np.ndarray]) -> float:
        """Rule-based: Check if overlap region is detected."""
        if overlap_region is None:
            return 0.3
        
        overlap_area = np.sum(overlap_region > 0)
        
        if overlap_area > 1000:  # Significant overlap detected
            return 1.0
        elif overlap_area > 500:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_point_marking(
        self, 
        marked_circles: List[Tuple[int, int]],
        overlap_region: Optional[np.ndarray],
        all_points: List[Tuple[int, int]]
    ) -> float:
        """Rule-based: Check if marked points are in overlap region."""
        if len(marked_circles) == 0:
            return 0.0
        
        if overlap_region is None:
            return 0.5
        
        # Count how many marked points are in overlap
        correct_marks = 0
        for circle in marked_circles:
            x, y = circle
            h, w = overlap_region.shape
            if 0 <= y < h and 0 <= x < w:
                if overlap_region[y, x] > 0:
                    correct_marks += 1
        
        if len(marked_circles) == 0:
            return 0.0
        
        return correct_marks / len(marked_circles)
    
    def _evaluate_completeness(
        self, 
        marked_circles: List[Tuple[int, int]],
        overlap_region: Optional[np.ndarray],
        all_points: List[Tuple[int, int]],
        gt_marked_circles: List[Tuple[int, int]] = None
    ) -> float:
        """Rule-based: Check marking completeness compared to GT."""
        if len(marked_circles) == 0:
            return 0.0
        
        # If GT markings available, compare with GT marking positions
        if gt_marked_circles and len(gt_marked_circles) > 0:
            # Count how many GT markings are matched by generated markings
            matched = 0
            for gt_circle in gt_marked_circles:
                for gen_circle in marked_circles:
                    dist = np.sqrt((gen_circle[0] - gt_circle[0])**2 + (gen_circle[1] - gt_circle[1])**2)
                    if dist < 50:  # Within matching distance
                        matched += 1
                        break
            
            # Score based on matching ratio
            recall = matched / len(gt_marked_circles)
            precision = matched / len(marked_circles) if len(marked_circles) > 0 else 0
            
            # F1 score
            if recall + precision > 0:
                return 2 * recall * precision / (recall + precision)
            return 0.0
        
        # Fallback: check if marking count is reasonable
        if overlap_region is None:
            return 0.5
        
        # Find points in overlap region
        points_in_overlap = []
        for point in all_points:
            x, y = point
            h, w = overlap_region.shape
            if 0 <= y < h and 0 <= x < w:
                if overlap_region[y, x] > 0:
                    points_in_overlap.append(point)
        
        if len(points_in_overlap) == 0:
            return 1.0 if len(marked_circles) == 0 else 0.5
        
        # Score based on marking at least some points
        marked_count = 0
        for point in points_in_overlap:
            for circle in marked_circles:
                dist = np.sqrt((circle[0] - point[0])**2 + (circle[1] - point[1])**2)
                if dist < 40:
                    marked_count += 1
                    break
        
        # If at least some points are marked correctly, give good score
        if marked_count >= len(marked_circles) * 0.8:
            return 1.0
        elif marked_count > 0:
            return 0.7
        return 0.3
    
    def _evaluate_marking_accuracy(
        self, 
        marked_circles: List[Tuple[int, int]],
        overlap_region: Optional[np.ndarray]
    ) -> float:
        """Rule-based: Check for false positives (precision)."""
        if len(marked_circles) == 0:
            return 0.0
        
        if overlap_region is None:
            return 0.5
        
        # Count marks that are in overlap (true positives)
        true_positives = 0
        for circle in marked_circles:
            x, y = circle
            h, w = overlap_region.shape
            if 0 <= y < h and 0 <= x < w:
                if overlap_region[y, x] > 0:
                    true_positives += 1
        
        return true_positives / len(marked_circles)
    
    def _detect_overlap_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect overlapping region of two shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find all non-white shapes
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find large shapes (potential overlapping regions)
        large_shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # Large shape threshold
                large_shapes.append((cnt, area))
        
        if large_shapes:
            # The largest shape is likely the overlap region itself
            # (when two shapes overlap, they form a single connected region)
            largest = max(large_shapes, key=lambda x: x[1])
            overlap = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(overlap, [largest[0]], -1, 255, -1)
            return overlap
        
        # Fallback: try color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect different colored shapes and find their intersection
        color_ranges = [
            ([100, 50, 50], [130, 255, 255]),  # Blue
            ([15, 50, 50], [45, 255, 255]),     # Yellow/Orange
            ([140, 50, 50], [170, 255, 255]),   # Magenta/Pink
            ([35, 50, 50], [85, 255, 255]),     # Green
        ]
        
        masks = []
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask) > 1000:
                masks.append(mask)
        
        # Find overlap between any two color masks
        if len(masks) >= 2:
            overlap = cv2.bitwise_and(masks[0], masks[1])
            if np.sum(overlap) > 100:
                return overlap
        
        return None
    
    def _detect_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect small point/dot markers in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 500:  # Small dot size range
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
        
        return points
    
    def _detect_circles(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect marking circles in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red circles
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    circles.append((cx, cy))
        
        return circles


class LocateTopmostFigureEvaluator(BaseEvaluator):
    """
    G-140: Locate topmost (unobscured) figure in overlapping shapes.
    
    Rule-based evaluation:
    - Z-order identification (45%): Outline the topmost (least occluded) shape
    - Outline accuracy (30%): Red outline follows shape boundary
    - Marking uniqueness (15%): Only one shape marked
    - Visual clarity (10%): Clear visible outline
    """
    
    TASK_WEIGHTS = {
        'z_order_identification': 0.45,
        'outline_accuracy': 0.30,
        'marking_uniqueness': 0.15,
        'visual_clarity': 0.10
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
        
        # 1. Z-order identification (45%)
        scores['z_order_identification'] = self._evaluate_z_order(
            first_frame, final_frame
        )
        
        # 2. Outline accuracy (30%)
        scores['outline_accuracy'] = self._evaluate_outline(final_frame)
        
        # 3. Marking uniqueness (15%)
        scores['marking_uniqueness'] = self._evaluate_uniqueness(final_frame)
        
        # 4. Visual clarity (10%)
        scores['visual_clarity'] = self._evaluate_clarity(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_z_order(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the topmost (least occluded) shape is outlined."""
        # Detect the topmost shape from first frame
        topmost_center = self._find_topmost_shape_center(first_frame)
        
        # Detect outline position in final frame
        outline = self._detect_outline(final_frame)
        
        if outline is None:
            return 0.0
        
        outline_center = self._get_contour_center(outline)
        
        if topmost_center is None or outline_center is None:
            return 0.3
        
        # Check if outline is around topmost shape
        dist = np.sqrt((outline_center[0] - topmost_center[0])**2 + 
                      (outline_center[1] - topmost_center[1])**2)
        
        if dist < 50:
            return 1.0
        elif dist < 100:
            return 0.7
        elif dist < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_outline(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate outline drawing quality."""
        outline = self._detect_outline(final_frame)
        
        if outline is None:
            return 0.0
        
        # Check if outline forms a closed shape
        perimeter = cv2.arcLength(outline, True)
        area = cv2.contourArea(outline)
        
        if perimeter > 100 and area > 500:
            # Check circularity (how well-formed the outline is)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            if circularity > 0.1:  # Reasonable shape
                return 1.0
            return 0.7
        elif perimeter > 50:
            return 0.5
        return 0.3
    
    def _evaluate_uniqueness(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if only one shape is outlined."""
        outlines = self._detect_all_outlines(final_frame)
        
        if len(outlines) == 0:
            return 0.0
        elif len(outlines) == 1:
            return 1.0
        else:
            return max(0.3, 1.0 - 0.3 * (len(outlines) - 1))
    
    def _evaluate_clarity(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate outline visual clarity."""
        outline = self._detect_outline(final_frame)
        if outline is None:
            return 0.0
        
        perimeter = cv2.arcLength(outline, True)
        if perimeter > 150:
            return 1.0
        elif perimeter > 100:
            return 0.8
        elif perimeter > 50:
            return 0.5
        return perimeter / 100
    
    def _find_topmost_shape_center(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the center of the topmost (least occluded) shape."""
        # Convert to grayscale and find shapes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find shape boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # The topmost shape typically has the most visible edge pixels
        # Find contour with most edge density
        best_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None
    
    def _detect_outline(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the main red outline in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _detect_all_outlines(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect all red outlines in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return [c for c in contours if cv2.contourArea(c) > 100]
    
    def _get_contour_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None


class IdentifyUniqueFigureEvaluator(BaseEvaluator):
    """
    G-147: Identify unique figure in uniform set.
    
    Rule-based evaluation:
    - Shape recognition (40%): Find the one shape that differs from others
    - Marking precision (35%): Red circle accurately marks the unique figure
    - Marking quality (15%): Circle color, size, line width
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'shape_recognition': 0.40,
        'marking_precision': 0.35,
        'marking_quality': 0.15,
        'scene_preservation': 0.10
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
        
        # 1. Shape recognition (40%)
        scores['shape_recognition'] = self._evaluate_recognition(
            first_frame, final_frame
        )
        
        # 2. Marking precision (35%)
        scores['marking_precision'] = self._evaluate_marking_position(final_frame)
        
        # 3. Marking quality (15%)
        scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if unique shape is identified."""
        # Find the unique shape from first frame
        unique_shape_center = self._find_unique_shape(first_frame)
        
        # Detect marking in final frame
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if unique_shape_center is None:
            # Can't verify, give partial credit
            return 0.5
        
        # Check if circle marks the unique shape
        dist = np.sqrt((circle[0] - unique_shape_center[0])**2 + 
                      (circle[1] - unique_shape_center[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle position accuracy."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in a reasonable position (not at edges)
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 50 < x < w - 50 and 50 < y < h - 50:
            return 1.0
        elif 20 < x < w - 20 and 20 < y < h - 20:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle quality."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Check if circle is reasonable size
        if 20 < r < 150:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Check color (should be red)
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        roi_y = max(0, y - r - 10)
        roi_x = max(0, x - r - 10)
        h, w = final_frame.shape[:2]
        roi = hsv[roi_y:min(h, y+r+10), roi_x:min(w, x+r+10)]
        
        if roi.size > 0:
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask = cv2.inRange(roi, lower_red1, upper_red1) | cv2.inRange(roi, lower_red2, upper_red2)
            red_ratio = np.sum(mask > 0) / max(1, mask.size)
            color_score = min(1.0, red_ratio * 10)
        else:
            color_score = 0.5
        
        return 0.6 * size_score + 0.4 * color_score
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original shapes are preserved."""
        # Count shapes in first frame
        first_shapes = self._count_shapes(first_frame)
        
        # Count shapes in final frame (excluding red marking)
        final_shapes = self._count_shapes_excluding_red(final_frame)
        
        if first_shapes == 0:
            return 0.5
        
        # Shapes should be preserved
        if abs(first_shapes - final_shapes) <= 1:
            return 1.0
        elif abs(first_shapes - final_shapes) <= 2:
            return 0.7
        else:
            return 0.4
    
    def _find_unique_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the unique shape that differs from others."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 3:
            return None
        
        # Analyze shape features
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                # Compute shape descriptor (approximate vertices)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_vertices = len(approx)
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, num_vertices, cv2.contourArea(cnt)))
        
        if len(shapes) < 3:
            return None
        
        # Find the outlier (different vertex count or significantly different area)
        vertex_counts = [s[2] for s in shapes]
        areas = [s[3] for s in shapes]
        
        # Check for vertex count outlier
        from collections import Counter
        vertex_counter = Counter(vertex_counts)
        
        for shape in shapes:
            if vertex_counter[shape[2]] == 1:  # Only one shape with this vertex count
                return (shape[0], shape[1])
        
        # Check for area outlier
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        for shape in shapes:
            if abs(shape[3] - mean_area) > 2 * std_area:
                return (shape[0], shape[1])
        
        return None
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count number of shapes in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 200)
    
    def _count_shapes_excluding_red(self, frame: np.ndarray) -> int:
        """Count shapes excluding red marking."""
        # Mask out red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Remove red from frame
        frame_no_red = frame.copy()
        frame_no_red[red_mask > 0] = [255, 255, 255]
        
        return self._count_shapes(frame_no_red)
    
    def _detect_marking_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the marking circle (red)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(largest)
            return (int(x), int(y), int(r))
        
        return None


class CircleLargestNumericalValueEvaluator(BaseEvaluator):
    """
    G-160: Circle the largest numerical value.
    
    Rule-based evaluation:
    - Numerical identification (40%): Circle marks the position of largest number
    - Circle position accuracy (30%): Circle center aligns with number
    - Circle style (20%): Red color, appropriate size
    - Animation quality (10%): Smooth expansion
    """
    
    TASK_WEIGHTS = {
        'numerical_identification': 0.40,
        'circle_position': 0.30,
        'circle_style': 0.20,
        'animation_quality': 0.10
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
        
        # 1. Numerical identification (40%)
        scores['numerical_identification'] = self._evaluate_number_selection(
            first_frame, final_frame
        )
        
        # 2. Circle position (30%)
        scores['circle_position'] = self._evaluate_circle_position(final_frame)
        
        # 3. Circle style (20%)
        scores['circle_style'] = self._evaluate_circle_style(final_frame)
        
        # 4. Animation quality (10%)
        scores['animation_quality'] = self._evaluate_animation_quality(video_frames)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_number_selection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the largest number is circled."""
        # Detect text regions (numbers) from first frame
        number_regions = self._detect_number_regions(first_frame)
        
        # Detect red circle in final frame
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if len(number_regions) == 0:
            return 0.5  # Can't verify, partial credit
        
        # Find which number region is circled
        circled_region = None
        min_dist = float('inf')
        for region in number_regions:
            dist = np.sqrt((region[0] - circle[0])**2 + (region[1] - circle[1])**2)
            if dist < min_dist:
                min_dist = dist
                circled_region = region
        
        if circled_region is None or min_dist > 100:
            return 0.3
        
        # The largest number should have the darkest/largest text region
        # (assuming larger digit values = visually larger or more prominent)
        largest_region = max(number_regions, key=lambda r: r[2])  # r[2] is area
        
        if circled_region[2] == largest_region[2]:
            return 1.0
        elif circled_region[2] >= largest_region[2] * 0.7:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_circle_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle position accuracy."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in reasonable position (not at edges)
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_circle_style(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle style (color, size)."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Size check
        if 30 < r < 200:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Color check
        color_score = self._check_red_color(final_frame, x, y, r)
        
        return 0.5 * size_score + 0.5 * color_score
    
    def _evaluate_animation_quality(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 5:
            return 0.5
        
        radii = []
        for frame in video_frames[len(video_frames)//3:]:
            circle = self._detect_red_circle(frame)
            if circle:
                radii.append(circle[2])
        
        if len(radii) < 2:
            return 0.5
        
        # Check for smooth increase
        smooth_count = sum(1 for i in range(1, len(radii)) if radii[i] >= radii[i-1] * 0.95)
        smoothness = smooth_count / (len(radii) - 1)
        
        return smoothness
    
    def _detect_number_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect text/number regions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 10000:  # Text size range
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    regions.append((cx, cy, area))
        
        return regions
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
    
    def _check_red_color(self, frame: np.ndarray, x: int, y: int, r: int) -> float:
        """Check if the circle is red colored."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r + 5, 255, 10)
        
        roi_hsv = hsv[mask > 0]
        if len(roi_hsv) == 0:
            return 0.5
        
        red_count = sum(1 for pixel in roi_hsv if pixel[0] < 10 or pixel[0] > 160)
        return red_count / len(roi_hsv)


class MarkSecondLargestShapeEvaluator(BaseEvaluator):
    """
    G-161: Mark the second largest shape.
    
    Rule-based evaluation:
    - Size recognition (40%): Identify second largest correctly
    - Marking precision (35%): Border accurately marks target
    - Marking quality (15%): Border style and color
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'size_recognition': 0.40,
        'marking_precision': 0.35,
        'marking_quality': 0.15,
        'scene_preservation': 0.10
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
        
        # CRITICAL CHECK: Only ONE marking should exist
        marking_count = self._count_red_markings(final_frame)
        if marking_count == 0:
            # No marking at all - task failed
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': 'no_marking_found'
            }
            return 0.0
        elif marking_count > 1:
            # Multiple markings - violation of "only one mark" rule
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': f'multiple_markings_found: {marking_count}'
            }
            return 0.0
        
        # CRITICAL CHECK: No new objects should be generated
        first_shapes = self._detect_shapes_with_area(first_frame)
        final_shapes_no_red = self._detect_shapes_without_red(final_frame)
        
        if len(final_shapes_no_red) > len(first_shapes):
            # New objects generated - violation
            self._last_task_details = {
                'size_recognition': 0.0,
                'marking_precision': 0.0,
                'marking_quality': 0.0,
                'scene_preservation': 0.0,
                'error': f'new_objects_generated: first={len(first_shapes)}, final={len(final_shapes_no_red)}'
            }
            return 0.0
        
        # 1. Size recognition (40%) - CRITICAL: Is the CORRECT shape marked?
        size_recognition_score = self._evaluate_size_recognition(
            first_frame, final_frame
        )
        scores['size_recognition'] = size_recognition_score
        
        # CRITICAL: If wrong shape is marked, other scores should be penalized
        correct_shape_marked = size_recognition_score > 0.5
        
        # 2. Marking precision (35%) - Only counts if correct shape is marked
        if correct_shape_marked:
            scores['marking_precision'] = self._evaluate_marking_precision(final_frame)
        else:
            scores['marking_precision'] = 0.0  # Wrong shape - no credit for marking
        
        # 3. Marking quality (15%) - Only counts if correct shape is marked
        if correct_shape_marked:
            scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        else:
            scores['marking_quality'] = 0.0  # Wrong shape - no credit
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _count_red_markings(self, frame: np.ndarray) -> int:
        """Count number of red markings in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Count significant red contours (filter noise)
        return len([c for c in contours if cv2.contourArea(c) > 100])
    
    def _detect_shapes_without_red(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes excluding red marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        frame_no_red = frame.copy()
        frame_no_red[red_mask > 0] = [255, 255, 255]
        return self._detect_shapes_with_area(frame_no_red)
    
    def _evaluate_size_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if second largest shape is correctly identified."""
        # Find shapes sorted by area
        shapes = self._detect_shapes_with_area(first_frame)
        
        if len(shapes) < 2:
            return 0.0  # STRICT: Not enough shapes to have "second largest"
        
        # Sort by area (descending)
        sorted_shapes = sorted(shapes, key=lambda s: s[2], reverse=True)
        second_largest_center = (sorted_shapes[1][0], sorted_shapes[1][1])
        
        # Detect red border/marking in final frame
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        marking_center = self._get_contour_center(marking)
        
        if marking_center is None:
            return 0.0  # STRICT: Can't determine marking position
        
        # Check if marking is on second largest
        dist = np.sqrt((marking_center[0] - second_largest_center[0])**2 + 
                      (marking_center[1] - second_largest_center[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.5  # STRICT: Reduced from 0.7
        else:
            return 0.0  # STRICT: Wrong shape marked
    
    def _evaluate_marking_precision(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border marking precision."""
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        # Check if marking forms a proper border
        perimeter = cv2.arcLength(marking, True)
        area = cv2.contourArea(marking)
        
        if perimeter > 100 and area > 500:
            return 1.0
        elif perimeter > 50:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border marking quality."""
        marking = self._detect_red_border(final_frame)
        
        if marking is None:
            return 0.0
        
        perimeter = cv2.arcLength(marking, True)
        if perimeter > 100:
            return 1.0
        else:
            return perimeter / 100
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original shapes are preserved - NO NEW OBJECTS."""
        first_count = len(self._detect_shapes_with_area(first_frame))
        
        # Count shapes excluding red marking
        final_count = len(self._detect_shapes_without_red(final_frame))
        
        # STRICT: No new objects allowed
        if final_count > first_count:
            return 0.0  # New objects generated - violation
        
        if final_count == first_count:
            return 1.0  # Perfect preservation
        elif final_count == first_count - 1:
            return 0.7  # One shape lost
        else:
            return 0.0  # STRICT: Too many shapes changed
    
    def _detect_shapes_with_area(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect shapes with (x, y, area)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, area))
        
        return shapes
    
    def _detect_red_border(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect red border/outline in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _get_contour_center(self, contour: np.ndarray) -> Optional[Tuple[int, int]]:
        """Get center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return None


class SelectLongestPolygonSideEvaluator(BaseEvaluator):
    """
    G-167: Select the longest polygon side.
    
    Rule-based evaluation:
    - Longest side identification (50%): Correctly identify longest edge
    - Marking position (25%): Circle/marker at midpoint of edge
    - Marking uniqueness (15%): Only one edge marked
    - Visual quality (10%): Circle style
    """
    
    TASK_WEIGHTS = {
        'longest_side_identification': 0.50,
        'marking_position': 0.25,
        'marking_uniqueness': 0.15,
        'visual_quality': 0.10
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
        
        # 1. Longest side identification (50%)
        scores['longest_side_identification'] = self._evaluate_side_identification(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking position (25%)
        scores['marking_position'] = self._evaluate_marking_position(final_frame)
        
        # 3. Marking uniqueness (15%)
        scores['marking_uniqueness'] = self._evaluate_marking_uniqueness(final_frame)
        
        # 4. Visual quality (10%)
        scores['visual_quality'] = self._evaluate_visual_quality_marking(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_side_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the correct side is identified."""
        # Detect marking in final frame
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_marking_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + 
                              (circle[1] - gt_circle[1])**2)
                if dist < 30:
                    return 1.0
                elif dist < 60:
                    return 0.8
                elif dist < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Find the longest edge midpoint from first frame
        longest_edge_midpoint = self._find_longest_edge_midpoint(first_frame)
        
        if longest_edge_midpoint is None:
            return 0.5  # Can't verify
        
        # Check if circle is near the longest edge midpoint
        dist = np.sqrt((circle[0] - longest_edge_midpoint[0])**2 + 
                      (circle[1] - longest_edge_midpoint[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.5
        else:
            return 0.2
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate if circle is at midpoint of an edge."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in a reasonable position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 20 < x < w - 20 and 20 < y < h - 20:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_marking_uniqueness(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if only one edge is marked."""
        circles = self._detect_all_marking_circles(final_frame)
        
        if len(circles) == 0:
            return 0.0
        elif len(circles) == 1:
            return 1.0
        else:
            return max(0.3, 1.0 - 0.3 * (len(circles) - 1))
    
    def _evaluate_visual_quality_marking(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle visual quality."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 5 < r < 50:
            return 1.0
        else:
            return 0.5
    
    def _find_longest_edge_midpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find midpoint of the longest edge in the polygon."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get the main polygon contour
        polygon = max(contours, key=cv2.contourArea)
        
        # Approximate to get vertices
        peri = cv2.arcLength(polygon, True)
        approx = cv2.approxPolyDP(polygon, 0.02 * peri, True)
        
        if len(approx) < 3:
            return None
        
        # Find longest edge
        max_length = 0
        longest_midpoint = None
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if length > max_length:
                max_length = length
                longest_midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        
        return longest_midpoint
    
    def _detect_marking_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the marking circle (red, orange, or yellow)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Warm colors (red, orange, yellow)
        lower_warm1 = np.array([0, 50, 50])
        upper_warm1 = np.array([35, 255, 255])
        lower_warm2 = np.array([160, 50, 50])
        upper_warm2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_warm1, upper_warm1) | cv2.inRange(hsv, lower_warm2, upper_warm2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10:
                (x, y), r = cv2.minEnclosingCircle(largest)
                if r > 3:
                    return (int(x), int(y), int(r))
        
        return None
    
    def _detect_all_marking_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect all marking circles."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_warm1 = np.array([0, 80, 80])
        upper_warm1 = np.array([35, 255, 255])
        lower_warm2 = np.array([160, 80, 80])
        upper_warm2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_warm1, upper_warm1) | cv2.inRange(hsv, lower_warm2, upper_warm2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circles.append((int(x), int(y), int(r)))
        
        return circles


class ArrangeCirclesByCircumferenceEvaluator(BaseEvaluator):
    """
    G-174: Arrange circles by circumference (large to small).
    
    Rule-based evaluation:
    - Sorting correctness (40%): Circles ordered by size (descending left to right)
    - Layout accuracy (30%): Horizontal alignment, even spacing
    - Object fidelity (20%): Circle properties preserved
    - Completeness (10%): All circles present
    """
    
    TASK_WEIGHTS = {
        'sorting_correctness': 0.40,
        'layout_accuracy': 0.30,
        'object_fidelity': 0.20,
        'completeness': 0.10
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
        
        # 1. Sorting correctness (40%)
        scores['sorting_correctness'] = self._evaluate_sorting(final_frame)
        
        # 2. Layout accuracy (30%)
        scores['layout_accuracy'] = self._evaluate_layout(final_frame)
        
        # 3. Object fidelity (20%)
        scores['object_fidelity'] = self._evaluate_fidelity(first_frame, final_frame)
        
        # 4. Completeness (10%)
        scores['completeness'] = self._evaluate_completeness(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_sorting(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circles are sorted by circumference (large to small)."""
        circles = self._detect_circles_with_size(final_frame)
        
        if len(circles) < 2:
            return 0.0  # STRICT: Not enough circles detected
        
        # Sort by x-position (left to right)
        sorted_by_x = sorted(circles, key=lambda c: c[0])
        radii = [c[2] for c in sorted_by_x]
        
        # Count inversions (smaller before larger - should be descending)
        inversions = sum(1 for i in range(len(radii)-1) if radii[i] < radii[i+1])
        max_inversions = len(radii) - 1
        
        if max_inversions == 0:
            return 1.0
        
        score = 1.0 - inversions / max_inversions
        return score
    
    def _evaluate_layout(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate horizontal alignment and spacing."""
        circles = self._detect_circles_with_size(final_frame)
        
        if len(circles) < 2:
            return 0.0  # STRICT: Not enough circles for layout eval
        
        # Check Y-coordinate alignment
        y_coords = [c[1] for c in circles]
        y_variance = np.var(y_coords)
        alignment_score = 1.0 / (1.0 + y_variance / 100)
        
        # Check spacing uniformity
        sorted_by_x = sorted(circles, key=lambda c: c[0])
        spacings = []
        for i in range(1, len(sorted_by_x)):
            spacing = sorted_by_x[i][0] - sorted_by_x[i-1][0]
            spacings.append(spacing)
        
        if len(spacings) > 1:
            spacing_variance = np.var(spacings) / (np.mean(spacings) + 1)
            spacing_score = 1.0 / (1.0 + spacing_variance)
        else:
            spacing_score = 1.0
        
        return 0.6 * alignment_score + 0.4 * spacing_score
    
    def _evaluate_fidelity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circle properties are preserved."""
        first_circles = self._detect_circles_with_size(first_frame)
        final_circles = self._detect_circles_with_size(final_frame)
        
        if len(first_circles) == 0 or len(final_circles) == 0:
            return 0.0  # STRICT: No circles detected
        
        # Compare radii distributions
        first_radii = sorted([c[2] for c in first_circles])
        final_radii = sorted([c[2] for c in final_circles])
        
        if len(first_radii) != len(final_radii):
            return 0.0  # STRICT: Circle count changed
        
        # Calculate radius similarity
        radius_diffs = [abs(fr - gr) for fr, gr in zip(first_radii, final_radii)]
        avg_diff = np.mean(radius_diffs)
        
        return max(0.0, 1.0 - avg_diff / 30)
    
    def _evaluate_completeness(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if all circles are present."""
        first_circles = self._detect_circles_with_size(first_frame)
        final_circles = self._detect_circles_with_size(final_frame)
        
        if len(first_circles) == 0:
            return 0.0  # STRICT: No circles in first frame
        
        completeness = min(1.0, len(final_circles) / len(first_circles))
        
        if len(final_circles) > len(first_circles):
            completeness *= 0.9  # Penalize extra circles
        
        return completeness
    
    def _detect_circles_with_size(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circles with their x, y, radius."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=30, minRadius=15, maxRadius=100
        )
        
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                result.append((int(x), int(y), int(r)))
        
        return result


class DrawNextSizedShapeEvaluator(BaseEvaluator):
    """
    G-193: Draw next sized shape in pattern.
    
    Rule-based evaluation:
    - Pattern recognition (30%): Identify "large-medium-small" size pattern
    - Figure drawing accuracy (35%): Correct type, color, smaller size
    - Label accuracy (25%): Correct Chinese label "小"
    - Animation quality (10%): Smooth growth animation
    """
    
    TASK_WEIGHTS = {
        'pattern_recognition': 0.30,
        'figure_drawing': 0.35,
        'label_accuracy': 0.25,
        'animation_quality': 0.10
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
        
        # CRITICAL: First check if the shape count is correct
        # Should add exactly ONE new shape
        first_shapes = self._detect_shapes_with_area(first_frame, exclude_boxes=True)
        final_shapes = self._detect_shapes_with_area(final_frame, exclude_boxes=True)
        
        shape_count_change = len(final_shapes) - len(first_shapes)
        
        # If more than 2 new shapes or shapes removed, task failed
        if shape_count_change > 2 or shape_count_change < 0:
            self._last_task_details = {
                'pattern_recognition': 0.0,
                'figure_drawing': 0.0,
                'label_accuracy': 0.0,
                'animation_quality': 0.3,
                'too_many_shapes_changed': True,
                'first_count': len(first_shapes),
                'final_count': len(final_shapes)
            }
            return 0.0
        
        # 1. Pattern recognition (30%) - CRITICAL: Is size pattern followed?
        pattern_score = self._evaluate_pattern_understanding(
            first_frame, final_frame
        )
        scores['pattern_recognition'] = pattern_score
        
        # CRITICAL: If pattern not followed, other scores should be penalized
        pattern_followed = pattern_score > 0.7
        
        # 2. Figure drawing (35%) - Only counts if pattern is followed
        if pattern_followed:
            scores['figure_drawing'] = self._evaluate_figure_drawing(
                first_frame, final_frame
            )
        else:
            scores['figure_drawing'] = 0.0  # Wrong pattern - no credit
        
        # 3. Label accuracy (25%) - Compare with GT if available
        if gt_final_frame is not None:
            # STRICT: Compare final frame with GT
            gen_final_resized = final_frame
            gt_final_resized = gt_final_frame
            if gen_final_resized.shape != gt_final_resized.shape:
                gt_final_resized = cv2.resize(gt_final_frame, (final_frame.shape[1], final_frame.shape[0]))
            
            diff = np.abs(gen_final_resized.astype(float) - gt_final_resized.astype(float)).mean()
            if diff < 15:
                scores['label_accuracy'] = 1.0
            elif diff < 30:
                scores['label_accuracy'] = 0.3
            else:
                scores['label_accuracy'] = 0.0
        else:
            scores['label_accuracy'] = self._evaluate_label(final_frame)
        
        # 4. Animation quality (10%)
        scores['animation_quality'] = self._evaluate_animation(video_frames)
        
        self._last_task_details = scores
        self._last_task_details['first_count'] = len(first_shapes)
        self._last_task_details['final_count'] = len(final_shapes)
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_pattern_understanding(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if size pattern is understood.
        
        The pattern is 'large-medium-small' repeating cyclically.
        We check if the rightmost new shape in the final frame follows the pattern.
        """
        # Detect shapes in both frames (excluding hollow boxes/designated areas)
        first_shapes = self._detect_shapes_with_area(first_frame, exclude_boxes=True)
        final_shapes = self._detect_shapes_with_area(final_frame, exclude_boxes=True)
        
        # Sort by x-position to get sequence
        first_sorted = sorted(first_shapes, key=lambda s: s[0])
        final_sorted = sorted(final_shapes, key=lambda s: s[0])
        
        if len(first_sorted) < 2 or len(final_sorted) < len(first_sorted):
            return 0.5
        
        # Get sizes from first frame to understand the pattern
        first_sizes = [s[2] for s in first_sorted]
        
        # Identify large, medium, small sizes from the first 3 shapes (if available)
        if len(first_sizes) >= 3:
            # Sort first 3 sizes to identify large, medium, small
            size_levels = sorted(first_sizes[:3], reverse=True)
            large_size = size_levels[0]
            medium_size = size_levels[1]
            small_size = size_levels[2]
            
            # Determine what the next shape should be based on position in pattern
            # Pattern: L-M-S-L-M-S-...
            # Position in pattern is (len(first_sorted)) % 3
            # 0 -> next is L, 1 -> next is M, 2 -> next is S
            pattern_position = len(first_sorted) % 3
            
            # Find the new shape (rightmost shape in final that wasn't in first)
            new_shapes = []
            for fs in final_sorted:
                is_new = True
                for ff in first_sorted:
                    if abs(fs[0] - ff[0]) < 50 and abs(fs[2] - ff[2]) / max(fs[2], ff[2]) < 0.3:
                        is_new = False
                        break
                if is_new:
                    new_shapes.append(fs)
            
            if new_shapes:
                # Get the rightmost new shape
                rightmost_new = max(new_shapes, key=lambda s: s[0])
                new_size = rightmost_new[2]
                
                # Check if new shape follows the pattern
                if pattern_position == 0:  # Should be large
                    expected_size = large_size
                elif pattern_position == 1:  # Should be medium
                    expected_size = medium_size
                else:  # Should be small
                    expected_size = small_size
                
                # Calculate size ratio
                size_ratio = min(new_size, expected_size) / max(new_size, expected_size)
                
                # Check if new size matches expected size category
                if size_ratio > 0.5:
                    return 1.0
                elif size_ratio > 0.3:
                    return 0.8
                else:
                    return 0.6
        
        return 0.1
    
    def _evaluate_figure_drawing(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate if correct figure is drawn in the box."""
        # Count shapes (excluding hollow boxes)
        first_shapes = self._detect_shapes_with_area(first_frame, exclude_boxes=True)
        final_shapes = self._detect_shapes_with_area(final_frame, exclude_boxes=True)
        first_count = len(first_shapes)
        final_count = len(final_shapes)
        
        # Should have one more shape
        if final_count == first_count + 1:
            if len(first_shapes) > 0 and len(final_shapes) > 0:
                # Get sizes sorted
                first_sizes = sorted([s[2] for s in first_shapes], reverse=True)
                final_sizes = sorted([s[2] for s in final_shapes], reverse=True)
                
                # The new shape should follow the pattern
                # If pattern is L-M-S-L-M, next should be S
                if len(first_sizes) >= 3:
                    small_size = first_sizes[2]  # Third largest = small
                    # Find the new shape size
                    new_size = None
                    for fs in final_sizes:
                        if fs not in first_sizes or final_sizes.count(fs) > first_sizes.count(fs):
                            new_size = fs
                            break
                    
                    if new_size is not None:
                        # Check if new size is close to small size
                        size_ratio = min(new_size, small_size) / max(new_size, small_size)
                        if size_ratio > 0.5:
                            return 1.0
                        elif size_ratio > 0.3:
                            return 0.8
                        return 0.6
                return 0.7
            return 0.6
        elif final_count >= first_count:
            return 0.5
        else:
            return 0.3
    
    def _evaluate_label(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate if label is present."""
        # Check for text/label in the new shape area
        h, w = final_frame.shape[:2]
        
        # Focus on right portion where new shape and label should be
        right_region = final_frame[:, 3*w//4:]
        
        gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY)
        
        # Count dark pixels (text)
        dark_pixels = np.sum(gray < 100)
        
        if dark_pixels > 500:  # Significant text present
            return 1.0
        elif dark_pixels > 200:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_animation(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 5:
            return 0.5
        
        # Check for smooth changes
        differences = []
        for i in range(1, min(len(video_frames), 30)):
            diff = np.mean(np.abs(
                video_frames[i].astype(float) - video_frames[i-1].astype(float)
            ))
            differences.append(diff)
        
        if len(differences) < 2:
            return 0.5
        
        # Smoothness: low variance in differences
        variance = np.var(differences)
        smoothness = 1.0 / (1.0 + variance / 100)
        
        return smoothness
    
    def _detect_shapes_with_area(self, frame: np.ndarray, exclude_boxes: bool = True, min_area: int = 2000) -> List[Tuple[int, int, int]]:
        """Detect shapes with (x, y, area).
        
        Args:
            frame: Input frame
            exclude_boxes: If True, exclude hollow boxes (designated areas)
            min_area: Minimum area threshold to filter out small labels/text
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple thresholds to handle different image styles
        best_shapes = []
        for thresh in [200, 220, 240]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shapes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    bbox_area = w * h
                    
                    # Exclude hollow boxes (designated areas) - they have low fill ratio
                    if exclude_boxes:
                        fill_ratio = area / bbox_area if bbox_area > 0 else 0
                        if fill_ratio < 0.3:  # Hollow box has low fill ratio
                            continue
                    
                    # Exclude small shapes that are likely labels/text
                    if area < min_area:
                        continue
                    
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        shapes.append((cx, cy, area))
            
            # Keep the threshold that finds the most shapes
            if len(shapes) > len(best_shapes):
                best_shapes = shapes
        
        return best_shapes


# Export all evaluators
HIDDEN40_EVALUATORS = {
    'G-47_multiple_keys_for_one_door_data-generator': MultipleKeysForOneDoorEvaluator,
    'G-135_select_next_figure_small_large_alternating_sequence_data-generator': SelectNextFigureAlternatingEvaluator,
    'G-136_locate_point_in_overlapping_area_data-generator': LocatePointInOverlappingAreaEvaluator,
    'G-140_locate_topmost_unobscured_figure_data-generator': LocateTopmostFigureEvaluator,
    'G-147_identify_unique_figure_in_uniform_set_data-generator': IdentifyUniqueFigureEvaluator,
    'G-160_circle_largest_numerical_value_data-generator': CircleLargestNumericalValueEvaluator,
    'G-161_mark_second_largest_shape_data-generator': MarkSecondLargestShapeEvaluator,
    'G-167_select_longest_polygon_side_data-generator': SelectLongestPolygonSideEvaluator,
    'G-174_arrange_circles_by_circumference_data-generator': ArrangeCirclesByCircumferenceEvaluator,
    'G-193_draw_next_sized_shape_data-generator': DrawNextSizedShapeEvaluator,
}
