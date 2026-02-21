"""Evaluator for G-135_select_next_figure_small_large_alternating_sequence_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-135_select_next_figure_small_large_alternating_sequence_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


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
