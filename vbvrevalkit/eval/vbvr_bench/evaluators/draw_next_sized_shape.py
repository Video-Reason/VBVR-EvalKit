"""Evaluator for G-193_draw_next_sized_shape_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-193_draw_next_sized_shape_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


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
