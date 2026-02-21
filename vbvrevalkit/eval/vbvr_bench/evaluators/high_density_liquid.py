"""Evaluator for G-273_high_density_liquid_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-273_high_density_liquid_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class HighDensityLiquidEvaluator(BaseEvaluator):
    """
    G-273: High density liquid evaluator.
    
    Rule-based evaluation:
    - Physics reasoning accuracy (45%): Identify high-density liquid (object floats)
    - Marking object correctness (30%): Mark object in high-density liquid
    - Marking standardization (20%): Red rectangle marking
    - Element preservation (5%): Original elements unchanged
    """
    
    TASK_WEIGHTS = {
        'physics_reasoning': 0.45,
        'marking_correctness': 0.30,
        'marking_standardization': 0.20,
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
        
        # 1. Physics reasoning (45%)
        scores['physics_reasoning'] = self._evaluate_physics_reasoning(
            first_frame, final_frame
        )
        
        # 2. Marking correctness (30%)
        scores['marking_correctness'] = self._evaluate_marking_correctness(
            first_frame, final_frame
        )
        
        # 3. Marking standardization (20%)
        scores['marking_standardization'] = self._evaluate_marking_standard(final_frame)
        
        # 4. Element preservation (5%)
        scores['element_preservation'] = self._evaluate_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_physics_reasoning(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if high-density liquid (floating yellow square) is identified.
        
        CRITICAL RULES:
        1. There must be TWO yellow squares floating in liquid containers
        2. The squares should be in the middle portion of the frame (not at top)
        3. A red rectangle marking should be around the HIGHER floating square
        """
        h, w = final_frame.shape[:2]
        
        # Find yellow squares
        yellow_objects = self._find_yellow_squares(final_frame)
        
        # CRITICAL: Must have exactly 2 yellow squares
        if len(yellow_objects) != 2:
            return 0.1
        
        # CRITICAL: Yellow squares should be in the middle portion (not at top edge)
        # If squares are at y < 10% of height, they're not in liquid containers
        for obj in yellow_objects:
            if obj[2] < h * 0.1:  # top_y < 10% of height
                return 0.1  # Objects at top edge - wrong scene
        
        # Find the higher floating square (smaller y = higher on screen)
        higher_obj = min(yellow_objects, key=lambda o: o[2])
        lower_obj = max(yellow_objects, key=lambda o: o[2])
        
        # CRITICAL: There should be a noticeable height difference
        height_diff = lower_obj[2] - higher_obj[2]
        if height_diff < 30:
            return 0.3  # Not enough difference to determine which is higher
        
        # Detect red rectangle marking
        rect = self._detect_red_rectangle(final_frame)
        
        if rect is None:
            return 0.2  # No marking found
        
        rect_center = ((rect[0] + rect[2])//2, (rect[1] + rect[3])//2)
        
        # Check if rectangle marks the HIGHER floating square
        dist_to_higher = np.sqrt((rect_center[0] - higher_obj[0])**2 + (rect_center[1] - higher_obj[1])**2)
        dist_to_lower = np.sqrt((rect_center[0] - lower_obj[0])**2 + (rect_center[1] - lower_obj[1])**2)
        
        # Rectangle should be closer to the higher floating square
        if dist_to_higher < dist_to_lower:
            if dist_to_higher < 80:
                return 1.0
            elif dist_to_higher < 150:
                return 0.8
            else:
                return 0.5
        else:
            # Marked the wrong square!
            return 0.2
    
    def _find_yellow_squares(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find yellow squares with their positions.
        
        Returns: List of (center_x, center_y, top_y, area)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Yellow squares should be reasonably sized
            if 2000 < area < 50000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    yellow_objects.append((cx, cy, y, area))
        
        return yellow_objects
    
    def _evaluate_marking_correctness(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the correct yellow square is marked with red rectangle."""
        rect = self._detect_red_rectangle(final_frame)
        floating_obj = self._find_floating_object(final_frame)
        
        if rect is None:
            return 0.0
        
        h, w = final_frame.shape[:2]
        rect_center = ((rect[0] + rect[2])//2, (rect[1] + rect[3])//2)
        
        if floating_obj is None:
            # Fallback: check if rectangle is in the expected region
            if h * 0.3 < rect_center[1] < h * 0.7:
                return 0.8  # Rectangle is in reasonable position
            return 0.0
        
        dist = np.sqrt((rect_center[0] - floating_obj[0])**2 + (rect_center[1] - floating_obj[1])**2)
        
        # More lenient distance threshold for this task
        return max(0.3, 1.0 - dist / 150)
    
    def _evaluate_marking_standard(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if red rectangle marking is used."""
        rect = self._detect_red_rectangle(final_frame)
        
        if rect is None:
            return 0.0
        
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        
        if 20 < w < 200 and 20 < h < 200:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if elements are preserved."""
        # Count objects
        first_objects = self._count_objects(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_objects = self._count_objects(final_no_red)
        
        if abs(first_objects - final_objects) <= 1:
            return 1.0
        else:
            return 0.6
    
    def _find_floating_object(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the yellow square (方块) that floats higher in the higher density liquid.
        
        This task has two containers with blue/green liquids. Yellow squares float
        in each. The one in higher density liquid floats HIGHER (top above liquid).
        We need to find the yellow square that floats higher and return its position.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Find yellow squares (方块)
        # Yellow hue range: 15-45 in OpenCV HSV
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Yellow squares are around 10000-15000 area
            if 5000 < area < 30000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Store center and top y position
                    yellow_objects.append((cx, cy, y, area))
        
        if len(yellow_objects) == 0:
            return None
        
        # The yellow square in higher density liquid floats HIGHER (smaller y = top)
        # Find the one with smallest top y (highest on screen)
        topmost = min(yellow_objects, key=lambda o: o[2])
        return (topmost[0], topmost[1])
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count objects in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if 500 < cv2.contourArea(cnt) < 10000)
    
    def _detect_red_rectangle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect red rectangle marking in the frame.
        
        CRITICAL: The marking should be a small-to-medium sized rectangle,
        not a large container. Filter out areas that are too large.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_frame, w_frame = frame.shape[:2]
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size - marking should be small to medium
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Marking rectangle should be:
            # - Not too small (> 500 pixels)
            # - Not too large (< 10% of frame area)
            # - Roughly square-ish (aspect ratio between 0.3 and 3)
            frame_area = h_frame * w_frame
            if 500 < area < frame_area * 0.1:
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3:
                    valid_contours.append((cnt, area))
        
        if valid_contours:
            # Take the largest valid contour
            largest = max(valid_contours, key=lambda x: x[1])[0]
            x, y, w, h = cv2.boundingRect(largest)
            return (x, y, x + w, y + h)
        
        return None


# Export all evaluators
HIDDEN40_EVALUATORS_PART2 = {
    'G-202_mark_wave_peaks_data-generator': MarkWavePeaksEvaluator,
    'G-212_find_incorrect_arrow_direction_data-generator': FindIncorrectArrowDirectionEvaluator,
    'G-217_circle_central_dot_data-generator': CircleCentralDotEvaluator,
    'G-218_identify_largest_angle_in_triangle_data-generator': IdentifyLargestAngleEvaluator,
    'G-219_select_leftmost_shape_data-generator': SelectLeftmostShapeEvaluator,
    'G-221_outline_innermost_square_data-generator': OutlineInnermostSquareEvaluator,
    'G-240_add_borders_to_unbordered_shapes_data-generator': AddBordersToUnborderedEvaluator,
    'G-247_identify_chinese_character_data-generator': IdentifyChineseCharacterEvaluator,
    'G-248_mark_asymmetrical_shape_data-generator': MarkAsymmetricalShapeEvaluator,
    'G-273_high_density_liquid_data-generator': HighDensityLiquidEvaluator,
