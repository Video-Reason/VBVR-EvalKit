"""Evaluator for G-221_outline_innermost_square_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-221_outline_innermost_square_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class OutlineInnermostSquareEvaluator(BaseEvaluator):
    """
    G-221: Outline innermost square evaluator.
    
    Rule-based evaluation for concentric squares centered at canvas:
    - Concentric structure preservation (40%): Squares remain concentric at canvas center
    - Color preservation (35%): Colors on all sides (上下左右) remain the same
    - Blue outline addition (20%): Blue outline added around innermost square
    - Element preservation (5%): Original squares unchanged
    """
    
    TASK_WEIGHTS = {
        'concentric_structure': 0.40,
        'color_preservation': 0.35,
        'outline_addition': 0.20,
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
        
        h, w = first_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Detect concentric squares in first frame (GT reference)
        gt_squares = self._detect_concentric_squares(first_frame)
        
        # 1. Check concentric structure preservation (40%)
        scores['concentric_structure'] = self._evaluate_concentric_structure(
            first_frame, final_frame, center_x, center_y
        )
        
        # If structure is completely broken, return early with low score
        if scores['concentric_structure'] < 0.3:
            self._last_task_details = {
                'concentric_structure': scores['concentric_structure'],
                'color_preservation': 0.0,
                'outline_addition': 0.0,
                'element_preservation': 0.0,
                'structure_broken': True
            }
            return 0.0
        
        # 2. Check color preservation (35%) - colors on all 4 sides should match
        scores['color_preservation'] = self._evaluate_color_preservation(
            first_frame, final_frame, center_x, center_y
        )
        
        # 3. Check blue outline addition (20%)
        scores['outline_addition'] = self._evaluate_outline_addition(
            first_frame, final_frame, center_x, center_y
        )
        
        # 4. Element preservation (5%)
        scores['element_preservation'] = self._evaluate_element_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_concentric_squares(self, frame: np.ndarray) -> List[Dict]:
        """Detect concentric squares by scanning from center outward."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        squares = []
        prev_color = None
        current_dist = 0
        
        # Scan horizontally from center to right edge
        for x in range(center_x, w):
            color = tuple(frame[center_y, x])
            if prev_color is not None:
                color_diff = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(color, prev_color))
                if color_diff > 30:  # Color transition = square boundary
                    squares.append({
                        'distance': x - center_x,
                        'color': prev_color
                    })
            prev_color = color
        
        # Add the outermost square
        if prev_color is not None:
            squares.append({
                'distance': w - center_x,
                'color': prev_color
            })
        
        return squares
    
    def _evaluate_concentric_structure(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        center_x: int,
        center_y: int
    ) -> float:
        """Check if concentric square structure is preserved."""
        h, w = first_frame.shape[:2]
        
        # Sample colors at multiple distances from center in all 4 directions
        # For concentric squares, colors at same distance should be similar
        distances = [50, 100, 150, 200, 250, 300, 350, 400]
        
        matches = 0
        total = 0
        
        for dist in distances:
            if dist >= min(center_x, center_y, w - center_x, h - center_y):
                continue
            
            # Get colors in 4 directions for first frame
            first_colors = []
            final_colors = []
            
            # Right
            first_colors.append(tuple(first_frame[center_y, min(center_x + dist, w-1)]))
            final_colors.append(tuple(final_frame[center_y, min(center_x + dist, w-1)]))
            # Left
            first_colors.append(tuple(first_frame[center_y, max(center_x - dist, 0)]))
            final_colors.append(tuple(final_frame[center_y, max(center_x - dist, 0)]))
            # Down
            first_colors.append(tuple(first_frame[min(center_y + dist, h-1), center_x]))
            final_colors.append(tuple(final_frame[min(center_y + dist, h-1), center_x]))
            # Up
            first_colors.append(tuple(first_frame[max(center_y - dist, 0), center_x]))
            final_colors.append(tuple(final_frame[max(center_y - dist, 0), center_x]))
            
            # Check if colors in final frame match first frame (ignoring blue outline)
            for fc, fnc in zip(first_colors, final_colors):
                total += 1
                # Allow for blue outline (high B, low G, low R)
                is_blue = fnc[0] > 200 and fnc[1] < 50 and fnc[2] < 50
                if is_blue:
                    matches += 1  # Blue outline is acceptable
                else:
                    color_diff = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(fc, fnc))
                    if color_diff < 50:
                        matches += 1
        
        if total == 0:
            return 0.5
        
        return matches / total
    
    def _evaluate_color_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        center_x: int,
        center_y: int
    ) -> float:
        """Check if colors on all 4 sides (上下左右) are preserved."""
        h, w = first_frame.shape[:2]
        
        # Sample at multiple distances
        distances = [100, 200, 300, 400]
        
        preserved = 0
        total = 0
        
        for dist in distances:
            if dist >= min(center_x, center_y, w - center_x, h - center_y):
                continue
            
            # Get first frame colors at 4 directions
            first_right = tuple(first_frame[center_y, min(center_x + dist, w-1)])
            first_left = tuple(first_frame[center_y, max(center_x - dist, 0)])
            first_down = tuple(first_frame[min(center_y + dist, h-1), center_x])
            first_up = tuple(first_frame[max(center_y - dist, 0), center_x])
            
            # Get final frame colors
            final_right = tuple(final_frame[center_y, min(center_x + dist, w-1)])
            final_left = tuple(final_frame[center_y, max(center_x - dist, 0)])
            final_down = tuple(final_frame[min(center_y + dist, h-1), center_x])
            final_up = tuple(final_frame[max(center_y - dist, 0), center_x])
            
            # Check if 4 sides have same color in first frame (concentric property)
            first_colors = [first_right, first_left, first_down, first_up]
            final_colors = [final_right, final_left, final_down, final_up]
            
            # For each direction, check if color is preserved
            for fc, fnc in zip(first_colors, final_colors):
                total += 1
                # Ignore blue outline
                is_blue = fnc[0] > 200 and fnc[1] < 50 and fnc[2] < 50
                if is_blue:
                    preserved += 1
                else:
                    color_diff = sum(abs(int(c1) - int(c2)) for c1, c2 in zip(fc, fnc))
                    if color_diff < 50:
                        preserved += 1
        
        if total == 0:
            return 0.5
        
        return preserved / total
    
    def _evaluate_outline_addition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        center_x: int,
        center_y: int
    ) -> float:
        """Check if blue outline was added around innermost square."""
        # Count blue pixels in first vs final
        first_blue = self._count_blue_pixels(first_frame)
        final_blue = self._count_blue_pixels(final_frame)
        
        blue_increase = final_blue - first_blue
        
        if blue_increase < 500:
            return 0.0  # No blue outline added
        
        # Check if blue outline is near center (around innermost square)
        blue_mask = self._get_blue_mask(final_frame)
        blue_points = np.where(blue_mask > 0)
        
        if len(blue_points[0]) == 0:
            return 0.0
        
        # Calculate average distance of blue pixels from center
        avg_y = np.mean(blue_points[0])
        avg_x = np.mean(blue_points[1])
        
        dist_from_center = np.sqrt((avg_x - center_x)**2 + (avg_y - center_y)**2)
        
        # Blue outline should be near center (innermost square)
        h, w = final_frame.shape[:2]
        max_dist = min(w, h) / 2
        
        # Closer to center = better
        if dist_from_center < max_dist * 0.3:
            return 1.0
        elif dist_from_center < max_dist * 0.5:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_element_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Check if overall structure is preserved."""
        # Compare histograms (excluding blue)
        first_hist = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        final_hist = cv2.calcHist([final_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        correlation = cv2.compareHist(first_hist, final_hist, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)
    
    def _count_blue_pixels(self, frame: np.ndarray) -> int:
        """Count pure blue pixels."""
        b, g, r = cv2.split(frame)
        blue_mask = (b > 200) & (g < 50) & (r < 50)
        return int(np.sum(blue_mask))
    
    def _get_blue_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get mask of blue pixels."""
        b, g, r = cv2.split(frame)
        return ((b > 200) & (g < 50) & (r < 50)).astype(np.uint8) * 255
