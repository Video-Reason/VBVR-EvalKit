"""Evaluator for G-240_add_borders_to_unbordered_shapes_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-240_add_borders_to_unbordered_shapes_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class AddBordersToUnborderedEvaluator(BaseEvaluator):
    """
    G-240: Add borders to unbordered shapes evaluator.
    
    Rule-based evaluation:
    - Border identification accuracy (40%): Identify shapes without borders
    - Border addition correctness (35%): Add black borders correctly
    - Border appearance quality (15%): Border style and width
    - Scene preservation (10%): Original attributes unchanged
    """
    
    TASK_WEIGHTS = {
        'border_identification': 0.40,
        'border_addition': 0.35,
        'border_appearance': 0.15,
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
        
        # First, detect colorful shapes and check which ones have borders
        first_shapes = self._detect_colorful_shapes(first_frame)
        first_bordered = self._count_shapes_with_black_border(first_frame, first_shapes)
        
        # Check if all shapes already have borders in the first frame
        # If so, no change is needed and the task is already complete
        if len(first_shapes) > 0 and first_bordered == len(first_shapes):
            self._last_task_details = {
                'border_identification': 1.0,
                'border_addition': 1.0,
                'border_appearance': 1.0,
                'scene_preservation': 1.0,
                'all_already_bordered': True,
                'first_shapes': len(first_shapes),
                'first_bordered': first_bordered
            }
            return 1.0  # Task already complete
        
        # Check if borders were added by looking at dark pixel increase
        # (Borders are BLACK lines around colorful shapes, so they add dark pixels)
        first_dark = self._count_dark_edge_pixels(first_frame)
        final_dark = self._count_dark_edge_pixels(final_frame)
        dark_increase = final_dark - first_dark
        
        # Also check frame difference as secondary metric
        frame_diff = np.mean(np.abs(first_frame.astype(float) - final_frame.astype(float)))
        
        # If no dark pixel increase AND no frame difference, task not completed
        # Note: borders are thin black lines, so frame_diff can be small even with borders
        if dark_increase < 500 and frame_diff < 1:
            # No meaningful change - task not completed - ALL scores 0
            self._last_task_details = {
                'border_identification': 0.0,
                'border_addition': 0.0,
                'border_appearance': 0.0,
                'scene_preservation': 0.0,
                'no_change_detected': True,
                'dark_increase': int(dark_increase),
                'frame_diff': float(frame_diff)
            }
            return 0.0
        
        # Need significant increase in dark pixels for borders
        # Borders are BLACK lines, so should add at least 1000 dark pixels
        if dark_increase < 1000:
            self._last_task_details = {
                'border_identification': 0.0,
                'border_addition': 0.0,
                'border_appearance': 0.0,
                'scene_preservation': 0.5,
                'no_borders_added': True,
                'dark_pixel_increase': int(dark_increase)
            }
            return 0.0
        
        # Detect colorful shapes (几何体) in both frames
        final_shapes = self._detect_colorful_shapes(final_frame)
        
        # Count shapes with black borders
        final_bordered = self._count_shapes_with_black_border(final_frame, final_shapes)
        new_borders = final_bordered - first_bordered
        
        # CRITICAL CHECK: Shape count should remain approximately the same
        # Adding borders should NOT create new shapes
        shape_count_change = len(final_shapes) - len(first_shapes)
        if shape_count_change > 2:
            # Too many new shapes created - model is not just adding borders
            self._last_task_details = {
                'border_identification': 0.0,
                'border_addition': 0.0,
                'border_appearance': 0.0,
                'scene_preservation': 0.0,
                'too_many_new_shapes': True,
                'first_shapes': int(len(first_shapes)),
                'final_shapes': int(len(final_shapes)),
                'shape_count_change': int(shape_count_change)
            }
            return 0.0
        
        # 1. Border identification (40%): Check if unbordered shapes were identified
        scores['border_identification'] = self._evaluate_border_id(
            first_shapes, first_bordered, final_bordered
        )
        
        # 2. Border addition (35%): Check if black borders were added correctly
        scores['border_addition'] = self._evaluate_border_addition(
            first_frame, final_frame, dark_increase
        )
        
        # 3. Border appearance (15%): Check border quality (black, clean lines)
        scores['border_appearance'] = self._evaluate_border_appearance(final_frame)
        
        # 4. Scene preservation (10%): Check shapes preserved
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame, first_shapes, final_shapes
        )
        
        self._last_task_details = scores
        self._last_task_details['first_shapes'] = int(len(first_shapes))
        self._last_task_details['final_shapes'] = int(len(final_shapes))
        self._last_task_details['first_bordered'] = int(first_bordered)
        self._last_task_details['final_bordered'] = int(final_bordered)
        self._last_task_details['dark_increase'] = int(dark_increase)
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_colorful_shapes(self, frame: np.ndarray) -> List[Dict]:
        """Detect colorful geometric shapes (几何体) in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colorful) regions - shapes are colorful
        saturation_mask = hsv[:, :, 1] > 50
        value_mask = hsv[:, :, 2] > 50
        colorful_mask = (saturation_mask & value_mask).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(colorful_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip small noise
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            
            shapes.append({
                'contour': cnt,
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })
        
        return shapes
    
    def _count_shapes_with_black_border(self, frame: np.ndarray, shapes: List[Dict]) -> int:
        """Count how many shapes have black borders around them."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bordered_count = 0
        
        for shape in shapes:
            x, y, w, h = shape['bbox']
            
            # Expand bbox slightly to check for border
            margin = 5
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            # Check for dark pixels (black border) around the shape
            # Look at the perimeter region
            perimeter_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.rectangle(perimeter_mask, (x1, y1), (x2, y2), 255, margin * 2)
            cv2.rectangle(perimeter_mask, (x, y), (x + w, y + h), 0, -1)
            
            # Count dark pixels in perimeter
            perimeter_region = gray[perimeter_mask > 0]
            if len(perimeter_region) > 0:
                dark_pixels = np.sum(perimeter_region < 50)
                dark_ratio = dark_pixels / len(perimeter_region)
                
                # If significant dark pixels around shape, it has a border
                if dark_ratio > 0.1:
                    bordered_count += 1
        
        return bordered_count
    
    def _evaluate_border_id(
        self,
        first_shapes: List[Dict],
        first_bordered: int,
        final_bordered: int
    ) -> float:
        """Rule-based: Check if unbordered shapes were identified and bordered."""
        if len(first_shapes) == 0:
            return 0.0
        
        # Calculate how many shapes needed borders
        unbordered_first = len(first_shapes) - first_bordered
        if unbordered_first == 0:
            return 1.0  # All were already bordered
        
        # How many got new borders
        new_borders = final_bordered - first_bordered
        
        # Score based on proportion of unbordered shapes that got borders
        if new_borders >= unbordered_first:
            return 1.0  # All unbordered shapes got borders
        elif new_borders > 0:
            return max(0.5, new_borders / unbordered_first)
        else:
            return 0.0
    
    def _evaluate_border_addition(
        self,
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        dark_increase: int
    ) -> float:
        """Rule-based: Check if black borders were added correctly."""
        # Borders should add significant dark pixels
        if dark_increase < 1000:
            return 0.0
        
        # Check if dark pixels form edges (borders should be lines, not blobs)
        edges = cv2.Canny(final_frame, 50, 150)
        first_edges = cv2.Canny(first_frame, 50, 150)
        
        edge_count = np.sum(edges > 0)
        first_edge_count = np.sum(first_edges > 0)
        edge_increase = edge_count - first_edge_count
        
        # Good borders should have both dark pixel increase AND edge increase
        if dark_increase > 3000 and edge_increase > 1000:
            return 1.0
        elif dark_increase > 2000 or edge_increase > 500:
            return 0.8
        elif dark_increase > 1000:
            return 0.6
        else:
            return 0.0
    
    def _evaluate_border_appearance(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate border appearance quality."""
        # Check edge structure
        edges = cv2.Canny(final_frame, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Also check for dark pixels (borders are typically dark)
        gray = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 50) / gray.size
        
        # Reasonable edge ratio indicates clean borders
        # Lower threshold since some images have sparse borders
        if edge_ratio > 0.001 and dark_ratio > 0.001:
            return 1.0
        elif edge_ratio > 0.0005 or dark_ratio > 0.0005:
            return 0.8
        else:
            return 0.2
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        first_shapes: List[Dict],
        final_shapes: List[Dict]
    ) -> float:
        """Rule-based: Check if shape colors and positions are preserved."""
        # Check if same number of shapes exist
        if len(first_shapes) == 0:
            return 0.5
        
        shape_count_diff = abs(len(first_shapes) - len(final_shapes))
        
        if shape_count_diff > 2:
            return 0.3  # Too many shapes changed
        
        # Compare color distributions
        first_colors = self._get_color_distribution(first_frame)
        final_colors = self._get_color_distribution(final_frame)
        
        # Similar color distributions indicate preservation
        color_diff = np.sum(np.abs(first_colors - final_colors))
        
        # Normalize by total histogram sum
        total = np.sum(first_colors) + np.sum(final_colors)
        if total > 0:
            normalized_diff = color_diff / total
        else:
            normalized_diff = 0
        
        # Use normalized difference for more consistent scoring
        if normalized_diff < 0.1 and shape_count_diff <= 1:
            return 1.0
        elif normalized_diff < 0.2:
            return 0.8
        elif normalized_diff < 0.4:
            return 0.6
        else:
            return 0.4
    
    def _count_bordered_shapes(self, frame: np.ndarray) -> int:
        """Count shapes with borders (dark edges)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count contours with substantial edges (bordered shapes)
        return sum(1 for cnt in contours if cv2.arcLength(cnt, True) > 100)
    
    def _count_dark_edge_pixels(self, frame: np.ndarray) -> int:
        """Count dark pixels (potential borders)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.sum(gray < 50)
    
    def _get_color_distribution(self, frame: np.ndarray) -> np.ndarray:
        """Get color histogram."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        return hist.flatten()
