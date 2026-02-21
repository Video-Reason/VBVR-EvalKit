"""Evaluator for G-136_locate_point_in_overlapping_area_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-136_locate_point_in_overlapping_area_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


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
