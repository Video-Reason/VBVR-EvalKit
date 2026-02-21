"""Evaluator for G-25_seperate_object_spinning_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-25_seperate_object_spinning_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class SeparateObjectsSpinningEvaluator(BaseEvaluator):
    """
    G-25: Separate objects with spinning evaluator.
    
    Evaluates:
    - Rotation correctness (40%): Objects rotated to match target orientation
    - Alignment precision (30%): Objects align with dashed outlines
    - Operation order (20%): Rotate first, then translate
    - Visual fidelity (10%): Shape, color, size preserved
    """
    
    TASK_WEIGHTS = {
        'rotation': 0.40,
        'alignment': 0.30,
        'order': 0.20,
        'fidelity': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate separate objects spinning task."""
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Detect objects in first and last frames
        first_objects = self._detect_colored_objects(first_frame)
        last_objects = self._detect_colored_objects(last_frame)
        gt_objects = self._detect_colored_objects(gt_last)
        
        # 1. Rotation correctness (40%): Check if objects reached correct orientation
        rotation_score = self._evaluate_rotation(last_objects, gt_objects)
        scores['rotation'] = rotation_score
        
        # 2. Alignment precision (30%): Check if objects align with target outlines
        alignment_score = self._evaluate_alignment(last_frame, gt_last)
        scores['alignment'] = alignment_score
        
        # 3. Operation order (20%): Check if rotation happens before translation
        order_score = self._evaluate_operation_order(video_frames, first_objects)
        scores['order'] = order_score
        
        # 4. Visual fidelity (10%): Check if shape, color, size preserved
        fidelity_score = self._evaluate_fidelity(first_objects, last_objects)
        scores['fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_colored_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect solid colored objects (not dashed outlines)."""
        objects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common object colors
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Filter small noise
                    continue
                
                # Get moments and orientation
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Calculate orientation using moments
                if M['mu20'] - M['mu02'] != 0:
                    angle = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
                else:
                    angle = 0
                
                # Get bounding box
                rect = cv2.minAreaRect(contour)
                
                objects.append({
                    'color': color_name,
                    'center': (cx, cy),
                    'area': area,
                    'angle': np.degrees(angle),
                    'contour': contour,
                    'rect': rect
                })
        
        return objects
    
    def _evaluate_rotation(self, last_objects: List[Dict], gt_objects: List[Dict]) -> float:
        """Evaluate if objects are rotated to correct orientation."""
        if not last_objects or not gt_objects:
            return 0.0
        
        total_score = 0.0
        matched = 0
        
        for gt_obj in gt_objects:
            # Find matching object by color and approximate position
            best_match = None
            best_dist = float('inf')
            
            for obj in last_objects:
                if obj['color'] == gt_obj['color']:
                    dist = safe_distance(obj['center'], gt_obj['center'])
                    if dist < best_dist:
                        best_dist = dist
                        best_match = obj
            
            if best_match:
                matched += 1
                # Compare angles (accounting for symmetry)
                angle_diff = abs(best_match['angle'] - gt_obj['angle'])
                angle_diff = min(angle_diff, 180 - angle_diff)  # Handle symmetry
                
                # Score based on angle error: <5° = 1.0, 5-10° = 0.8, 10-20° = 0.6, >20° = 0.4
                if angle_diff < 5:
                    total_score += 1.0
                elif angle_diff < 10:
                    total_score += 0.8
                elif angle_diff < 20:
                    total_score += 0.6
                else:
                    total_score += max(0.2, 1.0 - angle_diff / 90)
        
        return total_score / max(1, len(gt_objects))
    
    def _evaluate_alignment(self, last_frame: np.ndarray, gt_last: np.ndarray) -> float:
        """Evaluate if objects align with target outlines."""
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Compare using structural similarity in regions with objects
        gray_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        gray_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2GRAY)
        
        # Find object regions in GT
        _, thresh_gt = cv2.threshold(gray_gt, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0  # STRICT: No contours to compare
        
        alignment_scores = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            
            # Create mask for this region
            mask = np.zeros(gray_gt.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Expand mask slightly for alignment tolerance
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Compare regions
            roi_gen = gray_gen[mask > 0]
            roi_gt = gray_gt[mask > 0]
            
            if len(roi_gen) > 0 and len(roi_gt) > 0:
                diff = np.mean(np.abs(roi_gen.astype(float) - roi_gt.astype(float)))
                alignment_scores.append(max(0, 1.0 - diff / 128))
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def _evaluate_operation_order(self, frames: List[np.ndarray], first_objects: List[Dict]) -> float:
        """Evaluate if rotation happens before translation."""
        if len(frames) < 5 or not first_objects:
            return 0.0  # STRICT: Not enough data
        
        # Track object positions and angles through video
        n_samples = min(20, len(frames))
        sample_indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
        
        position_changes = []
        angle_changes = []
        
        prev_objects = first_objects
        for idx in sample_indices[1:]:
            frame = frames[idx]
            curr_objects = self._detect_colored_objects(frame)
            
            if not curr_objects:
                continue
            
            # Match objects and track changes
            for prev_obj in prev_objects:
                best_match = None
                best_dist = float('inf')
                
                for obj in curr_objects:
                    if obj['color'] == prev_obj['color']:
                        dist = safe_distance(obj['center'], prev_obj['center'])
                        if dist < best_dist:
                            best_dist = dist
                            best_match = obj
                
                if best_match:
                    position_changes.append(best_dist)
                    angle_diff = abs(best_match['angle'] - prev_obj['angle'])
                    angle_changes.append(min(angle_diff, 180 - angle_diff))
            
            prev_objects = curr_objects
        
        if len(position_changes) < 3:
            return 0.0  # STRICT: Not enough position data
        
        # Check if rotation (angle changes) happens before translation (position changes)
        # Split into first half and second half
        mid = len(position_changes) // 2
        
        first_half_angle = np.mean(angle_changes[:mid]) if angle_changes[:mid] else 0
        second_half_angle = np.mean(angle_changes[mid:]) if angle_changes[mid:] else 0
        first_half_pos = np.mean(position_changes[:mid]) if position_changes[:mid] else 0
        second_half_pos = np.mean(position_changes[mid:]) if position_changes[mid:] else 0
        
        # Ideal: more rotation in first half, more translation in second half
        rotation_first = first_half_angle > second_half_angle
        translation_second = second_half_pos > first_half_pos
        
        if rotation_first and translation_second:
            return 1.0
        elif rotation_first or translation_second:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_fidelity(self, first_objects: List[Dict], last_objects: List[Dict]) -> float:
        """Evaluate if shape, color, size are preserved."""
        if not first_objects or not last_objects:
            return 0.5
        
        preserved_count = 0
        total = len(first_objects)
        
        for first_obj in first_objects:
            # Find matching object by color
            for last_obj in last_objects:
                if first_obj['color'] == last_obj['color']:
                    # Check area preservation (within 20%)
                    area_ratio = last_obj['area'] / max(1, first_obj['area'])
                    if 0.8 <= area_ratio <= 1.2:
                        preserved_count += 1
                    break
        
        return preserved_count / max(1, total)
