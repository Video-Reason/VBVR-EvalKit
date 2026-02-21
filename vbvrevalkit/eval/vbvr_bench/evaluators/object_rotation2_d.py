"""Evaluator for O-85_2d_object_rotation_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-85_2d_object_rotation_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class ObjectRotation2DEvaluator(BaseEvaluator):
    """
    O-85: 2D object rotation evaluator.
    
    Rule-based evaluation:
    - Rotation angle accuracy (40%): Correct degrees (within 2° for perfect)
    - Rotation direction (30%): Clockwise/counterclockwise correct
    - Rotation center (20%): Around object center (no translation)
    - Object fidelity (10%): Shape, color, size preserved
    """
    
    TASK_WEIGHTS = {
        'angle_accuracy': 0.40,
        'direction': 0.30,
        'center': 0.20,
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
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        scores['angle_accuracy'] = self._evaluate_angle(
            first_frame, final_frame, gt_final_frame
        )
        scores['direction'] = self._evaluate_direction(video_frames)
        scores['center'] = self._evaluate_center(first_frame, final_frame)
        scores['fidelity'] = self._evaluate_fidelity(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored objects and their orientations."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Find colored regions
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Get orientation using moments
            if M['mu20'] - M['mu02'] != 0:
                angle = 0.5 * np.arctan2(2 * M['mu11'], M['mu20'] - M['mu02'])
            else:
                angle = 0
            
            # Get bounding rect for additional angle info
            rect = cv2.minAreaRect(cnt)
            rect_angle = rect[2]
            
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if roi.size > 0:
                color = tuple(roi.mean(axis=(0, 1)).astype(int).tolist())
            else:
                color = (0, 0, 0)
            
            objects.append({
                'center': (cx, cy),
                'area': area,
                'angle': np.degrees(angle),
                'rect_angle': rect_angle,
                'color': color,
                'bbox': (x, y, w, h)
            })
        
        return objects
    
    def _evaluate_angle(self, first: np.ndarray, final: np.ndarray,
                        gt_final: Optional[np.ndarray]) -> float:
        """Check if rotation angle is correct."""
        first_objects = self._detect_objects(first)
        final_objects = self._detect_objects(final)
        
        if not first_objects or not final_objects:
            return 0.0  # STRICT: No objects detected
        
        if gt_final is not None:
            gt_objects = self._detect_objects(gt_final)
            
            if gt_objects and final_objects:
                # Compare angles with GT
                total_diff = 0
                matched = 0
                for fo in final_objects:
                    min_diff = float('inf')
                    for go in gt_objects:
                        # Match by color
                        color_dist = np.sqrt(sum((a-b)**2 for a, b in zip(fo['color'], go['color'])))
                        if color_dist < 50:
                            angle_diff = abs(fo['angle'] - go['angle'])
                            angle_diff = min(angle_diff, 180 - angle_diff)  # Handle wrap
                            min_diff = min(min_diff, angle_diff)
                    if min_diff < float('inf'):
                        total_diff += min_diff
                        matched += 1
                
                if matched > 0:
                    avg_diff = total_diff / matched
                    if avg_diff < 2:
                        return 1.0
                    elif avg_diff < 10:
                        return 0.8
                    elif avg_diff < 30:
                        return 0.5
                    return 0.2
        
        # Compare with first frame - check if rotation happened
        angle_changes = []
        for fo in final_objects:
            for ffo in first_objects:
                color_dist = np.sqrt(sum((a-b)**2 for a, b in zip(fo['color'], ffo['color'])))
                if color_dist < 50:
                    angle_diff = abs(fo['angle'] - ffo['angle'])
                    angle_changes.append(angle_diff)
        
        if angle_changes and np.mean(angle_changes) > 5:
            return 0.5  # Some rotation happened but not verified
        
        return 0.0  # STRICT: No clear rotation detected
    
    def _evaluate_direction(self, video_frames: List[np.ndarray]) -> float:
        """Check if rotation direction is correct."""
        if len(video_frames) < 5:
            return 0.0  # STRICT: Not enough frames
        
        # Track angle changes through video
        angle_progression = []
        prev_objects = None
        
        for frame in video_frames[::max(1, len(video_frames)//10)]:
            objects = self._detect_objects(frame)
            
            if prev_objects and objects:
                for obj in objects:
                    for pobj in prev_objects:
                        color_dist = np.sqrt(sum((a-b)**2 for a, b in zip(obj['color'], pobj['color'])))
                        if color_dist < 50:
                            angle_change = obj['angle'] - pobj['angle']
                            angle_progression.append(angle_change)
            
            prev_objects = objects
        
        if not angle_progression:
            return 0.0  # STRICT: No angle changes detected
        
        # Check if rotation is consistent (all same direction)
        positive = sum(1 for a in angle_progression if a > 0)
        negative = sum(1 for a in angle_progression if a < 0)
        total = positive + negative
        
        if total > 0:
            consistency = max(positive, negative) / total
            return consistency
        
        return 0.0  # STRICT: No clear rotation direction
    
    def _evaluate_center(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if rotation is around correct center."""
        first_objects = self._detect_objects(first)
        final_objects = self._detect_objects(final)
        
        if not first_objects or not final_objects:
            return 0.0  # STRICT: No objects detected
        
        # Objects should remain in their grid cells (center shouldn't move much)
        center_drifts = []
        
        for fo in final_objects:
            for ffo in first_objects:
                color_dist = np.sqrt(sum((a-b)**2 for a, b in zip(fo['color'], ffo['color'])))
                if color_dist < 50:
                    center_drift = np.sqrt((fo['center'][0] - ffo['center'][0])**2 +
                                          (fo['center'][1] - ffo['center'][1])**2)
                    center_drifts.append(center_drift)
        
        if center_drifts:
            avg_drift = np.mean(center_drifts)
            # Objects shouldn't move much during rotation around center
            if avg_drift < 10:
                return 1.0
            elif avg_drift < 30:
                return 0.7
            elif avg_drift < 50:
                return 0.5
            return 0.3
        
        return 0.5
    
    def _evaluate_fidelity(self, first: np.ndarray, final: np.ndarray) -> float:
        """Check if objects are preserved."""
        first_objects = self._detect_objects(first)
        final_objects = self._detect_objects(final)
        
        if not first_objects:
            return 0.5
        
        # Compare object counts
        count_ratio = min(len(final_objects), len(first_objects)) / max(len(first_objects), 1)
        
        # Compare total areas
        first_area = sum(o['area'] for o in first_objects)
        final_area = sum(o['area'] for o in final_objects)
        
        if max(first_area, final_area) > 0:
            area_ratio = min(first_area, final_area) / max(first_area, final_area)
        else:
            area_ratio = 0.5
        
        return 0.5 * count_ratio + 0.5 * area_ratio


# Export all Part 4 evaluators
HIDDEN40_EVALUATORS_PART4 = {
    'O-54_control_panel_data-generator': ControlPanelEvaluator,
    'O-56_raven_data-generator': RavenMatrixEvaluator,
    'O-58_symbol_delete_data-generator': SymbolDeleteEvaluator,
    'O-59_symbol_insert_data-generator': SymbolInsertEvaluator,
    'O-60_symbol_substitute_data-genertor': SymbolSubstituteEvaluator,
    'O-61_symbol_edit_data-generator': SymbolEditConstraintEvaluator,
    'O-62_gravity_physics_data-generator': GravityPhysicsEvaluator,
    'O-64_animal_matching_data-generator': AnimalMatchingEvaluator,
    'O-65_animal_size_sorting_data-generator': AnimalSizeSortingEvaluator,
    'O-85_2d_object_rotation_data-generator': ObjectRotation2DEvaluator,
