"""Evaluator for G-5_multi_object_placement_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-5_multi_object_placement_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class MultiObjectPlacementEvaluator(BaseEvaluator):
    """
    G-5: Multi-object placement evaluator.
    
    Rule-based evaluation:
    - Color matching (30%): Objects moved to matching color star markers
    - Alignment precision (25%): Object centers aligned with star centers
    - Path optimality (20%): Straight-line movement paths
    - Object fidelity (15%): Shape, size, color preserved
    - Star invariance (10%): Star markers remain stationary
    """
    
    TASK_WEIGHTS = {
        'color_matching': 0.30,
        'alignment': 0.25,
        'path': 0.20,
        'fidelity': 0.15,
        'star_invariance': 0.10
    }
    
    def _detect_colored_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored objects (shapes) in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects = []
        
        # Define color ranges for common colors
        color_ranges = {
            'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            'blue': [([100, 100, 100], [130, 255, 255])],
            'green': [([35, 100, 100], [85, 255, 255])],
            'yellow': [([20, 100, 100], [35, 255, 255])],
        }
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                objects.append({'color': color_name, 'center': (cx, cy), 'area': area})
        
        return objects
    
    def _detect_star_markers(self, frame: np.ndarray) -> List[Dict]:
        """Detect star markers by looking for small star-shaped objects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        stars = []
        color_ranges = {
            'red': [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
            'blue': [([100, 100, 100], [130, 255, 255])],
            'green': [([35, 100, 100], [85, 255, 255])],
            'yellow': [([20, 100, 100], [35, 255, 255])],
        }
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Stars are typically smaller markers
                if area < 100 or area > 2000:
                    continue
                
                # Check if shape has star-like properties (many vertices)
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) >= 8:  # Star has many vertices
                    M = cv2.moments(cnt)
                    if M['m00'] == 0:
                        continue
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    stars.append({'color': color_name, 'center': (cx, cy), 'area': area})
        
        return stars
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate multi-object placement task."""
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Detect objects and stars in final frames
        gen_objects = self._detect_colored_objects(last_frame)
        gt_objects = self._detect_colored_objects(gt_final_frame)
        
        # 1. Color matching: Check if objects are at correct color-matched positions
        if gen_objects and gt_objects:
            matched = 0
            for gen_obj in gen_objects:
                for gt_obj in gt_objects:
                    if gen_obj['color'] == gt_obj['color']:
                        dist = safe_distance(gen_obj['center'], gt_obj['center'])
                        if dist < 30:  # Within 30 pixels
                            matched += 1
                            break
            scores['color_matching'] = matched / max(len(gt_objects), 1)
        else:
            scores['color_matching'] = 0.2  # Detection failed
        
        # 2. Alignment precision: Compare object positions with GT
        if gen_objects and gt_objects:
            total_dist = 0
            count = 0
            for gen_obj in gen_objects:
                min_dist = float('inf')
                for gt_obj in gt_objects:
                    if gen_obj['color'] == gt_obj['color']:
                        dist = safe_distance(gen_obj['center'], gt_obj['center'])
                        min_dist = min(min_dist, dist)
                if min_dist < float('inf'):
                    total_dist += min_dist
                    count += 1
            avg_dist = total_dist / count if count > 0 else 100
            scores['alignment'] = max(0, 1.0 - avg_dist / 50.0)
        else:
            scores['alignment'] = 0.2  # Detection failed
        
        # 3. Path optimality: Analyze motion smoothness
        if len(video_frames) > 2:
            motion_scores = []
            for i in range(1, min(len(video_frames), 10)):
                diff = cv2.absdiff(video_frames[i], video_frames[i-1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            # Smooth motion should have consistent changes
            if motion_scores:
                variance = np.var(motion_scores)
                scores['path'] = max(0, 1.0 - variance / 1000.0)
            else:
                scores['path'] = 0.2  # Detection failed
        else:
            scores['path'] = 0.2  # Detection failed
        
        # 4. Fidelity: Check object count and area preservation
        first_objects = self._detect_colored_objects(first_frame)
        if first_objects and gen_objects:
            count_ratio = min(len(gen_objects), len(first_objects)) / max(len(gen_objects), len(first_objects), 1)
            first_area = sum(o['area'] for o in first_objects)
            gen_area = sum(o['area'] for o in gen_objects)
            area_ratio = min(first_area, gen_area) / max(first_area, gen_area, 1)
            scores['fidelity'] = 0.5 * count_ratio + 0.5 * area_ratio
        else:
            scores['fidelity'] = 0.2  # Detection failed
        
        # 5. Star invariance: Check if star positions are preserved
        first_stars = self._detect_star_markers(first_frame)
        final_stars = self._detect_star_markers(last_frame)
        if first_stars and final_stars:
            preserved = 0
            for fs in first_stars:
                for ls in final_stars:
                    if fs['color'] == ls['color']:
                        dist = safe_distance(fs['center'], ls['center'])
                        if dist < 20:
                            preserved += 1
                            break
            scores['star_invariance'] = preserved / max(len(first_stars), 1)
        else:
            scores['star_invariance'] = 0.3  # No stars detected, assume OK
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
