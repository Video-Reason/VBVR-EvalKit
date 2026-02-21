"""Evaluator for G-54_connecting_color_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-54_connecting_color_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


class ConnectingColorEvaluator(BaseEvaluator):
    """
    G-54: Connecting color evaluator.
    
    STRICT rule-based evaluation for the "connect same color objects with curves" task:
    - Same color objects MUST be connected by same color curves/lines
    - Different color objects should NOT be connected  
    - Original objects should remain in place (not change position significantly)
    
    CRITICAL RULES:
    1. Detect ALL colored objects in the first frame (not just circles)
    2. Count CORRECT connections: same-color objects connected by same-color line
    3. Count WRONG connections: different-color objects connected by any line
    4. Objects should NOT change position or be destroyed
    
    Expected connections:
    - GT sample 0: 2 correct connections (blue pair + orange pair)
    - GT sample 1: 3 correct connections
    
    Evaluates:
    - Object preservation (20%): Original objects still exist at same positions
    - Correct connections (50%): Same-color objects are connected by same-color lines
    - No wrong connections (30%): No lines connecting different colors
    """
    
    TASK_WEIGHTS = {
        'object_preservation': 0.20,
        'correct_connections': 0.50,
        'no_wrong_connections': 0.30
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate connecting color task with proper connection counting."""
        scores = {}
        
        first_frame = video_frames[0] if len(video_frames) > 0 else None
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_first = gt_first_frame
        gt_last = gt_final_frame
        
        if first_frame is None or last_frame is None:
            return 0.0
        
        # Resize frames if needed
        if gt_last is not None and last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        if gt_first is not None and first_frame.shape != gt_first.shape:
            gt_first = cv2.resize(gt_first, (first_frame.shape[1], first_frame.shape[0]))
        
        # Detect circular objects in first frame (before curves are drawn)
        first_objects = self._detect_circular_objects(first_frame)
        
        # Detect objects in last frame
        last_objects = self._detect_circular_objects(last_frame)
        
        # Store detection info for debugging
        scores['num_first_objects'] = len(first_objects)
        scores['num_last_objects'] = len(last_objects)
        
        if len(first_objects) < 2:
            scores['object_preservation'] = 0.0
            scores['correct_connections'] = 0.0
            scores['no_wrong_connections'] = 0.0
            self._last_task_details = scores
            self._last_task_details['error'] = 'not_enough_objects_in_first_frame'
            return 0.0
        
        # 1. Object preservation (20%): Check if objects remain at similar positions
        preservation_score = self._evaluate_object_preservation(first_objects, last_objects)
        scores['object_preservation'] = preservation_score
        
        # Group first frame objects by color
        objects_by_color = {}
        for obj in first_objects:
            objects_by_color.setdefault(obj['color'], []).append(obj)
        
        # Count expected connections (pairs of same-color objects)
        expected_connections = sum(
            len(objs) * (len(objs) - 1) // 2 
            for objs in objects_by_color.values() 
            if len(objs) >= 2
        )
        
        # 2 & 3. Count correct and wrong connections
        correct_count, wrong_count, connection_details = self._count_connections(
            last_frame, first_objects, objects_by_color
        )
        
        # Store connection info
        scores['expected_connections'] = expected_connections
        scores['correct_connections_count'] = correct_count
        scores['wrong_connections_count'] = wrong_count
        
        # 2. Correct connections score (50%)
        if expected_connections > 0:
            scores['correct_connections'] = min(1.0, correct_count / expected_connections)
        else:
            scores['correct_connections'] = 0.0
        
        # 3. No wrong connections score (30%)
        if wrong_count == 0:
            scores['no_wrong_connections'] = 1.0
        else:
            # Each wrong connection heavily penalizes
            scores['no_wrong_connections'] = max(0.0, 1.0 - wrong_count * 0.4)
        
        scores['connection_details'] = connection_details
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _detect_circular_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect colored objects (dots/circles/blobs) in the frame.
        
        NOTE: Lowered circularity threshold to detect more shapes, not just perfect circles.
        The task involves connecting same-color objects which may be various shapes.
        """
        objects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Accept objects in reasonable size range (not too small, not too big)
                if area < 500 or area > 30000:
                    continue
                
                # Check circularity (lowered threshold to accept more shapes)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Accept reasonably compact objects (circularity > 0.3)
                    # This accepts ovals, squares with rounded corners, etc.
                    if circularity > 0.3:
                        M = cv2.moments(contour)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            objects.append({
                                'color': color_name,
                                'center': (cx, cy),
                                'area': area,
                                'circularity': circularity
                            })
        
        return objects
    
    def _evaluate_object_preservation(self, first_objects: List[Dict], 
                                      last_objects: List[Dict]) -> float:
        """Check if original objects are preserved in similar positions."""
        if not first_objects:
            return 0.0
        
        matched = 0
        for first_obj in first_objects:
            # Find matching object in last frame (same color, similar position)
            for last_obj in last_objects:
                if first_obj['color'] == last_obj['color']:
                    dist = safe_distance(first_obj['center'], last_obj['center'])
                    if dist < 50:  # Object within 50 pixels of original position
                        matched += 1
                        break
        
        return matched / len(first_objects)
    
    def _count_connections(self, frame: np.ndarray, first_objects: List[Dict],
                          objects_by_color: Dict) -> Tuple[int, int, Dict]:
        """
        Count correct and wrong line connections.
        
        - Correct: Same-color objects connected by same-color line
        - Wrong: Different-color objects connected by any line
        
        Returns: (correct_count, wrong_count, details)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'green': ([35, 100, 100], [85, 255, 255], None, None),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
        }
        
        correct_connections = []
        wrong_connections = []
        
        # Check correct connections: same-color objects connected by same-color line
        for color_name, objs in objects_by_color.items():
            if len(objs) < 2:
                continue
            
            ranges = color_ranges.get(color_name)
            if not ranges:
                continue
            
            # Create color mask for this color
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            # Check each pair of same-color objects
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    obj1, obj2 = objs[i], objs[j]
                    
                    # Check if there's a continuous path of this color between them
                    if self._has_color_path(mask, obj1['center'], obj2['center']):
                        correct_connections.append({
                            'color': color_name,
                            'obj1': obj1['center'],
                            'obj2': obj2['center']
                        })
        
        # Check wrong connections: different-color objects connected
        colors_list = list(objects_by_color.keys())
        for i in range(len(colors_list)):
            for j in range(i + 1, len(colors_list)):
                color1, color2 = colors_list[i], colors_list[j]
                objs1 = objects_by_color[color1]
                objs2 = objects_by_color[color2]
                
                for obj1 in objs1:
                    for obj2 in objs2:
                        if self._has_any_line_connection(frame, obj1['center'], obj2['center']):
                            wrong_connections.append({
                                'colors': (color1, color2),
                                'obj1': obj1['center'],
                                'obj2': obj2['center']
                            })
        
        details = {
            'correct': correct_connections,
            'wrong': wrong_connections
        }
        
        return len(correct_connections), len(wrong_connections), details
    
    def _has_color_path(self, color_mask: np.ndarray, 
                        p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if there's a continuous path of the specific color between two points.
        Uses line sampling with wider search region and checks for color presence along the path.
        
        NOTE: Lowered threshold to 30% because curves may not follow straight line exactly.
        """
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # Sample points along the line
        num_samples = 30
        connected_count = 0
        
        for t in np.linspace(0.15, 0.85, num_samples):  # Avoid endpoints (object areas)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < color_mask.shape[1] and 0 <= y < color_mask.shape[0]:
                # Check a LARGER region around the sample point (curves may not be straight)
                y_min, y_max = max(0, y-20), min(color_mask.shape[0], y+20)
                x_min, x_max = max(0, x-20), min(color_mask.shape[1], x+20)
                region = color_mask[y_min:y_max, x_min:x_max]
                
                if np.sum(region > 0) > 5:  # Some pixels of this color
                    connected_count += 1
        
        # Connection exists if more than 30% of samples have the color (curves may not follow straight line)
        return connected_count > num_samples * 0.3
    
    def _has_any_line_connection(self, frame: np.ndarray,
                                 p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Check if there's any drawn line connecting two different-colored objects.
        Looks for non-background colored pixels along the path.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # Sample points along the middle section of the line
        num_samples = 20
        colored_count = 0
        
        for t in np.linspace(0.3, 0.7, num_samples):  # Only check middle section
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Check if this pixel is colored (high saturation) and not background
                h, s, v = hsv[y, x]
                if s > 50 and v > 50:  # Colored pixel (not white/black/gray)
                    colored_count += 1
        
        # If more than 40% of middle samples have colored pixels, there's a connection
        return colored_count > num_samples * 0.4
