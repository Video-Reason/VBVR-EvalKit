"""
Specific evaluators for Open_60 tasks (Part 2).
These evaluators implement rule-based scoring for the second batch of Open_60 tasks.
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


class ChartExtremeEvaluator(BaseEvaluator):
    """
    G-29: Chart extreme with data evaluator.
    
    Evaluates:
    - Extreme identification accuracy (40%): Correct max/min identified
    - Marking correctness (35%): Red rectangle accurately marks target
    - Visual normality (15%): Border complete and accurate
    - Chart fidelity (10%): Chart elements preserved
    """
    
    TASK_WEIGHTS = {
        'extreme_id': 0.40,
        'marking': 0.35,
        'visual': 0.15,
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
        """Evaluate chart extreme task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect red rectangles in both frames
        gen_red_regions = self._detect_red_rectangle(last_frame)
        gt_red_regions = self._detect_red_rectangle(gt_last)
        
        # 1. Extreme identification (40%): Check if red rectangle is in correct location
        extreme_score = self._evaluate_extreme_identification(gen_red_regions, gt_red_regions, last_frame, gt_last)
        scores['extreme_id'] = extreme_score
        
        # 2. Marking correctness (35%): Red rectangle position and completeness
        marking_score = self._evaluate_marking(gen_red_regions, gt_red_regions)
        scores['marking'] = marking_score
        
        # 3. Visual normality (15%): Border completeness
        visual_score = self._evaluate_visual_normality(last_frame, gen_red_regions)
        scores['visual'] = visual_score
        
        # 4. Chart fidelity (10%): Chart elements preserved
        fidelity_score = self._evaluate_chart_fidelity(last_frame, gt_last, gen_red_regions, gt_red_regions)
        scores['fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_red_rectangle(self, frame: np.ndarray) -> List[Dict]:
        """Detect red rectangular borders in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color range (two ranges due to hue wrap-around)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter noise
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's rectangular (not filled)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            # Get center
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            regions.append({
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area,
                'contour': contour,
                'is_border': fill_ratio < 0.5  # Border has low fill ratio
            })
        
        return regions
    
    def _evaluate_extreme_identification(self, gen_regions: List[Dict], gt_regions: List[Dict], 
                                         gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Check if the correct extreme element is marked."""
        if not gen_regions or not gt_regions:
            return 0.0  # STRICT: Must have marking in both
        
        # Find the main red marking in both
        gen_main = max(gen_regions, key=lambda r: r['area'])
        gt_main = max(gt_regions, key=lambda r: r['area'])
        
        # Compare centers
        gen_center = gen_main['center']
        gt_center = gt_main['center']
        
        # Calculate distance normalized by frame size
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        # STRICTER: Score based on distance
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.7
        elif normalized_dist < 0.20:
            return 0.3
        else:
            return 0.0  # STRICT: Wrong position
    
    def _evaluate_marking(self, gen_regions: List[Dict], gt_regions: List[Dict]) -> float:
        """Evaluate red rectangle marking accuracy."""
        if not gen_regions:
            return 0.0
        if not gt_regions:
            return 0.0  # STRICT: No GT to compare
        
        gen_main = max(gen_regions, key=lambda r: r['area'])
        gt_main = max(gt_regions, key=lambda r: r['area'])
        
        # Compare bounding boxes using IoU
        gen_bbox = gen_main['bbox']
        gt_bbox = gt_main['bbox']
        
        # Calculate IoU
        x1 = max(gen_bbox[0], gt_bbox[0])
        y1 = max(gen_bbox[1], gt_bbox[1])
        x2 = min(gen_bbox[0] + gen_bbox[2], gt_bbox[0] + gt_bbox[2])
        y2 = min(gen_bbox[1] + gen_bbox[3], gt_bbox[1] + gt_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        gen_area = gen_bbox[2] * gen_bbox[3]
        gt_area = gt_bbox[2] * gt_bbox[3]
        union = gen_area + gt_area - intersection
        
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _evaluate_visual_normality(self, frame: np.ndarray, red_regions: List[Dict]) -> float:
        """Evaluate if the red border is complete and proper."""
        if not red_regions:
            return 0.0
        
        main_region = max(red_regions, key=lambda r: r['area'])
        
        # Check if it forms a complete rectangle (4 sides)
        contour = main_region['contour']
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # A rectangle should have 4 vertices
        if len(approx) == 4:
            return 1.0
        elif len(approx) >= 3 and len(approx) <= 6:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_chart_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray,
                                  gen_regions: List[Dict], gt_regions: List[Dict]) -> float:
        """Evaluate if chart elements are preserved."""
        # Mask out red regions and compare the rest
        gen_mask = np.ones(gen_frame.shape[:2], dtype=np.uint8) * 255
        gt_mask = np.ones(gt_frame.shape[:2], dtype=np.uint8) * 255
        
        for region in gen_regions:
            cv2.drawContours(gen_mask, [region['contour']], -1, 0, -1)
        for region in gt_regions:
            cv2.drawContours(gt_mask, [region['contour']], -1, 0, -1)
        
        # Compare non-red regions
        combined_mask = gen_mask & gt_mask
        
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        if np.sum(combined_mask > 0) == 0:
            return 0.0  # STRICT: No overlap to compare
        
        diff = np.abs(gen_gray.astype(float) - gt_gray.astype(float))
        masked_diff = diff[combined_mask > 0]
        
        avg_diff = np.mean(masked_diff) / 255.0
        return max(0, 1.0 - avg_diff * 2)


class DirectedGraphNavigationEvaluator(BaseEvaluator):
    """
    G-31: Directed graph navigation evaluator.
    
    CRITICAL RULES:
    1. Blue triangle (agent) must move from green circle to red circle
    2. Agent must reach the red circle (endpoint)
    3. All circle colors (green, red) must NOT change
    """
    
    TASK_WEIGHTS = {
        'completion': 0.35,           # Agent reaches red endpoint
        'circles_preserved': 0.50,    # Circle colors unchanged - MORE IMPORTANT
        'path_quality': 0.15          # Follows graph structure
    }
    
    def _count_circle_colors(self, frame: np.ndarray) -> Tuple[int, int]:
        """Count green and red pixels in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_count = np.sum(green_mask > 0)
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_count = np.sum(red_mask > 0)
        
        return green_count, red_count
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate directed graph navigation task.
        
        CRITICAL RULES:
        1. Agent must reach red endpoint
        2. Circle colors (green, red) must NOT change significantly
        """
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # CRITICAL: Check if circle colors are preserved
        first_green, first_red = self._count_circle_colors(first_frame)
        final_green, final_red = self._count_circle_colors(last_frame)
        
        first_total = first_green + first_red
        final_total = final_green + final_red
        
        # Circle colors should not change dramatically
        total_change = abs(final_total - first_total) / max(first_total, 1)
        
        if total_change > 1.0:  # More than 100% increase
            # Circle colors changed significantly - task failed
            scores['circles_preserved'] = 0.0
            scores['completion'] = 0.0
            scores['path_quality'] = 0.0
            self._last_task_details = scores
            self._last_task_details['circles_changed'] = True
            return 0.0
        else:
            scores['circles_preserved'] = max(0, 1.0 - total_change)
        
        # Detect nodes and agent
        nodes_first = self._detect_nodes(first_frame)
        gen_agent_final = self._detect_agent(last_frame)
        
        # 1. Completion: Check if agent reached red endpoint
        if gen_agent_final is not None and nodes_first.get('end') is not None:
            end_pos = nodes_first['end']
            dist = np.sqrt((gen_agent_final[0] - end_pos[0])**2 + 
                          (gen_agent_final[1] - end_pos[1])**2)
            if dist < 50:
                scores['completion'] = 1.0
            elif dist < 100:
                scores['completion'] = 0.3  # STRICT: Not at endpoint
            else:
                scores['completion'] = 0.0  # STRICT: Failed to reach endpoint
        else:
            scores['completion'] = 0.0
        
        # 2. Path quality: Check if agent followed graph structure
        agent_positions = self._track_agent(video_frames)
        if len(agent_positions) >= 2:
            # Check for smooth movement (no teleporting)
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 150 or dy > 150:
                    large_jumps += 1
            scores['path_quality'] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores['path_quality'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect blue triangular agent position."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest blue region (the agent)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _track_agent(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track agent position across all frames."""
        positions = []
        for frame in frames:
            pos = self._detect_agent(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_nodes(self, frame: np.ndarray) -> Dict:
        """Detect graph nodes (green=start, red=end, white=intermediate)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        nodes = {'start': None, 'end': None, 'intermediate': []}
        
        # Green (start node)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    nodes['start'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                break
        
        # Red (end node)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 200:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    nodes['end'] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                break
        
        return nodes
    
    def _evaluate_path_length(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent took shortest path."""
        if not agent_positions or not nodes.get('end'):
            return 0.0
        
        # Check if agent reached end node
        final_pos = agent_positions[-1]
        end_node = nodes['end']
        
        dist_to_end = np.sqrt((final_pos[0] - end_node[0])**2 + (final_pos[1] - end_node[1])**2)
        
        # If agent reached end (within threshold)
        if dist_to_end < 50:
            # Count number of significant position changes (steps)
            steps = 0
            prev_pos = agent_positions[0]
            for pos in agent_positions[1:]:
                dist = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                if dist > 20:  # Significant movement
                    steps += 1
                    prev_pos = pos
            
            # Fewer steps is better (assuming optimal is 3-5 steps for typical graph)
            if steps <= 5:
                return 1.0
            elif steps <= 7:
                return 0.8
            elif steps <= 10:
                return 0.6
            else:
                return 0.4
        else:
            # Didn't reach end
            return max(0.2, 1.0 - dist_to_end / 500)
    
    def _evaluate_direction_compliance(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent follows arrow directions."""
        if len(agent_positions) < 2:
            return 0.5
        
        # For now, check if movement is generally forward (left to right or top to bottom)
        # This is a simplification; full implementation would need edge detection
        forward_moves = 0
        total_moves = 0
        
        for i in range(1, len(agent_positions)):
            dx = agent_positions[i][0] - agent_positions[i-1][0]
            dy = agent_positions[i][1] - agent_positions[i-1][1]
            
            if abs(dx) > 10 or abs(dy) > 10:  # Significant movement
                total_moves += 1
                # Generally forward progress (towards end)
                if nodes.get('end') and nodes.get('start'):
                    # Check if moving towards end
                    prev_dist = np.sqrt((agent_positions[i-1][0] - nodes['end'][0])**2 + 
                                       (agent_positions[i-1][1] - nodes['end'][1])**2)
                    curr_dist = np.sqrt((agent_positions[i][0] - nodes['end'][0])**2 + 
                                       (agent_positions[i][1] - nodes['end'][1])**2)
                    if curr_dist < prev_dist:
                        forward_moves += 1
        
        return forward_moves / max(1, total_moves)
    
    def _evaluate_movement_legality(self, agent_positions: List[Tuple[int, int]], nodes: Dict) -> float:
        """Evaluate if agent moves along edges (not jumping)."""
        if len(agent_positions) < 2:
            return 0.5
        
        # Check for smooth movement (no large jumps)
        smooth_moves = 0
        total_moves = 0
        
        for i in range(1, len(agent_positions)):
            dist = np.sqrt((agent_positions[i][0] - agent_positions[i-1][0])**2 + 
                          (agent_positions[i][1] - agent_positions[i-1][1])**2)
            
            if dist > 5:  # Significant movement
                total_moves += 1
                if dist < 100:  # Reasonable step size
                    smooth_moves += 1
        
        return smooth_moves / max(1, total_moves)
    
    def _evaluate_graph_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if graph structure is preserved."""
        # Compare edge structures using edge detection
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)
        
        # Compare edge maps
        intersection = np.sum((gen_edges > 0) & (gt_edges > 0))
        union = np.sum((gen_edges > 0) | (gt_edges > 0))
        
        if union > 0:
            return intersection / union
        return 0.5


class AttentionShiftEvaluator(BaseEvaluator):
    """
    G-39: Attention shift different evaluator.
    
    CRITICAL RULES:
    1. Two objects should NOT change (remain stationary)
    2. Green box (框框) should move from one object to another
    3. Final frame must have green box around the other object
    """
    
    TASK_WEIGHTS = {
        'objects_preserved': 0.45,    # Two objects unchanged
        'green_box_shifted': 0.40,    # Green box moved to other object
        'box_fidelity': 0.15          # Green box maintained throughout
    }
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count non-green colored objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        
        # High saturation but not green
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        sat_mask = (sat > 50).astype(np.uint8) * 255
        obj_mask = cv2.bitwise_and(sat_mask, cv2.bitwise_not(green_mask))
        
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate attention shift task.
        
        CRITICAL RULES:
        1. Objects must be preserved (2 objects in both first and final frame)
        2. Green box must shift to the other object
        """
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # 1. CRITICAL: Objects must be preserved
        first_obj_count = self._count_objects(first_frame)
        final_obj_count = self._count_objects(last_frame)
        
        if first_obj_count == 0:
            first_obj_count = 2  # Assume 2 objects
        
        if final_obj_count == 0:
            # No objects detected in final frame - complete failure
            scores['objects_preserved'] = 0.0
            scores['green_box_shifted'] = 0.0
            scores['box_fidelity'] = 0.0
            self._last_task_details = scores
            self._last_task_details['objects_disappeared'] = True
            return 0.0
        
        # Check if object count is preserved
        if final_obj_count >= first_obj_count:
            scores['objects_preserved'] = 1.0
        elif final_obj_count == first_obj_count - 1:
            scores['objects_preserved'] = 0.5
        else:
            scores['objects_preserved'] = 0.0
        
        # 2. Green box must be present and shifted
        gen_box = self._detect_green_box(last_frame)
        gt_box = self._detect_green_box(gt_last)
        
        if gen_box is None:
            # No green box in final frame - failed
            scores['green_box_shifted'] = 0.0
        elif gt_box is None:
            scores['green_box_shifted'] = 0.0  # STRICT: No GT box to compare
        else:
            # Compare box centers
            gen_center = gen_box['center']
            gt_center = gt_box['center']
            
            frame_diag = np.sqrt(last_frame.shape[0]**2 + last_frame.shape[1]**2)
            distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
            normalized_dist = distance / frame_diag
            
            if normalized_dist < 0.05:
                scores['green_box_shifted'] = 1.0
            elif normalized_dist < 0.10:
                scores['green_box_shifted'] = 0.8
            elif normalized_dist < 0.20:
                scores['green_box_shifted'] = 0.5
            else:
                scores['green_box_shifted'] = 0.2
        
        # 3. Box fidelity: Green box maintained throughout
        box_fidelity_score = self._evaluate_box_fidelity(video_frames)
        scores['box_fidelity'] = box_fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_green_box(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect green attention box."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest green region (attention box)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(largest)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        return {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'area': area,
            'contour': largest
        }
    
    def _detect_objects_excluding_green(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects excluding green attention box."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mask out green
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Convert to grayscale and find objects
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[green_mask > 0] = 255  # Remove green regions
        
        # Find non-white regions (objects)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
            
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                objects.append({
                    'center': (cx, cy),
                    'area': area,
                    'contour': contour
                })
        
        return objects
    
    def _evaluate_attention_transfer(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if attention box transferred to correct position."""
        gen_box = self._detect_green_box(gen_frame)
        gt_box = self._detect_green_box(gt_frame)
        
        if gen_box is None:
            return 0.0
        if gt_box is None:
            return 0.5
        
        # Compare box centers
        gen_center = gen_box['center']
        gt_center = gt_box['center']
        
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.8
        elif normalized_dist < 0.20:
            return 0.6
        else:
            return max(0.2, 1.0 - normalized_dist)
    
    def _evaluate_stationarity(self, frames: List[np.ndarray]) -> float:
        """Evaluate if objects remain stationary throughout video."""
        if len(frames) < 2:
            return 0.5
        
        first_objects = self._detect_objects_excluding_green(frames[0])
        
        if not first_objects:
            return 0.5
        
        # Track objects through video
        max_movement = 0
        
        for frame in frames[::max(1, len(frames) // 10)]:  # Sample frames
            curr_objects = self._detect_objects_excluding_green(frame)
            
            for first_obj in first_objects:
                # Find closest matching object
                min_dist = float('inf')
                for curr_obj in curr_objects:
                    dist = safe_distance(first_obj['center'], curr_obj['center'])
                    min_dist = min(min_dist, dist)
                
                if min_dist < float('inf'):
                    max_movement = max(max_movement, min_dist)
        
        # Score based on maximum movement
        frame_diag = np.sqrt(frames[0].shape[0]**2 + frames[0].shape[1]**2)
        normalized_movement = max_movement / frame_diag
        
        if normalized_movement < 0.03:
            return 1.0
        elif normalized_movement < 0.05:
            return 0.8
        elif normalized_movement < 0.10:
            return 0.6
        else:
            return max(0.2, 1.0 - normalized_movement * 2)
    
    def _evaluate_box_fidelity(self, frames: List[np.ndarray]) -> float:
        """Evaluate if green attention box is maintained throughout."""
        box_present_count = 0
        single_box_count = 0
        
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            if valid_contours:
                box_present_count += 1
                if len(valid_contours) == 1:
                    single_box_count += 1
        
        presence_ratio = box_present_count / len(frames)
        single_ratio = single_box_count / max(1, box_present_count)
        
        return 0.6 * presence_ratio + 0.4 * single_ratio
    
    def _evaluate_transfer_quality(self, frames: List[np.ndarray]) -> float:
        """Evaluate smoothness of attention transfer."""
        if len(frames) < 3:
            return 0.5
        
        # Track green box positions
        positions = []
        for frame in frames:
            box = self._detect_green_box(frame)
            if box:
                positions.append(box['center'])
        
        if len(positions) < 3:
            return 0.5
        
        # Check for smooth movement (consistent velocity)
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))
        
        if len(velocities) < 2:
            return 0.5
        
        # Lower variance = smoother movement
        mean_vel = np.mean(velocities)
        std_vel = np.std(velocities)
        
        if mean_vel < 1:
            return 0.5  # No significant movement
        
        cv = std_vel / mean_vel  # Coefficient of variation
        
        return max(0.3, 1.0 - cv)


class GridHighestCostEvaluator(BaseEvaluator):
    """
    G-41: Grid highest cost path evaluator.
    
    CRITICAL RULES:
    1. Yellow dot (Pacman) must move from green grid to red grid
    2. Pacman must reach the red goal cell
    3. Grid structure (colors) must NOT change significantly
    4. Should follow high-cost path (follow GT route)
    """
    
    TASK_WEIGHTS = {
        'completion': 0.45,       # Pacman reaches red goal
        'grid_preserved': 0.35,   # Grid colors unchanged
        'movement': 0.20          # Step by step movement
    }
    
    def _count_grid_colors(self, frame: np.ndarray) -> Tuple[int, int]:
        """Count green and red pixels in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_count = np.sum(green_mask > 0)
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_count = np.sum(red_mask > 0)
        
        return green_count, red_count
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate grid highest cost task.
        
        CRITICAL RULES:
        1. Pacman must reach red goal
        2. Grid colors must be preserved
        """
        scores = {}
        
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        gt_last = gt_final_frame
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # CRITICAL: Check if grid colors are preserved
        first_green, first_red = self._count_grid_colors(first_frame)
        final_green, final_red = self._count_grid_colors(last_frame)
        
        first_total = first_green + first_red
        final_total = final_green + final_red
        
        total_change = abs(final_total - first_total) / max(first_total, 1)
        
        if total_change > 1.0:  # More than 100% increase
            scores['grid_preserved'] = 0.0
            scores['completion'] = 0.0
            scores['movement'] = 0.0
            self._last_task_details = scores
            self._last_task_details['grid_changed'] = True
            return 0.0
        else:
            scores['grid_preserved'] = max(0, 1.0 - total_change)
        
        # Detect Pacman and red goal
        agent = self._detect_pacman(last_frame)
        goal = self._detect_red_goal(first_frame)
        
        # 1. Completion: Check if Pacman reached red goal
        if agent is not None and goal is not None:
            dist = np.sqrt((agent[0] - goal[0])**2 + (agent[1] - goal[1])**2)
            if dist < 50:
                scores['completion'] = 1.0
            elif dist < 100:
                scores['completion'] = 0.5
            else:
                scores['completion'] = 0.1
        else:
            scores['completion'] = 0.0
        
        # 2. Movement: Check for step-by-step movement
        agent_positions = self._track_pacman(video_frames)
        if len(agent_positions) >= 2:
            large_jumps = 0
            for i in range(1, len(agent_positions)):
                dx = abs(agent_positions[i][0] - agent_positions[i-1][0])
                dy = abs(agent_positions[i][1] - agent_positions[i-1][1])
                if dx > 150 or dy > 150:
                    large_jumps += 1
            scores['movement'] = max(0, 1.0 - large_jumps * 0.3)
        else:
            scores['movement'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_red_goal(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red goal cell."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    def _detect_pacman(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect yellow Pac-Man agent."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow color range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def _track_pacman(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track Pac-Man position across frames."""
        positions = []
        for frame in frames:
            pos = self._detect_pacman(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_grid(self, frame: np.ndarray) -> Dict:
        """Detect 4x4 grid structure."""
        h, w = frame.shape[:2]
        
        # Assume 4x4 grid
        cell_w = w // 4
        cell_h = h // 4
        
        return {
            'rows': 4,
            'cols': 4,
            'cell_width': cell_w,
            'cell_height': cell_h
        }
    
    def _pos_to_cell(self, pos: Tuple[int, int], grid_info: Dict) -> Tuple[int, int]:
        """Convert pixel position to grid cell."""
        col = pos[0] // grid_info['cell_width']
        row = pos[1] // grid_info['cell_height']
        return (min(row, grid_info['rows'] - 1), min(col, grid_info['cols'] - 1))
    
    def _evaluate_path_cost(self, positions: List[Tuple[int, int]], grid_info: Dict, 
                            gt_frame: np.ndarray) -> float:
        """Evaluate if path has high cost."""
        if not positions:
            return 0.0
        
        # Get unique cells visited
        cells_visited = set()
        for pos in positions:
            cell = self._pos_to_cell(pos, grid_info)
            cells_visited.add(cell)
        
        # More cells visited generally means higher cost path (for max cost problem)
        # Optimal path should visit many high-value cells
        max_possible_cells = grid_info['rows'] * grid_info['cols']
        coverage = len(cells_visited) / max_possible_cells
        
        # For highest cost path, we expect more cells to be visited
        if coverage > 0.7:
            return 1.0
        elif coverage > 0.5:
            return 0.8
        elif coverage > 0.3:
            return 0.6
        else:
            return max(0.3, coverage * 2)
    
    def _evaluate_movement_legality(self, positions: List[Tuple[int, int]]) -> float:
        """Evaluate if movements are orthogonal (no diagonal)."""
        if len(positions) < 2:
            return 0.5
        
        legal_moves = 0
        total_moves = 0
        
        for i in range(1, len(positions)):
            dx = abs(positions[i][0] - positions[i-1][0])
            dy = abs(positions[i][1] - positions[i-1][1])
            
            if dx > 5 or dy > 5:  # Significant movement
                total_moves += 1
                # Orthogonal: one direction dominates
                if dx > 3 * dy or dy > 3 * dx:
                    legal_moves += 1
        
        return legal_moves / max(1, total_moves)
    
    def _evaluate_completeness(self, gen_frame: np.ndarray, gt_frame: np.ndarray, 
                               positions: List[Tuple[int, int]]) -> float:
        """Evaluate if agent reached destination (bottom-right)."""
        if not positions:
            return 0.0
        
        final_pos = positions[-1]
        h, w = gen_frame.shape[:2]
        
        # Destination is bottom-right corner
        dest = (w * 0.75, h * 0.75)  # Approximate center of bottom-right cell
        
        dist = np.sqrt((final_pos[0] - dest[0])**2 + (final_pos[1] - dest[1])**2)
        frame_diag = np.sqrt(h**2 + w**2)
        normalized_dist = dist / frame_diag
        
        if normalized_dist < 0.15:
            return 1.0
        elif normalized_dist < 0.25:
            return 0.7
        elif normalized_dist < 0.40:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_grid_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if grid structure is preserved."""
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect grid lines using edge detection
        gen_edges = cv2.Canny(gen_gray, 50, 150)
        gt_edges = cv2.Canny(gt_gray, 50, 150)
        
        # Compare edge density
        gen_edge_density = np.sum(gen_edges > 0) / gen_edges.size
        gt_edge_density = np.sum(gt_edges > 0) / gt_edges.size
        
        if gt_edge_density > 0:
            ratio = gen_edge_density / gt_edge_density
            return min(1.0, max(0.3, 1.0 - abs(1.0 - ratio)))
        
        return 0.5


class UnderstandSceneStructureEvaluator(BaseEvaluator):
    """
    G-43: Understand scene structure evaluator.
    
    Evaluates:
    - Room identification correctness (50%): Correct room type identified
    - Marking accuracy (30%): Green box accurately marks target room
    - Visual normality (15%): Border complete and accurate
    - Scene fidelity (5%): Floorplan preserved
    """
    
    TASK_WEIGHTS = {
        'room_id': 0.50,
        'marking': 0.30,
        'visual': 0.15,
        'fidelity': 0.05
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate scene structure task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect green markings
        gen_green = self._detect_green_marking(last_frame)
        gt_green = self._detect_green_marking(gt_last)
        
        # 1. Room identification (50%): Check if correct room is marked
        room_score = self._evaluate_room_identification(gen_green, gt_green, last_frame, gt_last)
        scores['room_id'] = room_score
        
        # 2. Marking accuracy (30%): Green box position
        marking_score = self._evaluate_marking_accuracy(gen_green, gt_green)
        scores['marking'] = marking_score
        
        # 3. Visual normality (15%): Border completeness
        visual_score = self._evaluate_visual_normality(gen_green)
        scores['visual'] = visual_score
        
        # 4. Scene fidelity (5%): Floorplan preserved
        fidelity_score = self._evaluate_scene_fidelity(last_frame, gt_last, gen_green, gt_green)
        scores['fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_green_marking(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect green rectangular marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(largest)
        M = cv2.moments(largest)
        
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        return {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'area': area,
            'contour': largest
        }
    
    def _evaluate_room_identification(self, gen_green: Optional[Dict], gt_green: Optional[Dict],
                                      gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if correct room is identified."""
        if gen_green is None:
            return 0.0
        if gt_green is None:
            return 0.0  # STRICT: No GT to compare
        
        # Compare marked regions
        gen_center = gen_green['center']
        gt_center = gt_green['center']
        
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        # STRICTER: Score based on distance
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.7
        elif normalized_dist < 0.20:
            return 0.3
        else:
            return 0.0  # STRICT: Wrong room marked
    
    def _evaluate_marking_accuracy(self, gen_green: Optional[Dict], gt_green: Optional[Dict]) -> float:
        """Evaluate green marking accuracy using IoU."""
        if gen_green is None:
            return 0.0
        if gt_green is None:
            return 0.0  # STRICT: No GT to compare
        
        gen_bbox = gen_green['bbox']
        gt_bbox = gt_green['bbox']
        
        # Calculate IoU
        x1 = max(gen_bbox[0], gt_bbox[0])
        y1 = max(gen_bbox[1], gt_bbox[1])
        x2 = min(gen_bbox[0] + gen_bbox[2], gt_bbox[0] + gt_bbox[2])
        y2 = min(gen_bbox[1] + gen_bbox[3], gt_bbox[1] + gt_bbox[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        gen_area = gen_bbox[2] * gen_bbox[3]
        gt_area = gt_bbox[2] * gt_bbox[3]
        union = gen_area + gt_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _evaluate_visual_normality(self, gen_green: Optional[Dict]) -> float:
        """Evaluate if marking is a proper rectangle."""
        if gen_green is None:
            return 0.0
        
        contour = gen_green['contour']
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            return 1.0
        elif len(approx) >= 3 and len(approx) <= 6:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_scene_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray,
                                  gen_green: Optional[Dict], gt_green: Optional[Dict]) -> float:
        """Evaluate if floorplan is preserved."""
        # Mask out green regions and compare
        gen_mask = np.ones(gen_frame.shape[:2], dtype=np.uint8) * 255
        gt_mask = np.ones(gt_frame.shape[:2], dtype=np.uint8) * 255
        
        if gen_green:
            cv2.drawContours(gen_mask, [gen_green['contour']], -1, 0, -1)
        if gt_green:
            cv2.drawContours(gt_mask, [gt_green['contour']], -1, 0, -1)
        
        combined_mask = gen_mask & gt_mask
        
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        if np.sum(combined_mask > 0) == 0:
            return 0.5
        
        diff = np.abs(gen_gray.astype(float) - gt_gray.astype(float))
        masked_diff = diff[combined_mask > 0]
        
        avg_diff = np.mean(masked_diff) / 255.0
        return max(0, 1.0 - avg_diff * 2)


class KeyDoorMatchingEvaluator(BaseEvaluator):
    """
    G-45: Key door matching evaluator.
    
    STRICT rule-based evaluation:
    - Agent (yellow/green dot) MUST physically move through the maze
    - Agent MUST move TO a key position to collect it (not just disappear)
    - Agent MUST move TO a door position after collecting key
    - Key color MUST match door color
    
    CRITICAL RULES:
    1. If agent doesn't move from starting position at all, score = 0
    2. Key collected ONLY if agent physically reached key's coordinates BEFORE key disappeared
    3. Door reached ONLY if agent physically reached door's coordinates AFTER collecting key
    
    Evaluates:
    - Agent movement (30%): Agent must PHYSICALLY move away from start position
    - Key collection (35%): Agent moved TO key position AND key disappeared
    - Door reached (25%): Agent moved TO door position after getting key
    - Sequence (10%): Correct order: key first, then door
    """
    
    TASK_WEIGHTS = {
        'agent_movement': 0.30,   # Critical: must physically move away from start
        'key_collected': 0.35,    # Must reach key position AND key disappears
        'door_reached': 0.25,     # Must reach door position after getting key
        'sequence': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate key door matching task with strict movement tracking."""
        scores = {}
        
        if len(video_frames) < 2:
            return 0.0
        
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # Track agent (yellow or green dot) across all frames
        agent_positions = self._track_agent(video_frames)
        
        if len(agent_positions) < 2:
            self._last_task_details = {
                'agent_movement': 0, 'key_collected': 0, 
                'door_reached': 0, 'sequence': 0, 
                'positions_found': len(agent_positions),
                'error': 'not_enough_agent_positions'
            }
            return 0.0
        
        # Detect keys and doors in first frame
        first_keys = self._detect_keys(first_frame)
        first_doors = self._detect_doors(first_frame)
        last_keys = self._detect_keys(last_frame)
        last_doors = self._detect_doors(last_frame)
        
        # Store detection info for debugging
        scores['num_first_keys'] = len(first_keys)
        scores['num_first_doors'] = len(first_doors)
        scores['num_last_keys'] = len(last_keys)
        scores['num_positions'] = len(agent_positions)
        
        # 1. CRITICAL: Agent movement (30%) - Agent must physically move
        movement_score, total_movement = self._evaluate_agent_movement(agent_positions)
        scores['agent_movement'] = movement_score
        scores['total_movement_distance'] = total_movement
        
        # If agent doesn't move, entire task fails
        if movement_score < 0.3:
            scores['key_collected'] = 0.0
            scores['door_reached'] = 0.0
            scores['sequence'] = 0.0
            self._last_task_details = scores
            self._last_task_details['failure_reason'] = 'agent_not_moving'
            return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
        
        # 2. Key collection (35%): Agent must move TO key position AND key disappears
        key_collected, collected_color, key_visit_score = self._check_key_collected_with_visit(
            first_keys, last_keys, agent_positions, video_frames
        )
        scores['key_collected'] = key_visit_score
        scores['collected_key_color'] = collected_color
        
        # 3. Door reached (25%): Agent must move TO door position (after getting key)
        door_reached, door_color, door_visit_score = self._check_door_reached_with_visit(
            first_doors, agent_positions, video_frames, key_collected
        )
        
        # Door only counts if key was collected first AND colors match
        if door_reached and key_collected:
            if collected_color and door_color and collected_color == door_color:
                scores['door_reached'] = door_visit_score  # Good: matching colors
            elif collected_color is None or door_color is None:
                scores['door_reached'] = door_visit_score * 0.5  # Partial: can't verify color match
            else:
                scores['door_reached'] = 0.0  # Wrong color key for door - FAIL
        elif door_reached and not key_collected:
            scores['door_reached'] = 0.0  # Reached door without key - FAIL
        else:
            scores['door_reached'] = 0.0
        
        scores['door_color'] = door_color
        
        # 4. Sequence (10%): Key collected before door reached
        sequence_score = self._evaluate_sequence_with_tracking(video_frames, first_keys, first_doors, agent_positions)
        scores['sequence'] = sequence_score if key_collected else 0.0
        
        self._last_task_details = scores
        return sum(scores.get(k, 0) * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS if k in scores)
    
    def _evaluate_agent_movement(self, positions: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        STRICT evaluation: Agent MUST physically move away from starting position.
        
        CRITICAL: If agent stays at/near starting position, score = 0
        The key-door task requires the agent to navigate through the maze.
        
        Returns: (movement_score, total_distance, max_dist_from_start)
        """
        if len(positions) < 2:
            return 0.0, 0.0
        
        start_pos = positions[0]
        
        # Calculate max distance agent ever reached from start position
        max_dist_from_start = 0.0
        for pos in positions:
            dist = safe_distance(pos, start_pos)
            max_dist_from_start = max(max_dist_from_start, dist)
        
        # CRITICAL CHECK: If agent never moved more than 30 pixels from start, FAIL
        if max_dist_from_start < 30:
            return 0.0, 0.0  # Agent stayed at starting position = COMPLETE FAILURE
        
        # Calculate total movement distance (for debugging)
        total_distance = 0.0
        for i in range(1, len(positions)):
            dist = safe_distance(positions[i], positions[i-1])
            total_distance += dist
        
        # Calculate displacement from start to end
        end_pos = positions[-1]
        displacement = safe_distance(start_pos, end_pos)
        
        # Score based on how far agent traveled from start
        # Agent should travel a significant distance to reach keys/doors
        if max_dist_from_start > 150:
            return 1.0, total_distance  # Agent traveled far - good movement
        elif max_dist_from_start > 100:
            return 0.8, total_distance
        elif max_dist_from_start > 60:
            return 0.6, total_distance
        elif max_dist_from_start > 30:
            return 0.3, total_distance  # Minimal movement
        else:
            return 0.0, total_distance  # No real movement
    
    def _check_key_collected_with_visit(
        self, first_keys: List[Dict], last_keys: List[Dict],
        positions: List[Tuple[int, int]], frames: List[np.ndarray]
    ) -> Tuple[bool, Optional[str], float]:
        """
        STRICT CHECK: Key is ONLY collected if:
        1. Agent PHYSICALLY reached the key's position (within threshold)
        2. Key disappeared from that position
        
        CRITICAL: Key disappearance WITHOUT agent visit = 0 score (not valid collection)
        
        Returns: (collected, color, score)
        """
        if not first_keys or not positions:
            return False, None, 0.0
        
        # Calculate starting position (first detected agent position)
        start_pos = positions[0]
        
        # For each key, check if agent ever reached it
        for key in first_keys:
            key_pos = key['center']
            key_color = key['color']
            
            # Find minimum distance agent got to this key
            min_dist_to_key = float('inf')
            visit_frame_idx = -1
            for i, pos in enumerate(positions):
                dist = safe_distance(pos, key_pos)
                if dist < min_dist_to_key:
                    min_dist_to_key = dist
                    visit_frame_idx = i
            
            # CRITICAL: Agent must have PHYSICALLY moved to the key position
            # Key position must be significantly different from start position
            key_dist_from_start = safe_distance(key_pos, start_pos)
            
            # Agent must have actually traveled toward the key (not stayed at start)
            if min_dist_to_key < 50 and key_dist_from_start > 50:
                # Agent reached this key's position
                # Now check if key disappeared
                key_still_exists = False
                for last_key in last_keys:
                    if last_key['color'] == key_color:
                        dist = safe_distance(last_key['center'], key_pos)
                        if dist < 50:  # Key still at same position
                            key_still_exists = True
                            break
                
                if not key_still_exists:
                    # Perfect: agent visited key AND key disappeared
                    return True, key_color, 1.0
                else:
                    # Agent visited key position but key didn't disappear
                    # Still give partial credit for reaching the key
                    return True, key_color, 0.6
        
        # STRICT: If agent never reached any key position, check if keys disappeared
        # This should NOT give credit (agent didn't do the work)
        first_key_colors = {}
        for key in first_keys:
            first_key_colors[key['color']] = first_key_colors.get(key['color'], 0) + 1
        
        last_key_colors = {}
        for key in last_keys:
            last_key_colors[key['color']] = last_key_colors.get(key['color'], 0) + 1
        
        for color, count in first_key_colors.items():
            if last_key_colors.get(color, 0) < count:
                # Key disappeared but agent NEVER reached it
                # This is WRONG - give 0 credit
                return False, None, 0.0
        
        # No key collection detected
        return False, None, 0.0
    
    def _check_door_reached_with_visit(
        self, doors: List[Dict], positions: List[Tuple[int, int]],
        frames: List[np.ndarray], key_collected: bool
    ) -> Tuple[bool, Optional[str], float]:
        """
        STRICT CHECK: Door is ONLY considered reached if:
        1. Agent PHYSICALLY moved to the door's position (not just stayed at start)
        2. This happened AFTER agent collected the key
        
        Returns: (reached, door_color, score)
        """
        if not doors or not positions:
            return False, None, 0.0
        
        start_pos = positions[0]
        
        best_door = None
        best_dist = float('inf')
        
        for door in doors:
            door_pos = door['center']
            
            # CRITICAL: Door must be at a different position than start
            door_dist_from_start = safe_distance(door_pos, start_pos)
            if door_dist_from_start < 50:
                continue  # Door is at starting position - skip
            
            # Check if agent visited this door position (especially in later frames)
            for pos in positions[-len(positions)//2:]:  # Check second half of trajectory
                dist = safe_distance(pos, door_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_door = door
        
        if best_door is None:
            return False, None, 0.0
        
        # CRITICAL: Agent must have actually moved toward the door
        # (best_dist should be small, meaning agent got close to door)
        door_dist_from_start = safe_distance(best_door['center'], start_pos)
        
        # Agent must have traveled away from start to reach the door
        if door_dist_from_start < 50:
            return False, None, 0.0  # Door is too close to start
        
        # Score based on how close agent got to door
        if best_dist < 40:
            return True, best_door['color'], 1.0  # Very close to door
        elif best_dist < 70:
            return True, best_door['color'], 0.7  # Reasonably close
        elif best_dist < 100:
            return True, best_door['color'], 0.4  # Somewhat close
        
        return False, None, 0.0
    
    def _evaluate_sequence_with_tracking(
        self, frames: List[np.ndarray], 
        keys: List[Dict], doors: List[Dict],
        positions: List[Tuple[int, int]]
    ) -> float:
        """
        Evaluate if key was visited BEFORE door was visited.
        Finds the closest key and door to the agent path and compares visit times.
        """
        if not keys or not doors or len(positions) < 2:
            return 0.0
        
        # Find when agent got closest to each key
        key_visits = []
        for key in keys:
            min_dist = float('inf')
            visit_frame = -1
            for i, pos in enumerate(positions):
                dist = safe_distance(pos, key['center'])
                if dist < min_dist:
                    min_dist = dist
                    visit_frame = i
            if min_dist < 80:  # Consider visited if within 80 pixels
                key_visits.append((key['color'], visit_frame, min_dist))
        
        # Find when agent got closest to each door
        door_visits = []
        for door in doors:
            min_dist = float('inf')
            visit_frame = -1
            for i, pos in enumerate(positions):
                dist = safe_distance(pos, door['center'])
                if dist < min_dist:
                    min_dist = dist
                    visit_frame = i
            if min_dist < 80:  # Consider visited if within 80 pixels
                door_visits.append((door['color'], visit_frame, min_dist))
        
        if not key_visits or not door_visits:
            return 0.3  # Partial credit: some movement detected
        
        # Check if any key was visited before any door
        earliest_key_frame = min(v[1] for v in key_visits)
        earliest_door_frame = min(v[1] for v in door_visits)
        
        if earliest_key_frame < earliest_door_frame:
            return 1.0  # Correct: visited a key before any door
        elif earliest_door_frame < earliest_key_frame:
            return 0.2  # Wrong: visited door first
        else:
            return 0.5  # Same frame (unlikely but handle it)
    
    def _detect_agent(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect agent dot - GREEN circular dot (primary) or yellow (fallback)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try GREEN first - most common agent color in key-door tasks
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for small roughly circular agents
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 100 < area < 4000:  # Small to medium size
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:  # Roughly circular
                        M = cv2.moments(c)
                        if M['m00'] > 0:
                            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        # Fallback to yellow if green not found
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if 100 < area < 4000:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:
                        M = cv2.moments(c)
                        if M['m00'] > 0:
                            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        return None
    
    def _track_agent(self, frames: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Track agent position across frames."""
        positions = []
        for frame in frames:
            pos = self._detect_agent(frame)
            if pos:
                positions.append(pos)
        return positions
    
    def _detect_keys(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect diamond-shaped keys of various colors.
        Keys are filled shapes (high fill ratio) with 4 vertices.
        """
        keys = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1
                
                # Keys are FILLED shapes (high fill ratio > 0.7)
                if fill_ratio < 0.7:
                    continue
                
                # Check if roughly diamond-shaped (4 vertices)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:  # Diamond-like shapes
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        keys.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'area': area,
                            'fill_ratio': fill_ratio
                        })
        
        return keys
    
    def _detect_doors(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect hollow rectangular doors.
        Doors are HOLLOW shapes (low fill ratio < 0.6) with 4 vertices.
        """
        doors = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255], None, None),
            'yellow': ([20, 100, 100], [35, 255, 255], None, None),
            'purple': ([130, 100, 100], [160, 255, 255], None, None),
            'cyan': ([85, 100, 100], [100, 255, 255], None, None),
            'orange': ([10, 100, 100], [20, 255, 255], None, None),
        }
        
        for color_name, ranges in color_ranges.items():
            lower1, upper1, lower2, upper2 = ranges
            mask = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            if lower2 is not None:
                mask |= cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 1
                
                # Doors are HOLLOW shapes (low fill ratio < 0.6)
                if fill_ratio >= 0.6:
                    continue
                
                # Check if roughly rectangular (4 vertices)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 3 <= len(approx) <= 6:  # Rectangular-like shapes
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        doors.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'area': area,
                            'fill_ratio': fill_ratio
                        })
        
        return doors
    


class PredictNextColorEvaluator(BaseEvaluator):
    """
    G-51: Predict next color evaluator.
    
    Evaluates:
    - Pattern identification accuracy (50%): Correct pattern recognized
    - Answer color accuracy (30%): Correct color predicted
    - Visual presentation quality (15%): Proper block style
    - Task understanding (5%): Only fill position 5
    """
    
    TASK_WEIGHTS = {
        'pattern': 0.50,
        'color': 0.30,
        'visual': 0.15,
        'understanding': 0.05
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate predict next color task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect color blocks
        gen_blocks = self._detect_color_blocks(last_frame)
        gt_blocks = self._detect_color_blocks(gt_last)
        
        # 1. Pattern identification (50%): Check if answer matches GT pattern
        pattern_score = self._evaluate_pattern(gen_blocks, gt_blocks)
        scores['pattern'] = pattern_score
        
        # 2. Color accuracy (30%): Check 5th block color
        color_score = self._evaluate_color_accuracy(gen_blocks, gt_blocks)
        scores['color'] = color_score
        
        # 3. Visual presentation (15%): Block style consistency
        visual_score = self._evaluate_visual_presentation(gen_blocks)
        scores['visual'] = visual_score
        
        # 4. Task understanding (5%): Only 5th position filled
        understanding_score = self._evaluate_task_understanding(gen_blocks, gt_blocks)
        scores['understanding'] = understanding_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_color_blocks(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored blocks in the frame."""
        blocks = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
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
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Filter noise
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if roughly square (block shape)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 <= aspect_ratio <= 1.4:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        blocks.append({
                            'color': color_name,
                            'center': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area
                        })
        
        # Sort by x position (left to right)
        blocks.sort(key=lambda b: b['center'][0])
        
        return blocks
    
    def _evaluate_pattern(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if the pattern was correctly identified."""
        if len(gen_blocks) < 5 or len(gt_blocks) < 5:
            return 0.0
        
        # Compare the 5th block (answer)
        gen_5th = gen_blocks[4] if len(gen_blocks) > 4 else None
        gt_5th = gt_blocks[4] if len(gt_blocks) > 4 else None
        
        if gen_5th is None or gt_5th is None:
            return 0.0
        
        if gen_5th['color'] == gt_5th['color']:
            return 1.0
        
        return 0.0
    
    def _evaluate_color_accuracy(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if the predicted color is correct."""
        if len(gen_blocks) < 5 or len(gt_blocks) < 5:
            return 0.0
        
        gen_5th = gen_blocks[4] if len(gen_blocks) > 4 else None
        gt_5th = gt_blocks[4] if len(gt_blocks) > 4 else None
        
        if gen_5th is None or gt_5th is None:
            return 0.0
        
        if gen_5th['color'] == gt_5th['color']:
            return 1.0
        
        # Partial credit for similar colors
        similar_colors = {
            ('red', 'orange'): 0.3,
            ('orange', 'yellow'): 0.3,
            ('blue', 'purple'): 0.3,
            ('green', 'blue'): 0.2,
        }
        
        color_pair = tuple(sorted([gen_5th['color'], gt_5th['color']]))
        return similar_colors.get(color_pair, 0.0)
    
    def _evaluate_visual_presentation(self, gen_blocks: List[Dict]) -> float:
        """Evaluate visual consistency of blocks."""
        if len(gen_blocks) < 5:
            return 0.0
        
        # Check if all blocks have similar size
        areas = [b['area'] for b in gen_blocks[:5]]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        if mean_area > 0:
            cv = std_area / mean_area
            return max(0.3, 1.0 - cv)
        
        return 0.5
    
    def _evaluate_task_understanding(self, gen_blocks: List[Dict], gt_blocks: List[Dict]) -> float:
        """Evaluate if only position 5 was filled."""
        # Check if first 4 blocks match GT (unchanged)
        if len(gen_blocks) < 4 or len(gt_blocks) < 4:
            return 0.0
        
        matches = 0
        for i in range(4):
            if gen_blocks[i]['color'] == gt_blocks[i]['color']:
                matches += 1
        
        return matches / 4


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


class SelectNextFigureIncreasingEvaluator(BaseEvaluator):
    """
    G-131: Select next figure increasing size evaluator.
    
    Evaluates:
    - Increasing pattern recognition (40%): Recognize small to large pattern
    - Shape type matching (30%): Correct shape type selected
    - Candidate selection accuracy (20%): Correct candidate marked
    - Visual annotation quality (10%): Red circle proper
    """
    
    TASK_WEIGHTS = {
        'pattern': 0.40,
        'shape_type': 0.30,
        'selection': 0.20,
        'annotation': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate select next figure increasing task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Resize if needed
        if last_frame.shape != gt_last.shape:
            gt_last = cv2.resize(gt_last, (last_frame.shape[1], last_frame.shape[0]))
        
        # Detect red circle marking
        gen_red = self._detect_red_circle(last_frame)
        gt_red = self._detect_red_circle(gt_last)
        
        # 1. Pattern recognition (40%): Check if correct answer selected
        pattern_score = self._evaluate_pattern_recognition(gen_red, gt_red, last_frame, gt_last)
        scores['pattern'] = pattern_score
        
        # 2. Shape type matching (30%): Correct shape type
        shape_score = self._evaluate_shape_type(gen_red, gt_red, last_frame, gt_last)
        scores['shape_type'] = shape_score
        
        # 3. Selection accuracy (20%): Red circle position
        selection_score = self._evaluate_selection_accuracy(gen_red, gt_red)
        scores['selection'] = selection_score
        
        # 4. Annotation quality (10%): Red circle proper
        annotation_score = self._evaluate_annotation_quality(gen_red)
        scores['annotation'] = annotation_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red circle annotation."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 100:
            return None
        
        # Check circularity
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2
        
        return {
            'center': (cx, cy),
            'area': area,
            'circularity': circularity,
            'contour': largest
        }
    
    def _evaluate_pattern_recognition(self, gen_red: Optional[Dict], gt_red: Optional[Dict],
                                       gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if the increasing pattern was recognized."""
        if gen_red is None:
            return 0.0
        if gt_red is None:
            return 0.5
        
        # Compare marked positions
        gen_center = gen_red['center']
        gt_center = gt_red['center']
        
        frame_diag = np.sqrt(gen_frame.shape[0]**2 + gen_frame.shape[1]**2)
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        normalized_dist = distance / frame_diag
        
        if normalized_dist < 0.05:
            return 1.0
        elif normalized_dist < 0.10:
            return 0.8
        elif normalized_dist < 0.20:
            return 0.5
        else:
            return max(0.2, 1.0 - normalized_dist)
    
    def _evaluate_shape_type(self, gen_red: Optional[Dict], gt_red: Optional[Dict],
                             gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if correct shape type was selected."""
        if gen_red is None or gt_red is None:
            return 0.0 if gen_red is None else 0.5
        
        # Extract region inside red circle and compare
        gen_center = gen_red['center']
        gt_center = gt_red['center']
        
        # Get regions around centers
        radius = 40
        
        gen_roi = self._extract_roi(gen_frame, gen_center, radius)
        gt_roi = self._extract_roi(gt_frame, gt_center, radius)
        
        if gen_roi is None or gt_roi is None:
            return 0.5
        
        # Compare using histogram
        gen_hist = cv2.calcHist([gen_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        gt_hist = cv2.calcHist([gt_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        cv2.normalize(gen_hist, gen_hist)
        cv2.normalize(gt_hist, gt_hist)
        
        similarity = cv2.compareHist(gen_hist, gt_hist, cv2.HISTCMP_CORREL)
        
        return max(0, similarity)
    
    def _extract_roi(self, frame: np.ndarray, center: Tuple[int, int], radius: int) -> Optional[np.ndarray]:
        """Extract region of interest around center."""
        h, w = frame.shape[:2]
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(w, center[0] + radius)
        y2 = min(h, center[1] + radius)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2]
    
    def _evaluate_selection_accuracy(self, gen_red: Optional[Dict], gt_red: Optional[Dict]) -> float:
        """Evaluate red circle position accuracy."""
        if gen_red is None:
            return 0.0
        if gt_red is None:
            return 0.5
        
        gen_center = gen_red['center']
        gt_center = gt_red['center']
        
        distance = np.sqrt((gen_center[0] - gt_center[0])**2 + (gen_center[1] - gt_center[1])**2)
        
        if distance < 20:
            return 1.0
        elif distance < 50:
            return 0.7
        elif distance < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_annotation_quality(self, gen_red: Optional[Dict]) -> float:
        """Evaluate quality of red circle annotation."""
        if gen_red is None:
            return 0.0
        
        # Check circularity (should be close to 1 for a circle)
        circularity = gen_red['circularity']
        
        if circularity > 0.7:
            return 1.0
        elif circularity > 0.5:
            return 0.7
        elif circularity > 0.3:
            return 0.4
        else:
            return 0.2


# Mapping of task names to evaluators
OPEN60_EVALUATORS_PART2 = {
    'G-25_seperate_object_spinning_data-generator': SeparateObjectsSpinningEvaluator,
    'G-29_chart_extreme_with_data_data-generator': ChartExtremeEvaluator,
    'G-31_directed_graph_navigation_data-generator': DirectedGraphNavigationEvaluator,
    'G-39_attention_shift_different_data-generator': AttentionShiftEvaluator,
    'G-41_grid_highest_cost_data-generator': GridHighestCostEvaluator,
    'G-43_understand_scene_structure_data-generator': UnderstandSceneStructureEvaluator,
    'G-45_key_door_matching_data-generator': KeyDoorMatchingEvaluator,
    'G-51_predict_next_color_data-generator': PredictNextColorEvaluator,
    'G-54_connecting_color_data-generator': ConnectingColorEvaluator,
    'G-131_select_next_figure_increasing_size_sequence_data-generator': SelectNextFigureIncreasingEvaluator,
}
