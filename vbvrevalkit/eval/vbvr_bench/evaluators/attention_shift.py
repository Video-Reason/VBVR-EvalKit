"""Evaluator for G-39_attention_shift_different_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-39_attention_shift_different_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, detect_shapes, color_distance, safe_distance


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
